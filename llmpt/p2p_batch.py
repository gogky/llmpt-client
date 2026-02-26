"""
P2P Batch Manager for coordinating libtorrent downloads and HTTP intercepts.

This module provides a singleton P2PBatchManager that aggregates concurrent 
HTTP download requests into a single, efficient BitTorrent session with 
dynamic file prioritization.
"""

import os
import threading
import time
import logging
from typing import Dict, Any, Optional

try:
    import libtorrent as lt
    LIBTORRENT_AVAILABLE = True
except ImportError:
    LIBTORRENT_AVAILABLE = False
    lt = None

logger = logging.getLogger('llmpt.p2p_batch')

class P2PBatchManager:
    """
    Manages global P2P download sessions for different HuggingFace repositories.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(P2PBatchManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        with self._lock:
            if getattr(self, '_initialized', False):
                return
            
            self._initialized = True
            # Mapping from (repo_id, revision) -> SessionContext
            self.sessions: Dict[tuple, "SessionContext"] = {}
            if LIBTORRENT_AVAILABLE:
                self.lt_session = lt.session()
                self.lt_session.listen_on(6881, 6891)
            else:
                self.lt_session = None

    def register_request(
        self,
        repo_id: str,
        revision: str,
        filename: str,
        temp_file_path: str,
        tracker_client: Any,
        timeout: int = 300
    ) -> bool:
        """
        Register a file download request.
        
        Args:
            repo_id: The HF repository ID.
            revision: The commit hash or branch name.
            filename: The relative path of the file to download within the repo.
            temp_file_path: The absolute path where HF expects the file to be saved.
            tracker_client: Instance of TrackerClient to query torrennt info.
            timeout: Max time to wait for the download.
            
        Returns:
            True if P2P download succeeds, False if it failed and should fallback to HTTP.
        """
        if not LIBTORRENT_AVAILABLE:
            return False

        repo_key = (repo_id, revision)
        
        with self._lock:
            if repo_key not in self.sessions:
                self.sessions[repo_key] = SessionContext(
                    repo_id=repo_id,
                    revision=revision,
                    tracker_client=tracker_client,
                    lt_session=self.lt_session,
                    timeout=timeout
                )
            session_ctx = self.sessions[repo_key]
        
        # Register the file with the session context and wait for it
        return session_ctx.download_file(filename, temp_file_path)

class SessionContext:
    """
    Manages a single libtorrent torrent_handle for a specific repo/revision.
    """
    def __init__(self, repo_id: str, revision: str, tracker_client: Any, lt_session: Any, timeout: int):
        self.repo_id = repo_id
        self.revision = revision
        self.tracker_client = tracker_client
        self.lt_session = lt_session
        self.timeout = timeout
        
        self.handle = None
        self.is_valid = True
        self.temp_dir = ""
        self.torrent_info_obj = None # libtorrent torrent_info object
        
        self.lock = threading.Lock()
        # Maps filename -> threading.Event (to notify completion)
        self.file_events: Dict[str, threading.Event] = {}
        # Maps filename -> expected output path
        self.file_destinations: Dict[str, str] = {}
        
        self.worker_thread = None

    def _init_torrent(self) -> bool:
        """Initialize the libtorrent handle if not already done."""
        if self.handle is not None:
            return True
            
        logger.info(f"[{self.repo_id}] Initializing P2P session for revision {self.revision}")
        
        # 1. Ask tracker for torrent info
        torrent_metadata = self.tracker_client.get_torrent_info(self.repo_id, self.revision)
        if not torrent_metadata or 'magnet_link' not in torrent_metadata:
            logger.warning(f"[{self.repo_id}] No torrent metadata found on tracker.")
            self.is_valid = False
            return False
            
        magnet_link = torrent_metadata['magnet_link']
        
        # 2. Add torrent based on magnet link
        try:
            params = lt.parse_magnet_uri(magnet_link)
            import tempfile
            self.temp_dir = tempfile.mkdtemp(prefix=f"llmpt_p2p_{self.repo_id.replace('/', '_')}_")
            params.save_path = self.temp_dir
            
            # Start paused to set priorities before downloading anything
            params.flags |= lt.torrent_flags.paused
            
            self.handle = self.lt_session.add_torrent(params)
            
            # 3. Wait for metadata to resolve to get the file tree
            logger.info(f"[{self.repo_id}] Waiting for torrent metadata resolution...")
            start_time = time.time()
            while not self.handle.has_metadata():
                if time.time() - start_time > 30: # 30s timeout for metadata
                    raise Exception("Timeout waiting for torrent metadata")
                time.sleep(1)
                
            self.torrent_info_obj = self.handle.get_torrent_info()
            
            # Initialize all file priorities to 0 (don't download)
            num_files = self.torrent_info_obj.num_files()
            self.handle.prioritize_files([0] * num_files)
            
            # Unpause
            self.handle.resume()
            logger.info(f"[{self.repo_id}] P2P session initialized. {num_files} files available.")
            
            # Start background monitoring thread
            self.worker_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.worker_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"[{self.repo_id}] Error initializing torrent: {e}")
            self.is_valid = False
            if self.handle:
                self.lt_session.remove_torrent(self.handle)
                self.handle = None
            return False

    def download_file(self, filename: str, destination: str) -> bool:
        """
        Request download of a specific file and block until done.
        """
        with self.lock:
            if not self.is_valid:
                return False
                
            if not self._init_torrent():
                return False
            
            event = self.file_events.get(filename)
            if not event:
                event = threading.Event()
                self.file_events[filename] = event
                self.file_destinations[filename] = destination
                
                # Find the file index in the torrent and prioritize it
                file_index = self._find_file_index(filename)
                if file_index is not None:
                    logger.info(f"[{self.repo_id}] Requesting file {filename} (Index {file_index}). Priority -> 1")
                    self.handle.file_priority(file_index, 1)
                else:
                    logger.warning(f"[{self.repo_id}] File {filename} not found in torrent metadata. Fallback to HTTP.")
                    return False
        
        # Block until the background thread signals completion
        logger.debug(f"[{self.repo_id}] Blocking waiting for P2P download of {filename}...")
        success = event.wait(timeout=self.timeout)
        
        if success:
            logger.info(f"[{self.repo_id}] P2P download of {filename} SUCCESS.")
            return True
        else:
            logger.warning(f"[{self.repo_id}] P2P download of {filename} TIMEOUT after {self.timeout}s.")
            return False

    def _find_file_index(self, target_filename: str) -> Optional[int]:
        """Find the libtorrent file index matching the requested filename."""
        if not self.torrent_info_obj:
            return None
            
        files = self.torrent_info_obj.files()
        for i in range(files.num_files()):
            # libtorrent path inside the torrent (e.g. meta-llama/Llama-2/config.json)
            # We want to match the end of the path
            lt_path = files.file_path(i)
            # HF requests clean paths like 'config.json' or 'models/part1.bin'
            if lt_path.endswith(target_filename):
                return i
        return None

    def _monitor_loop(self):
        """Background thread to monitor progress and trigger events."""
        logger.info(f"[{self.repo_id}] Monitor thread started.")
        
        while self.is_valid and self.handle:
            time.sleep(1)
            
            try:
                with self.lock:
                    if not self.handle:
                        break
                        
                    status = self.handle.status()
                    # Check if the whole torrent is suddenly removed or errored
                    if status.errc:
                        logger.error(f"[{self.repo_id}] Torrent error: {status.errc}")
                        self.is_valid = False
                        break
                    
                    # We only care about checking the specific files people are waiting for
                    pending_files = [f for f, e in self.file_events.items() if not e.is_set()]
                    if not pending_files:
                        continue
                        
                    file_progress = self.handle.file_progress()
                    files = self.torrent_info_obj.files()
                    
                    for filename in pending_files:
                        file_index = self._find_file_index(filename)
                        if file_index is not None:
                            file_size = files.file_size(file_index)
                            progress_bytes = file_progress[file_index]
                            
                            if progress_bytes == file_size and file_size > 0:
                                # File is fully downloaded!
                                lt_path = files.file_path(file_index)
                                full_local_path = os.path.join(self.temp_dir, lt_path)
                                destination = self.file_destinations[filename]
                                
                                # Move (hardlink or copy) logic
                                self._deliver_file(full_local_path, destination)
                                
                                # Unblock the HTTP thread
                                self.file_events[filename].set()

                                # Also lower priority to 0? Not strictly necessary since it's 100% 
                                # downloaded, but good for cleanup
                                self.handle.file_priority(file_index, 0)
                                
            except Exception as e:
                logger.error(f"[{self.repo_id}] Monitor loop exception: {e}")
                
    def _deliver_file(self, src: str, dst: str):
        """Deliver the downloaded file to the HF expected destination."""
        if not os.path.exists(src):
            logger.error(f"Could not find downloaded file at {src}")
            return
            
        import shutil
        import os
        
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        # Target HF temp_file already exists as an empty file because HF touched it.
        # We need to remove it first otherwise os.link fails.
        if os.path.exists(dst):
            os.remove(dst)
            
        try:
            os.link(src, dst)
            logger.debug(f"Hardlinked {src} -> {dst}")
        except OSError:
            logger.debug(f"Hardlink failed, falling back to copy {src} -> {dst}")
            shutil.copy2(src, dst)
