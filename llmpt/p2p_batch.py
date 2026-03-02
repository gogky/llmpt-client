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

    def register_seeding_task(self, repo_id: str, revision: str, tracker_client: Any, torrent_data: Optional[bytes] = None) -> bool:
        """
        Register a repository to be tracked for background seeding.
        This behaves like a download but without any specific HTTP interception blocks.
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
                    timeout=30, # short timeout for initialization
                    torrent_data=torrent_data,
                    is_seeder=True,  # enables seed_mode to skip piece hash verification
                )
            session_ctx = self.sessions[repo_key]
            
        # Ensure torrent is initialized with the magnet link via tracker
        if not session_ctx.is_valid:
            return False
            
        success = session_ctx._init_torrent()
        if not success:
            logger.error(f"[{repo_id}] Failed to initialize seeding session.")
            return False
                
        # For seeding, force map all files in the torrent to their HF blob equivalents!
        return session_ctx.map_all_files_for_seeding()

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
    def __init__(self, repo_id: str, revision: str, tracker_client: Any, lt_session: Any, timeout: int, torrent_data: Optional[bytes] = None, is_seeder: bool = False):
        self.repo_id = repo_id
        self.revision = revision
        self.tracker_client = tracker_client
        self.lt_session = lt_session
        self.timeout = timeout
        self.torrent_data = torrent_data
        self.is_seeder = is_seeder  # if True, use seed_mode to skip piece hash verification
        
        self.handle = None
        self.is_valid = True
        self.temp_dir = ""
        self.torrent_info_obj = None # libtorrent torrent_info object
        
        self.lock = threading.Lock()
        # Maps filename -> threading.Event (to notify completion)
        self.file_events: Dict[str, threading.Event] = {}
        self.file_destinations: Dict[str, str] = {}
        
        # Determine paths for fastresume data
        import os
        self.fastresume_dir = os.path.expanduser("~/.cache/llmpt/p2p_resume")
        os.makedirs(self.fastresume_dir, exist_ok=True)
        # Safe filename for the resume data
        safe_repo = self.repo_id.replace('/', '_')
        self.fastresume_path = os.path.join(self.fastresume_dir, f"{safe_repo}_{self.revision}.fastresume")
        
        self.worker_thread = None

    def _init_torrent(self) -> bool:
        """Initialize the libtorrent handle if not already done."""
        if self.handle is not None:
            return True
            
        logger.info(f"[{self.repo_id}] Initializing P2P session for revision {self.revision}")
        
        # 1. Ask tracker for torrent info
        torrent_metadata = self.tracker_client.get_torrent_info(self.repo_id, self.revision)
        
        # If not found and we used a commit hash naturally, try the human-readable 'main' fallback branch
        if not torrent_metadata and len(self.revision) >= 40:
            logger.info(f"[{self.repo_id}] Hash lookup failed, retrying tracker lookup using 'main' alias...")
            torrent_metadata = self.tracker_client.get_torrent_info(self.repo_id, "main")
            
        if not torrent_metadata or 'magnet_link' not in torrent_metadata:
            logger.warning(f"[{self.repo_id}] No torrent metadata found on tracker.")
            self.is_valid = False
            return False
            
        magnet_link = torrent_metadata['magnet_link']
        
        # 2. Add torrent based on magnet link or raw torrent data
        try:
            if self.torrent_data:
                logger.info(f"[{self.repo_id}] Initializing session with native raw torrent metadata.")
                info = lt.torrent_info(lt.bdecode(self.torrent_data))
                params = lt.add_torrent_params()
                params.ti = info
            else:
                params = lt.parse_magnet_uri(magnet_link)
            
            # We don't use tempdir anymore. We save everything into a common root 
            # (e.g., the HF blobs root) but we'll use rename_file to pinpoint exact locations.
            # We just need some dummy directory here, or maybe the HF cache dir root.
            import os
            # Using ~/.cache/huggingface/hub/p2p_root as a safe anchor point
            self.temp_dir = os.path.expanduser("~/.cache/huggingface/hub/p2p_root")
            os.makedirs(self.temp_dir, exist_ok=True)
            params.save_path = self.temp_dir
            
            # Start paused to set priorities before downloading anything
            params.flags |= lt.torrent_flags.paused
            
            # For seeding tasks: enable seed_mode to skip piece hash verification.
            # Rationale: HuggingFace already validates each blob via SHA256 (the blob
            # filename IS the hash). Re-verifying with libtorrent's SHA1 is redundant
            # and wastes time proportional to model size (minutes for large LLMs on HDD).
            # seed_mode uses lazy verification: only checks a piece if a peer reports it
            # as corrupted after downloading, which is sufficient for our trust model.
            if self.is_seeder:
                params.flags |= lt.torrent_flags.seed_mode
                logger.info(f"[{self.repo_id}] Seeder mode: using seed_mode flag (skipping piece hash verification)")
            
            # Load fastresume data if it exists
            if os.path.exists(self.fastresume_path):
                try:
                    with open(self.fastresume_path, "rb") as f:
                        resume_data = f.read()
                    try:
                        decoded = lt.bdecode(resume_data)
                        if isinstance(decoded, dict):
                            params.renamed_files = decoded.get(b'mapped_files', {})
                    except Exception:
                        pass
                    
                    # We inject the raw byte array and let libtorrent parse it
                    # depending on the libtorrent version, params may have a different interface
                    if hasattr(lt.add_torrent_params, "parse_resume_data"):
                         # lt>1.2 API
                         params = lt.read_resume_data(resume_data)
                         # gotta re-apply the magnet options
                         params.save_path = self.temp_dir
                         params.url = magnet_link
                         params.flags |= lt.torrent_flags.paused
                except Exception as e:
                    logger.warning(f"[{self.repo_id}] Failed to load resume data: {e}")
                    
            self.handle = self.lt_session.add_torrent(params)
            
            test_peer = os.environ.get('TEST_SEEDER_PEER')
            if test_peer:
                import socket
                try:
                    ip = socket.gethostbyname(test_peer)
                    logger.info(f"[{self.repo_id}] Test environment detected. Explicitly connecting to peer {test_peer} ({ip}):6881")
                    self.handle.connect_peer((ip, 6881), 0)
                except Exception as e:
                    logger.warning(f"Failed to resolve test peer {test_peer}: {e}")
            
            # 3. Wait for metadata to resolve to get the file tree
            logger.info(f"[{self.repo_id}] Waiting for torrent metadata resolution...")
            
            # Start background monitoring thread immediately so it can pick up metadata 
            # later if it times out here and we fall back to HTTP!
            self.worker_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.worker_thread.start()
            
            start_time = time.time()
            while not self.handle.has_metadata():
                if time.time() - start_time > 8: # Reduced to 8s for timeout
                    logger.warning(f"[{self.repo_id}] Timeout waiting for torrent metadata. Falling back.")
                    return False
                time.sleep(1)
                
            self.torrent_info_obj = self.handle.get_torrent_info()
            
            # Initialize all file priorities to 0 (don't download)
            num_files = self.torrent_info_obj.num_files()
            self.handle.prioritize_files([0] * num_files)
            
            # Unpause
            self.handle.resume()
            logger.info(f"[{self.repo_id}] P2P session initialized. {num_files} files available.")
            
            return True
        except Exception as e:
            logger.error(f"[{self.repo_id}] Error initializing torrent: {e}")
            # If a strict exception occurs (e.g. bad magnet link), we DO kill the handle.
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
                
            # Initialize torrent, but don't abort file tracking just because it timed out!
            # A timeout (Returns False) means we still have a valid handle in the background 
            # trying to get metadata, so we MUST track the file for fastresume mapping!
            init_success = self._init_torrent()
            if not init_success and not self.is_valid:
                return False
            
            event = self.file_events.get(filename)
            if not event:
                event = threading.Event()
                self.file_events[filename] = event
                self.file_destinations[filename] = destination
                
                # Try finding the file. BUT, if torrent_info_obj is None because we 
                # fell back to HTTP during metadata fetch, we just queue it up and
                # let the monitor thread prioritize it once metadata arrives!
                file_index = self._find_file_index(filename)
                if file_index is not None:
                    # Rename the libtorrent path for this file to be the ABSOLUTE path
                    # where HF expects the .incomplete file to sit. 
                    import os
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    self.handle.rename_file(file_index, destination)
                    
                    logger.info(f"[{self.repo_id}] Requesting file {filename} (Index {file_index}). Priority -> 1. Mapped to: {destination}")
                    self.handle.file_priority(file_index, 1)
                else:
                    if not self.torrent_info_obj:
                        logger.info(f"[{self.repo_id}] Meta still loading. Queueing file {filename} for background BT tracking.")
                    else:
                        logger.warning(f"[{self.repo_id}] File {filename} not found in metadata. Fallback to HTTP.")
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
        target_norm = target_filename.replace('\\', '/')
        
        for i in range(files.num_files()):
            # libtorrent path inside the torrent
            # e.g., "meta_llama_Llama_2_7b_main/onnx/model.onnx" or "config.json"
            lt_path = files.file_path(i).replace('\\', '/')
            
            # Case 1: Exact match (fallback for third-party single-file torrents)
            if lt_path == target_norm:
                return i
                
            # Case 2: Multi-file torrent (standard for our client, strips root folder)
            parts = lt_path.split('/', 1)
            if len(parts) == 2 and parts[1] == target_norm:
                return i
                
        return None

    def map_all_files_for_seeding(self) -> bool:
        """
        For a seeding task, we aggressively proactively map every single file in the torrent
        to its local Hugging Face blob so that libtorrent can hash-check the whole repo instantly.
        """
        if not self.handle or not self.torrent_info_obj:
            return False
        
        import os
        import tempfile
        pad_dir = os.path.join(self.temp_dir, ".pad_files")
        os.makedirs(pad_dir, exist_ok=True)
            
        logger.info(f"[{self.repo_id}] Mapping all files for background seeding...")
        
        files = self.torrent_info_obj.files()
        mapped_count = 0
        pad_count = 0
        for file_index in range(files.num_files()):
            lt_path = files.file_path(file_index).replace('\\', '/')
            file_size = files.file_size(file_index)
            
            parts = lt_path.split('/', 1)
            target_norm = parts[1] if len(parts) == 2 else lt_path
            
            # Handle libtorrent padding files (.pad/XXXXXX) - these are zero-byte virtual
            # files inserted to align file boundaries to piece boundaries. We must create
            # real zero-filled files of the correct size, otherwise piece hash checks fail
            # and the seeder cannot provide any data to peers.
            if target_norm.startswith('.pad/') or '/.pad/' in target_norm:
                pad_file_path = os.path.join(pad_dir, f"pad_{file_index}_{file_size}")
                if not os.path.exists(pad_file_path):
                    with open(pad_file_path, 'wb') as f:
                        f.write(b'\x00' * file_size)
                self.handle.rename_file(file_index, pad_file_path)
                pad_count += 1
                logger.info(f"[{self.repo_id}] Created padding file [{file_index}]: {target_norm} ({file_size} bytes)")
                continue
            
            # Request the local path from Hugging Face cache
            try:
                from huggingface_hub import try_to_load_from_cache
                local_path = try_to_load_from_cache(
                    repo_id=self.repo_id,
                    filename=target_norm,
                    revision=self.revision
                )
                if local_path and isinstance(local_path, str):
                    self.handle.rename_file(file_index, local_path)
                    mapped_count += 1
                    logger.info(f"[{self.repo_id}] Mapped for seeding [{file_index}]: {target_norm} -> {local_path}")
                else:
                    logger.warning(f"[{self.repo_id}] Cache miss for seeding [{file_index}]: {target_norm} (revision={self.revision})")
            except Exception as e:
                logger.warning(f"[{self.repo_id}] Failed to map file {target_norm} for seeding: {e}")
        
        if self.is_seeder:
            logger.info(f"[{self.repo_id}] Mapped {mapped_count}/{files.num_files()} real files + {pad_count} pad files. seed_mode active — skipping hash check, resuming immediately.")
        else:
            logger.info(f"[{self.repo_id}] Mapped {mapped_count}/{files.num_files()} real files + {pad_count} pad files. Starting libtorrent hash check...")
                
        # Resume the torrent (it was added paused).
        # With seed_mode, this immediately enters seeding state (no checking_files phase).
        # Without seed_mode, libtorrent will first verify all pieces before seeding.
        self.handle.resume()
        return True



    def _monitor_loop(self):
        """Background thread to monitor progress and trigger events."""
        logger.info(f"[{self.repo_id}] Monitor thread started.")
        
        last_save_time = time.time()
        
        while self.is_valid and self.handle:
            time.sleep(1)
            
            # Periodic fastresume save (every 5 seconds)
            if time.time() - last_save_time > 5:
                if self.handle and self.handle.is_valid():
                    self.handle.save_resume_data(lt.save_resume_flags_t.flush_disk_cache)
                last_save_time = time.time()
                
            try:
                # Check for libtorrent alerts
                alerts = self.lt_session.pop_alerts()
                for alert in alerts:
                    # If this is a fastresume save alert for our handle
                    if isinstance(alert, lt.save_resume_data_alert) and alert.handle == self.handle:
                        try:
                            resume_data = lt.bencode(alert.params)
                            with open(self.fastresume_path, "wb") as f:
                                f.write(resume_data)
                            logger.debug(f"[{self.repo_id}] Saved resume data to {self.fastresume_path}")
                        except Exception as e:
                            logger.warning(f"[{self.repo_id}] Failed to write resume data: {e}")
                            
                    elif isinstance(alert, lt.save_resume_data_failed_alert) and alert.handle == self.handle:
                         logger.debug(f"[{self.repo_id}] Save resume data failed: {alert.message()}")

                with self.lock:
                    if not self.handle:
                        break
                        
                    status = self.handle.status()
                    # Check if the whole torrent is suddenly removed or errored
                    has_error = False
                    if hasattr(status, 'errc') and status.errc:
                        # In some lt versions, errc is an object where bool() is true even for Success
                        if hasattr(status.errc, 'value') and status.errc.value() != 0:
                            has_error = True
                        elif hasattr(status.errc, 'message') and status.errc.message() != 'Success':
                            has_error = True
                    elif hasattr(status, 'error') and status.error:
                        has_error = True
                        
                    if has_error:
                        err_msg = status.errc.message() if hasattr(status, 'errc') and hasattr(status.errc, 'message') else str(getattr(status, 'error', getattr(status, 'errc', 'Unknown')))
                        logger.error(f"[{self.repo_id}] Torrent error: {err_msg}")
                        self.is_valid = False
                        break
                    
                    # We only care about checking the specific files people are waiting for
                    pending_files = [f for f, e in self.file_events.items() if not e.is_set()]
                    if not pending_files:
                        continue
                        
                    # If metadata finally arrived, we must belatedly map any requested files
                    if not self.torrent_info_obj and self.handle.has_metadata():
                        self.torrent_info_obj = self.handle.get_torrent_info()
                        num_files = self.torrent_info_obj.num_files()
                        self.handle.prioritize_files([0] * num_files)
                        logger.info(f"[{self.repo_id}] Background metadata resolved! {num_files} files.")
                        
                    if not self.torrent_info_obj:
                        continue # Still downloading metadata
                        
                    file_progress = self.handle.file_progress()
                    files = self.torrent_info_obj.files()
                    
                    for filename in list(pending_files):
                        file_index = self._find_file_index(filename)
                        if file_index is not None:
                            # IMPORTANT: Check if we need to map the file belatedly due to metadata delay
                            destination = self.file_destinations[filename]
                            
                            # There's no great way to check if rename_file was called, so we just
                            # proactively do it here for pending files if priority is 0
                            if self.handle.file_priorities()[file_index] == 0:
                                import os
                                os.makedirs(os.path.dirname(destination), exist_ok=True)
                                self.handle.rename_file(file_index, destination)
                                self.handle.file_priority(file_index, 1)
                                logger.info(f"[{self.repo_id}] Belatedly mapped {filename} -> {destination}")
                            
                            file_size = files.file_size(file_index)
                            progress_bytes = file_progress[file_index]
                            
                            if progress_bytes == file_size and file_size > 0:
                                # File is fully downloaded!
                                # Because we used rename_file, the data is ALREADY in destination!
                                destination = self.file_destinations[filename]
                                logger.info(f"[{self.repo_id}] File {filename} complete directly at {destination}")
                                
                                # Unblock the HTTP thread
                                self.file_events[filename].set()

                                # Also lower priority to 0? Not strictly necessary since it's 100% 
                                # downloaded, but good for cleanup
                                self.handle.file_priority(file_index, 0)
                                
                                # File is fully downloaded, good time to save resume state
                                self.handle.save_resume_data(lt.save_resume_flags_t.flush_disk_cache)
                                
            except Exception as e:
                logger.error(f"[{self.repo_id}] Monitor loop exception: {e}")
                
    def _deliver_file(self, src: str, dst: str):
        # DEPRECATED: We now use libtorrent's built-in target file renaming mechanism
        # into the HF blobs directy to achieve zero-copy. This is no longer called.
        pass
