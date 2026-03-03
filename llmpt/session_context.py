"""
Session context for managing a single libtorrent torrent handle.

Each SessionContext instance manages the lifecycle of a torrent for a specific
HuggingFace repository/revision pair: initialization, file download orchestration,
seeding file mapping, and background progress monitoring.
"""

import os
import shutil
import threading
import time
import logging
from typing import Dict, Any, Optional

from .monitor import run_monitor_loop

from .utils import lt, LIBTORRENT_AVAILABLE

logger = logging.getLogger('llmpt.p2p_batch')


class SessionContext:
    """
    Manages a single libtorrent torrent_handle for a specific repo/revision.
    """
    def __init__(self, repo_id: str, revision: str, tracker_client: Any, lt_session: Any, timeout: int, torrent_data: Optional[bytes] = None):
        self.repo_id = repo_id
        self.revision = revision
        self.tracker_client = tracker_client
        self.lt_session = lt_session
        self.timeout = timeout
        self.torrent_data = torrent_data
        
        self.handle = None
        self.is_valid = True
        self.temp_dir = ""
        self.torrent_info_obj = None # libtorrent torrent_info object
        
        self.lock = threading.Lock()
        # Maps filename -> threading.Event (to notify completion)
        self.file_events: Dict[str, threading.Event] = {}
        self.file_destinations: Dict[str, str] = {}
        
        # Determine paths for fastresume data
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
            # Using ~/.cache/huggingface/hub/p2p_root as a safe anchor point
            self.temp_dir = os.path.expanduser("~/.cache/huggingface/hub/p2p_root")
            os.makedirs(self.temp_dir, exist_ok=True)
            params.save_path = self.temp_dir
            
            # Start paused to set priorities before downloading anything
            params.flags |= lt.torrent_flags.paused
            
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
            self.worker_thread = threading.Thread(target=run_monitor_loop, args=(self,), daemon=True)
            self.worker_thread.start()
            
            start_time = time.time()
            while not self.handle.status().has_metadata:
                if time.time() - start_time > 8: # Reduced to 8s for timeout
                    logger.warning(f"[{self.repo_id}] Timeout waiting for torrent metadata. Falling back.")
                    return False
                time.sleep(1)
                
            self.torrent_info_obj = self.handle.torrent_file()
            
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
        
        Instead of using rename_file() (which causes piece invalidation when
        combined with force_recheck), we let libtorrent download to its default
        save_path and then copy/move the completed file to the HF destination.
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
                    # Don't rename_file()! Let libtorrent download to its default path.
                    # We only set priority to request this file. The monitor loop will
                    # copy the data to `destination` once the file is complete.
                    logger.info(f"[{self.repo_id}] Requesting file {filename} (Index {file_index}). Priority -> 1. Destination: {destination}")
                    self.handle.file_priority(file_index, 1)
                    
                    # If the torrent is already finished, data is already at the default path.
                    # Try to deliver it immediately.
                    status = self.handle.status()
                    if status.state in (4, 5):  # 4=finished, 5=seeding
                        files = self.torrent_info_obj.files()
                        file_progress = self.handle.file_progress()
                        file_size = files.file_size(file_index)
                        if file_progress[file_index] == file_size and file_size > 0:
                            src = self._get_lt_disk_path(file_index)
                            if os.path.exists(src):
                                self._deliver_file(src, destination)
                                logger.info(f"[{self.repo_id}] Torrent already complete, file {filename} delivered immediately.")
                                event.set()
                            else:
                                logger.info(f"[{self.repo_id}] Torrent complete but file not on disk at {src}. Monitor will handle.")
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

    def _get_lt_disk_path(self, file_index: int) -> str:
        """Return the absolute disk path where libtorrent stores a file by default."""
        lt_path = self.torrent_info_obj.files().file_path(file_index)
        return os.path.join(self.temp_dir, lt_path)

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
                    # CRITICAL: Resolve symlinks to real blob paths!
                    # HF cache uses symlinks: snapshots/xxx/config.json -> ../../blobs/sha256hash
                    # libtorrent cannot serve data through symlinks — it needs
                    # the actual blob file path to read and hash-check pieces.
                    real_path = os.path.realpath(local_path)
                    self.handle.rename_file(file_index, real_path)
                    mapped_count += 1
                    if real_path != local_path:
                        logger.info(f"[{self.repo_id}] Mapped for seeding [{file_index}]: {target_norm} -> {real_path} (resolved from symlink {local_path})")
                    else:
                        logger.info(f"[{self.repo_id}] Mapped for seeding [{file_index}]: {target_norm} -> {real_path}")
                else:
                    logger.warning(f"[{self.repo_id}] Cache miss for seeding [{file_index}]: {target_norm} (revision={self.revision})")
            except Exception as e:
                logger.warning(f"[{self.repo_id}] Failed to map file {target_norm} for seeding: {e}")
        
        logger.info(f"[{self.repo_id}] Mapped {mapped_count}/{files.num_files()} real files + {pad_count} pad files.")
                
        # Resume and force recheck so libtorrent verifies all pieces against the
        # renamed file paths. This is required because we used rename_file() to
        # point each torrent file to its actual HF blob path. Without a recheck,
        # libtorrent doesn't know the pieces are already present on disk.
        #
        # NOTE: We intentionally do NOT use seed_mode here. seed_mode is incompatible
        # with rename_file() — it assumes files are at their original torrent-internal
        # paths and silently reports 0 pieces when they've been remapped. The result is
        # a seeder that connects to peers but has nothing to offer, causing 100% timeouts.
        self.handle.resume()
        self.handle.force_recheck()
        logger.info(f"[{self.repo_id}] Force recheck initiated. Waiting for piece verification...")
        
        # Wait for the recheck to complete (state transitions: checking_files -> seeding/finished)
        recheck_start = time.time()
        recheck_timeout = 120  # 2 minutes max for hash check
        while time.time() - recheck_start < recheck_timeout:
            s = self.handle.status()
            # State 1 = checking_files, State 7 = checking_resume_data
            if s.state not in (1, 7):
                logger.info(f"[{self.repo_id}] Recheck complete. State: {s.state}, Progress: {s.progress*100:.1f}%, Pieces: {s.num_pieces}")
                break
            if int(time.time() - recheck_start) % 5 == 0:
                logger.info(f"[{self.repo_id}] Rechecking... {s.progress*100:.1f}%")
            time.sleep(0.5)
        else:
            logger.warning(f"[{self.repo_id}] Recheck timed out after {recheck_timeout}s")
        
        return True

    def _deliver_file(self, src: str, dst: str):
        """Copy a completed file from libtorrent's download path to the HF destination.
        
        Uses os.link() (hard link) for zero-copy on the same filesystem,
        falls back to shutil.copy2() for cross-device.
        """
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        # Remove destination if it already exists (e.g., stale .incomplete file)
        if os.path.exists(dst):
            os.unlink(dst)
        
        try:
            # Hard link: instant, no extra disk space on same filesystem
            os.link(src, dst)
            logger.debug(f"[{self.repo_id}] Hard-linked {src} -> {dst}")
        except OSError:
            # Cross-device fallback
            shutil.copy2(src, dst)
            logger.debug(f"[{self.repo_id}] Copied {src} -> {dst}")
