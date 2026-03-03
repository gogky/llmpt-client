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

# ── Timeout defaults (seconds) ────────────────────────────────────────────────
# Internal tuning knobs, NOT exposed to end users via the public API.
# Override via environment variables for debugging / testing only.
METADATA_TIMEOUT = int(os.environ.get('LLMPT_METADATA_TIMEOUT', '8'))


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
        self.test_peer_addr = None  # (ip, port) tuple for direct peer connection in test environments

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
                    # Parse host:port with IPv6-safe handling
                    # Supported formats: "hostname", "hostname:port",
                    #   "1.2.3.4:port", "[::1]:port"
                    if test_peer.startswith('['):
                        # IPv6 bracket notation: [::1]:port
                        bracket_end = test_peer.index(']')
                        host = test_peer[1:bracket_end]
                        port = int(test_peer[bracket_end + 2:]) if len(test_peer) > bracket_end + 2 else 6881
                    elif ':' in test_peer:
                        host, port_str = test_peer.rsplit(':', 1)
                        port = int(port_str)
                    else:
                        host = test_peer
                        port = 6881
                    
                    ip = socket.gethostbyname(host)
                    self.test_peer_addr = (ip, port)
                    logger.info(f"[{self.repo_id}] Test environment detected. Explicitly connecting to peer {host} ({ip}):{port}")
                    self.handle.connect_peer(self.test_peer_addr, 0)
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
                if time.time() - start_time > METADATA_TIMEOUT:
                    logger.warning(f"[{self.repo_id}] Timeout waiting for torrent metadata after {METADATA_TIMEOUT}s. Falling back.")
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
        Map every file in the torrent to its local HuggingFace blob for seeding.

        Strategy: create hardlinks at the paths libtorrent expects (its default
        save_path layout) pointing to the real HF blob files, then enable
        seed_mode so libtorrent starts serving immediately (0 seconds) with
        lazy per-piece SHA1 verification on first peer request.

        Falls back to the legacy rename_file() + force_recheck() approach if
        hardlinks fail (e.g. cross-filesystem).
        """
        if not self.handle or not self.torrent_info_obj:
            return False

        logger.info(f"[{self.repo_id}] Mapping all files for background seeding...")

        files = self.torrent_info_obj.files()
        use_seed_mode = True  # optimistic; set to False on first hardlink failure

        # Track created hardlinks so we can clean them up when seeding stops
        self.seeding_hardlinks = []

        mapped_count = 0
        for file_index in range(files.num_files()):
            lt_path = files.file_path(file_index).replace('\\', '/')
            file_size = files.file_size(file_index)

            parts = lt_path.split('/', 1)
            target_norm = parts[1] if len(parts) == 2 else lt_path

            # Handle libtorrent padding files (.pad/XXXXXX) — zero-filled virtual
            # files that align file boundaries to piece boundaries.
            if target_norm.startswith('.pad/') or '/.pad/' in target_norm:
                expected_path = os.path.join(self.temp_dir, lt_path)
                os.makedirs(os.path.dirname(expected_path), exist_ok=True)
                if not os.path.exists(expected_path):
                    with open(expected_path, 'wb') as f:
                        f.write(b'\x00' * file_size)
                    self.seeding_hardlinks.append(expected_path)
                logger.info(f"[{self.repo_id}] Created padding file [{file_index}]: {target_norm} ({file_size} bytes)")
                continue

            # Resolve the HF cache blob path
            try:
                from huggingface_hub import try_to_load_from_cache
                local_path = try_to_load_from_cache(
                    repo_id=self.repo_id,
                    filename=target_norm,
                    revision=self.revision
                )
                if not local_path or not isinstance(local_path, str):
                    logger.warning(f"[{self.repo_id}] Cache miss for seeding [{file_index}]: {target_norm} (revision={self.revision})")
                    continue

                # Resolve symlinks: snapshots/xxx/file -> ../../blobs/sha256hash
                real_path = os.path.realpath(local_path)

                # Create hardlink at the path libtorrent expects
                expected_path = os.path.join(self.temp_dir, lt_path)
                os.makedirs(os.path.dirname(expected_path), exist_ok=True)

                # Remove stale hardlink if it exists
                if os.path.exists(expected_path):
                    os.unlink(expected_path)

                try:
                    os.link(real_path, expected_path)
                    self.seeding_hardlinks.append(expected_path)
                    mapped_count += 1
                    logger.info(f"[{self.repo_id}] Hardlinked for seeding [{file_index}]: {target_norm} -> {real_path}")
                except OSError:
                    # Cross-filesystem: cannot hardlink, fall back to legacy approach
                    logger.warning(f"[{self.repo_id}] Hardlink failed (cross-filesystem?) for {target_norm}. Falling back to force_recheck.")
                    use_seed_mode = False
                    self._cleanup_seeding_hardlinks()
                    return self._map_all_files_legacy()

            except Exception as e:
                logger.warning(f"[{self.repo_id}] Failed to map file {target_norm} for seeding: {e}")

        logger.info(f"[{self.repo_id}] Hardlinked {mapped_count}/{files.num_files()} files for seeding.")

        # Enable seed_mode: libtorrent assumes all pieces are present and lazily
        # verifies each piece's SHA1 hash on first peer request. If any piece
        # fails verification, libtorrent automatically exits seed_mode and does
        # a full recheck. This is safe because HF blobs are content-addressed
        # (named by hash) and immutable after download.
        self.handle.set_flags(lt.torrent_flags.seed_mode)
        self.handle.resume()
        logger.info(f"[{self.repo_id}] Seeding started with seed_mode (0s startup, lazy verification).")

        return True

    def _map_all_files_legacy(self) -> bool:
        """Legacy seeding: rename_file() + force_recheck().

        Used as fallback when hardlinks fail (cross-filesystem scenario).
        This requires a full hash verification of all pieces, which can take
        minutes for large models.
        """
        files = self.torrent_info_obj.files()
        mapped_count = 0

        for file_index in range(files.num_files()):
            lt_path = files.file_path(file_index).replace('\\', '/')
            file_size = files.file_size(file_index)

            parts = lt_path.split('/', 1)
            target_norm = parts[1] if len(parts) == 2 else lt_path

            if target_norm.startswith('.pad/') or '/.pad/' in target_norm:
                pad_dir = os.path.join(self.temp_dir, ".pad_files")
                os.makedirs(pad_dir, exist_ok=True)
                pad_file_path = os.path.join(pad_dir, f"pad_{file_index}_{file_size}")
                if not os.path.exists(pad_file_path):
                    with open(pad_file_path, 'wb') as f:
                        f.write(b'\x00' * file_size)
                self.handle.rename_file(file_index, pad_file_path)
                continue

            try:
                from huggingface_hub import try_to_load_from_cache
                local_path = try_to_load_from_cache(
                    repo_id=self.repo_id,
                    filename=target_norm,
                    revision=self.revision
                )
                if local_path and isinstance(local_path, str):
                    real_path = os.path.realpath(local_path)
                    self.handle.rename_file(file_index, real_path)
                    mapped_count += 1
                    logger.info(f"[{self.repo_id}] [legacy] Mapped [{file_index}]: {target_norm} -> {real_path}")
                else:
                    logger.warning(f"[{self.repo_id}] [legacy] Cache miss [{file_index}]: {target_norm}")
            except Exception as e:
                logger.warning(f"[{self.repo_id}] [legacy] Failed to map {target_norm}: {e}")

        logger.info(f"[{self.repo_id}] [legacy] Mapped {mapped_count}/{files.num_files()} files. Starting force_recheck...")

        self.handle.resume()
        self.handle.force_recheck()

        # Wait for recheck (no hard timeout — large models can take 10+ min on HDD)
        recheck_start = time.time()
        last_log_time = 0.0
        while True:
            s = self.handle.status()
            if s.state not in (1, 7):
                elapsed = time.time() - recheck_start
                logger.info(f"[{self.repo_id}] [legacy] Recheck complete in {elapsed:.0f}s. Pieces: {s.num_pieces}")
                break
            now = time.time()
            if now - last_log_time >= 10:
                last_log_time = now
                elapsed = now - recheck_start
                logger.info(f"[{self.repo_id}] [legacy] Rechecking... {s.progress*100:.1f}% ({elapsed:.0f}s)")
            time.sleep(0.5)

        return True

    def _cleanup_seeding_hardlinks(self):
        """Remove hardlinks created for seeding in p2p_root."""
        for path in getattr(self, 'seeding_hardlinks', []):
            try:
                if os.path.exists(path):
                    os.unlink(path)
                    logger.debug(f"[{self.repo_id}] Cleaned up seeding hardlink: {path}")
            except OSError as e:
                logger.warning(f"[{self.repo_id}] Failed to clean up {path}: {e}")
        self.seeding_hardlinks = []

    def _deliver_file(self, src: str, dst: str):
        """Copy a completed file from libtorrent's download path to the HF destination.
        
        Uses os.link() (hard link) for zero-copy on the same filesystem,
        falls back to shutil.copy2() for cross-device.
        After delivery, removes the source file from p2p_root to avoid
        leaving behind duplicate directory entries (or real copies on
        cross-filesystem setups).
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
        
        # Clean up source: for hardlinks this just decrements the inode refcount
        # (data remains at dst); for copies this frees the duplicate disk space.
        try:
            os.unlink(src)
            logger.debug(f"[{self.repo_id}] Cleaned up source: {src}")
        except OSError as e:
            logger.debug(f"[{self.repo_id}] Could not remove source {src}: {e}")
