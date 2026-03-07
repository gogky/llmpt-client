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
from collections import deque
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .tracker_client import TrackerClient

from .monitor import run_monitor_loop

from .utils import lt, LIBTORRENT_AVAILABLE, strip_torrent_root

logger = logging.getLogger(__name__)


class SessionContext:
    """
    Manages a single libtorrent torrent_handle for a specific repo/revision.
    """
    def __init__(
        self,
        repo_id: str,
        revision: str,
        tracker_client: 'TrackerClient',
        lt_session: Optional[object],
        session_mode: str,
        timeout: int,
        torrent_data: Optional[bytes] = None,
        *,
        repo_type: str = 'model',
    ) -> None:
        if session_mode not in ('on_demand', 'full_seed'):
            raise ValueError("session_mode must be 'on_demand' or 'full_seed'")
        self.session_mode = session_mode
        self.repo_id = repo_id
        self.revision = revision
        self.repo_type = repo_type
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
        
        # Thread-safe alert inbox: populated by P2PBatchManager.dispatch_alerts(),
        # consumed by the monitor thread via _process_alerts().
        self.alert_lock = threading.Lock()
        self.pending_alerts: deque = deque()
        
        # Determine paths for fastresume data
        self.fastresume_dir = os.path.expanduser("~/.cache/llmpt/p2p_resume")
        os.makedirs(self.fastresume_dir, exist_ok=True)
        # Safe filename for the resume data
        safe_repo = self.repo_id.replace('/', '_')
        self.fastresume_path = os.path.join(self.fastresume_dir, f"{safe_repo}_{self.revision}.fastresume")
        
        self.worker_thread = None
        self.test_peer_addr = None  # (ip, port) tuple for direct peer connection in test environments
        self._has_webseed = False   # set during _init_torrent if WebSeed proxy is active
        self._peer_ready = threading.Event()  # signals that ≥1 peer is connected (test warmup)

    def _init_torrent(self) -> bool:
        """Initialize the libtorrent handle if not already done."""
        if self.handle is not None:
            return True

        from .torrent_init import acquire_torrent_data, build_add_torrent_params, resolve_test_peer

        logger.info(f"[{self.repo_id}] Initializing P2P session for revision {self.revision}")

        # 1. Obtain torrent data
        torrent_data = acquire_torrent_data(
            self.repo_id, self.revision, self.tracker_client, self.torrent_data, repo_type=self.repo_type
        )
        if not torrent_data:
            logger.warning(f"[{self.repo_id}] No torrent data available from tracker.")
            self.is_valid = False
            return False

        # 2. Build params and add torrent
        try:
            self.temp_dir = os.path.expanduser("~/.cache/huggingface/hub/p2p_root")
            os.makedirs(self.temp_dir, exist_ok=True)

            params, info = build_add_torrent_params(
                torrent_data=torrent_data,
                save_path=self.temp_dir,
                session_mode=self.session_mode,
                fastresume_path=self.fastresume_path,
                repo_id=self.repo_id,
            )
            self.handle = self.lt_session.add_torrent(params)

            # 3. Add WebSeed URL if proxy is running
            webseed_url = self._get_webseed_url()
            if webseed_url:
                self._has_webseed = True
                self.handle.add_url_seed(webseed_url)
                logger.info(f"[{self.repo_id}] WebSeed added: {webseed_url}")

            # 4. Resolve test peer if configured (connection handled by monitor loop)
            peer_addr = resolve_test_peer()
            if peer_addr:
                self.test_peer_addr = peer_addr
                logger.info(f"[{self.repo_id}] Test peer resolved to {peer_addr[0]}:{peer_addr[1]}")

            # torrent_info is immediately available — no metadata wait needed
            self.torrent_info_obj = self.handle.torrent_file()

            # Start background monitoring thread
            self.worker_thread = threading.Thread(target=run_monitor_loop, args=(self,), daemon=True)
            self.worker_thread.start()

            # Initialize all file priorities to 0 (don't download) for on-demand sessions.
            # Pure full_seed sessions keep default priority so seed_mode works.
            num_files = self.torrent_info_obj.num_files()
            if self.session_mode == 'on_demand':
                self.handle.prioritize_files([0] * num_files)

            logger.info(f"[{self.repo_id}] P2P session initialized. {num_files} files available.")
            return True
        except Exception as e:
            logger.error(f"[{self.repo_id}] Error initializing torrent: {e}")
            self.is_valid = False
            if self.handle:
                self.lt_session.remove_torrent(self.handle)
                self.handle = None
            return False

    def _wait_for_peer_ready(self, warmup_timeout: int = 60) -> None:
        """Wait until at least one peer is connected (test environments only).

        In pure-P2P test scenarios (no WebSeed), HuggingFace launches 10+
        concurrent download threads.  Without this warmup, all threads start
        their timeout countdowns *before* the peer handshake completes,
        causing early files to time out and fall back to HTTP.

        This method blocks until ``handle.status().num_peers > 0``, then
        sets ``_peer_ready`` so subsequent threads return immediately.

        Only called when ``test_peer_addr`` is set and ``_has_webseed`` is
        False — zero impact on production.
        """
        if self._peer_ready.is_set():
            return

        logger.info(f"[{self.repo_id}] Peer warmup: waiting for peer connection before starting downloads...")

        def is_fully_downloaded() -> bool:
            try:
                if not self.handle or not self.torrent_info_obj:
                    return False
                status = self.handle.status()
                if status.state in (4, 5):
                    progresses = self.handle.file_progress()
                    if sum(progresses) >= self.torrent_info_obj.total_size():
                        return True
            except Exception:
                pass
            return False

        # Ensure we're actively trying to connect
        if self.handle and self.test_peer_addr:
            try:
                # If torrent is already complete (or close to), it might not need peers
                if is_fully_downloaded():
                    logger.info(f"[{self.repo_id}] Peer warmup bypassed (already fully downloaded).")
                    self._peer_ready.set()
                    return
                self.handle.connect_peer(self.test_peer_addr, 0)
            except Exception:
                pass

        deadline = time.time() + warmup_timeout
        while time.time() < deadline:
            if not self.is_valid:
                break
            try:
                if self.handle:
                    status = self.handle.status()
                    if status.num_peers > 0:
                        logger.info(f"[{self.repo_id}] Peer warmup complete — {status.num_peers} peer(s) connected.")
                        self._peer_ready.set()
                        return
                    if is_fully_downloaded():
                        logger.info(f"[{self.repo_id}] Peer warmup bypassed (already fully downloaded).")
                        self._peer_ready.set()
                        return
            except Exception:
                pass
            time.sleep(0.5)

        logger.warning(f"[{self.repo_id}] Peer warmup timed out after {warmup_timeout}s. Proceeding anyway.")
        self._peer_ready.set()  # unblock all waiting threads

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
                        if self.handle:
                            self.handle.resume()
                        return False
            
            if self.handle:
                self.handle.resume()
                if self.test_peer_addr:
                    self.handle.connect_peer(self.test_peer_addr, 0)
                    logger.debug(f"[{self.repo_id}] Connected to test peer {self.test_peer_addr} after resume")

        # Peer warmup: in test environments without WebSeed, wait for the peer
        # handshake to complete BEFORE starting the download timer.  This
        # prevents the race where concurrent threads all time out because the
        # peer wasn't connected yet.
        if self.test_peer_addr and not self._has_webseed:
            self._wait_for_peer_ready()
        
        # Block until the background thread signals completion, but also check
        # if the monitor thread has exited (is_valid becomes False) so we can
        # fail fast instead of waiting the full timeout.
        # timeout=0 means "no timeout" (WebSeed guarantees progress).
        logger.debug(f"[{self.repo_id}] Blocking waiting for P2P download of {filename}...")
        deadline = time.time() + self.timeout if self.timeout > 0 else float('inf')
        
        # Use huggingface_hub's tqdm to integrate with native HF progress bars
        try:
            from huggingface_hub.utils import tqdm as hf_tqdm, are_progress_bars_disabled
            disable_pbar = are_progress_bars_disabled()
        except ImportError:
            # Fallback if huggingface_hub is not available (shouldn't happen in our context)
            hf_tqdm = None
            disable_pbar = True

        pbar = None
        last_progress = 0
        total_size = 0
        file_index = None

        try:
            while not event.is_set():
                if not self.is_valid:
                    logger.warning(f"[{self.repo_id}] P2P download of {filename} ABORTED (monitor stopped).")
                    return False
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                event.wait(timeout=min(1.0, remaining))
                
                # Fetch metadata and file properties dynamically if they arrive
                if file_index is None and self.torrent_info_obj:
                    file_index = self._find_file_index(filename)
                    if file_index is not None:
                        total_size = self.torrent_info_obj.files().file_size(file_index)
                
                # If we have a valid index and handle, we can update the progress bar
                if file_index is not None and self.handle and self.handle.is_valid():
                    if pbar is None and not disable_pbar and hf_tqdm:
                        pbar = hf_tqdm(
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            desc=f"{filename} (P2P)"
                        )
                    
                    try:
                        progresses = self.handle.file_progress()
                        current_progress = progresses[file_index]
                        if current_progress > last_progress:
                            # Avoid over-updating standard totals
                            if pbar:
                                pbar.update(current_progress - last_progress)
                            last_progress = current_progress
                        
                        if pbar:
                            s = self.handle.status()
                            pbar.set_postfix({"peers": s.num_peers, "dl_speed": f"{s.download_rate / 1024:.1f}KB/s"})
                    except Exception:
                        pass

        finally:
            if pbar is not None:
                if event.is_set() and last_progress < total_size:
                    pbar.update(total_size - last_progress)
                pbar.close()

        if event.is_set():
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
            lt_path = files.file_path(i)
            lt_norm = lt_path.replace('\\', '/')
            
            # Case 1: Exact match (fallback for third-party single-file torrents)
            if lt_norm == target_norm:
                return i
                
            # Case 2: Multi-file torrent (standard for our client, strips root folder)
            if strip_torrent_root(lt_path) == target_norm:
                return i
                
        return None

    def _get_webseed_url(self) -> Optional[str]:
        """Build the WebSeed URL for this session's repo, if the proxy is running.

        Reads ``webseed_proxy_port`` from the global config set by
        :func:`llmpt.enable_p2p`.  Returns ``None`` when:
        - The proxy was not started (port is None).
        - The import fails (should never happen in normal operation).
        """
        try:
            from . import get_config
            proxy_port = get_config().get('webseed_proxy_port')
            if proxy_port:
                return f"http://127.0.0.1:{proxy_port}/ws/{self.repo_type}/{self.repo_id}/"
        except Exception:
            pass
        return None

    def map_all_files_for_seeding(self) -> bool:
        """Map every file in the torrent to its local HuggingFace blob for seeding.

        Tries hardlinks first (instant seed_mode startup), falls back to
        rename_file + force_recheck if hardlinks fail (cross-filesystem).
        """
        if not self.handle or not self.torrent_info_obj:
            return False

        from .seeding_mapper import (
            hardlink_files_for_seeding,
            rename_files_for_seeding,
            cleanup_hardlinks,
        )

        logger.info(f"[{self.repo_id}] Mapping all files for background seeding...")

        try:
            hardlinks, mapped_count = hardlink_files_for_seeding(
                self.torrent_info_obj, self.temp_dir, self.repo_id, self.revision,
                repo_type=self.repo_type,
            )
            self.seeding_hardlinks = hardlinks

            files = self.torrent_info_obj.files()
            logger.info(f"[{self.repo_id}] Hardlinked {mapped_count}/{files.num_files()} files for seeding.")

            self.handle.set_flags(lt.torrent_flags.seed_mode)
            self.handle.resume()
            logger.info(f"[{self.repo_id}] Seeding started with seed_mode (0s startup, lazy verification).")
            return True

        except OSError:
            # Hardlink failed (cross-filesystem) — fall back to legacy
            logger.warning(f"[{self.repo_id}] Hardlink failed (cross-filesystem?). Falling back to force_recheck.")
            cleanup_hardlinks(self.repo_id, getattr(self, 'seeding_hardlinks', []))
            self.seeding_hardlinks = []

            rename_files_for_seeding(
                self.handle, self.torrent_info_obj,
                self.temp_dir, self.repo_id, self.revision,
                repo_type=self.repo_type,
            )
            return True

    def _cleanup_seeding_hardlinks(self):
        """Remove hardlinks created for seeding in p2p_root."""
        from .seeding_mapper import cleanup_hardlinks
        cleanup_hardlinks(self.repo_id, getattr(self, 'seeding_hardlinks', []))
        self.seeding_hardlinks = []

    def _deliver_file(self, src: str, dst: str):
        """Copy a completed file from libtorrent's download path to the HF destination.
        
        Uses os.link() (hard link) for zero-copy on the same filesystem,
        falls back to shutil.copy2() for cross-device.
        
        The source files in p2p_root MUST be preserved while the torrent is active 
        so libtorrent can read pieces/chunks across file boundaries. The entire
        session's temporary directory is cleaned up upon session teardown.
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

    def cleanup_temp_dir(self):
        """Delete the temporary directory containing download payloads."""
        if self.temp_dir and os.path.isdir(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.debug(f"[{self.repo_id}] Cleaned up temp dir: {self.temp_dir}")
            except Exception as e:
                logger.debug(f"[{self.repo_id}] Failed to clean temp dir {self.temp_dir}: {e}")
