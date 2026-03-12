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


def _format_live_transfer_postfix(stats: Optional[dict]) -> str:
    """Format a lightweight, user-facing transfer status for single-file bars."""
    stats = stats or {}
    active_peers = int(stats.get('active_p2p_peers', 0) or 0)
    peer_bytes = int(stats.get('peer_download', 0) or 0)
    webseed_bytes = int(stats.get('webseed_download', 0) or 0)

    if peer_bytes > 0:
        return f"peers={active_peers}"
    if webseed_bytes > 0:
        return "webseed"
    return ""


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
        cache_dir: Optional[str] = None,
        local_dir: Optional[str] = None,
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
        self.cache_dir = cache_dir
        self.local_dir = local_dir
        
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

        # Accumulated P2P transfer stats (sampled during download, survives peer disconnection)
        self._stats_lock = threading.Lock()
        self._acc_peer_download: int = 0
        self._acc_webseed_download: int = 0
        self._acc_total_payload_download: int = 0
        self._acc_peak_p2p_peers: int = 0
        self.seeding_mapped_files: int = 0
        self.seeding_total_files: int = 0
        self.full_mapping: bool = False
        self.seeding_hardlinks = []

    def get_file_progress(self, *, verified_only: bool = False):
        """Return file progress, optionally counting only hash-verified pieces."""
        if not self.handle:
            return []
        flags = 0
        if verified_only and hasattr(lt.torrent_handle, "piece_granularity"):
            flags = lt.torrent_handle.piece_granularity
        return self.handle.file_progress(flags)

    def _init_torrent(self, initial_filename: str = None) -> bool:
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
            if self.local_dir:
                self.temp_dir = os.path.join(self.local_dir, ".cache", "huggingface", "p2p_root")
            elif self.cache_dir:
                self.temp_dir = os.path.join(self.cache_dir, "p2p_root")
            else:
                self.temp_dir = os.path.expanduser("~/.cache/huggingface/hub/p2p_root")
            os.makedirs(self.temp_dir, exist_ok=True)

            params, info = build_add_torrent_params(
                torrent_data=torrent_data,
                save_path=self.temp_dir,
                session_mode=self.session_mode,
                fastresume_path=self.fastresume_path,
                repo_id=self.repo_id,
            )
            
            if self.session_mode == 'on_demand' and info is not None:
                from .utils import strip_torrent_root
                priorities = [0] * info.num_files()
                if initial_filename:
                    files = info.files()
                    for i in range(info.num_files()):
                        if strip_torrent_root(files.file_path(i).replace('\\', '/')) == initial_filename:
                            priorities[i] = 1
                            break
                # Also prioritize overlapping files to avoid partfile EOF bug
                if initial_filename and 1 in priorities:
                    s_idx = priorities.index(1)
                    for idx in self._get_overlapping_file_indices(s_idx, torrent_info=info):
                        priorities[idx] = 1
                params.file_priorities = priorities

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

            # File priorities are set in add_torrent_params now.
            num_files = self.torrent_info_obj.num_files()

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
                    progresses = self.get_file_progress(verified_only=True)
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

    def download_file(self, filename: str, destination: str, tqdm_class=None) -> bool:
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
            init_success = self._init_torrent(filename)
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

                    # Fix libtorrent partfile EOF bug: also prioritize files
                    # sharing piece boundaries with this file.
                    for idx in self._get_overlapping_file_indices(file_index):
                        logger.debug(f"[{self.repo_id}] Also prioritizing overlapping file index {idx} to avoid partfile EOF")
                        self.handle.file_priority(idx, 1)
                    
                    # If the torrent is already finished, data is already at the default path.
                    # Try to deliver it immediately.
                    status = self.handle.status()
                    if status.state in (4, 5):  # 4=finished, 5=seeding
                        files = self.torrent_info_obj.files()
                        file_progress = self.get_file_progress(verified_only=True)
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
            from huggingface_hub.utils import tqdm as hf_tqdm_lib, are_progress_bars_disabled
            disable_pbar = are_progress_bars_disabled()
            hf_tqdm = tqdm_class or hf_tqdm_lib
        except ImportError:
            # Fallback if huggingface_hub is not available (shouldn't happen in our context)
            hf_tqdm = tqdm_class
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
                        # Read current progress from libtorrent (may be non-zero if fastresume loaded)
                        initial_progress = 0
                        try:
                            progresses = self.handle.file_progress()
                            initial_progress = progresses[file_index]
                        except Exception:
                            pass
                        pbar = hf_tqdm(
                            total=total_size,
                            initial=initial_progress,
                            unit='B',
                            unit_scale=True,
                            desc=f"{filename} (P2P)"
                        )
                        last_progress = initial_progress
                    
                    try:
                        progresses = self.handle.file_progress()
                        current_progress = progresses[file_index]
                        if current_progress > last_progress:
                            # Avoid over-updating standard totals
                            if pbar:
                                pbar.update(current_progress - last_progress)
                            last_progress = current_progress
                        
                        # Sample peer stats while peers are still connected
                        self._snapshot_peer_stats()

                        postfix = _format_live_transfer_postfix(self.get_p2p_stats())
                        if hasattr(pbar, 'set_postfix_str'):
                            pbar.set_postfix_str(postfix, refresh=False)
                        elif postfix and hasattr(pbar, 'set_postfix'):
                            pbar.set_postfix({"status": postfix})
                    except Exception:
                        pass

        finally:
            if pbar is not None:
                if event.is_set() and last_progress < total_size:
                    pbar.update(total_size - last_progress)
                if hasattr(pbar, 'close'):
                    pbar.close()

        if event.is_set():
            logger.info(f"[{self.repo_id}] P2P download of {filename} SUCCESS.")
            return True
        else:
            logger.warning(f"[{self.repo_id}] P2P download of {filename} TIMEOUT after {self.timeout}s.")
            return False

    def _snapshot_peer_stats(self) -> None:
        """Sample current peer stats and update the accumulated high-water marks.

        ``get_peer_info()`` only returns *currently connected* peers.  Once a
        peer (including WebSeed) disconnects, its ``total_download`` is lost.
        By sampling periodically during the download and keeping the highest
        values seen, we preserve an accurate breakdown for the post-download
        summary.
        """
        try:
            with self.lock:
                handle = self.handle
            if not handle or not handle.is_valid():
                return

            peers = handle.get_peer_info()
            web_seed_flag = getattr(lt.peer_info, 'web_seed', None)
            total_payload_download = 0
            try:
                total_payload_download = handle.status().total_payload_download
            except Exception:
                pass

            peer_dl = 0
            webseed_dl = 0
            p2p_peers = 0

            for peer in peers:
                dl = getattr(peer, 'total_download', 0)
                if web_seed_flag is not None and (peer.flags & web_seed_flag):
                    webseed_dl += dl
                else:
                    peer_dl += dl
                    p2p_peers += 1

            with self._stats_lock:
                self._acc_peer_download = max(self._acc_peer_download, peer_dl)
                self._acc_webseed_download = max(self._acc_webseed_download, webseed_dl)
                self._acc_total_payload_download = max(
                    self._acc_total_payload_download,
                    total_payload_download,
                )
                self._acc_peak_p2p_peers = max(self._acc_peak_p2p_peers, p2p_peers)
        except Exception:
            pass

    def get_p2p_stats(self) -> dict:
        """Collect P2P transfer statistics from the current session.

        Distinguishes between bytes received from true P2P peers vs WebSeed
        (HTTP) sources.  Prefers live ``get_peer_info()`` data when available,
        but falls back to the accumulated high-water marks captured during the
        download (peers may have already disconnected by the time this is called).

        Returns:
            Dictionary with peer_download, webseed_download, total_payload_download,
            active_p2p_peers, peak_p2p_peers, num_webseeds, num_peers, num_seeds.
            Empty dict if handle is invalid and no accumulated data exists.
        """
        with self.lock:
            handle = self.handle

        # Try live data first
        peer_download = 0
        webseed_download = 0
        active_p2p_peers = 0
        num_webseeds = 0
        total_payload_download = 0
        num_peers = 0
        num_seeds = 0

        if handle and handle.is_valid():
            try:
                s = handle.status()
                total_payload_download = s.total_payload_download
                num_peers = s.num_peers
                num_seeds = s.num_seeds
                peers = handle.get_peer_info()

                web_seed_flag = getattr(lt.peer_info, 'web_seed', None)

                for peer in peers:
                    dl = getattr(peer, 'total_download', 0)
                    if web_seed_flag is not None and (peer.flags & web_seed_flag):
                        webseed_download += dl
                        num_webseeds += 1
                    else:
                        peer_download += dl
                        active_p2p_peers += 1
            except Exception as e:
                logger.debug(f"[{self.repo_id}] Failed to collect live P2P stats: {e}")

        with self._stats_lock:
            acc_peer_download = self._acc_peer_download
            acc_webseed_download = self._acc_webseed_download
            acc_total_payload_download = self._acc_total_payload_download
            acc_peak_p2p_peers = self._acc_peak_p2p_peers

        peer_download = max(peer_download, acc_peer_download)
        webseed_download = max(webseed_download, acc_webseed_download)
        total_payload_download = max(total_payload_download, acc_total_payload_download)
        peak_p2p_peers = max(active_p2p_peers, acc_peak_p2p_peers)

        # ``total_payload_download`` is the most trustworthy all-time byte count.
        # ``get_peer_info()`` can undercount WebSeed after disconnects, so
        # reconcile the source breakdown against the authoritative total.
        if total_payload_download > 0:
            peer_download = min(peer_download, total_payload_download)
            if peak_p2p_peers == 0:
                webseed_download = max(webseed_download, total_payload_download)
            else:
                webseed_download = max(
                    webseed_download,
                    total_payload_download - peer_download,
                )

        if peer_download == 0 and webseed_download == 0 and total_payload_download == 0:
            return {}

        return {
            'total_payload_download': total_payload_download,
            'peer_download': peer_download,
            'webseed_download': webseed_download,
            'active_p2p_peers': active_p2p_peers,
            'peak_p2p_peers': peak_p2p_peers,
            'num_webseeds': num_webseeds,
            'num_peers': num_peers,
            'num_seeds': num_seeds,
        }

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

    def _get_overlapping_file_indices(self, file_index: int, *, torrent_info=None) -> list:
        """Find file indices that share piece boundaries with the given file.

        Works around a libtorrent partfile bug where WebSeed downloads fail
        with "End of file" when a piece spans a priority-0 file boundary.
        See docs/libtorrent_partfile_eof_bug_fix.md for details.

        Args:
            file_index: The index of the target file.
            torrent_info: Optional torrent_info object. Falls back to
                self.torrent_info_obj when not provided (needed during
                _init_torrent before the handle exists).

        Returns:
            List of file indices whose byte range overlaps with the
            target file's piece range.
        """
        ti = torrent_info or self.torrent_info_obj
        if not ti:
            return []

        piece_len = ti.piece_length()
        files = ti.files()
        s_off = files.file_offset(file_index)
        e_off = s_off + files.file_size(file_index) - 1
        s_piece = s_off // piece_len
        e_piece = e_off // piece_len

        overlapping = []
        for i in range(files.num_files()):
            if i == file_index or files.file_size(i) == 0:
                continue
            fs = files.file_offset(i)
            fe = fs + files.file_size(i) - 1
            if fs // piece_len <= e_piece and fe // piece_len >= s_piece:
                overlapping.append(i)

        return overlapping

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
            is_padding_file,
        )

        logger.info(f"[{self.repo_id}] Mapping all files for background seeding...")
        files = self.torrent_info_obj.files()
        required_count = 0
        for file_index in range(files.num_files()):
            lt_path = files.file_path(file_index).replace('\\', '/')
            if not is_padding_file(strip_torrent_root(lt_path)):
                required_count += 1
        self.seeding_total_files = required_count
        self.seeding_mapped_files = 0
        self.full_mapping = False

        try:
            hardlinks, mapped_count = hardlink_files_for_seeding(
                self.torrent_info_obj, self.temp_dir, self.repo_id, self.revision,
                repo_type=self.repo_type, cache_dir=self.cache_dir, local_dir=self.local_dir
            )
            self.seeding_hardlinks = hardlinks
            self.seeding_mapped_files = mapped_count

            logger.info(f"[{self.repo_id}] Hardlinked {mapped_count}/{required_count} required files for seeding.")
            if mapped_count != required_count:
                logger.warning(
                    f"[{self.repo_id}] Refusing seed_mode: mapped "
                    f"{mapped_count}/{required_count} required files."
                )
                cleanup_hardlinks(self.repo_id, self.seeding_hardlinks)
                self.seeding_hardlinks = []
                return False

            self.handle.set_flags(lt.torrent_flags.seed_mode)
            self.handle.resume()
            self.full_mapping = True
            logger.info(f"[{self.repo_id}] Seeding started with seed_mode (0s startup, lazy verification).")
            return True

        except OSError:
            # Hardlink failed (cross-filesystem) — fall back to legacy
            logger.warning(f"[{self.repo_id}] Hardlink failed (cross-filesystem?). Falling back to force_recheck.")
            cleanup_hardlinks(self.repo_id, getattr(self, 'seeding_hardlinks', []))
            self.seeding_hardlinks = []

            mapped_count = rename_files_for_seeding(
                self.handle, self.torrent_info_obj,
                self.temp_dir, self.repo_id, self.revision,
                repo_type=self.repo_type, cache_dir=self.cache_dir, local_dir=self.local_dir
            )
            self.seeding_mapped_files = mapped_count
            if mapped_count != required_count:
                logger.warning(
                    f"[{self.repo_id}] Refusing partial legacy mapping: "
                    f"{mapped_count}/{required_count} required files."
                )
                return False
            self.full_mapping = True
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
        """Delete this torrent's subdirectory within p2p_root.

        Each torrent stores its files under ``p2p_root/{torrent_name}/``.
        We only clean up this torrent's subdirectory, leaving other
        sessions' data intact.
        """
        if not self.temp_dir or not self.torrent_info_obj:
            return
        try:
            torrent_subdir = os.path.join(self.temp_dir, self.torrent_info_obj.name())
            if os.path.isdir(torrent_subdir):
                shutil.rmtree(torrent_subdir, ignore_errors=True)
                logger.debug(f"[{self.repo_id}] Cleaned up torrent dir: {torrent_subdir}")
        except Exception as e:
            logger.debug(f"[{self.repo_id}] Failed to clean torrent dir: {e}")
