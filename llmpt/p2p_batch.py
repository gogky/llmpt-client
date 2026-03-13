"""
P2P Batch Manager for coordinating libtorrent downloads and HTTP intercepts.

This module provides a singleton P2PBatchManager that aggregates concurrent 
HTTP download requests into a single, efficient BitTorrent session with 
dynamic file prioritization.
"""

import os
import time
import threading
import logging
from typing import TYPE_CHECKING, Dict, Any, Optional

if TYPE_CHECKING:
    from .tracker_client import TrackerClient

from .utils import lt, LIBTORRENT_AVAILABLE, get_hf_hub_cache

# Re-export SessionContext for backward compatibility.
# All existing imports like ``from llmpt.p2p_batch import SessionContext`` continue to work.
from .session_context import SessionContext  # noqa: F401

logger = logging.getLogger(__name__)

# Default port range for libtorrent listen interface.
# Daemon (seeding) gets priority for the lower port; download clients
# use the next port so the two never clash even when a fixed port is
# specified by the user.
_DEFAULT_DAEMON_PORT = 6881
_DEFAULT_CLIENT_PORT = 6882
_MAX_PORT = 6999


def _storage_identity(
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> tuple[str, str]:
    """Return a normalized storage identity for session-level deduping."""
    if local_dir:
        return ("local_dir", os.path.realpath(os.path.abspath(os.path.expanduser(local_dir))))
    if cache_dir:
        return ("hub_cache", os.path.realpath(os.path.abspath(os.path.expanduser(cache_dir))))
    return ("hub_cache", get_hf_hub_cache())


def _is_port_available(port: int) -> bool:
    """Check whether *port* is free on **both** IPv4 and IPv6.

    On systems where IPv6 is not available (cannot create an ``AF_INET6``
    socket), the IPv6 check is silently skipped so that IPv4-only hosts
    are not penalised.
    """
    import socket

    # ── IPv4 ──────────────────────────────────────────────────────────
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', port))
    except OSError:
        return False

    # ── IPv6 ──────────────────────────────────────────────────────────
    try:
        s6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        # Kernel / Python built without IPv6 → skip check
        return True

    try:
        s6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, 'IPV6_V6ONLY'):
            s6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        s6.bind(('::', port))
        return True
    except OSError:
        return False
    finally:
        s6.close()


def _resolve_listen_interfaces(configured_port, role: str = 'daemon') -> str:
    """Resolve the ``listen_interfaces`` setting for libtorrent.

    Port assignment strategy (avoids daemon / client port conflicts)::

        ┌──────────────────────┬────────────┬─────────────┐
        │ configured_port      │  daemon    │  client     │
        ├──────────────────────┼────────────┼─────────────┤
        │ None / 0  (default)  │  6881      │  6882       │
        │ N  (explicit)        │  N         │  N + 1      │
        └──────────────────────┴────────────┴─────────────┘

    Starting from the target port, the function probes upward through
    *_MAX_PORT* checking **both IPv4 and IPv6** availability via
    ``_is_port_available``.  If every port in the range is busy it falls
    back to ``0`` (OS-assigned).

    There is a tiny TOCTOU window between releasing the probe socket and
    libtorrent binding, but in practice this is reliable enough and
    libtorrent will simply skip a failed interface.
    """
    has_explicit = configured_port and configured_port > 0

    if role == 'daemon':
        start_port = configured_port if has_explicit else _DEFAULT_DAEMON_PORT
    else:
        start_port = (configured_port + 1) if has_explicit else _DEFAULT_CLIENT_PORT

    end_port = max(start_port + (_MAX_PORT - _DEFAULT_DAEMON_PORT), _MAX_PORT) + 1
    for port in range(start_port, end_port):
        if _is_port_available(port):
            return f'0.0.0.0:{port},[::]:{port}'

    logger.warning(
        f"All ports {start_port}-{end_port - 1} occupied, "
        "falling back to OS-assigned port"
    )
    return '0.0.0.0:0,[::]:0'


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
            self.sessions: Dict[tuple, SessionContext] = {}
            if LIBTORRENT_AVAILABLE:
                from . import get_config
                config = get_config()
                configured_port = config.get('port')
                role = config.get('_role', 'client')
                # Resolve listen interfaces BEFORE creating lt.session()
                # so that lt.session() doesn't implicitly take 6881 before we check it.
                listen_ifaces = _resolve_listen_interfaces(configured_port, role=role)
                self.lt_session = lt.session()
                settings = self.lt_session.get_settings()
                settings['listen_interfaces'] = listen_ifaces
                self.lt_session.apply_settings(settings)
                self.listen_port = self.lt_session.listen_port()
                logger.info(f"libtorrent listening on port {self.listen_port}")
            else:
                self.lt_session = None
                self.listen_port = None

    def dispatch_alerts(self) -> None:
        """Pop alerts from the global lt_session and route each to the correct SessionContext.

        Each libtorrent session has a single alert queue shared by all torrent
        handles.  Previously, individual monitor threads called pop_alerts()
        independently, causing alerts belonging to *other* handles to be
        silently discarded (the "alert race" bug).

        This method should be called by each monitor thread before processing
        its own alerts.  It is safe to call concurrently — the _lock ensures
        only one thread pops at a time, and alerts are deposited into each
        SessionContext's thread-safe ``pending_alerts`` inbox.
        """
        if not self.lt_session:
            return

        with self._lock:
            alerts = self.lt_session.pop_alerts()
            if not alerts:
                return

            # Build a handle → SessionContext lookup (only active sessions)
            handle_to_ctx = {}
            for ctx in self.sessions.values():
                if ctx.handle is not None:
                    handle_to_ctx[ctx.handle] = ctx

            for alert in alerts:
                target_handle = getattr(alert, 'handle', None)
                target_ctx = handle_to_ctx.get(target_handle)
                if target_ctx is not None:
                    with target_ctx.alert_lock:
                        target_ctx.pending_alerts.append(alert)
                # Alerts without a matching handle (e.g. session-level alerts
                # like listen_succeeded_alert) are intentionally dropped here;
                # they are informational and not required for correctness.

    def register_seeding_task(self, repo_id: str, revision: str, tracker_client: 'TrackerClient', torrent_data: Optional[bytes] = None, *, repo_type: str = 'model', cache_dir: Optional[str] = None, local_dir: Optional[str] = None) -> bool:
        """
        Register a repository to be tracked for background seeding.
        This behaves like a download but without any specific HTTP interception blocks.
        """
        if not LIBTORRENT_AVAILABLE:
            return False

        storage_kind, storage_root = _storage_identity(cache_dir=cache_dir, local_dir=local_dir)
        repo_key = (repo_type, repo_id, revision, storage_kind, storage_root)
        with self._lock:
            if repo_key not in self.sessions:
                self.sessions[repo_key] = SessionContext(
                    repo_id=repo_id,
                    revision=revision,
                    repo_type=repo_type,
                    tracker_client=tracker_client,
                    lt_session=self.lt_session,
                    session_mode='full_seed',
                    timeout=0,  # unused: seeding path never calls download_file()
                    torrent_data=torrent_data,
                    cache_dir=cache_dir,
                    local_dir=local_dir,
                )
            session_ctx = self.sessions[repo_key]
            
        # Ensure torrent is initialized with .torrent data from tracker
        if not session_ctx.is_valid:
            return False
            
        success = session_ctx._init_torrent()
        if not success:
            logger.error(f"[{repo_id}] Failed to initialize seeding session.")
            return False
                
        # For seeding, force map all files in the torrent to their HF blob equivalents!
        success = session_ctx.map_all_files_for_seeding()
        if not success:
            self.remove_session(
                repo_id,
                revision,
                repo_type=repo_type,
                cache_dir=cache_dir,
                local_dir=local_dir,
            )
        return success

    def register_request(
        self,
        repo_id: str,
        revision: str,
        filename: str,
        temp_file_path: str,
        tracker_client: 'TrackerClient',
        timeout: int = 300,
        *,
        repo_type: str = 'model',
        cache_dir: Optional[str] = None,
        local_dir: Optional[str] = None,
        tqdm_class: Optional[Any] = None,
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
            repo_type: Repository type.
            
        Returns:
            True if P2P download succeeds, False if it failed and should fallback to HTTP.
        """
        if not LIBTORRENT_AVAILABLE:
            return False

        storage_kind, storage_root = _storage_identity(cache_dir=cache_dir, local_dir=local_dir)
        repo_key = (repo_type, repo_id, revision, storage_kind, storage_root)
        
        with self._lock:
            if repo_key not in self.sessions:
                from . import get_config
                config = get_config()
                self.sessions[repo_key] = SessionContext(
                    repo_id=repo_id,
                    revision=revision,
                    repo_type=repo_type,
                    tracker_client=tracker_client,
                    lt_session=self.lt_session,
                    session_mode='on_demand',
                    timeout=timeout,
                    cache_dir=cache_dir,
                    local_dir=local_dir,
                )
            session_ctx = self.sessions[repo_key]
        
        # Register the file with the session context and wait for it
        return session_ctx.download_file(filename, temp_file_path, tqdm_class=tqdm_class)

    def release_on_demand_session(
        self,
        repo_id: str,
        revision: str,
        *,
        repo_type: str = 'model',
        cache_dir: Optional[str] = None,
        local_dir: Optional[str] = None,
        completed: bool = False,
    ) -> bool:
        """Remove an idle on-demand session after handoff to the daemon.

        This is intended for the download client process only. The long-lived
        background seeding role belongs to the daemon's ``full_seed`` session;
        once a user-facing download operation completes, the temporary
        on-demand session should go away promptly to avoid duplicate seeders.
        """
        storage_kind, storage_root = _storage_identity(cache_dir=cache_dir, local_dir=local_dir)
        repo_key = (repo_type, repo_id, revision, storage_kind, storage_root)
        worker = None

        with self._lock:
            ctx = self.sessions.get(repo_key)
            if ctx is None or ctx.session_mode != 'on_demand':
                return False

        if not completed:
            self._checkpoint_on_demand_session(ctx)

        with self._lock:
            ctx = self.sessions.get(repo_key)
            if ctx is None or ctx.session_mode != 'on_demand':
                return False
            ctx = self.sessions.pop(repo_key)
            worker = self._teardown_session(
                ctx,
                purge_resumable_state=completed,
            )

        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=3)
        return True

    # ── Session lifecycle management ──────────────────────────────────────

    def _checkpoint_on_demand_session(
        self,
        ctx: 'SessionContext',
        *,
        timeout: float = 1.5,
    ) -> None:
        """Best-effort final fastresume save before tearing down an incomplete session."""
        if ctx.session_mode != 'on_demand':
            return

        with ctx.lock:
            handle = ctx.handle

        if not handle:
            return

        try:
            if not handle.is_valid():
                return
        except Exception:
            return

        fastresume_path = getattr(ctx, 'fastresume_path', None)
        baseline_mtime = None
        if fastresume_path and os.path.exists(fastresume_path):
            try:
                baseline_mtime = os.stat(fastresume_path).st_mtime_ns
            except OSError:
                baseline_mtime = None

        try:
            handle.save_resume_data(lt.save_resume_flags_t.flush_disk_cache)
        except Exception as exc:
            logger.debug(f"[{ctx.repo_id}] Final save_resume_data skipped: {exc}")
            return

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.dispatch_alerts()
            except Exception:
                pass

            try:
                from .monitor import _process_alerts

                _process_alerts(ctx)
            except Exception:
                pass

            if fastresume_path:
                try:
                    if os.path.exists(fastresume_path):
                        current_mtime = os.stat(fastresume_path).st_mtime_ns
                        if baseline_mtime is None or current_mtime != baseline_mtime:
                            return
                except OSError:
                    pass

            time.sleep(0.05)

    def _teardown_session(
        self,
        ctx: 'SessionContext',
        *,
        purge_resumable_state: bool = True,
    ) -> Optional[Any]:
        """Teardown a single session (called while self._lock is held).

        Cleans up hardlinks, invalidates
        the handle, removes the torrent from libtorrent, and returns the
        worker thread (if any) for the caller to join *outside* the lock.

        Args:
            ctx: The SessionContext to tear down.

        Returns:
            The worker thread to join, or None.
        """
        ctx._cleanup_seeding_hardlinks()
        with ctx.lock:
            handle = ctx.handle
            ctx.handle = None
            ctx.is_valid = False
        if handle:
            try:
                self.lt_session.remove_torrent(handle)
            except Exception:
                pass
        if purge_resumable_state:
            ctx.cleanup_temp_dir()
            ctx.cleanup_fastresume()
        return ctx.worker_thread

    def remove_session(self, repo_id: str, revision: str, *, repo_type: str = 'model', cache_dir: Optional[str] = None, local_dir: Optional[str] = None) -> bool:
        """Remove a single session (stop seeding a specific repo).

        Args:
            repo_id: HuggingFace repository ID.
            revision: Revision string.
            repo_type: Repository type.
            cache_dir: Optional cache directory.
            local_dir: Optional local directory.

        Returns:
            True if the session was found and removed, False otherwise.
        """
        storage_kind, storage_root = _storage_identity(cache_dir=cache_dir, local_dir=local_dir)
        repo_key = (repo_type, repo_id, revision, storage_kind, storage_root)
        worker = None

        with self._lock:
            if repo_key not in self.sessions:
                return False
            ctx = self.sessions.pop(repo_key)
            worker = self._teardown_session(ctx)

        if worker is not None:
            worker.join(timeout=3)
        return True

    def remove_all_sessions(self) -> int:
        """Remove all sessions.

        Returns:
            Number of sessions removed.
        """
        threads_to_join = []
        with self._lock:
            contexts = list(self.sessions.values())
        for ctx in contexts:
            if ctx.session_mode == 'on_demand':
                self._checkpoint_on_demand_session(ctx)
        with self._lock:
            count = len(self.sessions)
            for ctx in self.sessions.values():
                worker = self._teardown_session(
                    ctx,
                    purge_resumable_state=(ctx.session_mode != 'on_demand'),
                )
                if worker is not None:
                    threads_to_join.append(worker)
            self.sessions.clear()

        # Wait for monitor threads outside the lock to avoid deadlock
        # (monitor threads may try to acquire manager._lock via dispatch_alerts).
        for t in threads_to_join:
            t.join(timeout=3)

        return count

    def get_all_session_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active sessions with valid handles.

        Returns:
            Dictionary keyed by ``"repo_type:repo_id@revision"`` with status info.
        """
        status = {}
        with self._lock:
            for (repo_type, repo_id, revision, storage_kind, storage_root), ctx in self.sessions.items():
                if not ctx.handle or not ctx.handle.is_valid():
                    continue
                s = ctx.handle.status()
                # Represent the session uniquely
                key_suffix = f" ({storage_kind}={storage_root})" if storage_root else ""
                status[f"{repo_type}:{repo_id}@{revision}{key_suffix}"] = {
                    'repo_type': repo_type,
                    'repo_id': repo_id,
                    'revision': revision,
                    'cache_dir': ctx.cache_dir,
                    'local_dir': ctx.local_dir,
                    'uploaded': s.total_upload,
                    'peers': s.num_peers,
                    'upload_rate': s.upload_rate,
                    'progress': s.progress,
                    'state': str(s.state),
                    'mapped_files': ctx.seeding_mapped_files,
                    'total_files': ctx.seeding_total_files,
                    'full_mapping': ctx.full_mapping,
                }
        return status

    def get_repo_p2p_stats(
        self, repo_id: str, revision: str, repo_type: str = 'model'
    ) -> Dict[str, Any]:
        """Aggregate P2P stats across all sessions matching the given repo.

        Iterates over all active sessions whose repo identity matches and
        collects byte-level breakdowns (peer vs WebSeed) via
        ``SessionContext.get_p2p_stats()``.

        Args:
            repo_id: HuggingFace repository ID.
            revision: Resolved revision (commit hash).
            repo_type: Repository type.

        Returns:
            Aggregated stats dict with peer_download, webseed_download,
            total_payload_download, active_p2p_peers, max_p2p_peers.
        """
        total_peer_download = 0
        total_webseed_download = 0
        total_payload_download = 0
        active_p2p_peers = 0
        max_p2p_peers = 0

        with self._lock:
            for key, ctx in self.sessions.items():
                key_repo_type, key_repo_id, key_revision = key[:3]
                if (
                    key_repo_type == repo_type
                    and key_repo_id == repo_id
                    and key_revision == revision
                ):
                    stats = ctx.get_p2p_stats()
                    if stats:
                        total_peer_download += stats.get('peer_download', 0)
                        total_webseed_download += stats.get('webseed_download', 0)
                        total_payload_download += stats.get('total_payload_download', 0)
                        active_p2p_peers += stats.get('active_p2p_peers', 0)
                        max_p2p_peers = max(
                            max_p2p_peers, stats.get('peak_p2p_peers', 0)
                        )

        return {
            'peer_download': total_peer_download,
            'webseed_download': total_webseed_download,
            'total_payload_download': total_payload_download,
            'active_p2p_peers': active_p2p_peers,
            'max_p2p_peers': max_p2p_peers,
        }

    def shutdown(self) -> None:
        """Gracefully shut down all sessions and release resources.

        Called by atexit handler or directly by user code.  Cleans up
        seeding hardlinks, removes all torrent
        handles, and clears session state.  Waits for all monitor threads
        to exit before returning.
        """
        count = self.remove_all_sessions()
        logger.info(f"P2PBatchManager shutdown complete ({count} sessions removed).")
