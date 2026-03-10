"""
Monkey Patch implementation for huggingface_hub.

This module patches huggingface_hub's download functions to enable P2P acceleration.
"""

import sys
import atexit
import time
import threading
import logging
from typing import Optional, Any

logger = logging.getLogger('llmpt.patch')

# Store original functions
_original_hf_hub_download = None
_original_http_get = None
_original_snapshot_download = None
_original_snapshot_hf_tqdm = None

# Patching configuration (set by apply_patch)
_config = {}

# Thread-local storage for P2P context
_context = threading.local()

# Download statistics: tracks which files went through P2P vs HTTP fallback
_stats_lock = threading.Lock()
_download_stats = {
    'p2p': set(),      # filenames successfully delivered via P2P
    'http': set(),     # filenames that fell back to HTTP
}

# Fallback daemon notification via debounce.
# When _patched_snapshot_download runs, it increments _active_wrapper_counts[repo_id]
# so the fallback in _patched_hf_hub_download is suppressed.  When the user
# imports snapshot_download BEFORE enable_p2p(), _patched_snapshot_download
# never runs, and this fallback fires after a 2-second quiet period.
_deferred_lock = threading.Lock()
_deferred_timers: dict[tuple[str, str, str, str, str], threading.Timer] = {}
_deferred_contexts: dict[tuple[str, str, str, str, str], dict] = {}
_active_wrapper_counts: dict[str, int] = {}  # repo_id -> active wrapper depth
_active_download_counts: dict[str, int] = {}  # repo_id -> in-flight hf_hub_download calls
_SNAPSHOT_PROGRESS_BAR_NAME = "huggingface_hub.snapshot_download"
_SNAPSHOT_PROGRESS_UPDATE_INTERVAL = 0.25


def _deferred_key(
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> tuple[str, str, str, str, str]:
    """Build the dedupe key for deferred daemon notifications.

    The content identity is ``(repo_type, repo_id, revision)``.  Storage
    identity is tracked separately so simultaneous downloads of the same repo
    into different roots do not stomp each other.
    """
    storage_kind = "local_dir" if local_dir else "hub_cache"
    storage_root = local_dir or cache_dir or ""
    return (repo_type or "model", repo_id, revision or "main", storage_kind, storage_root)


def _is_wrapper_active(repo_id: str) -> bool:
    """Return True if patched snapshot_download is currently wrapping this repo."""
    with _deferred_lock:
        return _active_wrapper_counts.get(repo_id, 0) > 0


def _enter_wrapper(repo_id: str) -> None:
    """Increment active wrapper depth for a repo."""
    with _deferred_lock:
        _active_wrapper_counts[repo_id] = _active_wrapper_counts.get(repo_id, 0) + 1


def _exit_wrapper(repo_id: str) -> None:
    """Decrement active wrapper depth for a repo."""
    with _deferred_lock:
        depth = _active_wrapper_counts.get(repo_id, 0)
        if depth <= 1:
            _active_wrapper_counts.pop(repo_id, None)
        else:
            _active_wrapper_counts[repo_id] = depth - 1

def _flush_deferred_notifications():
    """Flush pending deferred notifications on process exit.

    The deferred timer threads are daemon threads, so they get killed
    when the main thread exits.  This atexit handler ensures that the
    verbose summary and daemon notifications fire even if the process
    exits within the 2-second debounce window.
    """
    with _deferred_lock:
        contexts = list(_deferred_contexts.values())
        for timer in _deferred_timers.values():
            timer.cancel()
        _deferred_timers.clear()
        _deferred_contexts.clear()

    for ctx in contexts:
        # Print verbose summary if enabled
        if _config.get('verbose'):
            elapsed = time.time() - ctx.get('start_time', time.time())
            _print_p2p_summary(
                stats=get_download_stats(),
                elapsed=elapsed,
                repo_id=ctx['repo_id'],
                resolved_revision=ctx['revision'],
                repo_type=ctx['repo_type'],
            )

        try:
            _notify_seed_daemon(
                repo_id=ctx['repo_id'],
                revision=ctx['revision'],
                repo_type=ctx['repo_type'],
                cache_dir=ctx.get('cache_dir'),
                local_dir=ctx.get('local_dir'),
                completed_snapshot=bool(ctx.get('completed_snapshot')),
            )
        except Exception:
            pass

atexit.register(_flush_deferred_notifications)


def get_download_stats() -> dict:
    """Return a snapshot of download statistics.

    Returns:
        Dictionary with 'p2p' and 'http' sets of filenames.
    """
    with _stats_lock:
        return {
            'p2p': set(_download_stats['p2p']),
            'http': set(_download_stats['http']),
        }


def reset_download_stats() -> None:
    """Clear all download statistics."""
    with _stats_lock:
        _download_stats['p2p'].clear()
        _download_stats['http'].clear()


def _truncate_temp_file(temp_file, filename: str) -> None:
    """Truncate temp file before HTTP fallback to prevent double-write.

    When P2P fails, libtorrent may have partially written data to temp_file.
    Truncating ensures the subsequent HTTP download starts from scratch.
    """
    try:
        temp_file.seek(0)
        temp_file.truncate(0)
        logger.debug(f"[P2P] Truncated temp_file for clean HTTP fallback: {filename}")
    except Exception as e:
        logger.warning(f"[P2P] Could not truncate temp_file for {filename}: {e}")


def _format_snapshot_p2p_postfix(stats: Optional[dict]) -> str:
    """Format live repo-level P2P stats for the snapshot progress bar."""
    stats = stats or {}
    peer_bytes = int(stats.get('peer_download', 0) or 0)
    webseed_bytes = int(stats.get('webseed_download', 0) or 0)
    peers = int(stats.get('active_p2p_peers', stats.get('max_p2p_peers', 0)) or 0)
    return (
        f"P2P {_format_bytes(peer_bytes)} | "
        f"WebSeed {_format_bytes(webseed_bytes)} | "
        f"Active peers {peers}"
    )


class _SnapshotProgressReporter:
    """Periodically updates snapshot_download's shared progress bar postfix."""

    def __init__(self, progress_bar: Any, repo_id: str, revision: str, repo_type: str) -> None:
        self.progress_bar = progress_bar
        self.repo_id = repo_id
        self.revision = revision
        self.repo_type = repo_type
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None or not self.repo_id:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="llmpt-snapshot-progress",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._update_once()
            self._stop_event.wait(_SNAPSHOT_PROGRESS_UPDATE_INTERVAL)
        self._update_once()

    def _update_once(self) -> None:
        try:
            from .p2p_batch import P2PBatchManager

            stats = P2PBatchManager().get_repo_p2p_stats(
                self.repo_id,
                self.revision,
                self.repo_type,
            )
            postfix = _format_snapshot_p2p_postfix(stats)
            if hasattr(self.progress_bar, 'set_postfix_str'):
                self.progress_bar.set_postfix_str(postfix, refresh=False)
        except Exception as e:
            logger.debug(f"[P2P] Live snapshot progress update failed: {e}")


def _wrap_snapshot_tqdm_class(
    base_tqdm_class: Any,
    repo_id: str,
    revision: str,
    repo_type: str,
):
    """Wrap a tqdm class so snapshot_download's shared bar shows live P2P stats."""

    class SnapshotTqdmProxy:
        _lock = getattr(base_tqdm_class, '_lock', None)

        @classmethod
        def get_lock(cls):
            return getattr(cls, '_lock', None) or base_tqdm_class.get_lock()

        @classmethod
        def set_lock(cls, lock):
            cls._lock = lock
            return base_tqdm_class.set_lock(lock)

        def __init__(self, *args, **kwargs):
            object.__setattr__(self, '_llmpt_inner', base_tqdm_class(*args, **kwargs))
            object.__setattr__(self, '_llmpt_reporter', None)

            is_snapshot_bar = kwargs.get('name') == _SNAPSHOT_PROGRESS_BAR_NAME
            is_disabled = getattr(self._llmpt_inner, 'disable', False)
            if is_snapshot_bar and not is_disabled:
                reporter = _SnapshotProgressReporter(
                    progress_bar=self._llmpt_inner,
                    repo_id=repo_id,
                    revision=revision,
                    repo_type=repo_type,
                )
                object.__setattr__(self, '_llmpt_reporter', reporter)
                reporter.start()

        def __getattr__(self, name):
            return getattr(self._llmpt_inner, name)

        def __setattr__(self, name, value):
            if name.startswith('_llmpt_'):
                object.__setattr__(self, name, value)
            else:
                setattr(self._llmpt_inner, name, value)

        def __enter__(self):
            entered = self._llmpt_inner.__enter__()
            return self if entered is self._llmpt_inner else entered

        def __exit__(self, exc_type, exc_value, traceback):
            return self._llmpt_inner.__exit__(exc_type, exc_value, traceback)

        def __iter__(self):
            return iter(self._llmpt_inner)

        def close(self):
            reporter = self._llmpt_reporter
            if reporter is not None:
                reporter.stop()
                object.__setattr__(self, '_llmpt_reporter', None)
            close_fn = getattr(self._llmpt_inner, 'close', None)
            if close_fn is not None:
                return close_fn()

    SnapshotTqdmProxy.__name__ = "SnapshotTqdmProxy"
    return SnapshotTqdmProxy


def _wrap_snapshot_tqdm_class_auto(base_tqdm_class: Any):
    """Wrap hf_tqdm so old snapshot_download references still get live P2P stats."""

    class SnapshotAutoTqdmProxy:
        _lock = getattr(base_tqdm_class, '_lock', None)

        @classmethod
        def get_lock(cls):
            return getattr(cls, '_lock', None) or base_tqdm_class.get_lock()

        @classmethod
        def set_lock(cls, lock):
            cls._lock = lock
            return base_tqdm_class.set_lock(lock)

        def __init__(self, *args, **kwargs):
            object.__setattr__(self, '_llmpt_inner', base_tqdm_class(*args, **kwargs))
            object.__setattr__(self, '_llmpt_reporter', None)

            is_snapshot_bar = kwargs.get('name') == _SNAPSHOT_PROGRESS_BAR_NAME
            is_disabled = getattr(self._llmpt_inner, 'disable', False)
            if is_snapshot_bar and not is_disabled:
                ctx = _extract_snapshot_context_from_stack()
                if ctx is not None:
                    reporter = _SnapshotProgressReporter(
                        progress_bar=self._llmpt_inner,
                        repo_id=ctx['repo_id'],
                        revision=ctx['revision'],
                        repo_type=ctx['repo_type'],
                    )
                    object.__setattr__(self, '_llmpt_reporter', reporter)
                    reporter.start()

        def __getattr__(self, name):
            return getattr(self._llmpt_inner, name)

        def __setattr__(self, name, value):
            if name.startswith('_llmpt_'):
                object.__setattr__(self, name, value)
            else:
                setattr(self._llmpt_inner, name, value)

        def __enter__(self):
            entered = self._llmpt_inner.__enter__()
            return self if entered is self._llmpt_inner else entered

        def __exit__(self, exc_type, exc_value, traceback):
            return self._llmpt_inner.__exit__(exc_type, exc_value, traceback)

        def __iter__(self):
            return iter(self._llmpt_inner)

        def close(self):
            reporter = self._llmpt_reporter
            if reporter is not None:
                reporter.stop()
                object.__setattr__(self, '_llmpt_reporter', None)
            close_fn = getattr(self._llmpt_inner, 'close', None)
            if close_fn is not None:
                return close_fn()

    SnapshotAutoTqdmProxy.__name__ = "SnapshotAutoTqdmProxy"
    return SnapshotAutoTqdmProxy


def _extract_context_from_stack() -> Optional[dict]:
    """Walk the call stack to extract download context from hf_hub_download.

    This is a fallback for when the user imports hf_hub_download BEFORE
    enable_p2p(), bypassing _patched_hf_hub_download's context injection.

    By the time http_get is called, the original hf_hub_download is still
    on the stack with all the context we need (repo_id, filename,
    commit_hash, etc.) as local variables.

    Returns:
        dict with repo_id, filename, revision, repo_type if found,
        or None if the frame cannot be located.
    """
    try:
        frame = sys._getframe(1)  # start from caller
        result = {"from_snapshot_download": False}
        for _ in range(60):       # walk up far enough for HF helper stacks
            frame = frame.f_back
            if frame is None:
                break
            name = frame.f_code.co_name
            if name in ('snapshot_download', '_inner_hf_hub_download'):
                result['from_snapshot_download'] = True
            if name in (
                'hf_hub_download',
                'snapshot_download',
                '_inner_hf_hub_download',
                '_hf_hub_download_to_cache_dir',
                '_hf_hub_download_to_local_dir'
            ):
                loc = frame.f_locals
                
                # Extract whatever we can find from this frame
                if 'repo_id' in loc and 'repo_id' not in result:
                    result['repo_id'] = loc.get('repo_id')
                if 'filename' in loc and 'filename' not in result:
                    result['filename'] = loc.get('filename')
                if 'repo_type' in loc and 'repo_type' not in result:
                    result['repo_type'] = loc.get('repo_type') or 'model'
                
                # Revisions and commit hashes
                rev = loc.get('commit_hash') or loc.get('revision')
                if rev and 'revision' not in result:
                    result['revision'] = rev

                # Subfolder resolution
                if 'subfolder' in loc and 'subfolder' not in result:
                    sub_val = loc.get('subfolder')
                    if sub_val and sub_val != '':
                        result['subfolder'] = sub_val

                # Directories
                for dir_key in ('cache_dir', 'local_dir'):
                    if dir_key in loc and dir_key not in result and loc[dir_key]:
                        result[dir_key] = loc[dir_key]
                    elif loc.get('kwargs') and dir_key in loc['kwargs'] and dir_key not in result and loc['kwargs'][dir_key]:
                        result[dir_key] = loc['kwargs'][dir_key]

        if result.get('repo_id') and result.get('filename'):
            # Default fallbacks
            result.setdefault('revision', 'main')
            result.setdefault('repo_type', 'model')
            
            # Apply subfolder if found
            if 'subfolder' in result:
                result['filename'] = f"{result['subfolder']}/{result['filename']}"
                
            return result
            
    except (AttributeError, ValueError):
        pass
    return None


def _extract_snapshot_context_from_stack() -> Optional[dict]:
    """Walk the call stack to recover snapshot_download context."""
    try:
        frame = sys._getframe(1)
        result = {}
        for _ in range(60):
            frame = frame.f_back
            if frame is None:
                break
            if frame.f_code.co_name not in ('snapshot_download', '_inner_hf_hub_download'):
                continue

            loc = frame.f_locals
            if 'repo_id' in loc and 'repo_id' not in result:
                result['repo_id'] = loc.get('repo_id')

            revision = loc.get('commit_hash') or loc.get('revision')
            if revision and 'revision' not in result:
                result['revision'] = revision

            if 'repo_type' not in result:
                result['repo_type'] = loc.get('repo_type') or 'model'

            for dir_key in ('cache_dir', 'local_dir'):
                if dir_key in loc and dir_key not in result and loc[dir_key]:
                    result[dir_key] = loc[dir_key]
                elif loc.get('kwargs') and dir_key in loc['kwargs'] and dir_key not in result and loc['kwargs'][dir_key]:
                    result[dir_key] = loc['kwargs'][dir_key]

            if result.get('repo_id') and result.get('revision'):
                return result
    except (AttributeError, ValueError):
        pass
    return None


def _fire_deferred_notification(key: tuple[str, str, str, str, str]) -> None:
    """Called by the debounce timer — sends a daemon seed notification.

    Also prints a P2P summary when verbose mode is enabled.  This is the
    fallback path for when ``_patched_snapshot_download`` was bypassed
    due to the user importing ``snapshot_download`` before calling
    ``enable_p2p()``.

    If downloads are still in progress for this repo, the timer is
    rescheduled instead of firing immediately.
    """
    with _deferred_lock:
        ctx = _deferred_contexts.get(key)
        if ctx is None:
            _deferred_timers.pop(key, None)
            return

        # If downloads are still in-flight, reschedule and wait.
        repo_id = ctx['repo_id']
        if _active_download_counts.get(repo_id, 0) > 0:
            old_timer = _deferred_timers.pop(key, None)
            if old_timer is not None:
                old_timer.cancel()
            new_timer = threading.Timer(
                2.0, _fire_deferred_notification, args=[key]
            )
            new_timer.daemon = True
            new_timer.start()
            _deferred_timers[key] = new_timer
            return

        # All downloads done — pop context and proceed.
        _deferred_contexts.pop(key, None)
        _deferred_timers.pop(key, None)

    # Print verbose summary if enabled (import-order bypass fallback)
    if _config.get('verbose'):
        elapsed = time.time() - ctx.get('start_time', time.time())
        _print_p2p_summary(
            stats=get_download_stats(),
            elapsed=elapsed,
            repo_id=ctx['repo_id'],
            resolved_revision=ctx['revision'],
            repo_type=ctx['repo_type'],
        )

    try:
        _notify_seed_daemon(
            repo_id=ctx['repo_id'],
            revision=ctx['revision'],
            repo_type=ctx['repo_type'],
            cache_dir=ctx.get('cache_dir'),
            local_dir=ctx.get('local_dir'),
            completed_snapshot=bool(ctx.get('completed_snapshot')),
        )
        logger.debug(
            f"[P2P] Deferred daemon notification sent for "
            f"{ctx['repo_id']}@{ctx['revision'][:8]}..."
        )
    except Exception as e:
        logger.debug(f"[P2P] Deferred daemon notification failed: {e}")


def _schedule_deferred_notification(
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    *,
    completed_snapshot: bool = False,
) -> None:
    """Schedule (or reschedule) a deferred daemon notification.

    Each call resets the 2-second timer.  When the timer finally fires
    (i.e. no new hf_hub_download calls for 2 s), we notify the daemon.
    This acts as a fallback for when _patched_snapshot_download is
    bypassed due to import-order issues.
    """
    key = _deferred_key(repo_id, revision, repo_type, cache_dir, local_dir)
    with _deferred_lock:
        # Cancel any existing timer for this specific repo+revision.
        timer = _deferred_timers.get(key)
        if timer is not None:
            timer.cancel()

        # Preserve start_time from the first schedule (for elapsed calculation)
        existing = _deferred_contexts.get(key)
        _deferred_contexts[key] = {
            'repo_id': repo_id,
            'revision': revision,
            'repo_type': repo_type,
            'cache_dir': cache_dir,
            'local_dir': local_dir,
            'completed_snapshot': (
                completed_snapshot or bool(existing and existing.get('completed_snapshot'))
            ),
            'start_time': existing['start_time'] if existing else time.time(),
        }
        
        # Start a new timer
        new_timer = threading.Timer(2.0, _fire_deferred_notification, args=[key])
        new_timer.daemon = True
        new_timer.start()
        _deferred_timers[key] = new_timer


def _notify_seed_daemon(
    *,
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    completed_snapshot: bool = False,
) -> None:
    from .ipc import notify_daemon

    notify_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "repo_type": repo_type,
    }
    if cache_dir is not None:
        notify_kwargs["cache_dir"] = cache_dir
    if local_dir is not None:
        notify_kwargs["local_dir"] = local_dir
    if completed_snapshot:
        notify_kwargs["completed_snapshot"] = True
    notify_daemon("seed", **notify_kwargs)


def _patched_hf_hub_download(repo_id: str, filename: str, **kwargs):
    """Patched version of hf_hub_download that injects P2P context."""
    from .tracker_client import TrackerClient
    from .utils import resolve_commit_hash

    # Track active downloads so the deferred summary waits for all to finish.
    with _deferred_lock:
        _active_download_counts[repo_id] = _active_download_counts.get(repo_id, 0) + 1

    # Query tracker for torrent info
    tracker = TrackerClient(_config['tracker_url'])

    # Resolve revision to a 40-char commit hash so the tracker query matches
    # what seeders registered.  If resolution fails (e.g. network error),
    # fall back to the raw value — the download still works via HTTP.
    raw_revision = kwargs.get('revision', 'main')
    repo_type = kwargs.get('repo_type', 'model')
    cache_dir = kwargs.get('cache_dir')
    local_dir = kwargs.get('local_dir')
    try:
        revision = resolve_commit_hash(repo_id, raw_revision, repo_type=repo_type)
    except Exception as e:
        logger.debug(f"[P2P] Could not resolve revision '{raw_revision}': {e}")
        revision = raw_revision

    # Resolve subfolder exactly as huggingface_hub does
    actual_filename = filename
    subfolder = kwargs.get('subfolder')
    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        actual_filename = f"{subfolder}/{filename}"

    # Backup previous context (in case of nested/recursive hf_hub_download calls)
    prev_repo_id = getattr(_context, 'repo_id', None)
    prev_repo_type = getattr(_context, 'repo_type', None)
    prev_filename = getattr(_context, 'filename', None)
    prev_revision = getattr(_context, 'revision', None)
    prev_tracker = getattr(_context, 'tracker', None)
    prev_config = getattr(_context, 'config', None)
    prev_cache_dir = getattr(_context, 'cache_dir', None)
    prev_local_dir = getattr(_context, 'local_dir', None)

    # Store context for http_get to use
    _context.repo_id = repo_id
    _context.repo_type = repo_type
    _context.filename = actual_filename
    _context.revision = revision
    _context.tracker = tracker
    _context.config = _config
    _context.cache_dir = cache_dir
    _context.local_dir = local_dir

    download_succeeded = False
    completed_snapshot = False
    try:
        # Call original function (will trigger patched http_get)
        result = _original_hf_hub_download(repo_id, filename, **kwargs)
        download_succeeded = True
        completed_snapshot = _extract_snapshot_context_from_stack() is not None
        return result
    finally:
        # Restore previous context instead of clearing it (supports recursion)
        _context.repo_id = prev_repo_id
        _context.repo_type = prev_repo_type
        _context.filename = prev_filename
        _context.revision = prev_revision
        _context.tracker = prev_tracker
        _context.config = prev_config
        _context.cache_dir = prev_cache_dir
        _context.local_dir = prev_local_dir

        # Decrement active download counter.
        with _deferred_lock:
            count = _active_download_counts.get(repo_id, 0)
            if count <= 1:
                _active_download_counts.pop(repo_id, None)
            else:
                _active_download_counts[repo_id] = count - 1

        # Fallback daemon notification: if _patched_snapshot_download is not
        # wrapping this call (import-order issue), schedule a deferred
        # notification so the daemon still learns about this download.
        if download_succeeded and not _is_wrapper_active(repo_id):
            _schedule_deferred_notification(
                repo_id,
                revision,
                repo_type,
                cache_dir,
                local_dir,
                completed_snapshot=completed_snapshot,
            )


def _patched_http_get(url: str, temp_file, **kwargs):
    """Patched version of http_get that uses P2P batch manager when available."""
    # Check if we have P2P context (injected by patched hf_hub_download)
    repo_id = getattr(_context, 'repo_id', None)
    repo_type = getattr(_context, 'repo_type', None) or 'model'
    filename = getattr(_context, 'filename', None)
    revision = getattr(_context, 'revision', None)
    tracker = getattr(_context, 'tracker', None)
    config = getattr(_context, 'config', {})
    cache_dir = getattr(_context, 'cache_dir', None)
    local_dir = getattr(_context, 'local_dir', None)
    schedule_deferred = False
    deferred_completed_snapshot = False
    truncated = False

    # Fallback: if _patched_hf_hub_download was bypassed (import-order issue),
    # try to recover context by inspecting the call stack.  The original
    # hf_hub_download is still on the stack with repo_id, filename, etc.
    if not (repo_id and filename and revision):
        stack_ctx = _extract_context_from_stack()
        if stack_ctx:
            from .tracker_client import TrackerClient
            repo_id = stack_ctx['repo_id']
            filename = stack_ctx['filename']
            revision = stack_ctx['revision']
            repo_type = stack_ctx['repo_type']
            cache_dir = stack_ctx.get('cache_dir')
            local_dir = stack_ctx.get('local_dir')
            tracker = TrackerClient(_config['tracker_url']) if _config.get('tracker_url') else None
            config = _config
            logger.debug(
                f"[P2P] Recovered context from stack frame: "
                f"{repo_id}/{filename}@{revision[:8]}..."
            )

            # Since we bypassed _patched_hf_hub_download, we must schedule the
            # deferred daemon notification here so the daemon knows to seed it.
            # Do this only after the file transfer succeeds.
            schedule_deferred = not _is_wrapper_active(repo_id)
            deferred_completed_snapshot = bool(stack_ctx.get('from_snapshot_download'))

    if repo_id and filename and tracker and revision:
        try:
            from .p2p_batch import P2PBatchManager
            logger.info(f"[P2P] Intercepted HTTP request for {repo_id}/{filename} (rev: {revision})")

            manager = P2PBatchManager()

            # With WebSeed enabled, download speed >= HTTP (WebSeed is a
            # guaranteed fallback source inside libtorrent). A fixed timeout
            # would only cause unnecessary HTTP fallbacks, so we disable it.
            # Without WebSeed (pure P2P), the timeout acts as a safety net.
            if config.get('webseed_proxy_port'):
                effective_timeout = 0  # 0 = no timeout
            else:
                effective_timeout = config.get('timeout', 300)

            register_kwargs = {
                "repo_id": repo_id,
                "revision": revision,
                "filename": filename,
                "temp_file_path": temp_file.name,
                "tracker_client": tracker,
                "timeout": effective_timeout,
                "repo_type": repo_type,
                "tqdm_class": kwargs.get("tqdm_class"),
            }
            if cache_dir is not None:
                register_kwargs["cache_dir"] = cache_dir
            if local_dir is not None:
                register_kwargs["local_dir"] = local_dir
            success = manager.register_request(**register_kwargs)

            if success:
                logger.info(f"[P2P] Successfully delivered {filename} via P2P.")
                with _stats_lock:
                    _download_stats['p2p'].add(filename)
                if schedule_deferred:
                    _schedule_deferred_notification(
                        repo_id,
                        revision,
                        repo_type,
                        cache_dir,
                        local_dir,
                        completed_snapshot=deferred_completed_snapshot,
                    )
                return  # Skip original http_get completely!
            else:
                logger.warning(f"[P2P] P2P fulfillment failed for {filename}. Falling back to HTTP.")
                _truncate_temp_file(temp_file, filename)
                truncated = True

        except Exception as e:
            logger.warning(f"[P2P] Exception in P2P intercept: {e}. Falling back to HTTP.")
            _truncate_temp_file(temp_file, filename)
            truncated = True

    # Fall back to original HTTP download if P2P failed or unavailable.
    # Only force resume_size=0 if we actually truncated the file after a
    # failed P2P attempt.  Otherwise, preserve the original resume_size so
    # that HuggingFace's native resume mechanism works correctly.
    if filename:
        with _stats_lock:
            _download_stats['http'].add(filename)
    fallback_kwargs = {**kwargs}
    if truncated:
        fallback_kwargs['resume_size'] = 0
    result = _original_http_get(url, temp_file, **fallback_kwargs)
    if schedule_deferred:
        _schedule_deferred_notification(
            repo_id,
            revision,
            repo_type,
            cache_dir,
            local_dir,
            completed_snapshot=deferred_completed_snapshot,
        )
    return result


def _format_bytes(n: int) -> str:
    """Format byte count to human-readable string."""
    try:
        from tqdm import tqdm as _tqdm
        formatted = _tqdm.format_sizeof(float(n), divisor=1000)
    except Exception:
        formatted = f"{n}B"

    if formatted == "0.00":
        return "0 B"
    if formatted.endswith("B"):
        return formatted
    return f"{formatted}B"


def _print_p2p_summary(
    stats: dict,
    elapsed: float,
    repo_id: str,
    resolved_revision: str,
    repo_type: str,
) -> None:
    """Print a P2P acceleration summary after snapshot_download completes.

    Only called when config['verbose'] is True.
    """
    p2p_files = stats.get('p2p', set())
    http_files = stats.get('http', set())
    total_files = len(p2p_files) + len(http_files)

    if total_files == 0:
        return

    # Collect byte-level stats from the P2P session
    try:
        from .p2p_batch import P2PBatchManager
        manager = P2PBatchManager()
        session_stats = manager.get_repo_p2p_stats(
            repo_id, resolved_revision, repo_type
        )
    except Exception:
        session_stats = {}

    sep = '-' * 56
    lines = [
        '',
        sep,
        f'  P2P Download Summary: {repo_id}',
        sep,
        f'  Files:     {len(p2p_files)}/{total_files} via P2P, '
        f'{len(http_files)}/{total_files} via HTTP fallback',
        f'  Time:      {elapsed:.1f}s',
    ]

    peer_bytes = session_stats.get('peer_download', 0)
    webseed_bytes = session_stats.get('webseed_download', 0)
    total_bytes = peer_bytes + webseed_bytes

    if total_bytes > 0:
        lines.append('')
        lines.append('  Data Sources:')
        if peer_bytes > 0:
            ratio = peer_bytes / total_bytes * 100
            lines.append(
                f'    From P2P peers:  {_format_bytes(peer_bytes)} ({ratio:.1f}%)'
            )
        if webseed_bytes > 0:
            ratio = webseed_bytes / total_bytes * 100
            lines.append(
                f'    From WebSeed:    {_format_bytes(webseed_bytes)} ({ratio:.1f}%)'
            )
        if peer_bytes > 0:
            lines.append('')
            lines.append(
                f'  Bandwidth saved: {_format_bytes(peer_bytes)} from server'
            )

    max_peers = session_stats.get('max_p2p_peers', 0)
    if max_peers > 0:
        lines.append(f'  Peak peers: {max_peers}')

    lines.append(sep)
    lines.append('')

    output = '\n'.join(lines)
    try:
        from tqdm import tqdm as _tqdm
        _tqdm.write(output)
    except ImportError:
        print(output)


def _patched_snapshot_download(*args, **kwargs):
    """Patched snapshot_download that notifies the daemon after completion.

    After the original snapshot_download finishes (all files downloaded),
    we notify the seeding daemon so it can create a .torrent (if needed)
    and start seeding the model for future downloaders.

    When verbose mode is enabled, prints a P2P acceleration summary
    showing file counts, data source breakdown (peer vs WebSeed),
    and bandwidth savings.
    """
    repo_id = args[0] if args else kwargs.get('repo_id')
    repo_type = kwargs.get('repo_type') or 'model'
    revision = kwargs.get('revision', 'main')
    resolved = revision

    if repo_id:
        try:
            from huggingface_hub.utils import tqdm as hf_tqdm_lib
        except ImportError:
            hf_tqdm_lib = None

        try:
            from .utils import resolve_commit_hash
            resolved = resolve_commit_hash(repo_id, revision, repo_type=repo_type)
        except Exception:
            resolved = revision

        base_tqdm_class = kwargs.get('tqdm_class') or hf_tqdm_lib
        if base_tqdm_class is not None and not kwargs.get('dry_run', False):
            kwargs = {
                **kwargs,
                'tqdm_class': _wrap_snapshot_tqdm_class(
                    base_tqdm_class=base_tqdm_class,
                    repo_id=repo_id,
                    revision=resolved,
                    repo_type=repo_type,
                ),
            }

    if repo_id:
        _enter_wrapper(repo_id)

    # Reset stats to track which files went through HTTP vs P2P in this batch
    reset_download_stats()
    start_time = time.time()

    try:
        result = _original_snapshot_download(*args, **kwargs)
    finally:
        if repo_id:
            _exit_wrapper(repo_id)

    elapsed = time.time() - start_time

    # After download completes, notify the daemon
    try:
        repo_id = args[0] if args else kwargs.get('repo_id')
        revision = kwargs.get('revision', 'main')
        repo_type = kwargs.get('repo_type') or 'model'

        if not repo_id:
            return result

        # Print P2P summary if verbose is enabled
        if _config.get('verbose'):
            _print_p2p_summary(
                stats=get_download_stats(),
                elapsed=elapsed,
                repo_id=repo_id,
                resolved_revision=resolved,
                repo_type=repo_type,
            )

        # Cancel any deferred notification for this exact snapshot identity —
        # we handle it here directly.
        keys_to_cancel = []
        with _deferred_lock:
            for key in list(_deferred_timers.keys()):
                key_repo_type, key_repo_id, key_revision = key[:3]
                if (
                    key_repo_type == repo_type and
                    key_repo_id == repo_id and
                    key_revision in {revision, resolved}
                ):
                    keys_to_cancel.append(key)

            for key in keys_to_cancel:
                timer = _deferred_timers.pop(key, None)
                if timer is not None:
                    timer.cancel()
                _deferred_contexts.pop(key, None)

        stats = get_download_stats()

        # Only a download that transferred at least one file is strong enough
        # evidence to mark a snapshot as complete. Pure cache hits are handled
        # by explicit import/verification flows.
        if not stats['p2p'] and not stats['http']:
            logger.debug(
                f"[P2P] Skipping daemon seed notification for {repo_id}@{resolved[:8]}... "
                f"(no transferred files observed)"
            )
            return result

        # Notify the daemon (fire-and-forget — safe even if daemon isn't running)
        cache_dir = kwargs.get('cache_dir')
        local_dir = kwargs.get('local_dir')
        _notify_seed_daemon(
            repo_id=repo_id,
            revision=resolved,
            repo_type=repo_type,
            cache_dir=cache_dir,
            local_dir=local_dir,
            completed_snapshot=True,
        )
        logger.debug(f"[P2P] Notified daemon to seed {repo_id}@{resolved[:8]}...")

    except Exception as e:
        # Never let notification failures break the download flow
        logger.debug(f"[P2P] Post-download daemon notification failed: {e}")

    return result


def apply_patch(config: dict) -> None:
    """
    Apply monkey patch to huggingface_hub.

    Args:
        config: Configuration dictionary containing tracker_url, etc.
    """
    global _original_hf_hub_download, _original_http_get, _original_snapshot_download, _original_snapshot_hf_tqdm, _config

    if _original_hf_hub_download is not None:
        logger.debug("Patch is already applied. Skipping.")
        return

    try:
        from huggingface_hub import file_download, _snapshot_download
        import huggingface_hub
    except ImportError:
        logger.error("huggingface_hub not installed")
        return

    # Save config and original functions to module globals
    _config = config
    _original_hf_hub_download = huggingface_hub.hf_hub_download
    _original_http_get = file_download.http_get
    _original_snapshot_download = _snapshot_download.snapshot_download
    _original_snapshot_hf_tqdm = _snapshot_download.hf_tqdm

    # Apply patches
    # NOTE: This top-level assignment has NO real effect on newer huggingface_hub versions.
    # The package uses a lazy __getattr__ that re-fetches attributes from sub-modules on
    # every access, silently overwriting anything set here. It is kept only for
    # documentation / clarity purposes. The two assignments below are what actually work.
    huggingface_hub.hf_hub_download = _patched_hf_hub_download
    file_download.hf_hub_download = _patched_hf_hub_download      # direct callers
    _snapshot_download.hf_hub_download = _patched_hf_hub_download  # snapshot_download() internals
    file_download.http_get = _patched_http_get

    # Patch snapshot_download to notify daemon after download completes
    huggingface_hub.snapshot_download = _patched_snapshot_download
    _snapshot_download.snapshot_download = _patched_snapshot_download
    _snapshot_download.hf_tqdm = _wrap_snapshot_tqdm_class_auto(_original_snapshot_hf_tqdm)

    logger.debug("Monkey patch applied successfully")


def remove_patch() -> None:
    """Remove monkey patch and restore original functions."""
    global _original_hf_hub_download, _original_http_get, _original_snapshot_download, _original_snapshot_hf_tqdm, _config

    if _original_hf_hub_download is None:
        return

    try:
        from huggingface_hub import file_download, _snapshot_download
        import huggingface_hub

        # Restore original functions
        huggingface_hub.hf_hub_download = _original_hf_hub_download
        file_download.hf_hub_download = _original_hf_hub_download
        _snapshot_download.hf_hub_download = _original_hf_hub_download
        file_download.http_get = _original_http_get

        # Restore snapshot_download
        if _original_snapshot_download:
            huggingface_hub.snapshot_download = _original_snapshot_download
            _snapshot_download.snapshot_download = _original_snapshot_download
        if _original_snapshot_hf_tqdm:
            _snapshot_download.hf_tqdm = _original_snapshot_hf_tqdm

        # Reset stored state
        _original_hf_hub_download = None
        _original_http_get = None
        _original_snapshot_download = None
        _original_snapshot_hf_tqdm = None
        _config = {}
        with _deferred_lock:
            for timer in _deferred_timers.values():
                timer.cancel()
            _deferred_timers.clear()
            _deferred_contexts.clear()
            _active_wrapper_counts.clear()
            _active_download_counts.clear()

        logger.debug("Monkey patch removed successfully")
    except ImportError:
        pass
