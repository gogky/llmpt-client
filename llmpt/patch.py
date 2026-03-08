"""
Monkey Patch implementation for huggingface_hub.

This module patches huggingface_hub's download functions to enable P2P acceleration.
"""

import sys
import atexit
import threading
import logging
from typing import Optional, Any

logger = logging.getLogger('llmpt.patch')

# Store original functions
_original_hf_hub_download = None
_original_http_get = None
_original_snapshot_download = None

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
_deferred_timers: dict[tuple[str, str, str], threading.Timer] = {}
_deferred_contexts: dict[tuple[str, str, str], dict] = {}   # (repo_type, repo_id, revision) -> context
_active_wrapper_counts: dict[str, int] = {}  # repo_id -> active wrapper depth


def _deferred_key(repo_id: str, revision: str, repo_type: str) -> tuple[str, str, str]:
    """Build the dedupe key for deferred daemon notifications."""
    return (repo_type or "model", repo_id, revision or "main")


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
    """Flush pending deferred notifications on process exit."""
    with _deferred_lock:
        contexts = list(_deferred_contexts.values())
        for timer in _deferred_timers.values():
            timer.cancel()
        _deferred_timers.clear()
        _deferred_contexts.clear()
        
    for ctx in contexts:
        try:
            from .ipc import notify_daemon
            notify_daemon(
                "seed",
                repo_id=ctx['repo_id'],
                revision=ctx['revision'],
                repo_type=ctx['repo_type'],
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
        for _ in range(10):       # walk up at most 10 frames
            frame = frame.f_back
            if frame is None:
                break
            if frame.f_code.co_name == 'hf_hub_download':
                loc = frame.f_locals
                repo_id = loc.get('repo_id')
                filename = loc.get('filename')
                if not (repo_id and filename):
                    return None

                # commit_hash (40-char SHA) is resolved by the HEAD request
                # inside hf_hub_download; prefer it over the raw `revision`.
                revision = loc.get('commit_hash') or loc.get('revision') or 'main'
                repo_type = loc.get('repo_type') or 'model'

                # Resolve subfolder exactly as huggingface_hub does
                subfolder = loc.get('subfolder')
                actual_filename = filename
                if subfolder and subfolder != '':
                    actual_filename = f"{subfolder}/{filename}"

                return {
                    'repo_id': repo_id,
                    'filename': actual_filename,
                    'revision': revision,
                    'repo_type': repo_type,
                }
    except (AttributeError, ValueError):
        pass
    return None


def _fire_deferred_notification(key: tuple[str, str, str]) -> None:
    """Called by the debounce timer — sends a daemon seed notification."""
    with _deferred_lock:
        ctx = _deferred_contexts.pop(key, None)
        _deferred_timers.pop(key, None)
        
    if ctx is None:
        return
        
    try:
        from .ipc import notify_daemon
        notify_daemon(
            "seed",
            repo_id=ctx['repo_id'],
            revision=ctx['revision'],
            repo_type=ctx['repo_type'],
        )
        logger.debug(
            f"[P2P] Deferred daemon notification sent for "
            f"{ctx['repo_id']}@{ctx['revision'][:8]}..."
        )
    except Exception as e:
        logger.debug(f"[P2P] Deferred daemon notification failed: {e}")


def _schedule_deferred_notification(
    repo_id: str, revision: str, repo_type: str
) -> None:
    """Schedule (or reschedule) a deferred daemon notification.

    Each call resets the 2-second timer.  When the timer finally fires
    (i.e. no new hf_hub_download calls for 2 s), we notify the daemon.
    This acts as a fallback for when _patched_snapshot_download is
    bypassed due to import-order issues.
    """
    key = _deferred_key(repo_id, revision, repo_type)
    with _deferred_lock:
        # Cancel any existing timer for this specific repo+revision.
        timer = _deferred_timers.get(key)
        if timer is not None:
            timer.cancel()
            
        _deferred_contexts[key] = {
            'repo_id': repo_id,
            'revision': revision,
            'repo_type': repo_type,
        }
        
        # Start a new timer
        new_timer = threading.Timer(2.0, _fire_deferred_notification, args=[key])
        new_timer.daemon = True
        new_timer.start()
        _deferred_timers[key] = new_timer


def _patched_hf_hub_download(repo_id: str, filename: str, **kwargs):
    """Patched version of hf_hub_download that injects P2P context."""
    from .tracker_client import TrackerClient
    from .utils import resolve_commit_hash

    # Query tracker for torrent info
    tracker = TrackerClient(_config['tracker_url'])

    # Resolve revision to a 40-char commit hash so the tracker query matches
    # what seeders registered.  If resolution fails (e.g. network error),
    # fall back to the raw value — the download still works via HTTP.
    raw_revision = kwargs.get('revision', 'main')
    repo_type = kwargs.get('repo_type', 'model')
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

    # Store context for http_get to use
    _context.repo_id = repo_id
    _context.repo_type = repo_type
    _context.filename = actual_filename
    _context.revision = revision
    _context.tracker = tracker
    _context.config = _config

    download_succeeded = False
    try:
        # Call original function (will trigger patched http_get)
        result = _original_hf_hub_download(repo_id, filename, **kwargs)
        download_succeeded = True
        return result
    finally:
        # Restore previous context instead of clearing it (supports recursion)
        _context.repo_id = prev_repo_id
        _context.repo_type = prev_repo_type
        _context.filename = prev_filename
        _context.revision = prev_revision
        _context.tracker = prev_tracker
        _context.config = prev_config

        # Fallback daemon notification: if _patched_snapshot_download is not
        # wrapping this call (import-order issue), schedule a deferred
        # notification so the daemon still learns about this download.
        if download_succeeded and not _is_wrapper_active(repo_id):
            _schedule_deferred_notification(repo_id, revision, repo_type)


def _patched_http_get(url: str, temp_file, **kwargs):
    """Patched version of http_get that uses P2P batch manager when available."""
    # Check if we have P2P context (injected by patched hf_hub_download)
    repo_id = getattr(_context, 'repo_id', None)
    repo_type = getattr(_context, 'repo_type', 'model')
    filename = getattr(_context, 'filename', None)
    revision = getattr(_context, 'revision', None)
    tracker = getattr(_context, 'tracker', None)
    config = getattr(_context, 'config', {})
    schedule_deferred = False
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

            success = manager.register_request(
                repo_id=repo_id,
                revision=revision,
                filename=filename,
                temp_file_path=temp_file.name,
                tracker_client=tracker,
                timeout=effective_timeout,
                repo_type=repo_type
            )

            if success:
                logger.info(f"[P2P] Successfully delivered {filename} via P2P.")
                with _stats_lock:
                    _download_stats['p2p'].add(filename)
                if schedule_deferred:
                    _schedule_deferred_notification(repo_id, revision, repo_type)
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
        _schedule_deferred_notification(repo_id, revision, repo_type)
    return result


def _patched_snapshot_download(*args, **kwargs):
    """Patched snapshot_download that notifies the daemon after completion.

    After the original snapshot_download finishes (all files downloaded),
    we notify the seeding daemon so it can create a .torrent (if needed)
    and start seeding the model for future downloaders.
    """
    repo_id = args[0] if args else kwargs.get('repo_id')

    if repo_id:
        _enter_wrapper(repo_id)

    # Reset stats to track which files went through HTTP vs P2P in this batch
    reset_download_stats()

    try:
        result = _original_snapshot_download(*args, **kwargs)
    finally:
        if repo_id:
            _exit_wrapper(repo_id)

    # After download completes, notify the daemon
    try:
        repo_id = args[0] if args else kwargs.get('repo_id')
        revision = kwargs.get('revision', 'main')
        repo_type = kwargs.get('repo_type', 'model')

        if not repo_id:
            return result

        # Resolve revision to commit hash for the daemon
        from .utils import resolve_commit_hash
        try:
            resolved = resolve_commit_hash(repo_id, revision, repo_type=repo_type)
        except Exception:
            resolved = revision

        # Cancel any deferred notification for this exact snapshot identity —
        # we handle it here directly.
        keys_to_cancel = []
        with _deferred_lock:
            for key in list(_deferred_timers.keys()):
                key_repo_type, key_repo_id, key_revision = key
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

        # Notify the daemon (fire-and-forget — safe even if daemon isn't running)
        from .ipc import notify_daemon
        notify_daemon("seed", repo_id=repo_id, revision=resolved, repo_type=repo_type)
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
    global _original_hf_hub_download, _original_http_get, _original_snapshot_download, _config

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

    logger.debug("Monkey patch applied successfully")


def remove_patch() -> None:
    """Remove monkey patch and restore original functions."""
    global _original_hf_hub_download, _original_http_get, _original_snapshot_download, _config

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

        # Reset stored state
        _original_hf_hub_download = None
        _original_http_get = None
        _original_snapshot_download = None
        _config = {}
        with _deferred_lock:
            for timer in _deferred_timers.values():
                timer.cancel()
            _deferred_timers.clear()
            _deferred_contexts.clear()
            _active_wrapper_counts.clear()

        logger.debug("Monkey patch removed successfully")
    except ImportError:
        pass
