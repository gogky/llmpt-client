"""
Monkey Patch implementation for huggingface_hub.

This module patches huggingface_hub's download functions to enable P2P acceleration.
"""

import atexit
import time
import threading
import logging
from typing import Optional, Any

import httpx

from .patch_context import (
    apply_thread_local_context,
    capture_thread_local_context,
    extract_context_from_stack as _extract_context_from_stack,
    extract_snapshot_context_from_stack as _extract_snapshot_context_from_stack,
    matches_snapshot_download_context as _matches_snapshot_download_context,
    read_thread_local_context,
    restore_thread_local_context,
)
from . import patch_runtime as _patch_runtime
from . import patch_ui as _patch_ui

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
_download_stats: dict[tuple[str, str, str, str, str], dict[str, set[str]]] = {}

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
_DEFAULT_METADATA_ERROR_RETRIES = 2
_DEFAULT_METADATA_RETRY_DELAY = 1.0


def _snapshot_stats_key(
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> tuple[str, str, str, str, str]:
    """Build the canonical key for one snapshot-level download stats bucket."""
    return _patch_runtime.snapshot_stats_key(
        repo_id,
        revision,
        repo_type,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )


def _record_download_stat(
    stats_key: tuple[str, str, str, str, str],
    stat_kind: str,
    filename: str,
) -> None:
    _patch_runtime.record_download_stat(
        stats_lock=_stats_lock,
        download_stats=_download_stats,
        stats_key=stats_key,
        stat_kind=stat_kind,
        filename=filename,
    )


def _deferred_key(
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> tuple[str, str, str, str, str]:
    """Build the dedupe key for deferred daemon notifications."""
    return _patch_runtime.deferred_key(
        repo_id,
        revision,
        repo_type,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )


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


def _iter_exception_chain(exc: BaseException):
    """Yield *exc* and its causal chain without looping forever."""
    seen: set[int] = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        yield current
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if context is not None and context is not cause:
            stack.append(context)
        if cause is not None:
            stack.append(cause)


def _is_retryable_hf_metadata_error(exc: BaseException) -> bool:
    """Return True when the failure came from transient HF metadata I/O."""
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        LocalEntryNotFoundError = tuple()  # type: ignore[assignment]

    retryable_roots = (
        httpx.ConnectError,
        httpx.TimeoutException,
    )
    wrapper_types = (
        ValueError,
        LocalEntryNotFoundError,
    )

    saw_wrapper = False
    for candidate in _iter_exception_chain(exc):
        if isinstance(candidate, retryable_roots):
            return True
        if isinstance(candidate, wrapper_types):
            saw_wrapper = True
    return False if saw_wrapper else isinstance(exc, retryable_roots)


def _call_with_hf_metadata_retries(
    operation,
    *,
    description: str,
    repo_id: Optional[str],
    revision: Optional[str],
):
    """Retry transient HF metadata failures a small number of times."""
    retries = max(0, int(_config.get("metadata_error_retries", _DEFAULT_METADATA_ERROR_RETRIES)))
    delay = max(0.0, float(_config.get("metadata_error_retry_delay", _DEFAULT_METADATA_RETRY_DELAY)))
    attempt = 0

    while True:
        try:
            return operation()
        except Exception as exc:
            if attempt >= retries or not _is_retryable_hf_metadata_error(exc):
                raise
            attempt += 1
            revision_display = (revision or "main")[:8]
            logger.warning(
                "[P2P] Retrying %s for %s@%s after transient HF metadata error "
                "(retry %d/%d): %s",
                description,
                repo_id or "?",
                revision_display,
                attempt,
                retries,
                exc,
            )
            if delay:
                time.sleep(delay)

def _flush_deferred_notifications():
    """Flush pending deferred notifications on process exit.

    The deferred timer threads are daemon threads, so they get killed
    when the main thread exits.  This atexit handler ensures that the
    verbose summary and daemon notifications fire even if the process
    exits within the 2-second debounce window.
    """
    _patch_runtime.flush_deferred_notifications(
        deferred_lock=_deferred_lock,
        deferred_contexts=_deferred_contexts,
        deferred_timers=_deferred_timers,
        config=_config,
        get_download_stats_fn=get_download_stats,
        reset_download_stats_fn=reset_download_stats,
        print_p2p_summary_fn=_print_p2p_summary,
        notify_seed_daemon_fn=_notify_seed_daemon,
    )

atexit.register(_flush_deferred_notifications)


def get_download_stats(
    *,
    stats_key: Optional[tuple[str, str, str, str, str]] = None,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> dict:
    """Return a snapshot of download statistics.

    Returns:
        Dictionary with 'p2p' and 'http' sets of filenames.
    """
    return _patch_runtime.get_download_stats(
        stats_lock=_stats_lock,
        download_stats=_download_stats,
        snapshot_key_builder=_snapshot_stats_key,
        stats_key=stats_key,
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )


def reset_download_stats(
    *,
    stats_key: Optional[tuple[str, str, str, str, str]] = None,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> None:
    """Clear all download statistics."""
    _patch_runtime.reset_download_stats(
        stats_lock=_stats_lock,
        download_stats=_download_stats,
        snapshot_key_builder=_snapshot_stats_key,
        stats_key=stats_key,
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )


def _format_snapshot_p2p_postfix(
    stats: Optional[dict],
    download_stats: Optional[dict] = None,
) -> str:
    """Format a short, user-facing snapshot status for the shared progress bar."""
    return _patch_ui.format_snapshot_p2p_postfix(
        stats,
        download_stats=download_stats,
    )


def _wrap_snapshot_tqdm_class(
    base_tqdm_class: Any,
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
):
    """Wrap a tqdm class so snapshot_download's shared bar shows live P2P stats."""
    return _patch_ui.wrap_snapshot_tqdm_class(
        base_tqdm_class,
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        cache_dir=cache_dir,
        local_dir=local_dir,
        progress_bar_name=_SNAPSHOT_PROGRESS_BAR_NAME,
        get_download_stats_fn=get_download_stats,
        logger=logger,
        update_interval=_SNAPSHOT_PROGRESS_UPDATE_INTERVAL,
    )


def _wrap_snapshot_tqdm_class_auto(base_tqdm_class: Any):
    """Wrap hf_tqdm so old snapshot_download references still get live P2P stats."""
    return _patch_ui.wrap_snapshot_tqdm_class_auto(
        base_tqdm_class,
        progress_bar_name=_SNAPSHOT_PROGRESS_BAR_NAME,
        extract_snapshot_context_from_stack_fn=_extract_snapshot_context_from_stack,
        get_download_stats_fn=get_download_stats,
        logger=logger,
        update_interval=_SNAPSHOT_PROGRESS_UPDATE_INTERVAL,
    )


def _fire_deferred_notification(key: tuple[str, str, str, str, str]) -> None:
    """Called by the debounce timer — sends a daemon seed notification.

    Also prints a P2P summary when verbose mode is enabled.  This is the
    fallback path for when ``_patched_snapshot_download`` was bypassed
    due to the user importing ``snapshot_download`` before calling
    ``enable_p2p()``.

    If downloads are still in progress for this repo, the timer is
    rescheduled instead of firing immediately.
    """
    _patch_runtime.fire_deferred_notification(
        key,
        deferred_lock=_deferred_lock,
        deferred_contexts=_deferred_contexts,
        deferred_timers=_deferred_timers,
        active_download_counts=_active_download_counts,
        config=_config,
        get_download_stats_fn=get_download_stats,
        reset_download_stats_fn=reset_download_stats,
        print_p2p_summary_fn=_print_p2p_summary,
        notify_seed_daemon_fn=_notify_seed_daemon,
        release_on_demand_session_fn=_release_on_demand_session,
        logger=logger,
    )


def _schedule_deferred_notification(
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    completed_snapshot: bool = False,
) -> None:
    """Schedule (or reschedule) a deferred daemon notification.

    Each call resets the 2-second timer.  When the timer finally fires
    (i.e. no new hf_hub_download calls for 2 s), we notify the daemon.
    This acts as a fallback for when _patched_snapshot_download is
    bypassed due to import-order issues.
    """
    _patch_runtime.schedule_deferred_notification(
        repo_id,
        revision,
        repo_type,
        cache_dir=cache_dir,
        local_dir=local_dir,
        completed_snapshot=completed_snapshot,
        deferred_lock=_deferred_lock,
        deferred_contexts=_deferred_contexts,
        deferred_timers=_deferred_timers,
        deferred_key_fn=_deferred_key,
        fire_deferred_notification_fn=_fire_deferred_notification,
    )


def _release_on_demand_session(
    *,
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    completed: bool = False,
) -> None:
    """Release the client-side on-demand session after handoff completes."""
    _patch_runtime.release_on_demand_session(
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        cache_dir=cache_dir,
        local_dir=local_dir,
        completed=completed,
        logger=logger,
    )


def _notify_seed_daemon(
    *,
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    completed_snapshot: bool = False,
) -> None:
    _patch_runtime.notify_seed_daemon(
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        cache_dir=cache_dir,
        local_dir=local_dir,
        completed_snapshot=completed_snapshot,
    )


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
    prev_context = capture_thread_local_context(_context)

    # Store context for http_get to use
    apply_thread_local_context(
        _context,
        repo_id=repo_id,
        repo_type=repo_type,
        filename=actual_filename,
        revision=revision,
        tracker=tracker,
        config=_config,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )

    download_succeeded = False
    try:
        # Call original function (will trigger patched http_get)
        result = _call_with_hf_metadata_retries(
            lambda: _original_hf_hub_download(repo_id, filename, **kwargs),
            description=f"hf_hub_download {actual_filename}",
            repo_id=repo_id,
            revision=revision,
        )
        download_succeeded = True
        return result
    finally:
        # Restore previous context instead of clearing it (supports recursion)
        restore_thread_local_context(_context, prev_context)

        # Decrement active download counter.
        with _deferred_lock:
            count = _active_download_counts.get(repo_id, 0)
            if count <= 1:
                _active_download_counts.pop(repo_id, None)
            else:
                _active_download_counts[repo_id] = count - 1
        downloads_finished = _active_download_counts.get(repo_id, 0) == 0
        wrapper_active = _is_wrapper_active(repo_id)

        # Single-file downloads should hand off to the daemon immediately once
        # the outer hf_hub_download call finishes. snapshot_download() uses its
        # own wrapper-level handoff, so we skip this branch while a wrapper is active.
        if not wrapper_active and downloads_finished:
            if download_succeeded:
                completed_snapshot = _matches_snapshot_download_context(
                    repo_id=repo_id,
                    revision=revision,
                    repo_type=repo_type,
                    cache_dir=cache_dir,
                    local_dir=local_dir,
                )
                try:
                    notify_kwargs = {
                        'repo_id': repo_id,
                        'revision': revision,
                        'repo_type': repo_type,
                        'cache_dir': cache_dir,
                        'local_dir': local_dir,
                    }
                    if completed_snapshot:
                        notify_kwargs['completed_snapshot'] = True
                    _notify_seed_daemon(
                        **notify_kwargs,
                    )
                except Exception as e:
                    logger.debug(f"[P2P] Immediate daemon notification failed: {e}")
            _release_on_demand_session(
                repo_id=repo_id,
                revision=revision,
                repo_type=repo_type,
                cache_dir=cache_dir,
                local_dir=local_dir,
                completed=download_succeeded,
            )


def _patched_http_get(url: str, temp_file, **kwargs):
    """Patched version of http_get that uses P2P batch manager when available."""
    # Check if we have P2P context (injected by patched hf_hub_download)
    current_context = read_thread_local_context(_context)
    repo_id = current_context.get('repo_id') if current_context else None
    repo_type = current_context.get('repo_type', 'model') if current_context else 'model'
    filename = current_context.get('filename') if current_context else None
    revision = current_context.get('revision') if current_context else None
    tracker = current_context.get('tracker') if current_context else None
    config = current_context.get('config', {}) if current_context else {}
    cache_dir = current_context.get('cache_dir') if current_context else None
    local_dir = current_context.get('local_dir') if current_context else None
    schedule_deferred = False
    deferred_completed_snapshot = False
    stats_key = None

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

    if repo_id and revision:
        stats_key = _snapshot_stats_key(
            repo_id,
            revision,
            repo_type,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )

    if repo_id and filename and tracker and revision:
        try:
            from .transfer_coordinator import TransferCoordinator
            logger.info(f"[P2P] Intercepted HTTP request for {repo_id}/{filename} (rev: {revision})")

            coordinator = TransferCoordinator()
            transfer = coordinator.fulfill_request(
                repo_id=repo_id,
                revision=revision,
                filename=filename,
                destination=temp_file.name,
                tracker_client=tracker,
                repo_type=repo_type,
                cache_dir=cache_dir,
                local_dir=local_dir,
                config=config,
                tqdm_class=kwargs.get("tqdm_class"),
            )

            if transfer.success:
                logger.info(f"[P2P] Successfully delivered {filename} via P2P.")
                if stats_key is not None:
                    _record_download_stat(stats_key, 'p2p', filename)
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

        except Exception as e:
            logger.warning(f"[P2P] Exception in P2P intercept: {e}. Falling back to HTTP.")

    # Fall back to original HTTP download if P2P failed or unavailable.
    # Preserve HuggingFace's native resume_size so interrupted HTTP transfers
    # can continue from existing partial data.
    if filename and stats_key is not None:
        _record_download_stat(stats_key, 'http', filename)
    fallback_kwargs = {**kwargs}
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
    return _patch_ui.format_bytes(n)


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
    _patch_ui.print_p2p_summary(
        stats=stats,
        elapsed=elapsed,
        repo_id=repo_id,
        resolved_revision=resolved_revision,
        repo_type=repo_type,
    )


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
    cache_dir = kwargs.get('cache_dir')
    local_dir = kwargs.get('local_dir')
    resolved = revision
    stats_key = None

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
                    cache_dir=cache_dir,
                    local_dir=local_dir,
                ),
            }

    if repo_id:
        _enter_wrapper(repo_id)
        stats_key = _snapshot_stats_key(
            repo_id,
            resolved,
            repo_type,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
        # Reset only this snapshot's stats bucket.
        reset_download_stats(stats_key=stats_key)
    start_time = time.time()
    download_completed = False

    try:
        result = _call_with_hf_metadata_retries(
            lambda: _original_snapshot_download(*args, **kwargs),
            description="snapshot_download",
            repo_id=repo_id,
            revision=resolved,
        )
        download_completed = True
    finally:
        if repo_id:
            _exit_wrapper(repo_id)
        if stats_key is not None and not download_completed:
            reset_download_stats(stats_key=stats_key)

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
                stats=get_download_stats(stats_key=stats_key),
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

        stats = get_download_stats(stats_key=stats_key)

        # Only a download that transferred at least one file is strong enough
        # evidence to mark a snapshot as complete. Pure cache hits are handled
        # by explicit import/verification flows.
        if not stats['p2p'] and not stats['http']:
            logger.debug(
                f"[P2P] Skipping daemon seed notification for {repo_id}@{resolved[:8]}... "
                f"(no transferred files observed)"
            )
            _release_on_demand_session(
                repo_id=repo_id,
                revision=resolved,
                repo_type=repo_type,
                cache_dir=cache_dir,
                local_dir=local_dir,
                completed=True,
            )
            return result

        try:
            _notify_seed_daemon(
                repo_id=repo_id,
                revision=resolved,
                repo_type=repo_type,
                cache_dir=cache_dir,
                local_dir=local_dir,
                completed_snapshot=True,
            )
            logger.debug(f"[P2P] Notified daemon to seed {repo_id}@{resolved[:8]}...")
        finally:
            _release_on_demand_session(
                repo_id=repo_id,
                revision=resolved,
                repo_type=repo_type,
                cache_dir=cache_dir,
                local_dir=local_dir,
                completed=True,
            )
            if stats_key is not None:
                reset_download_stats(stats_key=stats_key)

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
        reset_download_stats()

        logger.debug("Monkey patch removed successfully")
    except ImportError:
        pass
