"""
Monkey Patch implementation for huggingface_hub.

This module patches huggingface_hub's download functions to enable P2P acceleration.
"""

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
    try:
        revision = resolve_commit_hash(repo_id, raw_revision)
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
    prev_filename = getattr(_context, 'filename', None)
    prev_revision = getattr(_context, 'revision', None)
    prev_tracker = getattr(_context, 'tracker', None)
    prev_config = getattr(_context, 'config', None)

    # Store context for http_get to use
    _context.repo_id = repo_id
    _context.filename = actual_filename
    _context.revision = revision
    _context.tracker = tracker
    _context.config = _config

    try:
        # Call original function (will trigger patched http_get)
        return _original_hf_hub_download(repo_id, filename, **kwargs)
    finally:
        # Restore previous context instead of clearing it (supports recursion)
        _context.repo_id = prev_repo_id
        _context.filename = prev_filename
        _context.revision = prev_revision
        _context.tracker = prev_tracker
        _context.config = prev_config


def _patched_http_get(url: str, temp_file, **kwargs):
    """Patched version of http_get that uses P2P batch manager when available."""
    # Check if we have P2P context (injected by patched hf_hub_download)
    repo_id = getattr(_context, 'repo_id', None)
    filename = getattr(_context, 'filename', None)
    revision = getattr(_context, 'revision', None)
    tracker = getattr(_context, 'tracker', None)
    config = getattr(_context, 'config', {})

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
                timeout=effective_timeout
            )

            if success:
                logger.info(f"[P2P] Successfully delivered {filename} via P2P.")
                with _stats_lock:
                    _download_stats['p2p'].add(filename)
                return  # Skip original http_get completely!
            else:
                logger.warning(f"[P2P] P2P fulfillment failed for {filename}. Falling back to HTTP.")
                _truncate_temp_file(temp_file, filename)

        except Exception as e:
            logger.warning(f"[P2P] Exception in P2P intercept: {e}. Falling back to HTTP.")
            _truncate_temp_file(temp_file, filename)

    # Fall back to original HTTP download if P2P failed or unavailable
    # Pass resume_size=0 explicitly since we've truncated the file
    if filename:
        with _stats_lock:
            _download_stats['http'].add(filename)
    return _original_http_get(url, temp_file, **{**kwargs, 'resume_size': 0})


def _patched_snapshot_download(*args, **kwargs):
    """Patched snapshot_download that notifies the daemon after completion.

    After the original snapshot_download finishes (all files downloaded),
    we notify the seeding daemon so it can create a .torrent (if needed)
    and start seeding the model for future downloaders.
    """
    # Reset stats to track which files went through HTTP vs P2P in this batch
    reset_download_stats()

    result = _original_snapshot_download(*args, **kwargs)

    # After download completes, notify the daemon
    try:
        repo_id = args[0] if args else kwargs.get('repo_id')
        revision = kwargs.get('revision', 'main')

        if not repo_id:
            return result

        # Resolve revision to commit hash for the daemon
        from .utils import resolve_commit_hash
        try:
            resolved = resolve_commit_hash(repo_id, revision)
        except Exception:
            resolved = revision

        # Notify the daemon (fire-and-forget — safe even if daemon isn't running)
        from .ipc import notify_daemon
        notify_daemon("seed", repo_id=repo_id, revision=resolved)
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

        logger.debug("Monkey patch removed successfully")
    except ImportError:
        pass
