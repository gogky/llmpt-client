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

    # Query tracker for torrent info
    tracker = TrackerClient(_config['tracker_url'])

    # Get commit_hash from kwargs or resolve it
    revision = kwargs.get('revision', 'main')

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
            success = manager.register_request(
                repo_id=repo_id,
                revision=revision,
                filename=filename,
                temp_file_path=temp_file.name,
                tracker_client=tracker,
                timeout=config.get('timeout', 300)
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


def apply_patch(config: dict) -> None:
    """
    Apply monkey patch to huggingface_hub.

    Args:
        config: Configuration dictionary containing tracker_url, etc.
    """
    global _original_hf_hub_download, _original_http_get, _config

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

    # Apply patches
    # NOTE: This top-level assignment has NO real effect on newer huggingface_hub versions.
    # The package uses a lazy __getattr__ that re-fetches attributes from sub-modules on
    # every access, silently overwriting anything set here. It is kept only for
    # documentation / clarity purposes. The two assignments below are what actually work.
    huggingface_hub.hf_hub_download = _patched_hf_hub_download
    file_download.hf_hub_download = _patched_hf_hub_download      # direct callers
    _snapshot_download.hf_hub_download = _patched_hf_hub_download  # snapshot_download() internals
    file_download.http_get = _patched_http_get

    logger.debug("Monkey patch applied successfully")


def remove_patch() -> None:
    """Remove monkey patch and restore original functions."""
    global _original_hf_hub_download, _original_http_get, _config

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

        # Reset stored state
        _original_hf_hub_download = None
        _original_http_get = None
        _config = {}

        logger.debug("Monkey patch removed successfully")
    except ImportError:
        pass
