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

# Thread-local storage for P2P context
_context = threading.local()


def apply_patch(config: dict) -> None:
    """
    Apply monkey patch to huggingface_hub.

    Args:
        config: Configuration dictionary containing tracker_url, etc.
    """
    global _original_hf_hub_download, _original_http_get

    try:
        from huggingface_hub import file_download
        import huggingface_hub
    except ImportError:
        logger.error("huggingface_hub not installed")
        return

    # Save original functions
    _original_hf_hub_download = huggingface_hub.hf_hub_download
    _original_http_get = file_download.http_get

    # Create patched versions
    def patched_hf_hub_download(repo_id: str, filename: str, **kwargs):
        """Patched version of hf_hub_download that queries tracker."""
        from .tracker_client import TrackerClient

        # Query tracker for torrent info
        tracker = TrackerClient(config['tracker_url'])

        # Get commit_hash from kwargs or resolve it
        revision = kwargs.get('revision', 'main')

        # Store context for http_get to use
        _context.repo_id = repo_id
        _context.filename = filename
        _context.revision = revision
        _context.tracker = tracker
        _context.config = config

        try:
            # Call original function (will trigger patched http_get)
            return _original_hf_hub_download(repo_id, filename, **kwargs)
        finally:
            # Clean up context
            _context.repo_id = None
            _context.filename = None
            _context.revision = None
            _context.tracker = None

    def patched_http_get(url: str, temp_file, **kwargs):
        """Patched version of http_get that uses P2P when available."""
        # Check if we have P2P context
        repo_id = getattr(_context, 'repo_id', None)
        filename = getattr(_context, 'filename', None)
        tracker = getattr(_context, 'tracker', None)
        config = getattr(_context, 'config', None)

        if repo_id and filename and tracker:
            # Try P2P download
            try:
                from .downloader import try_p2p_download

                logger.info(f"[P2P] Attempting P2P download: {repo_id}/{filename}")

                success = try_p2p_download(
                    repo_id=repo_id,
                    filename=filename,
                    url=url,
                    temp_file=temp_file,
                    tracker=tracker,
                    config=config,
                    **kwargs
                )

                if success:
                    logger.info(f"[P2P] Successfully downloaded via P2P")
                    return
                else:
                    logger.warning(f"[P2P] P2P download failed, falling back to HTTP")
            except Exception as e:
                logger.warning(f"[P2P] P2P download error: {e}, falling back to HTTP")

        # Fall back to original HTTP download
        return _original_http_get(url, temp_file, **kwargs)

    # Apply patches
    huggingface_hub.hf_hub_download = patched_hf_hub_download
    file_download.http_get = patched_http_get

    logger.debug("Monkey patch applied successfully")


def remove_patch() -> None:
    """Remove monkey patch and restore original functions."""
    global _original_hf_hub_download, _original_http_get

    if _original_hf_hub_download is None:
        return

    try:
        from huggingface_hub import file_download
        import huggingface_hub

        # Restore original functions
        huggingface_hub.hf_hub_download = _original_hf_hub_download
        file_download.http_get = _original_http_get

        logger.debug("Monkey patch removed successfully")
    except ImportError:
        pass
