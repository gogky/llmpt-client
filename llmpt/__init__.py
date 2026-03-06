"""
llmpt - P2P-accelerated download client for HuggingFace Hub

This package provides seamless P2P acceleration for HuggingFace Hub downloads
using BitTorrent protocol. It works as a drop-in replacement with zero code changes.

Usage:
    # Method 1: Environment variable
    export HF_USE_P2P=1
    import llmpt
    from huggingface_hub import snapshot_download
    snapshot_download("meta-llama/Llama-2-7b")  # Automatically uses P2P

    # Method 2: Explicit enable
    from llmpt import enable_p2p
    enable_p2p()
    from huggingface_hub import snapshot_download
    snapshot_download("meta-llama/Llama-2-7b")
"""

import os
import atexit
import logging
from typing import Optional

from .utils import lt, LIBTORRENT_AVAILABLE as _LIBTORRENT_AVAILABLE

# Setup logging — use NullHandler per Python library best practices.
# This avoids "No handlers found" warnings while leaving log
# configuration (level, format, handlers) entirely to the end-user.
logger = logging.getLogger('llmpt')
logger.addHandler(logging.NullHandler())

# Global state
_patched = False
_atexit_registered = False
_config = {
    'tracker_url': None,
    'auto_seed': True,
    'seed_duration': 3600,  # 1 hour
    'timeout': 300,  # 5 minutes
    'port': None,  # None = use default 6881 with auto-fallback; set int to override
    'hf_token': None,  # HuggingFace token for WebSeed proxy (private/gated repos)
}
_webseed_proxy = None  # WebSeedProxy instance (created in enable_p2p)

__version__ = "0.1.0"
__all__ = [
    'enable_p2p',
    'disable_p2p',
    'is_enabled',
    'stop_seeding',
    'shutdown',
    'get_config',
]


def _disable_xet_storage(config: dict) -> None:
    """Disable HuggingFace's Xet Storage engine so ALL files go through http_get.

    Without this, huggingface_hub routes LFS files via xet_get() which
    completely bypasses our http_get monkey patch, making P2P ineffective
    for large model files.

    NOTE: Setting the env var alone is NOT enough — huggingface_hub
    evaluates HF_HUB_DISABLE_XET at import time and caches the result
    as a module constant.  We must also mutate the cached constant.
    """
    config['_original_xet_env'] = os.environ.get('HF_HUB_DISABLE_XET')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    try:
        from huggingface_hub import constants as hf_constants
        config['_original_xet_const'] = getattr(hf_constants, 'HF_HUB_DISABLE_XET', False)
        hf_constants.HF_HUB_DISABLE_XET = True
    except ImportError:
        pass


def _restore_xet_storage(config: dict) -> None:
    """Restore the original Xet Storage setting."""
    original_xet = config.get('_original_xet_env')
    if original_xet is None:
        os.environ.pop('HF_HUB_DISABLE_XET', None)
    else:
        os.environ['HF_HUB_DISABLE_XET'] = original_xet
    try:
        from huggingface_hub import constants as hf_constants
        hf_constants.HF_HUB_DISABLE_XET = config.get('_original_xet_const', False)
    except ImportError:
        pass


def enable_p2p(
    tracker_url: Optional[str] = None,
    auto_seed: bool = True,
    seed_duration: int = 3600,
    timeout: int = 300,
    port: Optional[int] = None,
    hf_token: Optional[str] = None,
    webseed: bool = True,
) -> None:
    """
    Enable P2P-accelerated downloads for HuggingFace Hub.

    Args:
        tracker_url: Tracker server URL. If None, uses HF_P2P_TRACKER env var
                     or default tracker.
        auto_seed: Whether to automatically seed downloaded files.
        seed_duration: How long to seed in seconds (0 = forever).
        timeout: P2P download timeout in seconds.
        port: The port to bind libtorrent to. Defaults to HF_P2P_PORT env var,
              or auto-selects from 6881-6999 range.
        hf_token: HuggingFace token for WebSeed proxy (private/gated repos).
        webseed: Whether to enable the WebSeed proxy. Defaults to True.
                 Set to False to disable WebSeed (useful for debugging).

    Example:
        >>> from llmpt import enable_p2p
        >>> enable_p2p(tracker_url="http://tracker.example.com")
        >>> from huggingface_hub import snapshot_download
        >>> snapshot_download("gpt2")  # Uses P2P
    """
    global _patched, _config, _atexit_registered, _webseed_proxy

    if not _LIBTORRENT_AVAILABLE:
        logger.warning(
            "⚠️  libtorrent not available. P2P downloads disabled.\n"
            "   Install with: pip install libtorrent"
        )
        return

    if _patched:
        logger.info("P2P already enabled")
        return

    # Update config
    _config['tracker_url'] = (
        tracker_url
        or os.getenv('HF_P2P_TRACKER')
        or 'http://localhost:8080'  # Default tracker
    )
    _config['auto_seed'] = auto_seed
    _config['seed_duration'] = seed_duration
    _config['timeout'] = timeout
    _config['hf_token'] = hf_token
    _config['webseed'] = webseed
    if port is not None:
        _config['port'] = port
    elif os.getenv('HF_P2P_PORT'):
        _config['port'] = int(os.getenv('HF_P2P_PORT'))

    _disable_xet_storage(_config)

    # Start WebSeed proxy (if enabled)
    if webseed:
        from .webseed_proxy import WebSeedProxy
        _webseed_proxy = WebSeedProxy(hf_token=hf_token)
        try:
            proxy_port = _webseed_proxy.start()
            _config['webseed_proxy_port'] = proxy_port
            logger.info(f"WebSeed proxy started on port {proxy_port}")
        except Exception as e:
            logger.warning(f"Failed to start WebSeed proxy: {e}. WebSeed disabled.")
            _webseed_proxy = None
            _config['webseed_proxy_port'] = None
    else:
        _config['webseed_proxy_port'] = None
        logger.info("WebSeed disabled by user")

    # Apply monkey patch
    from .patch import apply_patch
    apply_patch(_config)

    _patched = True

    # Auto-start the seeding daemon if auto_seed is enabled.
    # The daemon runs as an independent background process that survives
    # the current HF process exit, ensuring continuous seeding.
    if auto_seed:
        try:
            from .daemon import is_daemon_running, start_daemon
            if not is_daemon_running():
                daemon_pid = start_daemon(
                    tracker_url=_config['tracker_url'],
                )
                if daemon_pid:
                    logger.info(f"Seeding daemon auto-started (PID: {daemon_pid})")
                else:
                    logger.debug("Could not auto-start seeding daemon")
            else:
                logger.debug("Seeding daemon already running")
        except Exception as e:
            # Non-fatal: daemon auto-start failure should never break downloads
            logger.debug(f"Seeding daemon auto-start skipped: {e}")

    # Register atexit handler (once) so process exit always cleans up
    if not _atexit_registered:
        atexit.register(_cleanup_on_exit)
        _atexit_registered = True

    logger.info(f"✓ P2P enabled (tracker: {_config['tracker_url']})")


def disable_p2p() -> None:
    """
    Disable P2P downloads and restore original HTTP behavior.

    Example:
        >>> from llmpt import disable_p2p
        >>> disable_p2p()
    """
    global _patched, _webseed_proxy

    if not _patched:
        logger.info("P2P not enabled")
        return

    from .patch import remove_patch
    remove_patch()

    _restore_xet_storage(_config)

    # Stop WebSeed proxy
    if _webseed_proxy is not None:
        _webseed_proxy.stop()
        _webseed_proxy = None
        _config['webseed_proxy_port'] = None

    _patched = False
    logger.info("✓ P2P disabled")


def is_enabled() -> bool:
    """
    Check if P2P is currently enabled.

    Returns:
        True if P2P is enabled, False otherwise.

    Example:
        >>> from llmpt import is_enabled
        >>> is_enabled()
        False
    """
    return _patched


def stop_seeding() -> None:
    """
    Stop all active seeding tasks.

    Example:
        >>> from llmpt import stop_seeding
        >>> stop_seeding()
    """
    if not _patched:
        logger.info("P2P not enabled")
        return

    from .seeder import stop_all_seeding
    stop_all_seeding()
    logger.info("✓ All seeding stopped")


def get_config() -> dict:
    """
    Get current P2P configuration.

    Returns:
        Dictionary containing current configuration.

    Example:
        >>> from llmpt import get_config
        >>> config = get_config()
        >>> print(config['tracker_url'])
    """
    return _config.copy()


def shutdown() -> None:
    """
    Gracefully shut down P2P: stop all seeding, clean up temp files,
    and release the libtorrent session.

    This is called automatically via atexit when the process exits.
    You may also call it explicitly when you no longer need P2P.

    Example:
        >>> from llmpt import shutdown
        >>> shutdown()
    """
    global _webseed_proxy
    from .p2p_batch import P2PBatchManager
    try:
        manager = P2PBatchManager()
        manager.shutdown()
    except Exception:
        pass  # Best-effort during interpreter shutdown

    # Stop WebSeed proxy
    if _webseed_proxy is not None:
        try:
            _webseed_proxy.stop()
        except Exception:
            pass
        _webseed_proxy = None
        _config['webseed_proxy_port'] = None


def _cleanup_on_exit():
    """atexit callback — delegates to shutdown()."""
    if _patched:
        shutdown()


# Auto-enable from environment variable
def _auto_enable_from_env():
    """Check environment variables and auto-enable P2P if configured."""
    if os.getenv('HF_USE_P2P', '0') == '1':
        logger.info("[Auto] Detected HF_USE_P2P=1, enabling P2P...")
        enable_p2p(
            tracker_url=os.getenv('HF_P2P_TRACKER'),
            auto_seed=os.getenv('HF_P2P_AUTO_SEED', '1') == '1',
            seed_duration=int(os.getenv('HF_P2P_SEED_TIME', '3600')),
            timeout=int(os.getenv('HF_P2P_TIMEOUT', '300')),
            webseed=os.getenv('HF_P2P_WEBSEED', '1') == '1',
        )


# Execute on import
_auto_enable_from_env()
