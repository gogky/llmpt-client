"""
llmpt - P2P-accelerated download client for HuggingFace Hub

This package provides seamless P2P acceleration for HuggingFace Hub downloads
using BitTorrent protocol. It works as a drop-in replacement with zero code changes.

Usage:
    # Method 1: Environment variable
    export HF_USE_P2P=1
    from huggingface_hub import snapshot_download
    snapshot_download("meta-llama/Llama-2-7b")  # Automatically uses P2P

    # Method 2: Explicit enable
    from llmpt import enable_p2p
    enable_p2p()
    from huggingface_hub import snapshot_download
    snapshot_download("meta-llama/Llama-2-7b")
"""

import os
import logging
from typing import Optional

# Check libtorrent availability
try:
    import libtorrent as lt
    _LIBTORRENT_AVAILABLE = True
except ImportError:
    _LIBTORRENT_AVAILABLE = False
    lt = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('llmpt')

# Global state
_patched = False
_config = {
    'tracker_url': None,
    'auto_seed': True,
    'seed_duration': 3600,  # 1 hour
    'timeout': 300,  # 5 minutes
}

__version__ = "0.1.0"
__all__ = [
    'enable_p2p',
    'disable_p2p',
    'is_enabled',
    'stop_seeding',
    'get_config',
]


def enable_p2p(
    tracker_url: Optional[str] = None,
    auto_seed: bool = True,
    seed_duration: int = 3600,
    timeout: int = 300,
) -> None:
    """
    Enable P2P-accelerated downloads for HuggingFace Hub.

    Args:
        tracker_url: Tracker server URL. If None, uses HF_P2P_TRACKER env var
                     or default tracker.
        auto_seed: Whether to automatically seed downloaded files.
        seed_duration: How long to seed in seconds (0 = forever).
        timeout: P2P download timeout in seconds.

    Example:
        >>> from llmpt import enable_p2p
        >>> enable_p2p(tracker_url="http://tracker.example.com")
        >>> from huggingface_hub import snapshot_download
        >>> snapshot_download("gpt2")  # Uses P2P
    """
    global _patched, _config

    if not _LIBTORRENT_AVAILABLE:
        logger.warning(
            "⚠️  libtorrent not available. P2P downloads disabled.\n"
            "   Install with: pip install python-libtorrent"
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

    # Apply monkey patch
    from .patch import apply_patch
    apply_patch(_config)

    _patched = True
    logger.info(f"✓ P2P enabled (tracker: {_config['tracker_url']})")


def disable_p2p() -> None:
    """
    Disable P2P downloads and restore original HTTP behavior.

    Example:
        >>> from llmpt import disable_p2p
        >>> disable_p2p()
    """
    global _patched

    if not _patched:
        logger.info("P2P not enabled")
        return

    from .patch import remove_patch
    remove_patch()

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
        )


# Execute on import
_auto_enable_from_env()
