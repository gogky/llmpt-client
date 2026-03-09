"""
Basic tests for llmpt package.
"""

import pytest
import os


def test_import():
    """Test that package can be imported."""
    import llmpt
    assert llmpt.__version__ == "0.1.0"


def test_enable_disable():
    """Test enable/disable P2P."""
    import llmpt

    # Initially disabled
    assert not llmpt.is_enabled()

    # Enable
    llmpt.enable_p2p(tracker_url="http://test-tracker.com")
    assert llmpt.is_enabled()

    # Disable
    llmpt.disable_p2p()
    assert not llmpt.is_enabled()


def test_config():
    """Test configuration."""
    import llmpt

    llmpt.enable_p2p(
        tracker_url="http://test.com",
        timeout=600
    )

    config = llmpt.get_config()
    assert config['tracker_url'] == "http://test.com"
    assert config['timeout'] == 600

    llmpt.disable_p2p()


def test_import_does_not_auto_enable():
    """Test that importing llmpt does NOT auto-enable P2P.

    Users must explicitly call enable_p2p(). The old _auto_enable_from_env
    behaviour was removed because it caused unwanted side effects (libtorrent
    session creation, WebSeedProxy startup) in CLI commands and daemon
    sub-processes that merely import ``llmpt`` sub-modules.
    """
    import llmpt
    # Even if HF_USE_P2P=1 is set, importing should not enable P2P
    assert not llmpt.is_enabled()


def test_libtorrent_availability():
    """Test libtorrent availability check."""
    from llmpt import _LIBTORRENT_AVAILABLE
    # Just check it's a boolean
    assert isinstance(_LIBTORRENT_AVAILABLE, bool)
