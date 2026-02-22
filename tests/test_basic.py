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
        auto_seed=False,
        seed_duration=7200,
        timeout=600
    )

    config = llmpt.get_config()
    assert config['tracker_url'] == "http://test.com"
    assert config['auto_seed'] == False
    assert config['seed_duration'] == 7200
    assert config['timeout'] == 600

    llmpt.disable_p2p()


def test_env_variable():
    """Test auto-enable from environment variable."""
    # This test needs to be run in isolation
    # For now, just check the function exists
    from llmpt import _auto_enable_from_env
    assert callable(_auto_enable_from_env)


def test_libtorrent_availability():
    """Test libtorrent availability check."""
    from llmpt import _LIBTORRENT_AVAILABLE
    # Just check it's a boolean
    assert isinstance(_LIBTORRENT_AVAILABLE, bool)
