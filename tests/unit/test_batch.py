"""
Tests for P2PBatchManager.
"""

import os
import sys
import threading
import pytest
from unittest.mock import patch, MagicMock


def _make_mock_lt():
    """Create a comprehensive mock for libtorrent."""
    mock_lt = MagicMock()
    mock_lt.session = MagicMock()
    mock_lt.session_settings = MagicMock()
    mock_lt.add_torrent_params = MagicMock()
    mock_lt.torrent_info = MagicMock()
    mock_lt.create_torrent = MagicMock()
    mock_lt.set_piece_hashes = MagicMock()

    # Mock some enums and constants
    mock_lt.torrent_status.checking_files = 1
    mock_lt.torrent_status.downloading_metadata = 2
    mock_lt.torrent_status.downloading = 3
    mock_lt.torrent_status.finished = 4
    mock_lt.torrent_status.seeding = 5
    mock_lt.torrent_status.allocating = 6
    mock_lt.torrent_status.checking_resume_data = 7

    # Add flag
    mock_lt.torrent_flags.paused = 0

    mock_lt.alert.category_t.error_notification = 1
    mock_lt.torrent_error_alert = MagicMock()

    return mock_lt


@pytest.fixture
def mock_libtorrent():
    """Patch libtorrent in all modules that use it.
    
    session_context.py and p2p_batch.py import `lt` from `.utils` at module
    load time, so `sys.modules` injection alone won't work.  We must patch
    the module-level `lt` variable in every module that references it.
    """
    mock_lt = _make_mock_lt()

    with patch('llmpt.utils.lt', mock_lt), \
         patch('llmpt.utils.LIBTORRENT_AVAILABLE', True), \
         patch('llmpt.p2p_batch.lt', mock_lt), \
         patch('llmpt.p2p_batch.LIBTORRENT_AVAILABLE', True), \
         patch('llmpt.session_context.lt', mock_lt), \
         patch('llmpt.session_context.LIBTORRENT_AVAILABLE', True):
        yield mock_lt


@pytest.fixture(autouse=True)
def reset_batch_manager_singleton():
    """Reset P2PBatchManager singleton between tests."""
    from llmpt.p2p_batch import P2PBatchManager
    P2PBatchManager._instance = None
    yield
    P2PBatchManager._instance = None


def test_batch_manager_init(mock_libtorrent):
    """Test initialization of P2PBatchManager."""
    from llmpt.p2p_batch import P2PBatchManager

    manager1 = P2PBatchManager()
    manager2 = P2PBatchManager()

    assert manager1 is manager2
    assert manager1.lt_session is not None
    assert getattr(manager1, 'sessions') is not None


def test_register_request_no_torrent(mock_libtorrent):
    """Test registering a request when tracker has no torrent."""
    from llmpt.p2p_batch import P2PBatchManager
    tracker = MagicMock()
    tracker.get_torrent_info.return_value = None

    manager = P2PBatchManager()

    success = manager.register_request(
        repo_id="demo",
        revision="main",
        filename="model.bin",
        temp_file_path="/tmp/fake",
        tracker_client=tracker
    )

    # Should fail inside SessionContext._init_torrent
    assert success is False


def test_register_request_success(mock_libtorrent):
    """Test successfully registering a request and creating a session context."""
    from llmpt.p2p_batch import P2PBatchManager
    from llmpt.p2p_batch import SessionContext

    tracker = MagicMock()
    tracker.get_torrent_info.return_value = {
        "info_hash": "abc",
        "magnet_link": "magnet:?xt=urn:btih:abc"
    }

    manager = P2PBatchManager()

    # We want to mock `download_file` inside `SessionContext` so we don't
    # block on threading events in unit tests.
    with patch.object(SessionContext, 'download_file', return_value=True) as mock_download:
        success = manager.register_request(
            repo_id="demo",
            revision="main",
            filename="model.bin",
            temp_file_path="/tmp/fake",
            tracker_client=tracker
        )

        assert success is True
        mock_download.assert_called_once_with("model.bin", "/tmp/fake")
        assert ("demo", "main") in manager.sessions


def test_session_context_init_torrent(mock_libtorrent):
    """Test internal initialization of libtorrent session inside SessionContext.

    The _init_torrent() flow:
      1. tracker.get_torrent_info() → returns magnet link
      2. lt.parse_magnet_uri(magnet_link) → returns add_torrent_params
      3. lt_session.add_torrent(params) → returns handle
      4. Starts monitor thread (daemon)
      5. Waits for handle.status().has_metadata → True
      6. handle.torrent_file() → torrent_info_obj
      7. handle.prioritize_files([0] * num_files)
      8. handle.resume()
    """
    from llmpt.p2p_batch import SessionContext

    tracker = MagicMock()
    tracker.get_torrent_info.return_value = {
        "info_hash": "abc",
        "magnet_link": "magnet:?xt=urn:btih:abc"
    }

    mock_lt_session = MagicMock()

    # Mock add_torrent return handle
    mock_handle = MagicMock()

    # _init_torrent() checks handle.status().has_metadata in a while loop
    mock_status = MagicMock()
    mock_status.has_metadata = True
    mock_handle.status.return_value = mock_status

    # After metadata resolves, it calls handle.torrent_file()
    mock_torrent_info = MagicMock()
    mock_torrent_info.num_files.return_value = 1
    mock_handle.torrent_file.return_value = mock_torrent_info

    mock_lt_session.add_torrent.return_value = mock_handle

    # lt.parse_magnet_uri is called in _init_torrent
    mock_params = MagicMock()
    mock_params.flags = 0
    mock_libtorrent.parse_magnet_uri.return_value = mock_params

    ctx = SessionContext("demo", "main", tracker, mock_lt_session, timeout=10)

    # Mock the monitor loop to prevent real background thread work
    with patch('llmpt.session_context.run_monitor_loop'), \
         patch('os.path.exists', return_value=False):
        result = ctx._init_torrent()

    assert result is True
    mock_lt_session.add_torrent.assert_called_once()
    mock_handle.resume.assert_called_once()
    assert ctx.handle is mock_handle
    assert ctx.torrent_info_obj is mock_torrent_info

