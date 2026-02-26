"""
Tests for P2PBatchManager.
"""

import os
import sys
import threading
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_libtorrent():
    """Create a mock for libtorrent module."""
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
    mock_lt.torrent_flags.paused = getattr(mock_lt.torrent_flags, 'paused', 0)
    
    mock_lt.alert.category_t.error_notification = 1
    mock_lt.torrent_error_alert = MagicMock()
    
    sys.modules['libtorrent'] = mock_lt
    yield mock_lt


def test_batch_manager_init(mock_libtorrent):
    """Test initialization of P2PBatchManager."""
    from llmpt.p2p_batch import P2PBatchManager
    
    # P2PBatchManager is a singleton
    P2PBatchManager._instance = None
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
    
    P2PBatchManager._instance = None
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
    """Test internal initializtion of libtorrent session inside SessionContext."""
    from llmpt.p2p_batch import SessionContext
    
    tracker = MagicMock()
    tracker.get_torrent_info.return_value = {
        "info_hash": "abc",
        "magnet_link": "magnet:?xt=urn:btih:abc"
    }
    
    mock_lt_session = MagicMock()
    # Mock add_torrent return handle
    mock_handle = MagicMock()
    mock_handle.has_metadata.return_value = True
    
    mock_torrent_info = MagicMock()
    mock_torrent_info.num_files.return_value = 1
    mock_handle.get_torrent_info.return_value = mock_torrent_info
    
    mock_lt_session.add_torrent.return_value = mock_handle
    
    ctx = SessionContext("demo", "main", tracker, mock_lt_session, timeout=10)
    
    with patch('threading.Thread'):
        result = ctx._init_torrent()
    
    assert result is True
    mock_lt_session.add_torrent.assert_called_once()
    mock_handle.resume.assert_called_once()
    assert ctx.handle is mock_handle
