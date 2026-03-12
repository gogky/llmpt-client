"""
Tests for P2PBatchManager.

Uses shared fixtures from conftest.py:
- ``mock_lt_all_modules``: patches libtorrent across all llmpt modules
- ``reset_batch_manager_singleton`` (autouse): isolates the singleton
"""

import pytest
from unittest.mock import patch, MagicMock


def test_batch_manager_init(mock_lt_all_modules):
    """Test initialization of P2PBatchManager."""
    from llmpt.p2p_batch import P2PBatchManager

    manager1 = P2PBatchManager()
    manager2 = P2PBatchManager()

    assert manager1 is manager2
    assert manager1.lt_session is not None
    assert getattr(manager1, 'sessions') is not None


def test_register_request_no_torrent(mock_lt_all_modules):
    """Test registering a request when tracker has no torrent."""
    from llmpt.p2p_batch import P2PBatchManager
    tracker = MagicMock()
    tracker.download_torrent.return_value = None

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


def test_register_request_success(mock_lt_all_modules):
    """Test successfully registering a request and creating a session context."""
    from llmpt.p2p_batch import P2PBatchManager
    from llmpt.p2p_batch import SessionContext
    from llmpt.utils import get_hf_hub_cache

    tracker = MagicMock()
    tracker.download_torrent.return_value = b'fake_torrent_bytes'

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
        mock_download.assert_called_once_with("model.bin", "/tmp/fake", tqdm_class=None)
        assert ("model", "demo", "main", "hub_cache", get_hf_hub_cache()) in manager.sessions


def test_session_context_init_torrent(mock_lt_all_modules):
    """Test internal initialization of libtorrent session inside SessionContext.

    The _init_torrent() flow:
      1. torrent_cache.resolve_torrent_data() → returns raw .torrent bytes
         (three-layer: local cache → tracker → None)
      2. lt.bdecode(torrent_data) → decoded dict
      3. lt.torrent_info(decoded) → torrent_info object
      4. lt_session.add_torrent(params) → returns handle
      5. Starts monitor thread (daemon)
      6. handle.torrent_file() → torrent_info_obj (immediately available)
      7. handle.prioritize_files([0] * num_files)
      8. handle.resume()
    """
    from llmpt.p2p_batch import SessionContext

    tracker = MagicMock()

    mock_lt_session = MagicMock()

    # Mock add_torrent return handle
    mock_handle = MagicMock()

    # After init, it calls handle.torrent_file() (immediately, no metadata wait)
    mock_torrent_info = MagicMock()
    mock_torrent_info.num_files.return_value = 1
    mock_handle.torrent_file.return_value = mock_torrent_info

    mock_lt_session.add_torrent.return_value = mock_handle

    ctx = SessionContext('demo', 'main', tracker, mock_lt_session, 'on_demand', 10, repo_type='model')

    # Mock the monitor loop to prevent real background thread work,
    # and mock the three-layer torrent resolver to return fake data
    with patch('llmpt.session_context.run_monitor_loop'), \
         patch('os.path.exists', return_value=False), \
         patch('llmpt.torrent_cache.resolve_torrent_data', return_value=b'fake_torrent_bytes'):
        result = ctx._init_torrent()

    assert result is True
    mock_lt_session.add_torrent.assert_called_once()
    assert ctx.handle is mock_handle
    assert ctx.torrent_info_obj is mock_torrent_info
    # Verify we used bdecode + torrent_info path, not parse_magnet_uri
    mock_lt_all_modules.bdecode.assert_called_once_with(b'fake_torrent_bytes')
    mock_lt_all_modules.torrent_info.assert_called_once()
    mock_lt_all_modules.parse_magnet_uri.assert_not_called()
