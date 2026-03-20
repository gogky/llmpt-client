"""
Tests for P2PBatchManager.

Uses shared fixtures from conftest.py:
- ``mock_lt_all_modules``: patches libtorrent across all llmpt modules
- ``reset_batch_manager_singleton`` (autouse): isolates the singleton
"""

import threading
import pytest
from unittest.mock import patch, MagicMock
from typing import Optional


def _session_key(
    repo_id: str = "demo",
    revision: str = "main",
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
):
    from llmpt.session_identity import build_source_session_key

    return build_source_session_key(
        repo_type,
        repo_id,
        revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )


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

    with patch.object(manager, '_ensure_alert_pump_running') as mock_pump:
        success = manager.register_request(
            repo_id="demo",
            revision="main",
            filename="model.bin",
            temp_file_path="/tmp/fake",
            tracker_client=tracker
        )

    # Should fail inside SessionContext._init_torrent
    assert success is False
    mock_pump.assert_called_once()


def test_register_request_success(mock_lt_all_modules):
    """Test successfully registering a request and creating a session context."""
    from llmpt.p2p_batch import P2PBatchManager
    from llmpt.p2p_batch import SessionContext

    tracker = MagicMock()
    tracker.download_torrent.return_value = b'fake_torrent_bytes'

    manager = P2PBatchManager()

    # We want to mock `download_file` inside `SessionContext` so we don't
    # block on threading events in unit tests.
    with patch.object(SessionContext, 'download_file', return_value=True) as mock_download, \
         patch.object(manager, '_ensure_alert_pump_running') as mock_pump:
        success = manager.register_request(
            repo_id="demo",
            revision="main",
            filename="model.bin",
            temp_file_path="/tmp/fake",
            tracker_client=tracker
        )

        assert success is True
        mock_download.assert_called_once_with("model.bin", "/tmp/fake", tqdm_class=None)
        assert _session_key() in manager.sessions
        mock_pump.assert_called_once()


def test_execute_transfer_uses_source_filename_when_it_differs_from_target(mock_lt_all_modules):
    from llmpt.p2p_batch import P2PBatchManager
    from llmpt.p2p_batch import SessionContext
    from llmpt.transfer_types import (
        LogicalTorrentRef,
        SourceFileCandidate,
        StorageIdentity,
        TargetFileRequest,
        TorrentSourceRef,
        TransferPlan,
    )

    tracker = MagicMock()
    manager = P2PBatchManager()

    target = TargetFileRequest(
        logical=LogicalTorrentRef(repo_type="model", repo_id="demo", revision="main"),
        filename="new/model.bin",
        destination="/tmp/fake",
        storage=StorageIdentity(kind="hub_cache", root="/tmp/cache"),
    )
    plan = TransferPlan(
        target=target,
        source_file=SourceFileCandidate(
            source=TorrentSourceRef(
                logical=LogicalTorrentRef(repo_type="model", repo_id="demo", revision="oldrev"),
                storage=target.storage,
            ),
            filename="old/model.bin",
        ),
    )

    with patch.object(SessionContext, 'download_file', return_value=True) as mock_download, \
         patch.object(manager, '_ensure_alert_pump_running') as mock_pump:
        success = manager.execute_transfer(
            plan,
            tracker_client=tracker,
        )

    assert success is True
    mock_download.assert_called_once_with("old/model.bin", "/tmp/fake", tqdm_class=None)
    assert _session_key(revision="oldrev", cache_dir="/tmp/cache") in manager.sessions
    mock_pump.assert_called_once()


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


def test_release_on_demand_session_completed_purges_state(mock_lt_all_modules):
    from llmpt.p2p_batch import P2PBatchManager

    manager = P2PBatchManager()
    ctx = MagicMock(session_mode='on_demand', worker_thread=None)
    key = _session_key()
    manager.sessions[key] = ctx

    with patch.object(manager, '_checkpoint_on_demand_session') as mock_checkpoint, \
         patch.object(manager, '_teardown_session', return_value=None) as mock_teardown:
        released = manager.release_on_demand_session("demo", "main", completed=True)

    assert released is True
    mock_checkpoint.assert_not_called()
    mock_teardown.assert_called_once_with(ctx, purge_resumable_state=True)


def test_register_seeding_task_reuses_existing_logical_session(mock_lt_all_modules):
    from llmpt.p2p_batch import P2PBatchManager

    tracker = MagicMock()
    manager = P2PBatchManager()

    existing_ctx = MagicMock()
    existing_ctx.handle = MagicMock()
    existing_ctx.handle.is_valid.return_value = True
    existing_ctx.is_valid = True
    manager.sessions = {
        _session_key(cache_dir="/tmp/cache-a"): existing_ctx,
    }

    with patch.object(manager, '_ensure_alert_pump_running') as mock_pump:
        success = manager.register_seeding_task(
            repo_id="demo",
            revision="main",
            repo_type="model",
            tracker_client=tracker,
            torrent_data=b"fake",
            cache_dir="/tmp/cache-b",
        )

    assert success is True
    assert set(manager.sessions) == {
        _session_key(cache_dir="/tmp/cache-a"),
    }
    mock_pump.assert_called_once()


def test_release_on_demand_session_incomplete_preserves_state(mock_lt_all_modules):
    from llmpt.p2p_batch import P2PBatchManager

    manager = P2PBatchManager()
    ctx = MagicMock(session_mode='on_demand', worker_thread=None)
    key = _session_key()
    manager.sessions[key] = ctx

    with patch.object(manager, '_checkpoint_on_demand_session') as mock_checkpoint, \
         patch.object(manager, '_teardown_session', return_value=None) as mock_teardown:
        released = manager.release_on_demand_session("demo", "main", completed=False)

    assert released is True
    mock_checkpoint.assert_called_once_with(ctx)
    mock_teardown.assert_called_once_with(ctx, purge_resumable_state=False)


def test_remove_all_sessions_preserves_on_demand_partials(mock_lt_all_modules):
    from llmpt.p2p_batch import P2PBatchManager

    manager = P2PBatchManager()
    on_demand = MagicMock(session_mode='on_demand', worker_thread=None)
    full_seed = MagicMock(session_mode='full_seed', worker_thread=None)
    manager.sessions = {
        _session_key(cache_dir="/tmp/a"): on_demand,
        _session_key(revision="rev2", cache_dir="/tmp/b"): full_seed,
    }

    with patch.object(manager, '_checkpoint_on_demand_session') as mock_checkpoint, \
         patch.object(manager, '_teardown_session', side_effect=[None, None]) as mock_teardown, \
         patch.object(manager, '_stop_alert_pump') as mock_stop_pump:
        count = manager.remove_all_sessions()

    assert count == 2
    mock_checkpoint.assert_called_once_with(on_demand)
    assert mock_teardown.call_args_list == [
        ((on_demand,), {'purge_resumable_state': False}),
        ((full_seed,), {'purge_resumable_state': True}),
    ]
    mock_stop_pump.assert_called_once_with(join=True)


def test_checkpoint_wakes_alert_pump_instead_of_dispatch(mock_lt_all_modules):
    from llmpt.p2p_batch import P2PBatchManager

    manager = P2PBatchManager()
    ctx = MagicMock(session_mode='on_demand', fastresume_path='/tmp/demo.fastresume')
    ctx.lock = threading.Lock()
    ctx.handle = MagicMock()
    ctx.handle.is_valid.return_value = True

    stat_a = type('Stat', (), {'st_mtime_ns': 1})()
    stat_b = type('Stat', (), {'st_mtime_ns': 2})()

    with patch.object(manager, '_request_alert_pump_wakeup') as mock_wakeup, \
         patch.object(manager, 'dispatch_alerts') as mock_dispatch, \
         patch('llmpt.monitor._process_alerts') as mock_process_alerts, \
         patch('os.path.exists', return_value=True), \
         patch('os.stat', side_effect=[stat_a, stat_b]):
        manager._checkpoint_on_demand_session(ctx)

    mock_wakeup.assert_called()
    mock_dispatch.assert_not_called()
    mock_process_alerts.assert_called()


def test_alert_pump_loop_dispatches_alerts(mock_lt_all_modules):
    from llmpt.p2p_batch import P2PBatchManager

    manager = P2PBatchManager()
    manager.sessions = {_session_key(cache_dir="/tmp/cache"): MagicMock()}
    manager.lt_session.wait_for_alert = MagicMock()

    stop_event = threading.Event()
    with patch.object(manager._alert_pump_wakeup, 'wait', return_value=False) as mock_wait, \
         patch.object(manager, 'dispatch_alerts', side_effect=lambda: stop_event.set()) as mock_dispatch:
        manager._alert_pump_loop(stop_event)

    mock_wait.assert_called_once_with(0.2)
    manager.lt_session.wait_for_alert.assert_not_called()
    mock_dispatch.assert_called_once()
