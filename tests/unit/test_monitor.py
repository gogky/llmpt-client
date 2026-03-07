"""
Tests for the monitor module (llmpt.monitor).

All functions are tested with a mocked SessionContext and mocked libtorrent
so no real I/O or network activity occurs.
"""

import threading
from collections import deque
import pytest
from unittest.mock import patch, MagicMock, mock_open

from llmpt.monitor import (
    _log_diagnostics,
    _save_resume_data,
    _process_alerts,
    _check_pending_files,
    _check_session_health,
    _resolve_pending_metadata,
    _collect_ready_files,
    _has_torrent_error,
    _get_error_message,
)
from tests.unit.conftest import make_mock_ctx


# ─── _has_torrent_error ──────────────────────────────────────────────────────

class TestHasTorrentError:

    def test_no_error_attributes(self):
        """Status without errc or error → no error."""
        status = MagicMock(spec=[])  # no attributes at all
        assert _has_torrent_error(status) is False

    def test_errc_value_nonzero(self):
        status = MagicMock()
        status.errc.value.return_value = 5
        assert _has_torrent_error(status) is True

    def test_errc_value_zero(self):
        status = MagicMock()
        status.errc.value.return_value = 0
        status.errc.message.return_value = 'Success'
        assert _has_torrent_error(status) is False

    def test_errc_message_not_success(self):
        status = MagicMock(spec=['errc'])
        status.errc = MagicMock(spec=['message'])
        status.errc.message.return_value = 'disk read error'
        assert _has_torrent_error(status) is True

    def test_legacy_error_attribute(self):
        """Older libtorrent versions use `status.error` string."""
        status = MagicMock(spec=['error'])
        status.error = "something went wrong"
        assert _has_torrent_error(status) is True

    def test_legacy_error_empty(self):
        status = MagicMock(spec=['error'])
        status.error = ""
        assert _has_torrent_error(status) is False


# ─── _get_error_message ──────────────────────────────────────────────────────

class TestGetErrorMessage:

    def test_errc_message(self):
        status = MagicMock()
        status.errc.message.return_value = "disk full"
        assert _get_error_message(status) == "disk full"

    def test_fallback_to_error(self):
        status = MagicMock(spec=['error'])
        status.error = "old style error"
        assert "old style error" in _get_error_message(status)

    def test_unknown_fallback(self):
        status = MagicMock(spec=[])
        result = _get_error_message(status)
        assert "Unknown" in result


# ─── _log_diagnostics ────────────────────────────────────────────────────────

class TestLogDiagnostics:

    def test_valid_handle(self):
        """Should not raise when handle is valid."""
        ctx = make_mock_ctx()
        ctx.handle.is_valid.return_value = True
        status = ctx.handle.status.return_value
        status.state = 3  # downloading
        status.progress = 0.5
        status.num_peers = 2
        status.num_seeds = 1
        status.download_rate = 1024 * 100
        status.upload_rate = 1024 * 50
        status.num_pieces = 5
        ctx.torrent_info_obj = MagicMock()
        ctx.torrent_info_obj.num_pieces.return_value = 10

        # Should not raise
        _log_diagnostics(ctx)

    def test_invalid_handle(self):
        """Should not crash when handle is invalid."""
        ctx = make_mock_ctx()
        ctx.handle.is_valid.return_value = False
        _log_diagnostics(ctx)  # No exception

    def test_no_handle(self):
        """Should not crash when handle is None."""
        ctx = make_mock_ctx()
        ctx.handle = None
        _log_diagnostics(ctx)


# ─── _save_resume_data ────────────────────────────────────────────────────────

class TestSaveResumeData:

    @patch('llmpt.monitor.lt')
    def test_calls_save_on_valid_handle(self, mock_lt):
        ctx = make_mock_ctx()
        ctx.handle.is_valid.return_value = True
        _save_resume_data(ctx)
        ctx.handle.save_resume_data.assert_called_once()

    @patch('llmpt.monitor.lt')
    def test_skips_invalid_handle(self, mock_lt):
        ctx = make_mock_ctx()
        ctx.handle.is_valid.return_value = False
        _save_resume_data(ctx)
        ctx.handle.save_resume_data.assert_not_called()

    @patch('llmpt.monitor.lt')
    def test_no_handle(self, mock_lt):
        ctx = make_mock_ctx()
        ctx.handle = None
        _save_resume_data(ctx)  # Should not raise


# ─── _process_alerts ──────────────────────────────────────────────────────────

class TestProcessAlerts:

    @patch('llmpt.monitor.lt')
    def test_save_resume_data_alert(self, mock_lt):
        """Successful resume data save should write to disk."""
        ctx = make_mock_ctx()

        alert = MagicMock()
        alert.handle = ctx.handle
        alert.params = {'info-hash': 'abc'}
        mock_lt.save_resume_data_alert = type(alert)
        mock_lt.save_resume_data_failed_alert = type('Other', (), {})
        mock_lt.bencode.return_value = b'encoded_data'

        # Populate the inbox (as dispatch_alerts would do)
        ctx.pending_alerts.append(alert)

        with patch('builtins.open', mock_open()) as m:
            _process_alerts(ctx)
            m.assert_called_once_with(ctx.fastresume_path, "wb")
            m().write.assert_called_once_with(b'encoded_data')

    @patch('llmpt.monitor.lt')
    def test_no_alerts(self, mock_lt):
        ctx = make_mock_ctx()
        # Empty inbox
        _process_alerts(ctx)  # No crash


# ─── _check_pending_files ────────────────────────────────────────────────────

class TestCheckPendingFiles:

    @patch('llmpt.monitor.lt')
    def test_no_handle_returns_true(self, mock_lt):
        """Should break the loop if handle is None."""
        ctx = make_mock_ctx()
        ctx.handle = None
        assert _check_pending_files(ctx) is True

    @patch('llmpt.monitor.lt')
    def test_torrent_error_sets_invalid(self, mock_lt):
        """Torrent-level error should mark ctx as invalid and return True."""
        ctx = make_mock_ctx()
        status = ctx.handle.status.return_value
        status.errc.value.return_value = 42
        status.errc.message.return_value = "disk error"

        result = _check_pending_files(ctx)

        assert result is True
        assert ctx.is_valid is False

    @patch('llmpt.monitor.lt')
    def test_no_pending_files_returns_false(self, mock_lt):
        """No pending events → nothing to do, don't break."""
        ctx = make_mock_ctx()
        # No error
        status = MagicMock(spec=['state'])
        status.state = 3
        ctx.handle.status.return_value = status
        ctx.file_events = {}  # Nothing pending

        assert _check_pending_files(ctx) is False

    @patch('llmpt.monitor.lt')
    def test_file_completed_and_delivered(self, mock_lt):
        """When file progress == file_size, deliver and set event."""
        ctx = make_mock_ctx()

        event = threading.Event()
        ctx.file_events = {"model.bin": event}
        ctx.file_destinations = {"model.bin": "/dest/model.bin"}

        # No error in status
        status = MagicMock(spec=['state'])
        status.state = 3  # downloading
        ctx.handle.status.return_value = status

        # Torrent info setup
        mock_files = MagicMock()
        mock_files.file_size.return_value = 1000
        mock_files.file_path.return_value = "root/model.bin"
        mock_files.num_files.return_value = 1

        mock_ti = MagicMock()
        mock_ti.files.return_value = mock_files
        mock_ti.num_files.return_value = 1
        ctx.torrent_info_obj = mock_ti

        # _find_file_index returns 0
        ctx._find_file_index.return_value = 0

        # File is 100% done
        ctx.handle.file_progress.return_value = [1000]
        ctx.handle.file_priorities.return_value = [1]

        # _get_lt_disk_path and _deliver_file
        ctx._get_lt_disk_path.return_value = "/tmp/p2p/root/model.bin"
        ctx._deliver_file.return_value = None  # no-op mock

        result = _check_pending_files(ctx)

        assert result is False
        assert event.is_set()
        ctx._deliver_file.assert_called_once_with("/tmp/p2p/root/model.bin", "/dest/model.bin")

    @patch('llmpt.monitor.lt')
    def test_file_not_yet_complete(self, mock_lt):
        """File with partial progress should not trigger delivery."""
        ctx = make_mock_ctx()

        event = threading.Event()
        ctx.file_events = {"model.bin": event}
        ctx.file_destinations = {"model.bin": "/dest/model.bin"}

        status = MagicMock(spec=['state'])
        status.state = 3
        ctx.handle.status.return_value = status

        mock_files = MagicMock()
        mock_files.file_size.return_value = 1000
        mock_files.num_files.return_value = 1

        mock_ti = MagicMock()
        mock_ti.files.return_value = mock_files
        ctx.torrent_info_obj = mock_ti

        ctx._find_file_index.return_value = 0
        ctx.handle.file_progress.return_value = [500]  # Only 50%
        ctx.handle.file_priorities.return_value = [1]

        result = _check_pending_files(ctx)

        assert result is False
        assert not event.is_set()
        ctx._deliver_file.assert_not_called()

    @patch('llmpt.monitor.lt')
    def test_checking_state_skips_progress_check(self, mock_lt):
        """During recheck (state 1 or 7), progress checks should be skipped."""
        ctx = make_mock_ctx()

        event = threading.Event()
        ctx.file_events = {"model.bin": event}

        status = MagicMock(spec=['state', 'has_metadata'])
        status.state = 1  # checking_files
        ctx.handle.status.return_value = status
        ctx.torrent_info_obj = MagicMock()

        result = _check_pending_files(ctx)
        assert result is False

    @patch('llmpt.monitor.lt')
    def test_metadata_arrival_triggers_mapping(self, mock_lt):
        """When torrent_info_obj is None but metadata becomes available, it should be populated."""
        ctx = make_mock_ctx()

        event = threading.Event()
        ctx.file_events = {"model.bin": event}
        ctx.torrent_info_obj = None  # metadata not loaded yet

        # First call: no error, has_metadata → True
        status = MagicMock(spec=['state', 'has_metadata'])
        status.state = 3
        status.has_metadata = True
        ctx.handle.status.return_value = status

        mock_ti = MagicMock()
        mock_ti.num_files.return_value = 2
        mock_ti.files.return_value = MagicMock()
        ctx.handle.torrent_file.return_value = mock_ti

        # After metadata loads, _find_file_index returns None → no delivery
        ctx._find_file_index.return_value = None
        ctx.handle.file_progress.return_value = [0, 0]

        _check_pending_files(ctx)

        # torrent_info_obj should now be set
        assert ctx.torrent_info_obj is mock_ti
        ctx.handle.prioritize_files.assert_called_once_with([0, 0])


# ─── _check_session_health ───────────────────────────────────────────────────

class TestCheckSessionHealth:

    def test_no_handle_returns_true(self):
        ctx = make_mock_ctx()
        ctx.handle = None
        assert _check_session_health(ctx) is True

    def test_torrent_error_returns_true(self):
        ctx = make_mock_ctx()
        status = ctx.handle.status.return_value
        status.errc.value.return_value = 42
        status.errc.message.return_value = "disk error"

        result = _check_session_health(ctx)
        assert result is True
        assert ctx.is_valid is False

    def test_no_pending_files_returns_false(self):
        ctx = make_mock_ctx()
        status = MagicMock(spec=['state'])
        status.state = 3
        ctx.handle.status.return_value = status
        ctx.file_events = {}

        assert _check_session_health(ctx) is False

    def test_has_pending_files_returns_none(self):
        ctx = make_mock_ctx()
        status = MagicMock(spec=['state'])
        status.state = 3
        ctx.handle.status.return_value = status

        event = threading.Event()
        ctx.file_events = {"model.bin": event}

        assert _check_session_health(ctx) is None


# ─── _resolve_pending_metadata ───────────────────────────────────────────────

class TestResolvePendingMetadata:

    def test_resolves_when_metadata_arrives(self):
        ctx = make_mock_ctx()
        ctx.torrent_info_obj = None

        status = MagicMock()
        status.has_metadata = True
        ctx.handle.status.return_value = status

        mock_ti = MagicMock()
        mock_ti.num_files.return_value = 3
        ctx.handle.torrent_file.return_value = mock_ti

        _resolve_pending_metadata(ctx)

        assert ctx.torrent_info_obj is mock_ti
        ctx.handle.prioritize_files.assert_called_once_with([0, 0, 0])

    def test_skips_when_already_resolved(self):
        ctx = make_mock_ctx()
        existing_ti = MagicMock()
        ctx.torrent_info_obj = existing_ti

        _resolve_pending_metadata(ctx)

        # Should not call torrent_file() again
        ctx.handle.torrent_file.assert_not_called()
        assert ctx.torrent_info_obj is existing_ti


# ─── _collect_ready_files ────────────────────────────────────────────────────

class TestCollectReadyFiles:

    def test_returns_empty_without_torrent_info(self):
        ctx = make_mock_ctx()
        ctx.torrent_info_obj = None
        assert _collect_ready_files(ctx) == []

    def test_collects_completed_files(self):
        ctx = make_mock_ctx()

        event = threading.Event()
        ctx.file_events = {"model.bin": event}
        ctx.file_destinations = {"model.bin": "/dest/model.bin"}

        mock_files = MagicMock()
        mock_files.file_size.return_value = 1000

        mock_ti = MagicMock()
        mock_ti.files.return_value = mock_files
        ctx.torrent_info_obj = mock_ti

        ctx._find_file_index.return_value = 0
        ctx.handle.file_progress.return_value = [1000]
        ctx.handle.file_priorities.return_value = [1]
        ctx._get_lt_disk_path.return_value = "/tmp/p2p/model.bin"

        result = _collect_ready_files(ctx)

        assert len(result) == 1
        assert result[0] == ("/tmp/p2p/model.bin", "/dest/model.bin", "model.bin")

    def test_skips_incomplete_files(self):
        ctx = make_mock_ctx()

        event = threading.Event()
        ctx.file_events = {"model.bin": event}
        ctx.file_destinations = {"model.bin": "/dest/model.bin"}

        mock_files = MagicMock()
        mock_files.file_size.return_value = 1000

        mock_ti = MagicMock()
        mock_ti.files.return_value = mock_files
        ctx.torrent_info_obj = mock_ti

        ctx._find_file_index.return_value = 0
        ctx.handle.file_progress.return_value = [500]  # 50%
        ctx.handle.file_priorities.return_value = [1]

        result = _collect_ready_files(ctx)
        assert result == []


# ─── Integration: deliver outside lock ───────────────────────────────────────

class TestDeliverOutsideLock:

    @patch('llmpt.monitor.lt')
    def test_deliver_file_runs_without_holding_lock(self, mock_lt):
        """Critical test: _deliver_file must NOT be called while ctx.lock is held.
        This prevents large file copies from blocking download_file() callers."""
        ctx = make_mock_ctx()

        event = threading.Event()
        ctx.file_events = {"model.bin": event}
        ctx.file_destinations = {"model.bin": "/dest/model.bin"}
        ctx.auto_seed = False
        ctx.seed_start_time = None
        ctx.seed_duration = 0

        # No error
        status = MagicMock(spec=['state'])
        status.state = 3
        ctx.handle.status.return_value = status

        mock_files = MagicMock()
        mock_files.file_size.return_value = 1000
        mock_ti = MagicMock()
        mock_ti.files.return_value = mock_files
        ctx.torrent_info_obj = mock_ti

        ctx._find_file_index.return_value = 0
        ctx.handle.file_progress.return_value = [1000]
        ctx.handle.file_priorities.return_value = [1]
        ctx._get_lt_disk_path.return_value = "/tmp/p2p/model.bin"

        # Track whether the lock is held when _deliver_file is called
        lock_was_held_during_deliver = []

        original_deliver = ctx._deliver_file
        def tracking_deliver(src, dst):
            # Try to acquire the lock without blocking.
            # If the lock is already held by our thread, acquire() would
            # return True (reentrant for the same thread on non-reentrant Lock).
            # We use a separate flag instead.
            acquired = ctx.lock.acquire(blocking=False)
            if acquired:
                ctx.lock.release()
                lock_was_held_during_deliver.append(False)
            else:
                lock_was_held_during_deliver.append(True)

        ctx._deliver_file.side_effect = tracking_deliver

        _check_pending_files(ctx)

        assert len(lock_was_held_during_deliver) == 1, \
            "_deliver_file should have been called exactly once"
        assert lock_was_held_during_deliver[0] is False, \
            "_deliver_file was called while ctx.lock was held! This blocks download_file() callers."
        assert event.is_set()
