"""
Tests for llmpt.daemon module.
"""

import os
import signal
import time
from unittest.mock import MagicMock, patch

import pytest

from llmpt.daemon import (
    _read_pid,
    _refresh_tracker_registration_state,
    _seeding_key,
    _write_pid,
    _remove_pid,
    _is_process_running,
    is_daemon_running,
    PID_FILE,
)
from llmpt.utils import get_hf_hub_cache


@pytest.fixture
def mock_pid_file(tmp_path, monkeypatch):
    """Use a temporary PID file for testing."""
    pid_file = str(tmp_path / "test_daemon.pid")
    monkeypatch.setattr("llmpt.daemon.PID_FILE", pid_file)
    monkeypatch.setattr("llmpt.daemon.LLMPT_DIR", str(tmp_path))
    return pid_file


class TestPidManagement:
    """Tests for PID file read/write/remove."""

    def test_write_and_read_pid(self, mock_pid_file):
        _write_pid(12345)
        assert _read_pid() == 12345

    def test_read_nonexistent(self, mock_pid_file):
        assert _read_pid() is None

    def test_remove_pid(self, mock_pid_file):
        _write_pid(12345)
        _remove_pid()
        assert _read_pid() is None

    def test_remove_nonexistent_pid(self, mock_pid_file):
        # Should not raise
        _remove_pid()


class TestIsProcessRunning:
    """Tests for _is_process_running()."""

    def test_current_process_is_running(self):
        assert _is_process_running(os.getpid()) is True

    def test_nonexistent_pid(self):
        # PID 99999999 almost certainly doesn't exist
        assert _is_process_running(99999999) is False


class TestIsDaemonRunning:
    """Tests for is_daemon_running()."""

    def test_no_pid_file(self, mock_pid_file):
        assert is_daemon_running() is None

    def test_stale_pid_file(self, mock_pid_file):
        """Stale PID file (process not running) returns None and cleans up."""
        _write_pid(99999999)
        assert is_daemon_running() is None
        # PID file should be cleaned up
        assert _read_pid() is None

    def test_running_process(self, mock_pid_file):
        """If PID is our own process, it should be detected as running."""
        _write_pid(os.getpid())
        result = is_daemon_running()
        assert result == os.getpid()


class TestProcessSeedable:

    def test_stale_cached_torrent_is_regenerated(self):
        from llmpt.daemon import _process_seedable

        tracker = MagicMock()
        tracker.tracker_url = "http://tracker.example.com"
        manager = MagicMock()
        manager.register_seeding_task.return_value = True
        seeding_set = set()
        failed_attempts = {}

        with patch("llmpt.torrent_cache.resolve_torrent_data", return_value=b"stale"), \
             patch("llmpt.torrent_cache.delete_cached_torrent") as mock_delete, \
             patch("llmpt.torrent_creator.torrent_matches_completed_source", return_value=False), \
             patch("llmpt.torrent_creator.create_and_register_torrent", return_value={
                 "torrent_data": b"fresh",
                 "info_hash": "abc123",
             }) as mock_create, \
             patch("llmpt.torrent_cache.save_torrent_to_cache"), \
             patch("llmpt.torrent_creator.ensure_registered"):
            result = _process_seedable(
                "test/repo",
                "a" * 40,
                tracker,
                manager,
                seeding_set,
                failed_attempts,
            )

        assert result is True
        mock_delete.assert_called_once_with("test/repo", "a" * 40, repo_type="model")
        mock_create.assert_called_once()
        manager.register_seeding_task.assert_called_once_with(
            repo_id="test/repo",
            revision="a" * 40,
            repo_type="model",
            tracker_client=tracker,
            torrent_data=b"fresh",
            cache_dir=None,
            local_dir=None,
        )


class TestTrackerStateRefresh:

    def test_refresh_tracker_registration_repairs_missing_state(self, monkeypatch, tmp_path):
        from llmpt.torrent_state import get_torrent_state

        state_file = tmp_path / "torrent_state.json"
        monkeypatch.setattr("llmpt.torrent_state.TORRENT_STATE_FILE", str(state_file))

        tracker = MagicMock()
        tracker.tracker_url = "http://tracker.example.com"
        tracker.get_torrent_info.return_value = {"info_hash": "abc123"}

        state = _refresh_tracker_registration_state(
            "test/repo",
            "a" * 40,
            "model",
            tracker,
        )

        assert state["tracker_registered"] is True
        persisted = get_torrent_state(
            "test/repo",
            "a" * 40,
            tracker_url="http://tracker.example.com",
        )
        assert persisted["tracker_registered"] is True

    def test_refresh_tracker_registration_keeps_false_on_tracker_miss(self, monkeypatch, tmp_path):
        from llmpt.torrent_state import get_torrent_state

        state_file = tmp_path / "torrent_state.json"
        monkeypatch.setattr("llmpt.torrent_state.TORRENT_STATE_FILE", str(state_file))

        tracker = MagicMock()
        tracker.tracker_url = "http://tracker.example.com"
        tracker.get_torrent_info.return_value = None

        state = _refresh_tracker_registration_state(
            "test/repo",
            "b" * 40,
            "model",
            tracker,
        )

        assert state["tracker_registered"] is False
        persisted = get_torrent_state(
            "test/repo",
            "b" * 40,
            tracker_url="http://tracker.example.com",
        )
        assert persisted["tracker_registered"] is False


def test_seeding_key_normalizes_default_hub_cache():
    assert _seeding_key("model", "demo/repo", "a" * 40) == (
        "model",
        "demo/repo",
        "a" * 40,
        "hub_cache",
        get_hf_hub_cache(),
    )
