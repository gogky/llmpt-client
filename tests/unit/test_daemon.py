"""
Tests for llmpt.daemon module.
"""

import os
import signal
import time

import pytest

from llmpt.daemon import (
    _read_pid,
    _write_pid,
    _remove_pid,
    _is_process_running,
    is_daemon_running,
    PID_FILE,
)


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
