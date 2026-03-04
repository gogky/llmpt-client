"""
Tests for the seeder module (llmpt.seeder).

All P2PBatchManager interactions are mocked.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset P2PBatchManager singleton for isolation."""
    from llmpt.p2p_batch import P2PBatchManager
    P2PBatchManager._instance = None
    yield
    P2PBatchManager._instance = None


# ─── start_seeding ────────────────────────────────────────────────────────────

class TestStartSeeding:

    @patch('llmpt.seeder.LIBTORRENT_AVAILABLE', False)
    def test_no_libtorrent_returns_false(self):
        from llmpt.seeder import start_seeding
        result = start_seeding("test/repo", "main", MagicMock())
        assert result is False

    @patch('llmpt.seeder.LIBTORRENT_AVAILABLE', True)
    @patch('llmpt.seeder.P2PBatchManager')
    def test_success(self, MockManager):
        from llmpt.seeder import start_seeding

        mock_instance = MagicMock()
        mock_instance.register_seeding_task.return_value = True
        MockManager.return_value = mock_instance

        tracker = MagicMock()
        result = start_seeding("test/repo", "main", tracker)

        assert result is True
        mock_instance.register_seeding_task.assert_called_once_with(
            repo_id="test/repo",
            revision="main",
            tracker_client=tracker,
            torrent_data=None,
        )

    @patch('llmpt.seeder.LIBTORRENT_AVAILABLE', True)
    @patch('llmpt.seeder.P2PBatchManager')
    def test_failure(self, MockManager):
        from llmpt.seeder import start_seeding

        mock_instance = MagicMock()
        mock_instance.register_seeding_task.return_value = False
        MockManager.return_value = mock_instance

        result = start_seeding("test/repo", "main", MagicMock())
        assert result is False


# ─── stop_seeding ─────────────────────────────────────────────────────────────

class TestStopSeeding:

    @patch('llmpt.seeder.P2PBatchManager')
    def test_not_seeding(self, MockManager):
        """Stopping a non-existent seeding task should return False."""
        from llmpt.seeder import stop_seeding
        import threading

        mock_instance = MagicMock()
        mock_instance._lock = threading.Lock()
        mock_instance.sessions = {}
        MockManager.return_value = mock_instance

        result = stop_seeding("no/repo", "main")
        assert result is False

    @patch('llmpt.seeder.P2PBatchManager')
    def test_stop_existing(self, MockManager):
        """Stopping an existing task should remove it and return True."""
        from llmpt.seeder import stop_seeding
        import threading

        mock_handle = MagicMock()
        mock_session_info = MagicMock()
        mock_session_info.handle = mock_handle

        mock_instance = MagicMock()
        mock_instance._lock = threading.Lock()
        mock_instance.sessions = {("test/repo", "main"): mock_session_info}
        MockManager.return_value = mock_instance

        result = stop_seeding("test/repo", "main")

        assert result is True
        mock_instance.lt_session.remove_torrent.assert_called_once_with(mock_handle)
        assert ("test/repo", "main") not in mock_instance.sessions


# ─── stop_all_seeding ─────────────────────────────────────────────────────────

class TestStopAllSeeding:

    @patch('llmpt.seeder.P2PBatchManager')
    def test_empty(self, MockManager):
        from llmpt.seeder import stop_all_seeding
        import threading

        mock_instance = MagicMock()
        mock_instance._lock = threading.Lock()
        mock_instance.sessions = {}
        MockManager.return_value = mock_instance

        count = stop_all_seeding()
        assert count == 0

    @patch('llmpt.seeder.P2PBatchManager')
    def test_multiple_tasks(self, MockManager):
        from llmpt.seeder import stop_all_seeding
        import threading

        s1 = MagicMock()
        s1.handle = MagicMock()
        s2 = MagicMock()
        s2.handle = MagicMock()
        s3 = MagicMock()
        s3.handle = None  # No handle

        mock_instance = MagicMock()
        mock_instance._lock = threading.Lock()
        mock_instance.sessions = {
            ("a", "main"): s1,
            ("b", "main"): s2,
            ("c", "dev"): s3,
        }
        MockManager.return_value = mock_instance

        count = stop_all_seeding()

        assert count == 3
        assert mock_instance.lt_session.remove_torrent.call_count == 2  # Only s1, s2 had handles
        assert len(mock_instance.sessions) == 0


# ─── get_seeding_status ───────────────────────────────────────────────────────

class TestGetSeedingStatus:

    @patch('llmpt.seeder.P2PBatchManager')
    def test_no_sessions(self, MockManager):
        from llmpt.seeder import get_seeding_status
        import threading

        mock_instance = MagicMock()
        mock_instance._lock = threading.Lock()
        mock_instance.sessions = {}
        MockManager.return_value = mock_instance

        status = get_seeding_status()
        assert status == {}

    @patch('llmpt.seeder.P2PBatchManager')
    def test_active_sessions(self, MockManager):
        from llmpt.seeder import get_seeding_status
        import threading

        mock_status = MagicMock()
        mock_status.total_upload = 1024
        mock_status.num_peers = 3
        mock_status.upload_rate = 512
        mock_status.progress = 1.0
        mock_status.state = 5

        mock_handle = MagicMock()
        mock_handle.is_valid.return_value = True
        mock_handle.status.return_value = mock_status

        session_info = MagicMock()
        session_info.handle = mock_handle

        mock_instance = MagicMock()
        mock_instance._lock = threading.Lock()
        mock_instance.sessions = {("test/repo", "main"): session_info}
        MockManager.return_value = mock_instance

        status = get_seeding_status()

        assert "test/repo@main" in status
        info = status["test/repo@main"]
        assert info['repo_id'] == "test/repo"
        assert info['revision'] == "main"
        assert info['uploaded'] == 1024
        assert info['peers'] == 3
        assert info['progress'] == 1.0

    @patch('llmpt.seeder.P2PBatchManager')
    def test_invalid_handle_skipped(self, MockManager):
        """Sessions with invalid handles should be excluded from status."""
        from llmpt.seeder import get_seeding_status
        import threading

        mock_handle = MagicMock()
        mock_handle.is_valid.return_value = False

        session_info = MagicMock()
        session_info.handle = mock_handle

        mock_instance = MagicMock()
        mock_instance._lock = threading.Lock()
        mock_instance.sessions = {("test/repo", "main"): session_info}
        MockManager.return_value = mock_instance

        status = get_seeding_status()
        assert status == {}
