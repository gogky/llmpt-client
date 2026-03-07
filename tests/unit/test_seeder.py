"""
Tests for the seeder module (llmpt.seeder).

Now that seeder.py is a thin façade over P2PBatchManager's public API,
these tests verify that each function correctly delegates to the
corresponding manager method.

Singleton reset is handled by conftest autouse fixture.
"""

import pytest
from unittest.mock import patch, MagicMock


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
            repo_type="model",
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

        mock_instance = MagicMock()
        mock_instance.remove_session.return_value = False
        MockManager.return_value = mock_instance

        result = stop_seeding("no/repo", "main")
        assert result is False
        mock_instance.remove_session.assert_called_once_with("no/repo", "main", repo_type="model")

    @patch('llmpt.seeder.P2PBatchManager')
    def test_stop_existing(self, MockManager):
        """Stopping an existing task should delegate to manager and return True."""
        from llmpt.seeder import stop_seeding

        mock_instance = MagicMock()
        mock_instance.remove_session.return_value = True
        MockManager.return_value = mock_instance

        result = stop_seeding("test/repo", "main")
        assert result is True
        mock_instance.remove_session.assert_called_once_with("test/repo", "main", repo_type="model")


# ─── stop_all_seeding ─────────────────────────────────────────────────────────

class TestStopAllSeeding:

    @patch('llmpt.seeder.P2PBatchManager')
    def test_empty(self, MockManager):
        from llmpt.seeder import stop_all_seeding

        mock_instance = MagicMock()
        mock_instance.remove_all_sessions.return_value = 0
        MockManager.return_value = mock_instance

        count = stop_all_seeding()
        assert count == 0
        mock_instance.remove_all_sessions.assert_called_once()

    @patch('llmpt.seeder.P2PBatchManager')
    def test_multiple_tasks(self, MockManager):
        from llmpt.seeder import stop_all_seeding

        mock_instance = MagicMock()
        mock_instance.remove_all_sessions.return_value = 3
        MockManager.return_value = mock_instance

        count = stop_all_seeding()
        assert count == 3


# ─── get_seeding_status ───────────────────────────────────────────────────────

class TestGetSeedingStatus:

    @patch('llmpt.seeder.P2PBatchManager')
    def test_no_sessions(self, MockManager):
        from llmpt.seeder import get_seeding_status

        mock_instance = MagicMock()
        mock_instance.get_all_session_status.return_value = {}
        MockManager.return_value = mock_instance

        status = get_seeding_status()
        assert status == {}

    @patch('llmpt.seeder.P2PBatchManager')
    def test_active_sessions(self, MockManager):
        from llmpt.seeder import get_seeding_status

        mock_instance = MagicMock()
        mock_instance.get_all_session_status.return_value = {
            "test/repo@main": {
                'repo_id': "test/repo",
                'revision': "main",
                'uploaded': 1024,
                'peers': 3,
                'upload_rate': 512,
                'progress': 1.0,
                'state': '5',
            }
        }
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
    def test_delegates_to_manager(self, MockManager):
        """get_seeding_status should call get_all_session_status exactly once."""
        from llmpt.seeder import get_seeding_status

        mock_instance = MagicMock()
        mock_instance.get_all_session_status.return_value = {}
        MockManager.return_value = mock_instance

        get_seeding_status()
        mock_instance.get_all_session_status.assert_called_once()
