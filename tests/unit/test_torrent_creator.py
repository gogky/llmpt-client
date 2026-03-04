"""
Tests for torrent_creator module (llmpt.torrent_creator).

All libtorrent and huggingface_hub interactions are mocked.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from llmpt.utils import get_optimal_piece_length


# ─── create_torrent ───────────────────────────────────────────────────────────

class TestCreateTorrent:

    @patch('llmpt.torrent_creator.LIBTORRENT_AVAILABLE', False)
    def test_no_libtorrent_returns_none(self):
        """Should return None when libtorrent is not installed."""
        from llmpt.torrent_creator import create_torrent
        result = create_torrent("test/repo", "main", "http://tracker.example.com")
        assert result is None

    @patch('llmpt.torrent_creator.lt')
    @patch('llmpt.torrent_creator.LIBTORRENT_AVAILABLE', True)
    def test_snapshot_download_fails_returns_none(self, mock_lt):
        """If snapshot_download raises an exception, should return None."""
        from llmpt.torrent_creator import create_torrent

        with patch('llmpt.torrent_creator.snapshot_download', side_effect=Exception("no such repo"), create=True), \
             patch.dict('sys.modules', {'huggingface_hub': MagicMock()}):
            # Re-import to pick up the patched module
            import importlib
            import llmpt.torrent_creator as tc
            # Override the import inside the function
            with patch.object(tc, '__name__', 'llmpt.torrent_creator'):
                result = create_torrent("nonexistent/repo", "main", "http://tracker.example.com")

        assert result is None

    @patch('llmpt.torrent_creator.lt')
    @patch('llmpt.torrent_creator.LIBTORRENT_AVAILABLE', True)
    def test_successful_torrent_creation(self, mock_lt, tmp_path):
        """Full successful path should return a well-structured dict."""
        from llmpt.torrent_creator import create_torrent

        # Create a fake snapshot directory
        snap_dir = tmp_path / "snapshot_abc123"
        snap_dir.mkdir()
        (snap_dir / "config.json").write_text('{"key": "value"}')
        (snap_dir / "model.bin").write_bytes(b'\x00' * 1000)

        # Mock file_storage.total_size() so get_optimal_piece_length can run
        mock_fs = MagicMock()
        mock_fs.total_size.return_value = 1014  # small → 256KB piece
        mock_lt.file_storage.return_value = mock_fs

        mock_torrent_obj = MagicMock()
        mock_torrent_obj.generate.return_value = {'info': 'data'}
        mock_lt.create_torrent.return_value = mock_torrent_obj

        # Mock torrent_info and its files() for file_list extraction
        mock_files = MagicMock()
        mock_files.num_files.return_value = 2
        mock_files.file_path.side_effect = lambda i: [
            "snapshot_abc123/config.json",
            "snapshot_abc123/model.bin",
        ][i]
        mock_files.file_size.side_effect = lambda i: [14, 1000][i]

        mock_info = MagicMock()
        mock_info.info_hash.return_value = "abcdef1234567890"
        mock_info.num_pieces.return_value = 1
        mock_info.num_files.return_value = 2
        mock_info.files.return_value = mock_files
        mock_lt.torrent_info.return_value = mock_info
        mock_lt.bencode.return_value = b'bencoded_torrent'

        with patch('huggingface_hub.snapshot_download', return_value=str(snap_dir)), \
             patch('llmpt.torrent_cache.load_cached_torrent', return_value=None), \
             patch('llmpt.torrent_cache.save_torrent_to_cache'):
            result = create_torrent(
                "test/repo", "main", "http://tracker.example.com",
            )

        assert result is not None
        assert result['info_hash'] == 'abcdef1234567890'
        assert 'magnet_link' not in result
        # piece_length is auto-computed, just verify it's a positive power of two
        pl = result['piece_length']
        assert pl > 0 and (pl & (pl - 1)) == 0, f"piece_length {pl} is not a power of two"
        assert result['num_pieces'] == 1
        assert result['num_files'] == 2
        assert result['torrent_data'] == b'bencoded_torrent'
        assert result['file_size'] > 0
        # Verify files list
        assert result['files'] == [
            {'path': 'config.json', 'size': 14},
            {'path': 'model.bin', 'size': 1000},
        ]

        # Verify tracker URL was formatted correctly
        mock_torrent_obj.add_tracker.assert_called_once_with("http://tracker.example.com/announce")
        mock_torrent_obj.set_creator.assert_called_once_with("llmpt-client")

    @patch('llmpt.torrent_creator.lt')
    @patch('llmpt.torrent_creator.LIBTORRENT_AVAILABLE', True)
    def test_snapshot_dir_not_found(self, mock_lt, tmp_path):
        """If snapshot_download returns a non-existent path, should return None."""
        from llmpt.torrent_creator import create_torrent

        non_existent = str(tmp_path / "does_not_exist")

        with patch('huggingface_hub.snapshot_download', return_value=non_existent), \
             patch('llmpt.torrent_cache.load_cached_torrent', return_value=None), \
             patch('llmpt.torrent_cache.save_torrent_to_cache'):
            result = create_torrent("test/repo", "main", "http://tracker.example.com")

        assert result is None


# ─── create_and_register_torrent ──────────────────────────────────────────────

class TestCreateAndRegisterTorrent:

    @patch('llmpt.torrent_creator.create_torrent')
    def test_creation_fails_returns_false(self, mock_create):
        """If create_torrent returns None, should return False."""
        from llmpt.torrent_creator import create_and_register_torrent

        mock_create.return_value = None
        tracker = MagicMock()
        tracker.tracker_url = "http://tracker.example.com"

        result = create_and_register_torrent(
            "test/repo", "main", "model", "Test Model", tracker
        )
        assert result is False

    @patch('llmpt.torrent_creator.create_torrent')
    def test_registration_fails_returns_none(self, mock_create):
        """If tracker.register_torrent fails, should return None."""
        from llmpt.torrent_creator import create_and_register_torrent

        mock_create.return_value = {
            'info_hash': 'abc',
            'file_size': 1000,
            'piece_length': get_optimal_piece_length(1000),
            'num_files': 1,
            'torrent_data': b'fake_torrent',
            'files': [{'path': 'model.bin', 'size': 1000}],
        }
        tracker = MagicMock()
        tracker.tracker_url = "http://tracker.example.com"
        tracker.register_torrent.return_value = False

        result = create_and_register_torrent(
            "test/repo", "main", "model", "Test Model", tracker
        )
        assert result is None

    @patch('llmpt.torrent_creator.create_torrent')
    def test_successful_creation_and_registration(self, mock_create):
        """Full success: create_torrent returns info, register_torrent returns True."""
        from llmpt.torrent_creator import create_and_register_torrent

        torrent_info = {
            'info_hash': 'abc',
            'file_size': 5000,
            'piece_length': get_optimal_piece_length(5000),
            'num_files': 3,
            'num_pieces': 2,
            'torrent_data': b'fake_torrent_data',
            'files': [
                {'path': 'config.json', 'size': 100},
                {'path': 'model-00001.bin', 'size': 2450},
                {'path': 'model-00002.bin', 'size': 2450},
            ],
        }
        mock_create.return_value = torrent_info

        tracker = MagicMock()
        tracker.tracker_url = "http://tracker.example.com"
        tracker.register_torrent.return_value = True

        result = create_and_register_torrent(
            "test/repo", "main", "model", "Test Model", tracker
        )

        assert result is torrent_info
        tracker.register_torrent.assert_called_once_with(
            repo_id="test/repo",
            revision="main",
            repo_type="model",
            name="Test Model",
            info_hash='abc',
            total_size=5000,
            file_count=3,
            piece_length=get_optimal_piece_length(5000),
            torrent_data=b'fake_torrent_data',
            files=[
                {'path': 'config.json', 'size': 100},
                {'path': 'model-00001.bin', 'size': 2450},
                {'path': 'model-00002.bin', 'size': 2450},
            ],
        )
