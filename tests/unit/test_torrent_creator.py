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
        tracker = MagicMock()
        tracker.tracker_url = "http://tracker.example.com"
        result = create_torrent("test/repo", "main", tracker)
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
                tracker = MagicMock()
                tracker.tracker_url = "http://tracker.example.com"
                result = create_torrent("nonexistent/repo", "main", tracker)

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
             patch('llmpt.torrent_cache.resolve_torrent_data', return_value=None), \
             patch('llmpt.torrent_cache.save_torrent_to_cache'):
            tracker = MagicMock()
            tracker.tracker_url = "http://tracker.example.com"
            result = create_torrent(
                "test/repo", "main", tracker,
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

        # Verify tracker URL was formatted correctly (tracker object .tracker_url property)
        mock_torrent_obj.add_tracker.assert_called_once_with("http://tracker.example.com/announce")
        mock_torrent_obj.set_creator.assert_called_once_with("llmpt-client")

    @patch('llmpt.torrent_creator.lt')
    @patch('llmpt.torrent_creator.LIBTORRENT_AVAILABLE', True)
    def test_successful_local_dir_torrent_creation_uses_local_dir(self, mock_lt, tmp_path):
        """local_dir sources should only include verified manifest files."""
        from llmpt.torrent_creator import create_torrent

        local_dir = tmp_path / "local_dir"
        local_dir.mkdir()
        (local_dir / "config.json").write_text('{"key": "value"}')
        (local_dir / "model.bin").write_bytes(b'\x00' * 1000)
        cache_dir = local_dir / ".cache" / "huggingface" / "download"
        cache_dir.mkdir(parents=True)
        (cache_dir / "stale.lock").write_text("")

        mock_fs = MagicMock()
        mock_fs.total_size.return_value = 1014
        mock_lt.file_storage.return_value = mock_fs

        mock_torrent_obj = MagicMock()
        mock_torrent_obj.generate.return_value = {
            b'info': {
                b'name': local_dir.name.encode(),
                b'files': [
                    {b'length': 16, b'path': [b'config.json']},
                    {b'length': 1000, b'path': [b'model.bin']},
                ],
                b'piece length': 262144,
                b'pieces': b'0' * 20,
            }
        }
        mock_lt.create_torrent.return_value = mock_torrent_obj

        revision = "a" * 40
        mock_files = MagicMock()
        mock_files.num_files.return_value = 2
        mock_files.file_path.side_effect = lambda i: [
            f"{revision}/config.json",
            f"{revision}/model.bin",
        ][i]
        mock_files.file_size.side_effect = lambda i: [14, 1000][i]

        mock_info = MagicMock()
        mock_info.info_hash.return_value = "abcdef1234567890"
        mock_info.num_pieces.return_value = 1
        mock_info.num_files.return_value = 2
        mock_info.files.return_value = mock_files
        mock_lt.torrent_info.return_value = mock_info
        mock_lt.bencode.return_value = b'bencoded_torrent'

        with patch('huggingface_hub.snapshot_download', return_value=str(local_dir)) as mock_snapshot_download, \
             patch('llmpt.completed_registry.get_completed_manifest', return_value=["config.json", "model.bin"]), \
             patch('llmpt.torrent_cache.resolve_torrent_data', return_value=None), \
             patch('llmpt.torrent_cache.save_torrent_to_cache'):
            tracker = MagicMock()
            tracker.tracker_url = "http://tracker.example.com"
            result = create_torrent(
                "test/repo", revision, tracker, local_dir=str(local_dir)
            )

        assert result is not None
        assert result['commit_hash'] == revision
        mock_snapshot_download.assert_called_once_with(
            repo_id="test/repo",
            revision=revision,
            repo_type=None,
            local_files_only=True,
            local_dir=str(local_dir),
        )
        mock_lt.add_files.assert_not_called()
        mock_fs.add_file.assert_any_call("local_dir/config.json", 16)
        mock_fs.add_file.assert_any_call("local_dir/model.bin", 1000)
        assert mock_fs.add_file.call_count == 2
        mock_lt.set_piece_hashes.assert_called_once_with(mock_torrent_obj, str(local_dir.parent))

    @patch('llmpt.torrent_creator.lt')
    @patch('llmpt.torrent_creator.LIBTORRENT_AVAILABLE', True)
    def test_cached_local_dir_torrent_preserves_explicit_revision(self, mock_lt):
        """Reused local_dir torrents must not replace revision with the folder name."""
        from llmpt.torrent_creator import create_torrent

        tracker = MagicMock()
        tracker.tracker_url = "http://tracker.example.com"

        with patch('llmpt.torrent_cache.resolve_torrent_data', return_value=b'cached_torrent'), \
             patch('llmpt.torrent_creator.torrent_matches_completed_source', return_value=True), \
             patch('llmpt.torrent_creator._torrent_data_to_result', return_value={
                 'info_hash': 'abc',
                 'file_size': 1000,
                 'piece_length': 262144,
                 'num_pieces': 1,
                 'num_files': 1,
                 'torrent_data': b'cached_torrent',
                 'commit_hash': 'local_dir',
                 'files': [{'path': 'config.json', 'size': 1000}],
             }):
            result = create_torrent(
                "test/repo",
                "a" * 40,
                tracker,
                local_dir="/tmp/local_dir",
            )

        assert result is not None
        assert result["commit_hash"] == "a" * 40

    @patch('llmpt.torrent_creator.lt')
    @patch('llmpt.torrent_creator.LIBTORRENT_AVAILABLE', True)
    def test_stale_cached_local_dir_torrent_root_is_deleted_and_regenerated(self, mock_lt, tmp_path):
        """Old local_dir torrents rooted at the folder name must be regenerated."""
        from llmpt.torrent_creator import create_torrent

        revision = "a" * 40
        local_dir = tmp_path / "my_cache"
        local_dir.mkdir()
        (local_dir / "config.json").write_text('{"key": "value"}')

        mock_fs = MagicMock()
        mock_fs.total_size.return_value = 16
        mock_lt.file_storage.return_value = mock_fs

        mock_torrent_obj = MagicMock()
        mock_torrent_obj.generate.return_value = {
            b'info': {
                b'name': b'my_cache',
                b'files': [{b'length': 16, b'path': [b'config.json']}],
                b'piece length': 262144,
                b'pieces': b'0' * 20,
            }
        }
        mock_lt.create_torrent.return_value = mock_torrent_obj

        mock_files = MagicMock()
        mock_files.num_files.return_value = 1
        mock_files.file_path.return_value = f"{revision}/config.json"
        mock_files.file_size.return_value = 16

        mock_info = MagicMock()
        mock_info.info_hash.return_value = "abcdef1234567890"
        mock_info.num_pieces.return_value = 1
        mock_info.num_files.return_value = 1
        mock_info.files.return_value = mock_files
        mock_lt.torrent_info.return_value = mock_info
        mock_lt.bencode.return_value = b'bencoded_torrent'

        with patch('huggingface_hub.snapshot_download', return_value=str(local_dir)), \
             patch('llmpt.completed_registry.get_completed_manifest', return_value=["config.json"]), \
             patch('llmpt.torrent_cache.resolve_torrent_data', return_value=b'stale_torrent'), \
             patch('llmpt.torrent_creator._torrent_data_to_result', return_value={
                 'info_hash': 'old',
                 'file_size': 16,
                 'piece_length': 262144,
                 'num_pieces': 1,
                 'num_files': 1,
                 'torrent_data': b'stale_torrent',
                 'commit_hash': 'my_cache',
                 'files': [{'path': 'config.json', 'size': 16}],
             }), \
             patch('llmpt.torrent_cache.delete_cached_torrent') as mock_delete, \
             patch('llmpt.torrent_cache.save_torrent_to_cache'):
            tracker = MagicMock()
            tracker.tracker_url = "http://tracker.example.com"
            result = create_torrent(
                "test/repo", revision, tracker, local_dir=str(local_dir)
            )

        assert result is not None
        mock_delete.assert_called_once_with("test/repo", revision, repo_type="model")
        mock_lt.create_torrent.assert_called_once()

    @patch('llmpt.torrent_creator.lt')
    @patch('llmpt.torrent_creator.LIBTORRENT_AVAILABLE', True)
    def test_snapshot_dir_not_found(self, mock_lt, tmp_path):
        """If snapshot_download returns a non-existent path, should return None."""
        from llmpt.torrent_creator import create_torrent

        non_existent = str(tmp_path / "does_not_exist")

        with patch('huggingface_hub.snapshot_download', return_value=non_existent), \
             patch('llmpt.torrent_cache.resolve_torrent_data', return_value=None), \
             patch('llmpt.torrent_cache.save_torrent_to_cache'):
            tracker = MagicMock()
            tracker.tracker_url = "http://tracker.example.com"
            result = create_torrent("test/repo", "main", tracker)

        assert result is None

    @patch('llmpt.torrent_creator.lt')
    @patch('llmpt.torrent_creator.LIBTORRENT_AVAILABLE', True)
    def test_stale_cached_torrent_is_deleted_and_regenerated(self, mock_lt, tmp_path):
        """A cached torrent that does not match the completed manifest must be ignored."""
        from llmpt.torrent_creator import create_torrent

        snap_dir = tmp_path / "snapshot_abc123"
        snap_dir.mkdir()
        (snap_dir / "config.json").write_text('{"key": "value"}')
        (snap_dir / "model.bin").write_bytes(b'\x00' * 1000)

        mock_fs = MagicMock()
        mock_fs.total_size.return_value = 1014
        mock_lt.file_storage.return_value = mock_fs

        mock_torrent_obj = MagicMock()
        mock_torrent_obj.generate.return_value = {'info': 'data'}
        mock_lt.create_torrent.return_value = mock_torrent_obj

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
             patch('llmpt.torrent_cache.resolve_torrent_data', return_value=b'stale_torrent'), \
             patch('llmpt.torrent_creator.torrent_matches_completed_source', return_value=False), \
             patch('llmpt.torrent_cache.delete_cached_torrent') as mock_delete, \
             patch('llmpt.torrent_cache.save_torrent_to_cache'):
            tracker = MagicMock()
            tracker.tracker_url = "http://tracker.example.com"
            result = create_torrent("test/repo", "main", tracker)

        assert result is not None
        mock_delete.assert_called_once_with("test/repo", "main", repo_type="model")
        mock_lt.create_torrent.assert_called_once()


# ─── create_and_register_torrent ──────────────────────────────────────────────

class TestCreateAndRegisterTorrent:

    @patch('llmpt.torrent_creator.create_torrent')
    def test_creation_fails_returns_none(self, mock_create):
        """If create_torrent returns None, should return None."""
        from llmpt.torrent_creator import create_and_register_torrent

        mock_create.return_value = None
        tracker = MagicMock()
        tracker.tracker_url = "http://tracker.example.com"

        result = create_and_register_torrent(
            "test/repo", "main", "model", "Test Model", tracker
        )
        assert result is None

    @patch('llmpt.torrent_creator.create_torrent')
    def test_registration_fails_returns_none(self, mock_create, tmp_path):
        """If tracker.register_torrent fails, should return None."""
        from llmpt.torrent_creator import create_and_register_torrent
        from llmpt.torrent_state import get_torrent_state

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

        with patch("llmpt.torrent_state.TORRENT_STATE_FILE", str(tmp_path / "torrent_state_fail.json")):
            result = create_and_register_torrent(
                "test/repo", "main", "model", "Test Model", tracker
            )
            state = get_torrent_state(
                "test/repo",
                "main",
                tracker_url="http://tracker.example.com",
            )
        assert result is None
        assert state["tracker_registered"] is False
        assert state["last_registration_error"] == "register_failed"

    @patch('llmpt.torrent_creator.create_torrent')
    def test_successful_creation_and_registration(self, mock_create, tmp_path):
        """Full success: create_torrent returns info, register_torrent returns True."""
        from llmpt.torrent_creator import create_and_register_torrent
        from llmpt.torrent_state import get_torrent_state

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

        with patch("llmpt.torrent_state.TORRENT_STATE_FILE", str(tmp_path / "torrent_state_success.json")):
            result = create_and_register_torrent(
                "test/repo", "main", "model", "Test Model", tracker
            )
            state = get_torrent_state(
                "test/repo",
                "main",
                tracker_url="http://tracker.example.com",
            )

        assert result is torrent_info
        assert state["tracker_registered"] is True
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


class TestEnsureRegistered:
    def test_records_failed_registration_state(self, tmp_path):
        from llmpt.torrent_creator import ensure_registered
        from llmpt.torrent_state import get_torrent_state

        tracker = MagicMock()
        tracker.tracker_url = "http://tracker.example.com"
        tracker.get_torrent_info.return_value = None
        tracker.register_torrent.return_value = False

        with patch("llmpt.torrent_state.TORRENT_STATE_FILE", str(tmp_path / "torrent_state.json")), \
             patch("llmpt.torrent_creator._torrent_data_to_result", return_value={
                 "info_hash": "abc",
                 "file_size": 1000,
                 "piece_length": 262144,
                 "num_files": 1,
                 "files": [{"path": "model.bin", "size": 1000}],
                 "commit_hash": "a" * 40,
             }):
            result = ensure_registered(
                "test/repo",
                "a" * 40,
                "model",
                b"torrent",
                tracker,
            )
            state = get_torrent_state(
                "test/repo",
                "a" * 40,
                tracker_url="http://tracker.example.com",
            )

        assert result is False
        assert state["tracker_registered"] is False
        assert state["last_registration_error"] == "register_failed"

    @patch('llmpt.torrent_creator.create_torrent')
    def test_registration_passes_storage_args(self, mock_create):
        from llmpt.torrent_creator import create_and_register_torrent

        mock_create.return_value = None
        tracker = MagicMock()
        tracker.tracker_url = "http://tracker.example.com"

        create_and_register_torrent(
            "test/repo",
            "a" * 40,
            "model",
            "Test Model",
            tracker,
            cache_dir="/tmp/custom-cache",
            local_dir="/tmp/custom-local",
        )

        mock_create.assert_called_once_with(
            repo_id="test/repo",
            revision="a" * 40,
            tracker_client=tracker,
            repo_type="model",
            cache_dir="/tmp/custom-cache",
            local_dir="/tmp/custom-local",
        )
