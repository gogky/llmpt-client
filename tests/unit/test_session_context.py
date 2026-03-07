"""
Tests for SessionContext (llmpt.session_context).

Covers: _init_torrent, download_file, _find_file_index, _deliver_file,
        _get_lt_disk_path, map_all_files_for_seeding.
All libtorrent interactions are mocked.
"""

import os
import threading
import pytest
from unittest.mock import patch, MagicMock, call, mock_open


def _make_mock_lt():
    """Shared mock for libtorrent."""
    mock_lt = MagicMock()
    mock_lt.torrent_flags.paused = 0
    mock_lt.save_resume_flags_t.flush_disk_cache = 0
    return mock_lt


@pytest.fixture
def mock_lt():
    """Patch libtorrent in session_context and torrent_init modules."""
    m = _make_mock_lt()
    with patch('llmpt.session_context.lt', m), \
         patch('llmpt.session_context.LIBTORRENT_AVAILABLE', True), \
         patch('llmpt.torrent_init.lt', m):
        yield m


@pytest.fixture
def make_ctx(mock_lt):
    """Factory for creating a SessionContext with mocked libtorrent."""
    def _factory(repo_id="test/repo", revision="main", timeout=10, torrent_data=None, session_mode='on_demand'):
        from llmpt.session_context import SessionContext
        tracker = MagicMock()
        lt_session = MagicMock()
        with patch('os.makedirs'):
            ctx = SessionContext(repo_id, revision, tracker, lt_session, session_mode=session_mode, timeout=timeout, torrent_data=torrent_data)
        return ctx
    return _factory


def _setup_successful_init(ctx, mock_lt):
    """Helper: configure mocks for a successful _init_torrent() execution.

    NOTE: This sets up the libtorrent mocks. Callers must also patch
    ``llmpt.torrent_cache.resolve_torrent_data`` to return torrent bytes,
    since _init_torrent now uses the three-layer cache resolver.
    """

    mock_params = MagicMock()
    mock_params.flags = 0
    mock_lt.add_torrent_params.return_value = mock_params

    mock_handle = MagicMock()
    mock_ti = MagicMock()
    mock_ti.num_files.return_value = 1
    mock_handle.torrent_file.return_value = mock_ti
    ctx.lt_session.add_torrent.return_value = mock_handle

    return mock_handle, mock_ti


# ─── _find_file_index ─────────────────────────────────────────────────────────

class TestFindFileIndex:

    def test_returns_none_without_torrent_info(self, make_ctx):
        ctx = make_ctx()
        ctx.torrent_info_obj = None
        assert ctx._find_file_index("model.bin") is None

    def test_exact_match(self, make_ctx):
        """File path in torrent matches target exactly (single-file torrent)."""
        ctx = make_ctx()
        mock_files = MagicMock()
        mock_files.num_files.return_value = 1
        mock_files.file_path.return_value = "config.json"
        ctx.torrent_info_obj = MagicMock()
        ctx.torrent_info_obj.files.return_value = mock_files

        assert ctx._find_file_index("config.json") == 0

    def test_multi_file_strips_root(self, make_ctx):
        """Standard multi-file torrent: root_folder/filename → strips root."""
        ctx = make_ctx()
        mock_files = MagicMock()
        mock_files.num_files.return_value = 2
        mock_files.file_path.side_effect = lambda i: [
            "test_repo_main/config.json",
            "test_repo_main/model.bin",
        ][i]
        ctx.torrent_info_obj = MagicMock()
        ctx.torrent_info_obj.files.return_value = mock_files

        assert ctx._find_file_index("config.json") == 0
        assert ctx._find_file_index("model.bin") == 1

    def test_not_found(self, make_ctx):
        ctx = make_ctx()
        mock_files = MagicMock()
        mock_files.num_files.return_value = 1
        mock_files.file_path.return_value = "root/other.txt"
        ctx.torrent_info_obj = MagicMock()
        ctx.torrent_info_obj.files.return_value = mock_files

        assert ctx._find_file_index("model.bin") is None

    def test_backslash_normalization(self, make_ctx):
        """Windows-style backslashes should be normalized."""
        ctx = make_ctx()
        mock_files = MagicMock()
        mock_files.num_files.return_value = 1
        mock_files.file_path.return_value = "root\\subdir\\model.bin"
        ctx.torrent_info_obj = MagicMock()
        ctx.torrent_info_obj.files.return_value = mock_files

        assert ctx._find_file_index("subdir/model.bin") == 0


# ─── _get_lt_disk_path ────────────────────────────────────────────────────────

class TestGetLtDiskPath:

    def test_joins_temp_dir_and_file_path(self, make_ctx):
        ctx = make_ctx()
        ctx.temp_dir = "/tmp/p2p_root"
        mock_files = MagicMock()
        mock_files.file_path.return_value = "repo/model.bin"
        ctx.torrent_info_obj = MagicMock()
        ctx.torrent_info_obj.files.return_value = mock_files

        result = ctx._get_lt_disk_path(0)
        assert result == os.path.join("/tmp/p2p_root", "repo/model.bin")


# ─── _deliver_file ────────────────────────────────────────────────────────────

class TestDeliverFile:

    def test_hard_link_success(self, make_ctx, tmp_path):
        """Source file should be cleaned up after delivery."""
        ctx = make_ctx()

        src = tmp_path / "src.bin"
        src.write_bytes(b"hello")
        src_inode = os.stat(str(src)).st_ino
        dst = tmp_path / "subdir" / "dst.bin"

        ctx._deliver_file(str(src), str(dst))

        assert dst.exists()
        assert dst.read_bytes() == b"hello"
        # src MUST NOT be deleted so Libtorrent can continue reading chunks
        assert src.exists()
        assert os.stat(str(dst)).st_ino == src_inode

    def test_cross_device_fallback(self, make_ctx, tmp_path):
        """When os.link fails with OSError, should fall back to shutil.copy2."""
        ctx = make_ctx()

        src = tmp_path / "src.bin"
        src.write_bytes(b"data")
        dst = tmp_path / "dst.bin"

        with patch('os.link', side_effect=OSError("cross-device")), \
             patch('shutil.copy2') as mock_copy:
            ctx._deliver_file(str(src), str(dst))
            mock_copy.assert_called_once_with(str(src), str(dst))

    def test_overwrites_existing_destination(self, make_ctx, tmp_path):
        """If dst already exists, it should be removed before linking."""
        ctx = make_ctx()

        src = tmp_path / "src.bin"
        src.write_bytes(b"new data")
        dst = tmp_path / "dst.bin"
        dst.write_bytes(b"old data")

        ctx._deliver_file(str(src), str(dst))

        assert dst.read_bytes() == b"new data"


# ─── _init_torrent ────────────────────────────────────────────────────────────

class TestInitTorrent:

    def test_already_initialized(self, make_ctx, mock_lt):
        """If handle is not None, should return True immediately."""
        ctx = make_ctx()
        ctx.handle = MagicMock()
        assert ctx._init_torrent() is True

    def test_no_torrent_data_available(self, make_ctx, mock_lt):
        """If all three cache layers return None, should mark invalid and return False."""
        ctx = make_ctx()
        with patch('llmpt.torrent_cache.resolve_torrent_data', return_value=None):
            result = ctx._init_torrent()
        assert result is False
        assert ctx.is_valid is False

    def test_hash_revision_no_fallback_to_main(self, make_ctx, mock_lt):
        """After revision unification (1.1), there is no 'retry with main'
        fallback.  A missing cache/tracker entry should fail immediately."""
        ctx = make_ctx(revision="a" * 40)

        with patch('llmpt.torrent_cache.resolve_torrent_data', return_value=None) as mock_resolve:
            result = ctx._init_torrent()

        assert result is False
        assert ctx.is_valid is False
        # Should only call resolver once, no fallback
        mock_resolve.assert_called_once_with(
            "test/repo", "a" * 40, ctx.tracker_client, repo_type="model"
        )

    def test_successful_torrent_init(self, make_ctx, mock_lt):
        """Full successful path: cache resolve → bdecode → add_torrent → resume."""
        ctx = make_ctx()
        mock_handle, mock_ti = _setup_successful_init(ctx, mock_lt)
        mock_ti.num_files.return_value = 3

        with patch('llmpt.session_context.run_monitor_loop'), \
             patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch('llmpt.torrent_cache.resolve_torrent_data', return_value=b'fake_torrent_bytes'):
            result = ctx._init_torrent()

        assert result is True
        assert ctx.handle is mock_handle
        assert ctx.torrent_info_obj is mock_ti
        mock_handle.prioritize_files.assert_called_once_with([0, 0, 0])
        mock_lt.bdecode.assert_called_once_with(b'fake_torrent_bytes')
        mock_lt.torrent_info.assert_called_once()

    def test_local_torrent_data_skips_tracker(self, make_ctx, mock_lt):
        """When torrent_data is provided locally, should skip tracker download."""
        ctx = make_ctx(torrent_data=b'local_torrent_bytes')

        mock_handle = MagicMock()
        mock_ti = MagicMock()
        mock_ti.num_files.return_value = 1
        mock_handle.torrent_file.return_value = mock_ti
        ctx.lt_session.add_torrent.return_value = mock_handle

        with patch('llmpt.session_context.run_monitor_loop'), \
             patch('os.path.exists', return_value=False), \
             patch('os.makedirs'):
            result = ctx._init_torrent()

        assert result is True
        mock_lt.bdecode.assert_called_once_with(b'local_torrent_bytes')
        mock_lt.torrent_info.assert_called_once()
        # Should NOT call tracker since local data was available
        ctx.tracker_client.download_torrent.assert_not_called()

    def test_exception_marks_invalid(self, make_ctx, mock_lt):
        """Exception during init should set is_valid=False and return False."""
        ctx = make_ctx()
        mock_lt.bdecode.side_effect = RuntimeError("bad torrent data")

        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch('llmpt.torrent_cache.resolve_torrent_data', return_value=b'bad_data'):
            result = ctx._init_torrent()

        assert result is False
        assert ctx.is_valid is False

    def test_fastresume_loading(self, make_ctx, mock_lt):
        """When fastresume file exists, it should be loaded and applied."""
        ctx = make_ctx()
        mock_handle, mock_ti = _setup_successful_init(ctx, mock_lt)

        resume_bytes = b'resume_data_bytes'
        mock_lt.bdecode.return_value = {b'mapped_files': {}}
        # Simulate lt.add_torrent_params does NOT have parse_resume_data
        mock_lt.add_torrent_params.parse_resume_data = None
        delattr(mock_lt.add_torrent_params, 'parse_resume_data')

        with patch('llmpt.session_context.run_monitor_loop'), \
             patch('os.path.exists', return_value=True), \
             patch('os.makedirs'), \
             patch('builtins.open', mock_open(read_data=resume_bytes)), \
             patch('llmpt.torrent_cache.resolve_torrent_data', return_value=b'fake_torrent_bytes'):
            result = ctx._init_torrent()

        assert result is True
        assert ctx.handle is mock_handle

    def test_fastresume_with_read_resume_data(self, make_ctx, mock_lt):
        """When lt.add_torrent_params has parse_resume_data, should use read_resume_data API."""
        ctx = make_ctx()
        mock_handle, mock_ti = _setup_successful_init(ctx, mock_lt)

        resume_bytes = b'resume_data_bytes'
        mock_lt.bdecode.return_value = {b'mapped_files': {}}
        # Simulate lt.add_torrent_params HAS parse_resume_data
        mock_lt.add_torrent_params.parse_resume_data = True

        mock_resumed_params = MagicMock()
        mock_resumed_params.flags = 0
        mock_lt.read_resume_data.return_value = mock_resumed_params

        with patch('llmpt.session_context.run_monitor_loop'), \
             patch('os.path.exists', return_value=True), \
             patch('os.makedirs'), \
             patch('builtins.open', mock_open(read_data=resume_bytes)), \
             patch('llmpt.torrent_cache.resolve_torrent_data', return_value=b'fake_torrent_bytes'):
            result = ctx._init_torrent()

        assert result is True
        mock_lt.read_resume_data.assert_called_once_with(resume_bytes)

    def test_test_seeder_peer_env(self, make_ctx, mock_lt):
        """When TEST_SEEDER_PEER env is set, should connect to the specified peer."""
        ctx = make_ctx()
        mock_handle, mock_ti = _setup_successful_init(ctx, mock_lt)

        with patch('llmpt.session_context.run_monitor_loop'), \
             patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch.dict(os.environ, {'TEST_SEEDER_PEER': 'seeder-host:6881'}), \
             patch('llmpt.torrent_init.socket.gethostbyname', return_value='10.0.0.5') as mock_dns, \
             patch('llmpt.torrent_cache.resolve_torrent_data', return_value=b'fake_torrent_bytes'):
            result = ctx._init_torrent()

        assert result is True
        mock_dns.assert_called_once_with('seeder-host')
        assert ctx.test_peer_addr == ('10.0.0.5', 6881)

    def test_test_seeder_peer_no_port(self, make_ctx, mock_lt):
        """When TEST_SEEDER_PEER has no port, should default to 6881."""
        ctx = make_ctx()
        mock_handle, mock_ti = _setup_successful_init(ctx, mock_lt)

        with patch('llmpt.session_context.run_monitor_loop'), \
             patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch.dict(os.environ, {'TEST_SEEDER_PEER': 'seeder-host'}), \
             patch('llmpt.torrent_init.socket.gethostbyname', return_value='10.0.0.5') as mock_dns, \
             patch('llmpt.torrent_cache.resolve_torrent_data', return_value=b'fake_torrent_bytes'):
            result = ctx._init_torrent()

        assert result is True
        mock_dns.assert_called_once_with('seeder-host')
        assert ctx.test_peer_addr == ('10.0.0.5', 6881)

    def test_test_seeder_peer_ipv6_bracket(self, make_ctx, mock_lt):
        """When TEST_SEEDER_PEER uses [IPv6]:port notation, should parse correctly."""
        ctx = make_ctx()
        mock_handle, mock_ti = _setup_successful_init(ctx, mock_lt)

        with patch('llmpt.session_context.run_monitor_loop'), \
             patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch.dict(os.environ, {'TEST_SEEDER_PEER': '[::1]:7000'}), \
             patch('llmpt.torrent_init.socket.gethostbyname', return_value='::1') as mock_dns, \
             patch('llmpt.torrent_cache.resolve_torrent_data', return_value=b'fake_torrent_bytes'):
            result = ctx._init_torrent()

        assert result is True
        mock_dns.assert_called_once_with('::1')
        assert ctx.test_peer_addr == ('::1', 7000)



# ─── download_file ────────────────────────────────────────────────────────────

class TestDownloadFile:

    def test_returns_false_when_invalid(self, make_ctx, mock_lt):
        ctx = make_ctx()
        ctx.is_valid = False
        assert ctx.download_file("model.bin", "/dest/model.bin") is False

    def test_returns_false_when_init_fails_and_invalid(self, make_ctx, mock_lt):
        """If _init_torrent fails and marks invalid, should return False."""
        ctx = make_ctx()
        with patch('llmpt.torrent_cache.resolve_torrent_data', return_value=None):
            result = ctx.download_file("model.bin", "/dest/model.bin")
        assert result is False

    def test_file_not_in_torrent_returns_false(self, make_ctx, mock_lt):
        """If file is not in metadata, should return False (fallback to HTTP)."""
        ctx = make_ctx()
        ctx.handle = MagicMock()
        ctx.is_valid = True

        mock_ti = MagicMock()
        mock_files = MagicMock()
        mock_files.num_files.return_value = 1
        mock_files.file_path.return_value = "root/other_file.txt"
        mock_ti.files.return_value = mock_files
        ctx.torrent_info_obj = mock_ti

        with patch.object(ctx, '_init_torrent', return_value=True):
            result = ctx.download_file("nonexistent.bin", "/dest/nonexistent.bin")

        assert result is False

    def test_timeout_returns_false(self, make_ctx, mock_lt):
        """If event is never set within timeout, should return False."""
        ctx = make_ctx(timeout=1)  # 1 second timeout
        ctx.handle = MagicMock()
        ctx.is_valid = True

        mock_ti = MagicMock()
        mock_files = MagicMock()
        mock_files.num_files.return_value = 1
        mock_files.file_path.return_value = "root/model.bin"
        mock_files.file_size.return_value = 1000
        mock_ti.files.return_value = mock_files
        ctx.torrent_info_obj = mock_ti

        ctx.handle.status.return_value.state = 3  # downloading
        ctx.handle.file_progress.return_value = [0]

        with patch.object(ctx, '_init_torrent', return_value=True):
            result = ctx.download_file("model.bin", "/dest/model.bin")

        assert result is False

    def test_immediate_delivery_when_torrent_finished(self, make_ctx, mock_lt, tmp_path):
        """If torrent is already finished, file should be delivered immediately."""
        ctx = make_ctx(timeout=5)
        ctx.handle = MagicMock()
        ctx.is_valid = True

        mock_files = MagicMock()
        mock_files.num_files.return_value = 1
        mock_files.file_path.side_effect = lambda i: "root/model.bin"
        mock_files.file_size.return_value = 100

        mock_ti = MagicMock()
        mock_ti.files.return_value = mock_files
        ctx.torrent_info_obj = mock_ti
        ctx.temp_dir = str(tmp_path)

        ctx.handle.status.return_value.state = 4
        ctx.handle.file_progress.return_value = [100]

        src_dir = tmp_path / "root"
        src_dir.mkdir()
        src_file = src_dir / "model.bin"
        src_file.write_bytes(b"x" * 100)

        dst = tmp_path / "dest" / "model.bin"

        with patch.object(ctx, '_init_torrent', return_value=True):
            result = ctx.download_file("model.bin", str(dst))

        assert result is True
        assert dst.exists()


# ─── map_all_files_for_seeding ────────────────────────────────────────────────

class TestMapAllFilesForSeeding:

    def test_no_handle_returns_false(self, make_ctx):
        """Should return False if handle is None."""
        ctx = make_ctx()
        ctx.handle = None
        assert ctx.map_all_files_for_seeding() is False

    def test_no_torrent_info_returns_false(self, make_ctx):
        """Should return False if torrent_info_obj is None."""
        ctx = make_ctx()
        ctx.handle = MagicMock()
        ctx.torrent_info_obj = None
        assert ctx.map_all_files_for_seeding() is False

    def test_normal_file_mapping_uses_hardlinks_and_seed_mode(self, make_ctx, mock_lt):
        """Normal files should be hardlinked and seed_mode enabled (no force_recheck)."""
        ctx = make_ctx()
        ctx.handle = MagicMock()
        ctx.temp_dir = "/tmp/p2p_root"

        mock_files = MagicMock()
        mock_files.num_files.return_value = 2
        mock_files.file_path.side_effect = lambda i: [
            "root/config.json",
            "root/model.bin",
        ][i]
        mock_files.file_size.side_effect = lambda i: [100, 5000][i]

        mock_ti = MagicMock()
        mock_ti.files.return_value = mock_files
        mock_ti.num_files.return_value = 2
        ctx.torrent_info_obj = mock_ti

        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch('os.link') as mock_link, \
             patch('os.unlink'), \
             patch('huggingface_hub.try_to_load_from_cache') as mock_cache, \
             patch('os.path.realpath') as mock_realpath:

            mock_cache.side_effect = lambda repo_id, filename, revision, **kwargs: {
                "config.json": "/hf/snapshots/abc/config.json",
                "model.bin": "/hf/snapshots/abc/model.bin",
            }.get(filename)

            mock_realpath.side_effect = lambda p: p.replace("snapshots/abc", "blobs/sha256hash")

            result = ctx.map_all_files_for_seeding()

        assert result is True
        # Hardlinks should be created (not rename_file)
        assert mock_link.call_count == 2
        ctx.handle.rename_file.assert_not_called()
        # seed_mode should be set (not force_recheck)
        ctx.handle.set_flags.assert_called_once_with(mock_lt.torrent_flags.seed_mode)
        ctx.handle.resume.assert_called_once()
        ctx.handle.force_recheck.assert_not_called()
        # Hardlink paths should be tracked for cleanup
        assert len(ctx.seeding_hardlinks) == 2

    def test_padding_file_handling(self, make_ctx, mock_lt, tmp_path):
        """Padding files (.pad/) should create zero-filled files at the expected lt path."""
        ctx = make_ctx()
        ctx.handle = MagicMock()
        ctx.temp_dir = str(tmp_path)

        mock_files = MagicMock()
        mock_files.num_files.return_value = 1
        mock_files.file_path.return_value = "root/.pad/1024"
        mock_files.file_size.return_value = 1024

        mock_ti = MagicMock()
        mock_ti.files.return_value = mock_files
        mock_ti.num_files.return_value = 1
        ctx.torrent_info_obj = mock_ti

        result = ctx.map_all_files_for_seeding()

        assert result is True
        # Padding file should be created at the expected libtorrent path
        expected_pad_path = str(tmp_path / "root" / ".pad" / "1024")
        assert os.path.exists(expected_pad_path)
        assert os.path.getsize(expected_pad_path) == 1024
        # seed_mode should still be enabled
        ctx.handle.set_flags.assert_called_once_with(mock_lt.torrent_flags.seed_mode)
        ctx.handle.force_recheck.assert_not_called()

    def test_cache_miss_continues(self, make_ctx, mock_lt):
        """Files not in HF cache should be skipped but not fail."""
        ctx = make_ctx()
        ctx.handle = MagicMock()
        ctx.temp_dir = "/tmp/p2p_root"

        mock_files = MagicMock()
        mock_files.num_files.return_value = 1
        mock_files.file_path.return_value = "root/missing.bin"
        mock_files.file_size.return_value = 500

        mock_ti = MagicMock()
        mock_ti.files.return_value = mock_files
        mock_ti.num_files.return_value = 1
        ctx.torrent_info_obj = mock_ti

        with patch('os.makedirs'), \
             patch('huggingface_hub.try_to_load_from_cache', return_value=None):
            result = ctx.map_all_files_for_seeding()

        assert result is True
        # No hardlinks or rename_file for cache-missed files
        ctx.handle.rename_file.assert_not_called()
        # seed_mode and resume should still be called
        ctx.handle.set_flags.assert_called_once_with(mock_lt.torrent_flags.seed_mode)
        ctx.handle.resume.assert_called_once()
        ctx.handle.force_recheck.assert_not_called()

    def test_cross_filesystem_fallback(self, make_ctx, mock_lt):
        """When hardlink fails (OSError), should fall back to legacy rename_file + force_recheck."""
        ctx = make_ctx()
        ctx.handle = MagicMock()
        ctx.temp_dir = "/tmp/p2p_root"

        mock_files = MagicMock()
        mock_files.num_files.return_value = 1
        mock_files.file_path.return_value = "root/model.bin"
        mock_files.file_size.return_value = 5000

        mock_ti = MagicMock()
        mock_ti.files.return_value = mock_files
        mock_ti.num_files.return_value = 1
        ctx.torrent_info_obj = mock_ti

        # Simulate recheck completing immediately
        mock_status = MagicMock()
        mock_status.state = 5  # seeding
        mock_status.progress = 1.0
        mock_status.num_pieces = 10
        ctx.handle.status.return_value = mock_status

        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch('os.link', side_effect=OSError("cross-device link")), \
             patch('huggingface_hub.try_to_load_from_cache', return_value="/hf/blobs/abc"), \
             patch('os.path.realpath', return_value="/hf/blobs/abc"):
            result = ctx.map_all_files_for_seeding()

        assert result is True
        # Should have fallen back to legacy: rename_file + force_recheck
        ctx.handle.rename_file.assert_called()
        ctx.handle.force_recheck.assert_called_once()
        # seed_mode should NOT have been set
        ctx.handle.set_flags.assert_not_called()

    def test_cleanup_seeding_hardlinks(self, make_ctx, tmp_path):
        """_cleanup_seeding_hardlinks should remove tracked hardlink files."""
        ctx = make_ctx()

        # Create some fake hardlink files
        f1 = tmp_path / "link1.bin"
        f2 = tmp_path / "link2.bin"
        f1.write_bytes(b"data")
        f2.write_bytes(b"data")

        ctx.seeding_hardlinks = [str(f1), str(f2)]
        ctx._cleanup_seeding_hardlinks()

        assert not f1.exists()
        assert not f2.exists()
        assert ctx.seeding_hardlinks == []
