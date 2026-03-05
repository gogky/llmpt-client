"""
Tests for llmpt.seeding_mapper — helpers extracted from SessionContext.map_all_files_for_seeding().
"""

import os
import pytest
from unittest.mock import patch, MagicMock, mock_open


# ─── resolve_hf_blob ─────────────────────────────────────────────────────────

class TestResolveHfBlob:

    @patch('huggingface_hub.try_to_load_from_cache', return_value="/fake/blobs/abc123")
    @patch('os.path.realpath', return_value="/real/blobs/abc123")
    def test_resolves_path(self, mock_real, mock_cache):
        from llmpt.seeding_mapper import resolve_hf_blob

        result = resolve_hf_blob("test/repo", "config.json", "main")
        assert result == "/real/blobs/abc123"
        mock_cache.assert_called_once_with(repo_id="test/repo", filename="config.json", revision="main")

    @patch('huggingface_hub.try_to_load_from_cache', return_value=None)
    def test_returns_none_on_cache_miss(self, mock_cache):
        from llmpt.seeding_mapper import resolve_hf_blob

        result = resolve_hf_blob("test/repo", "missing.bin", "main")
        assert result is None

    @patch('huggingface_hub.try_to_load_from_cache', side_effect=Exception("hf error"))
    def test_returns_none_on_exception(self, mock_cache):
        from llmpt.seeding_mapper import resolve_hf_blob

        result = resolve_hf_blob("test/repo", "bad.bin", "main")
        assert result is None


# ─── is_padding_file ──────────────────────────────────────────────────────────

class TestIsPaddingFile:

    def test_pad_prefix(self):
        from llmpt.seeding_mapper import is_padding_file
        assert is_padding_file(".pad/12345") is True

    def test_pad_in_path(self):
        from llmpt.seeding_mapper import is_padding_file
        assert is_padding_file("subdir/.pad/12345") is True

    def test_normal_file(self):
        from llmpt.seeding_mapper import is_padding_file
        assert is_padding_file("model.bin") is False

    def test_pad_like_name(self):
        from llmpt.seeding_mapper import is_padding_file
        assert is_padding_file("padding_file.txt") is False


# ─── create_padding_file ─────────────────────────────────────────────────────

class TestCreatePaddingFile:

    def test_creates_file(self, tmp_path):
        from llmpt.seeding_mapper import create_padding_file

        target = str(tmp_path / "sub" / "pad_0_128")
        create_padding_file(target, 128)
        assert os.path.exists(target)
        assert os.path.getsize(target) == 128
        with open(target, 'rb') as f:
            assert f.read() == b'\x00' * 128

    def test_skips_existing(self, tmp_path):
        from llmpt.seeding_mapper import create_padding_file

        target = str(tmp_path / "pad_0_64")
        # Pre-create with different content
        with open(target, 'wb') as f:
            f.write(b'\xff' * 64)

        create_padding_file(target, 64)
        # Should NOT overwrite
        with open(target, 'rb') as f:
            assert f.read() == b'\xff' * 64


# ─── cleanup_hardlinks ────────────────────────────────────────────────────────

class TestCleanupHardlinks:

    def test_removes_existing_files(self, tmp_path):
        from llmpt.seeding_mapper import cleanup_hardlinks

        f1 = tmp_path / "link1"
        f2 = tmp_path / "link2"
        f1.write_text("a")
        f2.write_text("b")

        cleanup_hardlinks("test/repo", [str(f1), str(f2)])
        assert not f1.exists()
        assert not f2.exists()

    def test_skips_nonexistent(self):
        from llmpt.seeding_mapper import cleanup_hardlinks
        # Should not raise
        cleanup_hardlinks("test/repo", ["/nonexistent/path/link1"])

    def test_handles_os_error(self, tmp_path):
        from llmpt.seeding_mapper import cleanup_hardlinks

        f = tmp_path / "link"
        f.write_text("data")
        with patch('os.unlink', side_effect=OSError("permission denied")):
            # Should not raise, just warn
            cleanup_hardlinks("test/repo", [str(f)])


# ─── hardlink_files_for_seeding ───────────────────────────────────────────────

class TestHardlinkFilesForSeeding:

    @patch('llmpt.seeding_mapper.resolve_hf_blob')
    def test_successful_hardlink(self, mock_resolve, tmp_path):
        from llmpt.seeding_mapper import hardlink_files_for_seeding

        # Create a "blob" file to hardlink to
        blob = tmp_path / "blob"
        blob.write_text("model data")
        mock_resolve.return_value = str(blob)

        mock_ti = MagicMock()
        mock_files = MagicMock()
        mock_files.num_files.return_value = 1
        mock_files.file_path.return_value = "root/model.bin"
        mock_files.file_size.return_value = 10
        mock_ti.files.return_value = mock_files

        temp_dir = str(tmp_path / "p2p_root")
        os.makedirs(temp_dir, exist_ok=True)

        hardlinks, count = hardlink_files_for_seeding(mock_ti, temp_dir, "t/r", "main")
        assert count == 1
        assert len(hardlinks) == 1
        assert os.path.exists(hardlinks[0])

    @patch('llmpt.seeding_mapper.resolve_hf_blob', return_value=None)
    def test_cache_miss_skips(self, mock_resolve, tmp_path):
        from llmpt.seeding_mapper import hardlink_files_for_seeding

        mock_ti = MagicMock()
        mock_files = MagicMock()
        mock_files.num_files.return_value = 1
        mock_files.file_path.return_value = "root/model.bin"
        mock_files.file_size.return_value = 10
        mock_ti.files.return_value = mock_files

        hardlinks, count = hardlink_files_for_seeding(mock_ti, str(tmp_path), "t/r", "main")
        assert count == 0
        assert len(hardlinks) == 0
