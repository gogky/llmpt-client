"""
Tests for llmpt.torrent_cache module.

Covers: load_cached_torrent, save_torrent_to_cache, resolve_torrent_data.
"""

import os
import pytest
from unittest.mock import patch, MagicMock


# ─── load_cached_torrent ─────────────────────────────────────────────────────

class TestLoadCachedTorrent:

    def test_returns_none_when_no_cache(self, tmp_path):
        """No cached file → returns None."""
        from llmpt.torrent_cache import load_cached_torrent

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(tmp_path)):
            result = load_cached_torrent("test/repo", "abc123")

        assert result is None

    def test_returns_bytes_on_cache_hit(self, tmp_path):
        """Cached file exists → returns its bytes."""
        from llmpt.torrent_cache import load_cached_torrent

        # Create a fake cached torrent file
        cache_file = tmp_path / "model_test_repo_abc123.torrent"
        cache_file.write_bytes(b"torrent_data_bytes")

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(tmp_path)):
            result = load_cached_torrent("test/repo", "abc123")

        assert result == b"torrent_data_bytes"

    def test_empty_file_treated_as_miss(self, tmp_path):
        """An empty cached file is treated as a cache miss and cleaned up."""
        from llmpt.torrent_cache import load_cached_torrent

        cache_file = tmp_path / "model_test_repo_abc123.torrent"
        cache_file.write_bytes(b"")

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(tmp_path)):
            result = load_cached_torrent("test/repo", "abc123")

        assert result is None
        assert not cache_file.exists()  # empty file should be cleaned up

    def test_read_error_returns_none(self, tmp_path):
        """If reading the file fails, returns None gracefully."""
        from llmpt.torrent_cache import load_cached_torrent

        cache_file = tmp_path / "model_test_repo_abc123.torrent"
        cache_file.write_bytes(b"some_data")

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(tmp_path)), \
             patch('builtins.open', side_effect=OSError("permission denied")):
            result = load_cached_torrent("test/repo", "abc123")

        assert result is None


# ─── save_torrent_to_cache ────────────────────────────────────────────────────

class TestSaveTorrentToCache:

    def test_creates_cache_file(self, tmp_path):
        """Should write torrent data to the correct path."""
        from llmpt.torrent_cache import save_torrent_to_cache

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(tmp_path)):
            save_torrent_to_cache("test/repo", "abc123", b"torrent_bytes")

        cache_file = tmp_path / "model_test_repo_abc123.torrent"
        assert cache_file.exists()
        assert cache_file.read_bytes() == b"torrent_bytes"

    def test_creates_directory_if_missing(self, tmp_path):
        """Should create the cache directory if it doesn't exist."""
        from llmpt.torrent_cache import save_torrent_to_cache

        cache_dir = tmp_path / "nonexistent" / "dir"

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(cache_dir)):
            save_torrent_to_cache("test/repo", "abc123", b"data")

        assert (cache_dir / "model_test_repo_abc123.torrent").exists()

    def test_atomic_write_no_tmp_file_remains(self, tmp_path):
        """After save, only the final file should exist (no .tmp remnant)."""
        from llmpt.torrent_cache import save_torrent_to_cache

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(tmp_path)):
            save_torrent_to_cache("test/repo", "abc123", b"data")

        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "model_test_repo_abc123.torrent"
        assert not any(f.name.endswith('.tmp') for f in files)

    def test_overwrites_existing_cache(self, tmp_path):
        """Should overwrite an existing cached file with new data."""
        from llmpt.torrent_cache import save_torrent_to_cache

        cache_file = tmp_path / "model_test_repo_abc123.torrent"
        cache_file.write_bytes(b"old_data")

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(tmp_path)):
            save_torrent_to_cache("test/repo", "abc123", b"new_data")

        assert cache_file.read_bytes() == b"new_data"

    def test_write_error_handled_gracefully(self, tmp_path):
        """If write fails, should not raise and should clean up tmp file."""
        from llmpt.torrent_cache import save_torrent_to_cache

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(tmp_path)), \
             patch('builtins.open', side_effect=OSError("disk full")):
            # Should not raise
            save_torrent_to_cache("test/repo", "abc123", b"data")


# ─── resolve_torrent_data ─────────────────────────────────────────────────────

class TestResolveTorrentData:

    def test_layer1_cache_hit(self, tmp_path):
        """Layer 1: local cache hit → returns without contacting tracker."""
        from llmpt.torrent_cache import resolve_torrent_data

        cache_file = tmp_path / "model_test_repo_abc123.torrent"
        cache_file.write_bytes(b"cached_torrent")

        tracker = MagicMock()

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(tmp_path)):
            result = resolve_torrent_data("test/repo", "abc123", tracker)

        assert result == b"cached_torrent"
        tracker.download_torrent.assert_not_called()

    def test_layer2_tracker_hit(self, tmp_path):
        """Layer 2: cache miss → tracker returns data → data is cached locally."""
        from llmpt.torrent_cache import resolve_torrent_data

        tracker = MagicMock()
        tracker.download_torrent.return_value = b"tracker_torrent"

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(tmp_path)):
            result = resolve_torrent_data("test/repo", "abc123", tracker)

        assert result == b"tracker_torrent"
        tracker.download_torrent.assert_called_once_with("test/repo", "abc123", repo_type="model")
        # Verify it was cached locally
        cache_file = tmp_path / "model_test_repo_abc123.torrent"
        assert cache_file.exists()
        assert cache_file.read_bytes() == b"tracker_torrent"

    def test_layer3_nothing_found(self, tmp_path):
        """Layer 3: both cache and tracker miss → returns None."""
        from llmpt.torrent_cache import resolve_torrent_data

        tracker = MagicMock()
        tracker.download_torrent.return_value = None

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(tmp_path)):
            result = resolve_torrent_data("test/repo", "abc123", tracker)

        assert result is None
        tracker.download_torrent.assert_called_once_with("test/repo", "abc123", repo_type="model")

    def test_repo_id_with_slashes_in_filename(self, tmp_path):
        """Repo IDs with slashes should be sanitized in filenames."""
        from llmpt.torrent_cache import save_torrent_to_cache, load_cached_torrent

        with patch('llmpt.torrent_cache.TORRENT_CACHE_DIR', str(tmp_path)):
            save_torrent_to_cache("org/model-name", "abc123", b"data")
            result = load_cached_torrent("org/model-name", "abc123")

        assert result == b"data"
        # Filename should use underscore instead of slash
        cache_file = tmp_path / "model_org_model-name_abc123.torrent"
        assert cache_file.exists()
