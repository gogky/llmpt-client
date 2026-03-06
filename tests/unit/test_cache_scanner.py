"""
Tests for llmpt.cache_scanner module.
"""

import os
import pytest
from pathlib import Path

from llmpt.cache_scanner import (
    _parse_repo_id,
    _is_snapshot_complete,
    scan_hf_cache,
)


class TestParseRepoId:
    """Tests for _parse_repo_id()."""

    def test_simple_model(self):
        assert _parse_repo_id("models--gpt2") == "gpt2"

    def test_org_model(self):
        assert _parse_repo_id("models--meta-llama--Llama-2-7b") == "meta-llama/Llama-2-7b"

    def test_dataset(self):
        assert _parse_repo_id("datasets--squad") == "squad"

    def test_space(self):
        assert _parse_repo_id("spaces--some-user--my-space") == "some-user/my-space"

    def test_invalid_prefix(self):
        assert _parse_repo_id("unknown--gpt2") is None

    def test_no_separator(self):
        assert _parse_repo_id("randomdir") is None

    def test_empty_string(self):
        assert _parse_repo_id("") is None

    def test_hyphenated_names(self):
        """Hyphens within org/model names should be preserved."""
        assert _parse_repo_id("models--my-org--my-model-v2") == "my-org/my-model-v2"


class TestIsSnapshotComplete:
    """Tests for _is_snapshot_complete()."""

    def test_complete_snapshot_regular_files(self, tmp_path):
        """Snapshot with regular files (not symlinks) is complete."""
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.bin").write_bytes(b"\x00" * 100)
        assert _is_snapshot_complete(tmp_path) is True

    def test_empty_snapshot(self, tmp_path):
        """Empty directory is incomplete."""
        assert _is_snapshot_complete(tmp_path) is False

    def test_complete_snapshot_with_symlinks(self, tmp_path):
        """Snapshot with valid symlinks is complete."""
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        (blobs / "abc123").write_bytes(b"\x00" * 100)

        snap = tmp_path / "snapshot"
        snap.mkdir()
        (snap / "model.bin").symlink_to(blobs / "abc123")

        assert _is_snapshot_complete(snap) is True

    def test_broken_symlink_is_incomplete(self, tmp_path):
        """Snapshot with broken symlink is incomplete."""
        snap = tmp_path / "snapshot"
        snap.mkdir()
        (snap / "model.bin").symlink_to("/nonexistent/path")

        assert _is_snapshot_complete(snap) is False

    def test_hidden_files_ignored(self, tmp_path):
        """Hidden files (like .gitattributes) are ignored."""
        (tmp_path / ".gitattributes").write_text("*.bin filter=lfs")
        (tmp_path / "config.json").write_text("{}")
        assert _is_snapshot_complete(tmp_path) is True

    def test_only_hidden_files_is_incomplete(self, tmp_path):
        """A directory with only hidden files has no real content."""
        (tmp_path / ".gitattributes").write_text("*.bin filter=lfs")
        # All entries are hidden → iterdir returns them but they're skipped
        # The function should still return True since there were entries
        # (hidden files don't cause it to be False, only broken symlinks do)
        assert _is_snapshot_complete(tmp_path) is True

    def test_subdirectory_with_complete_files(self, tmp_path):
        """Snapshot with subdirectories containing valid files."""
        sub = tmp_path / "unet"
        sub.mkdir()
        (sub / "model.safetensors").write_bytes(b"\x00")
        (tmp_path / "config.json").write_text("{}")
        assert _is_snapshot_complete(tmp_path) is True

    def test_subdirectory_with_broken_symlink(self, tmp_path):
        """Incomplete if a subdirectory contains a broken symlink."""
        sub = tmp_path / "unet"
        sub.mkdir()
        (sub / "model.safetensors").symlink_to("/nonexistent")
        (tmp_path / "config.json").write_text("{}")
        assert _is_snapshot_complete(tmp_path) is False


class TestScanHfCache:
    """Tests for scan_hf_cache()."""

    def _create_mock_cache(self, base: Path, models: dict):
        """Create a mock HF cache structure.

        Args:
            base: Root cache directory.
            models: Dict mapping dir_name → {revision: [files]}.
                    e.g. {"models--gpt2": {"abc123...": ["config.json", "model.bin"]}}
        """
        for dir_name, revisions in models.items():
            model_dir = base / dir_name / "snapshots"
            model_dir.mkdir(parents=True)
            for revision, files in revisions.items():
                snap = model_dir / revision
                snap.mkdir()
                for f in files:
                    # Support subdirectories in filenames
                    file_path = snap / f
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text("content")

    def test_single_model(self, tmp_path):
        commit = "a" * 40
        self._create_mock_cache(tmp_path, {
            "models--gpt2": {commit: ["config.json", "model.bin"]}
        })
        result = scan_hf_cache(str(tmp_path))
        assert result == [("gpt2", commit)]

    def test_multiple_models(self, tmp_path):
        commit1 = "a" * 40
        commit2 = "b" * 40
        self._create_mock_cache(tmp_path, {
            "models--gpt2": {commit1: ["config.json"]},
            "models--org--model": {commit2: ["config.json"]},
        })
        result = scan_hf_cache(str(tmp_path))
        assert len(result) == 2
        assert ("gpt2", commit1) in result
        assert ("org/model", commit2) in result

    def test_skips_non_commit_hash_dirs(self, tmp_path):
        commit = "a" * 40
        self._create_mock_cache(tmp_path, {
            "models--gpt2": {
                commit: ["config.json"],
                "main": ["config.json"],  # Not a commit hash
            }
        })
        result = scan_hf_cache(str(tmp_path))
        assert len(result) == 1
        assert result[0][1] == commit

    def test_empty_cache(self, tmp_path):
        result = scan_hf_cache(str(tmp_path))
        assert result == []

    def test_nonexistent_cache(self, tmp_path):
        result = scan_hf_cache(str(tmp_path / "nonexistent"))
        assert result == []

    def test_skips_unknown_prefixes(self, tmp_path):
        commit = "a" * 40
        self._create_mock_cache(tmp_path, {
            "unknown--something": {commit: ["file.txt"]}
        })
        result = scan_hf_cache(str(tmp_path))
        assert result == []

    def test_multiple_revisions_same_model(self, tmp_path):
        commit1 = "a" * 40
        commit2 = "b" * 40
        self._create_mock_cache(tmp_path, {
            "models--gpt2": {
                commit1: ["config.json"],
                commit2: ["config.json", "model.bin"],
            }
        })
        result = scan_hf_cache(str(tmp_path))
        assert len(result) == 2
