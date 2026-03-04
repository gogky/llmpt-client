"""
Tests for utility functions in llmpt.utils.
"""

import os
import hashlib
import tempfile
import pytest

from llmpt.utils import calculate_file_hash, format_bytes, get_optimal_piece_length


# ─── calculate_file_hash ─────────────────────────────────────────────────────

class TestCalculateFileHash:

    def test_sha256_default(self, tmp_path):
        """Default algorithm should be sha256."""
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert calculate_file_hash(str(f)) == expected

    def test_sha1(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        expected = hashlib.sha1(b"hello world").hexdigest()
        assert calculate_file_hash(str(f), algorithm='sha1') == expected

    def test_md5(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        expected = hashlib.md5(b"hello world").hexdigest()
        assert calculate_file_hash(str(f), algorithm='md5') == expected

    def test_empty_file(self, tmp_path):
        """Hash of empty file should match hash of empty bytes."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert calculate_file_hash(str(f)) == expected

    def test_large_file_chunked_read(self, tmp_path):
        """File larger than 8192 bytes should be read in chunks correctly."""
        data = os.urandom(20000)  # ~20KB, exceeds 8192-byte chunk size
        f = tmp_path / "large.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert calculate_file_hash(str(f)) == expected


# ─── format_bytes ─────────────────────────────────────────────────────────────

class TestFormatBytes:

    def test_bytes(self):
        assert format_bytes(0) == "0.0 B"
        assert format_bytes(512) == "512.0 B"
        assert format_bytes(1023) == "1023.0 B"

    def test_kilobytes(self):
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1536) == "1.5 KB"

    def test_megabytes(self):
        assert format_bytes(1024 * 1024) == "1.0 MB"

    def test_gigabytes(self):
        assert format_bytes(1024 ** 3) == "1.0 GB"

    def test_terabytes(self):
        assert format_bytes(1024 ** 4) == "1.0 TB"

    def test_petabytes(self):
        assert format_bytes(1024 ** 5) == "1.0 PB"

    def test_very_large(self):
        """Values beyond PB should still return PB units."""
        result = format_bytes(2 * 1024 ** 5)
        assert "PB" in result


# ─── get_optimal_piece_length ─────────────────────────────────────────────────

class TestGetOptimalPieceLength:

    def test_small_file(self):
        """< 100MB → 256KB."""
        assert get_optimal_piece_length(50 * 1024 * 1024) == 256 * 1024
        assert get_optimal_piece_length(0) == 256 * 1024

    def test_small_file_boundary(self):
        """Exactly 100MB should go to the next tier (1MB)."""
        assert get_optimal_piece_length(100 * 1024 * 1024) == 1024 * 1024

    def test_medium_file(self):
        """100MB–1GB → 1MB."""
        assert get_optimal_piece_length(500 * 1024 * 1024) == 1024 * 1024

    def test_medium_file_boundary(self):
        """Exactly 1GB should go to 4MB tier."""
        assert get_optimal_piece_length(1024 * 1024 * 1024) == 4 * 1024 * 1024

    def test_large_file(self):
        """1GB–10GB → 4MB."""
        assert get_optimal_piece_length(5 * 1024 ** 3) == 4 * 1024 * 1024

    def test_very_large_file(self):
        """10GB–100GB → 16MB."""
        assert get_optimal_piece_length(20 * 1024 ** 3) == 16 * 1024 * 1024
        assert get_optimal_piece_length(10 * 1024 ** 3) == 16 * 1024 * 1024

    def test_huge_file(self):
        """100GB–1TB → 32MB (e.g. Llama-3.1-405B ~800GB)."""
        assert get_optimal_piece_length(100 * 1024 ** 3) == 32 * 1024 * 1024
        assert get_optimal_piece_length(500 * 1024 ** 3) == 32 * 1024 * 1024

    def test_massive_file(self):
        """≥1TB → 64MB (e.g. massive MoE models)."""
        assert get_optimal_piece_length(1024 * 1024 ** 3) == 64 * 1024 * 1024
        assert get_optimal_piece_length(2048 * 1024 ** 3) == 64 * 1024 * 1024
