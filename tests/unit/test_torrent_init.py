"""
Tests for llmpt.torrent_init — helpers extracted from SessionContext._init_torrent().
"""

import os
import pytest
from unittest.mock import patch, MagicMock


# ─── acquire_torrent_data ─────────────────────────────────────────────────────

class TestAcquireTorrentData:

    @patch('llmpt.torrent_cache.resolve_torrent_data')
    def test_prefers_supplied_data(self, mock_resolve):
        from llmpt.torrent_init import acquire_torrent_data

        result = acquire_torrent_data("r/1", "abc", MagicMock(), b"supplied")
        assert result == b"supplied"
        mock_resolve.assert_not_called()

    @patch('llmpt.torrent_cache.resolve_torrent_data', return_value=b"cached")
    def test_falls_back_to_cache(self, mock_resolve):
        from llmpt.torrent_init import acquire_torrent_data

        tracker = MagicMock()
        result = acquire_torrent_data("r/1", "abc", tracker, None)
        assert result == b"cached"
        mock_resolve.assert_called_once_with("r/1", "abc", tracker)

    @patch('llmpt.torrent_cache.resolve_torrent_data', return_value=None)
    def test_returns_none_when_unavailable(self, mock_resolve):
        from llmpt.torrent_init import acquire_torrent_data

        result = acquire_torrent_data("r/1", "abc", MagicMock(), None)
        assert result is None


# ─── build_add_torrent_params ─────────────────────────────────────────────────

class TestBuildAddTorrentParams:

    @patch('llmpt.torrent_init.lt')
    def test_full_seed_mode(self, mock_lt):
        from llmpt.torrent_init import build_add_torrent_params

        mock_info = MagicMock()
        mock_lt.torrent_info.return_value = mock_info
        mock_lt.bdecode.return_value = {}
        mock_lt.torrent_flags.paused = 1
        mock_lt.torrent_flags.seed_mode = 2

        mock_params = MagicMock()
        mock_params.flags = 0
        mock_lt.add_torrent_params.return_value = mock_params

        params, info = build_add_torrent_params(
            torrent_data=b"data",
            save_path="/tmp/dl",
            session_mode="full_seed",
            fastresume_path="/tmp/fake.fastresume",
            repo_id="test/repo",
        )

        assert info is mock_info
        assert params.save_path == "/tmp/dl"
        # Should have both paused and seed_mode flags
        assert params.flags & 1  # paused
        assert params.flags & 2  # seed_mode

    @patch('llmpt.torrent_init.lt')
    def test_on_demand_no_fastresume(self, mock_lt):
        from llmpt.torrent_init import build_add_torrent_params

        mock_info = MagicMock()
        mock_lt.torrent_info.return_value = mock_info
        mock_lt.bdecode.return_value = {}
        mock_lt.torrent_flags.paused = 1
        mock_lt.torrent_flags.seed_mode = 2

        mock_params = MagicMock()
        mock_params.flags = 0
        mock_lt.add_torrent_params.return_value = mock_params

        params, info = build_add_torrent_params(
            torrent_data=b"data",
            save_path="/tmp/dl",
            session_mode="on_demand",
            fastresume_path="/tmp/nonexistent.fastresume",
            repo_id="test/repo",
        )

        assert info is mock_info
        # Should have paused but NOT seed_mode
        assert params.flags & 1  # paused
        assert not (params.flags & 2)  # no seed_mode


# ─── _load_fastresume / fastresume integration ───────────────────────────────

class TestFastresumeLoading:
    """Tests for the fastresume bug fix.

    The original bug: _load_fastresume() called lt.read_resume_data() which
    returns a NEW add_torrent_params object.  The function assigned this to
    the local variable `params`, but Python doesn't pass-by-reference, so
    the caller's variable was never updated — fastresume data was silently
    discarded.

    The fix: _load_fastresume() now RETURNS the params object, and the caller
    uses the return value.
    """

    @patch('llmpt.torrent_init.lt')
    def test_fastresume_uses_read_resume_data_object(self, mock_lt, tmp_path):
        """Core regression test: the returned params must be the NEW object
        from lt.read_resume_data(), not the original one."""
        from llmpt.torrent_init import build_add_torrent_params

        # Create a fake fastresume file
        fastresume_path = str(tmp_path / "test.fastresume")
        with open(fastresume_path, "wb") as f:
            f.write(b"fake_resume_data")

        # Set up mock: lt.read_resume_data returns a DIFFERENT object
        mock_info = MagicMock()
        mock_lt.torrent_info.return_value = mock_info
        mock_lt.bdecode.return_value = {}
        mock_lt.torrent_flags.paused = 1
        mock_lt.torrent_flags.seed_mode = 2

        original_params = MagicMock()
        original_params.flags = 0
        mock_lt.add_torrent_params.return_value = original_params

        # This is the key: read_resume_data returns a BRAND NEW object
        resumed_params = MagicMock()
        resumed_params.flags = 0
        resumed_params._is_resumed = True  # marker to identify it
        mock_lt.read_resume_data.return_value = resumed_params

        # Ensure parse_resume_data attr exists so the code path is taken
        mock_lt.add_torrent_params.parse_resume_data = True

        params, info = build_add_torrent_params(
            torrent_data=b"data",
            save_path="/tmp/dl",
            session_mode="on_demand",
            fastresume_path=fastresume_path,
            repo_id="test/repo",
        )

        # THE FIX: params must be the resumed object, NOT the original
        assert params is resumed_params, (
            "build_add_torrent_params must return the object from "
            "lt.read_resume_data(), not the original params"
        )
        assert params is not original_params
        # Verify save_path, ti, and flags were set on the resumed params
        assert params.save_path == "/tmp/dl"
        assert params.ti is mock_info
        assert params.flags & 1  # paused flag set

    @patch('llmpt.torrent_init.lt')
    def test_no_fastresume_file_returns_original_params(self, mock_lt):
        """When no fastresume file exists, the original params are returned."""
        from llmpt.torrent_init import build_add_torrent_params

        mock_info = MagicMock()
        mock_lt.torrent_info.return_value = mock_info
        mock_lt.bdecode.return_value = {}
        mock_lt.torrent_flags.paused = 1
        mock_lt.torrent_flags.seed_mode = 2

        original_params = MagicMock()
        original_params.flags = 0
        mock_lt.add_torrent_params.return_value = original_params

        params, info = build_add_torrent_params(
            torrent_data=b"data",
            save_path="/tmp/dl",
            session_mode="on_demand",
            fastresume_path="/tmp/does_not_exist.fastresume",
            repo_id="test/repo",
        )

        assert params is original_params
        mock_lt.read_resume_data.assert_not_called()

    @patch('llmpt.torrent_init.lt')
    def test_fastresume_corrupt_data_returns_original_params(self, mock_lt, tmp_path):
        """If fastresume data is corrupt, original params are returned gracefully."""
        from llmpt.torrent_init import build_add_torrent_params

        fastresume_path = str(tmp_path / "corrupt.fastresume")
        with open(fastresume_path, "wb") as f:
            f.write(b"not_valid_bencode")

        mock_info = MagicMock()
        mock_lt.torrent_info.return_value = mock_info
        mock_lt.bdecode.side_effect = [
            {},              # first call: for torrent_data in build_add_torrent_params
            Exception("corrupt"),  # second call: for resume_data in _load_fastresume
        ]
        mock_lt.torrent_flags.paused = 1
        mock_lt.torrent_flags.seed_mode = 2

        original_params = MagicMock()
        original_params.flags = 0
        mock_lt.add_torrent_params.return_value = original_params

        # No parse_resume_data → won't try lt.read_resume_data
        if hasattr(mock_lt.add_torrent_params, 'parse_resume_data'):
            del mock_lt.add_torrent_params.parse_resume_data

        params, info = build_add_torrent_params(
            torrent_data=b"data",
            save_path="/tmp/dl",
            session_mode="on_demand",
            fastresume_path=fastresume_path,
            repo_id="test/repo",
        )

        # Should still get a valid params object (original)
        assert params is original_params

    @patch('llmpt.torrent_init.lt')
    def test_fastresume_old_libtorrent_without_parse_resume_data(self, mock_lt, tmp_path):
        """Older libtorrent without parse_resume_data uses in-place mutation path."""
        from llmpt.torrent_init import build_add_torrent_params

        fastresume_path = str(tmp_path / "old_lt.fastresume")
        with open(fastresume_path, "wb") as f:
            f.write(b"fake_resume_data")

        mock_info = MagicMock()
        mock_lt.torrent_info.return_value = mock_info
        mock_lt.bdecode.return_value = {b'mapped_files': {0: b'/some/path'}}
        mock_lt.torrent_flags.paused = 1
        mock_lt.torrent_flags.seed_mode = 2

        original_params = MagicMock()
        original_params.flags = 0
        mock_lt.add_torrent_params.return_value = original_params

        # Simulate old libtorrent: no parse_resume_data attribute
        if hasattr(mock_lt.add_torrent_params, 'parse_resume_data'):
            del mock_lt.add_torrent_params.parse_resume_data

        params, info = build_add_torrent_params(
            torrent_data=b"data",
            save_path="/tmp/dl",
            session_mode="on_demand",
            fastresume_path=fastresume_path,
            repo_id="test/repo",
        )

        # Should get the original params (mutated in-place with renamed_files)
        assert params is original_params
        assert params.renamed_files == {0: b'/some/path'}
        mock_lt.read_resume_data.assert_not_called()


# ─── resolve_test_peer ────────────────────────────────────────────────────────

class TestResolveTestPeer:

    def test_returns_none_when_unset(self):
        from llmpt.torrent_init import resolve_test_peer

        with patch.dict(os.environ, {}, clear=True):
            assert resolve_test_peer() is None

    @patch('llmpt.torrent_init.socket')
    def test_host_port(self, mock_socket):
        from llmpt.torrent_init import resolve_test_peer

        mock_socket.gethostbyname.return_value = "1.2.3.4"
        with patch.dict(os.environ, {'TEST_SEEDER_PEER': 'myhost:7000'}):
            result = resolve_test_peer()
        assert result == ("1.2.3.4", 7000)

    @patch('llmpt.torrent_init.socket')
    def test_host_only_default_port(self, mock_socket):
        from llmpt.torrent_init import resolve_test_peer

        mock_socket.gethostbyname.return_value = "10.0.0.1"
        with patch.dict(os.environ, {'TEST_SEEDER_PEER': 'myhost'}):
            result = resolve_test_peer()
        assert result == ("10.0.0.1", 6881)

    @patch('llmpt.torrent_init.socket')
    def test_ipv6_bracket_notation(self, mock_socket):
        from llmpt.torrent_init import resolve_test_peer

        mock_socket.gethostbyname.return_value = "::1"
        with patch.dict(os.environ, {'TEST_SEEDER_PEER': '[::1]:9000'}):
            result = resolve_test_peer()
        assert result == ("::1", 9000)

    @patch('llmpt.torrent_init.socket')
    def test_dns_failure_returns_none(self, mock_socket):
        from llmpt.torrent_init import resolve_test_peer

        mock_socket.gethostbyname.side_effect = OSError("DNS fail")
        with patch.dict(os.environ, {'TEST_SEEDER_PEER': 'badhost:6881'}):
            result = resolve_test_peer()
        assert result is None
