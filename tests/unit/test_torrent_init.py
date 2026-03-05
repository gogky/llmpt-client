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
