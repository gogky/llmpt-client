"""
Tests for _resolve_listen_interfaces in p2p_batch module.

Covers: explicit port, default port availability, fallback scanning, and
full-range-occupied OS-assigned fallback.

Uses ``mock_lt_all_modules`` from conftest (autouse via wrapper).
"""

import socket
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def patch_lt(mock_lt_all_modules):
    """Auto-use the shared multi-module libtorrent mock."""
    yield mock_lt_all_modules


class TestResolveListenInterfaces:

    def test_explicit_port(self):
        """When a positive port is configured, use it directly without probing."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        result = _resolve_listen_interfaces(7000)
        assert result == '0.0.0.0:7000,[::]:7000'

    def test_explicit_port_zero_triggers_fallback(self):
        """port=0 should behave like None (auto-select from default range)."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        with patch('socket.socket') as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value.__enter__ = MagicMock(return_value=mock_sock)
            mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)
            # bind succeeds on first try
            mock_sock.bind = MagicMock()

            result = _resolve_listen_interfaces(0)
            assert '6881' in result

    def test_none_triggers_fallback(self):
        """port=None should try default port range."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        with patch('socket.socket') as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value.__enter__ = MagicMock(return_value=mock_sock)
            mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_sock.bind = MagicMock()

            result = _resolve_listen_interfaces(None)
            assert '6881' in result

    def test_default_port_occupied_tries_next(self):
        """If 6881 is occupied, should try 6882."""
        from llmpt.p2p_batch import _resolve_listen_interfaces

        call_count = 0

        def fake_bind(addr):
            nonlocal call_count
            call_count += 1
            if addr[1] == 6881:
                raise OSError("Address already in use")
            # 6882 succeeds

        with patch('socket.socket') as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value.__enter__ = MagicMock(return_value=mock_sock)
            mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_sock.bind = fake_bind

            result = _resolve_listen_interfaces(None)
            assert '6882' in result
            assert call_count == 2

    def test_all_ports_occupied_falls_back_to_os(self):
        """If all ports 6881-6999 are occupied, should return port 0."""
        from llmpt.p2p_batch import _resolve_listen_interfaces

        def always_fail(addr):
            raise OSError("Address already in use")

        with patch('socket.socket') as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value.__enter__ = MagicMock(return_value=mock_sock)
            mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_sock.bind = always_fail

            result = _resolve_listen_interfaces(None)
            assert result == '0.0.0.0:0,[::]:0'
