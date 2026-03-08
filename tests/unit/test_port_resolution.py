"""
Tests for _resolve_listen_interfaces and _is_port_available in p2p_batch module.

Covers:
  - Role-based port assignment (daemon=N, client=N+1)
  - Default port assignment (daemon=6881, client=6882)
  - IPv4 + IPv6 dual-stack port probing
  - Graceful IPv6 skip on systems without IPv6
  - Fallback scanning when ports are occupied
  - Full-range-occupied OS-assigned fallback

Uses ``mock_lt_all_modules`` from conftest (autouse via wrapper).
"""

import socket
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def patch_lt(mock_lt_all_modules):
    """Auto-use the shared multi-module libtorrent mock."""
    yield mock_lt_all_modules


# ── helpers ────────────────────────────────────────────────────────────────

def _make_bind_tracker():
    """Return a (fake_bind, calls) pair that records (family, addr) per call.

    By default every bind succeeds.  Override ``fake_bind.fail_ports``
    (a set of int) to make specific ports raise ``OSError``.
    """
    calls = []

    def fake_bind(addr):
        calls.append(addr)
        port = addr[1]
        if port in getattr(fake_bind, 'fail_ports', set()):
            raise OSError("Address already in use")

    fake_bind.fail_ports = set()
    return fake_bind, calls


def _patch_socket(fake_bind, *, ipv6_available=True):
    """Context manager that patches ``socket.socket`` for port probing.

    Args:
        fake_bind: callable used as ``sock.bind``.
        ipv6_available: if False, creating an AF_INET6 socket raises OSError.
    """
    real_af_inet = socket.AF_INET
    real_af_inet6 = socket.AF_INET6

    def socket_factory(family, sock_type):
        if family == real_af_inet6 and not ipv6_available:
            raise OSError("Address family not supported")
        mock_sock = MagicMock()
        mock_sock.setsockopt = MagicMock()
        mock_sock.bind = fake_bind
        mock_sock.close = MagicMock()
        # Context-manager support
        mock_sock.__enter__ = MagicMock(return_value=mock_sock)
        mock_sock.__exit__ = MagicMock(return_value=False)
        return mock_sock

    return patch('socket.socket', side_effect=socket_factory)


# ── Role-based port assignment ─────────────────────────────────────────────

class TestRoleBasedPortAssignment:

    def test_explicit_port_daemon_role(self):
        """Daemon with explicit port N should use N."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        fake_bind, _ = _make_bind_tracker()
        with _patch_socket(fake_bind):
            result = _resolve_listen_interfaces(7000, role='daemon')
        assert result == '0.0.0.0:7000,[::]:7000'

    def test_explicit_port_client_role(self):
        """Client with explicit port N should use N+1."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        fake_bind, _ = _make_bind_tracker()
        with _patch_socket(fake_bind):
            result = _resolve_listen_interfaces(7000, role='client')
        assert result == '0.0.0.0:7001,[::]:7001'

    def test_default_port_daemon_role(self):
        """Daemon with no port (None) should default to 6881."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        fake_bind, _ = _make_bind_tracker()
        with _patch_socket(fake_bind):
            result = _resolve_listen_interfaces(None, role='daemon')
        assert '6881' in result

    def test_default_port_client_role(self):
        """Client with no port (None) should default to 6882."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        fake_bind, _ = _make_bind_tracker()
        with _patch_socket(fake_bind):
            result = _resolve_listen_interfaces(None, role='client')
        assert '6882' in result

    def test_zero_port_daemon_role(self):
        """port=0 for daemon should behave like None (start at 6881)."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        fake_bind, _ = _make_bind_tracker()
        with _patch_socket(fake_bind):
            result = _resolve_listen_interfaces(0, role='daemon')
        assert '6881' in result

    def test_zero_port_client_role(self):
        """port=0 for client should behave like None (start at 6882)."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        fake_bind, _ = _make_bind_tracker()
        with _patch_socket(fake_bind):
            result = _resolve_listen_interfaces(0, role='client')
        assert '6882' in result

    def test_default_role_is_daemon(self):
        """When role is omitted it should default to 'daemon'."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        fake_bind, _ = _make_bind_tracker()
        with _patch_socket(fake_bind):
            result = _resolve_listen_interfaces(None)
        assert '6881' in result


# ── IPv6 port probing ──────────────────────────────────────────────────────

class TestIPv6PortProbing:

    def test_ipv4_free_ipv6_occupied(self):
        """If IPv4 is free but IPv6 is occupied, port should be skipped."""
        from llmpt.p2p_batch import _resolve_listen_interfaces

        def selective_bind(addr):
            host, port = addr
            # IPv4 on 6881 succeeds, IPv6 on 6881 fails
            if host == '::' and port == 6881:
                raise OSError("Address already in use")
            # Everything else succeeds

        with _patch_socket(selective_bind):
            result = _resolve_listen_interfaces(None, role='daemon')
        # Should skip 6881 and land on 6882
        assert '6882' in result

    def test_ipv6_not_supported_skips_check(self):
        """On IPv4-only systems, IPv6 check is skipped gracefully."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        fake_bind, _ = _make_bind_tracker()
        with _patch_socket(fake_bind, ipv6_available=False):
            result = _resolve_listen_interfaces(None, role='daemon')
        # Should succeed on port 6881 (IPv4-only check)
        assert '6881' in result


# ── Fallback behaviour ────────────────────────────────────────────────────

class TestFallbackBehaviour:

    def test_first_port_occupied_tries_next(self):
        """If the target port is occupied, probe the next port."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        fake_bind, _ = _make_bind_tracker()
        fake_bind.fail_ports = {6881}
        with _patch_socket(fake_bind):
            result = _resolve_listen_interfaces(None, role='daemon')
        assert '6882' in result

    def test_all_ports_occupied_falls_back_to_os(self):
        """If all ports 6881-6999 are occupied, return port 0."""
        from llmpt.p2p_batch import _resolve_listen_interfaces

        def always_fail(addr):
            raise OSError("Address already in use")

        with _patch_socket(always_fail):
            result = _resolve_listen_interfaces(None, role='daemon')
        assert result == '0.0.0.0:0,[::]:0'

    def test_explicit_port_occupied_probes_upward(self):
        """If explicit port N is occupied, probe N+1, N+2, etc."""
        from llmpt.p2p_batch import _resolve_listen_interfaces
        fake_bind, _ = _make_bind_tracker()
        fake_bind.fail_ports = {7000}
        with _patch_socket(fake_bind):
            result = _resolve_listen_interfaces(7000, role='daemon')
        assert '7001' in result
