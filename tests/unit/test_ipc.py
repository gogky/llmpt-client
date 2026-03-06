"""
Tests for llmpt.ipc module.
"""

import json
import os
import socket
import threading
import time

import pytest

from llmpt.ipc import IPCServer, notify_daemon, query_daemon, SOCKET_DIR


@pytest.fixture
def socket_path(tmp_path):
    """Return a temp socket path for testing."""
    return str(tmp_path / "test_daemon.sock")


@pytest.fixture
def ipc_server(socket_path):
    """Create and start an IPC server, clean up after test."""
    received_messages = []

    def handler(msg):
        received_messages.append(msg)
        action = msg.get("action")
        if action == "ping":
            return {"status": "ok", "pid": os.getpid()}
        elif action == "status":
            return {"status": "ok", "seeding_count": 3}
        return {"status": "ok"}

    server = IPCServer(socket_path=socket_path, handler=handler)
    server.start()
    time.sleep(0.1)  # Allow server thread to start
    yield server, received_messages
    server.stop()


class TestIPCServer:
    """Tests for IPCServer."""

    def test_start_and_stop(self, socket_path):
        """Server starts and creates socket file, stops and removes it."""
        server = IPCServer(socket_path=socket_path)
        server.start()
        assert os.path.exists(socket_path)
        server.stop()
        assert not os.path.exists(socket_path)

    def test_receives_message(self, ipc_server, socket_path):
        """Server receives and processes messages."""
        server, received = ipc_server

        # Send a message directly
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(socket_path)
        msg = json.dumps({"action": "seed", "repo_id": "gpt2"}) + "\n"
        sock.sendall(msg.encode())
        sock.close()

        time.sleep(0.2)

        assert len(received) == 1
        assert received[0]["action"] == "seed"
        assert received[0]["repo_id"] == "gpt2"

    def test_receives_response(self, ipc_server, socket_path):
        """Server sends back response."""
        server, _ = ipc_server

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(socket_path)
        msg = json.dumps({"action": "ping"}) + "\n"
        sock.sendall(msg.encode())

        # Read response
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        sock.close()

        response = json.loads(data.strip())
        assert response["status"] == "ok"
        assert response["pid"] == os.getpid()

    def test_cleans_up_stale_socket(self, socket_path):
        """Server removes stale socket file on start."""
        # Create a stale socket file
        os.makedirs(os.path.dirname(socket_path), exist_ok=True)
        with open(socket_path, "w") as f:
            f.write("stale")

        server = IPCServer(socket_path=socket_path)
        server.start()
        # Should have replaced the stale file with a real socket
        assert os.path.exists(socket_path)
        server.stop()


class TestClientFunctions:
    """Tests for notify_daemon and query_daemon client functions."""

    def test_notify_daemon_success(self, ipc_server, socket_path, monkeypatch):
        """notify_daemon returns True when daemon is running."""
        server, received = ipc_server
        monkeypatch.setattr("llmpt.ipc.SOCKET_PATH", socket_path)

        result = notify_daemon("seed", repo_id="gpt2", revision="abc123")
        time.sleep(0.2)

        assert result is True
        assert len(received) == 1
        assert received[0]["action"] == "seed"

    def test_notify_daemon_no_daemon(self, tmp_path, monkeypatch):
        """notify_daemon returns False when daemon is not running."""
        monkeypatch.setattr("llmpt.ipc.SOCKET_PATH", str(tmp_path / "nonexistent.sock"))

        result = notify_daemon("seed", repo_id="gpt2")
        assert result is False

    def test_query_daemon_success(self, ipc_server, socket_path, monkeypatch):
        """query_daemon returns response dict."""
        server, _ = ipc_server
        monkeypatch.setattr("llmpt.ipc.SOCKET_PATH", socket_path)

        result = query_daemon("status")
        assert result is not None
        assert result["status"] == "ok"
        assert result["seeding_count"] == 3

    def test_query_daemon_no_daemon(self, tmp_path, monkeypatch):
        """query_daemon returns None when daemon is not running."""
        monkeypatch.setattr("llmpt.ipc.SOCKET_PATH", str(tmp_path / "nonexistent.sock"))

        result = query_daemon("status")
        assert result is None
