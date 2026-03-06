"""
IPC (Inter-Process Communication) for the llmpt daemon.

Uses Unix Domain Sockets for communication between the download client
and the background seeding daemon.

Protocol:
    - Messages are newline-delimited JSON
    - Each message has an "action" field
    - Responses (if any) are also JSON

Example messages:
    {"action": "seed", "repo_id": "gpt2", "revision": "abc123..."}
    {"action": "status"}
    {"action": "scan"}
"""

import json
import logging
import os
import select
import socket
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("llmpt.ipc")

# Default socket path
SOCKET_DIR = os.path.expanduser("~/.cache/llmpt")
SOCKET_PATH = os.path.join(SOCKET_DIR, "daemon.sock")


class IPCServer:
    """Unix Domain Socket server for the daemon.

    Runs a non-blocking accept loop in a background thread, dispatching
    incoming messages to a user-supplied callback.
    """

    def __init__(
        self,
        socket_path: str = SOCKET_PATH,
        handler: Optional[Callable[[dict], Optional[dict]]] = None,
    ):
        self.socket_path = socket_path
        self.handler = handler or (lambda msg: None)
        self._server_socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the IPC server in a background thread."""
        os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)

        # Clean up stale socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self._server_socket = socket.socket(
            socket.AF_UNIX, socket.SOCK_STREAM
        )
        self._server_socket.bind(self.socket_path)
        self._server_socket.listen(5)
        self._server_socket.setblocking(False)
        self._running = True

        self._thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="ipc-server"
        )
        self._thread.start()
        logger.info(f"IPC server started on {self.socket_path}")

    def stop(self) -> None:
        """Stop the IPC server."""
        self._running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        # Clean up socket file
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except OSError:
            pass
        logger.info("IPC server stopped")

    def _accept_loop(self) -> None:
        """Background thread: accept and handle connections."""
        while self._running:
            try:
                # Use select() to avoid blocking indefinitely
                readable, _, _ = select.select(
                    [self._server_socket], [], [], 1.0
                )
                if not readable:
                    continue

                conn, _ = self._server_socket.accept()
                conn.settimeout(5)
                try:
                    self._handle_connection(conn)
                except Exception as e:
                    logger.debug(f"IPC connection error: {e}")
                finally:
                    conn.close()

            except OSError:
                # Socket closed during shutdown
                break
            except Exception as e:
                logger.debug(f"IPC accept error: {e}")

    def _handle_connection(self, conn: socket.socket) -> None:
        """Handle a single IPC connection."""
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        if not data:
            return

        try:
            msg = json.loads(data.strip())
        except json.JSONDecodeError:
            logger.debug(f"IPC: invalid JSON: {data[:100]}")
            return

        logger.debug(f"IPC received: {msg.get('action', '?')}")

        # Dispatch to handler
        response = self.handler(msg)

        # Send response if handler returned one
        if response is not None:
            try:
                conn.sendall(json.dumps(response).encode() + b"\n")
            except (OSError, BrokenPipeError):
                pass


# ---------------------------------------------------------------------------
# Client functions (used by download client / CLI)
# ---------------------------------------------------------------------------


def notify_daemon(action: str, **kwargs) -> bool:
    """Send a fire-and-forget message to the daemon.

    Returns True if the message was sent successfully, False otherwise.
    Does NOT raise exceptions — safe to call even when the daemon is not
    running.
    """
    msg: Dict[str, Any] = {"action": action, **kwargs}
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(SOCKET_PATH)
        sock.sendall(json.dumps(msg).encode() + b"\n")
        sock.close()
        logger.debug(f"IPC notify sent: {action}")
        return True
    except Exception as e:
        logger.debug(f"IPC notify failed (daemon may not be running): {e}")
        return False


def query_daemon(action: str, **kwargs) -> Optional[dict]:
    """Send a message to the daemon and wait for a response.

    Returns the response dict, or None if communication failed.
    """
    msg: Dict[str, Any] = {"action": action, **kwargs}
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(SOCKET_PATH)
        sock.sendall(json.dumps(msg).encode() + b"\n")

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

        if data:
            return json.loads(data.strip())
        return None

    except Exception as e:
        logger.debug(f"IPC query failed: {e}")
        return None
