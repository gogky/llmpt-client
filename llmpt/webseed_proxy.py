"""
WebSeed reverse proxy server.

Starts a lightweight local HTTP server that translates libtorrent's
WebSeed requests into HuggingFace CDN requests, handling URL mapping
and authentication transparently.

Architecture overview:

    libtorrent ──GET /{commit_hash}/{file}──▶  WebSeedProxy (127.0.0.1)
                                                    │
                                                    ▼
                                       HuggingFace CDN (huggingface.co)
                                       GET /resolve/{commit_hash}/{file}
                                       + Authorization header

See docs/webseed_design.md.resolved for full design rationale.
"""

import os
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from urllib.parse import quote

import requests as _requests

logger = logging.getLogger(__name__)

# Chunk size for streaming proxy responses (64 KB).
_STREAM_CHUNK_SIZE = 65536

# Headers to forward from upstream to the client (lowercase).
_FORWARDED_HEADERS = frozenset({
    'content-length',
    'content-range',
    'content-type',
    'accept-ranges',
    'etag',
    'last-modified',
})


class WebSeedProxyHandler(BaseHTTPRequestHandler):
    """Handle libtorrent WebSeed HTTP requests.

    Expected request path format::

        /ws/{org}/{repo}/{commit_hash}/{file_path...}

    The handler:
    1. Parses the path to extract *repo_id*, *commit_hash*, and *file_path*.
    2. Constructs the upstream HuggingFace URL:
       ``https://huggingface.co/{repo_id}/resolve/{commit_hash}/{file_path}``
    3. Injects ``Authorization: Bearer <token>`` if a token is configured.
    4. Forwards the ``Range`` header from libtorrent (if present).
    5. Streams the upstream response back to libtorrent.
    """

    # Suppress per-request log lines from BaseHTTPRequestHandler.
    def log_message(self, format, *args):  # noqa: A002
        logger.debug("WebSeedProxy: %s", format % args)

    # ── Request handling ──────────────────────────────────────────────────

    def do_GET(self):  # noqa: N802
        """Handle GET requests from libtorrent."""
        try:
            self._handle_get()
        except Exception:
            logger.exception("WebSeedProxy: unexpected error handling %s", self.path)
            self.send_error(500, "Internal proxy error")

    def _handle_get(self):
        # ── 1. Parse the path ─────────────────────────────────────────────
        parts = self._parse_path()
        if parts is None:
            self.send_error(400, "Bad path format")
            return

        repo_type, repo_id, commit_hash, file_path = parts

        # ── 2. Build upstream URL ─────────────────────────────────────────
        hf_endpoint = self.server.hf_endpoint
        # URL-encode the file_path segments but keep '/' as-is.
        encoded_file_path = "/".join(quote(seg, safe="") for seg in file_path.split("/"))
        
        if repo_type == "dataset":
            upstream_url = f"{hf_endpoint}/datasets/{repo_id}/resolve/{commit_hash}/{encoded_file_path}"
        elif repo_type == "space":
            upstream_url = f"{hf_endpoint}/spaces/{repo_id}/resolve/{commit_hash}/{encoded_file_path}"
        else:
            upstream_url = f"{hf_endpoint}/{repo_id}/resolve/{commit_hash}/{encoded_file_path}"

        # ── 3. Build upstream headers ─────────────────────────────────────
        upstream_headers = {}
        token = self.server.hf_token
        if token:
            upstream_headers["Authorization"] = f"Bearer {token}"
        upstream_headers["User-Agent"] = "llmpt-webseed-proxy/0.1"

        # Forward Range header if present.
        range_header = self.headers.get("Range")
        if range_header:
            upstream_headers["Range"] = range_header

        # ── 4. Make the upstream request ──────────────────────────────────
        try:
            resp = _requests.get(
                upstream_url,
                headers=upstream_headers,
                stream=True,
                timeout=120,
                allow_redirects=True,
            )
        except _requests.RequestException as exc:
            logger.warning("WebSeedProxy: upstream request failed: %s", exc)
            self.send_error(502, "Upstream request failed")
            return

        # ── 5. Forward upstream response ──────────────────────────────────
        self.send_response(resp.status_code)
        for header_name, header_value in resp.headers.items():
            if header_name.lower() in _FORWARDED_HEADERS:
                self.send_header(header_name, header_value)
        self.end_headers()

        try:
            for chunk in resp.iter_content(chunk_size=_STREAM_CHUNK_SIZE):
                if chunk:
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Client (libtorrent) closed the connection early — not an error.
            pass
        finally:
            resp.close()

    # ── Path parsing ──────────────────────────────────────────────────────

    def _parse_path(self):
        """Parse the request path into (repo_type, repo_id, commit_hash, file_path).

        Expected format: ``/ws/{repo_type}/{repo_id...}/{commit_hash}/{file_path...}``

        Returns:
            Tuple of (repo_type, repo_id, commit_hash, file_path) or None on error.
        """
        path = self.path

        # Strip query string if present.
        if "?" in path:
            path = path.split("?", 1)[0]

        # Remove leading /ws/ prefix.
        if not path.startswith("/ws/"):
            return None

        remaining = path[4:]  # strip "/ws/"
        segments = remaining.split("/")
        if len(segments) < 3:
            return None

        repo_type_maybe = segments[0]
        if repo_type_maybe in ("model", "dataset", "space"):
            repo_type = repo_type_maybe
            repo_id_start = 1
        else:
            repo_type = "model"
            repo_id_start = 0

        commit_index = -1
        for i in range(repo_id_start, len(segments)):
            if len(segments[i]) == 40 and all(c in "0123456789abcdef" for c in segments[i].lower()):
                commit_index = i
                break

        if commit_index == -1 or commit_index == repo_id_start:
            return None

        repo_id = "/".join(segments[repo_id_start:commit_index])
        commit_hash = segments[commit_index]
        file_path = "/".join(segments[commit_index+1:])

        if not repo_id or not commit_hash or not file_path:
            return None

        return repo_type, repo_id, commit_hash, file_path


class _WebSeedHTTPServer(HTTPServer):
    """HTTPServer subclass that carries proxy configuration."""

    hf_token: Optional[str] = None
    hf_endpoint: str = "https://huggingface.co"


class WebSeedProxy:
    """Manage the lifecycle of the local WebSeed proxy server.

    Usage::

        proxy = WebSeedProxy(hf_token="hf_xxx")
        port = proxy.start()
        url = proxy.get_webseed_url("meta-llama/Llama-2-7b")
        # ... use url with libtorrent ...
        proxy.stop()
    """

    def __init__(self, hf_token: Optional[str] = None, hf_endpoint: Optional[str] = None) -> None:
        self.token = self._resolve_token(hf_token)
        self.hf_endpoint = hf_endpoint or os.getenv("HF_ENDPOINT", "https://huggingface.co")
        self.port: Optional[int] = None
        self.server: Optional[_WebSeedHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> int:
        """Start the proxy server and return the listening port.

        Uses ``port=0`` so the OS assigns a free port, avoiding conflicts.

        Returns:
            The port number the proxy is listening on.

        Raises:
            RuntimeError: If the proxy is already running.
        """
        if self.server is not None:
            raise RuntimeError("WebSeedProxy is already running")

        self.server = _WebSeedHTTPServer(("127.0.0.1", 0), WebSeedProxyHandler)
        self.server.hf_token = self.token
        self.server.hf_endpoint = self.hf_endpoint.rstrip("/")
        self.port = self.server.server_address[1]

        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="webseed-proxy",
            daemon=True,
        )
        self.thread.start()

        logger.info("WebSeedProxy started on 127.0.0.1:%d", self.port)
        return self.port

    def stop(self) -> None:
        """Stop the proxy server and release resources."""
        if self.server is None:
            return

        self.server.shutdown()
        if self.thread is not None:
            self.thread.join(timeout=5)

        self.server.server_close()
        logger.info("WebSeedProxy stopped (was on port %d)", self.port)

        self.server = None
        self.thread = None
        self.port = None

    def get_webseed_url(self, repo_id: str, repo_type: str = "model") -> str:
        """Build the WebSeed URL for a given repository.

        The trailing ``/`` is critical — it tells libtorrent to use BEP 19
        multi-file URL construction: ``{url}{torrent_name}/{file_path}``.

        Args:
            repo_id: The HuggingFace repository identifier (e.g. "meta-llama/Llama-2-7b").
            repo_type: The repository type.

        Returns:
            WebSeed URL string.

        Raises:
            RuntimeError: If the proxy is not running.
        """
        if self.port is None:
            raise RuntimeError("WebSeedProxy is not running")
        return f"http://127.0.0.1:{self.port}/ws/{repo_type}/{repo_id}/"

    @property
    def is_running(self) -> bool:
        """Return True if the proxy server is currently running."""
        return self.server is not None

    # ── Token resolution ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_token(explicit_token: Optional[str] = None) -> Optional[str]:
        """Resolve the HuggingFace token using priority order.

        Priority:
        1. Explicitly passed token.
        2. ``HF_TOKEN`` environment variable.
        3. Token saved by ``huggingface-cli login`` (``~/.cache/huggingface/token``).
        4. None (public repos only).
        """
        if explicit_token:
            return explicit_token

        env_token = os.getenv("HF_TOKEN")
        if env_token:
            return env_token

        try:
            from huggingface_hub import get_token as hf_get_token
            saved_token = hf_get_token()
            if saved_token:
                return saved_token
        except ImportError:
            pass

        return None
