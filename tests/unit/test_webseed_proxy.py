"""
Tests for WebSeedProxy (llmpt.webseed_proxy).

Covers: WebSeedProxyHandler path parsing, token resolution, proxy lifecycle,
        URL construction, Range forwarding, and error handling.
"""

import os
import time
import pytest
from http.client import HTTPConnection
from unittest.mock import patch, MagicMock

import requests as _real_requests  # keep a reference for test HTTP calls


def _http_get(port, path, headers=None):
    """Make a raw HTTP GET to the proxy using http.client (unaffected by patches)."""
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path, headers=headers or {})
    resp = conn.getresponse()
    body = resp.read()
    resp_headers = dict(resp.getheaders())
    conn.close()
    return resp.status, body, resp_headers


# ─── WebSeedProxyHandler._parse_path ──────────────────────────────────────────

class TestParsePathDirect:
    """Test path parsing via a directly instantiated handler (no server)."""

    def _make_handler(self, path):
        from llmpt.webseed_proxy import WebSeedProxyHandler
        handler = object.__new__(WebSeedProxyHandler)
        handler.path = path
        return handler

    def test_valid_path(self):
        h = self._make_handler("/ws/meta-llama/Llama-2-7b/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/config.json")
        result = h._parse_path()
        assert result == ("model", "meta-llama/Llama-2-7b", "a"*40, "config.json")

    def test_valid_nested_file_path(self):
        h = self._make_handler("/ws/org/repo/dddddddddddddddddddddddddddddddddddddddd/subdir/deep/model.bin")
        result = h._parse_path()
        assert result == ("model", "org/repo", "d"*40, "subdir/deep/model.bin")

    def test_missing_ws_prefix(self):
        h = self._make_handler("/other/org/repo/hash/file.bin")
        assert h._parse_path() is None

    def test_too_few_segments(self):
        h = self._make_handler("/ws/org/repo/hash")
        assert h._parse_path() is None

    def test_exactly_four_segments(self):
        h = self._make_handler("/ws/org/repo/cccccccccccccccccccccccccccccccccccccccc/file.txt")
        result = h._parse_path()
        assert result == ("model", "org/repo", "c"*40, "file.txt")

    def test_strips_query_string(self):
        h = self._make_handler("/ws/org/repo/1111111111111111111111111111111111111111/file.bin?foo=bar")
        result = h._parse_path()
        assert result == ("model", "org/repo", "1"*40, "file.bin")

    def test_empty_segments_rejected(self):
        h = self._make_handler("/ws///hash/file.bin")
        assert h._parse_path() is None

    def test_empty_file_path_rejected(self):
        h = self._make_handler("/ws/org/repo/hash/")
        assert h._parse_path() is None


# ─── WebSeedProxy token resolution ───────────────────────────────────────────

class TestTokenResolution:

    def test_explicit_token_wins(self):
        from llmpt.webseed_proxy import WebSeedProxy
        proxy = WebSeedProxy(hf_token="explicit_token")
        assert proxy.token == "explicit_token"

    def test_env_token_fallback(self):
        from llmpt.webseed_proxy import WebSeedProxy
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}, clear=False):
            proxy = WebSeedProxy(hf_token=None)
            assert proxy.token == "env_token"

    def test_hf_get_token_fallback(self):
        from llmpt.webseed_proxy import WebSeedProxy
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            with patch("huggingface_hub.get_token", return_value="saved_token"):
                proxy = WebSeedProxy(hf_token=None)
                assert proxy.token == "saved_token"

    def test_no_token_available(self):
        from llmpt.webseed_proxy import WebSeedProxy
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            with patch("huggingface_hub.get_token", return_value=None):
                proxy = WebSeedProxy(hf_token=None)
                assert proxy.token is None

    def test_explicit_overrides_env(self):
        from llmpt.webseed_proxy import WebSeedProxy
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}, clear=False):
            proxy = WebSeedProxy(hf_token="explicit")
            assert proxy.token == "explicit"


# ─── WebSeedProxy lifecycle ──────────────────────────────────────────────────

class TestProxyLifecycle:

    def test_start_and_stop(self):
        from llmpt.webseed_proxy import WebSeedProxy
        proxy = WebSeedProxy()
        port = proxy.start()
        assert isinstance(port, int)
        assert port > 0
        assert proxy.is_running

        proxy.stop()
        assert not proxy.is_running
        assert proxy.port is None

    def test_double_start_raises(self):
        from llmpt.webseed_proxy import WebSeedProxy
        proxy = WebSeedProxy()
        proxy.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                proxy.start()
        finally:
            proxy.stop()

    def test_double_stop_is_safe(self):
        from llmpt.webseed_proxy import WebSeedProxy
        proxy = WebSeedProxy()
        proxy.start()
        proxy.stop()
        proxy.stop()  # Should not raise

    def test_get_webseed_url_format(self):
        from llmpt.webseed_proxy import WebSeedProxy
        proxy = WebSeedProxy()
        port = proxy.start()
        try:
            url = proxy.get_webseed_url("meta-llama/Llama-2-7b", repo_type="model")
            assert url == f"http://127.0.0.1:{port}/ws/model/meta-llama/Llama-2-7b/"
            assert url.endswith("/")  # Critical for BEP 19 multi-file mode
        finally:
            proxy.stop()

    def test_get_webseed_url_before_start_raises(self):
        from llmpt.webseed_proxy import WebSeedProxy
        proxy = WebSeedProxy()
        with pytest.raises(RuntimeError, match="not running"):
            proxy.get_webseed_url("org/repo")


# ─── Integration: actual HTTP requests through the proxy ─────────────────────

class TestProxyHTTPIntegration:
    """These tests start a real proxy and make actual HTTP requests using
    http.client (unaffected by ``requests`` patches) while mocking the
    upstream ``_requests.get`` call inside the proxy handler."""

    @pytest.fixture(autouse=True)
    def proxy_fixture(self):
        from llmpt.webseed_proxy import WebSeedProxy
        self.proxy = WebSeedProxy(hf_token="test_token_123")
        self.port = self.proxy.start()
        yield
        self.proxy.stop()

    def _make_upstream_response(self, status_code=200, headers=None, chunks=None):
        """Create a properly configured mock Response for requests.get()."""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        # Make .headers behave like a real dict (items() for iteration, lowercase lookup)
        resp_headers = headers or {}
        mock_resp.headers = resp_headers
        mock_resp.iter_content.return_value = iter(chunks or [])
        mock_resp.close = MagicMock()
        return mock_resp

    def test_successful_proxy_request(self):
        """Full path: libtorrent → proxy → mock HF CDN."""
        mock_resp = self._make_upstream_response(
            status_code=200,
            headers={"Content-Length": "5", "Content-Type": "application/octet-stream"},
            chunks=[b"hello"],
        )

        with patch("llmpt.webseed_proxy._requests.get", return_value=mock_resp) as mock_get:
            status, body, _ = _http_get(self.port, "/ws/org/repo/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/config.json")

        assert status == 200
        assert body == b"hello"

        # Verify upstream URL was constructed correctly
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://huggingface.co/org/repo/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/config.json"
        # Verify token was injected
        assert call_args[1]["headers"]["Authorization"] == "Bearer test_token_123"

    def test_range_request_forwarding(self):
        """Range header should be forwarded to upstream."""
        mock_resp = self._make_upstream_response(
            status_code=206,
            headers={
                "Content-Length": "10",
                "Content-Range": "bytes 0-9/100",
                "Content-Type": "application/octet-stream",
            },
            chunks=[b"0123456789"],
        )

        with patch("llmpt.webseed_proxy._requests.get", return_value=mock_resp) as mock_get:
            status, body, _ = _http_get(
                self.port, "/ws/org/repo/1111111111111111111111111111111111111111/model.bin",
                headers={"Range": "bytes=0-9"},
            )

        assert status == 206
        # Verify Range was forwarded
        upstream_headers = mock_get.call_args[1]["headers"]
        assert upstream_headers["Range"] == "bytes=0-9"

    def test_bad_path_returns_400(self):
        """Malformed paths should return 400."""
        status, _, _ = _http_get(self.port, "/invalid/path")
        assert status == 400

    def test_upstream_failure_returns_502(self):
        """When upstream request fails, should return 502."""
        with patch("llmpt.webseed_proxy._requests.get",
                    side_effect=_real_requests.ConnectionError("Network down")):
            status, _, _ = _http_get(self.port, "/ws/org/repo/1111111111111111111111111111111111111111/file.bin")
        assert status == 502

    def test_no_token_no_auth_header(self):
        """When proxy has no token, no Authorization header should be sent upstream."""
        self.proxy.stop()

        from llmpt.webseed_proxy import WebSeedProxy
        proxy_no_token = WebSeedProxy(hf_token=None)
        proxy_no_token.token = None
        port = proxy_no_token.start()

        try:
            mock_resp = self._make_upstream_response(
                status_code=200,
                headers={"Content-Length": "3"},
                chunks=[b"abc"],
            )

            with patch("llmpt.webseed_proxy._requests.get", return_value=mock_resp) as mock_get:
                status, body, _ = _http_get(port, "/ws/org/repo/1111111111111111111111111111111111111111/file.bin")

            assert status == 200
            upstream_headers = mock_get.call_args[1]["headers"]
            assert "Authorization" not in upstream_headers
        finally:
            proxy_no_token.stop()

    def test_custom_hf_endpoint(self):
        """When custom HF endpoint is set, should use it for upstream URLs."""
        self.proxy.stop()

        from llmpt.webseed_proxy import WebSeedProxy
        proxy_custom = WebSeedProxy(hf_endpoint="https://hf-mirror.com")
        port = proxy_custom.start()

        try:
            mock_resp = self._make_upstream_response(
                status_code=200,
                headers={"Content-Length": "3"},
                chunks=[b"abc"],
            )

            with patch("llmpt.webseed_proxy._requests.get", return_value=mock_resp) as mock_get:
                _http_get(port, "/ws/org/repo/1111111111111111111111111111111111111111/file.bin")

            upstream_url = mock_get.call_args[0][0]
            assert upstream_url.startswith("https://hf-mirror.com/")
        finally:
            proxy_custom.stop()

    def test_nested_file_path(self):
        """Deep file paths should be correctly reconstructed."""
        mock_resp = self._make_upstream_response(
            status_code=200,
            headers={"Content-Length": "3"},
            chunks=[b"abc"],
        )

        with patch("llmpt.webseed_proxy._requests.get", return_value=mock_resp) as mock_get:
            _http_get(
                self.port,
                "/ws/org/repo/1111111111111111111111111111111111111111/subdir/deep/nested/file.safetensors",
            )

        upstream_url = mock_get.call_args[0][0]
        assert upstream_url == "https://huggingface.co/org/repo/resolve/1111111111111111111111111111111111111111/subdir/deep/nested/file.safetensors"


# ─── SessionContext._get_webseed_url ─────────────────────────────────────────

class TestSessionContextWebSeed:
    """Test the _get_webseed_url method on SessionContext."""

    def _make_ctx(self, repo_id="org/repo"):
        with patch('llmpt.session_context.lt', MagicMock()), \
             patch('llmpt.session_context.LIBTORRENT_AVAILABLE', True), \
             patch('os.makedirs'):
            from llmpt.session_context import SessionContext
            ctx = SessionContext(
                repo_id=repo_id, revision="main",
                tracker_client=MagicMock(), lt_session=MagicMock(),
                session_mode='on_demand', timeout=10,
            )
        return ctx

    def test_get_webseed_url_with_proxy_running(self):
        """When proxy port is configured, should return a valid URL."""
        ctx = self._make_ctx()
        with patch('llmpt.get_config', return_value={'webseed_proxy_port': 54321}):
            url = ctx._get_webseed_url()
        assert url == "http://127.0.0.1:54321/ws/model/org/repo/"

    def test_get_webseed_url_without_proxy(self):
        """When proxy port is None, should return None."""
        ctx = self._make_ctx()
        with patch('llmpt.get_config', return_value={'webseed_proxy_port': None}):
            url = ctx._get_webseed_url()
        assert url is None

    def test_get_webseed_url_no_port_key(self):
        """When config doesn't have the port key, should return None."""
        ctx = self._make_ctx()
        with patch('llmpt.get_config', return_value={}):
            url = ctx._get_webseed_url()
        assert url is None
