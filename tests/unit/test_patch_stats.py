"""
Tests for patch.py statistics and utility functions.

Covers: get_download_stats, reset_download_stats, _truncate_temp_file,
_patched_hf_hub_download context management, and _patched_http_get
P2P-success / P2P-failure / no-context branches.
"""

import threading
import pytest
from unittest.mock import patch, MagicMock

import llmpt.patch as patch_module
from llmpt.patch import (
    apply_patch,
    remove_patch,
    get_download_stats,
    reset_download_stats,
    _truncate_temp_file,
    _patched_http_get,
    _patched_hf_hub_download,
)


@pytest.fixture(autouse=True)
def clean_patch_state():
    """Ensure patch state is clean before and after each test."""
    remove_patch()
    patch_module._original_hf_hub_download = None
    patch_module._original_http_get = None
    patch_module._config = {}
    reset_download_stats()
    # Clear thread-local context
    for attr in ('repo_id', 'filename', 'revision', 'tracker', 'config'):
        if hasattr(patch_module._context, attr):
            delattr(patch_module._context, attr)
    yield
    remove_patch()
    patch_module._original_hf_hub_download = None
    patch_module._original_http_get = None
    patch_module._config = {}
    reset_download_stats()


# ─── get_download_stats ──────────────────────────────────────────────────────

class TestGetDownloadStats:

    def test_initial_empty(self):
        stats = get_download_stats()
        assert stats['p2p'] == set()
        assert stats['http'] == set()

    def test_returns_snapshot_not_reference(self):
        """Returned sets should be copies, not references to internal state."""
        stats1 = get_download_stats()
        stats1['p2p'].add("should_not_appear")
        stats2 = get_download_stats()
        assert "should_not_appear" not in stats2['p2p']

    def test_after_manual_insertion(self):
        """Verify stats reflect manual insertions (simulates real usage)."""
        with patch_module._stats_lock:
            patch_module._download_stats['p2p'].add("model.bin")
            patch_module._download_stats['http'].add("config.json")

        stats = get_download_stats()
        assert "model.bin" in stats['p2p']
        assert "config.json" in stats['http']


# ─── reset_download_stats ────────────────────────────────────────────────────

class TestResetDownloadStats:

    def test_clears_all(self):
        with patch_module._stats_lock:
            patch_module._download_stats['p2p'].add("a.bin")
            patch_module._download_stats['http'].add("b.bin")

        reset_download_stats()
        stats = get_download_stats()
        assert stats['p2p'] == set()
        assert stats['http'] == set()

    def test_idempotent(self):
        reset_download_stats()
        reset_download_stats()
        stats = get_download_stats()
        assert stats == {'p2p': set(), 'http': set()}


# ─── _truncate_temp_file ─────────────────────────────────────────────────────

class TestTruncateTempFile:

    def test_successful_truncate(self):
        temp_file = MagicMock()
        _truncate_temp_file(temp_file, "model.bin")
        temp_file.seek.assert_called_once_with(0)
        temp_file.truncate.assert_called_once_with(0)

    def test_exception_does_not_raise(self):
        """Should swallow exceptions gracefully."""
        temp_file = MagicMock()
        temp_file.seek.side_effect = OSError("file descriptor closed")
        # Should NOT raise
        _truncate_temp_file(temp_file, "model.bin")


# ─── _patched_http_get: P2P success path ─────────────────────────────────────

class TestPatchedHttpGetP2PSuccess:

    def test_p2p_success_skips_http(self):
        """When P2P succeeds, should NOT call original http_get and should record stats."""
        mock_original = MagicMock()
        patch_module._original_http_get = mock_original

        mock_manager = MagicMock()
        mock_manager.register_request.return_value = True

        patch_module._context.repo_id = "test/repo"
        patch_module._context.filename = "model.bin"
        patch_module._context.revision = "main"
        patch_module._context.tracker = MagicMock()
        patch_module._context.config = {'timeout': 60}

        temp_file = MagicMock()
        temp_file.name = "/tmp/fake"

        # Must patch where P2PBatchManager is defined, since _patched_http_get
        # does `from .p2p_batch import P2PBatchManager` (lazy import)
        with patch('llmpt.p2p_batch.P2PBatchManager', return_value=mock_manager):
            _patched_http_get("http://example.com/model.bin", temp_file=temp_file)

        # Original http_get should NOT be called
        mock_original.assert_not_called()

        # Stats should record P2P success
        stats = get_download_stats()
        assert "model.bin" in stats['p2p']
        assert "model.bin" not in stats['http']


# ─── _patched_http_get: P2P failure → HTTP fallback ──────────────────────────

class TestPatchedHttpGetP2PFailure:

    def test_p2p_failure_falls_back_to_http(self):
        """When P2P fails, should truncate temp file and call original http_get."""
        mock_original = MagicMock()
        patch_module._original_http_get = mock_original

        mock_manager = MagicMock()
        mock_manager.register_request.return_value = False

        patch_module._context.repo_id = "test/repo"
        patch_module._context.filename = "model.bin"
        patch_module._context.revision = "main"
        patch_module._context.tracker = MagicMock()
        patch_module._context.config = {'timeout': 60}

        temp_file = MagicMock()
        temp_file.name = "/tmp/fake"

        with patch('llmpt.patch.P2PBatchManager', return_value=mock_manager, create=True):
            _patched_http_get("http://example.com/model.bin", temp_file=temp_file)

        # Temp file should be truncated
        temp_file.seek.assert_called_with(0)
        temp_file.truncate.assert_called_with(0)

        # Original http_get should be called with resume_size=0
        mock_original.assert_called_once()
        _, kwargs = mock_original.call_args
        assert kwargs['resume_size'] == 0

        # Stats should record HTTP fallback
        stats = get_download_stats()
        assert "model.bin" in stats['http']

    def test_p2p_exception_falls_back_to_http(self):
        """Exception in P2P path should fall back to HTTP gracefully."""
        mock_original = MagicMock()
        patch_module._original_http_get = mock_original

        patch_module._context.repo_id = "test/repo"
        patch_module._context.filename = "model.bin"
        patch_module._context.revision = "main"
        patch_module._context.tracker = MagicMock()
        patch_module._context.config = {}

        temp_file = MagicMock()
        temp_file.name = "/tmp/fake"

        with patch('llmpt.patch.P2PBatchManager', side_effect=RuntimeError("boom"), create=True):
            _patched_http_get("http://example.com/model.bin", temp_file=temp_file)

        mock_original.assert_called_once()
        stats = get_download_stats()
        assert "model.bin" in stats['http']


# ─── _patched_http_get: no context → straight HTTP ───────────────────────────

class TestPatchedHttpGetNoContext:

    def test_no_context_goes_straight_to_http(self):
        """Without P2P context, should call original http_get directly."""
        mock_original = MagicMock()
        patch_module._original_http_get = mock_original

        # No context set at all
        temp_file = MagicMock()
        temp_file.name = "/tmp/fake"

        _patched_http_get("http://example.com/file.bin", temp_file=temp_file)

        mock_original.assert_called_once()
        # No stats should be recorded (filename is None)
        stats = get_download_stats()
        assert stats['p2p'] == set()
        assert stats['http'] == set()


# ─── _patched_hf_hub_download: context management ────────────────────────────

class TestPatchedHfHubDownload:

    def test_sets_and_restores_context(self):
        """Should set thread-local context before calling original and restore after."""
        mock_original = MagicMock(return_value="/path/to/file")
        patch_module._original_hf_hub_download = mock_original
        patch_module._config = {'tracker_url': 'http://tracker.example.com'}

        with patch('llmpt.tracker_client.TrackerClient') as MockTracker:
            mock_tracker = MagicMock()
            MockTracker.return_value = mock_tracker

            result = _patched_hf_hub_download("test/repo", "model.bin", revision="v1")

        assert result == "/path/to/file"
        mock_original.assert_called_once_with("test/repo", "model.bin", revision="v1")

        # Context should be restored (cleared since there was no previous context)
        assert getattr(patch_module._context, 'repo_id', None) is None

    def test_subfolder_handling(self):
        """Subfolder should be prepended to filename in context."""
        mock_original = MagicMock(return_value="/path/to/file")
        patch_module._original_hf_hub_download = mock_original
        patch_module._config = {'tracker_url': 'http://tracker.example.com'}

        contexts_seen = []

        def capture_context(*args, **kwargs):
            contexts_seen.append(getattr(patch_module._context, 'filename', None))
            return "/path/to/file"

        mock_original.side_effect = capture_context

        with patch('llmpt.tracker_client.TrackerClient'):
            _patched_hf_hub_download("test/repo", "model.bin", subfolder="onnx", revision="main")

        assert contexts_seen == ["onnx/model.bin"]

    def test_empty_subfolder_ignored(self):
        """Empty string subfolder should be treated as None."""
        mock_original = MagicMock(return_value="/path/to/file")
        patch_module._original_hf_hub_download = mock_original
        patch_module._config = {'tracker_url': 'http://tracker.example.com'}

        contexts_seen = []

        def capture_context(*args, **kwargs):
            contexts_seen.append(getattr(patch_module._context, 'filename', None))
            return "/path/to/file"

        mock_original.side_effect = capture_context

        with patch('llmpt.tracker_client.TrackerClient'):
            _patched_hf_hub_download("test/repo", "model.bin", subfolder="", revision="main")

        assert contexts_seen == ["model.bin"]  # Not "/model.bin"

    def test_context_restored_on_exception(self):
        """Context should be restored even if original function throws."""
        patch_module._original_hf_hub_download = MagicMock(side_effect=RuntimeError("fail"))
        patch_module._config = {'tracker_url': 'http://tracker.example.com'}

        # Set some pre-existing context
        patch_module._context.repo_id = "prev/repo"
        patch_module._context.filename = "prev.bin"

        with patch('llmpt.tracker_client.TrackerClient'):
            with pytest.raises(RuntimeError):
                _patched_hf_hub_download("test/repo", "model.bin", revision="main")

        # Context should be restored to previous values
        assert patch_module._context.repo_id == "prev/repo"
        assert patch_module._context.filename == "prev.bin"


# ─── apply_patch / remove_patch integration ───────────────────────────────────

class TestApplyRemovePatchIntegration:

    def test_apply_is_idempotent(self):
        """Calling apply_patch twice should not double-wrap."""
        from huggingface_hub import file_download

        apply_patch({'tracker_url': 'http://test'})
        first_http_get = file_download.http_get

        apply_patch({'tracker_url': 'http://test'})
        second_http_get = file_download.http_get

        assert first_http_get is second_http_get
        remove_patch()

    def test_remove_without_apply_is_safe(self):
        """remove_patch when not applied should be a no-op."""
        remove_patch()  # Should not raise

    def test_full_cycle(self):
        """apply → verify patched → remove → verify restored."""
        from huggingface_hub import file_download

        original_http_get = file_download.http_get

        apply_patch({'tracker_url': 'http://test'})
        assert file_download.http_get is not original_http_get

        remove_patch()
        assert file_download.http_get is original_http_get
