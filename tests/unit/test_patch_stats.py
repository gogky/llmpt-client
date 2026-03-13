"""
Tests for patch.py statistics and utility functions.

Covers: get_download_stats, reset_download_stats, _truncate_temp_file,
_patched_hf_hub_download context management, and _patched_http_get
P2P-success / P2P-failure / no-context branches.
"""

import time
import threading
import pytest
import httpx
from unittest.mock import patch, MagicMock
from huggingface_hub.errors import LocalEntryNotFoundError

import llmpt.patch as patch_module
from llmpt.patch import (
    apply_patch,
    remove_patch,
    get_download_stats,
    reset_download_stats,
    _record_download_stat,
    _format_snapshot_p2p_postfix,
    _snapshot_stats_key,
    _truncate_temp_file,
    _patched_http_get,
    _patched_hf_hub_download,
    _patched_snapshot_download,
)


@pytest.fixture(autouse=True)
def clean_patch_state():
    """Ensure patch state is clean before and after each test."""
    remove_patch()
    for timer in patch_module._deferred_timers.values():
        timer.cancel()
    patch_module._deferred_timers.clear()
    patch_module._deferred_contexts.clear()
    patch_module._active_wrapper_counts.clear()
    patch_module._active_download_counts.clear()
    patch_module._original_hf_hub_download = None
    patch_module._original_http_get = None
    patch_module._original_snapshot_download = None
    patch_module._original_snapshot_hf_tqdm = None
    patch_module._config = {}
    reset_download_stats()
    # Clear thread-local context
    for attr in ('repo_id', 'repo_type', 'filename', 'revision', 'tracker', 'config', 'cache_dir', 'local_dir'):
        if hasattr(patch_module._context, attr):
            delattr(patch_module._context, attr)
    yield
    remove_patch()
    for timer in patch_module._deferred_timers.values():
        timer.cancel()
    patch_module._deferred_timers.clear()
    patch_module._deferred_contexts.clear()
    patch_module._active_wrapper_counts.clear()
    patch_module._active_download_counts.clear()
    patch_module._original_hf_hub_download = None
    patch_module._original_http_get = None
    patch_module._original_snapshot_download = None
    patch_module._original_snapshot_hf_tqdm = None
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
        stats_key = _snapshot_stats_key("test/repo", "main", "model")
        _record_download_stat(stats_key, 'p2p', "model.bin")
        _record_download_stat(stats_key, 'http', "config.json")

        stats = get_download_stats()
        assert "model.bin" in stats['p2p']
        assert "config.json" in stats['http']


# ─── reset_download_stats ────────────────────────────────────────────────────

class TestResetDownloadStats:

    def test_clears_all(self):
        stats_key = _snapshot_stats_key("test/repo", "main", "model")
        _record_download_stat(stats_key, 'p2p', "a.bin")
        _record_download_stat(stats_key, 'http', "b.bin")

        reset_download_stats()
        stats = get_download_stats()
        assert stats['p2p'] == set()
        assert stats['http'] == set()

    def test_idempotent(self):
        reset_download_stats()
        reset_download_stats()
        stats = get_download_stats()
        assert stats == {'p2p': set(), 'http': set()}


class TestSnapshotPostfix:

    def test_prefers_current_peers_when_p2p_is_active(self):
        text = _format_snapshot_p2p_postfix({
            'active_p2p_peers': 3,
            'peer_download': 1024,
            'webseed_download': 2048,
        })
        assert text == "peers=3"

    def test_shows_http_when_http_fallback_was_observed(self):
        text = _format_snapshot_p2p_postfix(
            {'webseed_download': 2048},
            {'http': {'config.json'}},
        )
        assert text == "http"

    def test_shows_webseed_for_pure_webseed_transfers(self):
        text = _format_snapshot_p2p_postfix({'webseed_download': 2048})
        assert text == "webseed"

    def test_hides_postfix_when_source_is_not_yet_known(self):
        text = _format_snapshot_p2p_postfix({})
        assert text == ""


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

        with patch('llmpt.tracker_client.TrackerClient') as MockTracker, \
             patch('llmpt.utils.resolve_commit_hash', side_effect=lambda r, rev, **kw: rev):
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

        with patch('llmpt.tracker_client.TrackerClient'), \
             patch('llmpt.utils.resolve_commit_hash', side_effect=lambda r, rev, **kw: rev):
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

        with patch('llmpt.tracker_client.TrackerClient'), \
             patch('llmpt.utils.resolve_commit_hash', side_effect=lambda r, rev, **kw: rev):
            _patched_hf_hub_download("test/repo", "model.bin", subfolder="", revision="main")

        assert contexts_seen == ["model.bin"]  # Not "/model.bin"

    def test_context_restored_on_exception(self):
        """Context should be restored even if original function throws."""
        patch_module._original_hf_hub_download = MagicMock(side_effect=RuntimeError("fail"))
        patch_module._config = {'tracker_url': 'http://tracker.example.com'}

        # Set some pre-existing context
        patch_module._context.repo_id = "prev/repo"
        patch_module._context.filename = "prev.bin"

        with patch('llmpt.tracker_client.TrackerClient'), \
             patch('llmpt.utils.resolve_commit_hash', side_effect=lambda r, rev, **kw: rev):
            with pytest.raises(RuntimeError):
                _patched_hf_hub_download("test/repo", "model.bin", revision="main")

        # Context should be restored to previous values
        assert patch_module._context.repo_id == "prev/repo"
        assert patch_module._context.filename == "prev.bin"

    def test_successful_single_file_hands_off_and_releases(self):
        patch_module._original_hf_hub_download = MagicMock(return_value="/path/to/file")
        patch_module._config = {'tracker_url': 'http://tracker.example.com'}
        mock_manager = MagicMock()

        with patch('llmpt.tracker_client.TrackerClient'), \
             patch('llmpt.utils.resolve_commit_hash', side_effect=lambda r, rev, **kw: rev), \
             patch('llmpt.patch._notify_seed_daemon') as mock_notify, \
             patch('llmpt.p2p_batch.P2PBatchManager._instance', mock_manager):
            result = _patched_hf_hub_download("test/repo", "model.bin", revision="main")

        assert result == "/path/to/file"
        mock_notify.assert_called_once_with(
            repo_id="test/repo",
            revision="main",
            repo_type="model",
            cache_dir=None,
            local_dir=None,
        )
        mock_manager.release_on_demand_session.assert_called_once_with(
            repo_id="test/repo",
            revision="main",
            repo_type="model",
            cache_dir=None,
            local_dir=None,
        )

    def test_wrapper_active_skips_single_file_handoff(self):
        patch_module._original_hf_hub_download = MagicMock(return_value="/path/to/file")
        patch_module._config = {'tracker_url': 'http://tracker.example.com'}
        patch_module._active_wrapper_counts["test/repo"] = 1
        mock_manager = MagicMock()

        with patch('llmpt.tracker_client.TrackerClient'), \
             patch('llmpt.utils.resolve_commit_hash', side_effect=lambda r, rev, **kw: rev), \
             patch('llmpt.patch._notify_seed_daemon') as mock_notify, \
             patch('llmpt.p2p_batch.P2PBatchManager._instance', mock_manager):
            result = _patched_hf_hub_download("test/repo", "model.bin", revision="main")

        assert result == "/path/to/file"
        mock_notify.assert_not_called()
        mock_manager.release_on_demand_session.assert_not_called()

    def test_retries_transient_metadata_error(self):
        transient = httpx.ConnectError("[Errno 101] Network is unreachable")
        attempts = {"count": 0}

        def flaky_download(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ValueError("Force download failed due to the above error.") from transient
            return "/path/to/file"

        patch_module._original_hf_hub_download = MagicMock(side_effect=flaky_download)
        patch_module._config = {
            'tracker_url': 'http://tracker.example.com',
            'metadata_error_retries': 2,
            'metadata_error_retry_delay': 1.0,
        }
        mock_manager = MagicMock()

        with patch('llmpt.tracker_client.TrackerClient'), \
             patch('llmpt.utils.resolve_commit_hash', side_effect=lambda r, rev, **kw: rev), \
             patch('llmpt.patch.time.sleep') as mock_sleep, \
             patch('llmpt.patch._notify_seed_daemon'), \
             patch('llmpt.p2p_batch.P2PBatchManager._instance', mock_manager):
            result = _patched_hf_hub_download("test/repo", "model.bin", revision="main")

        assert result == "/path/to/file"
        assert patch_module._original_hf_hub_download.call_count == 2
        mock_sleep.assert_called_once_with(1.0)


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


class TestPatchedSnapshotDownload:

    def test_live_snapshot_progress_uses_proxy_even_when_verbose_disabled(self):
        """snapshot_download should always show live P2P stats on its shared bar."""
        patch_module._config = {'verbose': False}
        patch_module._original_snapshot_download = MagicMock(return_value="/tmp/model")

        class RecordingTqdm:
            instances = []

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.total = kwargs.get('total', 0)
                self.disable = kwargs.get('disable', False)
                self.postfixes = []
                self.closed = False
                RecordingTqdm.instances.append(self)

            def set_postfix_str(self, text, refresh=False):
                self.postfixes.append((text, refresh))

            def close(self):
                self.closed = True

        def fake_snapshot_download(*args, **kwargs):
            bar = kwargs['tqdm_class'](
                name="huggingface_hub.snapshot_download",
                total=0,
                disable=False,
            )
            bar.total += 8
            time.sleep(0.03)
            bar.close()
            return "/tmp/model"

        patch_module._original_snapshot_download.side_effect = fake_snapshot_download

        mock_manager = MagicMock()
        mock_manager.get_repo_p2p_stats.return_value = {
            'active_p2p_peers': 3,
            'peer_download': 1024,
            'webseed_download': 2048,
            'max_p2p_peers': 3,
        }

        with patch('llmpt.utils.resolve_commit_hash', return_value='a' * 40), \
             patch('llmpt.patch._notify_seed_daemon'), \
             patch('llmpt.p2p_batch.P2PBatchManager', return_value=mock_manager), \
             patch.object(patch_module, '_SNAPSHOT_PROGRESS_UPDATE_INTERVAL', 0.01):
            result = _patched_snapshot_download(
                repo_id="test/repo",
                revision="main",
                tqdm_class=RecordingTqdm,
            )

        assert result == "/tmp/model"
        assert len(RecordingTqdm.instances) == 1
        bar = RecordingTqdm.instances[0]
        assert bar.total == 8
        assert bar.closed is True
        assert any(
            text == "peers=3"
            for text, _ in bar.postfixes
        )

    def test_non_snapshot_bars_do_not_get_live_p2p_postfix(self):
        """Only the shared snapshot bar should start the live reporter."""
        patch_module._config = {'verbose': False}
        patch_module._original_snapshot_download = MagicMock(return_value="/tmp/model")

        class RecordingTqdm:
            instances = []

            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
                self.total = kwargs.get('total', 0)
                self.disable = kwargs.get('disable', False)
                self.postfixes = []
                RecordingTqdm.instances.append(self)

            def set_postfix_str(self, text, refresh=False):
                self.postfixes.append((text, refresh))

            def close(self):
                return None

        def fake_snapshot_download(*args, **kwargs):
            bar = kwargs['tqdm_class'](name="thread_map", total=0, disable=False)
            time.sleep(0.03)
            bar.close()
            return "/tmp/model"

        patch_module._original_snapshot_download.side_effect = fake_snapshot_download

        mock_manager = MagicMock()
        mock_manager.get_repo_p2p_stats.return_value = {
            'active_p2p_peers': 3,
            'peer_download': 1024,
            'webseed_download': 2048,
            'max_p2p_peers': 3,
        }

        with patch('llmpt.utils.resolve_commit_hash', return_value='a' * 40), \
             patch('llmpt.patch._notify_seed_daemon'), \
             patch('llmpt.p2p_batch.P2PBatchManager', return_value=mock_manager), \
             patch.object(patch_module, '_SNAPSHOT_PROGRESS_UPDATE_INTERVAL', 0.01):
            _patched_snapshot_download(
                repo_id="test/repo",
                revision="main",
                tqdm_class=RecordingTqdm,
            )

        assert len(RecordingTqdm.instances) == 1
        assert RecordingTqdm.instances[0].postfixes == []

    def test_skips_daemon_notification_without_transfers(self):
        """Pure cache hits must not auto-register a completed snapshot."""
        patch_module._config = {'verbose': False}
        patch_module._original_snapshot_download = MagicMock(return_value="/tmp/model")
        mock_manager = MagicMock()

        with patch('llmpt.utils.resolve_commit_hash', return_value='a' * 40), \
             patch('llmpt.patch._notify_seed_daemon') as mock_notify, \
             patch('llmpt.p2p_batch.P2PBatchManager._instance', mock_manager):
            result = _patched_snapshot_download(
                repo_id="test/repo",
                revision="main",
            )

        assert result == "/tmp/model"
        mock_notify.assert_not_called()
        mock_manager.release_on_demand_session.assert_called_once_with(
            repo_id="test/repo",
            revision='a' * 40,
            repo_type="model",
            cache_dir=None,
            local_dir=None,
        )

    def test_notifies_daemon_for_transferred_snapshot(self):
        """A snapshot that transferred files should mark the source complete."""
        patch_module._config = {'verbose': False}
        mock_manager = MagicMock()

        def fake_snapshot_download(*args, **kwargs):
            stats_key = _snapshot_stats_key(
                "test/repo",
                "a" * 40,
                "model",
            )
            _record_download_stat(stats_key, 'http', "config.json")
            return "/tmp/model"

        patch_module._original_snapshot_download = MagicMock(side_effect=fake_snapshot_download)

        with patch('llmpt.utils.resolve_commit_hash', return_value='a' * 40), \
             patch('llmpt.patch._notify_seed_daemon') as mock_notify, \
             patch('llmpt.p2p_batch.P2PBatchManager._instance', mock_manager):
            result = _patched_snapshot_download(
                repo_id="test/repo",
                revision="main",
            )

        assert result == "/tmp/model"
        mock_notify.assert_called_once_with(
            repo_id="test/repo",
            revision='a' * 40,
            repo_type="model",
            cache_dir=None,
            local_dir=None,
            completed_snapshot=True,
        )
        mock_manager.release_on_demand_session.assert_called_once_with(
            repo_id="test/repo",
            revision='a' * 40,
            repo_type="model",
            cache_dir=None,
            local_dir=None,
        )

    def test_snapshot_completion_ignores_foreign_stats_bucket(self):
        """One snapshot must not treat another snapshot's transfers as its own."""
        patch_module._config = {'verbose': False}
        patch_module._original_snapshot_download = MagicMock(return_value="/tmp/model-a")
        mock_manager = MagicMock()
        foreign_key = _snapshot_stats_key("other/repo", "b" * 40, "model")
        _record_download_stat(foreign_key, 'http', "foreign.bin")

        with patch('llmpt.utils.resolve_commit_hash', return_value='a' * 40), \
             patch('llmpt.patch._notify_seed_daemon') as mock_notify, \
             patch('llmpt.p2p_batch.P2PBatchManager._instance', mock_manager):
            result = _patched_snapshot_download(
                repo_id="test/repo",
                revision="main",
            )

        assert result == "/tmp/model-a"
        mock_notify.assert_not_called()
        mock_manager.release_on_demand_session.assert_called_once_with(
            repo_id="test/repo",
            revision='a' * 40,
            repo_type="model",
            cache_dir=None,
            local_dir=None,
        )

    def test_retries_transient_snapshot_metadata_error(self):
        transient = httpx.ConnectError("[Errno 101] Network is unreachable")
        attempts = {"count": 0}

        def flaky_snapshot_download(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise LocalEntryNotFoundError(
                    "An error happened while trying to locate the files on the Hub."
                ) from transient
            return "/tmp/model"

        patch_module._config = {
            'verbose': False,
            'metadata_error_retries': 2,
            'metadata_error_retry_delay': 1.0,
        }
        patch_module._original_snapshot_download = MagicMock(side_effect=flaky_snapshot_download)
        mock_manager = MagicMock()

        with patch('llmpt.utils.resolve_commit_hash', return_value='a' * 40), \
             patch('llmpt.patch.time.sleep') as mock_sleep, \
             patch('llmpt.p2p_batch.P2PBatchManager._instance', mock_manager):
            result = _patched_snapshot_download(
                repo_id="test/repo",
                revision="main",
            )

        assert result == "/tmp/model"
        assert patch_module._original_snapshot_download.call_count == 2
        mock_sleep.assert_called_once_with(1.0)


class TestDeferredNotification:

    def test_deferred_notification_releases_on_demand_session(self):
        mock_manager = MagicMock()
        key = patch_module._deferred_key("test/repo", "a" * 40, "model")
        patch_module._deferred_contexts[key] = {
            'repo_id': "test/repo",
            'revision': "a" * 40,
            'repo_type': "model",
            'cache_dir': None,
            'local_dir': None,
            'start_time': time.time(),
        }
        patch_module._deferred_timers[key] = MagicMock()

        with patch('llmpt.patch._notify_seed_daemon') as mock_notify, \
             patch('llmpt.p2p_batch.P2PBatchManager._instance', mock_manager):
            patch_module._fire_deferred_notification(key)

        mock_notify.assert_called_once_with(
            repo_id="test/repo",
            revision="a" * 40,
            repo_type="model",
            cache_dir=None,
            local_dir=None,
        )
        mock_manager.release_on_demand_session.assert_called_once_with(
            repo_id="test/repo",
            revision="a" * 40,
            repo_type="model",
            cache_dir=None,
            local_dir=None,
        )
