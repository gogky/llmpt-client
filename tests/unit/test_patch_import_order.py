"""
Test: apply_patch correctly patches _snapshot_download.hf_hub_download
regardless of the order in which huggingface_hub is imported.

This directly validates the README claim:
  "import order does not affect the patch; only enable_p2p() must be called
   before any download starts."

NOTE on huggingface_hub.__getattr__ (lazy-import):
  The top-level `huggingface_hub` package uses a lazy-import __getattr__ that
  re-fetches attributes from sub-modules on every access.  This means that
  `huggingface_hub.hf_hub_download = patched` is silently overwritten the next
  time __getattr__ runs, so we deliberately do NOT assert on the top-level name.
  The two places that actually matter are:
    - huggingface_hub.file_download.hf_hub_download   (used by direct callers)
    - huggingface_hub._snapshot_download.hf_hub_download  (used by snapshot_download)
"""

import pytest
from huggingface_hub import _snapshot_download  # intentional: imported BEFORE apply_patch
from huggingface_hub import file_download
from unittest.mock import MagicMock, patch as mock_patch
import time

from llmpt.patch import apply_patch, remove_patch


@pytest.fixture(autouse=True)
def cleanup():
    yield
    remove_patch()


def test_snapshot_download_module_is_patched_after_apply():
    """
    Even though we imported _snapshot_download before apply_patch(),
    the module-level reference must be replaced when apply_patch() runs.

    This is the critical import-order test: the patch targets the MODULE OBJECT's
    __dict__, which is the same object regardless of when the import happened.
    """
    original = _snapshot_download.hf_hub_download

    apply_patch({'tracker_url': 'http://test-tracker'})

    assert _snapshot_download.hf_hub_download is not original, (
        "_snapshot_download.hf_hub_download was NOT replaced by apply_patch(). "
        "Import order still matters — README warning must be kept!"
    )
    assert callable(_snapshot_download.hf_hub_download)


def test_file_download_module_is_patched_after_apply():
    """file_download.hf_hub_download (used by direct hf_hub_download callers) must be replaced."""
    original = file_download.hf_hub_download

    apply_patch({'tracker_url': 'http://test-tracker'})

    assert file_download.hf_hub_download is not original, (
        "file_download.hf_hub_download was NOT replaced."
    )
    assert callable(file_download.hf_hub_download)


def test_snapshot_hf_tqdm_is_patched_after_apply():
    """Old snapshot_download references should still see a patched hf_tqdm."""
    original = _snapshot_download.hf_tqdm

    apply_patch({'tracker_url': 'http://test-tracker'})

    assert _snapshot_download.hf_tqdm is not original, (
        "_snapshot_download.hf_tqdm was NOT replaced by apply_patch(). "
        "Import-order fallback for live snapshot postfix is missing."
    )
    assert callable(_snapshot_download.hf_tqdm)


def test_remove_patch_restores_both():
    """
    remove_patch() must restore both module-level references back to the originals.
    """
    original_snap = _snapshot_download.hf_hub_download
    original_fd   = file_download.hf_hub_download

    apply_patch({'tracker_url': 'http://test-tracker'})
    remove_patch()

    assert _snapshot_download.hf_hub_download is original_snap, \
        "_snapshot_download.hf_hub_download was not restored after remove_patch()"
    assert file_download.hf_hub_download is original_fd, \
        "file_download.hf_hub_download was not restored after remove_patch()"


def test_remove_patch_restores_snapshot_hf_tqdm():
    """remove_patch() must restore module-level hf_tqdm as well."""
    original = _snapshot_download.hf_tqdm

    apply_patch({'tracker_url': 'http://test-tracker'})
    remove_patch()

    assert _snapshot_download.hf_tqdm is original, \
        "_snapshot_download.hf_tqdm was not restored after remove_patch()"


def test_double_apply_is_idempotent():
    """Calling apply_patch twice must not double-wrap the function."""
    apply_patch({'tracker_url': 'http://test-tracker'})
    patched_once = _snapshot_download.hf_hub_download

    apply_patch({'tracker_url': 'http://test-tracker'})  # second call
    patched_twice = _snapshot_download.hf_hub_download

    assert patched_once is patched_twice, \
        "apply_patch() is not idempotent — double application wraps the function twice!"


# ---------------------------------------------------------------------------
# Deferred daemon notification (import-order fallback) tests
# ---------------------------------------------------------------------------

class TestDeferredDaemonNotification:
    """Verify that daemon notifications fire even when snapshot_download
    is imported BEFORE apply_patch (the import-order problem)."""

    def test_deferred_notification_fires_when_snapshot_wrapper_inactive(self, monkeypatch):
        """
        Simulate the import-order scenario:
          1) User does `from huggingface_hub import snapshot_download`
          2) Then calls `enable_p2p()` (apply_patch)
          3) Then calls the original snapshot_download reference

        Since _patched_snapshot_download never wraps the call,
        _patched_hf_hub_download should schedule a deferred notification
        that eventually fires.
        """
        import llmpt.patch as patch_mod
        from unittest.mock import MagicMock, patch as mock_patch
        import time

        apply_patch({'tracker_url': 'http://test-tracker'})

        # Mock the daemon notification
        mock_notify = MagicMock(return_value=True)
        # Mock resolve_commit_hash to avoid network calls
        mock_resolve = MagicMock(return_value='abc123' * 7)  # 42 chars

        with mock_patch.object(patch_mod, '_active_wrapper_counts', {}):
            with mock_patch('llmpt.utils.resolve_commit_hash', mock_resolve):
                with mock_patch('llmpt.ipc.notify_daemon', mock_notify):
                    # Directly call _patched_hf_hub_download with a fake original
                    original_fn = MagicMock(return_value='/fake/path')
                    patch_mod._original_hf_hub_download = original_fn

                    patch_mod._patched_hf_hub_download('test/repo', 'model.bin',
                                                       revision='main')

                    # The timer is 2 seconds — wait a bit longer
                    time.sleep(2.5)

                    # Deferred notification should have fired
                    mock_notify.assert_called_once()
                    call_args = mock_notify.call_args
                    assert call_args[0][0] == 'seed'
                    assert call_args[1]['repo_id'] == 'test/repo'

    def test_no_deferred_when_snapshot_wrapper_active(self, monkeypatch):
        """When _patched_snapshot_download is wrapping the call,
        no deferred notification should be scheduled."""
        import llmpt.patch as patch_mod
        from unittest.mock import MagicMock, patch as mock_patch
        import time

        apply_patch({'tracker_url': 'http://test-tracker'})

        mock_notify = MagicMock(return_value=True)
        mock_resolve = MagicMock(return_value='abc123' * 7)

        # Simulate active wrapper repos (the normal case)
        patch_mod._active_wrapper_counts['test/repo'] = 1
        try:
            with mock_patch('llmpt.utils.resolve_commit_hash', mock_resolve):
                with mock_patch('llmpt.ipc.notify_daemon', mock_notify):
                    original_fn = MagicMock(return_value='/fake/path')
                    patch_mod._original_hf_hub_download = original_fn

                    patch_mod._patched_hf_hub_download('test/repo', 'model.bin',
                                                       revision='main')

                    time.sleep(2.5)

                    # No deferred notification should have fired
                    mock_notify.assert_not_called()
        finally:
            patch_mod._active_wrapper_counts.pop('test/repo', None)

    def test_deferred_dedup_key_preserves_multiple_revisions(self, monkeypatch):
        """Different revisions for the same repo should trigger separate notifications."""
        import llmpt.patch as patch_mod
        from unittest.mock import MagicMock, patch as mock_patch
        import time

        apply_patch({'tracker_url': 'http://test-tracker'})

        mock_notify = MagicMock(return_value=True)
        revisions = ['a' * 40, 'b' * 40]

        with mock_patch.object(patch_mod, '_active_wrapper_counts', {}):
            with mock_patch('llmpt.utils.resolve_commit_hash', side_effect=revisions):
                with mock_patch('llmpt.ipc.notify_daemon', mock_notify):
                    patch_mod._original_hf_hub_download = MagicMock(return_value='/fake/path')

                    patch_mod._patched_hf_hub_download('test/repo', 'model.bin', revision='rev-a')
                    patch_mod._patched_hf_hub_download('test/repo', 'model.bin', revision='rev-b')

                    time.sleep(2.5)

                    assert mock_notify.call_count == 2
                    notified_revs = sorted(call.kwargs['revision'] for call in mock_notify.call_args_list)
                    assert notified_revs == sorted(revisions)

    def test_no_deferred_notification_when_hf_download_fails(self, monkeypatch):
        """If hf_hub_download raises, we should not notify daemon."""
        import llmpt.patch as patch_mod
        from unittest.mock import MagicMock, patch as mock_patch
        import time

        apply_patch({'tracker_url': 'http://test-tracker'})

        mock_notify = MagicMock(return_value=True)
        failing_fn = MagicMock(side_effect=RuntimeError("download failed"))

        with mock_patch.object(patch_mod, '_active_wrapper_counts', {}):
            with mock_patch('llmpt.utils.resolve_commit_hash', return_value='c' * 40):
                with mock_patch('llmpt.ipc.notify_daemon', mock_notify):
                    patch_mod._original_hf_hub_download = failing_fn

                    with pytest.raises(RuntimeError, match="download failed"):
                        patch_mod._patched_hf_hub_download('test/repo', 'model.bin', revision='main')

                    time.sleep(2.5)
                    mock_notify.assert_not_called()

    def test_hf_hub_download_deferred_marks_completed_snapshot_for_local_dir(self):
        """When old snapshot_download calls patched hf_hub_download, deferred notify must keep completed_snapshot."""
        import llmpt.patch as patch_mod
        from unittest.mock import MagicMock, patch as mock_patch
        import time

        apply_patch({'tracker_url': 'http://test-tracker'})

        mock_notify = MagicMock(return_value=True)

        with mock_patch.object(patch_mod, '_active_wrapper_counts', {}):
            with mock_patch('llmpt.ipc.notify_daemon', mock_notify):
                original_fn = MagicMock(return_value='/fake/path')
                patch_mod._original_hf_hub_download = original_fn

                def snapshot_download():
                    repo_id = 'org/local-model'
                    revision = 'main'
                    repo_type = 'model'
                    local_dir = '/tmp/local-model'
                    filename = 'weights.bin'
                    return patch_mod._patched_hf_hub_download(
                        repo_id,
                        filename,
                        revision=revision,
                        repo_type=repo_type,
                        local_dir=local_dir,
                    )

                snapshot_download()
                time.sleep(2.5)

                mock_notify.assert_called_once()
                call_args = mock_notify.call_args
                assert call_args[0][0] == 'seed'
                assert call_args[1]['repo_id'] == 'org/local-model'
                assert call_args[1]['local_dir'] == '/tmp/local-model'
                assert call_args[1]['completed_snapshot'] is True

    def test_hf_hub_download_deferred_marks_completed_snapshot_for_cache_dir(self):
        """Import-order snapshot_download(cache_dir=...) should also carry completed_snapshot."""
        import llmpt.patch as patch_mod
        from unittest.mock import MagicMock, patch as mock_patch
        import time

        apply_patch({'tracker_url': 'http://test-tracker'})

        mock_notify = MagicMock(return_value=True)

        with mock_patch.object(patch_mod, '_active_wrapper_counts', {}):
            with mock_patch('llmpt.ipc.notify_daemon', mock_notify):
                original_fn = MagicMock(return_value='/fake/path')
                patch_mod._original_hf_hub_download = original_fn

                def snapshot_download():
                    repo_id = 'org/cache-model'
                    revision = 'main'
                    repo_type = 'model'
                    cache_dir = '/tmp/custom-cache'
                    filename = 'weights.bin'
                    return patch_mod._patched_hf_hub_download(
                        repo_id,
                        filename,
                        revision=revision,
                        repo_type=repo_type,
                        cache_dir=cache_dir,
                    )

                snapshot_download()
                time.sleep(2.5)

                mock_notify.assert_called_once()
                call_args = mock_notify.call_args
                assert call_args[0][0] == 'seed'
                assert call_args[1]['repo_id'] == 'org/cache-model'
                assert call_args[1]['cache_dir'] == '/tmp/custom-cache'
                assert call_args[1]['completed_snapshot'] is True


# ---------------------------------------------------------------------------
# Stack frame inspection (hf_hub_download import-order fallback) tests
# ---------------------------------------------------------------------------

class TestStackFrameInspection:
    """Verify that _extract_context_from_stack correctly recovers P2P context
    when hf_hub_download is imported BEFORE apply_patch."""

    def test_extract_context_from_hf_hub_download_frame(self):
        """Simulate a call stack where hf_hub_download is an ancestor frame.

        _extract_context_from_stack should find the frame and extract
        repo_id, filename, and commit_hash.
        """
        from llmpt.patch import _extract_context_from_stack

        # Simulate being called from within an hf_hub_download call stack.
        # We create a function named 'hf_hub_download' with the expected
        # local variables, then call _extract_context_from_stack from within it.
        def hf_hub_download():
            repo_id = 'test-org/test-model'
            filename = 'model.safetensors'
            commit_hash = 'a' * 40  # 40-char SHA
            revision = 'main'
            repo_type = 'model'
            subfolder = None

            # Simulate the intermediate call chain
            def _download_to_tmp_and_move():
                return _extract_context_from_stack()

            return _download_to_tmp_and_move()

        result = hf_hub_download()

        assert result is not None, "_extract_context_from_stack returned None"
        assert result['repo_id'] == 'test-org/test-model'
        assert result['filename'] == 'model.safetensors'
        assert result['revision'] == 'a' * 40  # should prefer commit_hash
        assert result['repo_type'] == 'model'

    def test_extract_context_with_subfolder(self):
        """subfolder should be prepended to filename."""
        from llmpt.patch import _extract_context_from_stack

        def hf_hub_download():
            repo_id = 'test-org/test-model'
            filename = 'weights.bin'
            commit_hash = 'b' * 40
            revision = 'main'
            repo_type = 'model'
            subfolder = 'unet'

            def _download_to_tmp_and_move():
                return _extract_context_from_stack()

            return _download_to_tmp_and_move()

        result = hf_hub_download()
        assert result['filename'] == 'unet/weights.bin'

    def test_extract_context_prefers_commit_hash_over_revision(self):
        """When commit_hash is available, it should be used over revision."""
        from llmpt.patch import _extract_context_from_stack

        def hf_hub_download():
            repo_id = 'owner/repo'
            filename = 'file.bin'
            commit_hash = 'c' * 40
            revision = 'main'  # raw revision
            repo_type = 'dataset'
            subfolder = None

            def inner():
                return _extract_context_from_stack()
            return inner()

        result = hf_hub_download()
        assert result['revision'] == 'c' * 40, "Should prefer commit_hash over 'main'"
        assert result['repo_type'] == 'dataset'

    def test_extract_context_falls_back_to_revision(self):
        """When commit_hash is None, fall back to revision."""
        from llmpt.patch import _extract_context_from_stack

        def hf_hub_download():
            repo_id = 'owner/repo'
            filename = 'file.bin'
            commit_hash = None
            revision = 'v1.0'
            repo_type = None  # should default to 'model'
            subfolder = None

            def inner():
                return _extract_context_from_stack()
            return inner()

        result = hf_hub_download()
        assert result['revision'] == 'v1.0'
        assert result['repo_type'] == 'model'

    def test_returns_none_when_not_in_hf_hub_download(self):
        """When called outside of hf_hub_download, should return None."""
        from llmpt.patch import _extract_context_from_stack

        def some_other_function():
            def deeper():
                return _extract_context_from_stack()
            return deeper()

        result = some_other_function()
        assert result is None

    def test_extract_snapshot_context_from_snapshot_download_frame(self):
        """snapshot progress fallback should recover repo identity from stack."""
        from llmpt.patch import _extract_snapshot_context_from_stack

        def snapshot_download():
            repo_id = 'test-org/test-dataset'
            commit_hash = 'e' * 40
            revision = 'main'
            repo_type = 'dataset'

            def create_bar():
                return _extract_snapshot_context_from_stack()

            return create_bar()

        result = snapshot_download()

        assert result is not None
        assert result['repo_id'] == 'test-org/test-dataset'
        assert result['revision'] == 'e' * 40
        assert result['repo_type'] == 'dataset'

    def test_extract_snapshot_context_from_inner_snapshot_worker(self):
        """Real snapshot_download workers run in _inner_hf_hub_download frames."""
        from llmpt.patch import _extract_snapshot_context_from_stack

        def _inner_hf_hub_download():
            repo_id = 'test-org/test-dataset'
            commit_hash = '1' * 40
            revision = 'main'
            repo_type = 'dataset'
            cache_dir = '/tmp/custom-cache'

            def create_bar():
                return _extract_snapshot_context_from_stack()

            return create_bar()

        result = _inner_hf_hub_download()

        assert result is not None
        assert result['repo_id'] == 'test-org/test-dataset'
        assert result['revision'] == '1' * 40
        assert result['repo_type'] == 'dataset'
        assert result['cache_dir'] == '/tmp/custom-cache'

    def test_extract_context_from_stack_marks_snapshot_origin_and_storage(self):
        """Import-order fallback should preserve local_dir/cache_dir and snapshot origin."""
        from llmpt.patch import _extract_context_from_stack

        def snapshot_download():
            repo_id = 'test-org/test-model'
            commit_hash = 'f' * 40
            revision = 'main'
            repo_type = 'model'
            local_dir = '/tmp/local-model'

            def _hf_hub_download_to_local_dir():
                filename = 'weights.bin'

                def inner():
                    return _extract_context_from_stack()

                return inner()

            return _hf_hub_download_to_local_dir()

        result = snapshot_download()

        assert result is not None
        assert result['repo_id'] == 'test-org/test-model'
        assert result['revision'] == 'f' * 40
        assert result['local_dir'] == '/tmp/local-model'
        assert result['from_snapshot_download'] is True

    def test_extract_context_from_inner_snapshot_worker_marks_snapshot_origin(self):
        """Worker-thread stack extraction should still mark snapshot origin."""
        from llmpt.patch import _extract_context_from_stack

        def _inner_hf_hub_download():
            repo_id = 'test-org/test-model'
            filename = 'weights.bin'
            commit_hash = '0' * 40
            revision = 'main'
            repo_type = 'model'
            local_dir = '/tmp/local-model'

            def inner():
                return _extract_context_from_stack()

            return inner()

        result = _inner_hf_hub_download()

        assert result is not None
        assert result['repo_id'] == 'test-org/test-model'
        assert result['revision'] == '0' * 40
        assert result['local_dir'] == '/tmp/local-model'
        assert result['from_snapshot_download'] is True

    def test_snapshot_hf_tqdm_fallback_updates_live_postfix(self):
        """Patched module-level hf_tqdm should drive live postfix for old snapshot refs."""
        import llmpt.patch as patch_mod

        apply_patch({'tracker_url': 'http://test-tracker', 'verbose': False})

        class RecordingTqdm:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
                self.disable = kwargs.get('disable', False)
                self.postfixes = []
                self.closed = False

            @classmethod
            def get_lock(cls):
                return None

            @classmethod
            def set_lock(cls, lock):
                return None

            def set_postfix_str(self, text, refresh=False):
                self.postfixes.append((text, refresh))

            def close(self):
                self.closed = True

        patched_tqdm = patch_mod._wrap_snapshot_tqdm_class_auto(RecordingTqdm)
        mock_manager = MagicMock()
        mock_manager.get_repo_p2p_stats.return_value = {
            'peer_download': 1024,
            'webseed_download': 2048,
            'max_p2p_peers': 3,
        }

        with mock_patch('llmpt.p2p_batch.P2PBatchManager', return_value=mock_manager), \
             mock_patch.object(patch_mod, '_SNAPSHOT_PROGRESS_UPDATE_INTERVAL', 0.01):

            def snapshot_download():
                repo_id = 'owner/repo'
                commit_hash = 'f' * 40
                revision = 'main'
                repo_type = 'dataset'

                bar = patched_tqdm(
                    name="huggingface_hub.snapshot_download",
                    total=0,
                    disable=False,
                )
                time.sleep(0.03)
                bar.close()
                return bar

            bar = snapshot_download()

        assert bar.closed is True
        assert any(
            text == "P2P 1.02kB | WebSeed 2.05kB | Active peers 3"
            for text, _ in bar.postfixes
        )


    def test_http_get_schedules_notification_from_stack(self, monkeypatch):
        """When hf_hub_download is imported before enable_p2p, _patched_hf_hub_download
        is bypassed. _patched_http_get must schedule the deferred notification itself
        after recovering context from the stack."""
        import llmpt.patch as patch_mod
        apply_patch({'tracker_url': 'http://test-tracker'})

        mock_notify = MagicMock(return_value=True)

        with mock_patch.object(patch_mod, '_active_wrapper_counts', {}):
            with mock_patch('llmpt.ipc.notify_daemon', mock_notify):
                
                # Mock a call stack as if hf_hub_download was bypassed
                def hf_hub_download():
                    repo_id = 'org/model-from-stack'
                    filename = 'weights.bin'
                    commit_hash = 'd' * 40
                    revision = 'main'
                    repo_type = 'model'
                    
                    def _call_http_get():
                        # Call _patched_http_get. It should inspect the stack, find this frame,
                        # and then trigger a deferred notification.
                        mock_file = MagicMock()
                        mock_file.name = '/tmp/fake'
                        
                        # Mock the actual HTTP fetch or P2P bypass so it doesn't try to download
                        with mock_patch('llmpt.p2p_batch.P2PBatchManager') as mock_manager_cls:
                            mock_manager = MagicMock()
                            mock_manager.register_request.return_value = False
                            mock_manager_cls.return_value = mock_manager
                            # Also mock original_http_get to do nothing
                            with mock_patch.object(patch_mod, '_original_http_get', return_value=None):
                                patch_mod._patched_http_get('http://fake', mock_file)
                    
                    _call_http_get()

                hf_hub_download()

                # Wait for the deferred timer (2 seconds) to fire
                time.sleep(2.5)

                mock_notify.assert_called_once()
                call_args = mock_notify.call_args
                assert call_args[0][0] == 'seed'
                assert call_args[1]['repo_id'] == 'org/model-from-stack'
                assert call_args[1]['revision'] == 'd' * 40

    def test_http_get_does_not_schedule_when_http_fails(self, monkeypatch):
        """Stack fallback must not notify daemon if HTTP fallback raises."""
        import llmpt.patch as patch_mod
        from unittest.mock import MagicMock, patch as mock_patch
        import time

        apply_patch({'tracker_url': 'http://test-tracker'})

        mock_notify = MagicMock(return_value=True)

        with mock_patch.object(patch_mod, '_active_wrapper_counts', {}):
            with mock_patch('llmpt.ipc.notify_daemon', mock_notify):

                def hf_hub_download():
                    repo_id = 'org/model-from-stack'
                    filename = 'weights.bin'
                    commit_hash = 'e' * 40
                    revision = 'main'
                    repo_type = 'model'

                    def _call_http_get():
                        mock_file = MagicMock()
                        mock_file.name = '/tmp/fake'

                        with mock_patch('llmpt.p2p_batch.P2PBatchManager') as mock_manager_cls:
                            mock_manager = MagicMock()
                            mock_manager.register_request.return_value = False
                            mock_manager_cls.return_value = mock_manager

                            with mock_patch.object(
                                patch_mod,
                                '_original_http_get',
                                side_effect=RuntimeError("http failed"),
                            ):
                                with pytest.raises(RuntimeError, match="http failed"):
                                    patch_mod._patched_http_get('http://fake', mock_file)

                    _call_http_get()

                hf_hub_download()
                time.sleep(2.5)
                mock_notify.assert_not_called()

    def test_http_get_deferred_notification_marks_completed_snapshot_for_local_dir(self):
        """Import-order fallback should mark snapshot_download(local_dir=...) as completed."""
        import llmpt.patch as patch_mod
        apply_patch({'tracker_url': 'http://test-tracker'})

        mock_notify = MagicMock(return_value=True)

        with mock_patch.object(patch_mod, '_active_wrapper_counts', {}):
            with mock_patch('llmpt.ipc.notify_daemon', mock_notify):

                def snapshot_download():
                    repo_id = 'org/local-model'
                    commit_hash = '1' * 40
                    revision = 'main'
                    repo_type = 'model'
                    local_dir = '/tmp/local-model'

                    def _hf_hub_download_to_local_dir():
                        filename = 'weights.bin'

                        def _call_http_get():
                            mock_file = MagicMock()
                            mock_file.name = '/tmp/fake'

                            with mock_patch('llmpt.p2p_batch.P2PBatchManager') as mock_manager_cls:
                                mock_manager = MagicMock()
                                mock_manager.register_request.return_value = False
                                mock_manager_cls.return_value = mock_manager
                                with mock_patch.object(patch_mod, '_original_http_get', return_value=None):
                                    patch_mod._patched_http_get('http://fake', mock_file)

                        _call_http_get()

                    _hf_hub_download_to_local_dir()

                snapshot_download()
                time.sleep(2.5)

                mock_notify.assert_called_once()
                call_args = mock_notify.call_args
                assert call_args[0][0] == 'seed'
                assert call_args[1]['repo_id'] == 'org/local-model'
                assert call_args[1]['revision'] == '1' * 40
                assert call_args[1]['local_dir'] == '/tmp/local-model'
                assert call_args[1]['completed_snapshot'] is True

    def test_http_get_deferred_notification_marks_completed_snapshot_for_cache_dir(self):
        """Import-order fallback should mark snapshot_download(cache_dir=...) as completed."""
        import llmpt.patch as patch_mod
        apply_patch({'tracker_url': 'http://test-tracker'})

        mock_notify = MagicMock(return_value=True)

        with mock_patch.object(patch_mod, '_active_wrapper_counts', {}):
            with mock_patch('llmpt.ipc.notify_daemon', mock_notify):

                def snapshot_download():
                    repo_id = 'org/cache-model'
                    commit_hash = '2' * 40
                    revision = 'main'
                    repo_type = 'model'
                    cache_dir = '/tmp/custom-cache'

                    def _hf_hub_download_to_cache_dir():
                        filename = 'weights.bin'

                        def _call_http_get():
                            mock_file = MagicMock()
                            mock_file.name = '/tmp/fake'

                            with mock_patch('llmpt.p2p_batch.P2PBatchManager') as mock_manager_cls:
                                mock_manager = MagicMock()
                                mock_manager.register_request.return_value = False
                                mock_manager_cls.return_value = mock_manager
                                with mock_patch.object(patch_mod, '_original_http_get', return_value=None):
                                    patch_mod._patched_http_get('http://fake', mock_file)

                        _call_http_get()

                    _hf_hub_download_to_cache_dir()

                snapshot_download()
                time.sleep(2.5)

                mock_notify.assert_called_once()
                call_args = mock_notify.call_args
                assert call_args[0][0] == 'seed'
                assert call_args[1]['repo_id'] == 'org/cache-model'
                assert call_args[1]['revision'] == '2' * 40
                assert call_args[1]['cache_dir'] == '/tmp/custom-cache'
                assert call_args[1]['completed_snapshot'] is True
