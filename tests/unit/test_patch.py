"""
Tests for HuggingFace Hub patching.

Uses the REAL huggingface_hub package (which must be installed) so that
apply_patch / remove_patch can exercise the actual import path
(`from huggingface_hub import file_download, _snapshot_download`).

Previous approach injected fake sys.modules entries which broke apply_patch
and also polluted the module namespace for later tests in the same session.
"""

import pytest
from unittest.mock import patch, MagicMock

import huggingface_hub
from huggingface_hub import file_download

import llmpt
from llmpt.patch import apply_patch, remove_patch
from llmpt.utils import get_hf_hub_cache


@pytest.fixture(autouse=True)
def cleanup_patches():
    """Ensure patches are removed after each test."""
    yield
    remove_patch()
    # Also reset the module-level guard so apply_patch can be called again
    import llmpt.patch as pm
    pm._original_hf_hub_download = None
    pm._original_http_get = None
    pm._config = {}


def test_apply_patch_success():
    """Test that patch applies correctly to real hf_hub modules."""
    original_http_get = file_download.http_get

    apply_patch({'tracker_url': 'http://test'})

    # http_get on the real file_download module should be replaced
    assert file_download.http_get is not original_http_get
    assert callable(file_download.http_get)


def test_remove_patch_success():
    """Test that patch removal restores original functions."""
    original_http_get = file_download.http_get

    apply_patch({'tracker_url': 'http://test'})
    assert file_download.http_get is not original_http_get

    remove_patch()

    # After removal the original should be restored
    assert file_download.http_get is original_http_get

    import llmpt.patch as patch_module
    assert patch_module._original_hf_hub_download is None


def test_p2p_interception_flow():
    """Test the complete intercepted P2P flow inside http_get.

    We apply a real patch (so http_get is replaced), then manually set up
    the ThreadLocal context and call the patched http_get directly to verify
    that the transfer coordinator is invoked with the correct args.
    """
    config = {'tracker_url': 'http://test', 'timeout': 300}
    apply_patch(config)

    import llmpt.patch as patch_module
    from llmpt.transfer_types import TransferResult

    mock_tracker_client = MagicMock()
    mock_coordinator = MagicMock()
    mock_coordinator.fulfill_request.return_value = TransferResult(
        success=True,
        delivered_path="/tmp/fake",
        via="p2p",
    )

    with patch('llmpt.transfer_coordinator.TransferCoordinator', return_value=mock_coordinator):

        # Manually set ThreadLocal context (normally done by patched_hf_hub_download)
        patch_module._context.repo_id = "demo/repo"
        patch_module._context.filename = "model.bin"
        patch_module._context.revision = "main"
        patch_module._context.tracker = mock_tracker_client
        patch_module._context.config = config

        try:
            # Trigger the patched http_get via the real module reference
            patched_http_get = file_download.http_get
            temp_mock = MagicMock()
            temp_mock.name = "/tmp/fake"
            patched_http_get("http://dummy_url", temp_file=temp_mock)

            # Check that the intercept happened
            mock_coordinator.fulfill_request.assert_called_once_with(
                repo_id="demo/repo",
                revision="main",
                filename="model.bin",
                destination="/tmp/fake",
                tracker_client=mock_tracker_client,
                repo_type="model",
                cache_dir=None,
                local_dir=None,
                config=config,
                tqdm_class=None,
            )
        finally:
            # Cleanup context
            patch_module._context.repo_id = None
            patch_module._context.filename = None
            patch_module._context.revision = None
            patch_module._context.tracker = None
            patch_module._context.config = None


def test_snapshot_download_notification_handles_storage_key_shape():
    config = {'tracker_url': 'http://test', 'timeout': 300}
    apply_patch(config)

    import llmpt.patch as patch_module

    with patch('llmpt.patch._original_snapshot_download', return_value="/tmp/local"), \
         patch('llmpt.utils.resolve_commit_hash', return_value="a" * 40), \
         patch('llmpt.patch.get_download_stats', return_value={'p2p': {'config.json'}, 'http': set()}), \
         patch('llmpt.ipc.notify_daemon') as notify_daemon:

        key = patch_module._deferred_key(
            repo_id="demo/repo",
            revision="main",
            repo_type="model",
            local_dir="/tmp/model",
        )
        patch_module._deferred_timers[key] = MagicMock()
        patch_module._deferred_contexts[key] = {
            "repo_id": "demo/repo",
            "revision": "main",
            "repo_type": "model",
            "local_dir": "/tmp/model",
        }

        result = huggingface_hub.snapshot_download(
            repo_id="demo/repo",
            revision="main",
            local_dir="/tmp/model",
        )

        assert result == "/tmp/local"
        notify_daemon.assert_called_once_with(
            "seed",
            repo_id="demo/repo",
            revision="a" * 40,
            repo_type="model",
            local_dir="/tmp/model",
            completed_snapshot=True,
        )


def test_deferred_key_normalizes_default_hub_cache():
    import llmpt.patch as patch_module

    assert patch_module._deferred_key(
        repo_id="demo/repo",
        revision="main",
        repo_type="model",
    ) == ("model", "demo/repo", "main", "hub_cache", get_hf_hub_cache())
