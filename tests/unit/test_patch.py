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
    that P2PBatchManager.register_request is invoked with the correct args.
    """
    config = {'tracker_url': 'http://test', 'timeout': 300}
    apply_patch(config)

    import llmpt.patch as patch_module

    mock_tracker_client = MagicMock()
    mock_batch_manager = MagicMock()
    mock_batch_manager.register_request.return_value = True

    with patch('llmpt.p2p_batch.P2PBatchManager', return_value=mock_batch_manager):

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
            mock_batch_manager.register_request.assert_called_once_with(
                repo_id="demo/repo",
                revision="main",
                filename="model.bin",
                temp_file_path="/tmp/fake",
                tracker_client=mock_tracker_client,
                timeout=300,
                repo_type="model"
            )
        finally:
            # Cleanup context
            patch_module._context.repo_id = None
            patch_module._context.filename = None
            patch_module._context.revision = None
            patch_module._context.tracker = None
            patch_module._context.config = None
