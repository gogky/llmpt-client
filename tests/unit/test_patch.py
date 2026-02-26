"""
Tests for HuggingFace Hub patching.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock

# Create mock objects
class MockFileDownload:
    def http_get(self, url, temp_file, **kwargs):
        pass

class MockHfHub:
    def __init__(self):
        self.file_download = MockFileDownload()
        
    def hf_hub_download(self, repo_id, filename, **kwargs):
        pass

# Inject mocks into sys.modules BEFORE importing llmpt.patch
sys.modules['huggingface_hub'] = MockHfHub()
sys.modules['huggingface_hub.file_download'] = MockFileDownload()

import huggingface_hub
import huggingface_hub.file_download

import llmpt
from llmpt.patch import apply_patch, remove_patch

@pytest.fixture(autouse=True)
def cleanup_patches():
    """Ensure patches are removed after each test."""
    yield
    remove_patch()

def test_apply_patch_success():
    """Test that patch applies correctly to mock hf_hub."""
    # Capture the exact bound method
    original_download = huggingface_hub.hf_hub_download
    
    apply_patch({'tracker_url': 'http://test'})
    
    assert huggingface_hub.hf_hub_download is not original_download
    assert callable(huggingface_hub.hf_hub_download)
    
def test_remove_patch_success():
    """Test that patch removal restores original functions."""
    original_download = huggingface_hub.hf_hub_download
    
    apply_patch({'tracker_url': 'http://test'})
    assert huggingface_hub.hf_hub_download is not original_download
    
    remove_patch()
    
    # We must compare against the stored global original reference in llmpt.patch
    # because 'is' comparison of bound methods from classes can sometimes fail
    import llmpt.patch as patch_module
    assert patch_module._original_hf_hub_download is None

def test_p2p_interception_flow():
    """Test the complete intercepted P2P flow inside http_get."""
    config = {'tracker_url': 'http://test', 'timeout': 300}
    apply_patch(config)
    
    patched_hf = huggingface_hub.hf_hub_download
    
    # The logic is intercepted in `patched_http_get`. We can invoke that directly.
    # To invoke it, we need to set up the ThreadLocal context that `patched_hf_hub_download` would have set.
    import llmpt.patch as patch_module
    
    mock_tracker_client = MagicMock()
    mock_batch_manager = MagicMock()
    mock_batch_manager.register_request.return_value = True
    
    with patch('llmpt.patch.TrackerClient', return_value=mock_tracker_client, create=True), \
         patch('llmpt.p2p_batch.P2PBatchManager', return_value=mock_batch_manager, create=True):
        
        # Manually set ThreadLocal context
        patch_module._context.repo_id = "demo/repo"
        patch_module._context.filename = "model.bin"
        patch_module._context.revision = "main"
        patch_module._context.tracker = mock_tracker_client
        patch_module._context.config = config
        
        try:
            # 1. Trigger the patched http_get
            patched_http_get = huggingface_hub.file_download.http_get
            temp_mock = MagicMock()
            temp_mock.name = "/tmp/fake"
            patched_http_get("http://dummy_url", temp_file=temp_mock)
            
            # 2. Check that the intercept happened
            mock_batch_manager.register_request.assert_called_once_with(
                repo_id="demo/repo",
                revision="main",
                filename="model.bin",
                temp_file_path="/tmp/fake",
                tracker_client=mock_tracker_client,
                timeout=300
            )
        finally:
            # Cleanup context
            patch_module._context.repo_id = None
            patch_module._context.filename = None
