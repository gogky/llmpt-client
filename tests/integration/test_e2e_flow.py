"""
Integration tests for end-to-end P2P workflows.
"""

import os
import pytest
from huggingface_hub import snapshot_download

import llmpt

# This test requires a live tracker and internet connection
@pytest.mark.integration
def test_end_to_end_p2p_download(live_tracker_url, clean_hf_cache):
    """
    Test a full end-to-end P2P download using the live tracker.
    
    This will:
    1. Query the tracker for a small model (e.g. valid HuggingFace repo).
    2. Since the tracker might not have it, it will fallback to normal HTTP download.
    3. (Future enhancement): In a real seeder setup, the downloaded file is automatically seeded 
       and registered with the tracker. We verify that the system runs without errors.
    """
    # 1. Enable P2P with the real tracker
    llmpt.enable_p2p(tracker_url=live_tracker_url, timeout=30)
    
    assert llmpt.is_enabled() is True
    assert llmpt.get_config()["tracker_url"] == live_tracker_url
    
    # repo that is small and quick to download (~3KB)
    repo_id = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    filename = "config.json"
    
    # 2. Trigger the download 
    # Because our tracker might not have this specific file seeded yet by another peer,
    # the manager should gracefully query the tracker -> get 404/Empty -> fallback to HTTP -> save perfectly.
    local_path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[filename],
        local_files_only=False
    )
    
    # 3. Assertions
    assert os.path.exists(local_path)
    assert filename in os.listdir(local_path)
    
    # Cleanup runs automatically via fixtures
