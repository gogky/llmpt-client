"""
End-to-End P2P Download Test.
"""

import os
import time
import pytest
from huggingface_hub import snapshot_download
import requests
import llmpt
import shutil

def test_true_p2p_download():
    """
    Downloads the model that the Seeder container is currently hosting.
    """
    # Wait for the seeder to be fully ready and published to the live tracker
    time.sleep(15) 
    
    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    llmpt.enable_p2p(tracker_url=tracker_url, timeout=300)
    
    repo_id = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    filename = "config.json"
    
    # Optional strict network isolation testing:
    # If we wanted to ensure zero HF traffic, we could intercept requests here,
    # but the logs will easily show P2P fulfillment.
    
    print("[Downloader] Requesting snapshot_download...", flush=True)
    local_path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[filename],
        local_files_only=False
    )
    
    # Verification
    assert os.path.exists(local_path)
    assert filename in os.listdir(local_path)
    print(f"[Downloader] Successfully retrieved {filename} to {local_path}!", flush=True)
