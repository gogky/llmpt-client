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
    
    print("[Downloader] Waiting 15s for seeder and tracker to initialize...", flush=True)
    time.sleep(15)
    print("[Downloader] Requesting snapshot_download...", flush=True)
    local_path = snapshot_download(
        repo_id=repo_id,
        local_files_only=False
    )
    
    # Verification
    assert os.path.exists(local_path)
    # The whole repo should contain these core files
    files_in_repo = os.listdir(local_path)
    assert "config.json" in files_in_repo
    assert "pytorch_model.bin" in files_in_repo
    assert "tokenizer.json" in files_in_repo
    
    # Strict validation: Ensure it actually went through P2P
    # The monkey patching injects requests into P2PBatchManager.
    # We should see the session alive and files tracked.
    from llmpt.p2p_batch import P2PBatchManager
    manager = P2PBatchManager()
    
    # Find the active session using the repo_id (ignore the exact commit hash in the tuple)
    active_keys = [k for k in manager.sessions.keys() if k[0] == repo_id]
    assert len(active_keys) > 0, "P2P Manager did not intercept the download!"
    
    session_key = active_keys[0]
    session = manager.sessions[session_key]
    assert len(session.file_destinations) >= 3, "P2P session did not process the expected number of files!"
    
    print(f"[Downloader] Successfully P2P fetched {len(session.file_destinations)} files to {local_path}!", flush=True)
