import logging
import os
import threading
import time
import shutil
import huggingface_hub
from unittest.mock import patch
from huggingface_hub import HfApi
from llmpt.patch import apply_patch
from llmpt.tracker_client import TrackerClient

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# We use a tiny dataset or model repo for testing, e.g. "hf-internal-testing/tiny-random-gpt2"
TEST_REPO = "hf-internal-testing/tiny-random-gpt2"
TEST_FILE = "config.json"
TEST_REVISION = "main"

print(f"--- Starting Full E2E Test on {TEST_REPO} ---")

# Step 1: Clean local cache to ensure fresh state
print("1. Cleaning HuggingFace block-cache for this test repo...")
api = HfApi()
repo_str = TEST_REPO.replace('/', '--')
cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo_str}")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
resume_file = os.path.expanduser(f"~/.cache/llmpt/p2p_resume/{TEST_REPO.replace('/', '_')}_{TEST_REVISION}.fastresume")
if os.path.exists(resume_file):
    os.remove(resume_file)

# Mocking Tracker to pretend a torrent exists for the P2P layer
class MockTrackerClient:
    def __init__(self, *args, **kwargs):
        self.tracker_url = "http://localhost:8000"
    
    def get_torrent_info(self, repo_id, revision=None):
        if repo_id != TEST_REPO:
            return None
        return {
            "magnet_link": "magnet:?xt=urn:btih:1234567890abcdef1234567890abcdef12345678&dn=dummy",
            "info_hash": "1234567890abcdef1234567890abcdef12345678",
            "revision": TEST_REVISION
        }

# Inject the patch
print("2. Applying P2P Patches...")
apply_patch({"tracker_url": "http://localhost:8000", "timeout": 8})

# Step 2: Triggering a download (This will timeout from P2P and fallback to HTTP)
print("3. Attempting P2P Download (will timeout and fallback to HTTP)...")
try:
    # We expect this to take ~8 seconds (the timeout setting) then download via standard HF HTTP.
    with patch('llmpt.tracker_client.TrackerClient', return_value=MockTrackerClient()):
        path = huggingface_hub.hf_hub_download(repo_id=TEST_REPO, filename=TEST_FILE, revision=TEST_REVISION)
    print(f"   [SUCCESS] File successfully stored natively at: {path}")
except Exception as e:
    print(f"   [ERROR] Download failed: {e}")

# We should give the background daemon thread a few seconds to trigger `save_resume_data`
print("4. Waiting for daemon to save fastresume state...")
time.sleep(6)

# Step 3: Verify `.fastresume` tracking exists
print("5. Verifying .fastresume creation...")
if os.path.exists(resume_file):
    print(f"   [SUCCESS] Fastresume file correctly serialized at {resume_file}")
else:
    print(f"   [ERROR] Fastresume file missing!")

# Step 4: Test background seeding integration on the newly downloaded HTTP files
print("6. Simulating Background Seeding (retroactive hashing)...")
from llmpt.p2p_batch import P2PBatchManager
manager = P2PBatchManager()
# Verify it has properly mapped the file we just downloaded!
print("   Checking internal torrent mapping dict...")
print(f"   [DEBUG] Manager ID: {id(manager)}")
print(f"   [DEBUG] Manager sessions keys: {list(manager.sessions.keys())}")
session_ctx = manager.sessions.get((TEST_REPO, TEST_REVISION))

if session_ctx:
    print(f"   [SUCCESS] Session context exists in P2PBatch.")
    if TEST_FILE in session_ctx.file_destinations:
        print(f"   [SUCCESS] The file {TEST_FILE} is mapped internally to: {session_ctx.file_destinations[TEST_FILE]}")
    else:
        print(f"   [ERROR] File missing from P2P mapped destinations.")
else:
    print("   [ERROR] No active P2P session context found.")

print("\n--- All Tests Completed ---")
