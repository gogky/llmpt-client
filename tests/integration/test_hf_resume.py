import logging
import os
import time
from unittest.mock import patch
from huggingface_hub import hf_hub_download
from llmpt.patch import apply_patch
from llmpt.tracker_client import TrackerClient

logging.basicConfig(level=logging.DEBUG)

def mock_get_torrent_info(self, repo_id, revision=None):
    return {
        "magnet_link": "magnet:?xt=urn:btih:3fa5dc5617bd5b7ccff37fd7e2ec80dcf25dc8eb&dn=gpt2_main&tr=udp%3a%2f%2ftracker.opentrackr.org%3a1337%2fannounce",
        "info_hash": "3fa5dc5617bd5b7ccff37fd7e2ec80dcf25dc8eb",
        "revision": "main"
    }

TrackerClient.get_torrent_info = mock_get_torrent_info

config = {"tracker_url": "http://localhost:8000", "timeout": 8}
apply_patch(config)

try:
    print("--- Starting first download (will timeout and save resume data) ---")
    hf_hub_download(repo_id="gpt2", filename="config.json", force_download=True)
except Exception as e:
    print(f"Error during download: {e}")

time.sleep(3)

print("--- Checking for fastresume cache ---")
repo_str = "gpt2".replace('/', '_')
expected_resume = os.path.expanduser(f"~/.cache/llmpt/p2p_resume/{repo_str}_main.fastresume")

if os.path.exists(expected_resume):
    print(f"SUCCESS: Fastresume file exists at {expected_resume}! Size: {os.path.getsize(expected_resume)} bytes")
else:
    print(f"FAIL: Fastresume file not found at {expected_resume}")
