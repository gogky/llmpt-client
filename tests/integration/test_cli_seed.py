import logging
import os
import threading
import time
from unittest.mock import patch
from llmpt.cli import main
import sys

# Replace sys.argv to emulate CLI call
sys.argv = [
    "llmpt-cli", "seed",
    "--repo-id", "gpt2",
    "--revision", "main"
]

logging.basicConfig(level=logging.DEBUG)

def mock_register_torrent(self, repo_id, revision, repo_type, name, info_hash, total_size, file_count, magnet_link, piece_length):
    print(f"MOCK REGISTER CALLED: {repo_id}@{revision} | Hash: {info_hash} | Size: {total_size} | Link: {magnet_link}")
    # Also we need to mock get_torrent_info so P2PBatchManager can retrieve it when we seed
    self.cached_magnet = magnet_link
    return True
    
def mock_get_torrent_info(self, repo_id, revision=None):
    print("MOCK GET TORRENT INFO CALLED")
    return {
        "magnet_link": getattr(self, "cached_magnet", "magnet:?xt=urn:btih:dummy"),
        "info_hash": "dummy_hash",
        "revision": "main"
    }

from llmpt.tracker_client import TrackerClient
TrackerClient.register_torrent = mock_register_torrent
TrackerClient.get_torrent_info = mock_get_torrent_info

print("--- Running llmpt-cli seed ---")
# To prevent the infinite while True loop in cmd_seed, we'll run it in a thread and kill it after 5 seconds
def run_cli():
    try:
        main()
    except SystemExit:
        pass

thread = threading.Thread(target=run_cli, daemon=True)
thread.start()

time.sleep(8)
print("--- Test Complete ---")
import os
os._exit(0)
