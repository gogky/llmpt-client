"""
Run the background seeder for the true P2P test.
"""

import os
import time
import pytest
import llmpt
from huggingface_hub import snapshot_download

def test_seeder_initialization():
    """
    1. Create a dummy model.
    2. Start the seeder.
    3. Block and wait to serve downlaoder.
    """
    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    
    # Enable P2P logging and logic
    llmpt.enable_p2p(tracker_url=tracker_url)
    
    repo_id = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    filename = "config.json"
    
    # 1. Download it officially once so the Seeder actually has it in its cache
    print(f"[Seeder] Downloading {repo_id} from HuggingFace to act as the source of truth...", flush=True)
    local_path = snapshot_download(
        repo_id=repo_id,
        local_files_only=False
    )
    
    assert os.path.exists(local_path)
    
    from llmpt.tracker_client import TrackerClient
    from llmpt.p2p_batch import P2PBatchManager
    
    # 2. Start the unified background seeding daemon for this repository
    print(f"[Seeder] Registering seeding task for {repo_id}...", flush=True)
    tracker_client = TrackerClient(tracker_url)
    manager = P2PBatchManager()
    
    from llmpt.torrent_creator import create_and_register_torrent
    
    # Wait a bit for the mock tracker container to be fully live
    print("[Seeder] Waiting 5s for tracker to come online...", flush=True)
    time.sleep(5)
    
    print(f"[Seeder] Creating and registering .torrent metadata on tracker...", flush=True)
    torrent_info = create_and_register_torrent(
        repo_id=repo_id,
        revision="main",
        repo_type="model",
        name=repo_id.split("/")[-1],
        tracker_client=tracker_client,
    )
    assert torrent_info is not None, "Failed to create/register BT metadata"
    
    success = manager.register_seeding_task(repo_id, "main", tracker_client, torrent_data=torrent_info['torrent_data'])
    print(f"[Seeder] Seeding registration returned: {success}. P2P background thread handling remaining tracking.", flush=True)
    
    # 4. Keep the container alive long enough for the downloader to finish
    print("[Seeder] Seeder online and seeding. Waiting for peers...", flush=True)
    for _ in range(36):
        time.sleep(5)

if __name__ == "__main__":
    test_seeder_initialization()
