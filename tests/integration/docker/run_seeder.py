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
        allow_patterns=[filename],
        local_files_only=False
    )
    
    assert os.path.exists(local_path)
    
    from llmpt.torrent_creator import create_torrent
    from llmpt.tracker_client import TrackerClient
    from llmpt.seeder import start_seeding
    
    # 2. Force the creator to make a torrent of this repo and publish it
    print(f"[Seeder] Creating torrent for {repo_id}...", flush=True)
    tracker_client = TrackerClient(tracker_url)
    target_file = os.path.join(local_path, filename)
    
    torrent_info = create_torrent(target_file, tracker_url)
    assert torrent_info is not None, "Failed to create torrent"

    success = tracker_client.register_torrent(
        repo_id=repo_id,
        revision="main",
        repo_type="model",
        name="test model",
        info_hash=torrent_info['info_hash'],
        total_size=torrent_info['file_size'],
        file_count=1,
        magnet_link=torrent_info['magnet_link'],
        piece_length=torrent_info['piece_length']
    )
    assert success is True, "Failed to publish torrent to tracker"
    
    # 3. Start the seeder daemon in the background
    print("[Seeder] Starting seeder daemon...", flush=True)
    started = start_seeding(torrent_info=torrent_info, file_path=target_file, duration=180)
    assert started, "Seeder failed to start"
    
    # 4. Keep the container alive long enough for the downloader to finish
    print("[Seeder] Seeder online and seeding. Waiting for peers...", flush=True)
    for _ in range(36):
        time.sleep(5)
