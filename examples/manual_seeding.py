"""
Example: Manual seeding control.
"""

from llmpt import enable_p2p, stop_seeding
from llmpt.seeder import get_seeding_status
from llmpt.utils import format_bytes
from huggingface_hub import snapshot_download
import time

# Enable P2P without auto-seeding
enable_p2p(
    tracker_url="http://localhost:8080",
    auto_seed=False  # We'll control seeding manually
)

# Download model
print("Downloading gpt2...")
path = snapshot_download("gpt2")
print(f"Downloaded to: {path}")

# Manually start seeding
from llmpt.torrent_creator import create_and_register_torrent
from llmpt.seeder import start_seeding
from llmpt.tracker_client import TrackerClient

# (In real usage, you'd get commit_hash from the download)
# This is just an example

# Check seeding status periodically
print("\nSeeding status:")
for i in range(5):
    status = get_seeding_status()
    if status:
        for info_hash, info in status.items():
            print(f"  {info_hash[:16]}... - "
                  f"Uploaded: {format_bytes(info['uploaded'])}, "
                  f"Peers: {info['peers']}")
    else:
        print("  No active seeding")
    time.sleep(2)

# Stop seeding
print("\nStopping all seeding...")
stop_seeding()
print("Done!")
