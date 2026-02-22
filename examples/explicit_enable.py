"""
Example: Explicit enable with custom configuration.
"""

from llmpt import enable_p2p, get_config
from huggingface_hub import snapshot_download

# Enable P2P with custom settings
enable_p2p(
    tracker_url="http://localhost:8080",
    auto_seed=True,
    seed_duration=7200,  # Seed for 2 hours
    timeout=600  # 10 minute timeout
)

# Show configuration
config = get_config()
print(f"P2P Configuration:")
print(f"  Tracker: {config['tracker_url']}")
print(f"  Auto-seed: {config['auto_seed']}")
print(f"  Seed duration: {config['seed_duration']}s")
print(f"  Timeout: {config['timeout']}s")
print()

# Download model
print("Downloading meta-llama/Llama-2-7b...")
path = snapshot_download(
    "meta-llama/Llama-2-7b",
    local_dir="./models/llama-2-7b"
)
print(f"Downloaded to: {path}")
