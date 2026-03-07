"""
Example: Explicit enable with custom configuration.

Import order does NOT matter — you can import huggingface_hub functions
before or after calling enable_p2p().
"""

from huggingface_hub import snapshot_download   # OK to import first!
from llmpt import enable_p2p, get_config

# Enable P2P with custom settings — works even though snapshot_download
# was imported above.
enable_p2p(
    tracker_url="http://localhost:8080",
    timeout=600  # 10 minute timeout
)

# Show configuration
config = get_config()
print(f"P2P Configuration:")
print(f"  Tracker: {config['tracker_url']}")
print(f"  Timeout: {config['timeout']}s")
print()

# Download model
print("Downloading meta-llama/Llama-2-7b...")
path = snapshot_download(
    "meta-llama/Llama-2-7b",
    local_dir="./models/llama-2-7b"
)
print(f"Downloaded to: {path}")
