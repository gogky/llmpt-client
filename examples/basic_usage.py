"""
Example: Basic usage with environment variable.
"""

import os

# Enable P2P via environment variable
os.environ['HF_USE_P2P'] = '1'
os.environ['HF_P2P_TRACKER'] = 'http://localhost:8080'

# Import llmpt (will auto-enable P2P)
import llmpt

# Use huggingface_hub as normal
from huggingface_hub import snapshot_download

print("Downloading gpt2 model...")
path = snapshot_download("gpt2")
print(f"Downloaded to: {path}")

# Check seeding status
from llmpt.seeder import get_seeding_status
status = get_seeding_status()
print(f"\nActive seeding tasks: {len(status)}")
