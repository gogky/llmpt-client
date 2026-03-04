"""
Run the background seeder for the true P2P test.

Drives seeding through the public CLI entry point `cmd_seed`, verifying the
exact same code path that `llmpt-cli seed` uses (see todo 1.4).
"""

import os
import time
import signal
import threading
import pytest
from huggingface_hub import snapshot_download


def test_seeder_initialization():
    """
    Real-user seeding flow via CLI:
      1. Download the repo via HuggingFace HTTP (populates local cache).
      2. Invoke `cmd_seed()` — the same function behind `llmpt-cli seed`.
         A background timer sends KeyboardInterrupt after 3 minutes to break
         out of the infinite seeding loop, just like a real user pressing Ctrl+C.
    """
    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    repo_id = "hf-internal-testing/tiny-random-GPTJForCausalLM"

    # ── 1. Populate HF local cache (before enabling P2P) ──
    print(f"[Seeder] Downloading {repo_id} from HuggingFace (HTTP)...", flush=True)
    local_path = snapshot_download(repo_id=repo_id, local_files_only=False)
    assert os.path.exists(local_path), f"snapshot_download returned non-existent path: {local_path}"

    # Wait for tracker to be reachable
    print("[Seeder] Waiting 5s for tracker to come online...", flush=True)
    time.sleep(5)

    # ── 2. Seed via the CLI entry point (cmd_seed) ──
    # cmd_seed() enters an infinite `while True: sleep(1)` loop at the end,
    # which is how the real CLI keeps the seeder alive.  We use SIGALRM to
    # send ourselves a KeyboardInterrupt after 3 minutes — simulating the
    # user pressing Ctrl+C.
    from unittest.mock import MagicMock
    from llmpt.cli import cmd_seed

    args = MagicMock()
    args.tracker = tracker_url
    args.repo_id = repo_id
    args.revision = "main"
    args.repo_type = "model"
    args.name = repo_id.split("/")[-1]

    def _alarm_handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(180)  # 3 minutes

    try:
        cmd_seed(args)
    finally:
        signal.alarm(0)  # Cancel the alarm

    print("[Seeder] ✅ Seeding session completed.", flush=True)


if __name__ == "__main__":
    test_seeder_initialization()
