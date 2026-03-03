"""
End-to-End P2P Download Test.

Verifies that ALL files in a HuggingFace repository are downloaded via P2P,
with zero HTTP fallbacks.
"""

import os
import time
import pytest
from huggingface_hub import snapshot_download
import requests
import llmpt
import shutil


def test_true_p2p_download():
    """
    Downloads the model that the Seeder container is currently hosting.
    Asserts that every single file goes through P2P — no HTTP fallbacks.
    """
    # Wait for the seeder to be fully ready and published to the live tracker
    time.sleep(15)

    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    llmpt.enable_p2p(tracker_url=tracker_url, timeout=60)

    # Reset download stats before the test
    from llmpt.patch import reset_download_stats, get_download_stats
    reset_download_stats()

    repo_id = "hf-internal-testing/tiny-random-GPTJForCausalLM"

    # Wait for seeder to: (1) complete HTTP download, (2) register torrent, (3) finish piece hash check
    print("[Downloader] Waiting 30s for seeder to complete HTTP download and start seeding...", flush=True)
    time.sleep(30)
    print("[Downloader] Requesting snapshot_download...", flush=True)
    local_path = snapshot_download(
        repo_id=repo_id,
        local_files_only=False,
        force_download=True,  # Must bypass HF local cache to ensure http_get is always called
    )

    # ── 1. Basic file existence checks ──
    assert os.path.exists(local_path)
    files_in_repo = set()
    for root, dirs, files in os.walk(local_path):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), local_path)
            files_in_repo.add(rel)

    expected_files = {"config.json", "pytorch_model.bin", "tokenizer.json"}
    missing = expected_files - files_in_repo
    assert not missing, f"Core files missing from snapshot: {missing}"

    # ── 2. P2P download statistics validation ──
    stats = get_download_stats()
    p2p_files = stats['p2p']
    http_files = stats['http']

    print(f"\n{'='*60}", flush=True)
    print(f"  P2P Download Report", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Total files in repo:       {len(files_in_repo)}", flush=True)
    print(f"  Downloaded via P2P:        {len(p2p_files)}", flush=True)
    print(f"  Fell back to HTTP:         {len(http_files)}", flush=True)
    print(f"{'='*60}", flush=True)

    if p2p_files:
        print(f"\n  ✅ P2P files:", flush=True)
        for f in sorted(p2p_files):
            print(f"     ✓ {f}", flush=True)

    if http_files:
        print(f"\n  ❌ HTTP fallback files:", flush=True)
        for f in sorted(http_files):
            print(f"     ✗ {f}", flush=True)

    # Files in repo but not tracked by either P2P or HTTP
    # (these would have bypassed http_get entirely, e.g. via Xet)
    tracked_files = p2p_files | http_files
    untracked = files_in_repo - tracked_files
    if untracked:
        print(f"\n  ⚠️  Untracked files (bypassed http_get entirely!):", flush=True)
        for f in sorted(untracked):
            print(f"     ? {f}", flush=True)

    print(f"\n{'='*60}\n", flush=True)

    # ── 3. Strict assertions ──
    from llmpt.p2p_batch import P2PBatchManager
    manager = P2PBatchManager()

    # The P2P manager must have intercepted the download
    active_keys = [k for k in manager.sessions.keys() if k[0] == repo_id]
    assert len(active_keys) > 0, "P2P Manager did not intercept the download!"

    session_key = active_keys[0]
    session = manager.sessions[session_key]

    # All files in the repo must have been processed by P2P
    assert len(p2p_files) == len(files_in_repo), (
        f"Not all files were downloaded via P2P!\n"
        f"  P2P: {len(p2p_files)}/{len(files_in_repo)}\n"
        f"  HTTP fallbacks: {sorted(http_files) if http_files else 'none'}\n"
        f"  Untracked (bypassed http_get): {sorted(untracked) if untracked else 'none'}"
    )

    # No HTTP fallbacks should occur
    assert len(http_files) == 0, (
        f"Some files fell back to HTTP instead of P2P!\n"
        f"  HTTP fallback files: {sorted(http_files)}"
    )

    print(f"[Downloader] ✅ All {len(p2p_files)} files downloaded via P2P to {local_path}!", flush=True)
