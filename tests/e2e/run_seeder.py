"""
Seeder entry point using the daemon architecture.

Instead of the removed foreground seeding CLI,
this uses the daemon to:
  1. Download the repo via HuggingFace HTTP (populates local cache)
  2. Start the daemon, which scans the cache and creates torrents
  3. Wait until the torrent is registered on the tracker

This mirrors the real user experience where the daemon runs in the
background and automatically seeds all cached models.
"""

import os
import time
import signal
import pytest
from huggingface_hub import snapshot_download


def test_seeder_initialization():
    """
    Real-user seeding flow via daemon:
      1. Download the repo via HuggingFace HTTP (populates local cache).
      2. Start the daemon which auto-discovers cached models.
      3. Daemon creates torrent, registers with tracker, and starts seeding.
      4. Wait for the torrent to appear on the tracker.
      5. Keep seeding for 3 minutes (simulating a user leaving the daemon running).
    """
    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    repo_id = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    dataset_id = "fka/prompts.chat"

    # ── 1. Populate HF local cache (before enabling P2P) ──
    print(f"[Seeder] Downloading {repo_id} from HuggingFace (HTTP)...", flush=True)
    local_path = snapshot_download(repo_id=repo_id, local_files_only=False)
    assert os.path.exists(local_path), f"snapshot_download returned non-existent path: {local_path}"

    print(f"[Seeder] Downloading {dataset_id} from HuggingFace (HTTP)...", flush=True)
    dataset_path = snapshot_download(repo_id=dataset_id, repo_type="dataset", local_files_only=False)
    assert os.path.exists(dataset_path), f"snapshot_download returned non-existent path: {dataset_path}"

    # Wait for tracker to be reachable
    print("[Seeder] Waiting 5s for tracker to come online...", flush=True)
    time.sleep(5)

    # ── 2. Start the daemon (it will scan cache and create torrents) ──
    from llmpt.daemon import start_daemon, stop_daemon, is_daemon_running

    port = int(os.environ.get("HF_P2P_PORT", 0)) or None
    print(f"[Seeder] Starting daemon (tracker: {tracker_url}, port: {port})...", flush=True)
    pid = start_daemon(tracker_url=tracker_url, port=port, foreground=False)
    assert pid is not None, "Failed to start daemon"
    print(f"[Seeder] Daemon started (PID: {pid})", flush=True)

    # ── 3. Wait for the torrent to appear on the tracker ──
    import requests
    api_url = f"{tracker_url.rstrip('/')}/api/v1/torrents"
    deadline = time.time() + 180  # 3 minutes max wait
    registered = False

    print("[Seeder] Waiting for daemon to create and register torrent...", flush=True)
    model_registered = False
    dataset_registered = False

    while time.time() < deadline:
        try:
            resp = requests.get(api_url, timeout=5)
            if resp.status_code == 200:
                body = resp.json()
                torrents = body.get("data", body) if isinstance(body, dict) else body
                
                for t in torrents:
                    if t.get("repo_id") == repo_id:
                        if not model_registered:
                            print(f"[Seeder] ✅ Model Torrent registered! info_hash={t.get('info_hash', '?')[:16]}...", flush=True)
                            model_registered = True
                    elif t.get("repo_id") == dataset_id:
                        if not dataset_registered:
                            print(f"[Seeder] ✅ Dataset Torrent registered! info_hash={t.get('info_hash', '?')[:16]}...", flush=True)
                            dataset_registered = True
        except Exception as e:
            print(f"[Seeder] Tracker poll failed: {e}", flush=True)

        if model_registered and dataset_registered:
            break
        time.sleep(5)

    assert model_registered and dataset_registered, (
        f"Daemon did not register both torrents within 180s. "
        "Check daemon logs at ~/.cache/llmpt/daemon.log"
    )

    # ── 4. Wait for downloader to signal that P2P tests are done ──
    # Instead of blindly sleeping a fixed duration (which causes flaky tests
    # when the download takes longer than expected), we wait for the downloader
    # to write a signal file.  Both containers share /app via volume mount.
    signal_file = "/app/.e2e_tests_done"
    # Clean up stale signal from previous runs
    if os.path.exists(signal_file):
        os.remove(signal_file)

    max_wait = 600  # 10 minutes absolute safety net
    print(f"[Seeder] Seeding until downloader signals completion (max {max_wait}s)...", flush=True)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if os.path.exists(signal_file):
            print("[Seeder] Downloader signaled completion. Stopping.", flush=True)
            break
        time.sleep(5)
    else:
        print(f"[Seeder] Timed out after {max_wait}s waiting for downloader signal.", flush=True)

    # ── 5. Clean up ──
    print("[Seeder] Stopping daemon...", flush=True)
    stop_daemon()
    # Remove signal file so it doesn't affect next run
    if os.path.exists(signal_file):
        os.remove(signal_file)
    print("[Seeder] ✅ Seeding session completed.", flush=True)


if __name__ == "__main__":
    test_seeder_initialization()
