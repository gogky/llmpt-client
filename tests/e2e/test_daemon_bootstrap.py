"""
End-to-End test for daemon auto-seeding (cold start bootstrap).

This test verifies the complete cold-start scenario:
  1. Node A downloads a model via HTTP (no torrent exists on tracker)
  2. Node A's daemon auto-creates the torrent and registers it
  3. Node B downloads the same model — this time via P2P from Node A

This is the most critical E2E test: it proves that the P2P network
can bootstrap itself without any manual `llmpt-cli seed` intervention.

Run via Docker Compose:
    docker compose -f docker-compose.test-daemon.yml up --build
"""

import os
import time
import pytest
from huggingface_hub import snapshot_download
import requests
import llmpt
from llmpt.patch import reset_download_stats, get_download_stats


# ─── Constants ───────────────────────────────────────────────────────────────

REPO_ID = "hf-internal-testing/tiny-random-GPTJForCausalLM"
EXPECTED_FILES = {"config.json", "pytorch_model.bin", "tokenizer.json"}

DATASET_ID = "fka/prompts.chat"
EXPECTED_DATASET_FILES = {"prompts.csv"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _wait_for_torrent_on_tracker(tracker_url, repo_id, timeout=180):
    """Poll the tracker until a torrent for repo_id appears."""
    api_url = f"{tracker_url.rstrip('/')}/api/v1/torrents"
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            resp = requests.get(api_url, timeout=5)
            if resp.status_code == 200:
                body = resp.json()
                torrents = body.get("data", body) if isinstance(body, dict) else body
                for t in torrents:
                    if t.get("repo_id") == repo_id:
                        print(
                            f"[Test] Torrent found after {attempt} poll(s). "
                            f"info_hash={t.get('info_hash', '?')[:16]}...",
                            flush=True,
                        )
                        return True
        except Exception as e:
            print(f"[Test] Tracker poll #{attempt} failed: {e}", flush=True)
        time.sleep(5)
    return False


def _collect_files(local_path):
    """Walk a snapshot directory and return a set of relative file paths."""
    files_in_repo = set()
    for root, dirs, files in os.walk(local_path):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), local_path)
            files_in_repo.add(rel)
    return files_in_repo


def _print_download_report(stats, files_in_repo, label="P2P"):
    """Print a formatted download report for diagnostics."""
    p2p_files = stats['p2p']
    http_files = stats['http']

    print(f"\n{'='*60}", flush=True)
    print(f"  {label} Download Report", flush=True)
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

    print(f"\n{'='*60}\n", flush=True)


# ─── Test: Daemon cold-start bootstrap ───────────────────────────────────────

def test_daemon_cold_start_bootstrap():
    """
    Test the full cold-start bootstrap flow:
      1. Download model via plain HTTP (no P2P — this is the first user)
      2. Start the daemon, which scans the HF cache
      3. Daemon creates torrent and registers it on tracker
      4. Keep seeding so the second-user can download via P2P

    This simulates the "first user" experience — proving that the P2P network
    can self-bootstrap without manual intervention.

    NOTE: We do NOT call enable_p2p() here because:
      - The first user has no torrent to download from (cold start)
      - enable_p2p() creates a libtorrent session that competes for port 6881
      - We only need the daemon for seeding, not the download-side patches
    """
    time.sleep(10)

    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")

    # ── 0. Delete any stale torrent from previous test runs ──
    # Previous E2E tests may have registered torrents with different piece hashes.
    # The daemon would find those instead of creating a fresh one.
    api_url = f"{tracker_url.rstrip('/')}/api/v1/torrents"
    try:
        resp = requests.get(api_url, timeout=5)
        if resp.status_code == 200:
            body = resp.json()
            torrents = body.get("data", body) if isinstance(body, dict) else body
            for t in torrents:
                if t.get("repo_id") == REPO_ID:
                    delete_url = f"{tracker_url.rstrip('/')}/api/v1/torrents/{t['info_hash']}"
                    requests.delete(delete_url, timeout=5)
                    print(f"[Test] Deleted stale torrent {t['info_hash'][:16]}...", flush=True)
    except Exception as e:
        print(f"[Test] Could not clean tracker: {e}", flush=True)

    if os.path.exists("/app/.daemon_ready"):
        os.unlink("/app/.daemon_ready")

    # Also clear local torrent cache
    torrent_cache_dir = os.path.expanduser("~/.cache/llmpt/torrents")
    if os.path.exists(torrent_cache_dir):
        import shutil
        shutil.rmtree(torrent_cache_dir, ignore_errors=True)
        print("[Test] Cleared local torrent cache", flush=True)

    # ── 1. Download model via plain HTTP (no P2P patches needed) ──
    print(f"[Test] Downloading {REPO_ID} via plain HTTP (cold start)...", flush=True)
    local_path = snapshot_download(
        repo_id=REPO_ID,
        local_files_only=False,
        force_download=True,
    )

    assert os.path.exists(local_path)
    files_in_repo = _collect_files(local_path)
    missing = EXPECTED_FILES - files_in_repo
    assert not missing, f"Core files missing: {missing}"
    print(f"[Test] Downloaded {len(files_in_repo)} files to {local_path}", flush=True)

    # ── 2. Start the daemon (it will scan cache, create torrent, seed) ──
    from llmpt.daemon import start_daemon, stop_daemon, is_daemon_running

    port = int(os.environ.get("HF_P2P_PORT", 0)) or None
    print(f"[Test] Starting daemon (tracker: {tracker_url}, port: {port})...", flush=True)
    pid = start_daemon(tracker_url=tracker_url, port=port)
    assert pid is not None, "Failed to start daemon"
    print(f"[Test] Daemon started (PID: {pid})", flush=True)

    # ── 3. Wait for daemon to create and register the torrent ──
    print("[Test] Waiting for daemon to create torrent and register...", flush=True)
    torrent_registered = _wait_for_torrent_on_tracker(tracker_url, REPO_ID, timeout=180)

    assert torrent_registered, (
        f"Daemon did not register torrent for {REPO_ID} within 180s. "
        "The cold-start bootstrap failed!"
    )

    # ── 3.5 Wait for daemon to actually start seeding ──
    # (If the torrent was stale and not deleted from an external tracker, the above check may return immediately)
    print("[Test] Waiting for daemon to begin seeding...", flush=True)
    from llmpt.ipc import query_daemon
    seeding = False
    for _ in range(60):
        status = query_daemon("status")
        if status and status.get("seeding_count", 0) > 0:
            seeding = True
            break
        time.sleep(1)
    assert seeding, "Daemon did not start seeding within 60s."

    with open("/app/.daemon_ready", "w") as f:
        f.write("ready")

    print("[Test] ✅ Cold-start bootstrap successful!", flush=True)
    print(
        "  The daemon auto-detected the downloaded model, "
        "created a torrent, and registered it with the tracker.",
        flush=True,
    )

    # ── 4. Debug: dump daemon state ──
    # Print daemon log for diagnostics
    from llmpt.daemon import LOG_FILE
    try:
        with open(LOG_FILE) as f:
            log_content = f.read()
        print(f"\n{'='*60}", flush=True)
        print("  DAEMON LOG:", flush=True)
        print(f"{'='*60}", flush=True)
        for line in log_content.strip().split("\n"):
            print(f"  {line}", flush=True)
        print(f"{'='*60}\n", flush=True)
    except Exception as e:
        print(f"[Test] Could not read daemon log: {e}", flush=True)

    # Query daemon status via IPC
    status = query_daemon("status")
    print(f"[Test] Daemon IPC status: {status}", flush=True)

    # ── 5. Keep seeding so the second-user container can download via P2P ──
    # Docker kills all processes when the main process exits, so we must
    # keep pytest alive while the daemon seeds.
    print("[Test] Keeping alive to seed for second-user...", flush=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        if os.path.exists("/app/.second_user_done"):
            os.unlink("/app/.second_user_done")
            break
        time.sleep(1)


def test_daemon_p2p_download_after_bootstrap():
    """
    After the cold-start bootstrap (test above), verify that the SECOND
    downloader can use P2P to download the model.

    This test runs on a separate container that has the seeder's torrent
    available on the tracker.
    """
    time.sleep(15)

    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    llmpt.enable_p2p(tracker_url=tracker_url, timeout=60, webseed=False)

    reset_download_stats()

    # Wait for the seeder (first user / daemon) to have registered the torrent
    print("[Test] Polling tracker until torrent is registered...", flush=True)
    torrent_available = _wait_for_torrent_on_tracker(tracker_url, REPO_ID, timeout=180)
    assert torrent_available, f"No torrent for {REPO_ID} on tracker within 180s"

    print("[Test] Waiting for first-user to signal readiness...", flush=True)
    ready = False
    for _ in range(180):
        if os.path.exists("/app/.daemon_ready"):
            ready = True
            break
        time.sleep(1)
    assert ready, "first-user daemon did not become ready within 180s"

    # Download
    print(f"[Test] Downloading {REPO_ID} via P2P...", flush=True)
    local_path = snapshot_download(
        repo_id=REPO_ID,
        local_files_only=False,
        force_download=True,
    )

    assert os.path.exists(local_path)
    files_in_repo = _collect_files(local_path)
    missing = EXPECTED_FILES - files_in_repo
    assert not missing, f"Core files missing: {missing}"

    stats = get_download_stats()
    _print_download_report(stats, files_in_repo, label="Second User (P2P)")

    # All files should have gone through P2P this time
    assert len(stats['p2p']) == len(files_in_repo), (
        f"Not all files via P2P: {len(stats['p2p'])}/{len(files_in_repo)}\n"
        f"HTTP fallbacks: {sorted(stats['http'])}"
    )

    print(f"[Test] ✅ All {len(stats['p2p'])} files downloaded via P2P!", flush=True)

    with open("/app/.second_user_done", "w") as f:
        f.write("done")


def test_daemon_dataset_bootstrap():
    """
    Test the full cold-start bootstrap flow for a dataset:
      1. Download dataset via plain HTTP (no P2P)
      2. Start the daemon, which scans the HF dataset cache
      3. Daemon creates torrent and registers it on tracker
      4. Keep seeding so the second-user can download via P2P
    """
    time.sleep(10)

    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")

    # ── 0. Delete any stale torrent from previous test runs ──
    api_url = f"{tracker_url.rstrip('/')}/api/v1/torrents"
    try:
        resp = requests.get(api_url, timeout=5)
        if resp.status_code == 200:
            body = resp.json()
            torrents = body.get("data", body) if isinstance(body, dict) else body
            for t in torrents:
                if t.get("repo_id") == DATASET_ID:
                    delete_url = f"{tracker_url.rstrip('/')}/api/v1/torrents/{t['info_hash']}"
                    requests.delete(delete_url, timeout=5)
                    print(f"[Test] Deleted stale torrent {t['info_hash'][:16]}...", flush=True)
    except Exception as e:
        print(f"[Test] Could not clean tracker: {e}", flush=True)

    if os.path.exists("/app/.daemon_ready_dataset"):
        os.unlink("/app/.daemon_ready_dataset")

    # Also clear local torrent cache
    torrent_cache_dir = os.path.expanduser("~/.cache/llmpt/torrents")
    if os.path.exists(torrent_cache_dir):
        import shutil
        shutil.rmtree(torrent_cache_dir, ignore_errors=True)
        print("[Test] Cleared local torrent cache", flush=True)

    # ── 1. Download dataset via plain HTTP ──
    print(f"[Test] Downloading {DATASET_ID} via plain HTTP (cold start)...", flush=True)
    local_path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_files_only=False,
        force_download=True,
    )

    assert os.path.exists(local_path)
    files_in_repo = _collect_files(local_path)
    missing = EXPECTED_DATASET_FILES - files_in_repo
    assert not missing, f"Core files missing: {missing}"
    print(f"[Test] Downloaded {len(files_in_repo)} files to {local_path}", flush=True)

    # ── 2. Start the daemon (or reuse existing) and trigger a rescan ──
    from llmpt.daemon import start_daemon, stop_daemon

    port = int(os.environ.get("HF_P2P_PORT", 0)) or None
    print(f"[Test] Starting daemon (tracker: {tracker_url}, port: {port})...", flush=True)
    pid = start_daemon(tracker_url=tracker_url, port=port)
    assert pid is not None, "Failed to start daemon"
    print(f"[Test] Daemon started (PID: {pid})", flush=True)

    # The daemon may already be running from the model test and won't rescan
    # for 300s. Trigger an immediate rescan so it discovers the new dataset.
    from llmpt.ipc import query_daemon as _query_daemon
    scan_result = _query_daemon("scan")
    print(f"[Test] Triggered daemon rescan: {scan_result}", flush=True)
    # Give the daemon a moment to process the scan
    time.sleep(5)

    # ── 3. Wait for daemon to create and register the torrent ──
    print("[Test] Waiting for daemon to create torrent and register...", flush=True)
    torrent_registered = _wait_for_torrent_on_tracker(tracker_url, DATASET_ID, timeout=180)

    assert torrent_registered, (
        f"Daemon did not register torrent for {DATASET_ID} within 180s. "
        "The cold-start bootstrap failed!"
    )

    print("[Test] Waiting for daemon to begin seeding dataset...", flush=True)
    from llmpt.ipc import query_daemon
    seeding = False
    for _ in range(60):
        status = query_daemon("status")
        if status:
            sessions = status.get("sessions", {})
            # Check specifically for the dataset in sessions
            for key in sessions:
                if DATASET_ID in key:
                    seeding = True
                    break
        if seeding:
            break
        time.sleep(1)
    assert seeding, f"Daemon did not start seeding {DATASET_ID} within 60s. Sessions: {status.get('sessions', {}).keys() if status else 'N/A'}"

    with open("/app/.daemon_ready_dataset", "w") as f:
        f.write("ready")

    print("[Test] ✅ Cold-start bootstrap successful for dataset!", flush=True)

    print("[Test] Keeping alive to seed for second-user...", flush=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        if os.path.exists("/app/.second_user_dataset_done"):
            os.unlink("/app/.second_user_dataset_done")
            break
        time.sleep(1)


def test_daemon_p2p_dataset_download_after_bootstrap():
    """
    After the cold-start bootstrap (test above), verify that the SECOND
    downloader can use P2P to download the dataset.
    """
    time.sleep(15)

    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    llmpt.enable_p2p(tracker_url=tracker_url, timeout=60, webseed=False)

    reset_download_stats()

    print("[Test] Polling tracker until torrent is registered...", flush=True)
    torrent_available = _wait_for_torrent_on_tracker(tracker_url, DATASET_ID, timeout=180)
    assert torrent_available, f"No torrent for {DATASET_ID} on tracker within 180s"

    print("[Test] Waiting for first-user to signal readiness...", flush=True)
    ready = False
    for _ in range(180):
        if os.path.exists("/app/.daemon_ready_dataset"):
            ready = True
            break
        time.sleep(1)
    assert ready, "first-user daemon did not become ready within 180s"

    print(f"[Test] Downloading {DATASET_ID} via P2P...", flush=True)
    local_path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_files_only=False,
        force_download=True,
    )

    assert os.path.exists(local_path)
    files_in_repo = _collect_files(local_path)
    missing = EXPECTED_DATASET_FILES - files_in_repo
    assert not missing, f"Core files missing: {missing}"

    stats = get_download_stats()
    _print_download_report(stats, files_in_repo, label="Second User Dataset (P2P)")

    assert len(stats['p2p']) == len(files_in_repo), (
        f"Not all files via P2P: {len(stats['p2p'])}/{len(files_in_repo)}\n"
        f"HTTP fallbacks: {sorted(stats['http'])}"
    )

    print(f"[Test] ✅ All {len(stats['p2p'])} files downloaded via P2P for dataset!", flush=True)

    with open("/app/.second_user_dataset_done", "w") as f:
        f.write("done")
