"""
Seeder entry point for breakpoint-resume E2E tests.

Downloads a full small repository via HTTP, starts the daemon, waits for
torrent registration, then keeps seeding until the downloader test suite
signals completion. This matches the current product's whole-repository
seeding semantics.
"""

from __future__ import annotations

import os
import time

import requests
from huggingface_hub import snapshot_download


READY_SIGNAL = "/app/.resume_seeder_ready"
DONE_SIGNAL = "/app/.resume_e2e_done"


def _wait_for_torrent(tracker_url: str, repo_id: str, timeout: int = 600) -> None:
    api_url = f"{tracker_url.rstrip('/')}/api/v1/torrents"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                body = response.json()
                torrents = body.get("data", body) if isinstance(body, dict) else body
                for torrent in torrents:
                    if torrent.get("repo_id") == repo_id:
                        print(
                            f"[Resume Seeder] Torrent registered: "
                            f"{torrent.get('info_hash', '?')[:16]}...",
                            flush=True,
                        )
                        return
        except Exception as exc:  # noqa: BLE001
            print(f"[Resume Seeder] Tracker poll failed: {exc}", flush=True)
        time.sleep(5)
    raise AssertionError(f"Timed out waiting for torrent registration: {repo_id}")


def test_resume_seeder_initialization():
    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    repo_id = os.environ.get("RESUME_REPO_ID", "prajjwal1/bert-tiny")
    revision = os.environ.get("RESUME_REVISION", "main")

    for signal_file in (READY_SIGNAL, DONE_SIGNAL):
        if os.path.exists(signal_file):
            os.unlink(signal_file)

    print(f"[Resume Seeder] Downloading full repo via HTTP: {repo_id}", flush=True)
    local_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_files_only=False,
        force_download=True,
    )
    assert os.path.exists(local_path), f"snapshot_download returned non-existent path: {local_path}"

    from llmpt.daemon import start_daemon, stop_daemon

    port = int(os.environ.get("HF_P2P_PORT", 0)) or None
    pid = start_daemon(tracker_url=tracker_url, port=port, foreground=False)
    assert pid is not None, "Failed to start resume seeder daemon"
    print(f"[Resume Seeder] Daemon started (PID: {pid})", flush=True)

    _wait_for_torrent(tracker_url, repo_id)
    with open(READY_SIGNAL, "w", encoding="utf-8") as f:
        f.write("ready")

    print("[Resume Seeder] Waiting for downloader completion signal...", flush=True)
    deadline = time.time() + 1800
    while time.time() < deadline:
        if os.path.exists(DONE_SIGNAL):
            print("[Resume Seeder] Downloader signaled completion.", flush=True)
            break
        time.sleep(5)
    else:
        raise AssertionError("Timed out waiting for downloader completion signal")

    stop_daemon()

    for signal_file in (READY_SIGNAL, DONE_SIGNAL):
        if os.path.exists(signal_file):
            os.unlink(signal_file)


if __name__ == "__main__":
    test_resume_seeder_initialization()
