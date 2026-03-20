"""
Old-swarm seeder harness for Docker E2E tests.

This container downloads a specific historical revision, starts the daemon,
and keeps that old revision actively seeding until the downloader signals
completion.
"""

import os
import time

import requests
from huggingface_hub import snapshot_download

from llmpt.daemon import start_daemon, stop_daemon


TRACKER_URL = os.environ["TRACKER_URL"].rstrip("/")
REPO_ID = os.environ["REPO_ID"]
REPO_TYPE = os.environ.get("REPO_TYPE", "dataset")
OLD_REV = os.environ["OLD_REV"]
SIGNAL_FILE = os.environ.get("OLD_SWARM_SIGNAL_FILE", "/app/.old_swarm_e2e_done")


def wait_for_torrent(revision: str, timeout: int = 240) -> None:
    deadline = time.time() + timeout
    api_url = f"{TRACKER_URL}/api/v1/torrents"
    while time.time() < deadline:
        resp = requests.get(
            api_url,
            params={
                "repo_id": REPO_ID,
                "repo_type": REPO_TYPE,
                "revision": revision,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            print(f"[old_swarm_seeder] torrent registered for {revision}", flush=True)
            return
        time.sleep(5)
    raise RuntimeError(f"torrent for {revision} was not registered in time")


def main() -> None:
    if os.path.exists(SIGNAL_FILE):
        os.remove(SIGNAL_FILE)

    print(f"[old_swarm_seeder] downloading {REPO_ID}@{OLD_REV}", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=OLD_REV,
        local_files_only=False,
        force_download=True,
    )

    port = int(os.environ.get("HF_P2P_PORT", "6881"))
    pid = start_daemon(tracker_url=TRACKER_URL, port=port, foreground=False)
    if not pid:
        raise RuntimeError("failed to start daemon")
    print(f"[old_swarm_seeder] daemon pid={pid}", flush=True)

    try:
        wait_for_torrent(OLD_REV)
        deadline = time.time() + 900
        while time.time() < deadline:
            if os.path.exists(SIGNAL_FILE):
                print("[old_swarm_seeder] received completion signal", flush=True)
                return
            time.sleep(5)
        raise RuntimeError("timed out waiting for downloader completion signal")
    finally:
        print("[old_swarm_seeder] stopping daemon", flush=True)
        stop_daemon()
        if os.path.exists(SIGNAL_FILE):
            os.remove(SIGNAL_FILE)


if __name__ == "__main__":
    main()
