"""
Target-revision publisher for the old-swarm Docker E2E test.

This container registers the new revision torrent on the tracker but does not
start seeding it. That leaves the exact candidate present with zero seeders,
which is the condition we need in order to verify old-swarm reuse.
"""

import os
import time

import requests
from huggingface_hub import snapshot_download

from llmpt.torrent_creator import create_and_register_torrent
from llmpt.tracker_client import TrackerClient


TRACKER_URL = os.environ["TRACKER_URL"].rstrip("/")
REPO_ID = os.environ["REPO_ID"]
REPO_TYPE = os.environ.get("REPO_TYPE", "dataset")
NEW_REV = os.environ["NEW_REV"]
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
            print(f"[old_swarm_publisher] torrent registered for {revision}", flush=True)
            return
        time.sleep(5)
    raise RuntimeError(f"torrent for {revision} was not registered in time")


def main() -> None:
    print(f"[old_swarm_publisher] downloading {REPO_ID}@{NEW_REV}", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=NEW_REV,
        local_files_only=False,
        force_download=True,
    )

    tracker = TrackerClient(TRACKER_URL)
    info = create_and_register_torrent(
        repo_id=REPO_ID,
        revision=NEW_REV,
        repo_type=REPO_TYPE,
        name=REPO_ID,
        tracker_client=tracker,
    )
    if not info:
        raise RuntimeError("failed to create and register target revision torrent")

    print(
        f"[old_swarm_publisher] registered {NEW_REV} info_hash={info['info_hash'][:16]}...",
        flush=True,
    )
    wait_for_torrent(NEW_REV)

    deadline = time.time() + 900
    while time.time() < deadline:
        if os.path.exists(SIGNAL_FILE):
            print("[old_swarm_publisher] received completion signal", flush=True)
            return
        time.sleep(5)
    raise RuntimeError("timed out waiting for downloader completion signal")


if __name__ == "__main__":
    main()
