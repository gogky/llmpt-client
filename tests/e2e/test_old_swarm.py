"""
Docker E2E tests for old-swarm reuse.

This scenario verifies the cross-revision file-level fallback path:

1. The exact target revision is registered on the tracker but has zero seeders.
2. An older revision with the same file_root+size for the target file is
   actively seeding.
3. The client chooses the old revision as the source.
4. WebSeed is disabled, and runtime libtorrent stats confirm that the transfer
   came from true P2P peers rather than URL seeds.
"""

import os
import time

import requests
from huggingface_hub import snapshot_download

import llmpt
from llmpt.p2p_batch import P2PBatchManager
from llmpt.patch import get_download_stats, reset_download_stats
from llmpt.tracker_client import TrackerClient
from llmpt.transfer_coordinator import TransferCoordinator


REPO_ID = "fka/prompts.chat"
REPO_TYPE = "dataset"
OLD_REV = "7b8336efefee6e5f8fa3accf3b142151ff8fee1e"
NEW_REV = "e88423787cb39928434f2ef4ea678509dd7b4426"
TARGET_FILE = "README.md"
SIGNAL_FILE = "/app/.old_swarm_e2e_done"


def _wait_for_old_swarm_candidates(tracker_url: str, timeout: int = 300) -> dict:
    deadline = time.time() + timeout
    url = f"{tracker_url.rstrip('/')}/api/v1/file-sources"
    while time.time() < deadline:
        resp = requests.get(
            url,
            params={
                "repo_id": REPO_ID,
                "repo_type": REPO_TYPE,
                "revision": NEW_REV,
                "path": TARGET_FILE,
            },
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()["data"]
        candidates = body.get("candidates", [])
        exact = next(
            (
                c
                for c in candidates
                if c["revision"] == NEW_REV and c["path"] == TARGET_FILE
            ),
            None,
        )
        legacy = next((c for c in candidates if c["revision"] == OLD_REV), None)
        if exact and legacy and exact.get("seeders") == 0 and legacy.get("seeders", 0) > 0:
            print(
                f"[old_swarm_downloader] file-sources ready: exact={exact} legacy={legacy}",
                flush=True,
            )
            return body
        print(
            f"[old_swarm_downloader] waiting for expected candidates, current={body}",
            flush=True,
        )
        time.sleep(5)
    raise RuntimeError("timed out waiting for old-swarm candidate set")


def _get_old_revision_session():
    manager = P2PBatchManager()
    matches = [ctx for key, ctx in manager.sessions.items() if key.revision == OLD_REV]
    assert matches, "expected an active old-revision session"
    assert not any(key.revision == NEW_REV for key in manager.sessions.keys()), (
        "unexpected exact-target session present; old-swarm fallback should be using "
        "only the legacy source session"
    )
    return matches[0]


def test_old_swarm_single_file_without_webseed():
    """Download one unchanged file from an older revision with WebSeed disabled."""
    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    if os.path.exists(SIGNAL_FILE):
        os.remove(SIGNAL_FILE)

    llmpt.enable_p2p(tracker_url=tracker_url, timeout=120, webseed=False)

    config = llmpt.get_config()
    assert config.get("webseed_proxy_port") is None, (
        "WebSeed proxy should not start in the old-swarm pure-P2P test"
    )

    body = _wait_for_old_swarm_candidates(tracker_url)
    tracker = TrackerClient(tracker_url)
    coordinator = TransferCoordinator()
    target = coordinator.build_target_request(
        repo_id=REPO_ID,
        revision=NEW_REV,
        repo_type=REPO_TYPE,
        filename=TARGET_FILE,
        destination="/tmp/old-swarm-target",
    )
    plan = coordinator.plan_request(target, tracker_client=tracker)
    print(
        f"[old_swarm_downloader] planned source revision={plan.source.revision} "
        f"source_file={plan.source_filename}",
        flush=True,
    )
    assert plan.source.revision == OLD_REV, (
        f"expected old revision {OLD_REV}, got {plan.source.revision}; body={body}"
    )

    reset_download_stats()
    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=NEW_REV,
        allow_patterns=[TARGET_FILE],
        local_files_only=False,
        force_download=True,
    )
    target_path = os.path.join(local_path, TARGET_FILE)
    assert os.path.exists(target_path), f"target file not downloaded: {target_path}"

    stats = get_download_stats()
    print(f"[old_swarm_downloader] stats={stats}", flush=True)
    assert TARGET_FILE in stats["p2p"], "target file should be delivered via the P2P path"
    assert TARGET_FILE not in stats["http"], "target file should not fall back to HTTP"

    old_ctx = _get_old_revision_session()
    p2p_stats = old_ctx.get_p2p_stats()
    print(f"[old_swarm_downloader] old session stats={p2p_stats}", flush=True)
    assert p2p_stats, "expected runtime transfer stats on the old revision session"
    assert p2p_stats.get("num_webseeds", 0) == 0, (
        "old-swarm transfer should not attach any WebSeed peers"
    )
    assert p2p_stats.get("webseed_download", 0) == 0, (
        "old-swarm transfer should not download any bytes via WebSeed"
    )
    assert p2p_stats.get("peak_p2p_peers", 0) > 0, (
        "expected at least one real P2P peer to connect during the transfer"
    )
    assert p2p_stats.get("total_payload_download", 0) > 0, (
        "expected positive payload download on the old swarm session"
    )

    with open(SIGNAL_FILE, "w", encoding="utf-8") as fh:
        fh.write("done")
    print("[old_swarm_downloader] success", flush=True)
