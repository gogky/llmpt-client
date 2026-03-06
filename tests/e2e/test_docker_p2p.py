"""
End-to-End P2P Download Tests.

Provides three test scenarios:
  1. test_true_p2p_download      — Pure P2P (WebSeed disabled), all files via seeder
  2. test_webseed_only_download  — No seeder, all files via WebSeed proxy → HF CDN
  3. test_webseed_disabled       — Verify WebSeed proxy is NOT started when disabled

Run via Docker Compose:
  Pure P2P:   docker compose -f docker-compose.test.yml up --build
  WebSeed:    docker compose -f docker-compose.test-webseed.yml up --build
"""

import os
import time
import pytest
from huggingface_hub import snapshot_download
import requests
import llmpt
import shutil


# ─── Shared constants ────────────────────────────────────────────────────────

REPO_ID = "hf-internal-testing/tiny-random-GPTJForCausalLM"
EXPECTED_FILES = {"config.json", "pytorch_model.bin", "tokenizer.json"}


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _wait_for_seeder_ready(tracker_url: str, repo_id: str, timeout: int = 180) -> bool:
    """Poll the tracker until the seeder has registered a torrent for repo_id.

    This replaces the old fixed time.sleep(30): instead of guessing how long
    seeder setup takes, we poll the tracker API and return as soon as the
    seeder is confirmed registered.  This makes the test deterministic and
    independent of machine/network speed.

    Returns True if the torrent appears within *timeout* seconds, False otherwise.
    """
    api_url = f"{tracker_url.rstrip('/')}/api/v1/torrents"
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            resp = requests.get(api_url, timeout=5)
            if resp.status_code == 200:
                body = resp.json()
                # Tracker returns {"data": [...], "total": N}
                torrents = body.get("data", body) if isinstance(body, dict) else body
                for t in torrents:
                    if t.get("repo_id") == repo_id:
                        print(
                            f"[Downloader] Seeder registered torrent after {attempt} poll(s). "
                            f"info_hash={t.get('info_hash', '?')[:16]}...",
                            flush=True,
                        )
                        return True
        except Exception as e:
            print(f"[Downloader] Tracker poll #{attempt} failed: {e}", flush=True)
        time.sleep(5)

    print(f"[Downloader] Seeder did NOT register within {timeout}s.", flush=True)
    return False


def _collect_files(local_path: str) -> set:
    """Walk a snapshot directory and return a set of relative file paths."""
    files_in_repo = set()
    for root, dirs, files in os.walk(local_path):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), local_path)
            files_in_repo.add(rel)
    return files_in_repo


def _print_download_report(stats: dict, files_in_repo: set, label: str = "P2P") -> None:
    """Print a formatted download report for diagnostics."""
    p2p_files = stats['p2p']
    http_files = stats['http']
    tracked_files = p2p_files | http_files
    untracked = files_in_repo - tracked_files

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

    if untracked:
        print(f"\n  ⚠️  Untracked files (bypassed http_get entirely!):", flush=True)
        for f in sorted(untracked):
            print(f"     ? {f}", flush=True)

    print(f"\n{'='*60}\n", flush=True)


def _assert_all_p2p(stats: dict, files_in_repo: set) -> None:
    """Assert that ALL files were downloaded via P2P with zero HTTP fallbacks."""
    p2p_files = stats['p2p']
    http_files = stats['http']
    tracked_files = p2p_files | http_files
    untracked = files_in_repo - tracked_files

    assert len(p2p_files) == len(files_in_repo), (
        f"Not all files were downloaded via P2P!\n"
        f"  P2P: {len(p2p_files)}/{len(files_in_repo)}\n"
        f"  HTTP fallbacks: {sorted(http_files) if http_files else 'none'}\n"
        f"  Untracked (bypassed http_get): {sorted(untracked) if untracked else 'none'}"
    )

    assert len(http_files) == 0, (
        f"Some files fell back to HTTP instead of P2P!\n"
        f"  HTTP fallback files: {sorted(http_files)}"
    )


# ─── Test 1: Pure P2P (no WebSeed) ──────────────────────────────────────────

def test_true_p2p_download():
    """
    Downloads the model that the Seeder container is currently hosting.
    Asserts that every single file goes through P2P — no HTTP fallbacks.

    WebSeed is explicitly DISABLED to ensure we're testing the pure P2P path.
    If WebSeed were on, it would act as an invisible fallback and mask
    P2P connectivity issues.
    """
    # Give containers a moment to fully start up before polling
    time.sleep(15)

    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    llmpt.enable_p2p(tracker_url=tracker_url, timeout=60, webseed=False)

    # Reset download stats before the test
    from llmpt.patch import reset_download_stats, get_download_stats
    reset_download_stats()

    # Deterministically wait until the seeder has registered the torrent.
    print("[Downloader] Polling tracker until seeder is registered...", flush=True)
    seeder_ready = _wait_for_seeder_ready(tracker_url, REPO_ID, timeout=180)
    assert seeder_ready, (
        f"Seeder never registered torrent for {REPO_ID} within 180s. "
        "Check seeder container logs for errors."
    )

    print("[Downloader] Requesting snapshot_download...", flush=True)
    local_path = snapshot_download(
        repo_id=REPO_ID,
        local_files_only=False,
        force_download=True,  # Must bypass HF local cache to ensure http_get is always called
    )

    # ── 1. Basic file existence checks ──
    assert os.path.exists(local_path)
    files_in_repo = _collect_files(local_path)

    missing = EXPECTED_FILES - files_in_repo
    assert not missing, f"Core files missing from snapshot: {missing}"

    # ── 2. Download report and assertions ──
    stats = get_download_stats()
    _print_download_report(stats, files_in_repo, label="Pure P2P")
    _assert_all_p2p(stats, files_in_repo)

    # ── 3. Verify WebSeed was NOT active ──
    config = llmpt.get_config()
    assert config.get('webseed_proxy_port') is None, (
        "WebSeed proxy port should be None when webseed=False"
    )

    print(f"[Downloader] ✅ All {len(stats['p2p'])} files downloaded via pure P2P to {local_path}!", flush=True)


# ─── Test 2: WebSeed only (no seeder peer) ────────────────────────────────

def test_webseed_only_download():
    """
    Cold-start scenario: no P2P seeder available, WebSeed is the sole source.

    The WebSeed proxy translates libtorrent's piece requests into HTTP Range
    requests to the HuggingFace CDN. From libtorrent's perspective, the
    WebSeed acts as an always-online "seeder".

    Since we still need a .torrent from the tracker, this test first polls
    the tracker to ensure a torrent is registered (possibly by a prior
    seeder run or pre-registered data).

    NOTE: This test should be run WITHOUT a TEST_SEEDER_PEER env var,
    so the downloader has no P2P peers and relies entirely on WebSeed.
    """
    # Give containers a moment to fully start up before polling
    time.sleep(15)

    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    llmpt.enable_p2p(tracker_url=tracker_url, timeout=120, webseed=True)

    # Verify WebSeed proxy is actually running
    config = llmpt.get_config()
    assert config.get('webseed_proxy_port') is not None, (
        "WebSeed proxy should be running when webseed=True"
    )
    print(
        f"[Downloader] WebSeed proxy running on port {config['webseed_proxy_port']}",
        flush=True,
    )

    from llmpt.patch import reset_download_stats, get_download_stats
    reset_download_stats()

    # Wait for torrent to be registered on tracker.
    # In a real cold-start, the first node creates and registers the torrent.
    # In this test, we rely on a torrent that was previously registered.
    print("[Downloader] Polling tracker for torrent registration...", flush=True)
    torrent_available = _wait_for_seeder_ready(tracker_url, REPO_ID, timeout=180)
    assert torrent_available, (
        f"No torrent registered for {REPO_ID} within 180s. "
        "WebSeed test requires a torrent on the tracker (run seeder first)."
    )

    print("[Downloader] Requesting snapshot_download (WebSeed only)...", flush=True)
    local_path = snapshot_download(
        repo_id=REPO_ID,
        local_files_only=False,
        force_download=True,
    )

    # ── 1. Basic file existence checks ──
    assert os.path.exists(local_path)
    files_in_repo = _collect_files(local_path)

    missing = EXPECTED_FILES - files_in_repo
    assert not missing, f"Core files missing from snapshot: {missing}"

    # ── 2. Download report and assertions ──
    stats = get_download_stats()
    _print_download_report(stats, files_in_repo, label="WebSeed-Only")
    _assert_all_p2p(stats, files_in_repo)

    print(
        f"[Downloader] ✅ All {len(stats['p2p'])} files downloaded via WebSeed to {local_path}!",
        flush=True,
    )


# ─── Test 3: WebSeed explicitly disabled ─────────────────────────────────────

def test_webseed_disabled():
    """
    Verify that when webseed=False is passed to enable_p2p(), the WebSeed
    proxy is NOT started and the config reflects this.

    This is a lightweight config-validation test (no actual download needed).
    """
    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    llmpt.enable_p2p(tracker_url=tracker_url, timeout=10, webseed=False)

    config = llmpt.get_config()

    # Proxy port must be None
    assert config.get('webseed_proxy_port') is None, (
        f"Expected webseed_proxy_port=None, got {config.get('webseed_proxy_port')}"
    )

    # Config must reflect the user's choice
    assert config.get('webseed') is False, (
        f"Expected webseed=False in config, got {config.get('webseed')}"
    )

    print("[Test] ✅ Confirmed WebSeed proxy is NOT running when webseed=False", flush=True)


# ─── Test 4: WebSeed + P2P hybrid mode ───────────────────────────────────────

def test_webseed_with_p2p_download():
    """
    Hybrid scenario: both a P2P seeder AND WebSeed are active simultaneously.

    libtorrent will intelligently schedule piece requests between the P2P
    peer and the WebSeed HTTP source. When the P2P swarm is healthy, WebSeed
    is barely used; when a peer is slow or has missing pieces, WebSeed fills
    the gaps.

    This test verifies that:
    1. WebSeed proxy is running alongside P2P
    2. All files are downloaded successfully
    3. The download completes without HTTP fallback
    """
    # Give containers a moment to fully start up before polling
    time.sleep(15)

    tracker_url = os.environ.get("TRACKER_URL", "http://118.195.159.242")
    llmpt.enable_p2p(tracker_url=tracker_url, timeout=60, webseed=True)

    # Verify WebSeed proxy is running
    config = llmpt.get_config()
    assert config.get('webseed_proxy_port') is not None, (
        "WebSeed proxy should be running when webseed=True"
    )
    print(
        f"[Downloader] Hybrid mode: P2P + WebSeed (port {config['webseed_proxy_port']})",
        flush=True,
    )

    from llmpt.patch import reset_download_stats, get_download_stats
    reset_download_stats()

    # Wait for seeder to register the torrent
    print("[Downloader] Polling tracker until seeder is registered...", flush=True)
    seeder_ready = _wait_for_seeder_ready(tracker_url, REPO_ID, timeout=180)
    assert seeder_ready, (
        f"Seeder never registered torrent for {REPO_ID} within 180s. "
        "Check seeder container logs for errors."
    )

    print("[Downloader] Requesting snapshot_download (hybrid: P2P + WebSeed)...", flush=True)
    local_path = snapshot_download(
        repo_id=REPO_ID,
        local_files_only=False,
        force_download=True,
    )

    # ── 1. Basic file existence checks ──
    assert os.path.exists(local_path)
    files_in_repo = _collect_files(local_path)

    missing = EXPECTED_FILES - files_in_repo
    assert not missing, f"Core files missing from snapshot: {missing}"

    # ── 2. Download report and assertions ──
    stats = get_download_stats()
    _print_download_report(stats, files_in_repo, label="Hybrid (P2P + WebSeed)")
    _assert_all_p2p(stats, files_in_repo)

    print(
        f"[Downloader] ✅ All {len(stats['p2p'])} files downloaded via hybrid mode to {local_path}!",
        flush=True,
    )

