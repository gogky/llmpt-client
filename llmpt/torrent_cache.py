"""
Local .torrent file cache.

Provides a three-layer torrent resolution strategy:
  1. Local disk cache  (0 latency)
  2. Tracker server    (~1s latency)
  3. Return None       (caller decides: generate locally or give up)

Cache layout::

    ~/.cache/llmpt/torrents/
        meta-llama_Llama-2-7b_abc123def456...torrent
        gpt2_9a7cb3e...torrent

The cache key is ``(repo_id, revision)`` where *revision* is always
a 40-char commit hash.  Because piece_length is deterministically
computed from total_size, the same ``repo_id@revision`` always
produces the same torrent — so cached entries **never expire**.

Writes use atomic ``write-to-tmp + os.replace`` to prevent
half-written .torrent files from corrupting the cache.
"""

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .tracker_client import TrackerClient

logger = logging.getLogger("llmpt.torrent_cache")

TORRENT_CACHE_DIR = os.path.expanduser("~/.cache/llmpt/torrents")


def _cache_path(repo_id: str, revision: str, repo_type: str = "model") -> str:
    """Return the filesystem path for a cached .torrent file."""
    safe_repo = f"{repo_type}_{repo_id.replace('/', '_')}"
    return os.path.join(TORRENT_CACHE_DIR, f"{safe_repo}_{revision}.torrent")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cached_torrent(repo_id: str, revision: str, *, repo_type: str = "model") -> Optional[bytes]:
    """Load a .torrent from the local disk cache.

    Returns:
        Raw torrent bytes, or *None* if no cached entry exists.
    """
    path = _cache_path(repo_id, revision, repo_type)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as f:
            data = f.read()
        if data:
            logger.info(
                f"[{repo_id}] Torrent loaded from local cache "
                f"({len(data)} bytes)"
            )
            return data
        # Empty file — treat as cache miss and clean up
        os.unlink(path)
    except OSError as exc:
        logger.warning(f"[{repo_id}] Failed to read cached torrent: {exc}")

    return None


def save_torrent_to_cache(
    repo_id: str, revision: str, torrent_data: bytes, *, repo_type: str = "model"
) -> None:
    """Persist .torrent data to the local disk cache.

    Uses atomic *write-then-rename* so concurrent readers never see
    a partially-written file.
    """
    os.makedirs(TORRENT_CACHE_DIR, exist_ok=True)
    path = _cache_path(repo_id, revision, repo_type)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(torrent_data)
        os.replace(tmp_path, path)  # atomic on POSIX
        logger.debug(
            f"[{repo_id}] Torrent cached locally ({len(torrent_data)} bytes)"
        )
    except OSError as exc:
        logger.warning(f"[{repo_id}] Failed to cache torrent: {exc}")
        # Best-effort cleanup of the temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def resolve_torrent_data(
    repo_id: str,
    revision: str,
    tracker_client: "TrackerClient",
    *,
    repo_type: str = "model",
) -> Optional[bytes]:
    """Three-layer .torrent resolution.

    1. **Local cache** — instant, no I/O beyond a single file read.
    2. **Tracker server** — HTTP download; result is cached locally
       for next time.
    3. **None** — caller decides whether to generate locally (seeder)
       or fall back to HTTP (downloader).
    """
    # Layer 1: local disk cache
    cached = load_cached_torrent(repo_id, revision, repo_type=repo_type)
    if cached is not None:
        return cached

    # Layer 2: tracker server
    torrent_data = tracker_client.download_torrent(repo_id, revision, repo_type=repo_type)
    if torrent_data:
        logger.info(
            f"[{repo_id}] Torrent downloaded from tracker, caching locally"
        )
        save_torrent_to_cache(repo_id, revision, torrent_data, repo_type=repo_type)
        return torrent_data

    # Layer 3: not found
    logger.info(f"[{repo_id}] No torrent found in cache or tracker")
    return None
