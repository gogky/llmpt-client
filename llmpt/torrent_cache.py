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
import hashlib
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from .tracker_client import TrackerClient

logger = logging.getLogger("llmpt.torrent_cache")

TORRENT_CACHE_DIR = os.path.expanduser("~/.cache/llmpt/torrents")


def _mark_local_torrent_safe(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    present: bool,
) -> None:
    try:
        from .torrent_state import mark_local_torrent

        mark_local_torrent(repo_id, revision, repo_type=repo_type, present=present)
    except Exception as exc:
        logger.warning(
            f"[{repo_id}] Failed to persist local torrent state "
            f"(present={present}) for revision {revision[:8]}...: {exc}"
        )


def _mark_tracker_registration_safe(
    repo_id: str,
    revision: str,
    tracker_url: str,
    *,
    repo_type: str = "model",
    registered: bool,
) -> None:
    try:
        from .torrent_state import mark_tracker_registration

        mark_tracker_registration(
            repo_id,
            revision,
            repo_type=repo_type,
            tracker_url=tracker_url,
            registered=registered,
        )
    except Exception as exc:
        logger.warning(
            f"[{repo_id}] Failed to persist tracker registration state "
            f"(registered={registered}) for revision {revision[:8]}...: {exc}"
        )


def _cache_path(repo_id: str, revision: str, repo_type: str = "model") -> str:
    """Return the filesystem path for a cached .torrent file."""
    repo_digest = hashlib.sha1(repo_id.encode("utf-8")).hexdigest()[:16]
    return os.path.join(
        TORRENT_CACHE_DIR,
        f"{repo_type}_{repo_digest}_{revision}.torrent",
    )


def _safe_identity(repo_id: str, revision: str, repo_type: str = "model") -> tuple[str, str, str]:
    """Return a filesystem-safe torrent cache identity."""
    return (
        repo_type,
        hashlib.sha1(repo_id.encode("utf-8")).hexdigest()[:16],
        revision,
    )


def _parse_cached_torrent_name(filename: str) -> Optional[tuple[str, str, str]]:
    """Parse a cached torrent filename into ``(repo_type, repo_digest, revision)``."""
    if not filename.endswith(".torrent"):
        return None

    stem = filename[: -len(".torrent")]
    first_sep = stem.find("_")
    if first_sep <= 0:
        return None

    repo_type = stem[:first_sep]
    if repo_type not in {"model", "dataset", "space"}:
        return None

    remainder = stem[first_sep + 1 :]
    last_sep = remainder.rfind("_")
    if last_sep <= 0:
        return None

    repo_digest = remainder[:last_sep]
    revision = remainder[last_sep + 1 :]
    if not repo_digest or not revision:
        return None

    return repo_type, repo_digest, revision


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
            _mark_local_torrent_safe(
                repo_id,
                revision,
                repo_type=repo_type,
                present=True,
            )
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
        _mark_local_torrent_safe(
            repo_id,
            revision,
            repo_type=repo_type,
            present=True,
        )
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


def delete_cached_torrent(
    repo_id: str, revision: str, *, repo_type: str = "model"
) -> bool:
    """Remove a cached .torrent file if present."""
    path = _cache_path(repo_id, revision, repo_type)
    removed = False
    try:
        if os.path.exists(path):
            os.unlink(path)
            removed = True
    except OSError as exc:
        logger.warning(f"[{repo_id}] Failed to delete cached torrent: {exc}")
        return False

    _mark_local_torrent_safe(
        repo_id,
        revision,
        repo_type=repo_type,
        present=False,
    )
    return removed


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
        _mark_tracker_registration_safe(
            repo_id,
            revision,
            tracker_client.tracker_url,
            repo_type=repo_type,
            registered=True,
        )
        return torrent_data

    # Layer 3: not found
    logger.info(f"[{repo_id}] No torrent found in cache or tracker")
    return None


def cleanup_torrent_cache(
    protected: Iterable[tuple[str, str, str]],
) -> dict:
    """Delete cached torrents that are not needed for active or verified sources.

    Args:
        protected: Iterable of ``(repo_type, repo_id, revision)`` identities that
            must be retained. These typically come from active seeding sessions
            and verified completed sources.

    Returns:
        Summary dict containing counts for removed/kept/skipped entries.
    """
    summary = {
        "removed_torrents": 0,
        "removed_tmp_files": 0,
        "kept_torrents": 0,
        "skipped_unparsed": 0,
        "errors": 0,
    }
    if not os.path.isdir(TORRENT_CACHE_DIR):
        return summary

    protected_safe = {
        _safe_identity(repo_id, revision, repo_type)
        for repo_type, repo_id, revision in protected
    }

    state_entries = []
    try:
        from .torrent_state import load_all_torrent_states

        state_entries = load_all_torrent_states()
    except Exception as exc:
        logger.warning(f"Failed to load torrent state for cache cleanup: {exc}")

    state_map: dict[tuple[str, str, str], list[dict]] = {}
    for entry in state_entries:
        repo_type = entry.get("repo_type", "model")
        repo_id = entry.get("repo_id")
        revision = entry.get("revision")
        if not repo_id or not revision:
            continue
        key = _safe_identity(repo_id, revision, repo_type)
        state_map.setdefault(key, []).append(entry)

    for filename in os.listdir(TORRENT_CACHE_DIR):
        path = os.path.join(TORRENT_CACHE_DIR, filename)
        if not os.path.isfile(path):
            continue

        if filename.endswith(".tmp"):
            try:
                os.unlink(path)
                summary["removed_tmp_files"] += 1
            except OSError as exc:
                logger.warning(f"Failed to remove cached torrent temp file {filename}: {exc}")
                summary["errors"] += 1
            continue

        parsed = _parse_cached_torrent_name(filename)
        if parsed is None:
            summary["skipped_unparsed"] += 1
            continue

        if parsed in protected_safe:
            summary["kept_torrents"] += 1
            continue

        try:
            os.unlink(path)
            summary["removed_torrents"] += 1
        except OSError as exc:
            logger.warning(f"Failed to remove cached torrent {filename}: {exc}")
            summary["errors"] += 1
            continue

        matched_entries = state_map.get(parsed, [])
        if len(matched_entries) == 1:
            entry = matched_entries[0]
            _mark_local_torrent_safe(
                entry["repo_id"],
                entry["revision"],
                repo_type=entry.get("repo_type", "model"),
                present=False,
            )

    return summary
