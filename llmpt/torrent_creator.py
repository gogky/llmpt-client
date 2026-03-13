"""
Torrent creation utilities.
"""

import logging
import re
import os
from pathlib import Path
from typing import Optional, Any

from .utils import lt, LIBTORRENT_AVAILABLE, get_optimal_piece_length, format_bytes, strip_torrent_root

logger = logging.getLogger('llmpt.torrent_creator')
_COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{40}$")


def _is_padding_relative_path(relative_path: str) -> bool:
    """Return True when a libtorrent relative path refers to a padding file."""
    normalized = relative_path.replace("\\", "/")
    return normalized.startswith(".pad/") or "/.pad/" in normalized


def _extract_info_hash_metadata(info) -> dict[str, str]:
    """Extract the canonical application hash and announce key from torrent_info."""
    announce_key = str(info.info_hash())
    metadata = {"info_hash": announce_key, "announce_key": announce_key}

    try:
        hashes = info.info_hashes()
    except Exception:
        hashes = None

    if hashes is None:
        return metadata

    try:
        has_v2 = hashes.has_v2() if hasattr(hashes, "has_v2") else False
        if isinstance(has_v2, bool) and has_v2:
            full_v2 = str(hashes.v2)
            metadata["info_hash"] = full_v2
            metadata["info_hash_v2"] = full_v2
            return metadata
    except Exception:
        pass

    try:
        has_v1 = hashes.has_v1() if hasattr(hashes, "has_v1") else False
        if isinstance(has_v1, bool) and has_v1:
            metadata["info_hash_v1"] = str(hashes.v1)
    except Exception:
        pass

    return metadata


def _collect_payload_files(info) -> tuple[list[dict], int]:
    """Return only the real payload files from torrent_info, excluding .pad/ entries."""
    files = info.files()
    file_list: list[dict] = []
    total_size = 0

    for i in range(files.num_files()):
        lt_file_path = files.file_path(i)
        relative_path = strip_torrent_root(lt_file_path)
        if _is_padding_relative_path(relative_path):
            continue

        size = files.file_size(i)
        file_list.append({
            'path': relative_path,
            'size': size,
        })
        total_size += size

    return file_list, total_size


def _build_local_dir_file_storage(
    repo_id: str,
    revision: str,
    local_dir: str,
) -> tuple[Optional[Any], Optional[str]]:
    """Build file_storage for a verified local_dir source using its manifest only."""
    from .completed_registry import get_completed_manifest

    manifest = get_completed_manifest(
        repo_id,
        revision,
        local_dir=local_dir,
    )
    if not manifest:
        logger.error(
            f"[{repo_id}] No completed manifest available for local_dir source: {local_dir}"
        )
        return None, None

    source_root = Path(local_dir)
    # Piece hashing must read files from the real local_dir layout on disk.
    # We rewrite the torrent root to ``revision`` after generation so the
    # resulting torrent still matches the hub-cache swarm identity.
    torrent_root = source_root.name
    fs = lt.file_storage()

    for relative_path in manifest:
        file_path = source_root / relative_path
        if not file_path.exists() or not file_path.is_file():
            logger.error(
                f"[{repo_id}] Verified local_dir file missing while building torrent: {relative_path}"
            )
            return None, None
        fs.add_file(f"{torrent_root}/{relative_path}", file_path.stat().st_size)

    return fs, str(source_root.parent)


def _rewrite_torrent_root_name(
    torrent: Any,
    root_name: str,
) -> bool:
    """Rewrite the top-level name of a generated multi-file torrent."""
    if not isinstance(torrent, dict):
        return False

    info = torrent.get(b"info")
    if not isinstance(info, dict):
        return False

    encoded_name = str(root_name).encode("utf-8")
    info[b"name"] = encoded_name
    return True


def _expected_completed_files(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> Optional[list[dict]]:
    from .completed_registry import get_completed_manifest

    manifest = get_completed_manifest(
        repo_id,
        revision,
        repo_type=repo_type,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )
    if manifest is None:
        return None

    files = []
    if local_dir:
        base_dir = Path(local_dir)
        for relative_path in manifest:
            file_path = base_dir / relative_path
            if not file_path.exists() or not file_path.is_file():
                logger.warning(
                    f"[{repo_id}] Completed local_dir manifest no longer resolves: {relative_path}"
                )
                return []
            files.append({"path": relative_path, "size": file_path.stat().st_size})
    else:
        from huggingface_hub import try_to_load_from_cache

        lookup_repo_type = repo_type if repo_type != "model" else None
        for relative_path in manifest:
            resolved = try_to_load_from_cache(
                repo_id=repo_id,
                filename=relative_path,
                revision=revision,
                repo_type=lookup_repo_type,
                cache_dir=cache_dir,
            )
            if not resolved or not os.path.exists(resolved):
                logger.warning(
                    f"[{repo_id}] Completed cache manifest no longer resolves: {relative_path}"
                )
                return []
            files.append({"path": relative_path, "size": os.path.getsize(resolved)})

    files.sort(key=lambda item: item["path"])
    return files


def torrent_matches_completed_source(
    repo_id: str,
    revision: str,
    torrent_data: bytes,
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> bool:
    """Return True when a torrent matches the verified completed source manifest.

    If no completed manifest is available for this source, this check is skipped
    and only syntactic torrent parsing is required.
    """
    result = _torrent_data_to_result(torrent_data, repo_id)
    if result is None:
        return False

    if local_dir and _COMMIT_HASH_RE.match(revision or ""):
        actual_root = str(result.get("commit_hash", "") or "")
        if actual_root != revision:
            logger.warning(
                f"[{repo_id}] Rejecting stale local_dir torrent: "
                f"root name {actual_root!r} != revision {revision!r}"
            )
            return False

    expected_files = _expected_completed_files(
        repo_id,
        revision,
        repo_type=repo_type,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )
    if expected_files is None:
        return True

    actual_files = sorted(
        [
            {
                "path": str(item.get("path", "")).replace("\\", "/"),
                "size": int(item.get("size", 0) or 0),
            }
            for item in (result.get("files") or [])
        ],
        key=lambda item: item["path"],
    )

    if actual_files != expected_files:
        logger.warning(
            f"[{repo_id}] Rejecting stale torrent: files do not match completed manifest "
            f"(torrent={len(actual_files)} files, expected={len(expected_files)} files)"
        )
        return False

    expected_total_size = sum(item["size"] for item in expected_files)
    if result.get("file_size") != expected_total_size or result.get("num_files") != len(expected_files):
        logger.warning(
            f"[{repo_id}] Rejecting stale torrent: metadata mismatch "
            f"(torrent size={result.get('file_size')}, expected size={expected_total_size})"
        )
        return False

    return True


def _torrent_data_to_result(torrent_data: bytes, repo_id: str) -> Optional[dict]:
    """Build a torrent info result dict from raw cached .torrent bytes.

    This mirrors the return value of :func:`create_torrent` but skips the
    expensive ``set_piece_hashes`` step by parsing the already-generated
    bencode data.
    """
    try:
        info = lt.torrent_info(lt.bdecode(torrent_data))
        files = info.files()
        hash_metadata = _extract_info_hash_metadata(info)
        file_list, total_size = _collect_payload_files(info)

        # Extract the root folder name (= commit hash) from the first file path
        first_path = files.file_path(0).replace('\\', '/')
        commit_hash = first_path.split('/')[0] if '/' in first_path else ''

        result = {
            'file_size': total_size,
            'piece_length': info.piece_length(),
            'num_pieces': info.num_pieces(),
            'num_files': len(file_list),
            'torrent_data': torrent_data,
            'commit_hash': commit_hash,
            'files': file_list,
        }
        result.update(hash_metadata)
        return result
    except Exception as e:
        logger.warning(f"[{repo_id}] Failed to parse cached torrent: {e}")
        return None


def _normalized_result_commit_hash(
    result: dict,
    revision: str,
    *,
    local_dir: Optional[str] = None,
) -> str:
    """Prefer the explicit revision over the torrent root for local_dir torrents."""
    if local_dir or _COMMIT_HASH_RE.match(revision or ""):
        return revision
    return result.get("commit_hash", revision) or revision


def create_torrent(
    repo_id: str,
    revision: str,
    tracker_client: Any,
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> Optional[dict]:
    """
    Create a torrent file for an entire HuggingFace repository snapshot.

    The piece_length is automatically determined from the total repository
    size via ``get_optimal_piece_length()``.  This is intentionally **not**
    caller-configurable: everyone who creates a torrent for the same
    ``repo_id@revision`` must get the same ``info_hash``, otherwise users
    would be split into separate swarms and unable to share data.

    Args:
        repo_id: HuggingFace repository ID (e.g., meta-llama/Llama-2-7b).
        revision: Git commit hash or branch name.
        tracker_client: TrackerClient instance to check for existing torrents and get tracker URL.

    Returns:
        Dictionary containing torrent info (info_hash, torrent_data, files, etc.)
        or None if creation failed.
    """
    if not LIBTORRENT_AVAILABLE:
        logger.error("libtorrent not available")
        return None

    try:
        from huggingface_hub import snapshot_download
        from .torrent_cache import (
            delete_cached_torrent,
            resolve_torrent_data,
            save_torrent_to_cache,
        )

        # ── Layer 1 & 2: check local .torrent cache AND tracker ───────────
        # If we or another seeder already generated a torrent for this repo@revision,
        # reuse it. This skips the expensive set_piece_hashes() call which can take
        # 30+ minutes for large models on HDD.
        cached_or_downloaded = resolve_torrent_data(repo_id, revision, tracker_client, repo_type=repo_type)
        if cached_or_downloaded:
            result = None
            if torrent_matches_completed_source(
                repo_id,
                revision,
                cached_or_downloaded,
                repo_type=repo_type,
                cache_dir=cache_dir,
                local_dir=local_dir,
            ):
                logger.info(f"Using existing torrent for {repo_id}@{revision} (from cache or tracker)")
                result = _torrent_data_to_result(cached_or_downloaded, repo_id)
                if result is not None:
                    result["commit_hash"] = _normalized_result_commit_hash(
                        result,
                        revision,
                        local_dir=local_dir,
                    )
                    return result
            else:
                logger.warning(
                    f"[{repo_id}] Existing torrent does not match verified source; regenerating"
                )
                delete_cached_torrent(repo_id, revision, repo_type=repo_type)
            # Corrupt cache/downloaded entry — fall through to regenerate
            logger.warning(f"Existing torrent for {repo_id}@{revision} is invalid, regenerating")

        # ── Layer 3: No cache/tracker hit — generate from scratch ─────────
        #
        # Resolve the snapshot path from the LOCAL HF cache only.
        # This function must never trigger a network download — the caller is
        # responsible for ensuring files are already cached (e.g. via a prior
        # `snapshot_download()` call).  Using local_files_only=True also avoids
        # the P2P self-interception problem: if enable_p2p() has already been
        # called, a network snapshot_download here would be intercepted by the
        # monkey patch, trying to download via P2P with no peers → 300s timeout.
        logger.info(f"Resolving HF snapshot for {repo_id}@{revision}")
        
        snapshot_kwargs = {
            "repo_id": repo_id,
            "revision": revision,
            "repo_type": repo_type if repo_type != "model" else None,
            "local_files_only": True,
        }
        if cache_dir is not None:
            snapshot_kwargs["cache_dir"] = cache_dir
        if local_dir is not None:
            snapshot_kwargs["local_dir"] = local_dir

        snapshot_path = snapshot_download(**snapshot_kwargs)
        file_path = Path(snapshot_path)
        
        if not file_path.exists() or not file_path.is_dir():
            logger.error(f"Snapshot directory not found or valid: {file_path}")
            return None

        commit_hash = revision if _COMMIT_HASH_RE.match(revision) else file_path.name

        if local_dir is not None:
            fs, piece_hash_base = _build_local_dir_file_storage(
                repo_id,
                revision,
                local_dir,
            )
            if fs is None or piece_hash_base is None:
                return None
        else:
            # Create file storage
            fs = lt.file_storage()

            # We must add files relative to the snapshot path, but since libtorrent handles
            # the wrapping folder automatically, we just point it to the snapshot root directory.
            lt.add_files(fs, str(file_path))
            piece_hash_base = str(file_path.parent)

        # ── Deterministic piece_length selection ───────────────────────────
        # piece_length is derived solely from total_size so that every client
        # creating a torrent for the same repo@revision produces the same
        # info_hash → same swarm.  This must NOT be caller-configurable.
        total_size = fs.total_size()
        piece_length = get_optimal_piece_length(total_size)
        logger.info(
            f"piece_length={format_bytes(piece_length)} "
            f"for total_size={format_bytes(total_size)}"
        )

        # Create a pure BitTorrent v2 torrent.
        #
        # libtorrent will insert .pad/ entries into the logical file table so that each
        # payload file starts on a piece boundary. These padding files are internal to the
        # torrent layout and must not leak into the business-layer file manifest we publish
        # to the tracker or compare against completed HF snapshots.
        t = lt.create_torrent(fs, piece_length, flags=lt.create_torrent.v2_only)


        # Add tracker
        announce_url = f"{tracker_client.tracker_url.rstrip('/')}/announce"
        t.add_tracker(announce_url)

        # Set creator and comment
        t.set_creator("llmpt-client")
        comment_target = commit_hash if _COMMIT_HASH_RE.match(commit_hash or "") else file_path.name
        t.set_comment(f"Created by llmpt for {comment_target}")

        # Generate piece hashes
        logger.info(f"Generating piece hashes for {file_path.name}...")
        lt.set_piece_hashes(t, piece_hash_base)

        # Generate torrent
        torrent = t.generate()
        if local_dir is not None and _COMMIT_HASH_RE.match(commit_hash or ""):
            if not _rewrite_torrent_root_name(torrent, commit_hash):
                logger.error(
                    f"[{repo_id}] Failed to rewrite generated local_dir torrent root "
                    f"to revision {commit_hash}"
                )
                return None
        torrent_data = lt.bencode(torrent)

        # Cache the generated torrent for future use
        save_torrent_to_cache(repo_id, revision, torrent_data, repo_type=repo_type)

        # Extract hash metadata and payload files for the application-layer manifest.
        info = lt.torrent_info(torrent)
        hash_metadata = _extract_info_hash_metadata(info)
        file_list, payload_total_size = _collect_payload_files(info)

        logger.info(
            f"Torrent created: info_hash={hash_metadata['info_hash']} "
            f"announce_key={hash_metadata['announce_key']}"
        )

        result = {
            'file_size': total_size,
            'piece_length': piece_length,
            'num_pieces': info.num_pieces(),
            'num_files': len(file_list),
            'torrent_data': torrent_data,
            'commit_hash': commit_hash,
            'files': file_list,
        }
        if payload_total_size != total_size:
            logger.warning(
                f"[{repo_id}] Payload file total ({payload_total_size}) does not match "
                f"file_storage total ({total_size}); using file_storage total"
            )
        result.update(hash_metadata)
        return result

    except Exception as e:
        logger.error(f"Failed to create torrent: {e}")
        return None


def ensure_registered(
    repo_id: str,
    revision: str,
    repo_type: str,
    torrent_data: bytes,
    tracker_client: Any,
) -> bool:
    """
    Ensure the torrent is registered with the tracker.

    If the tracker already knows about this torrent, this is a no-op.
    Otherwise, parse the torrent data and register it.

    Args:
        repo_id: HuggingFace repository ID.
        revision: Git commit hash.
        repo_type: "model", "dataset", or "space".
        torrent_data: Raw .torrent file bytes.
        tracker_client: TrackerClient instance.

    Returns:
        True if the torrent is (now) registered, False on failure.
    """
    # Check if tracker already has this torrent
    existing = tracker_client.get_torrent_info(repo_id, revision, repo_type=repo_type)
    if existing:
        try:
            from .torrent_state import mark_tracker_registration

            mark_tracker_registration(
                repo_id,
                revision,
                repo_type=repo_type,
                tracker_url=tracker_client.tracker_url,
                registered=True,
                info_hash=existing.get("info_hash"),
            )
        except Exception:
            pass
        logger.debug(f"[{repo_id}] Torrent already registered on tracker")
        return True

    # Parse torrent to extract metadata needed for registration
    result = _torrent_data_to_result(torrent_data, repo_id)
    if result is None:
        logger.warning(f"[{repo_id}] Cannot parse torrent data for registration")
        return False

    resolved_revision = _normalized_result_commit_hash(result, revision)

    register_kwargs = {
        "repo_id": repo_id,
        "revision": resolved_revision,
        "repo_type": repo_type,
        "name": repo_id,
        "info_hash": result['info_hash'],
        "total_size": result['file_size'],
        "file_count": result.get('num_files', 1),
        "piece_length": result['piece_length'],
        "torrent_data": torrent_data,
        "files": result['files'],
    }
    if result.get("announce_key"):
        register_kwargs["announce_key"] = result["announce_key"]

    success = tracker_client.register_torrent(
        **register_kwargs,
    )

    if success:
        try:
            from .torrent_state import mark_tracker_registration

            mark_tracker_registration(
                repo_id,
                revision,
                repo_type=repo_type,
                tracker_url=tracker_client.tracker_url,
                registered=True,
                info_hash=result["info_hash"],
            )
        except Exception:
            pass
        logger.info(f"[{repo_id}] ✓ Torrent registered on tracker (was missing)")
    else:
        try:
            from .torrent_state import mark_tracker_registration

            mark_tracker_registration(
                repo_id,
                revision,
                repo_type=repo_type,
                tracker_url=tracker_client.tracker_url,
                registered=False,
                info_hash=result["info_hash"],
                error="register_failed",
            )
        except Exception:
            pass
        logger.warning(f"[{repo_id}] ✗ Failed to register torrent on tracker")

    return success


def create_and_register_torrent(
    repo_id: str,
    revision: str,
    repo_type: str,
    name: str,
    tracker_client: Any,
    *,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> Optional[dict]:
    """
    Create torrent and register it with the tracker.

    Args:
        repo_id: HuggingFace repository ID.
        revision: Git commit hash or branch name.
        repo_type: The type of repository (e.g., 'model').
        name: The display name of the model.
        tracker_client: TrackerClient instance.

    Returns:
        The torrent info dictionary if successful, None otherwise.
    """
    # Create torrent natively from the Hugging Face cache (or reuse existing)
    torrent_info = create_torrent(
        repo_id=repo_id,
        revision=revision,
        tracker_client=tracker_client,
        repo_type=repo_type,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )

    if not torrent_info:
        return None

    # Always register with the resolved commit hash from the snapshot path.
    # This provides defense-in-depth: even if the caller passed a branch name
    # like "main", the tracker entry will use the immutable commit hash.
    resolved_revision = torrent_info.get('commit_hash', revision)
    if resolved_revision != revision:
        logger.info(
            f"Revision override: registering with commit hash "
            f"'{resolved_revision}' instead of '{revision}'"
        )

    # Register with tracker using the resolved commit hash
    register_kwargs = {
        "repo_id": repo_id,
        "revision": resolved_revision,
        "repo_type": repo_type,
        "name": name,
        "info_hash": torrent_info['info_hash'],
        "total_size": torrent_info['file_size'],
        "file_count": torrent_info.get('num_files', 1),
        "piece_length": torrent_info['piece_length'],
        "torrent_data": torrent_info['torrent_data'],
        "files": torrent_info['files'],
    }
    if torrent_info.get("announce_key"):
        register_kwargs["announce_key"] = torrent_info["announce_key"]

    success = tracker_client.register_torrent(**register_kwargs)

    if success:
        try:
            from .torrent_state import mark_tracker_registration

            mark_tracker_registration(
                repo_id,
                revision,
                repo_type=repo_type,
                tracker_url=tracker_client.tracker_url,
                registered=True,
                info_hash=torrent_info["info_hash"],
            )
        except Exception:
            pass
        return torrent_info
    try:
        from .torrent_state import mark_tracker_registration

        mark_tracker_registration(
            repo_id,
            revision,
            repo_type=repo_type,
            tracker_url=tracker_client.tracker_url,
            registered=False,
            info_hash=torrent_info["info_hash"],
            error="register_failed",
        )
    except Exception:
        pass
    return None
