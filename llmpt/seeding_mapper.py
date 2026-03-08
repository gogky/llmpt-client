"""
Seeding file-mapping helpers.

These functions handle the creation of hardlinks (or rename_file fallback)
so that libtorrent can seed files already present in the HuggingFace cache
without duplicating data on disk.
"""

import os
import logging
import time
from typing import List, Optional, Tuple

from .utils import lt, strip_torrent_root

logger = logging.getLogger(__name__)


# ── HF cache resolution ──────────────────────────────────────────────────────

def resolve_hf_blob(repo_id: str, filename: str, revision: str, *, repo_type: str = "model", cache_dir: Optional[str] = None, local_dir: Optional[str] = None) -> Optional[str]:
    """Look up a file in the local HuggingFace cache and return its real path.

    Returns:
        Absolute real path (symlinks resolved) to the blob, or None if
        the file is not in cache.
    """
    try:
        if local_dir:
            # For local_dir, files are stored directly in the local folder, no blob mapping
            local_path = os.path.join(local_dir, filename)
            if os.path.exists(local_path):
                return os.path.realpath(local_path)
            # If not found directly, maybe it hasn't been moved yet. We fall through to cache.
            
        from huggingface_hub import try_to_load_from_cache
        cache_lookup_kwargs = {
            "repo_id": repo_id,
            "filename": filename,
            "revision": revision,
            "repo_type": repo_type if repo_type != "model" else None,
        }
        if cache_dir is not None:
            cache_lookup_kwargs["cache_dir"] = cache_dir
        local_path = try_to_load_from_cache(**cache_lookup_kwargs)
        if local_path and isinstance(local_path, str):
            return os.path.realpath(local_path)
    except Exception as e:
        logger.warning(f"[{repo_id}] Failed to resolve HF blob for {filename}: {e}")
    return None


# ── Padding files ─────────────────────────────────────────────────────────────

def is_padding_file(target_norm: str) -> bool:
    """Return True if *target_norm* is a libtorrent padding file."""
    return target_norm.startswith('.pad/') or '/.pad/' in target_norm


def create_padding_file(expected_path: str, size: int) -> None:
    """Create a zero-filled padding file if it doesn't already exist."""
    os.makedirs(os.path.dirname(expected_path), exist_ok=True)
    if not os.path.exists(expected_path):
        with open(expected_path, 'wb') as f:
            f.write(b'\x00' * size)


# ── Hardlink strategy (fast seed_mode startup) ───────────────────────────────

def hardlink_files_for_seeding(
    torrent_info,
    temp_dir: str,
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> Tuple[List[str], int]:
    """Create hardlinks at paths libtorrent expects, pointing to HF blobs.

    Args:
        torrent_info: libtorrent ``torrent_info`` object.
        temp_dir: Root download directory (libtorrent's ``save_path``).
        repo_id: HuggingFace repository ID.
        revision: Commit hash.

    Returns:
        ``(hardlinks, mapped_count)`` — list of created files, and count of
        successfully mapped real files.

    Raises:
        OSError: If a hardlink fails (e.g. cross-filesystem), signalling that
                 the caller should fall back to the legacy strategy.
    """
    files = torrent_info.files()
    hardlinks: List[str] = []
    mapped_count = 0

    for file_index in range(files.num_files()):
        lt_path = files.file_path(file_index).replace('\\', '/')
        file_size = files.file_size(file_index)
        target_norm = strip_torrent_root(lt_path)

        # Handle padding files
        if is_padding_file(target_norm):
            expected_path = os.path.join(temp_dir, lt_path)
            create_padding_file(expected_path, file_size)
            hardlinks.append(expected_path)
            logger.info(f"[{repo_id}] Created padding file [{file_index}]: {target_norm} ({file_size} bytes)")
            continue

        # Resolve the HF cache blob path
        real_path = resolve_hf_blob(repo_id, target_norm, revision, repo_type=repo_type, cache_dir=cache_dir, local_dir=local_dir)
        if not real_path:
            logger.warning(f"[{repo_id}] Cache miss for seeding [{file_index}]: {target_norm} (revision={revision})")
            continue

        # Create hardlink at the path libtorrent expects
        expected_path = os.path.join(temp_dir, lt_path)
        os.makedirs(os.path.dirname(expected_path), exist_ok=True)

        if os.path.exists(expected_path):
            os.unlink(expected_path)

        # Let OSError propagate to trigger legacy fallback
        os.link(real_path, expected_path)
        hardlinks.append(expected_path)
        mapped_count += 1
        logger.info(f"[{repo_id}] Hardlinked for seeding [{file_index}]: {target_norm} -> {real_path}")

    return hardlinks, mapped_count


# ── Legacy strategy (rename_file + force_recheck) ────────────────────────────

def rename_files_for_seeding(
    handle,
    torrent_info,
    temp_dir: str,
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> int:
    """Map files via rename_file() and trigger a force_recheck().

    This is the fallback when hardlinks fail (cross-filesystem).
    It requires a full SHA1 verification of all pieces, which can
    take minutes for large models.

    Args:
        handle: libtorrent torrent handle.
        torrent_info: libtorrent torrent_info object.
        temp_dir: Root download directory.
        repo_id: HuggingFace repository ID.
        revision: Commit hash.

    Returns:
        Number of files successfully mapped.
    """
    files = torrent_info.files()
    mapped_count = 0

    for file_index in range(files.num_files()):
        lt_path = files.file_path(file_index).replace('\\', '/')
        file_size = files.file_size(file_index)
        target_norm = strip_torrent_root(lt_path)

        if is_padding_file(target_norm):
            pad_dir = os.path.join(temp_dir, ".pad_files")
            os.makedirs(pad_dir, exist_ok=True)
            pad_file_path = os.path.join(pad_dir, f"pad_{file_index}_{file_size}")
            if not os.path.exists(pad_file_path):
                with open(pad_file_path, 'wb') as f:
                    f.write(b'\x00' * file_size)
            handle.rename_file(file_index, pad_file_path)
            continue

        real_path = resolve_hf_blob(repo_id, target_norm, revision, repo_type=repo_type, cache_dir=cache_dir, local_dir=local_dir)
        if real_path:
            handle.rename_file(file_index, real_path)
            mapped_count += 1
            logger.info(f"[{repo_id}] [legacy] Mapped [{file_index}]: {target_norm} -> {real_path}")
        else:
            logger.warning(f"[{repo_id}] [legacy] Cache miss [{file_index}]: {target_norm}")

    logger.info(f"[{repo_id}] [legacy] Mapped {mapped_count}/{files.num_files()} files. Starting force_recheck...")

    handle.resume()
    handle.force_recheck()

    # Wait for recheck (no hard timeout — large models can take 10+ min on HDD)
    recheck_start = time.time()
    last_log_time = 0.0
    while True:
        s = handle.status()
        if s.state not in (1, 7):
            elapsed = time.time() - recheck_start
            logger.info(f"[{repo_id}] [legacy] Recheck complete in {elapsed:.0f}s. Pieces: {s.num_pieces}")
            break
        now = time.time()
        if now - last_log_time >= 10:
            last_log_time = now
            elapsed = now - recheck_start
            logger.info(f"[{repo_id}] [legacy] Rechecking... {s.progress*100:.1f}% ({elapsed:.0f}s)")
        time.sleep(0.5)

    return mapped_count


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_hardlinks(repo_id: str, hardlinks: List[str]) -> None:
    """Remove hardlinks created for seeding in p2p_root."""
    for path in hardlinks:
        try:
            if os.path.exists(path):
                os.unlink(path)
                logger.debug(f"[{repo_id}] Cleaned up seeding hardlink: {path}")
        except OSError as e:
            logger.warning(f"[{repo_id}] Failed to clean up {path}: {e}")
