"""
Torrent creation utilities.
"""

import logging
from pathlib import Path
from typing import Optional, Any

from .utils import lt, LIBTORRENT_AVAILABLE, get_optimal_piece_length, format_bytes, strip_torrent_root

logger = logging.getLogger('llmpt.torrent_creator')


def _torrent_data_to_result(torrent_data: bytes, repo_id: str) -> Optional[dict]:
    """Build a torrent info result dict from raw cached .torrent bytes.

    This mirrors the return value of :func:`create_torrent` but skips the
    expensive ``set_piece_hashes`` step by parsing the already-generated
    bencode data.
    """
    try:
        info = lt.torrent_info(lt.bdecode(torrent_data))
        info_hash = str(info.info_hash())
        files = info.files()

        file_list = []
        total_size = 0
        for i in range(files.num_files()):
            lt_file_path = files.file_path(i)
            relative_path = strip_torrent_root(lt_file_path)
            size = files.file_size(i)
            file_list.append({'path': relative_path, 'size': size})
            total_size += size

        # Extract the root folder name (= commit hash) from the first file path
        first_path = files.file_path(0).replace('\\', '/')
        commit_hash = first_path.split('/')[0] if '/' in first_path else ''

        return {
            'info_hash': info_hash,
            'file_size': total_size,
            'piece_length': info.piece_length(),
            'num_pieces': info.num_pieces(),
            'num_files': info.num_files(),
            'torrent_data': torrent_data,
            'commit_hash': commit_hash,
            'files': file_list,
        }
    except Exception as e:
        logger.warning(f"[{repo_id}] Failed to parse cached torrent: {e}")
        return None


def create_torrent(
    repo_id: str,
    revision: str,
    tracker_client: Any,
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
        from .torrent_cache import resolve_torrent_data, save_torrent_to_cache

        # ── Layer 1 & 2: check local .torrent cache AND tracker ───────────
        # If we or another seeder already generated a torrent for this repo@revision,
        # reuse it. This skips the expensive set_piece_hashes() call which can take
        # 30+ minutes for large models on HDD.
        cached_or_downloaded = resolve_torrent_data(repo_id, revision, tracker_client)
        if cached_or_downloaded:
            logger.info(f"Using existing torrent for {repo_id}@{revision} (from cache or tracker)")
            result = _torrent_data_to_result(cached_or_downloaded, repo_id)
            if result is not None:
                return result
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
        
        snapshot_path = snapshot_download(
            repo_id=repo_id, 
            revision=revision, 
            local_files_only=True,
        )
        file_path = Path(snapshot_path)
        
        if not file_path.exists() or not file_path.is_dir():
            logger.error(f"Snapshot directory not found or valid: {file_path}")
            return None

        # Create file storage
        fs = lt.file_storage()
        
        # We must add files relative to the snapshot path, but since libtorrent handles 
        # the wrapping folder automatically, we just point it to the snapshot root directory.
        lt.add_files(fs, str(file_path))

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

        # Create torrent with v1_only flag to eliminate .pad/ padding files.
        #
        # Background on the padding problem:
        #   By default (and in v2/hybrid modes), libtorrent inserts virtual .pad/ files to
        #   align each file's start to a piece boundary (BEP 47 / BEP 52 canonical layout).
        #   In our use case these padding files are never in the HF cache, so the seeder's
        #   piece hash check always fails → seeder can't serve any data to peers.
        #
        # Why NOT v2 or hybrid?
        #   Despite BT v2's per-file Merkle hash trees, libtorrent's Python bindings still
        #   insert .pad/ files in v2_only (=32) and canonical_files (=128) modes.
        #   Tested with libtorrent 2.0.10: all modes except v1_only produce padding files.
        #
        # Why v1_only (=64)?
        #   - Zero .pad/ virtual files produced (verified experimentally)
        #   - Piece hash verification completes in <1s (vs 300s+ timeout with padding)
        #   - Fully compatible with libtorrent 2.x (which is our minimum requirement)
        #   - Each file's data runs contiguous across piece boundaries (standard BT v1 behavior)
        t = lt.create_torrent(fs, piece_length, flags=lt.create_torrent.v1_only)


        # Add tracker
        announce_url = f"{tracker_client.tracker_url.rstrip('/')}/announce"
        t.add_tracker(announce_url)

        # Set creator and comment
        t.set_creator("llmpt-client")
        t.set_comment(f"Created by llmpt for {file_path.name}")

        # Generate piece hashes
        logger.info(f"Generating piece hashes for {file_path.name}...")
        lt.set_piece_hashes(t, str(file_path.parent))

        # Generate torrent
        torrent = t.generate()
        torrent_data = lt.bencode(torrent)

        # Cache the generated torrent for future use
        save_torrent_to_cache(repo_id, revision, torrent_data)

        # Get info hash
        info = lt.torrent_info(torrent)
        info_hash = str(info.info_hash())

        # Extract per-file metadata
        files = info.files()
        file_list = []
        for i in range(files.num_files()):
            lt_file_path = files.file_path(i)
            relative_path = strip_torrent_root(lt_file_path)
            file_list.append({
                'path': relative_path,
                'size': files.file_size(i),
            })

        logger.info(f"Torrent created: {info_hash}")

        return {
            'info_hash': info_hash,
            'file_size': total_size,
            'piece_length': piece_length,
            'num_pieces': info.num_pieces(),
            'num_files': info.num_files(),
            'torrent_data': torrent_data,
            'commit_hash': file_path.name,
            'files': file_list,
        }

    except Exception as e:
        logger.error(f"Failed to create torrent: {e}")
        return None


def create_and_register_torrent(
    repo_id: str,
    revision: str,
    repo_type: str,
    name: str,
    tracker_client: Any,
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
    success = tracker_client.register_torrent(
        repo_id=repo_id,
        revision=resolved_revision,
        repo_type=repo_type,
        name=name,
        info_hash=torrent_info['info_hash'],
        total_size=torrent_info['file_size'],
        file_count=torrent_info.get('num_files', 1),
        piece_length=torrent_info['piece_length'],
        torrent_data=torrent_info['torrent_data'],
        files=torrent_info['files'],
    )

    if success:
        return torrent_info
    return None
