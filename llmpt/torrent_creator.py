"""
Torrent creation utilities.
"""

import logging
from pathlib import Path
from typing import Optional, Any

try:
    import libtorrent as lt
    LIBTORRENT_AVAILABLE = True
except ImportError:
    LIBTORRENT_AVAILABLE = False
    lt = None

logger = logging.getLogger('llmpt.torrent_creator')


def create_torrent(
    repo_id: str,
    revision: str,
    tracker_url: str,
    piece_length: int = 16 * 1024 * 1024,  # 16MB
) -> Optional[dict]:
    """
    Create a torrent file for an entire HuggingFace repository snapshot.

    Args:
        repo_id: HuggingFace repository ID (e.g., meta-llama/Llama-2-7b).
        revision: Git commit hash or branch name.
        tracker_url: Tracker announce URL.
        piece_length: Piece length in bytes (default: 16MB for large files).

    Returns:
        Dictionary containing torrent info (info_hash, magnet_link, etc.)
        or None if creation failed.
    """
    if not LIBTORRENT_AVAILABLE:
        logger.error("libtorrent not available")
        return None

    try:
        from huggingface_hub import snapshot_download
        
        # Download or resolve the snapshot path without re-downloading existing blobs
        logger.info(f"Resolving HF snapshot for {repo_id}@{revision}")
        
        # Since we use this tool mostly for files that exist, this should return instantly
        snapshot_path = snapshot_download(
            repo_id=repo_id, 
            revision=revision, 
            local_files_only=False  # Allow network check if needed, but caches will be used
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
        announce_url = f"{tracker_url.rstrip('/')}/announce"
        t.add_tracker(announce_url)

        # Set creator and comment
        t.set_creator("llmpt-client")
        t.set_comment(f"Created by llmpt for {file_path.name}")

        # Generate piece hashes
        logger.info(f"Generating piece hashes for {file_path.name}...")
        lt.set_piece_hashes(t, str(file_path.parent))

        # Generate torrent
        torrent = t.generate()

        # Get info hash
        info = lt.torrent_info(torrent)
        info_hash = str(info.info_hash())

        # Generate magnet link
        magnet_link = lt.make_magnet_uri(info)

        logger.info(f"Torrent created: {info_hash}")

        return {
            'info_hash': info_hash,
            'magnet_link': magnet_link,
            'file_size': sum(f.stat().st_size for f in file_path.rglob('*') if f.is_file()) if file_path.is_dir() else file_path.stat().st_size,
            'piece_length': piece_length,
            'num_pieces': info.num_pieces(),
            'num_files': info.num_files(),
            'torrent_data': lt.bencode(torrent),
            'commit_hash': file_path.name,
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
    piece_length: int = 16 * 1024 * 1024,
) -> bool:
    """
    Create torrent and register it with the tracker.

    Args:
        repo_id: HuggingFace repository ID.
        revision: Git commit hash or branch name.
        repo_type: The type of repository (e.g., 'model').
        name: The display name of the model.
        tracker_client: TrackerClient instance.
        piece_length: Piece length in bytes.

    Returns:
        The torrent info dictionary if successful, None otherwise.
    """
    # Create torrent natively from the Hugging Face cache
    torrent_info = create_torrent(
        repo_id=repo_id,
        revision=revision,
        tracker_url=tracker_client.tracker_url,
        piece_length=piece_length,
    )

    if not torrent_info:
        return False

    # Register with tracker using the human-readable revision
    success = tracker_client.register_torrent(
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        name=name,
        info_hash=torrent_info['info_hash'],
        total_size=torrent_info['file_size'],
        file_count=torrent_info.get('num_files', 1),
        magnet_link=torrent_info['magnet_link'],
        piece_length=torrent_info['piece_length'],
    )

    if success:
        return torrent_info
    return None
