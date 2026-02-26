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
    file_path: str,
    tracker_url: str,
    piece_length: int = 16 * 1024 * 1024,  # 16MB
) -> Optional[dict]:
    """
    Create a torrent file for a given file.

    Args:
        file_path: Path to the file to create torrent for.
        tracker_url: Tracker announce URL.
        piece_length: Piece length in bytes (default: 16MB for large files).

    Returns:
        Dictionary containing torrent info (info_hash, magnet_link, etc.)
        or None if creation failed.

    Example:
        >>> info = create_torrent(
        ...     file_path="/path/to/model.bin",
        ...     tracker_url="http://tracker.example.com/announce"
        ... )
        >>> print(info['magnet_link'])
    """
    if not LIBTORRENT_AVAILABLE:
        logger.error("libtorrent not available")
        return None

    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        # Create file storage
        fs = lt.file_storage()
        if file_path.is_dir():
            lt.add_files(fs, str(file_path))
        else:
            fs.add_file(file_path.name, file_path.stat().st_size)

        # Create torrent
        t = lt.create_torrent(fs, piece_length)

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
        }

    except Exception as e:
        logger.error(f"Failed to create torrent: {e}")
        return None


def create_and_register_torrent(
    file_path: str,
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
        file_path: Path to the file.
        repo_id: HuggingFace repository ID.
        revision: Git commit hash or branch name.
        repo_type: The type of repository (e.g., 'model').
        name: The display name of the model.
        tracker_client: TrackerClient instance.
        piece_length: Piece length in bytes.

    Returns:
        True if successful, False otherwise.
    """
    # Create torrent
    torrent_info = create_torrent(
        file_path=file_path,
        tracker_url=tracker_client.tracker_url,
        piece_length=piece_length,
    )

    if not torrent_info:
        return False

    # Register with tracker
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

    return success
