"""
Seeding manager for background seeding tasks.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import libtorrent as lt
    LIBTORRENT_AVAILABLE = True
except ImportError:
    LIBTORRENT_AVAILABLE = False
    lt = None

logger = logging.getLogger('llmpt.seeder')

# Global seeding state
_seeding_sessions: Dict[str, Dict[str, Any]] = {}
_seeding_lock = threading.Lock()


def start_seeding(
    torrent_info: dict,
    file_path: str,
    duration: int = 3600,
) -> bool:
    """
    Start seeding a file in the background.

    Args:
        torrent_info: Torrent information dictionary.
        file_path: Path to the file to seed.
        duration: How long to seed in seconds (0 = forever).

    Returns:
        True if seeding started successfully, False otherwise.
    """
    if not LIBTORRENT_AVAILABLE:
        logger.warning("libtorrent not available, cannot seed")
        return False

    info_hash = torrent_info.get('info_hash')
    if not info_hash:
        logger.error("Missing info_hash in torrent_info")
        return False

    # Check if already seeding
    with _seeding_lock:
        if info_hash in _seeding_sessions:
            logger.info(f"Already seeding: {info_hash}")
            return True

    # Start seeding in background thread
    thread = threading.Thread(
        target=_seed_worker,
        args=(torrent_info, file_path, duration),
        daemon=True
    )
    thread.start()

    logger.info(f"Started seeding: {info_hash} (duration: {duration}s)")
    return True


def _seed_worker(torrent_info: dict, file_path: str, duration: int):
    """Background worker for seeding."""
    info_hash = torrent_info['info_hash']
    magnet_link = torrent_info['magnet_link']

    try:
        # Create libtorrent session
        session = lt.session()
        session.listen_on(6881, 6891)

        # Add torrent
        params = lt.parse_magnet_uri(magnet_link)
        params.save_path = str(Path(file_path).parent)

        handle = session.add_torrent(params)

        # Store session info
        with _seeding_lock:
            _seeding_sessions[info_hash] = {
                'session': session,
                'handle': handle,
                'start_time': time.time(),
                'file_path': file_path,
            }

        logger.info(f"Seeding started: {info_hash}")

        # Seed for specified duration
        start_time = time.time()
        while True:
            # Check if should stop
            if duration > 0 and time.time() - start_time > duration:
                logger.info(f"Seeding duration reached: {info_hash}")
                break

            # Check if manually stopped
            with _seeding_lock:
                if info_hash not in _seeding_sessions:
                    logger.info(f"Seeding manually stopped: {info_hash}")
                    break

            # Log status periodically
            status = handle.status()
            if int(time.time()) % 300 == 0:  # Every 5 minutes
                logger.info(
                    f"Seeding {info_hash}: "
                    f"uploaded: {status.total_upload / 1024 / 1024:.1f} MB, "
                    f"peers: {status.num_peers}"
                )

            time.sleep(1)

    except Exception as e:
        logger.error(f"Seeding error: {e}")

    finally:
        # Clean up
        with _seeding_lock:
            if info_hash in _seeding_sessions:
                session_info = _seeding_sessions[info_hash]
                session_info['session'].remove_torrent(session_info['handle'])
                del _seeding_sessions[info_hash]

        logger.info(f"Seeding stopped: {info_hash}")


def stop_seeding(info_hash: str) -> bool:
    """
    Stop seeding a specific torrent.

    Args:
        info_hash: Info hash of the torrent to stop seeding.

    Returns:
        True if stopped successfully, False if not seeding.
    """
    with _seeding_lock:
        if info_hash not in _seeding_sessions:
            logger.warning(f"Not seeding: {info_hash}")
            return False

        # Remove from dict (worker thread will detect and stop)
        del _seeding_sessions[info_hash]

    logger.info(f"Stopping seeding: {info_hash}")
    return True


def stop_all_seeding() -> int:
    """
    Stop all active seeding tasks.

    Returns:
        Number of seeding tasks stopped.
    """
    with _seeding_lock:
        count = len(_seeding_sessions)
        _seeding_sessions.clear()

    logger.info(f"Stopped all seeding ({count} tasks)")
    return count


def get_seeding_status() -> Dict[str, Dict[str, Any]]:
    """
    Get status of all active seeding tasks.

    Returns:
        Dictionary mapping info_hash to seeding status.
    """
    with _seeding_lock:
        status = {}
        for info_hash, session_info in _seeding_sessions.items():
            handle = session_info['handle']
            s = handle.status()
            status[info_hash] = {
                'file_path': session_info['file_path'],
                'start_time': session_info['start_time'],
                'uploaded': s.total_upload,
                'peers': s.num_peers,
                'upload_rate': s.upload_rate,
            }
        return status
