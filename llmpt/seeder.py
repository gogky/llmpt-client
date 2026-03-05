"""
Seeding manager for background seeding tasks.
We now unify all seeding directly into the P2PBatchManager.
"""

import logging
from typing import Dict, Any, Optional

from .utils import lt, LIBTORRENT_AVAILABLE

from llmpt.p2p_batch import P2PBatchManager

logger = logging.getLogger('llmpt.seeder')


def start_seeding(
    repo_id: str,
    revision: str,
    tracker_client: Any,
    torrent_data: Optional[bytes] = None,
) -> bool:
    """
    Start seeding a repository in the background using the unified P2PBatchManager.

    Args:
        repo_id: The HuggingFace repository ID.
        revision: The HuggingFace revision/branch.
        tracker_client: TrackerClient instance.
        torrent_data: Raw .torrent file bytes.  When provided, libtorrent can
                      initialise the torrent immediately without fetching
                      metadata from peers (avoids timeout when the seeder is
                      the first/only peer).

    Returns:
        True if seeding started successfully, False otherwise.
    """
    if not LIBTORRENT_AVAILABLE:
        logger.warning("libtorrent not available, cannot seed")
        return False

    manager = P2PBatchManager()
    success = manager.register_seeding_task(
        repo_id=repo_id,
        revision=revision,
        tracker_client=tracker_client,
        torrent_data=torrent_data,
    )

    if success:
        logger.info(f"Started unified seeding for: {repo_id}@{revision}")
    else:
        logger.error(f"Failed to start unified seeding for: {repo_id}@{revision}")
        
    return success


def stop_seeding(repo_id: str, revision: str) -> bool:
    """
    Stop seeding a specific repository.
    """
    manager = P2PBatchManager()
    repo_key = (repo_id, revision)
    worker = None

    with manager._lock:
        if repo_key not in manager.sessions:
            logger.warning(f"Not seeding: {repo_id}@{revision}")
            return False
            
        session_info = manager.sessions[repo_key]
        session_info._cleanup_seeding_hardlinks()
        with session_info.lock:
            handle = session_info.handle
            session_info.handle = None
            session_info.is_valid = False
        if handle:
            manager.lt_session.remove_torrent(handle)
        worker = session_info.worker_thread

        del manager.sessions[repo_key]

    if worker is not None:
        worker.join(timeout=3)

    logger.info(f"Stopped seeding for: {repo_id}@{revision}")
    return True


def stop_all_seeding() -> int:
    """
    Stop all active seeding tasks.
    """
    manager = P2PBatchManager()
    count = 0
    threads_to_join = []
    with manager._lock:
        for repo_key, session_info in list(manager.sessions.items()):
            session_info._cleanup_seeding_hardlinks()
            with session_info.lock:
                handle = session_info.handle
                session_info.handle = None
                session_info.is_valid = False
            if handle:
                manager.lt_session.remove_torrent(handle)
            if session_info.worker_thread is not None:
                threads_to_join.append(session_info.worker_thread)
            del manager.sessions[repo_key]
            count += 1

    for t in threads_to_join:
        t.join(timeout=3)

    logger.info(f"Stopped all seeding ({count} tasks)")
    return count


def get_seeding_status() -> Dict[str, Dict[str, Any]]:
    """
    Get status of all active seeding tasks.
    """
    manager = P2PBatchManager()
    status = {}
    with manager._lock:
        for repo_key, session_info in manager.sessions.items():
            if not session_info.handle or not session_info.handle.is_valid():
                continue
            
            s = session_info.handle.status()
            repo_id, revision = repo_key
            status[f"{repo_id}@{revision}"] = {
                'repo_id': repo_id,
                'revision': revision,
                'uploaded': s.total_upload,
                'peers': s.num_peers,
                'upload_rate': s.upload_rate,
                'progress': s.progress,
                'state': str(s.state)
            }
    return status
