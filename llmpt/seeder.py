"""
Seeding manager for background seeding tasks.

This is a thin façade around P2PBatchManager's public session lifecycle
methods.  All state management and cleanup is handled internally by the
manager — this module simply provides convenient module-level functions.
"""

import logging
from typing import TYPE_CHECKING, Dict, Any, Optional

if TYPE_CHECKING:
    from .tracker_client import TrackerClient

from .utils import LIBTORRENT_AVAILABLE
from .p2p_batch import P2PBatchManager

logger = logging.getLogger(__name__)


def start_seeding(
    repo_id: str,
    revision: str,
    tracker_client: 'TrackerClient',
    torrent_data: Optional[bytes] = None,
    repo_type: str = 'model',
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
        repo_type=repo_type,
        tracker_client=tracker_client,
        torrent_data=torrent_data,
    )

    if success:
        logger.info(f"Started unified seeding for: {repo_id}@{revision}")
    else:
        logger.error(f"Failed to start unified seeding for: {repo_id}@{revision}")
        
    return success


def stop_seeding(repo_id: str, revision: str, repo_type: str = 'model') -> bool:
    """Stop seeding a specific repository."""
    manager = P2PBatchManager()
    removed = manager.remove_session(repo_id, revision, repo_type=repo_type)

    if removed:
        logger.info(f"Stopped seeding for: {repo_id}@{revision}")
    else:
        logger.warning(f"Not seeding: {repo_id}@{revision}")
    return removed


def stop_all_seeding() -> int:
    """Stop all active seeding tasks."""
    manager = P2PBatchManager()
    count = manager.remove_all_sessions()
    logger.info(f"Stopped all seeding ({count} tasks)")
    return count


def get_seeding_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all active seeding tasks."""
    manager = P2PBatchManager()
    return manager.get_all_session_status()
