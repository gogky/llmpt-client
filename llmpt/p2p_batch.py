"""
P2P Batch Manager for coordinating libtorrent downloads and HTTP intercepts.

This module provides a singleton P2PBatchManager that aggregates concurrent 
HTTP download requests into a single, efficient BitTorrent session with 
dynamic file prioritization.
"""

import threading
import logging
from typing import Dict, Any, Optional

from .utils import lt, LIBTORRENT_AVAILABLE

# Re-export SessionContext for backward compatibility.
# All existing imports like ``from llmpt.p2p_batch import SessionContext`` continue to work.
from .session_context import SessionContext  # noqa: F401

logger = logging.getLogger('llmpt.p2p_batch')

class P2PBatchManager:
    """
    Manages global P2P download sessions for different HuggingFace repositories.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(P2PBatchManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        with self._lock:
            if getattr(self, '_initialized', False):
                return
            
            self._initialized = True
            # Mapping from (repo_id, revision) -> SessionContext
            self.sessions: Dict[tuple, SessionContext] = {}
            if LIBTORRENT_AVAILABLE:
                self.lt_session = lt.session()
                settings = self.lt_session.get_settings()
                settings['listen_interfaces'] = '0.0.0.0:6881'
                self.lt_session.apply_settings(settings)
            else:
                self.lt_session = None

    def register_seeding_task(self, repo_id: str, revision: str, tracker_client: Any, torrent_data: Optional[bytes] = None) -> bool:
        """
        Register a repository to be tracked for background seeding.
        This behaves like a download but without any specific HTTP interception blocks.
        """
        if not LIBTORRENT_AVAILABLE:
            return False

        repo_key = (repo_id, revision)
        with self._lock:
            if repo_key not in self.sessions:
                self.sessions[repo_key] = SessionContext(
                    repo_id=repo_id,
                    revision=revision,
                    tracker_client=tracker_client,
                    lt_session=self.lt_session,
                    timeout=30, # short timeout for initialization
                    torrent_data=torrent_data,
                )
            session_ctx = self.sessions[repo_key]
            
        # Ensure torrent is initialized with the magnet link via tracker
        if not session_ctx.is_valid:
            return False
            
        success = session_ctx._init_torrent()
        if not success:
            logger.error(f"[{repo_id}] Failed to initialize seeding session.")
            return False
                
        # For seeding, force map all files in the torrent to their HF blob equivalents!
        return session_ctx.map_all_files_for_seeding()

    def register_request(
        self,
        repo_id: str,
        revision: str,
        filename: str,
        temp_file_path: str,
        tracker_client: Any,
        timeout: int = 300
    ) -> bool:
        """
        Register a file download request.
        
        Args:
            repo_id: The HF repository ID.
            revision: The commit hash or branch name.
            filename: The relative path of the file to download within the repo.
            temp_file_path: The absolute path where HF expects the file to be saved.
            tracker_client: Instance of TrackerClient to query torrennt info.
            timeout: Max time to wait for the download.
            
        Returns:
            True if P2P download succeeds, False if it failed and should fallback to HTTP.
        """
        if not LIBTORRENT_AVAILABLE:
            return False

        repo_key = (repo_id, revision)
        
        with self._lock:
            if repo_key not in self.sessions:
                self.sessions[repo_key] = SessionContext(
                    repo_id=repo_id,
                    revision=revision,
                    tracker_client=tracker_client,
                    lt_session=self.lt_session,
                    timeout=timeout
                )
            session_ctx = self.sessions[repo_key]
        
        # Register the file with the session context and wait for it
        return session_ctx.download_file(filename, temp_file_path)
