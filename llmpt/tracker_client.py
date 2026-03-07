"""
Tracker client for communicating with the llmpt tracker server.
"""

import logging
import requests
from typing import Optional, Dict, Any
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class TrackerClient:
    """Client for interacting with llmpt tracker server."""

    def __init__(self, tracker_url: str, timeout: int = 10):
        """
        Initialize tracker client.

        Args:
            tracker_url: Base URL of the tracker server.
            timeout: Request timeout in seconds.
        """
        self.tracker_url = tracker_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def get_torrent_info(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        *,
        repo_type: str = 'model'
    ) -> Optional[Dict[str, Any]]:
        """
        Query tracker for torrent information.

        Args:
            repo_id: HuggingFace repository ID (e.g., "meta-llama/Llama-2-7b").
            revision: Git commit hash or branch name. If None, returns the most recent.

        Returns:
            Dictionary containing torrent info (info_hash, total_size, files, etc.)
            or None if torrent doesn't exist.
        """
        try:
            url = urljoin(self.tracker_url, '/api/v1/torrents')
            params = {'repo_id': repo_id, 'repo_type': repo_type}
            if revision:
                params['revision'] = revision

            logger.debug(f"Querying tracker: {url} with params {params}")

            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 404:
                logger.debug(f"No torrent found for {repo_id}")
                return None

            response.raise_for_status()
            data = response.json()

            # Server returns {"total": N, "data": [...]}
            torrents = data.get('data', [])
            
            if not torrents:
                logger.debug(f"No torrent found for {repo_id}" + (f" (revision: {revision})" if revision else ""))
                return None

            # Server now filters by revision, so the first result is the match
            torrent = torrents[0]
            logger.info(f"Found torrent for {repo_id} (revision: {torrent.get('revision', 'N/A')}): {torrent.get('info_hash', 'N/A')}")
            return torrent

        except requests.RequestException as e:
            logger.warning(f"Failed to query tracker: {e}")
            return None

    def download_torrent(
        self,
        repo_id: str,
        revision: str,
        *,
        repo_type: str = 'model'
    ) -> Optional[bytes]:
        """
        Download raw .torrent file from tracker.

        Args:
            repo_id: HuggingFace repository ID.
            revision: Git commit hash.

        Returns:
            Raw torrent bytes, or None if not found.
        """
        try:
            url = urljoin(self.tracker_url, '/api/v1/torrents/torrent')

            response = self.session.get(
                url,
                params={'repo_id': repo_id, 'revision': revision, 'repo_type': repo_type},
                timeout=self.timeout
            )

            if response.status_code == 404:
                logger.debug(f"No .torrent file found for {repo_id}@{revision}")
                return None

            response.raise_for_status()
            logger.info(f"Downloaded .torrent for {repo_id}@{revision} ({len(response.content)} bytes)")
            return response.content

        except requests.RequestException as e:
            logger.warning(f"Failed to download .torrent: {e}")
            return None

    def register_torrent(
        self,
        repo_id: str,
        revision: str,
        repo_type: str,
        name: str,
        info_hash: str,
        total_size: int,
        file_count: int,
        piece_length: int,
        torrent_data: bytes,
        files: list,
    ) -> bool:
        """
        Register a new torrent with the tracker.

        Args:
            repo_id: HuggingFace repository ID.
            revision: Git commit hash or branch name.
            repo_type: "model", "dataset", or "space".
            name: Display name for the model.
            info_hash: BitTorrent info hash.
            total_size: Total size in bytes.
            file_count: Number of files in the torrent.
            piece_length: Piece length in bytes.
            torrent_data: Raw .torrent file bytes (will be base64-encoded for transport).
            files: List of dicts with 'path' and 'size' keys.

        Returns:
            True if registration successful, False otherwise.
        """
        import base64

        try:
            url = urljoin(self.tracker_url, '/api/v1/publish')
            data = {
                'repo_id': repo_id,
                'revision': revision,
                'repo_type': repo_type,
                'name': name,
                'info_hash': info_hash,
                'total_size': total_size,
                'file_count': file_count,
                'piece_length': piece_length,
                'torrent_data': base64.b64encode(torrent_data).decode('ascii'),
                'files': files,
            }

            logger.debug(f"Registering torrent: {url}")

            response = self.session.post(
                url,
                json=data,
                timeout=self.timeout
            )

            response.raise_for_status()
            logger.info(f"Successfully registered torrent for {repo_id} (revision: {revision})")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to register torrent: {e}")
            return False
