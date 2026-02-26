"""
Tracker client for communicating with the llmpt tracker server.
"""

import logging
import requests
from typing import Optional, Dict, Any
from urllib.parse import urljoin

logger = logging.getLogger('llmpt.tracker')


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
        filename: str,
        revision: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query tracker for torrent information.

        Args:
            repo_id: HuggingFace repository ID (e.g., "meta-llama/Llama-2-7b").
            filename: File name within the repository.
            revision: Git commit hash or branch name. If None, uses latest.

        Returns:
            Dictionary containing torrent info (magnet_link, info_hash, etc.)
            or None if torrent doesn't exist.
        """
        try:
            url = urljoin(self.tracker_url, '/api/v1/torrents')
            # Server only filters by repo_id
            params = {'repo_id': repo_id}
            
            logger.debug(f"Querying tracker: {url} with params {params}")

            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 404:
                logger.debug(f"No torrent found for {repo_id}/{filename}")
                return None

            response.raise_for_status()
            data = response.json()

            # Server returns {"total": N, "data": [...]}
            torrents = data.get('data', [])
            
            if not torrents:
                logger.debug(f"No torrent found for {repo_id}")
                return None

            # Filter locally by revision if provided
            if revision:
                for t in torrents:
                    if t.get('revision') == revision:
                        logger.info(f"Found torrent for {repo_id} (revision: {revision}): {t.get('info_hash', 'N/A')}")
                        return t
                logger.debug(f"No torrent found for {repo_id} at revision {revision}")
                return None
            else:
                # Return the most recently created one (server sorts by created_at DESC)
                torrent = torrents[0]
                logger.info(f"Found latest torrent for {repo_id}: {torrent.get('info_hash', 'N/A')}")
                return torrent

        except requests.RequestException as e:
            logger.warning(f"Failed to query tracker: {e}")
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
        magnet_link: str,
        piece_length: int,
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
            magnet_link: Magnet link for the torrent.
            piece_length: Piece length in bytes.

        Returns:
            True if registration successful, False otherwise.
        """
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
                'magnet_link': magnet_link,
                'piece_length': piece_length,
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

    def announce(
        self,
        info_hash: str,
        peer_id: str,
        port: int,
        uploaded: int = 0,
        downloaded: int = 0,
        left: int = 0,
        event: str = 'started',
    ) -> Optional[Dict[str, Any]]:
        """
        Send announce request to tracker (BitTorrent protocol).

        Args:
            info_hash: Torrent info hash.
            peer_id: Client peer ID.
            port: Listening port.
            uploaded: Bytes uploaded.
            downloaded: Bytes downloaded.
            left: Bytes left to download.
            event: Event type ('started', 'completed', 'stopped').

        Returns:
            Dictionary containing peer list and interval, or None if failed.
        """
        try:
            url = urljoin(self.tracker_url, '/announce')
            params = {
                'info_hash': info_hash,
                'peer_id': peer_id,
                'port': port,
                'uploaded': uploaded,
                'downloaded': downloaded,
                'left': left,
                'event': event,
                'compact': 1,
            }

            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )

            response.raise_for_status()

            # Parse bencoded response (simplified - libtorrent handles this)
            # For now, just return success
            return {'interval': 1800}

        except requests.RequestException as e:
            logger.warning(f"Announce failed: {e}")
            return None
