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
        commit_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query tracker for torrent information.

        Args:
            repo_id: HuggingFace repository ID (e.g., "meta-llama/Llama-2-7b").
            filename: File name within the repository.
            commit_hash: Git commit hash. If None, uses latest.

        Returns:
            Dictionary containing torrent info (magnet_link, info_hash, etc.)
            or None if torrent doesn't exist.

        Example:
            >>> client = TrackerClient("http://tracker.example.com")
            >>> info = client.get_torrent_info("gpt2", "config.json")
            >>> if info:
            ...     print(info['magnet_link'])
        """
        try:
            url = urljoin(self.tracker_url, '/api/v1/torrents')
            params = {
                'repo_id': repo_id,
                'filename': filename,
            }
            if commit_hash:
                params['commit_hash'] = commit_hash

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

            if data.get('torrents') and len(data['torrents']) > 0:
                torrent = data['torrents'][0]
                logger.info(
                    f"Found torrent for {repo_id}/{filename}: "
                    f"{torrent.get('info_hash', 'N/A')}"
                )
                return torrent
            else:
                logger.debug(f"No torrent found for {repo_id}/{filename}")
                return None

        except requests.RequestException as e:
            logger.warning(f"Failed to query tracker: {e}")
            return None

    def register_torrent(
        self,
        repo_id: str,
        filename: str,
        commit_hash: str,
        info_hash: str,
        magnet_link: str,
        file_size: int,
        piece_length: int,
    ) -> bool:
        """
        Register a new torrent with the tracker.

        Args:
            repo_id: HuggingFace repository ID.
            filename: File name within the repository.
            commit_hash: Git commit hash.
            info_hash: BitTorrent info hash.
            magnet_link: Magnet link for the torrent.
            file_size: File size in bytes.
            piece_length: Piece length in bytes.

        Returns:
            True if registration successful, False otherwise.

        Example:
            >>> client = TrackerClient("http://tracker.example.com")
            >>> success = client.register_torrent(
            ...     repo_id="gpt2",
            ...     filename="config.json",
            ...     commit_hash="abc123...",
            ...     info_hash="def456...",
            ...     magnet_link="magnet:?xt=...",
            ...     file_size=1234,
            ...     piece_length=16777216
            ... )
        """
        try:
            url = urljoin(self.tracker_url, '/api/v1/publish')
            data = {
                'repo_id': repo_id,
                'filename': filename,
                'commit_hash': commit_hash,
                'info_hash': info_hash,
                'magnet_link': magnet_link,
                'file_size': file_size,
                'piece_length': piece_length,
            }

            logger.debug(f"Registering torrent: {url}")

            response = self.session.post(
                url,
                json=data,
                timeout=self.timeout
            )

            response.raise_for_status()
            logger.info(f"Successfully registered torrent for {repo_id}/{filename}")
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
