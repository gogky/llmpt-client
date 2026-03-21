"""
Tracker client for communicating with the llmpt tracker server.
"""

import logging
import requests
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin

from .transfer_types import LogicalTorrentRef, SourceFileCandidate, TorrentSourceRef

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

    def resolve_file_sources(
        self,
        repo_id: str,
        revision: str,
        filename: str,
        *,
        repo_type: str = 'model'
    ) -> List[SourceFileCandidate]:
        """Resolve candidate source files for one logical target file.

        The server-side API is expected to return file-level source candidates
        keyed by the target file's ``file_root``. Older trackers may not expose
        this endpoint yet; in that case we simply return an empty list and let
        the caller fall back to the target torrent itself.
        """
        try:
            url = urljoin(self.tracker_url, '/api/v1/file-sources')
            params = {
                'repo_id': repo_id,
                'revision': revision,
                'path': filename,
                'repo_type': repo_type,
            }

            logger.debug(f"Resolving file sources: {url} with params {params}")

            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout,
            )

            if response.status_code == 404:
                logger.debug(
                    "Tracker does not expose file sources for "
                    f"{repo_id}/{filename}@{revision}"
                )
                return []

            response.raise_for_status()
            payload = response.json()
            items = self._extract_candidate_items(payload)
            candidates = []
            for item in items:
                candidate = self._parse_source_file_candidate(item)
                if candidate is not None:
                    candidates.append(candidate)
            return candidates
        except requests.RequestException as e:
            logger.warning(f"Failed to resolve file sources: {e}")
            return []

    def _extract_candidate_items(self, payload: Any) -> list[dict[str, Any]]:
        """Extract candidate rows from either flat or wrapped tracker payloads."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if not isinstance(payload, dict):
            return []

        data = payload.get('data', payload)
        if isinstance(data, dict):
            candidates = data.get('candidates', [])
        elif isinstance(data, list):
            candidates = data
        else:
            candidates = []

        return [item for item in candidates if isinstance(item, dict)]

    def _parse_source_file_candidate(
        self,
        item: dict[str, Any],
    ) -> Optional[SourceFileCandidate]:
        """Best-effort parse of one file-source candidate returned by the tracker."""
        repo_id = item.get('repo_id') or item.get('source_repo_id')
        revision = item.get('revision') or item.get('source_revision')
        repo_type = item.get('repo_type') or item.get('source_repo_type') or 'model'
        filename = item.get('path') or item.get('filename') or item.get('source_path')
        if not (repo_id and revision and filename):
            return None

        size = item.get('size')
        seeders = item.get('seeders')
        score = item.get('score', 0.0)

        try:
            size = int(size) if size is not None else None
        except (TypeError, ValueError):
            size = None

        try:
            seeders = int(seeders) if seeders is not None else None
        except (TypeError, ValueError):
            seeders = None

        try:
            score = float(score or 0.0)
        except (TypeError, ValueError):
            score = 0.0

        return SourceFileCandidate(
            source=TorrentSourceRef(
                logical=LogicalTorrentRef(
                    repo_type=repo_type,
                    repo_id=repo_id,
                    revision=revision,
                ),
            ),
            filename=filename,
            file_root=item.get('file_root'),
            size=size,
            seeders=seeders,
            score=score,
        )

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
        announce_key: Optional[str] = None,
    ) -> bool:
        """
        Register a new torrent with the tracker.

        Args:
            repo_id: HuggingFace repository ID.
            revision: Git commit hash or branch name.
            repo_type: "model", "dataset", or "space".
            name: Display name for the model.
            info_hash: Canonical torrent identity stored by the application.
            total_size: Total size in bytes.
            file_count: Number of files in the torrent.
            piece_length: Piece length in bytes.
            torrent_data: Raw .torrent file bytes (will be base64-encoded for transport).
            files: List of dicts with 'path' and 'size' keys.
            announce_key: Optional tracker announce key. For pure v2 torrents this
                is the 20-byte truncated swarm key used on the wire.

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
            if announce_key:
                data['announce_key'] = announce_key

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
