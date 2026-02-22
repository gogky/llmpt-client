"""
P2P downloader using libtorrent.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Any

try:
    import libtorrent as lt
    LIBTORRENT_AVAILABLE = True
except ImportError:
    LIBTORRENT_AVAILABLE = False
    lt = None

logger = logging.getLogger('llmpt.downloader')


def try_p2p_download(
    repo_id: str,
    filename: str,
    url: str,
    temp_file: Any,
    tracker: Any,
    config: dict,
    resume_size: int = 0,
    expected_size: Optional[int] = None,
    **kwargs
) -> bool:
    """
    Attempt to download file via P2P.

    Args:
        repo_id: HuggingFace repository ID.
        filename: File name to download.
        url: HTTP URL (for fallback).
        temp_file: File object to write to.
        tracker: TrackerClient instance.
        config: Configuration dictionary.
        resume_size: Bytes already downloaded (for resume).
        expected_size: Expected file size in bytes.
        **kwargs: Additional arguments from http_get.

    Returns:
        True if P2P download succeeded, False otherwise.
    """
    if not LIBTORRENT_AVAILABLE:
        logger.warning("libtorrent not available")
        return False

    # Query tracker for torrent info
    # Note: We need commit_hash, but it's not available here yet
    # For now, query without commit_hash (tracker returns latest)
    torrent_info = tracker.get_torrent_info(repo_id, filename)

    if not torrent_info:
        logger.info(f"No torrent found for {repo_id}/{filename}")
        return False

    magnet_link = torrent_info.get('magnet_link')
    if not magnet_link:
        logger.warning("Torrent info missing magnet_link")
        return False

    # Start P2P download
    try:
        downloader = P2PDownloader(
            magnet_link=magnet_link,
            output_file=temp_file,
            tracker_url=tracker.tracker_url,
            timeout=config.get('timeout', 300),
        )

        success = downloader.download()

        if success and config.get('auto_seed', True):
            # Start seeding in background
            from .seeder import start_seeding
            start_seeding(
                torrent_info=torrent_info,
                file_path=temp_file.name,
                duration=config.get('seed_duration', 3600)
            )

        return success

    except Exception as e:
        logger.error(f"P2P download failed: {e}")
        return False


class P2PDownloader:
    """P2P downloader using libtorrent."""

    def __init__(
        self,
        magnet_link: str,
        output_file: Any,
        tracker_url: str,
        timeout: int = 300,
    ):
        """
        Initialize P2P downloader.

        Args:
            magnet_link: Magnet link to download.
            output_file: File object to write to.
            tracker_url: Tracker URL.
            timeout: Download timeout in seconds.
        """
        self.magnet_link = magnet_link
        self.output_file = output_file
        self.tracker_url = tracker_url
        self.timeout = timeout
        self.session = None
        self.handle = None

    def download(self) -> bool:
        """
        Execute P2P download.

        Returns:
            True if download succeeded, False otherwise.
        """
        try:
            # Create libtorrent session
            self.session = lt.session()
            self.session.listen_on(6881, 6891)

            # Add torrent from magnet link
            params = lt.parse_magnet_uri(self.magnet_link)

            # Set download path to temp directory
            temp_path = Path(self.output_file.name).parent
            params.save_path = str(temp_path)

            self.handle = self.session.add_torrent(params)

            logger.info(f"Starting P2P download: {self.magnet_link[:50]}...")

            # Wait for download to complete
            start_time = time.time()
            last_progress = 0

            while not self.handle.is_seed():
                status = self.handle.status()

                # Check timeout
                if time.time() - start_time > self.timeout:
                    logger.warning(f"P2P download timeout after {self.timeout}s")
                    return False

                # Log progress
                if status.progress != last_progress:
                    logger.info(
                        f"P2P progress: {status.progress * 100:.1f}% "
                        f"({status.download_rate / 1024:.1f} KB/s, "
                        f"peers: {status.num_peers})"
                    )
                    last_progress = status.progress

                # Check if download is stalled (no peers)
                if status.num_peers == 0 and time.time() - start_time > 30:
                    logger.warning("No peers found, P2P download stalled")
                    return False

                time.sleep(1)

            logger.info("P2P download completed successfully")

            # Copy downloaded file to output_file
            # (libtorrent saves to its own location)
            downloaded_path = Path(temp_path) / self.handle.name()
            if downloaded_path.exists():
                with open(downloaded_path, 'rb') as src:
                    self.output_file.write(src.read())
                downloaded_path.unlink()  # Clean up

            return True

        except Exception as e:
            logger.error(f"P2P download error: {e}")
            return False

        finally:
            # Clean up session
            if self.session and self.handle:
                self.session.remove_torrent(self.handle)
