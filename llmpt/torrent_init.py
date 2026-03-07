"""
Torrent initialization helpers.

These functions encapsulate the individual steps of torrent session setup
(data acquisition, add_torrent_params construction, test-peer resolution)
so they can be unit-tested independently of SessionContext.
"""

import os
import logging
import socket
from typing import Optional

from .utils import lt

logger = logging.getLogger(__name__)


def acquire_torrent_data(
    repo_id: str,
    revision: str,
    tracker_client: object,
    supplied_data: Optional[bytes],
    *,
    repo_type: str = "model",
) -> Optional[bytes]:
    """Resolve torrent data from supplied bytes or three-layer cache.

    Priority order:
      1. *supplied_data* (passed directly by the seeder path)
      2. Three-layer lookup via ``torrent_cache.resolve_torrent_data``
         (local cache → tracker → None)

    Args:
        repo_id: HuggingFace repository ID.
        revision: Commit hash or branch name.
        tracker_client: TrackerClient instance for remote lookups.
        supplied_data: Pre-existing raw .torrent bytes, or None.

    Returns:
        Raw .torrent bytes, or None if not available from any source.
    """
    if supplied_data:
        return supplied_data

    from .torrent_cache import resolve_torrent_data
    return resolve_torrent_data(repo_id, revision, tracker_client, repo_type=repo_type)


def build_add_torrent_params(
    torrent_data: bytes,
    save_path: str,
    session_mode: str,
    fastresume_path: str,
    repo_id: str,
) -> tuple:
    """Construct libtorrent ``add_torrent_params`` from raw torrent data.

    Args:
        torrent_data: Raw .torrent file bytes.
        save_path: Directory where libtorrent writes downloaded pieces.
        session_mode: ``'on_demand'`` or ``'full_seed'``.
        fastresume_path: Path to fastresume file (loaded only for on_demand).
        repo_id: Repository ID (for logging only).

    Returns:
        A ``(params, torrent_info)`` tuple.  *params* is ready for
        ``lt_session.add_torrent()``.
    """
    info = lt.torrent_info(lt.bdecode(torrent_data))
    params = lt.add_torrent_params()
    params.ti = info
    params.save_path = save_path

    # Start paused to set priorities before downloading anything
    params.flags |= lt.torrent_flags.paused

    # Full-seed sessions start in seed_mode; downloaders may load fastresume
    if session_mode == 'full_seed':
        params.flags |= lt.torrent_flags.seed_mode
    else:
        params = _load_fastresume(params, info, save_path, fastresume_path, repo_id)

    return params, info


def _load_fastresume(params, info, save_path: str, path: str, repo_id: str):
    """Attempt to load fastresume data into *params* (best-effort).

    Returns:
        The ``add_torrent_params`` object to use — either the original
        *params* (mutated in-place) or a brand-new object produced by
        ``lt.read_resume_data()``.
    """
    if not os.path.exists(path):
        return params

    try:
        with open(path, "rb") as f:
            resume_data = f.read()

        try:
            decoded = lt.bdecode(resume_data)
            if isinstance(decoded, dict):
                params.renamed_files = decoded.get(b'mapped_files', {})
        except Exception:
            pass

        if hasattr(lt.add_torrent_params, "parse_resume_data"):
            params = lt.read_resume_data(resume_data)
            params.save_path = save_path
            params.ti = info
            params.flags |= lt.torrent_flags.paused
            logger.info(f"[{repo_id}] Loaded fastresume data from {path}")
    except Exception as e:
        logger.warning(f"[{repo_id}] Failed to load resume data: {e}")

    return params


def resolve_test_peer(env_var: str = 'TEST_SEEDER_PEER') -> Optional[tuple]:
    """Parse test peer address from an environment variable.

    Supported formats:
      - ``host:port``
      - ``host``           (defaults to port 6881)
      - ``[ipv6]:port``

    Returns:
        ``(ip, port)`` tuple, or None if the env var is unset.
    """
    test_peer = os.environ.get(env_var)
    if not test_peer:
        return None

    try:
        if test_peer.startswith('['):
            bracket_end = test_peer.index(']')
            host = test_peer[1:bracket_end]
            port = int(test_peer[bracket_end + 2:]) if len(test_peer) > bracket_end + 2 else 6881
        elif ':' in test_peer:
            host, port_str = test_peer.rsplit(':', 1)
            port = int(port_str)
        else:
            host = test_peer
            port = 6881

        ip = socket.gethostbyname(host)
        return (ip, port)
    except Exception as e:
        logger.warning(f"Failed to resolve test peer {test_peer}: {e}")
        return None
