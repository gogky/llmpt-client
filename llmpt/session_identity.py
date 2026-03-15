"""
Helpers for explicit torrent/session identity construction.

These helpers preserve the current on-disk/session naming behavior while
removing hand-written tuple assembly from the call sites.
"""

import hashlib
import os
from typing import Optional

from .transfer_types import (
    LogicalTorrentRef,
    SourceSessionKey,
    StorageIdentity,
    TorrentSourceRef,
)
from .utils import get_hf_hub_cache


def normalize_storage_root(path: Optional[str]) -> Optional[str]:
    """Return a canonical absolute storage root path."""
    if not path:
        return None
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def build_storage_identity(
    *,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> StorageIdentity:
    """Build the normalized storage identity used by a transfer/session."""
    if local_dir:
        return StorageIdentity(
            kind="local_dir",
            root=normalize_storage_root(local_dir) or "",
        )
    if cache_dir:
        return StorageIdentity(
            kind="hub_cache",
            root=normalize_storage_root(cache_dir) or "",
        )
    return StorageIdentity(
        kind="hub_cache",
        root=get_hf_hub_cache(),
    )


def storage_identity_to_kwargs(storage: StorageIdentity) -> dict[str, str]:
    """Convert a storage identity back to SessionContext constructor kwargs."""
    if storage.kind == "local_dir":
        return {"local_dir": storage.root}
    if storage.kind == "hub_cache":
        return {"cache_dir": storage.root}
    raise ValueError(f"unsupported storage kind: {storage.kind}")


def build_logical_torrent_ref(
    repo_type: str,
    repo_id: str,
    revision: str,
) -> LogicalTorrentRef:
    """Build the logical identity shared by all copies of one torrent."""
    return LogicalTorrentRef(
        repo_type=repo_type,
        repo_id=repo_id,
        revision=revision,
    )


def build_torrent_source_ref(
    repo_type: str,
    repo_id: str,
    revision: str,
    *,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> TorrentSourceRef:
    """Build the source-torrent reference used by one source session."""
    return TorrentSourceRef(
        logical=build_logical_torrent_ref(repo_type, repo_id, revision),
        storage=build_storage_identity(cache_dir=cache_dir, local_dir=local_dir),
    )


def build_source_session_key(
    repo_type: str,
    repo_id: str,
    revision: str,
    *,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> SourceSessionKey:
    """Build the explicit key for one source-backed session."""
    return SourceSessionKey(
        logical=build_logical_torrent_ref(repo_type, repo_id, revision),
        storage=build_storage_identity(cache_dir=cache_dir, local_dir=local_dir),
    )


def build_fastresume_filename(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    session_mode: str = "on_demand",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> str:
    """Build the fastresume filename for one storage-backed torrent session."""
    storage = build_storage_identity(cache_dir=cache_dir, local_dir=local_dir)
    identity = "|".join(
        [
            repo_type or "model",
            repo_id,
            revision,
            session_mode,
            storage.kind,
            storage.root,
        ]
    )
    repo_slug = hashlib.sha1(repo_id.encode("utf-8")).hexdigest()[:8]
    identity_digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    return (
        f"{repo_type}_{repo_slug}_{revision}_{session_mode}_{identity_digest}.fastresume"
    )
