"""
Transfer-related value objects.

These lightweight dataclasses keep target identity, source identity, and
session identity explicit so future routing changes do not have to lean on
ad-hoc tuples and loosely-typed dictionaries.
"""

from dataclasses import dataclass
from typing import Optional


_VALID_STORAGE_KINDS = frozenset({"hub_cache", "local_dir"})


@dataclass(frozen=True)
class StorageIdentity:
    """Normalized description of the local storage root backing a transfer."""

    kind: str
    root: str

    def __post_init__(self) -> None:
        if self.kind not in _VALID_STORAGE_KINDS:
            raise ValueError(f"unsupported storage kind: {self.kind}")
        if not self.root:
            raise ValueError("storage root must be non-empty")

    def as_legacy_tuple(self) -> tuple[str, str]:
        """Return the historical tuple shape used before explicit models."""
        return (self.kind, self.root)


@dataclass(frozen=True)
class LogicalTorrentRef:
    """Logical identity shared by all copies of the same torrent."""

    repo_type: str
    repo_id: str
    revision: str

    def as_legacy_tuple(self) -> tuple[str, str, str]:
        """Return the historical tuple shape used before explicit models."""
        return (self.repo_type, self.repo_id, self.revision)


@dataclass(frozen=True)
class SourceSessionKey:
    """Identity of one libtorrent-backed source session in local storage."""

    logical: LogicalTorrentRef
    storage: StorageIdentity

    @property
    def repo_type(self) -> str:
        return self.logical.repo_type

    @property
    def repo_id(self) -> str:
        return self.logical.repo_id

    @property
    def revision(self) -> str:
        return self.logical.revision

    @property
    def storage_kind(self) -> str:
        return self.storage.kind

    @property
    def storage_root(self) -> str:
        return self.storage.root

    def as_legacy_tuple(self) -> tuple[str, str, str, str, str]:
        """Return the historical tuple shape used before explicit models."""
        return self.logical.as_legacy_tuple() + self.storage.as_legacy_tuple()


@dataclass(frozen=True)
class TorrentSourceRef:
    """Source torrent chosen to fulfill a file transfer."""

    logical: LogicalTorrentRef
    storage: Optional[StorageIdentity] = None

    @property
    def repo_type(self) -> str:
        return self.logical.repo_type

    @property
    def repo_id(self) -> str:
        return self.logical.repo_id

    @property
    def revision(self) -> str:
        return self.logical.revision


@dataclass(frozen=True)
class TargetFileRequest:
    """One logical file request made by the Hugging Face download path."""

    logical: LogicalTorrentRef
    filename: str
    destination: str
    storage: StorageIdentity

    @property
    def repo_type(self) -> str:
        return self.logical.repo_type

    @property
    def repo_id(self) -> str:
        return self.logical.repo_id

    @property
    def revision(self) -> str:
        return self.logical.revision


@dataclass(frozen=True)
class TransferPlan:
    """Plan describing how one target file request should be fulfilled."""

    target: TargetFileRequest
    source: TorrentSourceRef


@dataclass(frozen=True)
class TransferResult:
    """Outcome of attempting to execute one transfer plan."""

    success: bool
    delivered_path: Optional[str] = None
    via: str = "p2p"
    error: Optional[str] = None
