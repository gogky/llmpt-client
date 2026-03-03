"""
Utility functions.
"""

import hashlib
import logging
import re
import threading

# Centralized libtorrent availability check.
# All modules should use: from .utils import lt, LIBTORRENT_AVAILABLE
try:
    import libtorrent as lt
    LIBTORRENT_AVAILABLE = True
except ImportError:
    LIBTORRENT_AVAILABLE = False
    lt = None
from pathlib import Path
from typing import Optional

logger = logging.getLogger('llmpt.utils')

# ── Revision resolution ────────────────────────────────────────────────────────
# Matches a 40-character lowercase hexadecimal string (Git commit hash format).
_COMMIT_HASH_RE = re.compile(r'^[0-9a-f]{40}$')

# In-memory cache: (repo_id, revision, repo_type) → resolved commit hash.
# Populated lazily by resolve_commit_hash(). Never expires within a process
# because commit hashes are immutable.
_revision_cache: dict = {}
_revision_cache_lock = threading.Lock()


def resolve_commit_hash(
    repo_id: str,
    revision: str = 'main',
    repo_type: str = 'model',
) -> str:
    """Resolve a branch name, tag, or partial ref to a full 40-char commit hash.

    If *revision* is already a 40-character hex string it is returned as-is
    (no network call).  Otherwise ``HfApi.repo_info()`` is called once and the
    result is cached for the lifetime of the process.

    Args:
        repo_id: HuggingFace repository ID (e.g. ``"meta-llama/Llama-2-7b"``).
        revision: Branch name, tag, or commit hash to resolve.
        repo_type: Repository type (``"model"``, ``"dataset"``, ``"space"``).

    Returns:
        The resolved 40-character commit hash.

    Raises:
        Exception: If the HuggingFace Hub API call fails (caller should handle).
    """
    # Fast-path: already a full commit hash
    if _COMMIT_HASH_RE.match(revision):
        return revision

    cache_key = (repo_id, revision, repo_type)

    with _revision_cache_lock:
        if cache_key in _revision_cache:
            return _revision_cache[cache_key]

    # Call HuggingFace Hub API to resolve the revision
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo_id=repo_id, revision=revision, repo_type=repo_type)
    commit_hash = info.sha

    if not commit_hash or not _COMMIT_HASH_RE.match(commit_hash):
        raise ValueError(
            f"HfApi.repo_info() returned unexpected sha '{commit_hash}' "
            f"for {repo_id}@{revision}"
        )

    with _revision_cache_lock:
        _revision_cache[cache_key] = commit_hash

    logger.info(
        f"Resolved revision '{revision}' -> '{commit_hash}' for {repo_id}"
    )
    return commit_hash


def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of a file.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm ('sha256', 'sha1', 'md5').

    Returns:
        Hex digest of the file hash.
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string.

    Args:
        bytes_value: Number of bytes.

    Returns:
        Formatted string (e.g., "1.5 GB").
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def get_optimal_piece_length(file_size: int) -> int:
    """
    Calculate optimal piece length for a torrent based on file size.

    Args:
        file_size: File size in bytes.

    Returns:
        Optimal piece length in bytes.

    Note:
        - Small files (<100MB): 256KB
        - Medium files (100MB-1GB): 1MB
        - Large files (1GB-10GB): 4MB
        - Very large files (>10GB): 16MB
    """
    if file_size < 100 * 1024 * 1024:  # <100MB
        return 256 * 1024  # 256KB
    elif file_size < 1024 * 1024 * 1024:  # <1GB
        return 1024 * 1024  # 1MB
    elif file_size < 10 * 1024 * 1024 * 1024:  # <10GB
        return 4 * 1024 * 1024  # 4MB
    else:  # >10GB
        return 16 * 1024 * 1024  # 16MB
