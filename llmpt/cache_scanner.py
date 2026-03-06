"""
HuggingFace cache scanner.

Discovers all complete model snapshots in the local HuggingFace Hub cache
(~/.cache/huggingface/hub/) that are eligible for BitTorrent seeding.

Cache layout::

    ~/.cache/huggingface/hub/
    ├── models--gpt2/
    │   ├── refs/
    │   │   └── main          (contains commit hash text)
    │   ├── snapshots/
    │   │   └── abc123.../    (40-char commit hash directory)
    │   │       ├── config.json    → ../../blobs/xxx  (symlink)
    │   │       └── model.safetensors → ../../blobs/yyy
    │   └── blobs/
    │       ├── xxx
    │       └── yyy
    └── models--meta-llama--Llama-2-7b/
        └── ...
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("llmpt.cache_scanner")

# Default HF cache directory
HF_HUB_CACHE = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join("~", ".cache", "huggingface", "hub"))
)

# 40-char lowercase hex commit hash
_COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{40}$")


def _parse_repo_id(dirname: str) -> Optional[str]:
    """Convert a HuggingFace cache directory name to a repo_id.

    Examples:
        ``models--gpt2``                      → ``gpt2``
        ``models--meta-llama--Llama-2-7b``    → ``meta-llama/Llama-2-7b``
        ``datasets--squad``                   → ``squad``

    Returns:
        The repository ID, or ``None`` if the directory name doesn't follow
        the expected pattern.
    """
    # Expected format: {type}--{org}--{name} or {type}--{name}
    parts = dirname.split("--", 1)
    if len(parts) < 2:
        return None

    repo_type_prefix = parts[0]  # e.g. "models", "datasets", "spaces"
    if repo_type_prefix not in ("models", "datasets", "spaces"):
        return None

    # The remaining part uses "--" as separator for org/name
    # e.g. "meta-llama--Llama-2-7b" → "meta-llama/Llama-2-7b"
    # But "gpt2" stays as "gpt2"
    repo_name = parts[1].replace("--", "/", 1)
    return repo_name


def _is_snapshot_complete(snapshot_dir: Path) -> bool:
    """Check if a snapshot directory is complete.

    A snapshot is considered complete when:
    - It contains at least one file/symlink
    - All symlinks resolve to existing files (blobs)

    Returns:
        True if the snapshot is fully intact, False otherwise.
    """
    entries = list(snapshot_dir.iterdir())
    if not entries:
        return False

    for entry in entries:
        # Skip .gitattributes and other hidden files
        if entry.name.startswith("."):
            continue

        if entry.is_symlink():
            # Symlink must resolve to an existing blob
            target = entry.resolve()
            if not target.exists():
                logger.debug(
                    f"Incomplete snapshot: broken symlink {entry.name} "
                    f"→ {target}"
                )
                return False
        elif entry.is_dir():
            # Subdirectories (e.g. "unet/") — recurse
            if not _is_snapshot_complete(entry):
                return False

    return True


def scan_hf_cache(
    cache_dir: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Scan the HuggingFace cache and return seedable (repo_id, revision) pairs.

    Args:
        cache_dir: Path to the HF hub cache directory.  Defaults to
                   ``~/.cache/huggingface/hub`` (or ``$HF_HOME``).

    Returns:
        List of ``(repo_id, commit_hash)`` tuples for all complete snapshots.
    """
    hub_dir = Path(cache_dir or HF_HUB_CACHE)
    if not hub_dir.exists():
        logger.info(f"HF cache directory not found: {hub_dir}")
        return []

    seedable: List[Tuple[str, str]] = []

    for model_dir in sorted(hub_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        repo_id = _parse_repo_id(model_dir.name)
        if not repo_id:
            continue

        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists() or not snapshots_dir.is_dir():
            continue

        for snapshot in sorted(snapshots_dir.iterdir()):
            if not snapshot.is_dir():
                continue

            # Only consider 40-char commit hash directories
            if not _COMMIT_HASH_RE.match(snapshot.name):
                logger.debug(
                    f"Skipping non-commit-hash snapshot dir: {snapshot.name}"
                )
                continue

            if _is_snapshot_complete(snapshot):
                seedable.append((repo_id, snapshot.name))
                logger.debug(f"Seedable: {repo_id}@{snapshot.name[:8]}...")
            else:
                logger.debug(
                    f"Incomplete snapshot, skipping: {repo_id}@{snapshot.name[:8]}..."
                )

    logger.info(f"Scan complete: {len(seedable)} seedable models found")
    return seedable
