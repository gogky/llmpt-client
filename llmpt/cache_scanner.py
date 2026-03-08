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

from dataclasses import dataclass
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("llmpt.cache_scanner")

# Default HF cache directory.
# HuggingFace convention: $HF_HOME/hub (not $HF_HOME directly).
# Try to use the official constant from huggingface_hub first.
def _resolve_hf_hub_cache() -> str:
    try:
        from huggingface_hub import constants
        return constants.HF_HUB_CACHE
    except (ImportError, AttributeError):
        pass
    hf_home = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return os.path.join(hf_home, "hub")

HF_HUB_CACHE = _resolve_hf_hub_cache()

# 40-char lowercase hex commit hash
_COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{40}$")

# Persistent registry for non-default storage locations that should survive
# daemon restarts.
KNOWN_STORAGE_FILE = os.path.expanduser("~/.cache/llmpt/known_storage.json")


@dataclass(frozen=True)
class SeedableSource:
    repo_type: str
    repo_id: str
    revision: str
    storage_kind: str
    storage_root: str
    cache_dir: Optional[str] = None
    local_dir: Optional[str] = None


def _normalize_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _default_registry() -> Dict[str, List[dict]]:
    return {"hub_cache_roots": [], "local_dir_sources": []}


def _load_storage_registry() -> Dict[str, List[dict]]:
    """Load the persistent storage registry.

    Backward compatibility:
    - Older versions stored a bare JSON list of hub cache roots.
    """
    if not os.path.exists(KNOWN_STORAGE_FILE):
        return _default_registry()

    try:
        with open(KNOWN_STORAGE_FILE, "r") as f:
            payload = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load storage registry: {e}")
        return _default_registry()

    if isinstance(payload, list):
        # Legacy schema: just a list of custom hub cache roots.
        return {
            "hub_cache_roots": [
                {"root": _normalize_path(root)}
                for root in payload
                if isinstance(root, str)
            ],
            "local_dir_sources": [],
        }

    if not isinstance(payload, dict):
        return _default_registry()

    registry = _default_registry()
    for item in payload.get("hub_cache_roots", []):
        if isinstance(item, str):
            registry["hub_cache_roots"].append({"root": _normalize_path(item)})
        elif isinstance(item, dict) and item.get("root"):
            registry["hub_cache_roots"].append({"root": _normalize_path(item["root"])})

    for item in payload.get("local_dir_sources", []):
        if not isinstance(item, dict):
            continue
        local_dir = item.get("local_dir")
        repo_id = item.get("repo_id")
        revision = item.get("revision")
        if not (local_dir and repo_id and revision):
            continue
        registry["local_dir_sources"].append(
            {
                "repo_type": item.get("repo_type", "model"),
                "repo_id": repo_id,
                "revision": revision,
                "local_dir": _normalize_path(local_dir),
            }
        )
    return registry


def _save_storage_registry(registry: Dict[str, List[dict]]) -> None:
    os.makedirs(os.path.dirname(KNOWN_STORAGE_FILE), exist_ok=True)
    tmp_path = KNOWN_STORAGE_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    os.replace(tmp_path, KNOWN_STORAGE_FILE)


def register_seedable_storage(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> None:
    """Persist non-default storage locations needed for daemon cold starts."""
    if not cache_dir and not local_dir:
        return

    try:
        registry = _load_storage_registry()

        if cache_dir:
            normalized_cache_dir = _normalize_path(cache_dir)
            roots = {
                item["root"]
                for item in registry["hub_cache_roots"]
                if item.get("root")
            }
            if normalized_cache_dir not in roots:
                registry["hub_cache_roots"].append({"root": normalized_cache_dir})

        if local_dir:
            normalized_local_dir = _normalize_path(local_dir)
            local_entry = {
                "repo_type": repo_type or "model",
                "repo_id": repo_id,
                "revision": revision,
                "local_dir": normalized_local_dir,
            }
            entries = [
                item
                for item in registry["local_dir_sources"]
                if not (
                    item.get("repo_type") == local_entry["repo_type"]
                    and item.get("repo_id") == local_entry["repo_id"]
                    and item.get("revision") == local_entry["revision"]
                    and item.get("local_dir") == local_entry["local_dir"]
                )
            ]
            entries.append(local_entry)
            registry["local_dir_sources"] = entries

        _save_storage_registry(registry)
    except Exception as e:
        logger.warning(
            f"Failed to register seedable storage for {repo_id}@{revision}: {e}"
        )


def _parse_repo_id(dirname: str) -> Optional[str]:
    """Convert a HuggingFace cache directory name to a repo_id.

    Examples:
        ``models--gpt2``                      → ``gpt2``
        ``models--meta-llama--Llama-2-7b``    → ``meta-llama/Llama-2-7b``
        ``datasets--squad``                   → ``squad``

    Returns:
        A tuple of `(repo_type, repo_id)`, or `None` if the directory name doesn't follow
        the expected pattern.
    """
    # Expected format: {type}--{org}--{name} or {type}--{name}
    parts = dirname.split("--", 1)
    if len(parts) < 2:
        return None

    repo_type_prefix = parts[0]  # e.g. "models", "datasets", "spaces"
    if repo_type_prefix not in ("models", "datasets", "spaces"):
        return None

    # "models--gpt2" -> ("model", "gpt2")
    # "datasets--squad" -> ("dataset", "squad")
    repo_type = repo_type_prefix.rstrip("s") # "models" -> "model", "datasets" -> "dataset", "spaces" -> "space"
    
    # The remaining part uses "--" as separator for org/name
    # e.g. "meta-llama--Llama-2-7b" → "meta-llama/Llama-2-7b"
    # But "gpt2" stays as "gpt2"
    repo_name = parts[1].replace("--", "/", 1)
    return repo_type, repo_name


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


def _scan_hf_cache_root(cache_dir: Optional[str]) -> List[SeedableSource]:
    """Scan one HF hub-style cache root and return structured seedable sources."""
    hub_dir = Path(cache_dir or HF_HUB_CACHE)
    if not hub_dir.exists():
        logger.info(f"HF cache directory not found: {hub_dir}")
        return []

    seedable: List[SeedableSource] = []
    for model_dir in sorted(hub_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        parsed = _parse_repo_id(model_dir.name)
        if not parsed:
            continue
        repo_type, repo_id = parsed

        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists() or not snapshots_dir.is_dir():
            continue

        for snapshot in sorted(snapshots_dir.iterdir()):
            if not snapshot.is_dir():
                continue

            if not _COMMIT_HASH_RE.match(snapshot.name):
                logger.debug(f"Skipping non-commit-hash snapshot dir: {snapshot.name}")
                continue

            if _is_snapshot_complete(snapshot):
                seedable.append(
                    SeedableSource(
                        repo_type=repo_type,
                        repo_id=repo_id,
                        revision=snapshot.name,
                        storage_kind="hub_cache",
                        storage_root=str(hub_dir),
                        cache_dir=str(hub_dir),
                    )
                )
            else:
                logger.debug(
                    f"Incomplete snapshot, skipping: {repo_id}@{snapshot.name[:8]}..."
                )
    return seedable


def _local_dir_matches_revision(local_dir: str, revision: str) -> bool:
    """Return True when the local_dir still has metadata for the requested revision."""
    metadata_root = Path(local_dir) / ".cache" / "huggingface" / "download"
    if not metadata_root.exists():
        return False

    for metadata_path in metadata_root.rglob("*.metadata"):
        try:
            with metadata_path.open("r") as f:
                commit_hash = f.readline().strip()
            if commit_hash != revision:
                continue

            relative = metadata_path.relative_to(metadata_root)
            file_relative = Path(str(relative)[: -len(".metadata")])
            if (Path(local_dir) / file_relative).exists():
                return True
        except OSError:
            continue
    return False


def scan_seedable_sources() -> List[SeedableSource]:
    """Scan default/custom hub caches plus registered local_dir sources."""
    seedable = _scan_hf_cache_root(None)
    registry = _load_storage_registry()

    seen_hub_roots = {_normalize_path(HF_HUB_CACHE)}
    seen_hub_roots.update(
        _normalize_path(source.storage_root)
        for source in seedable
        if source.storage_kind == "hub_cache"
    )

    # Track which registry entries are still valid for pruning.
    live_hub_roots: List[dict] = []
    for item in registry["hub_cache_roots"]:
        root = item.get("root")
        if not root:
            continue
        normalized_root = _normalize_path(root)
        if not os.path.isdir(normalized_root):
            logger.info(f"Pruning stale hub_cache_root from registry: {root}")
            continue
        live_hub_roots.append(item)
        if normalized_root in seen_hub_roots:
            continue
        seen_hub_roots.add(normalized_root)
        seedable.extend(_scan_hf_cache_root(normalized_root))

    seen_local_sources = set()
    live_local_sources: List[dict] = []
    for item in registry["local_dir_sources"]:
        local_dir = item["local_dir"]
        source_key = (
            item["repo_type"],
            item["repo_id"],
            item["revision"],
            local_dir,
        )
        if source_key in seen_local_sources:
            continue
        seen_local_sources.add(source_key)
        if not os.path.isdir(local_dir):
            logger.info(f"Pruning stale local_dir from registry: {local_dir}")
            continue
        if not _local_dir_matches_revision(local_dir, item["revision"]):
            logger.info(
                f"Pruning stale local_dir from registry (revision mismatch): "
                f"{item['repo_id']}@{item['revision'][:8]}... in {local_dir}"
            )
            continue
        live_local_sources.append(item)
        seedable.append(
            SeedableSource(
                repo_type=item["repo_type"],
                repo_id=item["repo_id"],
                revision=item["revision"],
                storage_kind="local_dir",
                storage_root=local_dir,
                local_dir=local_dir,
            )
        )

    # Prune stale entries from the registry (atomic write).
    if (
        len(live_hub_roots) != len(registry["hub_cache_roots"])
        or len(live_local_sources) != len(registry["local_dir_sources"])
    ):
        pruned = (
            len(registry["hub_cache_roots"]) - len(live_hub_roots)
            + len(registry["local_dir_sources"]) - len(live_local_sources)
        )
        logger.info(f"Pruned {pruned} stale entries from storage registry")
        _save_storage_registry(
            {"hub_cache_roots": live_hub_roots, "local_dir_sources": live_local_sources}
        )

    logger.info(f"Scan complete: {len(seedable)} seedable sources found")
    return seedable


def scan_hf_cache(
    cache_dir: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """Scan the HuggingFace cache and return seedable (repo_type, repo_id, revision) pairs.

    Args:
        cache_dir: Path to the HF hub cache directory.  Defaults to
                   ``~/.cache/huggingface/hub`` (or ``$HF_HOME``).

    Returns:
        List of ``(repo_type, repo_id, commit_hash)`` tuples for all complete snapshots.
    """
    sources = _scan_hf_cache_root(cache_dir)
    logger.info(f"Scan complete: {len(sources)} seedable models found")
    return [(item.repo_type, item.repo_id, item.revision) for item in sources]
