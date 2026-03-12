"""
Persistent registry of HuggingFace revisions verified as complete downloads.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger("llmpt.completed_registry")

COMPLETED_SOURCES_FILE = os.path.expanduser("~/.cache/llmpt/completed_sources.json")


def _resolve_hf_hub_cache() -> str:
    try:
        from huggingface_hub import constants

        return constants.HF_HUB_CACHE
    except (ImportError, AttributeError):
        pass

    hf_home = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return os.path.join(hf_home, "hub")


HF_HUB_CACHE = _resolve_hf_hub_cache()


def _normalize_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _normalize_manifest(manifest: Optional[Iterable[str]]) -> List[str]:
    return sorted(
        {
            str(path).replace("\\", "/")
            for path in (manifest or [])
            if isinstance(path, str) and path
        }
    )


def _load_payload() -> List[dict]:
    if not os.path.exists(COMPLETED_SOURCES_FILE):
        return []

    try:
        with open(COMPLETED_SOURCES_FILE, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load completed source registry: {exc}")
        return []

    if not isinstance(payload, list):
        return []

    entries: List[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        normalized = {
            "repo_type": item.get("repo_type", "model"),
            "repo_id": item.get("repo_id"),
            "revision": item.get("revision"),
            "storage_kind": item.get("storage_kind"),
            "storage_root": item.get("storage_root"),
            "cache_dir": item.get("cache_dir"),
            "local_dir": item.get("local_dir"),
            "manifest": _normalize_manifest(item.get("manifest")),
            "captured_at": float(item.get("captured_at", 0.0) or 0.0),
        }
        if not (
            normalized["repo_id"]
            and normalized["revision"]
            and normalized["storage_kind"]
            and normalized["storage_root"]
        ):
            continue
        if normalized["cache_dir"]:
            normalized["cache_dir"] = _normalize_path(normalized["cache_dir"])
        if normalized["local_dir"]:
            normalized["local_dir"] = _normalize_path(normalized["local_dir"])
        normalized["storage_root"] = _normalize_path(normalized["storage_root"])
        entries.append(normalized)
    return entries


def save_completed_sources(entries: Iterable[dict]) -> None:
    payload = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        if normalized.get("storage_root"):
            normalized["storage_root"] = _normalize_path(normalized["storage_root"])
        if normalized.get("cache_dir"):
            normalized["cache_dir"] = _normalize_path(normalized["cache_dir"])
        if normalized.get("local_dir"):
            normalized["local_dir"] = _normalize_path(normalized["local_dir"])
        normalized["manifest"] = _normalize_manifest(normalized.get("manifest"))
        normalized["captured_at"] = float(normalized.get("captured_at", 0.0) or time.time())
        payload.append(normalized)

    os.makedirs(os.path.dirname(COMPLETED_SOURCES_FILE), exist_ok=True)
    tmp_path = COMPLETED_SOURCES_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, COMPLETED_SOURCES_FILE)


def load_completed_sources() -> List[dict]:
    return _load_payload()


def _hub_snapshot_dir(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
) -> Path:
    from huggingface_hub.file_download import repo_folder_name

    hub_root = _normalize_path(cache_dir or HF_HUB_CACHE)
    repo_folder = repo_folder_name(repo_id=repo_id, repo_type=repo_type)
    return Path(hub_root) / repo_folder / "snapshots" / revision


def _snapshot_manifest(snapshot_dir: Path) -> List[str]:
    if not snapshot_dir.is_dir():
        return []

    manifest = []
    for path in sorted(snapshot_dir.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        manifest.append(path.relative_to(snapshot_dir).as_posix())
    return manifest


def _local_dir_manifest(local_dir: str, revision: str) -> List[str]:
    metadata_root = Path(local_dir) / ".cache" / "huggingface" / "download"
    if not metadata_root.is_dir():
        return []

    manifest = []
    for metadata_path in sorted(metadata_root.rglob("*.metadata")):
        try:
            with metadata_path.open("r") as f:
                commit_hash = f.readline().strip()
            if commit_hash != revision:
                continue
            relative = metadata_path.relative_to(metadata_root).with_suffix("")
            file_path = Path(local_dir) / relative
            if file_path.exists():
                manifest.append(relative.as_posix())
        except OSError:
            continue
    return manifest


def load_upstream_manifest(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
) -> List[str]:
    from huggingface_hub import snapshot_download

    dry_run = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        dry_run=True,
    )
    manifest = _normalize_manifest(
        item.filename
        for item in dry_run
        if getattr(item, "filename", None)
    )
    if not manifest:
        raise ValueError(f"empty upstream manifest for {repo_id}@{revision}")
    return manifest


def get_current_storage_manifest(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> List[str]:
    if local_dir:
        return _normalize_manifest(_local_dir_manifest(_normalize_path(local_dir), revision))

    snapshot_dir = _hub_snapshot_dir(
        repo_id,
        revision,
        repo_type=repo_type,
        cache_dir=cache_dir,
    )
    return _normalize_manifest(_snapshot_manifest(snapshot_dir))


def _hub_cache_files_present(
    repo_id: str,
    revision: str,
    manifest: List[str],
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False

    lookup_repo_type = repo_type if repo_type != "model" else None
    for filename in manifest:
        resolved = try_to_load_from_cache(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type=lookup_repo_type,
            cache_dir=cache_dir,
        )
        if not resolved or not os.path.exists(resolved):
            return False
    return True


def is_completed_entry_current(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False

    repo_type = entry.get("repo_type", "model")
    repo_id = entry.get("repo_id")
    revision = entry.get("revision")
    storage_kind = entry.get("storage_kind")
    manifest = _normalize_manifest(entry.get("manifest"))
    if not (repo_id and revision and storage_kind and manifest):
        return False

    if storage_kind == "local_dir":
        local_dir = entry.get("local_dir") or entry.get("storage_root")
        if not local_dir or not os.path.isdir(local_dir):
            return False
        current_manifest = get_current_storage_manifest(
            repo_id,
            revision,
            repo_type=repo_type,
            local_dir=local_dir,
        )
        return current_manifest == manifest

    if storage_kind == "hub_cache":
        cache_dir = entry.get("cache_dir") or entry.get("storage_root")
        if not cache_dir or not os.path.isdir(cache_dir):
            return False
        current_manifest = get_current_storage_manifest(
            repo_id,
            revision,
            repo_type=repo_type,
            cache_dir=cache_dir,
        )
        if current_manifest != manifest:
            return False
        return _hub_cache_files_present(
            repo_id,
            revision,
            manifest,
            repo_type=repo_type,
            cache_dir=cache_dir,
        )

    return False


def register_completed_source(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    manifest: Optional[Iterable[str]] = None,
) -> bool:
    if local_dir:
        storage_kind = "local_dir"
        storage_root = _normalize_path(local_dir)
    else:
        storage_kind = "hub_cache"
        storage_root = _normalize_path(cache_dir or HF_HUB_CACHE)
    current_manifest = get_current_storage_manifest(
        repo_id,
        revision,
        repo_type=repo_type,
        cache_dir=storage_root if storage_kind == "hub_cache" else None,
        local_dir=storage_root if storage_kind == "local_dir" else None,
    )
    manifest_list = _normalize_manifest(manifest)
    if not manifest_list:
        try:
            manifest_list = load_upstream_manifest(
                repo_id,
                revision,
                repo_type=repo_type,
            )
        except Exception as exc:
            logger.warning(
                f"Refusing to register completed source without upstream manifest: "
                f"{repo_id}@{revision} ({exc})"
            )
            return False
    if not manifest_list:
        logger.warning(
            f"Refusing to register completed source without manifest: {repo_id}@{revision}"
        )
        return False
    if current_manifest != manifest_list:
        logger.warning(
            f"Refusing to register completed source with mismatched manifest: "
            f"{repo_id}@{revision} (current={len(current_manifest)}, expected={len(manifest_list)})"
        )
        return False
    if storage_kind == "hub_cache" and not _hub_cache_files_present(
        repo_id,
        revision,
        manifest_list,
        repo_type=repo_type,
        cache_dir=storage_root,
    ):
        logger.warning(
            f"Refusing to register completed hub_cache source with unresolved files: "
            f"{repo_id}@{revision}"
        )
        return False

    entry = {
        "repo_type": repo_type or "model",
        "repo_id": repo_id,
        "revision": revision,
        "storage_kind": storage_kind,
        "storage_root": storage_root,
        "cache_dir": storage_root if storage_kind == "hub_cache" else None,
        "local_dir": storage_root if storage_kind == "local_dir" else None,
        "manifest": manifest_list,
        "captured_at": time.time(),
    }

    entries = _load_payload()
    kept = []
    for item in entries:
        if (
            item.get("repo_type") == entry["repo_type"]
            and item.get("repo_id") == entry["repo_id"]
            and item.get("revision") == entry["revision"]
            and item.get("storage_kind") == entry["storage_kind"]
            and item.get("storage_root") == entry["storage_root"]
        ):
            continue
        kept.append(item)
    kept.append(entry)
    save_completed_sources(kept)
    return True


def has_completed_source(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> bool:
    storage_kind = "local_dir" if local_dir else "hub_cache"
    storage_root = _normalize_path(local_dir or cache_dir or HF_HUB_CACHE)
    for item in _load_payload():
        if (
            item.get("repo_type") == (repo_type or "model")
            and item.get("repo_id") == repo_id
            and item.get("revision") == revision
            and item.get("storage_kind") == storage_kind
            and item.get("storage_root") == storage_root
        ):
            return is_completed_entry_current(item)
    return False


def get_completed_manifest(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> Optional[List[str]]:
    storage_kind = "local_dir" if local_dir else "hub_cache"
    storage_root = _normalize_path(local_dir or cache_dir or HF_HUB_CACHE)
    for item in _load_payload():
        if (
            item.get("repo_type") == (repo_type or "model")
            and item.get("repo_id") == repo_id
            and item.get("revision") == revision
            and item.get("storage_kind") == storage_kind
            and item.get("storage_root") == storage_root
        ):
            if not is_completed_entry_current(item):
                return None
            return _normalize_manifest(item.get("manifest"))
    return None


def forget_completed_source(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> int:
    storage_kind = "local_dir" if local_dir else "hub_cache"
    storage_root = _normalize_path(local_dir or cache_dir or HF_HUB_CACHE)
    entries = _load_payload()
    kept = []
    removed = 0
    for item in entries:
        if (
            item.get("repo_type") == (repo_type or "model")
            and item.get("repo_id") == repo_id
            and item.get("revision") == revision
            and item.get("storage_kind") == storage_kind
            and item.get("storage_root") == storage_root
        ):
            removed += 1
            continue
        kept.append(item)

    if removed:
        save_completed_sources(kept)
    return removed
