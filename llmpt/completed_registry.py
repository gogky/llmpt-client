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
        manifest = item.get("manifest")
        if not isinstance(manifest, list):
            manifest = []
        normalized = {
            "repo_type": item.get("repo_type", "model"),
            "repo_id": item.get("repo_id"),
            "revision": item.get("revision"),
            "storage_kind": item.get("storage_kind"),
            "storage_root": item.get("storage_root"),
            "cache_dir": item.get("cache_dir"),
            "local_dir": item.get("local_dir"),
            "manifest": sorted(
                {
                    str(path).replace("\\", "/")
                    for path in manifest
                    if isinstance(path, str) and path
                }
            ),
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
        manifest = normalized.get("manifest") or []
        normalized["manifest"] = sorted(
            {
                str(path).replace("\\", "/")
                for path in manifest
                if isinstance(path, str) and path
            }
        )
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
        derived_manifest = _local_dir_manifest(storage_root, revision)
    else:
        storage_kind = "hub_cache"
        storage_root = _normalize_path(cache_dir or HF_HUB_CACHE)
        derived_manifest = _snapshot_manifest(
            _hub_snapshot_dir(
                repo_id,
                revision,
                repo_type=repo_type,
                cache_dir=storage_root,
            )
        )

    manifest_list = sorted(
        {
            str(path).replace("\\", "/")
            for path in (manifest if manifest is not None else derived_manifest)
            if isinstance(path, str) and path
        }
    )
    if not manifest_list:
        logger.warning(
            f"Refusing to register completed source without manifest: {repo_id}@{revision}"
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
            return True
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
            manifest = item.get("manifest") or []
            return [
                str(path).replace("\\", "/")
                for path in manifest
                if isinstance(path, str) and path
            ]
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
