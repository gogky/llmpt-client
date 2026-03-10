"""
Best-effort import of pre-existing HuggingFace cache entries into completed registry.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

from .cache_scanner import (
    SeedableSource,
    _load_storage_registry,
    _local_dir_matches_revision,
    _scan_hf_cache_root,
)
from .completed_registry import has_completed_source, register_completed_source

logger = logging.getLogger("llmpt.cache_importer")

IMPORT_STATE_FILE = os.path.expanduser("~/.cache/llmpt/cache_import_state.json")

PARTIAL_RETRY_DELAY = 30 * 60
BLOCKED_RETRY_DELAY = 6 * 60 * 60
ERROR_RETRY_DELAY = 60 * 60


class _QuietTqdm:
    """No-op tqdm compatible enough for HuggingFace dry-run calls."""

    _lock = None

    @classmethod
    def get_lock(cls):
        return getattr(cls, "_lock", None)

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock
        return lock

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        self.disable = True

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def close(self):
        return None

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)


def _normalize_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _candidate_key(source: SeedableSource) -> str:
    return "|".join(
        [
            source.repo_type,
            source.repo_id,
            source.revision,
            source.storage_kind,
            source.storage_root,
        ]
    )


def _load_import_state() -> Dict[str, dict]:
    if not os.path.exists(IMPORT_STATE_FILE):
        return {}

    try:
        with open(IMPORT_STATE_FILE, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load cache import state: {exc}")
        return {}

    if not isinstance(payload, dict):
        return {}
    return {str(k): v for k, v in payload.items() if isinstance(v, dict)}


def load_import_state() -> Dict[str, dict]:
    """Public read-only accessor for cache import status."""
    return _load_import_state()


def clear_import_state() -> None:
    """Clear cached import retry state so the next scan re-verifies candidates."""
    _save_import_state({})


def _save_import_state(state: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(IMPORT_STATE_FILE), exist_ok=True)
    tmp_path = IMPORT_STATE_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp_path, IMPORT_STATE_FILE)


def _collect_hub_candidates() -> List[SeedableSource]:
    candidates: List[SeedableSource] = []
    seen = set()

    def add_sources(items: Iterable[SeedableSource]) -> None:
        for item in items:
            key = (
                item.repo_type,
                item.repo_id,
                item.revision,
                item.storage_kind,
                item.storage_root,
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(item)

    add_sources(_scan_hf_cache_root(None))

    registry = _load_storage_registry()
    for item in registry.get("hub_cache_roots", []):
        root = item.get("root")
        if root and os.path.isdir(root):
            add_sources(_scan_hf_cache_root(root))

    return candidates


def _collect_local_dir_candidates() -> List[SeedableSource]:
    candidates: List[SeedableSource] = []
    seen = set()
    registry = _load_storage_registry()
    for item in registry.get("local_dir_sources", []):
        local_dir = item.get("local_dir")
        if not local_dir or not os.path.isdir(local_dir):
            continue
        key = (
            item.get("repo_type", "model"),
            item.get("repo_id"),
            item.get("revision"),
            "local_dir",
            _normalize_path(local_dir),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            SeedableSource(
                repo_type=item.get("repo_type", "model"),
                repo_id=item["repo_id"],
                revision=item["revision"],
                storage_kind="local_dir",
                storage_root=_normalize_path(local_dir),
                local_dir=_normalize_path(local_dir),
            )
        )
    return candidates


def _retry_delay_for(status: str) -> int:
    if status == "partial":
        return PARTIAL_RETRY_DELAY
    if status == "blocked":
        return BLOCKED_RETRY_DELAY
    return ERROR_RETRY_DELAY


def _record_import_state(
    state: Dict[str, dict],
    source: SeedableSource,
    *,
    status: str,
    reason: str,
    delay: Optional[int] = None,
) -> None:
    now = time.time()
    state[_candidate_key(source)] = {
        "repo_type": source.repo_type,
        "repo_id": source.repo_id,
        "revision": source.revision,
        "storage_kind": source.storage_kind,
        "storage_root": source.storage_root,
        "status": status,
        "reason": reason,
        "last_checked": now,
        "next_retry_ts": now + (delay if delay is not None else _retry_delay_for(status)),
    }


def _classify_exception(exc: Exception) -> Tuple[str, str]:
    from huggingface_hub import errors as hf_errors

    if isinstance(
        exc,
        (
            hf_errors.GatedRepoError,
            hf_errors.RepositoryNotFoundError,
            hf_errors.RevisionNotFoundError,
            hf_errors.DisabledRepoError,
        ),
    ):
        return "blocked", type(exc).__name__
    if isinstance(exc, hf_errors.HfHubHTTPError):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {401, 403, 404}:
            return "blocked", f"{type(exc).__name__}:{status_code}"
    return "error", type(exc).__name__


def _verify_hub_candidate(source: SeedableSource) -> Tuple[str, Optional[List[str]], str]:
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    try:
        dry_run = snapshot_download(
            repo_id=source.repo_id,
            revision=source.revision,
            repo_type=source.repo_type,
            cache_dir=source.cache_dir,
            dry_run=True,
            tqdm_class=_QuietTqdm,
        )
    except Exception as exc:
        status, reason = _classify_exception(exc)
        return status, None, reason

    manifest = sorted({item.filename for item in dry_run if getattr(item, "filename", None)})
    if not manifest:
        return "error", None, "empty_manifest"

    lookup_repo_type = source.repo_type if source.repo_type != "model" else None
    missing = []
    for filename in manifest:
        resolved = try_to_load_from_cache(
            repo_id=source.repo_id,
            filename=filename,
            revision=source.revision,
            repo_type=lookup_repo_type,
            cache_dir=source.cache_dir,
        )
        if not resolved or not os.path.exists(resolved):
            missing.append(filename)

    if missing:
        return "partial", manifest, f"missing:{len(missing)}"

    return "imported", manifest, "ok"


def _import_local_dir_candidate(source: SeedableSource) -> Tuple[str, Optional[List[str]], str]:
    if not source.local_dir or not _local_dir_matches_revision(source.local_dir, source.revision):
        return "partial", None, "metadata_mismatch"
    if register_completed_source(
        repo_id=source.repo_id,
        revision=source.revision,
        repo_type=source.repo_type,
        local_dir=source.local_dir,
    ):
        return "imported", None, "ok"
    return "error", None, "register_failed"


def import_verified_cache_sources() -> Dict[str, int]:
    """
    Scan existing cache entries and import only candidates verified as complete.

    Returns a summary with counts for imported / skipped / blocked / partial / error.
    """

    summary = {
        "imported": 0,
        "skipped_completed": 0,
        "skipped_backoff": 0,
        "blocked": 0,
        "partial": 0,
        "error": 0,
    }
    state = _load_import_state()
    live_keys = set()
    now = time.time()

    candidates = _collect_hub_candidates() + _collect_local_dir_candidates()
    for source in candidates:
        key = _candidate_key(source)
        live_keys.add(key)

        if has_completed_source(
            repo_id=source.repo_id,
            revision=source.revision,
            repo_type=source.repo_type,
            cache_dir=source.cache_dir,
            local_dir=source.local_dir,
        ):
            state.pop(key, None)
            summary["skipped_completed"] += 1
            continue

        pending = state.get(key)
        if pending and float(pending.get("next_retry_ts", 0.0)) > now:
            summary["skipped_backoff"] += 1
            continue

        if source.storage_kind == "local_dir":
            status, _manifest, reason = _import_local_dir_candidate(source)
        else:
            status, manifest, reason = _verify_hub_candidate(source)
            if status == "imported":
                imported = register_completed_source(
                    repo_id=source.repo_id,
                    revision=source.revision,
                    repo_type=source.repo_type,
                    cache_dir=source.cache_dir,
                    manifest=manifest,
                )
                if not imported:
                    status = "error"
                    reason = "register_failed"

        if status == "imported":
            state.pop(key, None)
            summary["imported"] += 1
            continue

        summary[status] += 1
        _record_import_state(state, source, status=status, reason=reason)

    stale_keys = [key for key in state if key not in live_keys]
    for key in stale_keys:
        state.pop(key, None)

    _save_import_state(state)
    return summary
