"""Helpers for patch-layer download context management."""

import sys
from typing import Any, Optional

from .session_identity import normalize_storage_root


_CONTEXT_FIELDS = (
    "repo_id",
    "repo_type",
    "filename",
    "revision",
    "tracker",
    "config",
    "cache_dir",
    "local_dir",
)


def capture_thread_local_context(context_local) -> dict[str, Any]:
    """Snapshot the patch thread-local state for later restoration."""
    return {
        field: getattr(context_local, field, None)
        for field in _CONTEXT_FIELDS
    }


def apply_thread_local_context(
    context_local,
    *,
    repo_id: Optional[str] = None,
    repo_type: Optional[str] = None,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    tracker: Any = None,
    config: Optional[dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> None:
    """Install a new patch thread-local context."""
    context_local.repo_id = repo_id
    context_local.repo_type = repo_type
    context_local.filename = filename
    context_local.revision = revision
    context_local.tracker = tracker
    context_local.config = config
    context_local.cache_dir = cache_dir
    context_local.local_dir = local_dir


def restore_thread_local_context(context_local, snapshot: dict[str, Any]) -> None:
    """Restore a previously captured patch thread-local context."""
    apply_thread_local_context(
        context_local,
        repo_id=snapshot.get("repo_id"),
        repo_type=snapshot.get("repo_type"),
        filename=snapshot.get("filename"),
        revision=snapshot.get("revision"),
        tracker=snapshot.get("tracker"),
        config=snapshot.get("config"),
        cache_dir=snapshot.get("cache_dir"),
        local_dir=snapshot.get("local_dir"),
    )


def read_thread_local_context(context_local) -> Optional[dict[str, Any]]:
    """Read the current patch thread-local context if it is complete enough."""
    context = capture_thread_local_context(context_local)
    if not (context.get("repo_id") and context.get("filename") and context.get("revision")):
        return None
    context["repo_type"] = context.get("repo_type") or "model"
    context["config"] = context.get("config") or {}
    return context


def _extract_storage_context(result: dict[str, Any], locals_dict: dict[str, Any]) -> None:
    """Populate cache/local-dir context from a Hugging Face stack frame."""
    for dir_key in ("cache_dir", "local_dir"):
        if dir_key in locals_dict and dir_key not in result and locals_dict[dir_key]:
            result[dir_key] = locals_dict[dir_key]
        elif (
            locals_dict.get("kwargs")
            and dir_key in locals_dict["kwargs"]
            and dir_key not in result
            and locals_dict["kwargs"][dir_key]
        ):
            result[dir_key] = locals_dict["kwargs"][dir_key]


def extract_context_from_stack() -> Optional[dict[str, Any]]:
    """Walk the call stack to recover hf_hub_download request context."""
    try:
        frame = sys._getframe(1)
        result = {"from_snapshot_download": False}
        for _ in range(60):
            frame = frame.f_back
            if frame is None:
                break
            name = frame.f_code.co_name
            if name in ("snapshot_download", "_inner_hf_hub_download"):
                result["from_snapshot_download"] = True
            if name not in (
                "hf_hub_download",
                "snapshot_download",
                "_inner_hf_hub_download",
                "_hf_hub_download_to_cache_dir",
                "_hf_hub_download_to_local_dir",
            ):
                continue

            locals_dict = frame.f_locals
            if "repo_id" in locals_dict and "repo_id" not in result:
                result["repo_id"] = locals_dict.get("repo_id")
            if "filename" in locals_dict and "filename" not in result:
                result["filename"] = locals_dict.get("filename")
            if "repo_type" in locals_dict and "repo_type" not in result:
                result["repo_type"] = locals_dict.get("repo_type") or "model"

            revision = locals_dict.get("commit_hash") or locals_dict.get("revision")
            if revision and "revision" not in result:
                result["revision"] = revision

            if "subfolder" in locals_dict and "subfolder" not in result:
                subfolder = locals_dict.get("subfolder")
                if subfolder and subfolder != "":
                    result["subfolder"] = subfolder

            _extract_storage_context(result, locals_dict)

        if result.get("repo_id") and result.get("filename"):
            result.setdefault("revision", "main")
            result.setdefault("repo_type", "model")
            if "subfolder" in result:
                result["filename"] = f"{result['subfolder']}/{result['filename']}"
            return result
    except (AttributeError, ValueError):
        pass
    return None


def extract_snapshot_context_from_stack() -> Optional[dict[str, Any]]:
    """Walk the call stack to recover snapshot_download context."""
    try:
        frame = sys._getframe(1)
        result: dict[str, Any] = {}
        for _ in range(60):
            frame = frame.f_back
            if frame is None:
                break
            if frame.f_code.co_name not in ("snapshot_download", "_inner_hf_hub_download"):
                continue

            locals_dict = frame.f_locals
            if "repo_id" in locals_dict and "repo_id" not in result:
                result["repo_id"] = locals_dict.get("repo_id")

            revision = locals_dict.get("commit_hash") or locals_dict.get("revision")
            if revision and "revision" not in result:
                result["revision"] = revision

            if "repo_type" not in result:
                result["repo_type"] = locals_dict.get("repo_type") or "model"

            _extract_storage_context(result, locals_dict)

            if result.get("repo_id") and result.get("revision"):
                return result
    except (AttributeError, ValueError):
        pass
    return None


def matches_snapshot_download_context(
    *,
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> bool:
    """Return True when the current stack still belongs to this snapshot request."""
    snapshot_ctx = extract_snapshot_context_from_stack()
    if snapshot_ctx is None:
        return False

    if snapshot_ctx.get("repo_id") != repo_id:
        return False
    if (snapshot_ctx.get("repo_type") or "model") != (repo_type or "model"):
        return False
    if (snapshot_ctx.get("revision") or "main") != (revision or "main"):
        return False

    expected_local_dir = normalize_storage_root(local_dir)
    actual_local_dir = normalize_storage_root(snapshot_ctx.get("local_dir"))
    if expected_local_dir or actual_local_dir:
        return expected_local_dir == actual_local_dir

    expected_cache_dir = normalize_storage_root(cache_dir)
    actual_cache_dir = normalize_storage_root(snapshot_ctx.get("cache_dir"))
    if expected_cache_dir or actual_cache_dir:
        return expected_cache_dir == actual_cache_dir

    return True
