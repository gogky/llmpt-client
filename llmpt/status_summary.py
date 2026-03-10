"""
Lightweight unified status summary for source/torrent/session state.
"""

from __future__ import annotations

from typing import Optional

from .cache_importer import load_import_state
from .completed_registry import load_completed_sources
from .torrent_state import get_torrent_state


def _source_entries(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
) -> list[dict]:
    return [
        item
        for item in load_completed_sources()
        if item.get("repo_type") == repo_type
        and item.get("repo_id") == repo_id
        and item.get("revision") == revision
    ]


def _import_entries(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
) -> list[dict]:
    return [
        item
        for item in load_import_state().values()
        if item.get("repo_type") == repo_type
        and item.get("repo_id") == repo_id
        and item.get("revision") == revision
    ]


def get_source_status(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
) -> str:
    if _source_entries(repo_id, revision, repo_type=repo_type):
        return "verified"

    statuses = {item.get("status") for item in _import_entries(repo_id, revision, repo_type=repo_type)}
    if "blocked" in statuses:
        return "blocked"
    if "partial" in statuses:
        return "partial"
    if "error" in statuses:
        return "error"
    return "unknown"


def get_torrent_status(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    tracker_url: Optional[str] = None,
) -> str:
    state = get_torrent_state(
        repo_id,
        revision,
        repo_type=repo_type,
        tracker_url=tracker_url,
    )
    if state.get("tracker_registered"):
        return "registered"
    if state.get("local_torrent_present"):
        return "local_only"
    return "absent"


def get_session_status(
    *,
    active: bool,
    full_mapping: bool,
    tracker_registered: bool,
) -> str:
    if not active:
        return "inactive"
    if not full_mapping:
        return "degraded"
    if not tracker_registered:
        return "degraded"
    return "active"


def summarize_status(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    tracker_url: Optional[str] = None,
    active: bool = False,
    full_mapping: bool = False,
) -> dict:
    torrent_status = get_torrent_status(
        repo_id,
        revision,
        repo_type=repo_type,
        tracker_url=tracker_url,
    )
    tracker_registered = torrent_status == "registered"
    return {
        "source_status": get_source_status(
            repo_id,
            revision,
            repo_type=repo_type,
        ),
        "torrent_status": torrent_status,
        "session_status": get_session_status(
            active=active,
            full_mapping=full_mapping,
            tracker_registered=tracker_registered,
        ),
    }
