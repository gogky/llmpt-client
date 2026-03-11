"""
Persistent state for local .torrent cache presence and tracker registration.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger("llmpt.torrent_state")

TORRENT_STATE_FILE = os.path.expanduser("~/.cache/llmpt/torrent_state.json")


def _state_key(repo_id: str, revision: str, repo_type: str = "model") -> str:
    return f"{repo_type}|{repo_id}|{revision}"


def _load_state() -> dict[str, dict]:
    if not os.path.exists(TORRENT_STATE_FILE):
        return {}

    try:
        with open(TORRENT_STATE_FILE, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load torrent state registry: {exc}")
        return {}

    if not isinstance(payload, dict):
        return {}
    return {str(k): v for k, v in payload.items() if isinstance(v, dict)}


def _save_state(state: dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(TORRENT_STATE_FILE), exist_ok=True)
    tmp_path = TORRENT_STATE_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp_path, TORRENT_STATE_FILE)


def _upsert(
    repo_id: str,
    revision: str,
    repo_type: str,
    *,
    update: dict,
) -> None:
    state = _load_state()
    key = _state_key(repo_id, revision, repo_type)
    current = state.get(
        key,
        {
            "repo_type": repo_type,
            "repo_id": repo_id,
            "revision": revision,
            "local_torrent_present": False,
            "tracker_registered": False,
            "tracker_url": None,
            "info_hash": None,
            "last_registration_error": None,
            "updated_at": 0.0,
        },
    )
    current.update(update)
    current["repo_type"] = repo_type
    current["repo_id"] = repo_id
    current["revision"] = revision
    current["updated_at"] = time.time()
    state[key] = current
    _save_state(state)


def mark_local_torrent(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    present: bool = True,
    info_hash: Optional[str] = None,
) -> None:
    update = {"local_torrent_present": bool(present)}
    if info_hash is not None:
        update["info_hash"] = info_hash
    _upsert(repo_id, revision, repo_type, update=update)


def mark_tracker_registration(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    tracker_url: str,
    registered: bool,
    info_hash: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    update = {
        "tracker_registered": bool(registered),
        "tracker_url": tracker_url.rstrip("/"),
        "last_registration_error": None if registered else error,
        "last_registration_at": time.time(),
    }
    if info_hash is not None:
        update["info_hash"] = info_hash
    _upsert(repo_id, revision, repo_type, update=update)


def get_torrent_state(
    repo_id: str,
    revision: str,
    *,
    repo_type: str = "model",
    tracker_url: Optional[str] = None,
) -> dict:
    entry = _load_state().get(_state_key(repo_id, revision, repo_type), {}).copy()
    if not entry:
        entry = {
            "repo_type": repo_type,
            "repo_id": repo_id,
            "revision": revision,
            "local_torrent_present": False,
            "tracker_registered": False,
            "tracker_url": None,
            "info_hash": None,
            "last_registration_error": None,
        }
    if tracker_url:
        normalized = tracker_url.rstrip("/")
        entry["tracker_registered"] = bool(
            entry.get("tracker_registered") and entry.get("tracker_url") == normalized
        )
    return entry


def load_all_torrent_states() -> list[dict]:
    """Return all persisted torrent state entries."""
    return list(_load_state().values())
