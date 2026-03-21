"""Helpers for patch-layer download stats and daemon handoff."""

import threading
import time
from typing import Any, Callable, Optional

from .session_identity import normalize_storage_root
from .utils import get_hf_hub_cache


StatsKey = tuple[str, str, str, str, str]


def snapshot_stats_key(
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> StatsKey:
    """Build the canonical key for one snapshot-level download stats bucket."""
    if local_dir:
        storage_kind = "local_dir"
        storage_root = normalize_storage_root(local_dir)
    else:
        storage_kind = "hub_cache"
        storage_root = normalize_storage_root(cache_dir) or get_hf_hub_cache()
    return (repo_type or "model", repo_id, revision or "main", storage_kind, storage_root)


def empty_download_stats() -> dict[str, set[str]]:
    """Build an empty stats bucket."""
    return {
        "p2p": set(),
        "http": set(),
    }


def record_download_stat(
    *,
    stats_lock: threading.Lock,
    download_stats: dict[StatsKey, dict[str, set[str]]],
    stats_key: StatsKey,
    stat_kind: str,
    filename: str,
) -> None:
    """Record one file into the patch-layer stats bucket."""
    with stats_lock:
        bucket = download_stats.setdefault(stats_key, empty_download_stats())
        bucket[stat_kind].add(filename)


def get_download_stats(
    *,
    stats_lock: threading.Lock,
    download_stats: dict[StatsKey, dict[str, set[str]]],
    snapshot_key_builder: Callable[..., StatsKey],
    stats_key: Optional[StatsKey] = None,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> dict[str, set[str]]:
    """Return a snapshot of patch-layer transfer statistics."""
    if stats_key is None and repo_id is not None and revision is not None:
        stats_key = snapshot_key_builder(
            repo_id,
            revision,
            repo_type,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
    with stats_lock:
        if stats_key is not None:
            bucket = download_stats.get(stats_key, empty_download_stats())
            return {
                "p2p": set(bucket["p2p"]),
                "http": set(bucket["http"]),
            }

        merged = empty_download_stats()
        for bucket in download_stats.values():
            merged["p2p"].update(bucket["p2p"])
            merged["http"].update(bucket["http"])
        return merged


def reset_download_stats(
    *,
    stats_lock: threading.Lock,
    download_stats: dict[StatsKey, dict[str, set[str]]],
    snapshot_key_builder: Callable[..., StatsKey],
    stats_key: Optional[StatsKey] = None,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> None:
    """Clear patch-layer transfer stats."""
    if stats_key is None and repo_id is not None and revision is not None:
        stats_key = snapshot_key_builder(
            repo_id,
            revision,
            repo_type,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
    with stats_lock:
        if stats_key is None:
            download_stats.clear()
        else:
            download_stats.pop(stats_key, None)


def deferred_key(
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> StatsKey:
    """Build the dedupe key for deferred daemon notifications."""
    storage_kind = "local_dir" if local_dir else "hub_cache"
    storage_root = (
        normalize_storage_root(local_dir)
        if local_dir
        else (normalize_storage_root(cache_dir) or get_hf_hub_cache())
    )
    return (repo_type or "model", repo_id, revision or "main", storage_kind, storage_root)


def _build_notify_kwargs(
    *,
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    completed_snapshot: bool = False,
) -> dict[str, Any]:
    """Build the kwargs passed to daemon notification helpers."""
    notify_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "repo_type": repo_type,
    }
    if cache_dir is not None:
        notify_kwargs["cache_dir"] = cache_dir
    if local_dir is not None:
        notify_kwargs["local_dir"] = local_dir
    if completed_snapshot:
        notify_kwargs["completed_snapshot"] = True
    return notify_kwargs


def notify_seed_daemon(
    *,
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    completed_snapshot: bool = False,
) -> None:
    """Notify the daemon that a torrent should be seeded."""
    from .ipc import notify_daemon

    notify_daemon(
        "seed",
        **_build_notify_kwargs(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            cache_dir=cache_dir,
            local_dir=local_dir,
            completed_snapshot=completed_snapshot,
        ),
    )


def release_on_demand_session(
    *,
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    completed: bool = False,
    logger=None,
) -> None:
    """Release the temporary client-side on-demand session after handoff."""
    try:
        from .p2p_batch import P2PBatchManager

        manager = P2PBatchManager._instance
        if manager is None:
            return

        released = manager.release_on_demand_session(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            cache_dir=cache_dir,
            local_dir=local_dir,
            completed=completed,
        )
        if released and logger is not None:
            logger.debug(
                f"[P2P] Released on-demand session for "
                f"{repo_id}@{revision[:8]}..."
            )
    except Exception as exc:
        if logger is not None:
            logger.debug(
                f"[P2P] Failed to release on-demand session for "
                f"{repo_id}@{revision[:8]}...: {exc}"
            )


def fire_deferred_notification(
    key: StatsKey,
    *,
    deferred_lock: threading.Lock,
    deferred_contexts: dict[StatsKey, dict[str, Any]],
    deferred_timers: dict[StatsKey, threading.Timer],
    active_download_counts: dict[str, int],
    config: dict[str, Any],
    get_download_stats_fn: Callable[..., dict[str, set[str]]],
    reset_download_stats_fn: Callable[..., None],
    print_p2p_summary_fn: Callable[..., None],
    notify_seed_daemon_fn: Callable[..., None],
    release_on_demand_session_fn: Callable[..., None],
    logger,
    timer_factory: Callable[..., threading.Timer] = threading.Timer,
) -> None:
    """Debounce callback that notifies the daemon after downloads go quiet."""
    with deferred_lock:
        ctx = deferred_contexts.get(key)
        if ctx is None:
            deferred_timers.pop(key, None)
            return

        repo_id = ctx["repo_id"]
        if active_download_counts.get(repo_id, 0) > 0:
            old_timer = deferred_timers.pop(key, None)
            if old_timer is not None:
                old_timer.cancel()
            new_timer = timer_factory(2.0, fire_deferred_notification, args=[key], kwargs={
                "deferred_lock": deferred_lock,
                "deferred_contexts": deferred_contexts,
                "deferred_timers": deferred_timers,
                "active_download_counts": active_download_counts,
                "config": config,
                "get_download_stats_fn": get_download_stats_fn,
                "reset_download_stats_fn": reset_download_stats_fn,
                "print_p2p_summary_fn": print_p2p_summary_fn,
                "notify_seed_daemon_fn": notify_seed_daemon_fn,
                "release_on_demand_session_fn": release_on_demand_session_fn,
                "logger": logger,
                "timer_factory": timer_factory,
            })
            new_timer.daemon = True
            new_timer.start()
            deferred_timers[key] = new_timer
            return

        deferred_contexts.pop(key, None)
        deferred_timers.pop(key, None)

    if config.get("verbose"):
        elapsed = time.time() - ctx.get("start_time", time.time())
        print_p2p_summary_fn(
            stats=get_download_stats_fn(
                repo_id=ctx["repo_id"],
                revision=ctx["revision"],
                repo_type=ctx["repo_type"],
                cache_dir=ctx.get("cache_dir"),
                local_dir=ctx.get("local_dir"),
            ),
            elapsed=elapsed,
            repo_id=ctx["repo_id"],
            resolved_revision=ctx["revision"],
            repo_type=ctx["repo_type"],
        )

    try:
        notify_kwargs = {
            "repo_id": ctx["repo_id"],
            "revision": ctx["revision"],
            "repo_type": ctx["repo_type"],
            "cache_dir": ctx.get("cache_dir"),
            "local_dir": ctx.get("local_dir"),
        }
        if ctx.get("completed_snapshot"):
            notify_kwargs["completed_snapshot"] = True
        notify_seed_daemon_fn(**notify_kwargs)
        logger.debug(
            f"[P2P] Deferred daemon notification sent for "
            f"{ctx['repo_id']}@{ctx['revision'][:8]}..."
        )
    except Exception as exc:
        logger.debug(f"[P2P] Deferred daemon notification failed: {exc}")
    finally:
        release_on_demand_session_fn(
            repo_id=ctx["repo_id"],
            revision=ctx["revision"],
            repo_type=ctx["repo_type"],
            cache_dir=ctx.get("cache_dir"),
            local_dir=ctx.get("local_dir"),
            completed=True,
        )
        reset_download_stats_fn(
            repo_id=ctx["repo_id"],
            revision=ctx["revision"],
            repo_type=ctx["repo_type"],
            cache_dir=ctx.get("cache_dir"),
            local_dir=ctx.get("local_dir"),
        )


def schedule_deferred_notification(
    repo_id: str,
    revision: str,
    repo_type: str,
    *,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    completed_snapshot: bool = False,
    deferred_lock: threading.Lock,
    deferred_contexts: dict[StatsKey, dict[str, Any]],
    deferred_timers: dict[StatsKey, threading.Timer],
    deferred_key_fn: Callable[..., StatsKey],
    fire_deferred_notification_fn: Callable[[StatsKey], None],
    timer_factory: Callable[..., threading.Timer] = threading.Timer,
) -> None:
    """Schedule or reschedule the deferred daemon handoff timer."""
    key = deferred_key_fn(repo_id, revision, repo_type, cache_dir, local_dir)
    with deferred_lock:
        timer = deferred_timers.get(key)
        if timer is not None:
            timer.cancel()

        existing = deferred_contexts.get(key)
        deferred_contexts[key] = {
            "repo_id": repo_id,
            "revision": revision,
            "repo_type": repo_type,
            "cache_dir": cache_dir,
            "local_dir": local_dir,
            "completed_snapshot": bool(
                completed_snapshot or (existing or {}).get("completed_snapshot")
            ),
            "start_time": existing["start_time"] if existing else time.time(),
        }

        new_timer = timer_factory(2.0, fire_deferred_notification_fn, args=[key])
        new_timer.daemon = True
        new_timer.start()
        deferred_timers[key] = new_timer


def flush_deferred_notifications(
    *,
    deferred_lock: threading.Lock,
    deferred_contexts: dict[StatsKey, dict[str, Any]],
    deferred_timers: dict[StatsKey, threading.Timer],
    config: dict[str, Any],
    get_download_stats_fn: Callable[..., dict[str, set[str]]],
    reset_download_stats_fn: Callable[..., None],
    print_p2p_summary_fn: Callable[..., None],
    notify_seed_daemon_fn: Callable[..., None],
) -> None:
    """Flush pending deferred notifications during process exit."""
    with deferred_lock:
        contexts = list(deferred_contexts.values())
        for timer in deferred_timers.values():
            timer.cancel()
        deferred_timers.clear()
        deferred_contexts.clear()

    for ctx in contexts:
        if config.get("verbose"):
            elapsed = time.time() - ctx.get("start_time", time.time())
            print_p2p_summary_fn(
                stats=get_download_stats_fn(
                    repo_id=ctx["repo_id"],
                    revision=ctx["revision"],
                    repo_type=ctx["repo_type"],
                    cache_dir=ctx.get("cache_dir"),
                    local_dir=ctx.get("local_dir"),
                ),
                elapsed=elapsed,
                repo_id=ctx["repo_id"],
                resolved_revision=ctx["revision"],
                repo_type=ctx["repo_type"],
            )

        try:
            notify_seed_daemon_fn(
                repo_id=ctx["repo_id"],
                revision=ctx["revision"],
                repo_type=ctx["repo_type"],
                cache_dir=ctx.get("cache_dir"),
                local_dir=ctx.get("local_dir"),
            )
        except Exception:
            pass
        finally:
            reset_download_stats_fn(
                repo_id=ctx["repo_id"],
                revision=ctx["revision"],
                repo_type=ctx["repo_type"],
                cache_dir=ctx.get("cache_dir"),
                local_dir=ctx.get("local_dir"),
            )
