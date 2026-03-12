"""
llmptd — Background seeding daemon for llmpt.

This module implements a standalone daemon process that:
1. Scans the local HuggingFace cache for seedable models
2. Creates .torrent files and registers them with the tracker
3. Seeds continuously in the background via libtorrent
4. Accepts IPC commands from download clients

Lifecycle:
    llmpt-cli start   →  fork to background, return immediately
    llmpt-cli stop    →  send SIGTERM, daemon exits gracefully
    llmpt-cli status  →  query daemon via IPC
"""

import atexit
import json
import logging
import os
import random
import signal
import sys
import time
from typing import Dict, Optional, Set, Tuple

from .utils import LIBTORRENT_AVAILABLE, lt, get_hf_hub_cache

logger = logging.getLogger("llmpt.daemon")

# Paths
LLMPT_DIR = os.path.expanduser("~/.cache/llmpt")
PID_FILE = os.path.join(LLMPT_DIR, "daemon.pid")
LOG_FILE = os.path.join(LLMPT_DIR, "daemon.log")

# Daemon configuration
SCAN_INTERVAL = 300  # seconds between full HF cache scans
RETRY_BASE_DELAY = 60   # first retry after 60s
RETRY_MAX_DELAY = 3600  # cap backoff at 1h
RETRY_JITTER_RATIO = 0.10

SeedSessionKey = Tuple[str, str, str, str, str]


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------


def _read_pid() -> Optional[int]:
    """Read PID from the PID file. Returns None if not found or invalid."""
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _write_pid(pid: int) -> None:
    """Write PID to the PID file."""
    os.makedirs(LLMPT_DIR, exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(pid))


def _remove_pid() -> None:
    """Remove the PID file."""
    try:
        os.unlink(PID_FILE)
    except OSError:
        pass


def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _normalize_storage_path(path: str) -> str:
    """Normalize a storage path for stable in-memory session keys."""
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _seeding_key(
    repo_type: str,
    repo_id: str,
    revision: str,
    *,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> SeedSessionKey:
    """Return the daemon's canonical key for one seeding session."""
    if local_dir:
        storage_kind = "local_dir"
        storage_root = _normalize_storage_path(local_dir)
    elif cache_dir:
        storage_kind = "hub_cache"
        storage_root = _normalize_storage_path(cache_dir)
    else:
        storage_kind = "hub_cache"
        storage_root = get_hf_hub_cache()
    return (repo_type, repo_id, revision, storage_kind, storage_root)


def _discovered_seeding_keys(seedable) -> Set[SeedSessionKey]:
    """Build daemon session keys for the current scan result."""
    discovered_keys: Set[SeedSessionKey] = set()
    for item in seedable:
        discovered_keys.add(
            _seeding_key(
                item.repo_type,
                item.repo_id,
                item.revision,
                cache_dir=item.cache_dir,
                local_dir=item.local_dir,
            )
        )
    return discovered_keys


def _reconcile_seeding_sessions(
    manager,
    seeding_set: Set[SeedSessionKey],
    failed_attempts: Dict[SeedSessionKey, Dict[str, object]],
    suppressed_set: Set[SeedSessionKey],
    seedable,
) -> Set[SeedSessionKey]:
    """Remove sessions that no longer correspond to a live seedable source."""
    discovered_keys = _discovered_seeding_keys(seedable)
    stale_keys = [key for key in seeding_set if key not in discovered_keys]
    for stale_key in stale_keys:
        repo_type, repo_id, revision, storage_kind, storage_root = stale_key
        cache_dir = storage_root if storage_kind == "hub_cache" and storage_root else None
        local_dir = storage_root if storage_kind == "local_dir" else None
        logger.info(
            f"[{repo_id}@{revision[:8]}] Removing stale seeding session "
            f"({storage_kind}={storage_root or '<default>'})"
        )
        try:
            manager.remove_session(
                repo_id,
                revision,
                repo_type=repo_type,
                cache_dir=cache_dir,
                local_dir=local_dir,
            )
        finally:
            seeding_set.discard(stale_key)
            failed_attempts.pop(stale_key, None)
    suppressed_set.intersection_update(discovered_keys)
    return discovered_keys


def _matching_seeding_keys(
    seeding_set: Set[SeedSessionKey],
    *,
    repo_type: Optional[str],
    repo_id: str,
    revision: Optional[str] = None,
) -> list[SeedSessionKey]:
    """Find active seeding sessions matching the given repo identity."""
    matches = []
    for key in seeding_set:
        key_repo_type, key_repo_id, key_revision, _storage_kind, _storage_root = key
        if key_repo_id != repo_id:
            continue
        if repo_type is not None and key_repo_type != repo_type:
            continue
        if revision is not None and key_revision != revision:
            continue
        matches.append(key)
    return matches


def _unseed_matching_sessions(
    manager,
    seeding_set: Set[SeedSessionKey],
    failed_attempts: Dict[SeedSessionKey, Dict[str, object]],
    suppressed_set: Set[SeedSessionKey],
    *,
    repo_type: Optional[str],
    repo_id: str,
    revision: Optional[str] = None,
    forget: bool = False,
) -> dict:
    """Stop active seeding sessions and optionally forget custom storage."""
    matches = _matching_seeding_keys(
        seeding_set,
        repo_type=repo_type,
        repo_id=repo_id,
        revision=revision,
    )
    if not matches:
        return {
            "status": "error",
            "message": "no matching active seeding session",
            "removed_count": 0,
            "forgotten": {
                "hub_cache_roots_removed": 0,
                "local_dir_sources_removed": 0,
                "completed_sources_removed": 0,
            },
        }

    matched_repo_types = sorted({key[0] for key in matches})
    if repo_type is None and len(matched_repo_types) > 1:
        return {
            "status": "error",
            "message": (
                "multiple repo types match this repo_id; "
                f"please specify --repo-type ({', '.join(matched_repo_types)})"
            ),
            "removed_count": 0,
            "forgotten": {
                "hub_cache_roots_removed": 0,
                "local_dir_sources_removed": 0,
                "completed_sources_removed": 0,
            },
        }

    forgotten = {
        "hub_cache_roots_removed": 0,
        "local_dir_sources_removed": 0,
        "completed_sources_removed": 0,
    }
    removed_sessions = []

    for key in matches:
        match_repo_type, match_repo_id, match_revision, storage_kind, storage_root = key
        cache_dir = storage_root if storage_kind == "hub_cache" and storage_root else None
        local_dir = storage_root if storage_kind == "local_dir" else None

        manager.remove_session(
            match_repo_id,
            match_revision,
            repo_type=match_repo_type,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
        seeding_set.discard(key)
        failed_attempts.pop(key, None)
        suppressed_set.add(key)

        if forget and (cache_dir or local_dir):
            from llmpt.cache_scanner import forget_seedable_storage
            from llmpt.completed_registry import forget_completed_source

            counts = forget_seedable_storage(
                repo_id=match_repo_id,
                revision=match_revision,
                repo_type=match_repo_type,
                cache_dir=cache_dir,
                local_dir=local_dir,
            )
            forgotten["hub_cache_roots_removed"] += counts["hub_cache_roots_removed"]
            forgotten["local_dir_sources_removed"] += counts["local_dir_sources_removed"]
            forgotten["completed_sources_removed"] += forget_completed_source(
                repo_id=match_repo_id,
                revision=match_revision,
                repo_type=match_repo_type,
                cache_dir=cache_dir,
                local_dir=local_dir,
            )

        removed_sessions.append(
            {
                "repo_type": match_repo_type,
                "repo_id": match_repo_id,
                "revision": match_revision,
                "storage_kind": storage_kind,
                "storage_root": storage_root,
            }
        )

    return {
        "status": "ok",
        "removed_count": len(removed_sessions),
        "removed_sessions": removed_sessions,
        "forgotten": forgotten,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_daemon_running() -> Optional[int]:
    """Check if the daemon is running.

    Returns:
        PID of the running daemon, or None if not running.
    """
    pid = _read_pid()
    if pid and _is_process_running(pid):
        return pid
    # Stale PID file
    if pid:
        _remove_pid()
    return None


def start_daemon(
    tracker_url: str = "http://localhost:8080",
    port: Optional[int] = None,
    foreground: bool = False,
) -> Optional[int]:
    """Start the seeding daemon.

    Args:
        tracker_url: URL of the tracker server.
        port: libtorrent listen port (None = auto-select).
        foreground: If True, run in the foreground (for debugging).

    Returns:
        PID of the started daemon, or None if it was already running.
    """
    existing_pid = is_daemon_running()
    if existing_pid:
        logger.info(f"Daemon already running (PID: {existing_pid})")
        return existing_pid

    if foreground:
        # Run directly in the current process (useful for debugging)
        _write_pid(os.getpid())
        _daemon_main(tracker_url, port=port)
        return os.getpid()

    # Fork to background
    pid = _daemonize(tracker_url, port=port)
    return pid


def stop_daemon() -> bool:
    """Stop the running daemon.

    Returns:
        True if the daemon was stopped, False if it wasn't running.
    """
    pid = is_daemon_running()
    if not pid:
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for the process to exit
        for _ in range(30):  # Max 3 seconds
            if not _is_process_running(pid):
                _remove_pid()
                return True
            time.sleep(0.1)
        # Force kill if still running
        os.kill(pid, signal.SIGKILL)
        _remove_pid()
        return True
    except (OSError, ProcessLookupError):
        _remove_pid()
        return True


# ---------------------------------------------------------------------------
# Daemonize (double-fork)
# ---------------------------------------------------------------------------


def _daemonize(tracker_url: str, port: Optional[int] = None) -> int:
    """Start the daemon as a subprocess.

    Uses subprocess.Popen to launch a clean, detached Python process via
    the CLI's hidden `_internal_daemon_start` command to avoid deadlocks.

    Returns the daemon PID.
    """
    import subprocess

    os.makedirs(LLMPT_DIR, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "llmpt.cli", 
        "_internal_daemon_start", 
        "--tracker", tracker_url
    ]
    if port is not None:
        cmd.extend(["--port", str(port)])

    # Start a detached subprocess
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=open(os.path.join(LLMPT_DIR, "daemon_stderr.log"), "w"),
        start_new_session=True,  # Detach from parent's session
    )

    # Wait for PID file to be written
    for _ in range(50):
        time.sleep(0.1)
        daemon_pid = _read_pid()
        if daemon_pid and _is_process_running(daemon_pid):
            return daemon_pid

    # Fallback: use subprocess PID
    logger.warning(f"Daemon PID file not written in time, using subprocess PID {proc.pid}")
    return proc.pid



# ---------------------------------------------------------------------------
# Daemon main loop
# ---------------------------------------------------------------------------


def _daemon_main(tracker_url: str, port: Optional[int] = None) -> None:
    """Main loop of the seeding daemon."""
    if not LIBTORRENT_AVAILABLE:
        logger.error("libtorrent not available, daemon cannot start")
        return

    from llmpt.cache_scanner import scan_hf_cache
    from .ipc import IPCServer
    from .p2p_batch import P2PBatchManager
    from .tracker_client import TrackerClient

    # Set the global config so P2PBatchManager can read port and other settings.
    # In a forked daemon process, the parent's _config is not inherited.
    from . import _config as parent_config
    import llmpt
    llmpt._config['tracker_url'] = tracker_url
    if port is not None:
        llmpt._config['port'] = port
    elif os.getenv('HF_P2P_PORT'):
        llmpt._config['port'] = int(os.getenv('HF_P2P_PORT'))

    logger.info(f"Daemon starting (PID: {os.getpid()}, tracker: {tracker_url}, port: {llmpt._config.get('port', 'auto')})")

    # Mark this process as the seeding daemon so P2PBatchManager binds
    # to the daemon port (N) rather than the client port (N+1).
    llmpt._config['_role'] = 'daemon'

    tracker_client = TrackerClient(tracker_url)
    manager = P2PBatchManager()

    # Track what we're already seeding to avoid duplicate work
    seeding_set: Set[SeedSessionKey] = set()
    suppressed_set: Set[SeedSessionKey] = set()
    # Failed seed attempts with retry/backoff metadata.
    # key -> {"attempts": int, "next_retry_ts": float, "last_error": str}
    failed_attempts: Dict[SeedSessionKey, Dict[str, object]] = {}

    running = True

    def _signal_handler(signum, frame):
        nonlocal running
        logger.info(f"Received signal {signum}, shutting down...")
        running = False

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # IPC message handler
    def _handle_ipc(msg: dict) -> Optional[dict]:
        nonlocal tracker_url, tracker_client, last_scan_time
        action = msg.get("action")

        if action == "seed":
            repo_id = msg.get("repo_id")
            revision = msg.get("revision")
            repo_type = msg.get("repo_type", "model")
            cache_dir = msg.get("cache_dir")
            local_dir = msg.get("local_dir")
            completed_snapshot = bool(msg.get("completed_snapshot"))
            if repo_id and revision:
                key = _seeding_key(
                    repo_type,
                    repo_id,
                    revision,
                    cache_dir=cache_dir,
                    local_dir=local_dir,
                )
                if key not in seeding_set:
                    logger.info(f"IPC: seed request for {repo_id}@{revision[:8]}...")
                    # Explicit seed requests should not wait for backoff.
                    failed_attempts.pop(key, None)
                    suppressed_set.discard(key)

                    from llmpt.cache_scanner import register_seedable_storage
                    from llmpt.completed_registry import (
                        has_completed_source,
                        register_completed_source,
                    )

                    completed_registered = has_completed_source(
                        repo_id=repo_id,
                        revision=revision,
                        repo_type=repo_type,
                        cache_dir=cache_dir,
                        local_dir=local_dir,
                    )
                    if completed_snapshot and not completed_registered:
                        completed_registered = register_completed_source(
                            repo_id=repo_id,
                            revision=revision,
                            repo_type=repo_type,
                            cache_dir=cache_dir,
                            local_dir=local_dir,
                        )

                    if not completed_registered:
                        logger.info(
                            f"IPC seed skipped for unverified source: "
                            f"{repo_id}@{revision[:8]}..."
                        )
                        return {
                            "status": "skipped",
                            "message": "source is not verified complete",
                        }

                    register_seedable_storage(
                        repo_id=repo_id,
                        revision=revision,
                        repo_type=repo_type,
                        cache_dir=cache_dir,
                        local_dir=local_dir,
                    )
                        
                    _process_seedable(
                        repo_id, revision, tracker_client, manager,
                        seeding_set, failed_attempts, repo_type=repo_type,
                        cache_dir=cache_dir, local_dir=local_dir,
                    )
                return {"status": "ok"}

        elif action == "status":
            try:
                from llmpt.cache_scanner import scan_seedable_sources
                seedable = scan_seedable_sources()
                _reconcile_seeding_sessions(
                    manager, seeding_set, failed_attempts, suppressed_set, seedable
                )
            except Exception as e:
                logger.warning(f"Failed to reconcile seeding sessions before status: {e}")
            status = manager.get_all_session_status()
            from llmpt.torrent_state import get_torrent_state
            from llmpt.status_summary import summarize_status

            session_states = {}
            unified_states = {}
            for key, value in status.items():
                session_states[key] = get_torrent_state(
                    value.get("repo_id"),
                    value.get("revision"),
                    repo_type=value.get("repo_type", "model"),
                    tracker_url=tracker_url,
                )
                unified_states[key] = summarize_status(
                    value.get("repo_id"),
                    value.get("revision"),
                    repo_type=value.get("repo_type", "model"),
                    tracker_url=tracker_url,
                    active=True,
                    full_mapping=bool(value.get("full_mapping", False)),
                )
            return {
                "status": "ok",
                "pid": os.getpid(),
                "port": getattr(manager, 'listen_port', None),
                "tracker_url": tracker_url,
                "seeding_count": len(status),
                "sessions": {
                    k: {
                        "progress": v.get("progress", 0),
                        "state": v.get("state", "unknown"),
                        "uploaded": v.get("uploaded", 0),
                        "peers": v.get("peers", 0),
                        "upload_rate": v.get("upload_rate", 0),
                        "mapped_files": v.get("mapped_files", 0),
                        "total_files": v.get("total_files", 0),
                        "full_mapping": v.get("full_mapping", False),
                        "tracker_registered": session_states[k].get("tracker_registered", False),
                        "last_registration_error": session_states[k].get("last_registration_error"),
                        "source_status": unified_states[k]["source_status"],
                        "torrent_status": unified_states[k]["torrent_status"],
                        "session_status": unified_states[k]["session_status"],
                    }
                    for k, v in status.items()
                },
            }

        elif action == "update_tracker":
            new_url = msg.get("tracker_url")
            if not new_url:
                return {"status": "error", "message": "tracker_url required"}
            if new_url == tracker_url:
                return {"status": "ok", "message": "no change needed"}

            old_url = tracker_url
            tracker_url = new_url
            tracker_client = TrackerClient(new_url)
            llmpt._config['tracker_url'] = new_url

            # Update announce URL on all active torrent handles
            announce_url = f"{new_url.rstrip('/')}/announce"
            from llmpt.torrent_cache import resolve_torrent_data
            from llmpt.torrent_creator import ensure_registered
            with manager._lock:
                for ctx in manager.sessions.values():
                    with ctx.lock:
                        handle = ctx.handle
                    if handle:
                        try:
                            if handle.is_valid():
                                handle.replace_trackers([lt.announce_entry(announce_url)])
                                handle.force_reannounce()
                        except Exception as e:
                            logger.warning(f"Failed to update tracker for {ctx.repo_id}: {e}")
                    try:
                        torrent_data = resolve_torrent_data(
                            ctx.repo_id,
                            ctx.revision,
                            tracker_client,
                            repo_type=ctx.repo_type,
                        )
                        if torrent_data:
                            ensure_registered(
                                ctx.repo_id,
                                ctx.revision,
                                ctx.repo_type,
                                torrent_data,
                                tracker_client,
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to re-register torrent after tracker update for {ctx.repo_id}: {e}"
                        )

            logger.info(f"Tracker URL updated: {old_url} -> {new_url}")
            return {"status": "ok", "old_tracker": old_url, "new_tracker": new_url}

        elif action == "scan":
            # Clear failure backoff and reset scan timer to trigger immediate rescan
            failed_attempts.clear()
            try:
                from llmpt.cache_importer import clear_import_state

                clear_import_state()
            except Exception as e:
                logger.warning(f"Failed to clear cache import state before scan: {e}")
            last_scan_time = 0.0
            return {"status": "ok", "message": "scan triggered"}

        elif action == "unseed":
            repo_id = msg.get("repo_id")
            revision = msg.get("revision")
            repo_type = msg.get("repo_type")
            forget = bool(msg.get("forget"))
            if not repo_id:
                return {"status": "error", "message": "repo_id required"}
            return _unseed_matching_sessions(
                manager,
                seeding_set,
                failed_attempts,
                suppressed_set,
                repo_type=repo_type,
                repo_id=repo_id,
                revision=revision,
                forget=forget,
            )

        elif action == "ping":
            return {"status": "ok", "pid": os.getpid()}

        return {"status": "error", "message": f"unknown action: {action}"}

    # Start IPC server
    ipc_server = IPCServer(handler=_handle_ipc)
    ipc_server.start()

    # Write PID file AFTER IPC is listening.  This guarantees that when
    # is_daemon_running() finds a valid PID, the IPC socket is already
    # accepting connections — eliminating the "running but not responding
    # to IPC" race window.
    _write_pid(os.getpid())

    # Clean up on exit
    def _cleanup():
        ipc_server.stop()
        manager.shutdown()
        _remove_pid()

    atexit.register(_cleanup)

    # Initial scan
    last_scan_time = 0.0

    logger.info("Daemon started, entering main loop")

    while running:
        try:
            # Periodic HF cache scan
            now = time.time()
            if now - last_scan_time >= SCAN_INTERVAL:
                last_scan_time = now
                _scan_and_seed(
                    tracker_client, manager, seeding_set, failed_attempts,
                    suppressed_set,
                )

            time.sleep(1)

        except Exception as e:
            logger.error(f"Error in daemon main loop: {e}", exc_info=True)
            time.sleep(5)

    # Graceful shutdown
    logger.info("Daemon shutting down...")
    ipc_server.stop()
    manager.shutdown()
    _remove_pid()
    logger.info("Daemon stopped")


def _rewrite_announce_url(torrent_data: bytes, tracker_url: str, repo_id: str) -> bytes:
    """Rewrite the announce URL in a .torrent file to match the current tracker.

    The announce field is outside the ``info`` dict and therefore does NOT
    affect the info_hash (swarm identity).  This lets us correct stale
    announce URLs (e.g. ``localhost:8080/announce`` baked in during an earlier
    run) without splitting the swarm.

    Returns the original bytes unchanged if the URL is already correct or if
    parsing fails.
    """
    from .utils import lt

    correct_announce = f"{tracker_url.rstrip('/')}/announce"

    try:
        decoded = lt.bdecode(torrent_data)
        if not isinstance(decoded, dict):
            return torrent_data

        current = decoded.get(b'announce', b'').decode('utf-8', errors='replace')
        if current == correct_announce:
            return torrent_data

        logger.info(
            f"[{repo_id}] Rewriting announce URL: {current!r} → {correct_announce!r}"
        )
        decoded[b'announce'] = correct_announce.encode('utf-8')

        # Also fix announce-list if present
        announce_list = decoded.get(b'announce-list')
        if announce_list:
            decoded[b'announce-list'] = [[correct_announce.encode('utf-8')]]

        return lt.bencode(decoded)

    except Exception as exc:
        logger.warning(f"[{repo_id}] Failed to rewrite announce URL: {exc}")
        return torrent_data


def _scan_and_seed(
    tracker_client,
    manager,
    seeding_set: Set[SeedSessionKey],
    failed_attempts: Dict[SeedSessionKey, Dict[str, object]],
    suppressed_set: Set[SeedSessionKey],
) -> None:
    """Scan HF cache and start seeding any new models found."""
    try:
        from llmpt.cache_importer import import_verified_cache_sources

        import_summary = import_verified_cache_sources()
        if any(import_summary.values()):
            logger.info(f"Cache import summary: {import_summary}")
    except Exception as e:
        logger.warning(f"Cache import verification failed: {e}")

    from llmpt.cache_scanner import scan_seedable_sources

    seedable = scan_seedable_sources()
    new_count = 0
    now = time.time()
    discovered_keys = _discovered_seeding_keys(seedable)

    for item in seedable:
        repo_type = item.repo_type
        repo_id = item.repo_id
        revision = item.revision
        cache_dir = item.cache_dir
        local_dir = item.local_dir

        key = _seeding_key(
            repo_type,
            repo_id,
            revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
        if key in seeding_set:
            continue
        if key in suppressed_set:
            continue

        failure = failed_attempts.get(key)
        if failure is not None:
            next_retry_ts = float(failure.get("next_retry_ts", 0.0))
            if now < next_retry_ts:
                continue

        _process_seedable(
            repo_id, revision, tracker_client, manager,
            seeding_set, failed_attempts, repo_type=repo_type,
            cache_dir=cache_dir, local_dir=local_dir
        )
        new_count += 1

    if new_count > 0:
        logger.info(f"Scan processed {new_count} new models")
    _reconcile_seeding_sessions(
        manager, seeding_set, failed_attempts, suppressed_set, seedable
    )

    try:
        from .torrent_cache import cleanup_torrent_cache

        protected = {
            (item.repo_type, item.repo_id, item.revision)
            for item in seedable
        }
        protected.update(
            (repo_type, repo_id, revision)
            for repo_type, repo_id, revision, _storage_kind, _storage_root in seeding_set
        )
        cleanup_summary = cleanup_torrent_cache(protected)
        if any(cleanup_summary.values()):
            logger.info(f"Torrent cache cleanup summary: {cleanup_summary}")
    except Exception as e:
        logger.warning(f"Torrent cache cleanup failed: {e}")


def _record_seed_failure(
    key: SeedSessionKey,
    failed_attempts: Dict[SeedSessionKey, Dict[str, object]],
    reason: str,
) -> None:
    """Record a failed seed attempt and compute its next retry deadline."""
    state = failed_attempts.get(key, {"attempts": 0})
    attempts = int(state.get("attempts", 0)) + 1
    backoff = min(RETRY_BASE_DELAY * (2 ** (attempts - 1)), RETRY_MAX_DELAY)
    jitter = random.uniform(0, backoff * RETRY_JITTER_RATIO)
    next_retry = time.time() + backoff + jitter
    failed_attempts[key] = {
        "attempts": attempts,
        "next_retry_ts": next_retry,
        "last_error": reason,
    }

    _, repo_id, revision, _, _ = key
    wait_seconds = max(1, int(next_retry - time.time()))
    logger.warning(
        f"[{repo_id}@{revision[:8]}] Seeding attempt failed ({reason}). "
        f"Retry in {wait_seconds}s (attempt {attempts})."
    )


def _process_seedable(
    repo_id: str,
    revision: str,
    tracker_client,
    manager,
    seeding_set: Set[SeedSessionKey],
    failed_attempts: Dict[SeedSessionKey, Dict[str, object]],
    *,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> bool:
    """Process a single seedable model: create torrent if needed, start seeding."""
    key = _seeding_key(
        repo_type,
        repo_id,
        revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )

    logger.info(f"[{repo_id}@{revision[:8]}] Processing seedable model...")

    try:
        from .torrent_cache import delete_cached_torrent, resolve_torrent_data, save_torrent_to_cache
        from .torrent_creator import (
            create_and_register_torrent,
            ensure_registered,
            torrent_matches_completed_source,
        )

        # Try to get existing torrent (local cache or tracker)
        logger.info(f"[{repo_id}] Checking for existing torrent...")
        torrent_data = resolve_torrent_data(repo_id, revision, tracker_client, repo_type=repo_type)
        if torrent_data and not torrent_matches_completed_source(
            repo_id,
            revision,
            torrent_data,
            repo_type=repo_type,
            cache_dir=cache_dir,
            local_dir=local_dir,
        ):
            logger.warning(
                f"[{repo_id}] Ignoring cached/downloaded torrent because it does not match the verified source"
            )
            delete_cached_torrent(repo_id, revision, repo_type=repo_type)
            torrent_data = None

        if not torrent_data:
            # No torrent exists anywhere — we need to create it
            logger.info(
                f"[{repo_id}] No torrent found, creating and registering..."
            )
            torrent_info = create_and_register_torrent(
                repo_id=repo_id,
                revision=revision,
                repo_type=repo_type,
                name=repo_id,
                tracker_client=tracker_client,
                cache_dir=cache_dir,
                local_dir=local_dir,
            )
            if torrent_info:
                torrent_data = torrent_info.get("torrent_data")
                logger.info(
                    f"[{repo_id}] ✓ Torrent created and registered "
                    f"(info_hash: {torrent_info['info_hash'][:16]}...)"
                )
            else:
                logger.warning(f"[{repo_id}] Failed to create torrent")
                _record_seed_failure(key, failed_attempts, "create_torrent_failed")
                return False
        else:
            logger.info(f"[{repo_id}] Found existing torrent ({len(torrent_data)} bytes)")
            # The announce URL embedded in a cached/downloaded torrent may be stale
            # (e.g. created when the tracker was localhost:8080).  Rewrite it to
            # match the current tracker so libtorrent announces to the right server.
            torrent_data = _rewrite_announce_url(torrent_data, tracker_client.tracker_url, repo_id)
            save_torrent_to_cache(repo_id, revision, torrent_data, repo_type=repo_type)
            # Ensure the tracker knows about this torrent — it may only exist
            # in local cache (e.g. created when the tracker was unreachable).
            ensure_registered(repo_id, revision, repo_type, torrent_data, tracker_client)

        # Start seeding
        logger.info(f"[{repo_id}] Starting seeding task...")
        success = manager.register_seeding_task(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            tracker_client=tracker_client,
            torrent_data=torrent_data,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )

        if success:
            seeding_set.add(key)
            failed_attempts.pop(key, None)
            logger.info(f"[{repo_id}] ✓ Seeding started successfully")
            return True
        else:
            logger.warning(f"[{repo_id}] ✗ Failed to start seeding (register_seeding_task returned False)")
            _record_seed_failure(key, failed_attempts, "register_seeding_task_false")
            return False

    except Exception as e:
        logger.error(f"[{repo_id}] Error processing seedable: {e}", exc_info=True)
        _record_seed_failure(key, failed_attempts, f"exception:{type(e).__name__}")
        return False
