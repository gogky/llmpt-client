"""
llmptd — Background seeding daemon for llmpt.

This module implements a standalone daemon process that:
1. Scans the local HuggingFace cache for seedable models
2. Creates .torrent files and registers them with the tracker
3. Seeds continuously in the background via libtorrent
4. Accepts IPC commands from download clients

Lifecycle:
    llmpt-cli daemon start   →  fork to background, return immediately
    llmpt-cli daemon stop    →  send SIGTERM, daemon exits gracefully
    llmpt-cli daemon status  →  query daemon via IPC
"""

import atexit
import json
import logging
import os
import signal
import sys
import time
from typing import Dict, Optional, Set, Tuple

from .utils import LIBTORRENT_AVAILABLE

logger = logging.getLogger("llmpt.daemon")

# Paths
LLMPT_DIR = os.path.expanduser("~/.cache/llmpt")
PID_FILE = os.path.join(LLMPT_DIR, "daemon.pid")
LOG_FILE = os.path.join(LLMPT_DIR, "daemon.log")

# Daemon configuration
SCAN_INTERVAL = 300  # seconds between full HF cache scans


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

    from .cache_scanner import scan_hf_cache
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

    tracker_client = TrackerClient(tracker_url)
    manager = P2PBatchManager()

    # Track what we're already seeding to avoid duplicate work
    seeding_set: Set[Tuple[str, str, str]] = set()
    # Track what we've already attempted (to avoid re-processing failures)
    attempted_set: Set[Tuple[str, str, str]] = set()

    running = True

    def _signal_handler(signum, frame):
        nonlocal running
        logger.info(f"Received signal {signum}, shutting down...")
        running = False

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # IPC message handler
    def _handle_ipc(msg: dict) -> Optional[dict]:
        action = msg.get("action")

        if action == "seed":
            repo_id = msg.get("repo_id")
            revision = msg.get("revision")
            repo_type = msg.get("repo_type", "model")
            if repo_id and revision:
                key = (repo_id, revision, repo_type)
                if key not in seeding_set:
                    logger.info(f"IPC: seed request for {repo_id}@{revision[:8]}...")
                    _process_seedable(
                        repo_id, revision, repo_type, tracker_client, manager,
                        seeding_set, attempted_set,
                    )
                return {"status": "ok"}

        elif action == "status":
            status = manager.get_all_session_status()
            return {
                "status": "ok",
                "pid": os.getpid(),
                "port": getattr(manager, 'listen_port', None),
                "seeding_count": len(seeding_set),
                "sessions": {
                    k: {
                        "progress": v.get("progress", 0),
                        "state": v.get("state", "unknown"),
                        "uploaded": v.get("uploaded", 0),
                        "peers": v.get("peers", 0),
                        "upload_rate": v.get("upload_rate", 0),
                    }
                    for k, v in status.items()
                },
            }

        elif action == "scan":
            # Clear attempted set to allow re-scanning
            attempted_set.clear()
            return {"status": "ok", "message": "scan triggered"}

        elif action == "ping":
            return {"status": "ok", "pid": os.getpid()}

        return {"status": "error", "message": f"unknown action: {action}"}

    # Start IPC server
    ipc_server = IPCServer(handler=_handle_ipc)
    ipc_server.start()

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
                    tracker_client, manager, seeding_set, attempted_set
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


def _scan_and_seed(
    tracker_client,
    manager,
    seeding_set: Set[Tuple[str, str, str]],
    attempted_set: Set[Tuple[str, str, str]],
) -> None:
    """Scan HF cache and start seeding any new repositories found."""
    from .cache_scanner import scan_hf_cache

    seedable = scan_hf_cache()
    new_count = 0

    for repo_id, revision, repo_type in seedable:
        key = (repo_id, revision, repo_type)
        if key in seeding_set or key in attempted_set:
            continue

        _process_seedable(
            repo_id, revision, repo_type, tracker_client, manager,
            seeding_set, attempted_set,
        )
        new_count += 1

    if new_count > 0:
        logger.info(f"Scan processed {new_count} new models")


def _process_seedable(
    repo_id: str,
    revision: str,
    repo_type: str,
    tracker_client,
    manager,
    seeding_set: Set[Tuple[str, str, str]],
    attempted_set: Set[Tuple[str, str, str]],
) -> None:
    """Process a single seedable repo: create torrent if needed, start seeding."""
    key = (repo_id, revision, repo_type)
    attempted_set.add(key)

    logger.info(f"[{repo_id}@{revision[:8]}] Processing seedable model...")

    try:
        from .torrent_cache import resolve_torrent_data
        from .torrent_creator import create_and_register_torrent

        # Try to get existing torrent (local cache or tracker)
        logger.info(f"[{repo_id}] Checking for existing torrent...")
        torrent_data = resolve_torrent_data(repo_id, revision, tracker_client)

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
            )
            if torrent_info:
                torrent_data = torrent_info.get("torrent_data")
                logger.info(
                    f"[{repo_id}] ✓ Torrent created and registered "
                    f"(info_hash: {torrent_info['info_hash'][:16]}...)"
                )
            else:
                logger.warning(f"[{repo_id}] Failed to create torrent")
                return
        else:
            logger.info(f"[{repo_id}] Found existing torrent ({len(torrent_data)} bytes)")

        # Start seeding
        logger.info(f"[{repo_id}] Starting seeding task...")
        success = manager.register_seeding_task(
            repo_id=repo_id,
            revision=revision,
            tracker_client=tracker_client,
            torrent_data=torrent_data,
        )

        if success:
            seeding_set.add(key)
            logger.info(f"[{repo_id}] ✓ Seeding started successfully")
        else:
            logger.warning(f"[{repo_id}] ✗ Failed to start seeding (register_seeding_task returned False)")

    except Exception as e:
        logger.error(f"[{repo_id}] Error processing seedable: {e}", exc_info=True)

