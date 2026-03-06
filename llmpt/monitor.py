"""
Background monitor loop for P2P torrent sessions.

Extracted from SessionContext._monitor_loop to separate the progress monitoring,
diagnostic logging, and file delivery concerns from the torrent lifecycle management.
"""

import os
import time
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session_context import SessionContext

from .utils import lt

logger = logging.getLogger(__name__)

# Human-readable names for libtorrent torrent states
_STATE_NAMES = [
    'queued', 'checking_files', 'downloading_metadata',
    'downloading', 'finished', 'seeding', 'allocating',
    'checking_resume_data',
]


def run_monitor_loop(ctx: "SessionContext") -> None:
    """Background thread entry point: monitor progress and trigger file-completion events.

    Args:
        ctx: The SessionContext whose torrent handle is being monitored.
    """
    logger.info(f"[{ctx.repo_id}] Monitor thread started.")

    last_save_time = time.time()
    last_diag_time = 0  # Force first diagnostic log immediately
    last_peer_retry_time = 0  # Track last connect_peer retry

    try:
        while True:
            time.sleep(1)

            # Snapshot shared state under lock to avoid TOCTOU races with
            # stop_seeding() / shutdown() which set handle=None / is_valid=False,
            # and download_file() which resets seed_start_time=None.
            with ctx.lock:
                if not ctx.is_valid or not ctx.handle:
                    break
                seed_start = ctx.seed_start_time
                seed_dur = ctx.seed_duration

            try:
                now = time.time()

                # --- Periodic diagnostic logging (every 5 s) ---
                if now - last_diag_time > 5:
                    last_diag_time = now
                    _log_diagnostics(ctx)

                # --- Periodic peer reconnection for test environments (every 10 s) ---
                if now - last_peer_retry_time > 10:
                    last_peer_retry_time = now
                    _retry_test_peer_connection(ctx)

                # --- Periodic fastresume save (every 5 s) ---
                if now - last_save_time > 5:
                    _save_resume_data(ctx)
                    last_save_time = now

                # --- Check seed_duration timeout (using snapshot taken under lock) ---
                if seed_start is not None and seed_dur > 0:
                    elapsed = now - seed_start
                    if elapsed >= seed_dur:
                        logger.info(
                            f"[{ctx.repo_id}] Seed duration {seed_dur}s elapsed. "
                            f"Stopping auto-seed."
                        )
                        ctx._cleanup_download_sources()
                        ctx.is_valid = False
                        break

                # --- Dispatch alerts from the global lt_session to per-session inboxes ---
                from .p2p_batch import P2PBatchManager
                P2PBatchManager().dispatch_alerts()

                # --- Process this session's alerts from its inbox ---
                _process_alerts(ctx)

                # --- Check pending file completions ---
                should_break = _check_pending_files(ctx)
                if should_break:
                    break

            except Exception as e:
                logger.error(f"[{ctx.repo_id}] Monitor loop exception: {e}")

    except Exception as e:
        logger.error(f"[{ctx.repo_id}] Monitor thread crashed: {e}")

    finally:
        # Ensure is_valid is False so that download_file() callers waiting on
        # events can detect that the monitor is no longer running and fail fast
        # instead of blocking until timeout.
        ctx.is_valid = False
        logger.info(f"[{ctx.repo_id}] Monitor thread exited.")


def _log_diagnostics(ctx: "SessionContext") -> None:
    """Emit periodic status diagnostics for the torrent session."""
    with ctx.lock:
        handle = ctx.handle
        ti = ctx.torrent_info_obj
    if not handle:
        return
    try:
        if handle.is_valid():
            s = handle.status()
            state_str = _STATE_NAMES[s.state] if s.state < len(_STATE_NAMES) else str(s.state)
            total_pieces = ti.num_pieces() if ti else s.num_pieces
            logger.debug(
                f"[{ctx.repo_id}] STATUS: state={state_str} "
                f"progress={s.progress * 100:.1f}% "
                f"peers={s.num_peers} seeds={s.num_seeds} "
                f"dl={s.download_rate / 1024:.1f}KB/s ul={s.upload_rate / 1024:.1f}KB/s "
                f"pieces={s.num_pieces}/{total_pieces}"
            )
    except Exception as diag_err:
        logger.debug(f"[{ctx.repo_id}] Diagnostic log error: {diag_err}")


def _retry_test_peer_connection(ctx: "SessionContext") -> None:
    """Periodically retry connect_peer in Docker test environments.

    In Docker bridge networks, the external tracker returns the host's public
    NAT IP which is unreachable from within the container network. This function
    retries the direct Docker-internal connection when no peers are connected.
    """
    if not ctx.test_peer_addr:
        return

    with ctx.lock:
        handle = ctx.handle
    if not handle:
        return

    try:
        if handle.is_valid():
            s = handle.status()
            if s.num_peers == 0:
                handle.connect_peer(ctx.test_peer_addr, 0)
                logger.info(
                    f"[{ctx.repo_id}] Retrying connect_peer to "
                    f"{ctx.test_peer_addr[0]}:{ctx.test_peer_addr[1]} (peers=0)"
                )
    except Exception as e:
        logger.debug(f"[{ctx.repo_id}] Peer reconnect attempt failed: {e}")


def _save_resume_data(ctx: "SessionContext") -> None:
    """Request a fastresume data save from libtorrent."""
    with ctx.lock:
        handle = ctx.handle
    if not handle:
        return
    try:
        if handle.is_valid():
            handle.save_resume_data(lt.save_resume_flags_t.flush_disk_cache)
    except Exception as e:
        logger.debug(f"[{ctx.repo_id}] save_resume_data error: {e}")


def _process_alerts(ctx: "SessionContext") -> None:
    """Process alerts from this session's inbox (populated by dispatch_alerts)."""
    with ctx.alert_lock:
        alerts = list(ctx.pending_alerts)
        ctx.pending_alerts.clear()

    for alert in alerts:
        if isinstance(alert, lt.save_resume_data_alert):
            try:
                resume_data = lt.bencode(alert.params)
                with open(ctx.fastresume_path, "wb") as f:
                    f.write(resume_data)
                logger.debug(f"[{ctx.repo_id}] Saved resume data to {ctx.fastresume_path}")
            except Exception as e:
                logger.warning(f"[{ctx.repo_id}] Failed to write resume data: {e}")

        elif isinstance(alert, lt.save_resume_data_failed_alert):
            logger.debug(f"[{ctx.repo_id}] Save resume data failed: {alert.message()}")
            
        elif isinstance(alert, lt.peer_error_alert):
            logger.warning(f"[{ctx.repo_id}] PEER ERROR: {alert.message()}")
            
        elif isinstance(alert, lt.peer_disconnected_alert):
            logger.warning(f"[{ctx.repo_id}] PEER DISCONNECTED: {alert.message()}")
            
        elif isinstance(alert, lt.torrent_error_alert):
            logger.warning(f"[{ctx.repo_id}] TORRENT ERROR: {alert.message()}")
            
        elif isinstance(alert, lt.hash_failed_alert):
            logger.warning(f"[{ctx.repo_id}] HASH FAILED: {alert.message()}")

        elif isinstance(alert, lt.file_error_alert):
            logger.warning(f"[{ctx.repo_id}] FILE ERROR: {alert.message()}")


def _check_pending_files(ctx: "SessionContext") -> bool:
    """Check for completed files and deliver them. Returns True if the loop should break.

    Three-phase pipeline to minimize lock contention:
      Phase 1 (lock): health check, metadata resolution, snapshot ready files
      Phase 2 (no lock): file delivery I/O
      Phase 3 (lock): event notification, seed timer update
    """
    # ── Phase 1: read-only snapshot under lock ─────────────────────────
    with ctx.lock:
        health = _check_session_health(ctx)
        if health is not None:
            return health

        _resolve_pending_metadata(ctx)

        status = ctx.handle.status()
        if status.state in (1, 7):  # checking_files / checking_resume_data
            return False

        ready_files = _collect_ready_files(ctx)

    # ── Phase 2: file delivery I/O (no lock) ──────────────────────────
    delivered = []
    for src, dst, filename in ready_files:
        try:
            ctx._deliver_file(src, dst)
            logger.info(f"[{ctx.repo_id}] File {filename} complete. Delivered {src} -> {dst}")
            delivered.append(filename)
        except Exception as deliver_err:
            logger.error(f"[{ctx.repo_id}] Failed to deliver {filename}: {deliver_err}")
            # Don't add to delivered — let it timeout and fallback to HTTP

    # ── Phase 3: notify waiters and update state (lock) ───────────────
    if delivered:
        with ctx.lock:
            for filename in delivered:
                if filename in ctx.file_events:
                    ctx.file_events[filename].set()

            handle = ctx.handle
        if handle:
            try:
                handle.save_resume_data(lt.save_resume_flags_t.flush_disk_cache)
            except Exception:
                pass

    # Check if all files are now delivered → start seed timer
    with ctx.lock:
        _update_seed_timer(ctx)

    return False


def _check_session_health(ctx: "SessionContext"):
    """Check handle and torrent-level errors. Must be called under ctx.lock.

    Returns:
        True  - loop should break (error or no handle)
        False - no pending files, nothing to do
        None  - continue to next phase
    """
    if not ctx.handle:
        return True

    status = ctx.handle.status()
    if _has_torrent_error(status):
        err_msg = _get_error_message(status)
        logger.error(f"[{ctx.repo_id}] Torrent error: {err_msg}")
        ctx.is_valid = False
        return True

    pending_files = [f for f, e in ctx.file_events.items() if not e.is_set()]
    if not pending_files:
        _update_seed_timer(ctx)
        return False

    return None


def _resolve_pending_metadata(ctx: "SessionContext") -> None:
    """If metadata just arrived, initialize torrent_info and file priorities.
    Must be called under ctx.lock.
    """
    if not ctx.torrent_info_obj and ctx.handle.status().has_metadata:
        ctx.torrent_info_obj = ctx.handle.torrent_file()
        num_files = ctx.torrent_info_obj.num_files()
        ctx.handle.prioritize_files([0] * num_files)
        logger.info(f"[{ctx.repo_id}] Background metadata resolved! {num_files} files.")


def _collect_ready_files(ctx: "SessionContext") -> list:
    """Scan pending files and return those that are fully downloaded.
    Must be called under ctx.lock.

    Returns:
        List of (src_path, dst_path, filename) tuples ready for delivery.
    """
    if not ctx.torrent_info_obj:
        return []

    file_progress = ctx.handle.file_progress()
    files = ctx.torrent_info_obj.files()
    ready = []

    pending_files = [f for f, e in ctx.file_events.items() if not e.is_set()]
    for filename in pending_files:
        file_index = ctx._find_file_index(filename)
        if file_index is None:
            continue

        # Belatedly set priority for files queued before metadata arrived
        if ctx.handle.file_priorities()[file_index] == 0:
            ctx.handle.file_priority(file_index, 1)
            logger.info(f"[{ctx.repo_id}] Belatedly prioritized {filename} (Index {file_index})")

        file_size = files.file_size(file_index)
        progress_bytes = file_progress[file_index]

        if progress_bytes == file_size and file_size > 0:
            src = ctx._get_lt_disk_path(file_index)
            destination = ctx.file_destinations[filename]
            ready.append((src, destination, filename))

    return ready


def _update_seed_timer(ctx: "SessionContext") -> None:
    """Start the seed timer if all registered downloads are delivered.
    Must be called under ctx.lock.
    """
    pending = [f for f, e in ctx.file_events.items() if not e.is_set()]
    if pending:
        return
    if ctx.file_events and ctx.auto_seed and ctx.seed_start_time is None:
        ctx.seed_start_time = time.time()
        if ctx.seed_duration > 0:
            logger.info(
                f"[{ctx.repo_id}] All files delivered. "
                f"Auto-seeding for {ctx.seed_duration}s."
            )
        else:
            logger.info(
                f"[{ctx.repo_id}] All files delivered. "
                f"Auto-seeding indefinitely."
            )


def _has_torrent_error(status) -> bool:
    """Check if the torrent status indicates an error."""
    if hasattr(status, 'errc') and status.errc:
        if hasattr(status.errc, 'value') and status.errc.value() != 0:
            return True
        if hasattr(status.errc, 'message') and status.errc.message() != 'Success':
            return True
    elif hasattr(status, 'error') and status.error:
        return True
    return False


def _get_error_message(status) -> str:
    """Extract a human-readable error message from the torrent status."""
    if hasattr(status, 'errc') and hasattr(status.errc, 'message'):
        return status.errc.message()
    return str(getattr(status, 'error', getattr(status, 'errc', 'Unknown')))
