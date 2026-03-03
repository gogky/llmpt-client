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

try:
    import libtorrent as lt
except ImportError:
    lt = None

logger = logging.getLogger('llmpt.p2p_batch')

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

    while ctx.is_valid and ctx.handle:
        time.sleep(1)

        now = time.time()

        # --- Periodic diagnostic logging (every 5 s) ---
        if now - last_diag_time > 5:
            last_diag_time = now
            _log_diagnostics(ctx)

        # --- Periodic fastresume save (every 5 s) ---
        if now - last_save_time > 5:
            _save_resume_data(ctx)
            last_save_time = now

        try:
            # --- Process libtorrent alerts ---
            _process_alerts(ctx)

            # --- Check pending file completions ---
            should_break = _check_pending_files(ctx)
            if should_break:
                break

        except Exception as e:
            logger.error(f"[{ctx.repo_id}] Monitor loop exception: {e}")


def _log_diagnostics(ctx: "SessionContext") -> None:
    """Emit periodic status diagnostics for the torrent session."""
    try:
        if ctx.handle and ctx.handle.is_valid():
            s = ctx.handle.status()
            state_str = _STATE_NAMES[s.state] if s.state < len(_STATE_NAMES) else str(s.state)
            total_pieces = (
                ctx.torrent_info_obj.num_pieces() if ctx.torrent_info_obj else s.num_pieces
            )
            logger.info(
                f"[{ctx.repo_id}] STATUS: state={state_str} "
                f"progress={s.progress * 100:.1f}% "
                f"peers={s.num_peers} seeds={s.num_seeds} "
                f"dl={s.download_rate / 1024:.1f}KB/s ul={s.upload_rate / 1024:.1f}KB/s "
                f"pieces={s.num_pieces}/{total_pieces}"
            )
    except Exception as diag_err:
        logger.debug(f"[{ctx.repo_id}] Diagnostic log error: {diag_err}")


def _save_resume_data(ctx: "SessionContext") -> None:
    """Request a fastresume data save from libtorrent."""
    if ctx.handle and ctx.handle.is_valid():
        ctx.handle.save_resume_data(lt.save_resume_flags_t.flush_disk_cache)


def _process_alerts(ctx: "SessionContext") -> None:
    """Pop and handle libtorrent alerts (fastresume save results)."""
    alerts = ctx.lt_session.pop_alerts()
    for alert in alerts:
        if isinstance(alert, lt.save_resume_data_alert) and alert.handle == ctx.handle:
            try:
                resume_data = lt.bencode(alert.params)
                with open(ctx.fastresume_path, "wb") as f:
                    f.write(resume_data)
                logger.debug(f"[{ctx.repo_id}] Saved resume data to {ctx.fastresume_path}")
            except Exception as e:
                logger.warning(f"[{ctx.repo_id}] Failed to write resume data: {e}")

        elif isinstance(alert, lt.save_resume_data_failed_alert) and alert.handle == ctx.handle:
            logger.debug(f"[{ctx.repo_id}] Save resume data failed: {alert.message()}")


def _check_pending_files(ctx: "SessionContext") -> bool:
    """Check for completed files and deliver them. Returns True if the loop should break."""
    with ctx.lock:
        if not ctx.handle:
            return True

        status = ctx.handle.status()

        # Check for torrent-level errors
        if _has_torrent_error(status):
            err_msg = _get_error_message(status)
            logger.error(f"[{ctx.repo_id}] Torrent error: {err_msg}")
            ctx.is_valid = False
            return True

        # Identify files still waiting for completion
        pending_files = [f for f, e in ctx.file_events.items() if not e.is_set()]
        if not pending_files:
            return False

        # If metadata finally arrived, belatedly map requested files
        if not ctx.torrent_info_obj and ctx.handle.status().has_metadata:
            ctx.torrent_info_obj = ctx.handle.torrent_file()
            num_files = ctx.torrent_info_obj.num_files()
            ctx.handle.prioritize_files([0] * num_files)
            logger.info(f"[{ctx.repo_id}] Background metadata resolved! {num_files} files.")

        if not ctx.torrent_info_obj:
            return False  # Still downloading metadata

        # Skip progress checking while a recheck is in progress
        # (state 1=checking_files, 7=checking_resume_data)
        if status.state in (1, 7):
            return False

        file_progress = ctx.handle.file_progress()
        files = ctx.torrent_info_obj.files()

        for filename in list(pending_files):
            file_index = ctx._find_file_index(filename)
            if file_index is not None:
                destination = ctx.file_destinations[filename]

                # Belatedly set priority for files queued before metadata arrived
                if ctx.handle.file_priorities()[file_index] == 0:
                    ctx.handle.file_priority(file_index, 1)
                    logger.info(f"[{ctx.repo_id}] Belatedly prioritized {filename} (Index {file_index})")

                file_size = files.file_size(file_index)
                progress_bytes = file_progress[file_index]

                if progress_bytes == file_size and file_size > 0:
                    # File is fully downloaded — deliver it to the HF destination
                    src = ctx._get_lt_disk_path(file_index)
                    try:
                        ctx._deliver_file(src, destination)
                        logger.info(f"[{ctx.repo_id}] File {filename} complete. Delivered {src} -> {destination}")
                        ctx.file_events[filename].set()
                    except Exception as deliver_err:
                        logger.error(f"[{ctx.repo_id}] Failed to deliver {filename}: {deliver_err}")
                        # Don't set event — let it timeout and fallback to HTTP

                    ctx.handle.file_priority(file_index, 0)
                    ctx.handle.save_resume_data(lt.save_resume_flags_t.flush_disk_cache)

    return False


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
