"""
CLI tool for llmpt.
"""

import sys
import os
import re
import argparse
import logging
import signal
from collections import defaultdict


def main():
    """Main entry point for llmpt-cli."""
    parser = argparse.ArgumentParser(
        description='llmpt-cli: P2P-accelerated HuggingFace Hub CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a model via P2P
  llmpt-cli download meta-llama/Llama-2-7b

  # Download with custom tracker
  llmpt-cli download gpt2 --tracker http://tracker.example.com

  # Manage the background seeding daemon
  llmpt-cli start
  llmpt-cli scan
  llmpt-cli status
  llmpt-cli unseed gpt2
  llmpt-cli unseed model/gpt2@71034c5
        """
    )

    parser.add_argument(
        '--tracker',
        default=None,
        help='Tracker server URL'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show informational logs during downloads'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show debug logs for troubleshooting'
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Commands',
        metavar='{download,start,status,scan,unseed,stop,restart}',
    )

    # Download command
    download_parser = subparsers.add_parser(
        'download',
        help='Download a model via P2P'
    )
    download_parser.add_argument(
        'repo_id',
        help='Repository ID (e.g., meta-llama/Llama-2-7b)'
    )
    download_parser.add_argument(
        '--local-dir',
        help='Local directory to save files'
    )
    download_parser.add_argument(
        '--repo-type',
        default='model',
        choices=['model', 'dataset', 'space'],
        help='Repository type'
    )
    download_parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='P2P download timeout in seconds (default: HF_P2P_TIMEOUT or 300)'
    )
    download_parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='libtorrent listen port (default: HF_P2P_PORT or auto-select)'
    )
    download_parser.add_argument(
        '--hf-token',
        default=None,
        help='Hugging Face token for WebSeed access to private/gated repos'
    )
    webseed_group = download_parser.add_mutually_exclusive_group()
    webseed_group.add_argument(
        '--webseed',
        dest='webseed',
        action='store_true',
        default=None,
        help='Enable WebSeed fallback (default: HF_P2P_WEBSEED or true)'
    )
    webseed_group.add_argument(
        '--no-webseed',
        dest='webseed',
        action='store_false',
        help='Disable WebSeed fallback'
    )
    download_parser.add_argument(
        '--disable-utp',
        dest='disable_utp',
        action='store_true',
        default=None,
        help='Disable uTP and force TCP-only BitTorrent transport'
    )

    # Start command
    start_parser = subparsers.add_parser(
        'start',
        help='Start the background seeding daemon'
    )
    start_parser.add_argument(
        '--foreground',
        action='store_true',
        help='Run in the foreground (for debugging)'
    )
    start_parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='libtorrent listen port (default: HF_P2P_PORT or auto-select)'
    )
    start_parser.add_argument(
        '--disable-utp',
        dest='disable_utp',
        action='store_true',
        default=None,
        help='Disable uTP and force TCP-only BitTorrent transport'
    )

    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show background seeding status'
    )

    # Scan command
    scan_parser = subparsers.add_parser(
        'scan',
        help='Force an immediate cache rescan and import verification'
    )

    # Unseed command
    unseed_parser = subparsers.add_parser(
        'unseed',
        help='Stop seeding a specific active daemon session'
    )
    unseed_parser.add_argument(
        'repo_id',
        help='Repository ID or target like model/org/repo@7362d24'
    )
    unseed_parser.add_argument(
        '--revision',
        default=None,
        help='Git commit hash or branch name'
    )
    unseed_parser.add_argument(
        '--repo-type',
        default=None,
        choices=['model', 'dataset', 'space'],
        help='Optional repository type filter (only needed if repo_id is ambiguous)'
    )
    unseed_parser.add_argument(
        '--forget',
        action='store_true',
        help='Also forget the matched custom storage entry from known_storage.json'
    )

    # Stop command
    stop_parser = subparsers.add_parser(
        'stop',
        help='Stop the background seeding daemon'
    )

    # Restart command
    restart_parser = subparsers.add_parser(
        'restart',
        help='Restart the background seeding daemon'
    )
    restart_parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='libtorrent listen port (default: HF_P2P_PORT or auto-select)'
    )
    restart_parser.add_argument(
        '--disable-utp',
        dest='disable_utp',
        action='store_true',
        default=None,
        help='Disable uTP and force TCP-only BitTorrent transport'
    )

    # Internal hidden command for daemon subprocess
    internal_daemon_parser = subparsers.add_parser(
        '_internal_daemon_start',
        help=argparse.SUPPRESS
    )
    internal_daemon_parser.add_argument('--tracker', required=True)
    internal_daemon_parser.add_argument('--port', type=int, default=None)
    internal_daemon_parser.add_argument('--disable-utp', action='store_true', default=False)
    subparsers._choices_actions = [
        action
        for action in subparsers._choices_actions
        if action.dest != '_internal_daemon_start'
    ]

    args = parser.parse_args()

    # Setup logging
    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format='[%(name)s] %(levelname)s: %(message)s'
    )

    # Handle SIGTERM (e.g. `kill <pid>`) gracefully — same as Ctrl+C.
    # Without this, SIGTERM kills the process without running atexit callbacks,
    # leaving hardlinks and temp files behind.
    def _handle_sigterm(signum, frame):
        from llmpt import shutdown
        shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Execute command
    if args.command == 'download':
        cmd_download(args)
    elif args.command == 'start':
        cmd_start(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'scan':
        cmd_scan(args)
    elif args.command == 'unseed':
        cmd_unseed(args)
    elif args.command == 'stop':
        cmd_stop(args)
    elif args.command == 'restart':
        cmd_restart(args)
    elif args.command == '_internal_daemon_start':
        _cmd_internal_daemon_start(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_download(args):
    """Execute download command."""
    from llmpt import enable_p2p

    # Enable P2P FIRST, so patches are applied
    enable_p2p(
        tracker_url=args.tracker,
        timeout=args.timeout,
        port=args.port,
        hf_token=args.hf_token,
        webseed=args.webseed,
        disable_utp=args.disable_utp,
        verbose=args.verbose or args.debug,
    )

    # NOW import the patched function
    from huggingface_hub import snapshot_download

    print(f"Downloading {args.repo_id}...")

    # Download
    path = snapshot_download(
        args.repo_id,
        repo_type=args.repo_type,
        local_dir=args.local_dir
    )

    print(f"✓ Downloaded to: {path}")



def cmd_start(args):
    """Execute start command."""
    from llmpt.daemon import start_daemon, is_daemon_running, LOG_FILE

    tracker_url = _resolve_tracker_url(args.tracker)
    port = _resolve_port(args.port)

    existing = is_daemon_running()
    if existing:
        print(f"Daemon already running (PID: {existing})")
        return

    print("Starting daemon...")
    pid = start_daemon(
        tracker_url=tracker_url,
        port=port,
        foreground=args.foreground,
        disable_utp=args.disable_utp,
    )
    if pid:
        print(f"✓ Daemon started (PID: {pid})")
        print(f"  Tracker: {tracker_url}")
        print(f"  Logs: {LOG_FILE}")
    else:
        print("Error: Failed to start daemon")
        sys.exit(1)


def cmd_stop(args):
    """Execute stop command."""
    from llmpt.daemon import stop_daemon

    if stop_daemon():
        print("✓ Daemon stopped")
    else:
        print("Daemon is not running")


def cmd_restart(args):
    """Execute restart command."""
    from llmpt.daemon import start_daemon, stop_daemon
    import time

    tracker_url = _resolve_tracker_url(args.tracker)
    port = _resolve_port(args.port)

    stop_daemon()
    time.sleep(0.5)
    pid = start_daemon(
        tracker_url=tracker_url,
        port=port,
        disable_utp=args.disable_utp,
    )
    if pid:
        print(f"✓ Daemon restarted (PID: {pid})")
    else:
        print("Error: Failed to restart daemon")
        sys.exit(1)


def _resolve_tracker_url(explicit_tracker):
    """Resolve tracker URL from CLI arg, env, or hardcoded default."""
    return explicit_tracker or os.getenv('HF_P2P_TRACKER') or 'http://localhost:8080'


def _resolve_port(explicit_port):
    """Resolve listen port from CLI arg or env var."""
    if explicit_port is not None:
        return explicit_port
    port_env = os.getenv('HF_P2P_PORT')
    return int(port_env) if port_env else None


_TARGET_REPO_TYPES = {"model", "dataset", "space"}
_COMMIT_PREFIX_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _parse_unseed_target(target: str) -> tuple[bool, str | None, str, str | None]:
    """Parse ``repo_id`` or ``repo_type/repo_id@revision_prefix`` syntax."""
    if "/" not in target:
        return False, None, target, None

    repo_type, remainder = target.split("/", 1)
    if repo_type not in _TARGET_REPO_TYPES:
        return False, None, target, None

    if "@" not in remainder:
        return True, repo_type, remainder, None

    repo_id, revision = remainder.rsplit("@", 1)
    if not repo_id or not revision:
        raise ValueError(
            "invalid target syntax; expected <repo_type>/<repo_id>@<revision-prefix>"
        )
    return True, repo_type, repo_id, revision


def _revision_prefix_lengths(items, *, minimum: int = 7) -> dict[tuple[str, str, str], int]:
    """Compute the shortest unique revision prefix per ``repo_type/repo_id``."""
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in items:
        grouped[(item["repo_type"], item["repo_id"])].append(item["revision"])

    result: dict[tuple[str, str, str], int] = {}
    for (repo_type, repo_id), revisions in grouped.items():
        unique_revisions = sorted(set(revisions))
        for revision in unique_revisions:
            prefix_len = minimum
            while prefix_len < len(revision):
                prefix = revision[:prefix_len]
                clashes = [
                    candidate
                    for candidate in unique_revisions
                    if candidate != revision and candidate.startswith(prefix)
                ]
                if not clashes:
                    break
                prefix_len += 1
            result[(repo_type, repo_id, revision)] = min(prefix_len, len(revision))
    return result


def _format_target(
    repo_type: str,
    repo_id: str,
    revision: str,
    prefix_lengths: dict[tuple[str, str, str], int],
) -> str:
    prefix_len = prefix_lengths.get((repo_type, repo_id, revision), min(7, len(revision)))
    return f"{repo_type}/{repo_id}@{revision[:prefix_len]}"


def _aggregate_status_rows(sessions: dict[str, dict]) -> list[dict]:
    """Collapse multiple storage-backed sessions into logical repo/revision rows."""
    groups: dict[tuple[str, str, str], dict] = {}
    for info in sessions.values():
        repo_type = info.get("repo_type", "model")
        repo_id = info.get("repo_id")
        revision = info.get("revision")
        if not repo_id or not revision:
            continue

        key = (repo_type, repo_id, revision)
        group = groups.setdefault(
            key,
            {
                "repo_type": repo_type,
                "repo_id": repo_id,
                "revision": revision,
                "uploaded": 0,
                "peers": 0,
                "upload_rate": 0,
                "source_count": 0,
                "source_statuses": set(),
                "torrent_statuses": set(),
                "session_statuses": set(),
                "mapping_complete": True,
            },
        )
        group["uploaded"] += int(info.get("uploaded", 0) or 0)
        group["peers"] += int(info.get("peers", 0) or 0)
        group["upload_rate"] += int(info.get("upload_rate", 0) or 0)
        reported_source_count = info.get("source_count")
        if reported_source_count is None:
            group["source_count"] += 1
        else:
            group["source_count"] = max(
                group["source_count"],
                int(reported_source_count or 0),
            )
        group["source_statuses"].add(info.get("source_status", "unknown"))
        group["torrent_statuses"].add(info.get("torrent_status", "unknown"))
        group["session_statuses"].add(info.get("session_status", "unknown"))

        mapped_files = info.get("mapped_files")
        total_files = info.get("total_files")
        if total_files and mapped_files is not None and int(mapped_files) < int(total_files):
            group["mapping_complete"] = False
        elif info.get("full_mapping") is False:
            group["mapping_complete"] = False

    rows = list(groups.values())
    prefix_lengths = _revision_prefix_lengths(rows)
    for row in rows:
        row["target"] = _format_target(
            row["repo_type"],
            row["repo_id"],
            row["revision"],
            prefix_lengths,
        )
    rows.sort(key=lambda row: row["target"])
    return rows


def _status_rows(sessions: dict[str, dict]) -> list[dict]:
    """Render-ready rows for daemon status responses.

    The daemon now reports one logical torrent session per repo/revision and
    includes ``source_count`` explicitly, so the CLI no longer needs to
    aggregate multiple storage-backed sessions here.
    """
    rows = []
    for info in sessions.values():
        repo_type = info.get("repo_type", "model")
        repo_id = info.get("repo_id")
        revision = info.get("revision")
        if not repo_id or not revision:
            continue

        mapped_files = info.get("mapped_files")
        total_files = info.get("total_files")
        mapping_complete = True
        if total_files and mapped_files is not None and int(mapped_files) < int(total_files):
            mapping_complete = False
        elif info.get("full_mapping") is False:
            mapping_complete = False

        rows.append(
            {
                "repo_type": repo_type,
                "repo_id": repo_id,
                "revision": revision,
                "uploaded": int(info.get("uploaded", 0) or 0),
                "peers": int(info.get("peers", 0) or 0),
                "upload_rate": int(info.get("upload_rate", 0) or 0),
                "source_count": max(1, int(info.get("source_count", 1) or 1)),
                "source_statuses": {info.get("source_status", "unknown")},
                "torrent_statuses": {info.get("torrent_status", "unknown")},
                "session_statuses": {info.get("session_status", "unknown")},
                "mapping_complete": mapping_complete,
            }
        )

    prefix_lengths = _revision_prefix_lengths(rows)
    for row in rows:
        row["target"] = _format_target(
            row["repo_type"],
            row["repo_id"],
            row["revision"],
            prefix_lengths,
        )
    rows.sort(key=lambda row: row["target"])
    return rows


def _display_status_label(row: dict) -> str:
    source_statuses = row.get("source_statuses", set())
    torrent_statuses = row.get("torrent_statuses", set())
    session_statuses = row.get("session_statuses", set())

    if "blocked" in source_statuses:
        return "blocked"
    if "partial" in source_statuses:
        return "partial"
    if "error" in source_statuses:
        return "error"
    if not row.get("mapping_complete", True):
        return "partial-map"
    if "registered" in torrent_statuses and "active" in session_statuses:
        return "active"
    if "local_only" in torrent_statuses:
        return "local-only"
    if "absent" in torrent_statuses and "degraded" in session_statuses:
        return "unregistered"
    if "degraded" in session_statuses:
        return "degraded"
    if "inactive" in session_statuses:
        return "inactive"
    return "unknown"


def _print_removed_targets(removed_sessions: list[dict]) -> None:
    if not removed_sessions:
        return

    rows = _aggregate_status_rows(
        {
            str(index): {
                "repo_type": item.get("repo_type", "model"),
                "repo_id": item.get("repo_id"),
                "revision": item.get("revision"),
                "uploaded": 0,
                "peers": 0,
                "upload_rate": 0,
                "source_status": "unknown",
                "torrent_status": "unknown",
                "session_status": "inactive",
                "mapped_files": 0,
                "total_files": 0,
                "full_mapping": True,
            }
            for index, item in enumerate(removed_sessions)
        }
    )
    for row in rows:
        suffix = f" ({row['source_count']} sources)" if row["source_count"] > 1 else ""
        print(f"  {row['target']}{suffix}")
def cmd_status(args):
    """Execute status command."""
    from llmpt.daemon import is_daemon_running
    from llmpt.ipc import query_daemon
    from llmpt.utils import format_bytes

    pid = is_daemon_running()
    if not pid:
        print("Daemon is not running")
        print("  Start with: llmpt-cli start")
        return

    # Query detailed status via IPC
    response = query_daemon("status")
    if not response:
        print(f"Daemon running (PID: {pid}) but not responding to IPC")
        return

    print(f"llmptd (PID: {pid}) — running")
    
    daemon_tracker = response.get('tracker_url')
    if daemon_tracker:
        print(f"  Tracker: {daemon_tracker}")

    listen_port = response.get('port')
    if listen_port:
        print(f"  Port: {listen_port}")
        
    sessions = response.get('sessions', {})
    if not sessions:
        print("  No active seeding tasks")
        return

    rows = _status_rows(sessions)
    if not rows:
        print("Error: daemon status format is outdated; restart the daemon")
        sys.exit(1)
    print(f"\nActive seeding: {len(rows)}\n")
    width = max(len(row["target"]) for row in rows)
    for row in rows:
        sources_suffix = f"  {row['source_count']} sources" if row["source_count"] > 1 else ""
        print(
            f"  {row['target']:<{width}}  "
            f"↑{format_bytes(row['uploaded'])}  "
            f"{row['peers']} peers  "
            f"{format_bytes(row['upload_rate'])}/s  "
            f"{_display_status_label(row)}"
            f"{sources_suffix}"
        )


def cmd_scan(args):
    """Execute scan command."""
    from llmpt.daemon import is_daemon_running
    from llmpt.ipc import query_daemon

    pid = is_daemon_running()
    if not pid:
        print("Daemon is not running")
        print("  Start with: llmpt-cli start")
        return

    response = query_daemon("scan")
    if not response:
        print(f"Daemon running (PID: {pid}) but not responding to IPC")
        return

    if response.get("status") != "ok":
        print(f"Error: {response.get('message', 'scan failed')}")
        sys.exit(1)

    print("✓ Cache rescan triggered")


def cmd_unseed(args):
    """Execute unseed command."""
    from llmpt.daemon import is_daemon_running
    from llmpt.ipc import query_daemon
    from llmpt.utils import resolve_commit_hash

    pid = is_daemon_running()
    if not pid:
        print("Daemon is not running")
        print("  Start with: llmpt-cli start")
        return

    try:
        used_target_syntax, target_repo_type, repo_id, target_revision = _parse_unseed_target(args.repo_id)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if used_target_syntax and args.repo_type and args.repo_type != target_repo_type:
        print("Error: target repo type conflicts with --repo-type")
        sys.exit(1)
    if used_target_syntax and args.revision and target_revision and args.revision != target_revision:
        print("Error: target revision conflicts with --revision")
        sys.exit(1)

    repo_type = target_repo_type or args.repo_type
    raw_revision = target_revision if target_revision is not None else args.revision
    revision = None
    if raw_revision is not None:
        normalized_raw_revision = raw_revision.lower() if _COMMIT_PREFIX_RE.match(raw_revision.lower()) else raw_revision
        if used_target_syntax or _COMMIT_PREFIX_RE.match(normalized_raw_revision):
            revision = normalized_raw_revision
        else:
            try:
                revision = resolve_commit_hash(
                    repo_id, normalized_raw_revision, repo_type=repo_type or "model"
                )
            except Exception:
                revision = normalized_raw_revision

            if revision != normalized_raw_revision:
                print(f"Resolved revision: {normalized_raw_revision} → {revision}")

    response = query_daemon(
        "unseed",
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        forget=args.forget,
    )
    if not response:
        print(f"Daemon running (PID: {pid}) but not responding to IPC")
        return

    if response.get("status") != "ok":
        print(f"Error: {response.get('message', 'unseed failed')}")
        sys.exit(1)

    removed_count = response.get("removed_count", 0)
    forgotten = response.get("forgotten", {})
    print(f"✓ Removed {removed_count} active seeding session(s)")
    _print_removed_targets(response.get("removed_sessions", []))
    if args.forget:
        print(
            "  Forgot registry entries: "
            f"hub_cache_roots={forgotten.get('hub_cache_roots_removed', 0)}, "
            f"local_dir_sources={forgotten.get('local_dir_sources_removed', 0)}"
        )


def _cmd_internal_daemon_start(args):
    """Internal entrypoint for the daemon subprocess."""
    import os
    import logging
    from llmpt.daemon import _daemon_main, _write_pid, _remove_pid, LLMPT_DIR, LOG_FILE
    
    os.makedirs(LLMPT_DIR, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_log = logging.getLogger("llmpt")
    root_log.addHandler(fh)
    root_log.setLevel(logging.INFO)

    try:
        _daemon_main(args.tracker, port=args.port, disable_utp=args.disable_utp)
    except Exception as e:
        logger = logging.getLogger("llmpt.daemon")
        logger.error(f"Daemon crashed: {e}", exc_info=True)
    finally:
        _remove_pid()


if __name__ == '__main__':
    main()
