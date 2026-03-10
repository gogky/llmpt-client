"""
CLI tool for llmpt.
"""

import sys
import os
import argparse
import logging
import signal


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
        help='Enable verbose logging'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

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
        help='Repository ID'
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

    # Internal hidden command for daemon subprocess
    internal_daemon_parser = subparsers.add_parser(
        '_internal_daemon_start',
        help=argparse.SUPPRESS
    )
    internal_daemon_parser.add_argument('--tracker', required=True)
    internal_daemon_parser.add_argument('--port', type=int, default=None)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
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
        verbose=args.verbose,
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
    pid = start_daemon(tracker_url=tracker_url, port=port)
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

    print(f"\nActive seeding sessions: {len(sessions)}\n")
    for repo_key, info in sessions.items():
        uploaded = info.get('uploaded', 0)
        peers = info.get('peers', 0)
        rate = info.get('upload_rate', 0)
        mapped_files = info.get('mapped_files')
        total_files = info.get('total_files')
        source_status = info.get('source_status', 'unknown')
        torrent_status = info.get('torrent_status', 'unknown')
        session_status = info.get('session_status', 'unknown')
        state_summary = (
            f"source:{source_status} │ "
            f"torrent:{torrent_status} │ "
            f"session:{session_status}"
        )
        mapping_suffix = ""
        if total_files:
            mapping_suffix = f" │ mapped {mapped_files}/{total_files}"
        print(
            f"  {repo_key:<45} "
            f"↑ {format_bytes(uploaded):>8}  │ "
            f"{peers} peers │ "
            f"{format_bytes(rate)}/s │ "
            f"{state_summary}"
            f"{mapping_suffix}"
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

    revision = None
    if args.revision is not None:
        raw_revision = args.revision
        try:
            revision = resolve_commit_hash(
                args.repo_id, raw_revision, repo_type=args.repo_type or "model"
            )
        except Exception:
            revision = raw_revision

        if revision != raw_revision:
            print(f"Resolved revision: {raw_revision} → {revision}")

    response = query_daemon(
        "unseed",
        repo_id=args.repo_id,
        revision=revision,
        repo_type=args.repo_type,
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
        _daemon_main(args.tracker, port=args.port)
    except Exception as e:
        logger = logging.getLogger("llmpt.daemon")
        logger.error(f"Daemon crashed: {e}", exc_info=True)
    finally:
        _remove_pid()


if __name__ == '__main__':
    main()
