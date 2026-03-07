"""
CLI tool for llmpt.
"""

import sys
import argparse
import logging
import signal
from pathlib import Path


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

  # Create and seed a torrent
  llmpt-cli seed /path/to/model.bin --repo-id gpt2 --filename model.bin

  # Show seeding status
  llmpt-cli status
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

    # Seed command
    seed_parser = subparsers.add_parser(
        'seed',
        help='Create torrent and start seeding'
    )
    seed_parser.add_argument(
        '--repo-id',
        required=True,
        help='Repository ID'
    )
    seed_parser.add_argument(
        '--revision',
        required=True,
        help='Git commit hash or branch name'
    )
    seed_parser.add_argument(
        '--repo-type',
        default='model',
        help='Repository type'
    )
    seed_parser.add_argument(
        '--name',
        default='HF Model',
        help='Display name'
    )

    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show seeding status'
    )

    # Stop command
    stop_parser = subparsers.add_parser(
        'stop',
        help='Stop seeding'
    )
    stop_parser.add_argument(
        'repo_key',
        nargs='?',
        help='repo_id@revision to stop (e.g. meta-llama/Llama-2-7b@main). Omit to stop all.'
    )

    # Daemon command
    daemon_parser = subparsers.add_parser(
        'daemon',
        help='Manage the background seeding daemon'
    )
    daemon_parser.add_argument(
        'daemon_action',
        choices=['start', 'stop', 'status', 'restart'],
        help='Daemon action: start, stop, status, or restart'
    )
    daemon_parser.add_argument(
        '--foreground',
        action='store_true',
        help='Run daemon in the foreground (for debugging)'
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
    elif args.command == 'seed':
        cmd_seed(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'stop':
        cmd_stop(args)
    elif args.command == 'daemon':
        cmd_daemon(args)
    elif args.command == '_internal_daemon_start':
        _cmd_internal_daemon_start(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_download(args):
    """Execute download command."""
    from llmpt import enable_p2p, get_config

    # Enable P2P FIRST, so patches are applied
    enable_p2p(
        tracker_url=args.tracker
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


def cmd_seed(args):
    """Execute seed command."""
    from llmpt.tracker_client import TrackerClient
    from llmpt.torrent_creator import create_and_register_torrent
    from llmpt.seeder import start_seeding
    from llmpt.utils import resolve_commit_hash

    tracker = TrackerClient(args.tracker or 'http://localhost:8080')

    # Resolve revision (e.g. "main") to a 40-char commit hash so the tracker
    # entry is always keyed by the immutable commit identifier.
    raw_revision = args.revision
    try:
        revision = resolve_commit_hash(
            args.repo_id, raw_revision, repo_type=args.repo_type
        )
    except Exception as e:
        print(f"Error: Could not resolve revision '{raw_revision}': {e}")
        sys.exit(1)

    if revision != raw_revision:
        print(f"Resolved revision: {raw_revision} → {revision}")

    print(f"Resolving caching structure and creating torrent for {args.repo_id}@{revision}...")

    # Create and register torrent
    torrent_info = create_and_register_torrent(
        repo_id=args.repo_id,
        revision=revision,
        repo_type=args.repo_type,
        name=args.name,
        tracker_client=tracker,
    )

    if not torrent_info:
        print("Error: Failed to create or register torrent.")
        print("Hint: Make sure the model is already downloaded locally:")
        print(f"  llmpt-cli download {args.repo_id}")
        sys.exit(1)

    print("✓ Torrent created and registered")
    print("Starting background seeding engine (press Ctrl+C to stop)...")

    # Start unified seeding natively in P2PBatch
    start_seeding(
        repo_id=args.repo_id,
        revision=revision,
        tracker_client=tracker,
        torrent_data=torrent_info.get('torrent_data'),
    )
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n✓ Seeding stopped")


def cmd_status(args):
    """Execute status command."""
    from llmpt.seeder import get_seeding_status
    from llmpt.utils import format_bytes

    status = get_seeding_status()

    if not status:
        print("No active seeding tasks")
        return

    print(f"Active seeding tasks: {len(status)}\n")

    for repo_key, info in status.items():
        print(f"Repo: {repo_key}")
        print(f"  Progress: {info['progress']*100:.1f}%")
        print(f"  State: {info['state']}")
        print(f"  Uploaded: {format_bytes(info['uploaded'])}")
        print(f"  Peers: {info['peers']}")
        print(f"  Upload Rate: {format_bytes(info['upload_rate'])}/s")
        print()


def cmd_stop(args):
    """Execute stop command."""
    from llmpt.seeder import stop_seeding, stop_all_seeding

    if args.repo_key:
        # Expected format: "repo_type:repo_id@revision", e.g. "model:meta-llama/Llama-2-7b@main"
        if '@' not in args.repo_key:
            print("Error: repo_key must be in format repo_type:repo_id@revision (e.g. model:gpt2@main) or repo_id@revision")
            sys.exit(1)
        
        repo_type = 'model'
        repo_id_rev = args.repo_key
        if ':' in args.repo_key:
            repo_type, repo_id_rev = args.repo_key.split(':', 1)
            
        repo_id, revision = repo_id_rev.rsplit('@', 1)
        success = stop_seeding(repo_id, revision, repo_type=repo_type)
        if success:
            print(f"✓ Stopped seeding: {args.repo_key}")
        else:
            print(f"Error: Not seeding: {args.repo_key}")
    else:
        count = stop_all_seeding()
        print(f"✓ Stopped all seeding ({count} tasks)")


def cmd_daemon(args):
    """Execute daemon command."""
    from llmpt.daemon import start_daemon, stop_daemon, is_daemon_running, LOG_FILE
    from llmpt.ipc import query_daemon
    from llmpt.utils import format_bytes

    import os
    tracker_url = args.tracker or os.getenv('HF_P2P_TRACKER') or 'http://localhost:8080'

    if args.daemon_action == 'start':
        existing = is_daemon_running()
        if existing:
            print(f"Daemon already running (PID: {existing})")
            return

        print("Starting daemon...")
        pid = start_daemon(
            tracker_url=tracker_url,
            foreground=args.foreground,
        )
        if pid:
            print(f"✓ Daemon started (PID: {pid})")
            print(f"  Tracker: {tracker_url}")
            print(f"  Logs: {LOG_FILE}")
        else:
            print("Error: Failed to start daemon")
            sys.exit(1)

    elif args.daemon_action == 'stop':
        if stop_daemon():
            print("✓ Daemon stopped")
        else:
            print("Daemon is not running")

    elif args.daemon_action == 'restart':
        stop_daemon()
        import time
        time.sleep(0.5)
        pid = start_daemon(tracker_url=tracker_url)
        if pid:
            print(f"✓ Daemon restarted (PID: {pid})")
        else:
            print("Error: Failed to restart daemon")
            sys.exit(1)

    elif args.daemon_action == 'status':
        pid = is_daemon_running()
        if not pid:
            print("Daemon is not running")
            print("  Start with: llmpt-cli daemon start")
            return

        # Query detailed status via IPC
        response = query_daemon("status")
        if not response:
            print(f"Daemon running (PID: {pid}) but not responding to IPC")
            return

        print(f"llmptd (PID: {pid}) — running")
        
        listen_port = response.get('port')
        if listen_port:
            print(f"  Port: {listen_port}")
            
        sessions = response.get('sessions', {})
        if not sessions:
            print("  No active seeding tasks")
            return

        print(f"\nSeeding {len(sessions)} model(s):\n")
        for repo_key, info in sessions.items():
            state = info.get('state', '?')
            uploaded = info.get('uploaded', 0)
            peers = info.get('peers', 0)
            rate = info.get('upload_rate', 0)
            progress = info.get('progress', 0)
            print(
                f"  {repo_key:<45} "
                f"↑ {format_bytes(uploaded):>8}  │ "
                f"{peers} peers │ "
                f"{format_bytes(rate)}/s"
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

    _write_pid(os.getpid())

    try:
        _daemon_main(args.tracker, port=args.port)
    except Exception as e:
        logger = logging.getLogger("llmpt.daemon")
        logger.error(f"Daemon crashed: {e}", exc_info=True)
    finally:
        _remove_pid()


if __name__ == '__main__':
    main()
