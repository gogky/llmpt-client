"""
CLI tool for llmpt.
"""

import sys
import argparse
import logging
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
        '--no-seed',
        action='store_true',
        help='Do not seed after download'
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

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[%(name)s] %(levelname)s: %(message)s'
    )

    # Execute command
    if args.command == 'download':
        cmd_download(args)
    elif args.command == 'seed':
        cmd_seed(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'stop':
        cmd_stop(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_download(args):
    """Execute download command."""
    from llmpt import enable_p2p
    from huggingface_hub import snapshot_download

    # Enable P2P
    enable_p2p(
        tracker_url=args.tracker,
        auto_seed=not args.no_seed
    )

    print(f"Downloading {args.repo_id}...")

    # Download
    path = snapshot_download(
        args.repo_id,
        local_dir=args.local_dir
    )

    print(f"✓ Downloaded to: {path}")


def cmd_seed(args):
    """Execute seed command."""
    from llmpt.tracker_client import TrackerClient
    from llmpt.torrent_creator import create_and_register_torrent
    from llmpt.seeder import start_seeding

    tracker = TrackerClient(args.tracker or 'http://localhost:8080')

    print(f"Resolving caching structure and creating torrent for {args.repo_id}@{args.revision}...")

    # Create and register torrent
    success = create_and_register_torrent(
        repo_id=args.repo_id,
        revision=args.revision,
        repo_type=args.repo_type,
        name=args.name,
        tracker_client=tracker,
    )

    if not success:
        print("Error: Failed to create or register torrent")
        sys.exit(1)

    print("✓ Torrent created and registered")
    print("Starting background seeding engine (press Ctrl+C to stop)...")

    # Start unified seeding natively in P2PBatch
    start_seeding(
        repo_id=args.repo_id,
        revision=args.revision,
        tracker_client=tracker
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
        # Expected format: "repo_id@revision", e.g. "meta-llama/Llama-2-7b@main"
        if '@' not in args.repo_key:
            print("Error: repo_key must be in format repo_id@revision (e.g. gpt2@main)")
            sys.exit(1)
        repo_id, revision = args.repo_key.rsplit('@', 1)
        success = stop_seeding(repo_id, revision)
        if success:
            print(f"✓ Stopped seeding: {args.repo_key}")
        else:
            print(f"Error: Not seeding: {args.repo_key}")
    else:
        count = stop_all_seeding()
        print(f"✓ Stopped all seeding ({count} tasks)")


if __name__ == '__main__':
    main()
