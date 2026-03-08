"""
Tests for the CLI module (llmpt.cli).

Tests argument parsing and command dispatching via `main()`, plus
individual command functions with all external dependencies mocked.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock, call
from io import StringIO

from llmpt.cli import main, cmd_download, cmd_seed

# ─── main() argument parsing & dispatch ───────────────────────────────────────

class TestMainDispatch:

    def test_no_command_exits_with_error(self):
        """No subcommand → print help and exit(1)."""
        with patch('sys.argv', ['llmpt-cli']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch('llmpt.cli.cmd_download')
    def test_download_command(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', 'download', 'gpt2']):
            main()
        mock_cmd.assert_called_once()
        args = mock_cmd.call_args[0][0]
        assert args.repo_id == 'gpt2'
        assert args.command == 'download'

    @patch('llmpt.cli.cmd_download')
    def test_download_with_options(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', '--tracker', 'http://t.com',
                                'download', 'gpt2', '--local-dir', '/tmp/out']):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.tracker == 'http://t.com'
        assert args.local_dir == '/tmp/out'

    @patch('llmpt.cli.cmd_seed')
    def test_seed_command(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', 'seed', '--repo-id', 'gpt2', '--revision', 'main']):
            main()
        mock_cmd.assert_called_once()
        args = mock_cmd.call_args[0][0]
        assert args.repo_id == 'gpt2'
        assert args.revision == 'main'
        assert args.repo_type == 'model'  # default
        assert args.name == 'HF Model'  # default



    @patch('llmpt.cli.cmd_download')
    def test_verbose_flag(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', '-v', 'download', 'gpt2']):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.verbose is True


# ─── cmd_download ─────────────────────────────────────────────────────────────

class TestCmdDownload:

    @patch('time.sleep', side_effect=KeyboardInterrupt)
    @patch('llmpt.cli.snapshot_download', create=True)
    @patch('llmpt.cli.enable_p2p', create=True)
    def test_basic_download(self, mock_enable, mock_download, mock_sleep):
        """Should enable P2P with tracker URL, download, and block for seeding."""
        mock_download.return_value = "/path/to/download"

        args = MagicMock()
        args.tracker = "http://tracker.example.com"
        args.repo_id = "gpt2"
        args.local_dir = None

        with patch('huggingface_hub.snapshot_download', mock_download), \
             patch('llmpt.enable_p2p', mock_enable):
            cmd_download(args)

        mock_enable.assert_called_once_with(
            tracker_url="http://tracker.example.com"
        )
        mock_download.assert_called_once_with("gpt2", repo_type=args.repo_type, local_dir=None)

# ─── cmd_seed ─────────────────────────────────────────────────────────────────

class TestCmdSeed:

    @patch('llmpt.cli.start_seeding', create=True)
    @patch('llmpt.cli.create_and_register_torrent', create=True)
    @patch('llmpt.cli.TrackerClient', create=True)
    @patch('llmpt.utils.resolve_commit_hash', side_effect=lambda repo, rev, repo_type='model': rev)
    def test_seed_creation_failure(self, mock_resolve, MockTracker, mock_create, mock_start):
        """If create_and_register_torrent fails (returns None), should exit(1)."""
        mock_create.return_value = None

        args = MagicMock()
        args.tracker = 'http://tracker.example.com'
        args.repo_id = 'gpt2'
        args.revision = 'main'
        args.repo_type = 'model'
        args.name = 'GPT2'

        with patch('llmpt.torrent_creator.create_and_register_torrent', mock_create), \
             patch('llmpt.tracker_client.TrackerClient', MockTracker), \
             patch('llmpt.seeder.start_seeding', mock_start):
            with pytest.raises(SystemExit) as exc_info:
                cmd_seed(args)
            assert exc_info.value.code == 1

    @patch('time.sleep', side_effect=KeyboardInterrupt)
    @patch('llmpt.cli.start_seeding', create=True)
    @patch('llmpt.cli.create_and_register_torrent', create=True)
    @patch('llmpt.cli.TrackerClient', create=True)
    @patch('llmpt.utils.resolve_commit_hash', side_effect=lambda repo, rev, repo_type='model': rev)
    def test_seed_success_and_ctrl_c(self, mock_resolve, MockTracker, mock_create, mock_start, mock_sleep):
        """Successful seed should pass torrent_data through and loop until KeyboardInterrupt."""
        mock_create.return_value = {
            'info_hash': 'abc123',
            'torrent_data': b'fake_torrent_bytes',
        }

        args = MagicMock()
        args.tracker = 'http://tracker.example.com'
        args.repo_id = 'gpt2'
        args.revision = 'main'
        args.repo_type = 'model'
        args.name = 'GPT2'

        with patch('llmpt.torrent_creator.create_and_register_torrent', mock_create), \
             patch('llmpt.tracker_client.TrackerClient', MockTracker), \
             patch('llmpt.seeder.start_seeding', mock_start):
            cmd_seed(args)  # Should not raise, KeyboardInterrupt is caught

        mock_start.assert_called_once()
        # Verify torrent_data is passed through to start_seeding
        assert mock_start.call_args.kwargs.get('torrent_data') == b'fake_torrent_bytes'



