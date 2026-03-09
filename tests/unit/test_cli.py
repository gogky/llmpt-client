"""
Tests for the CLI module (llmpt.cli).

Tests argument parsing and command dispatching via `main()`, plus
individual command functions with all external dependencies mocked.
"""

import pytest
from unittest.mock import patch, MagicMock

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
        assert args.timeout is None
        assert args.port is None
        assert args.hf_token is None
        assert args.webseed is None
        assert args.verbose is False

    @patch('llmpt.cli.cmd_download')
    def test_download_with_options(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', '--tracker', 'http://t.com',
                                'download', 'gpt2', '--local-dir', '/tmp/out',
                                '--timeout', '123', '--port', '6881',
                                '--hf-token', 'hf_test', '--no-webseed']):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.tracker == 'http://t.com'
        assert args.local_dir == '/tmp/out'
        assert args.timeout == 123
        assert args.port == 6881
        assert args.hf_token == 'hf_test'
        assert args.webseed is False

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

    @patch('llmpt.cli.cmd_start')
    def test_start_with_port(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', 'start', '--port', '7001']):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.port == 7001


# ─── cmd_download ─────────────────────────────────────────────────────────────

class TestCmdDownload:

    @patch('llmpt.cli.snapshot_download', create=True)
    @patch('llmpt.cli.enable_p2p', create=True)
    def test_basic_download(self, mock_enable, mock_download):
        """Should pass CLI options through to enable_p2p and snapshot_download."""
        mock_download.return_value = "/path/to/download"

        args = MagicMock()
        args.tracker = "http://tracker.example.com"
        args.repo_id = "gpt2"
        args.local_dir = None
        args.repo_type = "model"
        args.timeout = 456
        args.port = 6888
        args.hf_token = "hf_token_123"
        args.webseed = False
        args.verbose = True

        with patch('huggingface_hub.snapshot_download', mock_download), \
             patch('llmpt.enable_p2p', mock_enable):
            cmd_download(args)

        mock_enable.assert_called_once_with(
            tracker_url="http://tracker.example.com",
            timeout=456,
            port=6888,
            hf_token="hf_token_123",
            webseed=False,
            verbose=True,
        )
        mock_download.assert_called_once_with("gpt2", repo_type=args.repo_type, local_dir=None)


class TestCmdStart:

    @patch.dict('os.environ', {'HF_P2P_TRACKER': 'http://env-tracker', 'HF_P2P_PORT': '7010'}, clear=False)
    @patch('llmpt.daemon.is_daemon_running', return_value=None)
    @patch('llmpt.daemon.start_daemon', return_value=12345)
    def test_start_uses_env_defaults(self, mock_start_daemon, mock_is_running):
        from llmpt.cli import cmd_start

        args = MagicMock()
        args.tracker = None
        args.port = None
        args.foreground = False

        cmd_start(args)

        mock_start_daemon.assert_called_once_with(
            tracker_url='http://env-tracker',
            port=7010,
            foreground=False,
        )

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

