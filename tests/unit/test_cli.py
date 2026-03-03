"""
Tests for the CLI module (llmpt.cli).

Tests argument parsing and command dispatching via `main()`, plus
individual command functions with all external dependencies mocked.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock, call
from io import StringIO

from llmpt.cli import main, cmd_download, cmd_seed, cmd_status, cmd_stop


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
                                'download', 'gpt2', '--local-dir', '/tmp/out', '--no-seed']):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.tracker == 'http://t.com'
        assert args.local_dir == '/tmp/out'
        assert args.no_seed is True

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

    @patch('llmpt.cli.cmd_status')
    def test_status_command(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', 'status']):
            main()
        mock_cmd.assert_called_once()

    @patch('llmpt.cli.cmd_stop')
    def test_stop_command_with_key(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', 'stop', 'gpt2@main']):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.repo_key == 'gpt2@main'

    @patch('llmpt.cli.cmd_stop')
    def test_stop_command_no_key(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', 'stop']):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.repo_key is None

    @patch('llmpt.cli.cmd_download')
    def test_verbose_flag(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', '-v', 'download', 'gpt2']):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.verbose is True


# ─── cmd_download ─────────────────────────────────────────────────────────────

class TestCmdDownload:

    @patch('llmpt.cli.snapshot_download', create=True)
    @patch('llmpt.cli.enable_p2p', create=True)
    def test_basic_download(self, mock_enable, mock_download):
        """Should enable P2P with tracker URL and call snapshot_download."""
        mock_download.return_value = "/path/to/download"

        args = MagicMock()
        args.tracker = "http://tracker.example.com"
        args.no_seed = False
        args.repo_id = "gpt2"
        args.local_dir = None

        with patch('huggingface_hub.snapshot_download', mock_download):
            with patch('llmpt.enable_p2p', mock_enable):
                cmd_download(args)

        mock_enable.assert_called_once_with(
            tracker_url="http://tracker.example.com",
            auto_seed=True,
        )
        mock_download.assert_called_once_with("gpt2", local_dir=None)

    @patch('llmpt.cli.snapshot_download', create=True)
    @patch('llmpt.cli.enable_p2p', create=True)
    def test_download_no_seed(self, mock_enable, mock_download):
        mock_download.return_value = "/path"
        args = MagicMock()
        args.tracker = None
        args.no_seed = True
        args.repo_id = "gpt2"
        args.local_dir = "/tmp/out"

        with patch('huggingface_hub.snapshot_download', mock_download):
            with patch('llmpt.enable_p2p', mock_enable):
                cmd_download(args)

        mock_enable.assert_called_once_with(tracker_url=None, auto_seed=False)


# ─── cmd_seed ─────────────────────────────────────────────────────────────────

class TestCmdSeed:

    @patch('llmpt.cli.start_seeding', create=True)
    @patch('llmpt.cli.create_and_register_torrent', create=True)
    @patch('llmpt.cli.TrackerClient', create=True)
    def test_seed_creation_failure(self, MockTracker, mock_create, mock_start):
        """If create_and_register_torrent fails, should exit(1)."""
        mock_create.return_value = False

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
    def test_seed_success_and_ctrl_c(self, MockTracker, mock_create, mock_start, mock_sleep):
        """Successful seed should loop until KeyboardInterrupt."""
        mock_create.return_value = True

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


# ─── cmd_status ───────────────────────────────────────────────────────────────

class TestCmdStatus:

    def test_no_active_tasks(self, capsys):
        args = MagicMock()

        with patch('llmpt.seeder.get_seeding_status', return_value={}):
            cmd_status(args)

        captured = capsys.readouterr()
        assert "No active seeding tasks" in captured.out

    def test_with_active_tasks(self, capsys):
        args = MagicMock()

        status = {
            "gpt2@main": {
                'repo_id': 'gpt2',
                'revision': 'main',
                'progress': 1.0,
                'state': '5',
                'uploaded': 1024 * 1024,
                'peers': 2,
                'upload_rate': 512 * 1024,
            }
        }

        with patch('llmpt.seeder.get_seeding_status', return_value=status):
            cmd_status(args)

        captured = capsys.readouterr()
        assert "Active seeding tasks: 1" in captured.out
        assert "gpt2@main" in captured.out
        assert "100.0%" in captured.out
        assert "Peers: 2" in captured.out


# ─── cmd_stop ─────────────────────────────────────────────────────────────────

class TestCmdStop:

    def test_stop_specific_success(self, capsys):
        args = MagicMock()
        args.repo_key = "gpt2@main"

        with patch('llmpt.seeder.stop_seeding', return_value=True) as mock_stop:
            cmd_stop(args)

        mock_stop.assert_called_once_with("gpt2", "main")
        captured = capsys.readouterr()
        assert "Stopped seeding" in captured.out

    def test_stop_specific_not_found(self, capsys):
        args = MagicMock()
        args.repo_key = "gpt2@main"

        with patch('llmpt.seeder.stop_seeding', return_value=False):
            cmd_stop(args)

        captured = capsys.readouterr()
        assert "Not seeding" in captured.out

    def test_stop_invalid_format(self):
        """repo_key without '@' should exit(1)."""
        args = MagicMock()
        args.repo_key = "gpt2_no_at_sign"

        with pytest.raises(SystemExit) as exc_info:
            cmd_stop(args)
        assert exc_info.value.code == 1

    def test_stop_all(self, capsys):
        args = MagicMock()
        args.repo_key = None

        with patch('llmpt.seeder.stop_all_seeding', return_value=3) as mock_stop:
            cmd_stop(args)

        mock_stop.assert_called_once()
        captured = capsys.readouterr()
        assert "3 tasks" in captured.out

    def test_stop_repo_with_at_in_name(self, capsys):
        """rsplit('@', 1) should handle repo IDs that contain '@'."""
        args = MagicMock()
        args.repo_key = "user@org/model@v2"

        with patch('llmpt.seeder.stop_seeding', return_value=True) as mock_stop:
            cmd_stop(args)

        # Should split on the LAST '@': repo_id="user@org/model", revision="v2"
        mock_stop.assert_called_once_with("user@org/model", "v2")
