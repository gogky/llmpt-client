"""
Tests for the CLI module (llmpt.cli).

Tests argument parsing and command dispatching via `main()`, plus
individual command functions with all external dependencies mocked.
"""

import logging

import pytest
from unittest.mock import patch, MagicMock

from llmpt.cli import main, cmd_download, cmd_scan, cmd_status, cmd_unseed

# ─── main() argument parsing & dispatch ───────────────────────────────────────

class TestMainDispatch:

    def test_no_command_exits_with_error(self):
        """No subcommand → print help and exit(1)."""
        with patch('sys.argv', ['llmpt-cli']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_help_hides_internal_daemon_command(self, capsys):
        with patch('sys.argv', ['llmpt-cli', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert '_internal_daemon_start' not in output

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
        assert args.debug is False

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

    def test_seed_command_is_rejected(self):
        with patch('sys.argv', ['llmpt-cli', 'seed', '--repo-id', 'gpt2', '--revision', 'main']):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 2

    @patch('llmpt.cli.cmd_unseed')
    def test_unseed_command(self, mock_cmd):
        with patch('sys.argv', [
            'llmpt-cli', 'unseed',
            'gpt2',
            '--forget',
        ]):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.repo_id == 'gpt2'
        assert args.revision is None
        assert args.repo_type is None
        assert args.forget is True

    @patch('llmpt.cli.cmd_unseed')
    def test_unseed_target_command(self, mock_cmd):
        with patch('sys.argv', [
            'llmpt-cli', 'unseed',
            'model/org/repo@7362d24',
        ]):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.repo_id == 'model/org/repo@7362d24'
        assert args.revision is None
        assert args.repo_type is None
        assert args.forget is False



    @patch('llmpt.cli.cmd_download')
    def test_verbose_flag(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', '-v', 'download', 'gpt2']):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.verbose is True
        assert args.debug is False

    @patch('llmpt.cli.cmd_download')
    def test_debug_flag(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', '--debug', 'download', 'gpt2']):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.verbose is False
        assert args.debug is True

    @patch('llmpt.cli.cmd_download')
    @patch('llmpt.cli.logging.basicConfig')
    def test_default_logging_is_warning(self, mock_basic_config, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', 'download', 'gpt2']):
            main()
        mock_basic_config.assert_called_once_with(
            level=logging.WARNING,
            format='[%(name)s] %(levelname)s: %(message)s',
        )

    @patch('llmpt.cli.cmd_download')
    @patch('llmpt.cli.logging.basicConfig')
    def test_verbose_logging_is_info(self, mock_basic_config, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', '-v', 'download', 'gpt2']):
            main()
        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format='[%(name)s] %(levelname)s: %(message)s',
        )

    @patch('llmpt.cli.cmd_download')
    @patch('llmpt.cli.logging.basicConfig')
    def test_debug_logging_is_debug(self, mock_basic_config, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', '--debug', 'download', 'gpt2']):
            main()
        mock_basic_config.assert_called_once_with(
            level=logging.DEBUG,
            format='[%(name)s] %(levelname)s: %(message)s',
        )

    @patch('llmpt.cli.cmd_start')
    def test_start_with_port(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', 'start', '--port', '7001']):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.port == 7001

    @patch('llmpt.cli.cmd_scan')
    def test_scan_command(self, mock_cmd):
        with patch('sys.argv', ['llmpt-cli', 'scan']):
            main()
        mock_cmd.assert_called_once()
        args = mock_cmd.call_args[0][0]
        assert args.command == 'scan'


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
        args.debug = False

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

    @patch('llmpt.cli.snapshot_download', create=True)
    @patch('llmpt.cli.enable_p2p', create=True)
    def test_debug_download_enables_verbose_summary(self, mock_enable, mock_download):
        """Debug mode should preserve detailed P2P summary output."""
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
        args.verbose = False
        args.debug = True

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


class TestCmdUnseed:

    @patch('llmpt.utils.resolve_commit_hash', side_effect=lambda repo, rev, repo_type='model': rev)
    @patch('llmpt.ipc.query_daemon', return_value={
        'status': 'ok',
        'removed_count': 1,
        'forgotten': {'hub_cache_roots_removed': 1, 'local_dir_sources_removed': 0},
    })
    @patch('llmpt.daemon.is_daemon_running', return_value=12345)
    def test_unseed_success(self, mock_is_running, mock_query, mock_resolve):
        args = MagicMock()
        args.repo_id = 'gpt2'
        args.revision = None
        args.repo_type = None
        args.forget = True

        cmd_unseed(args)

        mock_query.assert_called_once_with(
            'unseed',
            repo_id='gpt2',
            revision=None,
            repo_type=None,
            forget=True,
        )

    @patch('llmpt.utils.resolve_commit_hash')
    @patch('llmpt.ipc.query_daemon', return_value={
        'status': 'ok',
        'removed_count': 2,
        'removed_sessions': [
            {
                'repo_type': 'model',
                'repo_id': 'org/repo',
                'revision': '7362d24ca596daa0c15c0caad7407413599c78d4',
                'storage_kind': 'hub_cache',
                'storage_root': '/tmp/a',
            },
            {
                'repo_type': 'model',
                'repo_id': 'org/repo',
                'revision': '7362d24ca596daa0c15c0caad7407413599c78d4',
                'storage_kind': 'hub_cache',
                'storage_root': '/tmp/b',
            },
        ],
        'forgotten': {'hub_cache_roots_removed': 0, 'local_dir_sources_removed': 0},
    })
    @patch('llmpt.daemon.is_daemon_running', return_value=12345)
    def test_unseed_target_success(self, mock_is_running, mock_query, mock_resolve, capsys):
        args = MagicMock()
        args.repo_id = 'model/org/repo@7362d24'
        args.revision = None
        args.repo_type = None
        args.forget = False

        cmd_unseed(args)

        mock_resolve.assert_not_called()
        mock_query.assert_called_once_with(
            'unseed',
            repo_id='org/repo',
            revision='7362d24',
            repo_type='model',
            forget=False,
        )
        output = capsys.readouterr().out
        assert 'model/org/repo@7362d24' in output
        assert '(2 sources)' in output

    @patch('llmpt.utils.resolve_commit_hash', side_effect=RuntimeError('network down'))
    @patch('llmpt.ipc.query_daemon', return_value={
        'status': 'ok',
        'removed_count': 1,
        'forgotten': {'hub_cache_roots_removed': 0, 'local_dir_sources_removed': 0},
    })
    @patch('llmpt.daemon.is_daemon_running', return_value=12345)
    def test_unseed_falls_back_to_raw_revision_when_resolution_fails(self, mock_is_running, mock_query, mock_resolve):
        args = MagicMock()
        args.repo_id = 'gpt2'
        args.revision = 'main'
        args.repo_type = None
        args.forget = False

        cmd_unseed(args)

        mock_query.assert_called_once_with(
            'unseed',
            repo_id='gpt2',
            revision='main',
            repo_type=None,
            forget=False,
        )

    @patch('llmpt.utils.resolve_commit_hash', side_effect=lambda repo, rev, repo_type='model': rev)
    @patch('llmpt.ipc.query_daemon', return_value={'status': 'error', 'message': 'no matching active seeding session'})
    @patch('llmpt.daemon.is_daemon_running', return_value=12345)
    def test_unseed_error_exits(self, mock_is_running, mock_query, mock_resolve):
        args = MagicMock()
        args.repo_id = 'gpt2'
        args.revision = 'main'
        args.repo_type = None
        args.forget = False

        with pytest.raises(SystemExit) as exc_info:
            cmd_unseed(args)

        assert exc_info.value.code == 1


class TestCmdStatus:

    @patch('llmpt.ipc.query_daemon', return_value={
        'status': 'ok',
        'tracker_url': 'http://localhost:8080',
        'port': 6881,
        'sessions': {
            'a': {
                'repo_type': 'model',
                'repo_id': 'hf-internal-testing/tiny-random-GPTJForCausalLM',
                'revision': '7362d24ca596daa0c15c0caad7407413599c78d4',
                'uploaded': 3072,
                'peers': 0,
                'upload_rate': 12,
                'mapped_files': 10,
                'total_files': 10,
                'full_mapping': True,
                'source_count': 2,
                'source_status': 'verified',
                'torrent_status': 'local_only',
                'session_status': 'degraded',
            },
            'b': {
                'repo_type': 'model',
                'repo_id': 'hf-internal-testing/tiny-random-gpt2',
                'revision': '71034c5d8bde858ff824298bdedc65515b97d2b9',
                'uploaded': 4096,
                'peers': 0,
                'upload_rate': 0,
                'mapped_files': 10,
                'total_files': 10,
                'full_mapping': True,
                'source_status': 'verified',
                'torrent_status': 'local_only',
                'session_status': 'degraded',
            },
        },
    })
    @patch('llmpt.daemon.is_daemon_running', return_value=12345)
    def test_status_displays_reported_source_counts(self, mock_is_running, mock_query, capsys):
        args = MagicMock()

        cmd_status(args)

        output = capsys.readouterr().out
        assert 'Active seeding: 2' in output
        assert 'model/hf-internal-testing/tiny-random-GPTJForCausalLM@7362d24' in output
        assert 'model/hf-internal-testing/tiny-random-gpt2@71034c5' in output
        assert 'local-only  2 sources' in output

    @patch('llmpt.ipc.query_daemon', return_value={
        'status': 'ok',
        'tracker_url': 'http://localhost:8080',
        'port': 6881,
        'sessions': {
            'a': {
                'repo_type': 'model',
                'repo_id': 'hf-internal-testing/tiny-random-GPTJForCausalLM',
                'revision': '7362d24ca596daa0c15c0caad7407413599c78d4',
                'uploaded': 1024,
                'peers': 0,
                'upload_rate': 4,
                'mapped_files': 10,
                'total_files': 10,
                'full_mapping': True,
                'source_count': 2,
                'source_status': 'verified',
                'torrent_status': 'local_only',
                'session_status': 'degraded',
            },
        },
    })
    @patch('llmpt.daemon.is_daemon_running', return_value=12345)
    def test_status_uses_reported_source_count(self, mock_is_running, mock_query, capsys):
        args = MagicMock()

        cmd_status(args)

        output = capsys.readouterr().out
        assert 'Active seeding: 1' in output
        assert 'local-only  2 sources' in output


class TestCmdScan:

    @patch('llmpt.ipc.query_daemon', return_value={'status': 'ok', 'message': 'scan triggered'})
    @patch('llmpt.daemon.is_daemon_running', return_value=12345)
    def test_scan_success(self, mock_is_running, mock_query):
        args = MagicMock()

        cmd_scan(args)

        mock_query.assert_called_once_with('scan')
