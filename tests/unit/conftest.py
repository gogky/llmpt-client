"""
Shared pytest fixtures for llmpt unit tests.

Central repository for commonly-used mocks so each test file does
not need to redefine them.  Using these fixtures is optional — existing
tests that already have their own local helpers continue to work.
"""

import threading
from collections import deque

import pytest
from unittest.mock import patch, MagicMock


# ─── libtorrent mocks ────────────────────────────────────────────────────────

def make_mock_lt(*, include_alerts: bool = False):
    """Build a mock libtorrent module with commonly-used attributes.

    Args:
        include_alerts: If True, add save_resume_data_alert /
                        save_resume_data_failed_alert types used by
                        monitor and alert-race tests.
    """
    mock_lt = MagicMock()
    mock_lt.torrent_flags.paused = 0
    mock_lt.save_resume_flags_t.flush_disk_cache = 0

    # Constants for status states
    mock_lt.torrent_status.checking_files = 1
    mock_lt.torrent_status.downloading_metadata = 2
    mock_lt.torrent_status.downloading = 3
    mock_lt.torrent_status.finished = 4
    mock_lt.torrent_status.seeding = 5
    mock_lt.torrent_status.allocating = 6
    mock_lt.torrent_status.checking_resume_data = 7

    mock_lt.alert.category_t.error_notification = 1
    mock_lt.torrent_error_alert = MagicMock()

    if include_alerts:
        mock_lt.save_resume_data_alert = type('save_resume_data_alert', (), {})
        mock_lt.save_resume_data_failed_alert = type(
            'save_resume_data_failed_alert', (), {}
        )
        mock_lt.bencode = MagicMock(return_value=b'\x00')

    return mock_lt


@pytest.fixture
def mock_lt_basic():
    """Provide a basic mock libtorrent and patch it into session_context."""
    m = make_mock_lt()
    with patch('llmpt.session_context.lt', m), \
         patch('llmpt.session_context.LIBTORRENT_AVAILABLE', True):
        yield m


@pytest.fixture
def mock_lt_all_modules():
    """Patch libtorrent into utils, p2p_batch, session_context, and torrent_init.

    Use this when the test exercises code paths that cross module
    boundaries (e.g. P2PBatchManager → SessionContext → torrent_init).
    """
    m = make_mock_lt(include_alerts=True)
    with patch('llmpt.utils.lt', m), \
         patch('llmpt.utils.LIBTORRENT_AVAILABLE', True), \
         patch('llmpt.p2p_batch.lt', m), \
         patch('llmpt.p2p_batch.LIBTORRENT_AVAILABLE', True), \
         patch('llmpt.session_context.lt', m), \
         patch('llmpt.session_context.LIBTORRENT_AVAILABLE', True), \
         patch('llmpt.torrent_init.lt', m):
        yield m


# ─── P2PBatchManager singleton reset ─────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_batch_manager_singleton():
    """Reset P2PBatchManager singleton between every test.

    This is autouse so tests are always isolated, even if they don't
    explicitly request it.
    """
    from llmpt.p2p_batch import P2PBatchManager
    P2PBatchManager._instance = None
    yield
    P2PBatchManager._instance = None


# ─── Mock SessionContext (for monitor / alert tests) ─────────────────────────

def make_mock_ctx(repo_id: str = "test/repo", **overrides) -> MagicMock:
    """Build a minimal mock SessionContext with real locks and deque.

    The returned object uses MagicMock for everything except the
    threading primitives, which must be real to avoid deadlocks
    in tests.
    """
    ctx = MagicMock()
    ctx.repo_id = repo_id
    ctx.is_valid = True
    ctx.lock = threading.Lock()
    ctx.alert_lock = threading.Lock()
    ctx.pending_alerts = deque()
    ctx.file_events = {}
    ctx.file_destinations = {}
    ctx.torrent_info_obj = None
    ctx.fastresume_path = f"/tmp/{repo_id.replace('/', '_')}.fastresume"
    ctx.__dict__.update(overrides)
    return ctx


@pytest.fixture
def mock_ctx():
    """Fixture wrapper for ``make_mock_ctx()`` with default arguments."""
    return make_mock_ctx()
