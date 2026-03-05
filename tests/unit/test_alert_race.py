"""
Tests for the global alert dispatch mechanism.

Bug (before fix): Each SessionContext's monitor thread called
`lt_session.pop_alerts()`, which pops ALL alerts from the global session —
including alerts belonging to OTHER torrent handles.  Alerts not matching the
caller's handle were silently discarded, causing fastresume data loss.

Fix: `P2PBatchManager.dispatch_alerts()` centrally pops alerts and routes them
to each SessionContext's `pending_alerts` inbox.  `_process_alerts()` now reads
from the inbox instead of the global session.
"""

import threading
from collections import deque
from unittest.mock import patch, MagicMock, mock_open

from llmpt.monitor import _process_alerts
from tests.unit.conftest import make_mock_lt, make_mock_ctx


class TestDispatchAlerts:
    """Test the centralized alert dispatch mechanism in P2PBatchManager."""

    def test_alerts_routed_to_correct_sessions(self):
        """
        dispatch_alerts() should pop all alerts from lt_session once and deposit
        each alert into the correct SessionContext's pending_alerts inbox.
        """
        from llmpt.p2p_batch import P2PBatchManager

        handle_a = MagicMock(name="handle_A")
        handle_b = MagicMock(name="handle_B")

        ctx_a = make_mock_ctx(repo_id="owner/repo_a")
        ctx_a.handle = handle_a

        ctx_b = make_mock_ctx(repo_id="owner/repo_b")
        ctx_b.handle = handle_b

        alert_for_a = MagicMock()
        alert_for_a.handle = handle_a

        alert_for_b = MagicMock()
        alert_for_b.handle = handle_b

        # Setup: create a manager with a mock lt_session
        manager = P2PBatchManager.__new__(P2PBatchManager)
        manager._initialized = True
        manager.lt_session = MagicMock()
        manager.lt_session.pop_alerts.return_value = [alert_for_a, alert_for_b]
        manager.sessions = {
            ("owner/repo_a", "main"): ctx_a,
            ("owner/repo_b", "main"): ctx_b,
        }

        manager.dispatch_alerts()

        # Each context should have received exactly its own alert
        assert len(ctx_a.pending_alerts) == 1
        assert ctx_a.pending_alerts[0] is alert_for_a

        assert len(ctx_b.pending_alerts) == 1
        assert ctx_b.pending_alerts[0] is alert_for_b

    def test_alerts_not_stolen_across_sessions(self):
        """
        Regression test for the original bug: session A must NOT steal session B's
        alerts.  After dispatch, each session processes only its own alerts.
        """
        from llmpt.p2p_batch import P2PBatchManager

        handle_a = MagicMock(name="handle_A")
        handle_b = MagicMock(name="handle_B")

        ctx_a = make_mock_ctx(repo_id="owner/repo_a")
        ctx_a.handle = handle_a

        ctx_b = make_mock_ctx(repo_id="owner/repo_b")
        ctx_b.handle = handle_b

        SaveResumeAlert = type('save_resume_data_alert', (), {})

        alert_for_a = MagicMock()
        alert_for_a.__class__ = SaveResumeAlert
        alert_for_a.handle = handle_a
        alert_for_a.params = {'info-hash': 'aaa'}

        alert_for_b = MagicMock()
        alert_for_b.__class__ = SaveResumeAlert
        alert_for_b.handle = handle_b
        alert_for_b.params = {'info-hash': 'bbb'}

        # Setup manager
        manager = P2PBatchManager.__new__(P2PBatchManager)
        manager._initialized = True
        manager.lt_session = MagicMock()
        manager.lt_session.pop_alerts.return_value = [alert_for_a, alert_for_b]
        manager.sessions = {
            ("owner/repo_a", "main"): ctx_a,
            ("owner/repo_b", "main"): ctx_b,
        }

        # Dispatch (centralized pop + route)
        manager.dispatch_alerts()

        # Now process_alerts for each session should only handle its own alerts
        with patch('llmpt.monitor.lt') as mock_lt:
            mock_lt.save_resume_data_alert = SaveResumeAlert
            mock_lt.save_resume_data_failed_alert = type('FailedAlert', (), {})
            mock_lt.bencode.return_value = b'encoded_resume_data'

            with patch('builtins.open', mock_open()) as mock_file:
                _process_alerts(ctx_a)
                mock_file.assert_called_once_with(ctx_a.fastresume_path, "wb")

            with patch('builtins.open', mock_open()) as mock_file:
                _process_alerts(ctx_b)
                mock_file.assert_called_once_with(ctx_b.fastresume_path, "wb")

    def test_concurrent_dispatch_safe(self):
        """
        Multiple threads calling dispatch_alerts() concurrently should be safe —
        each alert is delivered exactly once to the correct session.
        """
        from llmpt.p2p_batch import P2PBatchManager

        handle_a = MagicMock(name="handle_A")
        handle_b = MagicMock(name="handle_B")

        ctx_a = make_mock_ctx(repo_id="owner/repo_a")
        ctx_a.handle = handle_a

        ctx_b = make_mock_ctx(repo_id="owner/repo_b")
        ctx_b.handle = handle_b

        SaveResumeAlert = type('save_resume_data_alert', (), {})

        alert_a = MagicMock()
        alert_a.__class__ = SaveResumeAlert
        alert_a.handle = handle_a
        alert_a.params = {'info': 'a'}

        alert_b = MagicMock()
        alert_b.__class__ = SaveResumeAlert
        alert_b.handle = handle_b
        alert_b.params = {'info': 'b'}

        # Simulate: pop_alerts returns both on first call, empty on subsequent
        pop_lock = threading.Lock()
        alerts_queue = [alert_a, alert_b]

        def pop_side_effect():
            with pop_lock:
                result = list(alerts_queue)
                alerts_queue.clear()
                return result

        manager = P2PBatchManager.__new__(P2PBatchManager)
        manager._initialized = True
        manager._lock = threading.Lock()
        manager.lt_session = MagicMock()
        manager.lt_session.pop_alerts.side_effect = pop_side_effect
        manager.sessions = {
            ("owner/repo_a", "main"): ctx_a,
            ("owner/repo_b", "main"): ctx_b,
        }

        # Run dispatch concurrently from multiple threads
        barrier = threading.Barrier(4)

        def dispatch_thread():
            barrier.wait()
            manager.dispatch_alerts()

        threads = [threading.Thread(target=dispatch_thread) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Each session should have exactly 1 alert (no duplication, no loss)
        assert len(ctx_a.pending_alerts) == 1, \
            f"Expected 1 alert for A, got {len(ctx_a.pending_alerts)}"
        assert ctx_a.pending_alerts[0] is alert_a

        assert len(ctx_b.pending_alerts) == 1, \
            f"Expected 1 alert for B, got {len(ctx_b.pending_alerts)}"
        assert ctx_b.pending_alerts[0] is alert_b

    def test_process_alerts_reads_from_inbox(self):
        """_process_alerts() should consume from pending_alerts, not lt_session."""
        ctx = make_mock_ctx()

        SaveResumeAlert = type('save_resume_data_alert', (), {})
        alert = MagicMock()
        alert.__class__ = SaveResumeAlert
        alert.params = {'info-hash': 'xyz'}

        # Pre-populate the inbox (as dispatch_alerts would)
        ctx.pending_alerts.append(alert)

        with patch('llmpt.monitor.lt') as mock_lt:
            mock_lt.save_resume_data_alert = SaveResumeAlert
            mock_lt.save_resume_data_failed_alert = type('FailedAlert', (), {})
            mock_lt.bencode.return_value = b'resume_bytes'

            with patch('builtins.open', mock_open()) as mock_file:
                _process_alerts(ctx)

            mock_file.assert_called_once_with(ctx.fastresume_path, "wb")

        # Inbox should be empty after processing
        assert len(ctx.pending_alerts) == 0

        # lt_session.pop_alerts should NOT have been called
        ctx.lt_session.pop_alerts.assert_not_called()

    def test_unmatched_alerts_dropped_gracefully(self):
        """Alerts for handles not in any session should be silently dropped."""
        from llmpt.p2p_batch import P2PBatchManager

        orphan_handle = MagicMock(name="orphan_handle")
        orphan_alert = MagicMock()
        orphan_alert.handle = orphan_handle

        manager = P2PBatchManager.__new__(P2PBatchManager)
        manager._initialized = True
        manager._lock = threading.Lock()
        manager.lt_session = MagicMock()
        manager.lt_session.pop_alerts.return_value = [orphan_alert]
        manager.sessions = {}

        # Should not raise
        manager.dispatch_alerts()
