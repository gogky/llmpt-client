"""Tests for persistent torrent lifecycle state."""

import threading


def test_mark_local_torrent_and_get_state(monkeypatch, tmp_path):
    from llmpt.torrent_state import get_torrent_state, mark_local_torrent

    state_file = tmp_path / "torrent_state.json"
    monkeypatch.setattr("llmpt.torrent_state.TORRENT_STATE_FILE", str(state_file))

    mark_local_torrent("org/model", "a" * 40, info_hash="abc")
    state = get_torrent_state("org/model", "a" * 40)

    assert state["local_torrent_present"] is True
    assert state["info_hash"] == "abc"
    assert state["tracker_registered"] is False


def test_tracker_registered_is_scoped_to_tracker_url(monkeypatch, tmp_path):
    from llmpt.torrent_state import get_torrent_state, mark_tracker_registration

    state_file = tmp_path / "torrent_state.json"
    monkeypatch.setattr("llmpt.torrent_state.TORRENT_STATE_FILE", str(state_file))

    mark_tracker_registration(
        "org/model",
        "b" * 40,
        tracker_url="http://tracker-a",
        registered=True,
    )

    state_same = get_torrent_state(
        "org/model",
        "b" * 40,
        tracker_url="http://tracker-a",
    )
    state_other = get_torrent_state(
        "org/model",
        "b" * 40,
        tracker_url="http://tracker-b",
    )

    assert state_same["tracker_registered"] is True
    assert state_other["tracker_registered"] is False


def test_failed_registration_persists_error(monkeypatch, tmp_path):
    from llmpt.torrent_state import get_torrent_state, mark_tracker_registration

    state_file = tmp_path / "torrent_state.json"
    monkeypatch.setattr("llmpt.torrent_state.TORRENT_STATE_FILE", str(state_file))

    mark_tracker_registration(
        "org/model",
        "c" * 40,
        tracker_url="http://tracker-a",
        registered=False,
        error="register_failed",
    )
    state = get_torrent_state(
        "org/model",
        "c" * 40,
        tracker_url="http://tracker-a",
    )

    assert state["tracker_registered"] is False
    assert state["last_registration_error"] == "register_failed"


def test_concurrent_updates_preserve_combined_state(monkeypatch, tmp_path):
    from llmpt.torrent_state import get_torrent_state, mark_local_torrent, mark_tracker_registration

    state_file = tmp_path / "torrent_state.json"
    monkeypatch.setattr("llmpt.torrent_state.TORRENT_STATE_FILE", str(state_file))

    barrier = threading.Barrier(2)

    def set_local():
        barrier.wait()
        mark_local_torrent("org/model", "d" * 40, present=True)

    def set_tracker():
        barrier.wait()
        mark_tracker_registration(
            "org/model",
            "d" * 40,
            tracker_url="http://tracker-a",
            registered=True,
        )

    t1 = threading.Thread(target=set_local)
    t2 = threading.Thread(target=set_tracker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    state = get_torrent_state(
        "org/model",
        "d" * 40,
        tracker_url="http://tracker-a",
    )

    assert state["local_torrent_present"] is True
    assert state["tracker_registered"] is True
