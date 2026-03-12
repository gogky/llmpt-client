"""Tests for lightweight unified status summaries."""

from llmpt.status_summary import (
    get_session_status,
    get_source_status,
    get_torrent_status,
    summarize_status,
)


def test_source_status_verified_beats_import_state(monkeypatch, tmp_path):
    from llmpt.completed_registry import register_completed_source

    completed_file = tmp_path / "completed_sources.json"
    import_state_file = tmp_path / "cache_import_state.json"
    cache_root = tmp_path / "hub"
    commit = "a" * 40

    snapshot = cache_root / "models--org--model" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")

    monkeypatch.setattr("llmpt.completed_registry.COMPLETED_SOURCES_FILE", str(completed_file))
    monkeypatch.setattr("llmpt.cache_importer.IMPORT_STATE_FILE", str(import_state_file))
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir=None, revision=None, repo_type=None: str(snapshot / filename),
    )

    register_completed_source(
        "org/model",
        commit,
        cache_dir=str(cache_root),
        manifest=["config.json"],
    )

    import_state_file.write_text(
        '{"model|org/model|' + commit + '|hub_cache|/tmp/x": {"repo_type":"model","repo_id":"org/model","revision":"' + commit + '","status":"blocked"}}'
    )

    assert get_source_status("org/model", commit) == "verified"


def test_source_status_uses_import_state_when_not_verified(monkeypatch, tmp_path):
    import_state_file = tmp_path / "cache_import_state.json"
    monkeypatch.setattr("llmpt.cache_importer.IMPORT_STATE_FILE", str(import_state_file))
    monkeypatch.setattr("llmpt.completed_registry.COMPLETED_SOURCES_FILE", str(tmp_path / "completed_sources.json"))

    commit = "b" * 40
    import_state_file.write_text(
        '{"model|org/model|' + commit + '|hub_cache|/tmp/x": {"repo_type":"model","repo_id":"org/model","revision":"' + commit + '","status":"partial"}}'
    )

    assert get_source_status("org/model", commit) == "partial"


def test_torrent_status_from_local_and_registered_state(monkeypatch, tmp_path):
    from llmpt.torrent_state import mark_local_torrent, mark_tracker_registration

    state_file = tmp_path / "torrent_state.json"
    monkeypatch.setattr("llmpt.torrent_state.TORRENT_STATE_FILE", str(state_file))

    commit = "c" * 40
    assert get_torrent_status("org/model", commit, tracker_url="http://tracker") == "absent"

    mark_local_torrent("org/model", commit)
    assert get_torrent_status("org/model", commit, tracker_url="http://tracker") == "local_only"

    mark_tracker_registration(
        "org/model",
        commit,
        tracker_url="http://tracker",
        registered=True,
    )
    assert get_torrent_status("org/model", commit, tracker_url="http://tracker") == "registered"


def test_session_status_degraded_without_tracker_or_mapping():
    assert get_session_status(active=False, full_mapping=False, tracker_registered=False) == "inactive"
    assert get_session_status(active=True, full_mapping=False, tracker_registered=True) == "degraded"
    assert get_session_status(active=True, full_mapping=True, tracker_registered=False) == "degraded"
    assert get_session_status(active=True, full_mapping=True, tracker_registered=True) == "active"


def test_summarize_status_aggregates_three_layers(monkeypatch, tmp_path):
    from llmpt.completed_registry import register_completed_source
    from llmpt.torrent_state import mark_local_torrent

    completed_file = tmp_path / "completed_sources.json"
    state_file = tmp_path / "torrent_state.json"
    monkeypatch.setattr("llmpt.completed_registry.COMPLETED_SOURCES_FILE", str(completed_file))
    monkeypatch.setattr("llmpt.cache_importer.IMPORT_STATE_FILE", str(tmp_path / "cache_import_state.json"))
    monkeypatch.setattr("llmpt.torrent_state.TORRENT_STATE_FILE", str(state_file))

    cache_root = tmp_path / "hub"
    commit = "d" * 40
    snapshot = cache_root / "models--org--model" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir=None, revision=None, repo_type=None: str(snapshot / filename),
    )

    register_completed_source(
        "org/model",
        commit,
        cache_dir=str(cache_root),
        manifest=["config.json"],
    )
    mark_local_torrent("org/model", commit)

    summary = summarize_status(
        "org/model",
        commit,
        tracker_url="http://tracker",
        active=True,
        full_mapping=True,
    )

    assert summary == {
        "source_status": "verified",
        "torrent_status": "local_only",
        "session_status": "degraded",
    }
