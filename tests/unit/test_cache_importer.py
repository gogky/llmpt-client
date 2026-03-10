"""Tests for automatic cache import verification."""

from types import SimpleNamespace

from llmpt.cache_scanner import SeedableSource
from llmpt.completed_registry import load_completed_sources


def test_import_verified_hub_candidate(monkeypatch, tmp_path):
    import llmpt.cache_importer as importer

    state_file = tmp_path / "cache_import_state.json"
    completed_file = tmp_path / "completed_sources.json"
    cache_root = tmp_path / "hub"
    commit = "a" * 40
    source = SeedableSource(
        repo_type="model",
        repo_id="org/model",
        revision=commit,
        storage_kind="hub_cache",
        storage_root=str(cache_root),
        cache_dir=str(cache_root),
    )

    file_a = cache_root / "models--org--model" / "snapshots" / commit / "config.json"
    file_b = cache_root / "models--org--model" / "snapshots" / commit / "model.bin"
    file_a.parent.mkdir(parents=True, exist_ok=True)
    file_a.write_text("{}")
    file_b.write_bytes(b"123")

    monkeypatch.setattr("llmpt.cache_importer.IMPORT_STATE_FILE", str(state_file))
    monkeypatch.setattr("llmpt.completed_registry.COMPLETED_SOURCES_FILE", str(completed_file))
    monkeypatch.setattr("llmpt.cache_importer._collect_hub_candidates", lambda: [source])
    monkeypatch.setattr("llmpt.cache_importer._collect_local_dir_candidates", lambda: [])
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: [
            SimpleNamespace(filename="config.json"),
            SimpleNamespace(filename="model.bin"),
        ],
    )
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir=None, revision=None, repo_type=None: {
            "config.json": str(file_a),
            "model.bin": str(file_b),
        }[filename],
    )

    summary = importer.import_verified_cache_sources()

    assert summary["imported"] == 1
    entries = load_completed_sources()
    assert len(entries) == 1
    assert entries[0]["manifest"] == ["config.json", "model.bin"]


def test_blocked_candidate_uses_retry_backoff(monkeypatch, tmp_path):
    import llmpt.cache_importer as importer

    state_file = tmp_path / "cache_import_state.json"
    cache_root = tmp_path / "hub"
    source = SeedableSource(
        repo_type="model",
        repo_id="org/gated",
        revision="b" * 40,
        storage_kind="hub_cache",
        storage_root=str(cache_root),
        cache_dir=str(cache_root),
    )

    calls = []

    monkeypatch.setattr("llmpt.cache_importer.IMPORT_STATE_FILE", str(state_file))
    monkeypatch.setattr("llmpt.cache_importer._collect_hub_candidates", lambda: [source])
    monkeypatch.setattr("llmpt.cache_importer._collect_local_dir_candidates", lambda: [])

    def fake_verify(candidate):
        calls.append(candidate.repo_id)
        return "blocked", None, "GatedRepoError"

    monkeypatch.setattr("llmpt.cache_importer._verify_hub_candidate", fake_verify)

    first = importer.import_verified_cache_sources()
    second = importer.import_verified_cache_sources()

    assert first["blocked"] == 1
    assert second["skipped_backoff"] == 1
    assert calls == ["org/gated"]


def test_import_local_dir_candidate(monkeypatch, tmp_path):
    import llmpt.cache_importer as importer

    state_file = tmp_path / "cache_import_state.json"
    completed_file = tmp_path / "completed_sources.json"
    local_dir = tmp_path / "local-model"
    commit = "c" * 40

    weights = local_dir / "weights.bin"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"123")
    metadata = local_dir / ".cache" / "huggingface" / "download" / "weights.bin.metadata"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(f"{commit}\netag\n123\n")

    source = SeedableSource(
        repo_type="model",
        repo_id="org/local",
        revision=commit,
        storage_kind="local_dir",
        storage_root=str(local_dir),
        local_dir=str(local_dir),
    )

    monkeypatch.setattr("llmpt.cache_importer.IMPORT_STATE_FILE", str(state_file))
    monkeypatch.setattr("llmpt.completed_registry.COMPLETED_SOURCES_FILE", str(completed_file))
    monkeypatch.setattr("llmpt.cache_importer._collect_hub_candidates", lambda: [])
    monkeypatch.setattr("llmpt.cache_importer._collect_local_dir_candidates", lambda: [source])

    summary = importer.import_verified_cache_sources()

    assert summary["imported"] == 1
    entries = load_completed_sources()
    assert len(entries) == 1
    assert entries[0]["local_dir"] == str(local_dir.resolve())
    assert entries[0]["manifest"] == ["weights.bin"]


def test_quiet_tqdm_is_iterable():
    from llmpt.cache_importer import _QuietTqdm

    items = list(_QuietTqdm(iter([1, 2, 3])))
    assert items == [1, 2, 3]


def test_quiet_tqdm_get_lock_tolerates_missing_class_attr(monkeypatch):
    from llmpt.cache_importer import _QuietTqdm

    monkeypatch.delattr(_QuietTqdm, "_lock", raising=False)

    assert _QuietTqdm.get_lock() is None


def test_clear_import_state(monkeypatch, tmp_path):
    import llmpt.cache_importer as importer

    state_file = tmp_path / "cache_import_state.json"
    monkeypatch.setattr("llmpt.cache_importer.IMPORT_STATE_FILE", str(state_file))
    state_file.write_text('{"x": {"status": "error"}}')

    importer.clear_import_state()

    assert importer.load_import_state() == {}
