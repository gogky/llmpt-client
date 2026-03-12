from types import SimpleNamespace

from llmpt.completed_registry import (
    get_completed_manifest,
    has_completed_source,
    load_completed_sources,
    register_completed_source,
    save_completed_sources,
)


def test_register_completed_source_local_dir_uses_upstream_manifest(monkeypatch, tmp_path):
    completed_file = tmp_path / "completed_sources.json"
    local_dir = tmp_path / "local-model"
    commit = "a" * 40

    model_file = local_dir / "weights.bin"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(b"123")
    metadata_path = (
        local_dir
        / ".cache"
        / "huggingface"
        / "download"
        / "weights.bin.metadata"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(f"{commit}\netag\n123\n")

    monkeypatch.setattr("llmpt.completed_registry.COMPLETED_SOURCES_FILE", str(completed_file))
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: [SimpleNamespace(filename="weights.bin")],
    )

    assert register_completed_source(
        repo_id="org/local",
        revision=commit,
        local_dir=str(local_dir),
    )

    entries = load_completed_sources()
    assert len(entries) == 1
    assert entries[0]["manifest"] == ["weights.bin"]


def test_stale_local_dir_subset_is_not_treated_as_completed(monkeypatch, tmp_path):
    completed_file = tmp_path / "completed_sources.json"
    local_dir = tmp_path / "local-model"
    commit = "b" * 40

    for filename in ("config.json", "weights.bin"):
        file_path = local_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("{}")
        metadata_path = (
            local_dir
            / ".cache"
            / "huggingface"
            / "download"
            / f"{filename}.metadata"
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(f"{commit}\netag\n123\n")

    monkeypatch.setattr("llmpt.completed_registry.COMPLETED_SOURCES_FILE", str(completed_file))
    save_completed_sources(
        [
            {
                "repo_type": "model",
                "repo_id": "org/local",
                "revision": commit,
                "storage_kind": "local_dir",
                "storage_root": str(local_dir),
                "local_dir": str(local_dir),
                "manifest": ["config.json"],
            }
        ]
    )

    assert has_completed_source("org/local", commit, local_dir=str(local_dir)) is False
    assert get_completed_manifest("org/local", commit, local_dir=str(local_dir)) is None


def test_stale_hub_cache_subset_is_not_treated_as_completed(monkeypatch, tmp_path):
    completed_file = tmp_path / "completed_sources.json"
    cache_root = tmp_path / "hub"
    commit = "c" * 40
    snapshot = cache_root / "models--org--model" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.bin").write_bytes(b"123")

    monkeypatch.setattr("llmpt.completed_registry.COMPLETED_SOURCES_FILE", str(completed_file))
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir=None, revision=None, repo_type=None: str(snapshot / filename),
    )
    save_completed_sources(
        [
            {
                "repo_type": "model",
                "repo_id": "org/model",
                "revision": commit,
                "storage_kind": "hub_cache",
                "storage_root": str(cache_root),
                "cache_dir": str(cache_root),
                "manifest": ["config.json"],
            }
        ]
    )

    assert has_completed_source("org/model", commit, cache_dir=str(cache_root)) is False
    assert get_completed_manifest("org/model", commit, cache_dir=str(cache_root)) is None
