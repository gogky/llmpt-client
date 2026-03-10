"""Unit tests for daemon retry/backoff behavior."""

import time


def test_record_seed_failure_increments_attempts_and_backoff(monkeypatch):
    import llmpt.daemon as daemon

    key = ("model", "org/repo", "a" * 40, "hub_cache", "/tmp/cache-a")
    failed = {}

    # Remove jitter so assertions are deterministic.
    monkeypatch.setattr(daemon.random, "uniform", lambda *_args, **_kwargs: 0.0)

    daemon._record_seed_failure(key, failed, "first")
    first = failed[key]
    assert first["attempts"] == 1
    first_delay = float(first["next_retry_ts"]) - time.time()
    assert first_delay > 50

    daemon._record_seed_failure(key, failed, "second")
    second = failed[key]
    assert second["attempts"] == 2
    second_delay = float(second["next_retry_ts"]) - time.time()
    assert second_delay > first_delay


def test_scan_and_seed_respects_retry_deadline(monkeypatch):
    import llmpt.daemon as daemon

    from llmpt.cache_scanner import SeedableSource

    key = ("model", "org/repo", "b" * 40, "hub_cache", "/tmp/cache-b")
    source = SeedableSource(
        repo_type="model", repo_id="org/repo", revision="b" * 40,
        storage_kind="hub_cache", storage_root="/tmp/cache-b",
        cache_dir="/tmp/cache-b",
    )
    monkeypatch.setattr("llmpt.cache_scanner.scan_seedable_sources", lambda: [source])
    monkeypatch.setattr(
        "llmpt.cache_importer.import_verified_cache_sources",
        lambda: {
            "imported": 0,
            "skipped_completed": 0,
            "skipped_backoff": 0,
            "blocked": 0,
            "partial": 0,
            "error": 0,
        },
    )

    calls = []

    def fake_process_seedable(
        repo_id,
        revision,
        tracker_client,
        manager,
        seeding_set,
        failed_attempts,
        *,
        repo_type="model",
        cache_dir=None,
        local_dir=None,
    ):
        calls.append((repo_type, repo_id, revision))
        return True

    monkeypatch.setattr(daemon, "_process_seedable", fake_process_seedable)

    seeding_set = set()
    failed = {
        key: {"attempts": 1, "next_retry_ts": time.time() + 120, "last_error": "network"}
    }

    # Still in backoff window: should skip.
    daemon._scan_and_seed(None, None, seeding_set, failed, set())
    assert calls == []

    # Retry deadline reached: should attempt again.
    failed[key]["next_retry_ts"] = time.time() - 1
    daemon._scan_and_seed(None, None, seeding_set, failed, set())
    assert calls == [("model", "org/repo", "b" * 40)]


def test_scan_and_seed_removes_stale_session(monkeypatch):
    import llmpt.daemon as daemon

    from llmpt.cache_scanner import SeedableSource

    live_source = SeedableSource(
        repo_type="model",
        repo_id="org/live",
        revision="c" * 40,
        storage_kind="hub_cache",
        storage_root="/tmp/cache-live",
        cache_dir="/tmp/cache-live",
    )
    monkeypatch.setattr("llmpt.cache_scanner.scan_seedable_sources", lambda: [live_source])
    monkeypatch.setattr(
        "llmpt.cache_importer.import_verified_cache_sources",
        lambda: {
            "imported": 0,
            "skipped_completed": 0,
            "skipped_backoff": 0,
            "blocked": 0,
            "partial": 0,
            "error": 0,
        },
    )
    monkeypatch.setattr(
        daemon,
        "_process_seedable",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not seed live session again")),
    )

    removed = []

    class FakeManager:
        def remove_session(self, repo_id, revision, *, repo_type="model", cache_dir=None, local_dir=None):
            removed.append((repo_type, repo_id, revision, cache_dir, local_dir))
            return True

    live_key = ("model", "org/live", "c" * 40, "hub_cache", "/tmp/cache-live")
    stale_key = ("dataset", "org/stale", "d" * 40, "hub_cache", "/tmp/cache-stale")
    seeding_set = {live_key, stale_key}
    failed = {stale_key: {"attempts": 1, "next_retry_ts": time.time() + 60, "last_error": "old"}}

    daemon._scan_and_seed(None, FakeManager(), seeding_set, failed, set())

    assert removed == [
        ("dataset", "org/stale", "d" * 40, "/tmp/cache-stale", None)
    ]
    assert seeding_set == {live_key}
    assert stale_key not in failed


def test_reconcile_seeding_sessions_removes_stale_without_waiting_for_scan_tick():
    import llmpt.daemon as daemon

    from llmpt.cache_scanner import SeedableSource

    live_source = SeedableSource(
        repo_type="model",
        repo_id="org/live",
        revision="f" * 40,
        storage_kind="hub_cache",
        storage_root="/tmp/cache-live",
        cache_dir="/tmp/cache-live",
    )

    removed = []

    class FakeManager:
        def remove_session(self, repo_id, revision, *, repo_type="model", cache_dir=None, local_dir=None):
            removed.append((repo_type, repo_id, revision, cache_dir, local_dir))
            return True

    live_key = ("model", "org/live", "f" * 40, "hub_cache", "/tmp/cache-live")
    stale_key = ("dataset", "org/stale", "0" * 40, "hub_cache", "/tmp/cache-stale")
    seeding_set = {live_key, stale_key}
    failed = {stale_key: {"attempts": 1, "next_retry_ts": time.time() + 60, "last_error": "old"}}

    discovered = daemon._reconcile_seeding_sessions(
        FakeManager(), seeding_set, failed, set(), [live_source]
    )

    assert discovered == {live_key}
    assert removed == [
        ("dataset", "org/stale", "0" * 40, "/tmp/cache-stale", None)
    ]
    assert seeding_set == {live_key}
    assert stale_key not in failed


def test_scan_and_seed_dedupes_by_storage_root(monkeypatch):
    import llmpt.daemon as daemon

    from llmpt.cache_scanner import SeedableSource

    source_a = SeedableSource(
        repo_type="dataset",
        repo_id="org/shared",
        revision="e" * 40,
        storage_kind="hub_cache",
        storage_root="/tmp/cache-a",
        cache_dir="/tmp/cache-a",
    )
    source_b = SeedableSource(
        repo_type="dataset",
        repo_id="org/shared",
        revision="e" * 40,
        storage_kind="hub_cache",
        storage_root="/tmp/cache-b",
        cache_dir="/tmp/cache-b",
    )
    monkeypatch.setattr("llmpt.cache_scanner.scan_seedable_sources", lambda: [source_a, source_b])
    monkeypatch.setattr(
        "llmpt.cache_importer.import_verified_cache_sources",
        lambda: {
            "imported": 0,
            "skipped_completed": 0,
            "skipped_backoff": 0,
            "blocked": 0,
            "partial": 0,
            "error": 0,
        },
    )

    calls = []

    def fake_process_seedable(
        repo_id,
        revision,
        tracker_client,
        manager,
        seeding_set,
        failed_attempts,
        *,
        repo_type="model",
        cache_dir=None,
        local_dir=None,
    ):
        calls.append((repo_type, repo_id, revision, cache_dir, local_dir))
        return True

    monkeypatch.setattr(daemon, "_process_seedable", fake_process_seedable)

    seeding_set = {("dataset", "org/shared", "e" * 40, "hub_cache", "/tmp/cache-a")}
    failed = {}

    daemon._scan_and_seed(None, None, seeding_set, failed, set())

    assert calls == [
        ("dataset", "org/shared", "e" * 40, "/tmp/cache-b", None)
    ]


def test_matching_seeding_keys_can_match_all_revisions_for_repo():
    import llmpt.daemon as daemon

    seeding_set = {
        ("dataset", "org/shared", "1" * 40, "hub_cache", "/tmp/cache-a"),
        ("dataset", "org/shared", "2" * 40, "hub_cache", "/tmp/cache-b"),
        ("dataset", "org/other", "1" * 40, "hub_cache", "/tmp/cache-a"),
    }

    matches = daemon._matching_seeding_keys(
        seeding_set,
        repo_type="dataset",
        repo_id="org/shared",
        revision=None,
    )

    assert set(matches) == {
        ("dataset", "org/shared", "1" * 40, "hub_cache", "/tmp/cache-a"),
        ("dataset", "org/shared", "2" * 40, "hub_cache", "/tmp/cache-b"),
    }


def test_unseed_matching_sessions_can_forget_registry(monkeypatch):
    import llmpt.daemon as daemon

    forgotten_calls = []

    def fake_forget_seedable_storage(
        repo_id,
        revision,
        *,
        repo_type="model",
        cache_dir=None,
        local_dir=None,
    ):
        forgotten_calls.append((repo_type, repo_id, revision, cache_dir, local_dir))
        return {"hub_cache_roots_removed": 1, "local_dir_sources_removed": 0}

    monkeypatch.setattr(
        "llmpt.completed_registry.forget_completed_source",
        lambda **kwargs: 1,
    )

    monkeypatch.setattr(
        "llmpt.cache_scanner.forget_seedable_storage",
        fake_forget_seedable_storage,
    )

    removed = []

    class FakeManager:
        def remove_session(self, repo_id, revision, *, repo_type="model", cache_dir=None, local_dir=None):
            removed.append((repo_type, repo_id, revision, cache_dir, local_dir))
            return True

    key = ("dataset", "org/shared", "2" * 40, "hub_cache", "/tmp/cache-a")
    seeding_set = {key}
    suppressed_set = set()
    failed = {key: {"attempts": 1, "next_retry_ts": time.time() + 60, "last_error": "old"}}

    result = daemon._unseed_matching_sessions(
        FakeManager(),
        seeding_set,
        failed,
        suppressed_set,
        repo_type="dataset",
        repo_id="org/shared",
        revision=None,
        forget=True,
    )

    assert result["status"] == "ok"
    assert result["removed_count"] == 1
    assert result["forgotten"] == {
        "hub_cache_roots_removed": 1,
        "local_dir_sources_removed": 0,
        "completed_sources_removed": 1,
    }
    assert removed == [
        ("dataset", "org/shared", "2" * 40, "/tmp/cache-a", None)
    ]
    assert forgotten_calls == [
        ("dataset", "org/shared", "2" * 40, "/tmp/cache-a", None)
    ]
    assert seeding_set == set()
    assert suppressed_set == {key}
    assert failed == {}


def test_unseed_matching_sessions_requires_repo_type_when_ambiguous():
    import llmpt.daemon as daemon

    class FakeManager:
        def remove_session(self, *args, **kwargs):
            raise AssertionError("remove_session should not be called on ambiguity")

    seeding_set = {
        ("dataset", "org/shared", "3" * 40, "hub_cache", "/tmp/cache-a"),
        ("model", "org/shared", "4" * 40, "hub_cache", "/tmp/cache-b"),
    }
    suppressed_set = set()
    failed = {}

    result = daemon._unseed_matching_sessions(
        FakeManager(),
        seeding_set,
        failed,
        suppressed_set,
        repo_type=None,
        repo_id="org/shared",
        revision=None,
        forget=False,
    )

    assert result["status"] == "error"
    assert "multiple repo types match this repo_id" in result["message"]


def test_scan_and_seed_skips_suppressed_keys(monkeypatch):
    import llmpt.daemon as daemon

    from llmpt.cache_scanner import SeedableSource

    source = SeedableSource(
        repo_type="dataset",
        repo_id="org/suppressed",
        revision="5" * 40,
        storage_kind="local_dir",
        storage_root="/tmp/local-dir",
        local_dir="/tmp/local-dir",
    )
    key = ("dataset", "org/suppressed", "5" * 40, "local_dir", "/tmp/local-dir")
    monkeypatch.setattr("llmpt.cache_scanner.scan_seedable_sources", lambda: [source])
    monkeypatch.setattr(
        "llmpt.cache_importer.import_verified_cache_sources",
        lambda: {
            "imported": 0,
            "skipped_completed": 0,
            "skipped_backoff": 0,
            "blocked": 0,
            "partial": 0,
            "error": 0,
        },
    )

    calls = []

    def fake_process_seedable(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(daemon, "_process_seedable", fake_process_seedable)

    daemon._scan_and_seed(
        None,
        None,
        set(),
        {},
        {key},
    )

    assert calls == []


def test_scan_and_seed_runs_cache_import_before_seeding(monkeypatch):
    import llmpt.daemon as daemon

    calls = []

    monkeypatch.setattr(
        "llmpt.cache_importer.import_verified_cache_sources",
        lambda: calls.append("import") or {
            "imported": 1,
            "skipped_completed": 0,
            "skipped_backoff": 0,
            "blocked": 0,
            "partial": 0,
            "error": 0,
        },
    )
    monkeypatch.setattr("llmpt.cache_scanner.scan_seedable_sources", lambda: [])

    daemon._scan_and_seed(None, None, set(), {}, set())

    assert calls == ["import"]


def test_reconcile_keeps_only_live_suppressed_keys():
    import llmpt.daemon as daemon

    from llmpt.cache_scanner import SeedableSource

    live_source = SeedableSource(
        repo_type="dataset",
        repo_id="org/live",
        revision="6" * 40,
        storage_kind="local_dir",
        storage_root="/tmp/live",
        local_dir="/tmp/live",
    )
    live_key = ("dataset", "org/live", "6" * 40, "local_dir", "/tmp/live")
    stale_suppressed = ("dataset", "org/stale", "7" * 40, "local_dir", "/tmp/stale")
    suppressed_set = {live_key, stale_suppressed}

    class FakeManager:
        def remove_session(self, *args, **kwargs):
            return True

    daemon._reconcile_seeding_sessions(
        FakeManager(),
        set(),
        {},
        suppressed_set,
        [live_source],
    )

    assert suppressed_set == {live_key}
