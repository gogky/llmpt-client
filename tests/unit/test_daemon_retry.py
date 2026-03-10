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
    daemon._scan_and_seed(None, None, seeding_set, failed)
    assert calls == []

    # Retry deadline reached: should attempt again.
    failed[key]["next_retry_ts"] = time.time() - 1
    daemon._scan_and_seed(None, None, seeding_set, failed)
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

    daemon._scan_and_seed(None, FakeManager(), seeding_set, failed)

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

    discovered = daemon._reconcile_seeding_sessions(FakeManager(), seeding_set, failed, [live_source])

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

    daemon._scan_and_seed(None, None, seeding_set, failed)

    assert calls == [
        ("dataset", "org/shared", "e" * 40, "/tmp/cache-b", None)
    ]
