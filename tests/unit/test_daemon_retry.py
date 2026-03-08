"""Unit tests for daemon retry/backoff behavior."""

import time


def test_record_seed_failure_increments_attempts_and_backoff(monkeypatch):
    import llmpt.daemon as daemon

    key = ("model", "org/repo", "a" * 40)
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

    key = ("model", "org/repo", "b" * 40)
    source = SeedableSource(
        repo_type="model", repo_id="org/repo", revision="b" * 40,
        storage_kind="hub_cache", storage_root="",
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
    assert calls == [key]

