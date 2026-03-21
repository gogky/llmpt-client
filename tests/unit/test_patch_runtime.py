"""Tests for extracted patch runtime helpers."""

import threading
from unittest.mock import patch


def test_snapshot_and_deferred_keys_use_default_hub_cache():
    from llmpt.patch_runtime import deferred_key, snapshot_stats_key
    from llmpt.utils import get_hf_hub_cache

    expected = ("model", "demo/repo", "main", "hub_cache", get_hf_hub_cache())

    assert snapshot_stats_key("demo/repo", "main", "model") == expected
    assert deferred_key("demo/repo", "main", "model") == expected


def test_stats_helpers_round_trip_and_reset_specific_bucket():
    from llmpt.patch_runtime import (
        get_download_stats,
        record_download_stat,
        reset_download_stats,
        snapshot_stats_key,
    )

    stats_lock = threading.Lock()
    download_stats = {}
    key = snapshot_stats_key("demo/repo", "main", "model")

    record_download_stat(
        stats_lock=stats_lock,
        download_stats=download_stats,
        stats_key=key,
        stat_kind="p2p",
        filename="model.bin",
    )

    stats = get_download_stats(
        stats_lock=stats_lock,
        download_stats=download_stats,
        snapshot_key_builder=snapshot_stats_key,
        stats_key=key,
    )
    assert stats == {"p2p": {"model.bin"}, "http": set()}

    reset_download_stats(
        stats_lock=stats_lock,
        download_stats=download_stats,
        snapshot_key_builder=snapshot_stats_key,
        stats_key=key,
    )

    stats = get_download_stats(
        stats_lock=stats_lock,
        download_stats=download_stats,
        snapshot_key_builder=snapshot_stats_key,
    )
    assert stats == {"p2p": set(), "http": set()}


def test_notify_seed_daemon_omits_completed_snapshot_when_false():
    from llmpt.patch_runtime import notify_seed_daemon

    with patch("llmpt.ipc.notify_daemon") as mock_notify:
        notify_seed_daemon(
            repo_id="demo/repo",
            revision="main",
            repo_type="model",
        )

    mock_notify.assert_called_once_with(
        "seed",
        repo_id="demo/repo",
        revision="main",
        repo_type="model",
    )
