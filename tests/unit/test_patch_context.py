"""Tests for patch-layer context helpers."""

import threading


def test_read_thread_local_context_requires_core_fields():
    from llmpt.patch_context import read_thread_local_context

    context = threading.local()
    context.repo_id = "demo/repo"
    context.filename = "model.bin"

    assert read_thread_local_context(context) is None


def test_apply_and_read_thread_local_context_round_trip():
    from llmpt.patch_context import apply_thread_local_context, read_thread_local_context

    context = threading.local()
    config = {"timeout": 60}

    apply_thread_local_context(
        context,
        repo_id="demo/repo",
        repo_type="model",
        filename="model.bin",
        revision="main",
        tracker="tracker",
        config=config,
        cache_dir="/tmp/cache",
    )

    result = read_thread_local_context(context)

    assert result is not None
    assert result["repo_id"] == "demo/repo"
    assert result["repo_type"] == "model"
    assert result["filename"] == "model.bin"
    assert result["revision"] == "main"
    assert result["tracker"] == "tracker"
    assert result["config"] is config
    assert result["cache_dir"] == "/tmp/cache"


def test_capture_and_restore_thread_local_context():
    from llmpt.patch_context import (
        apply_thread_local_context,
        capture_thread_local_context,
        restore_thread_local_context,
    )

    context = threading.local()
    apply_thread_local_context(
        context,
        repo_id="first/repo",
        repo_type="model",
        filename="first.bin",
        revision="main",
    )
    snapshot = capture_thread_local_context(context)

    apply_thread_local_context(
        context,
        repo_id="second/repo",
        repo_type="dataset",
        filename="second.bin",
        revision="dev",
        local_dir="/tmp/local",
    )
    restore_thread_local_context(context, snapshot)

    assert context.repo_id == "first/repo"
    assert context.repo_type == "model"
    assert context.filename == "first.bin"
    assert context.revision == "main"
    assert context.local_dir is None
