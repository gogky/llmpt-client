"""Tests for transfer-related value objects."""

import pytest


def test_storage_identity_rejects_unknown_kind():
    from llmpt.transfer_types import StorageIdentity

    with pytest.raises(ValueError, match="unsupported storage kind"):
        StorageIdentity(kind="unknown", root="/tmp/demo")


def test_storage_identity_requires_non_empty_root():
    from llmpt.transfer_types import StorageIdentity

    with pytest.raises(ValueError, match="storage root must be non-empty"):
        StorageIdentity(kind="hub_cache", root="")


def test_source_session_key_legacy_tuple_shape():
    from llmpt.transfer_types import LogicalTorrentRef, SourceSessionKey, StorageIdentity

    key = SourceSessionKey(
        logical=LogicalTorrentRef(
            repo_type="model",
            repo_id="demo",
            revision="main",
        ),
        storage=StorageIdentity(
            kind="hub_cache",
            root="/tmp/cache",
        ),
    )

    assert key.as_legacy_tuple() == (
        "model",
        "demo",
        "main",
        "hub_cache",
        "/tmp/cache",
    )


def test_transfer_plan_keeps_target_and_source_explicit():
    from llmpt.transfer_types import (
        LogicalTorrentRef,
        StorageIdentity,
        TargetFileRequest,
        TorrentSourceRef,
        TransferPlan,
    )

    logical = LogicalTorrentRef(
        repo_type="model",
        repo_id="demo",
        revision="main",
    )
    storage = StorageIdentity(kind="hub_cache", root="/tmp/cache")
    target = TargetFileRequest(
        logical=logical,
        filename="model.bin",
        destination="/tmp/model.bin",
        storage=storage,
    )
    source = TorrentSourceRef(logical=logical, storage=storage)

    plan = TransferPlan(target=target, source=source)

    assert plan.target.filename == "model.bin"
    assert plan.source.repo_id == "demo"
