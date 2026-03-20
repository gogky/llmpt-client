"""Tests for the transfer coordinator abstraction."""

from unittest.mock import MagicMock


def test_plan_request_keeps_target_and_source_explicit():
    from llmpt.transfer_coordinator import TransferCoordinator

    coordinator = TransferCoordinator(batch_manager=MagicMock())
    target = coordinator.build_target_request(
        repo_id="demo/repo",
        revision="main",
        filename="model.bin",
        destination="/tmp/model.bin",
        repo_type="model",
        local_dir="/tmp/model",
    )

    plan = coordinator.plan_request(target)

    assert plan.target is target
    assert plan.source.logical == target.logical
    assert plan.source.storage == target.storage
    assert plan.source_filename == target.filename


def test_plan_request_uses_legacy_swarm_when_exact_candidate_has_no_seeders():
    from llmpt.transfer_coordinator import TransferCoordinator
    from llmpt.transfer_types import (
        LogicalTorrentRef,
        SourceFileCandidate,
        TorrentSourceRef,
    )

    tracker = MagicMock()
    tracker.resolve_file_sources.return_value = [
        SourceFileCandidate(
            source=TorrentSourceRef(
                logical=LogicalTorrentRef(
                    repo_type="model",
                    repo_id="demo/repo",
                    revision="main",
                ),
            ),
            filename="model.bin",
            file_root="abc123",
            size=100,
            seeders=0,
        ),
        SourceFileCandidate(
            source=TorrentSourceRef(
                logical=LogicalTorrentRef(
                    repo_type="model",
                    repo_id="demo/repo",
                    revision="oldrev",
                ),
            ),
            filename="model.bin",
            file_root="abc123",
            size=100,
            seeders=4,
        ),
    ]

    coordinator = TransferCoordinator(batch_manager=MagicMock())
    target = coordinator.build_target_request(
        repo_id="demo/repo",
        revision="main",
        filename="model.bin",
        destination="/tmp/model.bin",
        repo_type="model",
        cache_dir="/tmp/cache",
    )

    plan = coordinator.plan_request(target, tracker_client=tracker)

    assert plan.source.repo_id == "demo/repo"
    assert plan.source.revision == "oldrev"
    assert plan.source_filename == "model.bin"
    assert len(plan.candidates) == 2


def test_plan_request_keeps_exact_target_when_legacy_swarm_file_root_differs():
    from llmpt.transfer_coordinator import TransferCoordinator
    from llmpt.transfer_types import (
        LogicalTorrentRef,
        SourceFileCandidate,
        TorrentSourceRef,
    )

    tracker = MagicMock()
    tracker.resolve_file_sources.return_value = [
        SourceFileCandidate(
            source=TorrentSourceRef(
                logical=LogicalTorrentRef(
                    repo_type="model",
                    repo_id="demo/repo",
                    revision="main",
                ),
            ),
            filename="model.bin",
            file_root="abc123",
            size=100,
            seeders=0,
        ),
        SourceFileCandidate(
            source=TorrentSourceRef(
                logical=LogicalTorrentRef(
                    repo_type="model",
                    repo_id="demo/repo",
                    revision="oldrev",
                ),
            ),
            filename="renamed.bin",
            file_root="different-root",
            size=100,
            seeders=99,
        ),
    ]

    coordinator = TransferCoordinator(batch_manager=MagicMock())
    target = coordinator.build_target_request(
        repo_id="demo/repo",
        revision="main",
        filename="model.bin",
        destination="/tmp/model.bin",
        repo_type="model",
        cache_dir="/tmp/cache",
    )

    plan = coordinator.plan_request(target, tracker_client=tracker)

    assert plan.source.repo_id == "demo/repo"
    assert plan.source.revision == "main"
    assert plan.source_filename == "model.bin"


def test_fulfill_request_executes_plan_with_configured_timeout():
    from llmpt.transfer_coordinator import TransferCoordinator

    batch_manager = MagicMock()
    batch_manager.execute_transfer.return_value = True
    tracker = MagicMock()
    coordinator = TransferCoordinator(batch_manager=batch_manager)

    result = coordinator.fulfill_request(
        repo_id="demo/repo",
        revision="main",
        filename="model.bin",
        destination="/tmp/model.bin",
        tracker_client=tracker,
        repo_type="model",
        cache_dir="/tmp/cache",
        config={"timeout": 42},
        tqdm_class="fake_tqdm",
    )

    assert result.success is True
    assert result.delivered_path == "/tmp/model.bin"

    call = batch_manager.execute_transfer.call_args
    plan = call.args[0]
    assert plan.target.filename == "model.bin"
    assert plan.target.destination == "/tmp/model.bin"
    assert plan.target.storage.kind == "hub_cache"
    assert plan.target.storage.root == "/tmp/cache"
    assert plan.source_filename == "model.bin"
    assert call.kwargs["tracker_client"] is tracker
    assert call.kwargs["timeout"] == 42
    assert call.kwargs["tqdm_class"] == "fake_tqdm"


def test_webseed_config_disables_timeout():
    from llmpt.transfer_coordinator import TransferCoordinator

    batch_manager = MagicMock()
    batch_manager.execute_transfer.return_value = False
    coordinator = TransferCoordinator(batch_manager=batch_manager)

    result = coordinator.fulfill_request(
        repo_id="demo/repo",
        revision="main",
        filename="model.bin",
        destination="/tmp/model.bin",
        tracker_client=MagicMock(),
        config={"timeout": 60, "webseed_proxy_port": 8080},
    )

    assert result.success is False
    assert batch_manager.execute_transfer.call_args.kwargs["timeout"] == 0
