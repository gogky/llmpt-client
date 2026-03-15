"""
Coordinate logical transfer planning separately from libtorrent execution.

Today, every target file request is fulfilled by the matching torrent for the
same repo/revision. Keeping that planning step explicit now gives future
cross-swarm routing a single place to evolve without pushing more policy into
patch.py or P2PBatchManager.
"""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .tracker_client import TrackerClient

from .p2p_batch import P2PBatchManager
from .session_identity import build_logical_torrent_ref, build_storage_identity
from .transfer_types import (
    TargetFileRequest,
    TorrentSourceRef,
    TransferPlan,
    TransferResult,
)


class TransferCoordinator:
    """Plan and execute one logical file transfer."""

    def __init__(self, batch_manager: Optional[P2PBatchManager] = None) -> None:
        self.batch_manager = batch_manager or P2PBatchManager()

    def build_target_request(
        self,
        *,
        repo_id: str,
        revision: str,
        filename: str,
        destination: str,
        repo_type: str = "model",
        cache_dir: Optional[str] = None,
        local_dir: Optional[str] = None,
    ) -> TargetFileRequest:
        """Build the logical target request from Hugging Face download inputs."""
        return TargetFileRequest(
            logical=build_logical_torrent_ref(repo_type, repo_id, revision),
            filename=filename,
            destination=destination,
            storage=build_storage_identity(
                cache_dir=cache_dir,
                local_dir=local_dir,
            ),
        )

    def plan_request(self, target: TargetFileRequest) -> TransferPlan:
        """Plan how to fulfill a target request.

        Current behavior stays identical to the pre-refactor client: the source
        torrent is the torrent identified by the same repo/revision/storage.
        """
        return TransferPlan(
            target=target,
            source=TorrentSourceRef(
                logical=target.logical,
                storage=target.storage,
            ),
        )

    def resolve_timeout(self, config: Optional[dict[str, Any]] = None) -> int:
        """Resolve the execution timeout from patch-layer config."""
        config = config or {}
        if config.get("webseed_proxy_port"):
            return 0
        return int(config.get("timeout", 300))

    def execute_plan(
        self,
        plan: TransferPlan,
        *,
        tracker_client: "TrackerClient",
        timeout: int = 300,
        tqdm_class: Optional[Any] = None,
    ) -> TransferResult:
        """Execute one transfer plan through the batch manager."""
        success = self.batch_manager.execute_transfer(
            plan,
            tracker_client=tracker_client,
            timeout=timeout,
            tqdm_class=tqdm_class,
        )
        if success:
            return TransferResult(
                success=True,
                delivered_path=plan.target.destination,
                via="p2p",
            )
        return TransferResult(success=False, via="fallback")

    def fulfill_request(
        self,
        *,
        repo_id: str,
        revision: str,
        filename: str,
        destination: str,
        tracker_client: "TrackerClient",
        repo_type: str = "model",
        cache_dir: Optional[str] = None,
        local_dir: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        tqdm_class: Optional[Any] = None,
    ) -> TransferResult:
        """Build, plan, and execute a file transfer request."""
        target = self.build_target_request(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            destination=destination,
            repo_type=repo_type,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
        plan = self.plan_request(target)
        return self.execute_plan(
            plan,
            tracker_client=tracker_client,
            timeout=self.resolve_timeout(config),
            tqdm_class=tqdm_class,
        )
