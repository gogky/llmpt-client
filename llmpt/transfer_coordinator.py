"""
Coordinate logical transfer planning separately from libtorrent execution.

Today, every target file request prefers the matching torrent for the same
repo/revision, but can fall back to older swarms when the tracker returns
file-level candidates whose torrent-v2 ``file_root`` and size prove they are
the exact same file content. Keeping that planning step explicit gives future
routing changes a single place to evolve without pushing more policy into
patch.py or P2PBatchManager.
"""

from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    from .tracker_client import TrackerClient

from .p2p_batch import P2PBatchManager
from .session_identity import build_logical_torrent_ref, build_storage_identity
from .transfer_types import (
    SourceFileCandidate,
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

    def build_primary_candidate(self, target: TargetFileRequest) -> SourceFileCandidate:
        """Build the default exact-revision candidate for one target file."""
        return SourceFileCandidate(
            source=TorrentSourceRef(
                logical=target.logical,
                storage=target.storage,
            ),
            filename=target.filename,
        )

    def _candidate_with_target_storage(
        self,
        candidate: SourceFileCandidate,
        target: TargetFileRequest,
    ) -> SourceFileCandidate:
        """Attach the target storage identity when a tracker candidate omits it."""
        if candidate.source.storage == target.storage:
            return candidate
        return SourceFileCandidate(
            source=TorrentSourceRef(
                logical=candidate.source.logical,
                storage=target.storage,
            ),
            filename=candidate.filename,
            file_root=candidate.file_root,
            size=candidate.size,
            seeders=candidate.seeders,
            score=candidate.score,
        )

    def resolve_source_candidates(
        self,
        target: TargetFileRequest,
        *,
        tracker_client: Optional["TrackerClient"] = None,
    ) -> list[SourceFileCandidate]:
        """Resolve the candidate source files for one target request.

        The target torrent itself always remains a candidate. File-level source
        discovery is best-effort: if the tracker does not yet implement the new
        API, callers transparently fall back to the exact target torrent.
        """
        primary = self.build_primary_candidate(target)
        candidates = [primary]
        if tracker_client is not None:
            for candidate in tracker_client.resolve_file_sources(
                target.repo_id,
                target.revision,
                target.filename,
                repo_type=target.repo_type,
            ):
                candidates.append(self._candidate_with_target_storage(candidate, target))
        return self._dedupe_candidates(candidates)

    def _dedupe_candidates(
        self,
        candidates: Iterable[SourceFileCandidate],
    ) -> list[SourceFileCandidate]:
        """Keep the highest-signal candidate per source torrent and file path."""
        deduped: dict[tuple[str, str, str, str], SourceFileCandidate] = {}
        for candidate in candidates:
            key = (
                candidate.repo_type,
                candidate.repo_id,
                candidate.revision,
                candidate.filename,
            )
            current = deduped.get(key)
            if current is None or self._candidate_metadata_key(candidate) > self._candidate_metadata_key(current):
                deduped[key] = candidate
        return list(deduped.values())

    def _candidate_is_exact_target(
        self,
        candidate: SourceFileCandidate,
        target: TargetFileRequest,
    ) -> bool:
        """Return True when a candidate points at the target torrent and path."""
        return (
            candidate.source.logical == target.logical
            and candidate.filename == target.filename
        )

    def _candidate_has_available_seeders(self, candidate: SourceFileCandidate) -> bool:
        """Treat missing seeder counts as healthy so legacy trackers stay compatible."""
        return candidate.seeders is None or candidate.seeders > 0

    def _candidate_metadata_key(
        self,
        candidate: SourceFileCandidate,
    ) -> tuple[int, int, int]:
        """Prefer candidates that carry content identity and seeder metadata."""
        return (
            int(bool(candidate.file_root)),
            int(candidate.size is not None),
            candidate.seeders if candidate.seeders is not None else -1,
        )

    def _candidate_matches_content(
        self,
        candidate: SourceFileCandidate,
        reference: SourceFileCandidate,
    ) -> bool:
        """Return True only when both candidates describe the same file content."""
        return (
            bool(reference.file_root)
            and reference.size is not None
            and candidate.file_root == reference.file_root
            and candidate.size == reference.size
        )

    def _candidate_sort_key(
        self,
        candidate: SourceFileCandidate,
        target: Optional[TargetFileRequest],
    ) -> tuple[int]:
        """Rank alternates primarily by swarm health after content identity matches."""
        del target
        seeders = candidate.seeders if candidate.seeders is not None else -1
        return (seeders,)

    def choose_source_candidate(
        self,
        target: TargetFileRequest,
        candidates: Iterable[SourceFileCandidate],
    ) -> SourceFileCandidate:
        """Choose the final source candidate for one target file.

        Policy is intentionally conservative:
        1. Prefer the exact target torrent whenever it has peers (or peer count
           is unknown because the tracker does not expose file-level stats yet).
        2. Otherwise, only consider alternates whose ``file_root`` and size
           match the exact candidate.
        3. Among those exact-content alternates, choose the healthiest swarm by
           seeder count.
        """
        candidate_list = list(candidates)
        exact_candidates = [
            candidate for candidate in candidate_list
            if self._candidate_is_exact_target(candidate, target)
        ]
        if exact_candidates:
            exact = max(exact_candidates, key=self._candidate_metadata_key)
            if self._candidate_has_available_seeders(exact):
                return exact

        alternates = [
            candidate for candidate in candidate_list
            if not self._candidate_is_exact_target(candidate, target)
            and exact_candidates
            and self._candidate_matches_content(candidate, exact)
        ]
        if alternates:
            return max(alternates, key=lambda candidate: self._candidate_sort_key(candidate, target))

        if exact_candidates:
            return exact_candidates[0]
        return self.build_primary_candidate(target)

    def plan_request(
        self,
        target: TargetFileRequest,
        *,
        tracker_client: Optional["TrackerClient"] = None,
    ) -> TransferPlan:
        """Plan how to fulfill a target request."""
        candidates = self.resolve_source_candidates(
            target,
            tracker_client=tracker_client,
        )
        source_file = self.choose_source_candidate(target, candidates)
        return TransferPlan(
            target=target,
            source_file=source_file,
            candidates=tuple(candidates),
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
        plan = self.plan_request(
            target,
            tracker_client=tracker_client,
        )
        return self.execute_plan(
            plan,
            tracker_client=tracker_client,
            timeout=self.resolve_timeout(config),
            tqdm_class=tqdm_class,
        )
