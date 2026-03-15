"""Helpers for patch-layer progress display and summary output."""

import threading
from typing import Any, Callable, Optional


def format_snapshot_p2p_postfix(
    stats: Optional[dict],
    download_stats: Optional[dict] = None,
) -> str:
    """Format a short, user-facing snapshot status for the shared progress bar."""
    stats = stats or {}
    download_stats = download_stats or {}

    active_peers = int(stats.get("active_p2p_peers", 0) or 0)
    peer_bytes = int(stats.get("peer_download", 0) or 0)
    webseed_bytes = int(stats.get("webseed_download", 0) or 0)
    http_seen = bool(download_stats.get("http"))

    if active_peers > 0 or peer_bytes > 0:
        return f"peers={active_peers}"
    if http_seen:
        return "http"
    if webseed_bytes > 0:
        return "webseed"
    return ""


class SnapshotProgressReporter:
    """Periodically updates snapshot_download's shared progress bar postfix."""

    def __init__(
        self,
        progress_bar: Any,
        repo_id: str,
        revision: str,
        repo_type: str,
        *,
        cache_dir: Optional[str] = None,
        local_dir: Optional[str] = None,
        get_download_stats_fn: Callable[..., dict],
        logger,
        update_interval: float = 0.25,
    ) -> None:
        self.progress_bar = progress_bar
        self.repo_id = repo_id
        self.revision = revision
        self.repo_type = repo_type
        self.cache_dir = cache_dir
        self.local_dir = local_dir
        self.get_download_stats_fn = get_download_stats_fn
        self.logger = logger
        self.update_interval = update_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None or not self.repo_id:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="llmpt-snapshot-progress",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._update_once()
            self._stop_event.wait(self.update_interval)
        self._update_once()

    def _update_once(self) -> None:
        try:
            from .p2p_batch import P2PBatchManager

            stats = P2PBatchManager().get_repo_p2p_stats(
                self.repo_id,
                self.revision,
                self.repo_type,
            )
            postfix = format_snapshot_p2p_postfix(
                stats,
                download_stats=self.get_download_stats_fn(
                    repo_id=self.repo_id,
                    revision=self.revision,
                    repo_type=self.repo_type,
                    cache_dir=self.cache_dir,
                    local_dir=self.local_dir,
                ),
            )
            if hasattr(self.progress_bar, "set_postfix_str"):
                self.progress_bar.set_postfix_str(postfix, refresh=False)
        except Exception as exc:
            self.logger.debug(f"[P2P] Live snapshot progress update failed: {exc}")


def _build_snapshot_tqdm_proxy(
    base_tqdm_class: Any,
    *,
    progress_bar_name: str,
    reporter_factory: Callable[[Any], Optional[SnapshotProgressReporter]],
) -> Any:
    """Build a tqdm proxy that wires snapshot progress reporting into one bar."""

    class SnapshotTqdmProxy:
        _lock = getattr(base_tqdm_class, "_lock", None)

        @classmethod
        def get_lock(cls):
            return getattr(cls, "_lock", None) or base_tqdm_class.get_lock()

        @classmethod
        def set_lock(cls, lock):
            cls._lock = lock
            return base_tqdm_class.set_lock(lock)

        def __init__(self, *args, **kwargs):
            object.__setattr__(self, "_llmpt_inner", base_tqdm_class(*args, **kwargs))
            object.__setattr__(self, "_llmpt_reporter", None)

            is_snapshot_bar = kwargs.get("name") == progress_bar_name
            is_disabled = getattr(self._llmpt_inner, "disable", False)
            if is_snapshot_bar and not is_disabled:
                reporter = reporter_factory(self._llmpt_inner)
                if reporter is not None:
                    object.__setattr__(self, "_llmpt_reporter", reporter)
                    reporter.start()

        def __getattr__(self, name):
            return getattr(self._llmpt_inner, name)

        def __setattr__(self, name, value):
            if name.startswith("_llmpt_"):
                object.__setattr__(self, name, value)
            else:
                setattr(self._llmpt_inner, name, value)

        def __enter__(self):
            entered = self._llmpt_inner.__enter__()
            return self if entered is self._llmpt_inner else entered

        def __exit__(self, exc_type, exc_value, traceback):
            return self._llmpt_inner.__exit__(exc_type, exc_value, traceback)

        def __iter__(self):
            return iter(self._llmpt_inner)

        def close(self):
            reporter = self._llmpt_reporter
            if reporter is not None:
                reporter.stop()
                object.__setattr__(self, "_llmpt_reporter", None)
            close_fn = getattr(self._llmpt_inner, "close", None)
            if close_fn is not None:
                return close_fn()

    SnapshotTqdmProxy.__name__ = "SnapshotTqdmProxy"
    return SnapshotTqdmProxy


def wrap_snapshot_tqdm_class(
    base_tqdm_class: Any,
    *,
    repo_id: str,
    revision: str,
    repo_type: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    progress_bar_name: str,
    get_download_stats_fn: Callable[..., dict],
    logger,
    update_interval: float = 0.25,
) -> Any:
    """Wrap a tqdm class so snapshot_download's shared bar shows live P2P stats."""

    def reporter_factory(progress_bar: Any) -> SnapshotProgressReporter:
        return SnapshotProgressReporter(
            progress_bar=progress_bar,
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            cache_dir=cache_dir,
            local_dir=local_dir,
            get_download_stats_fn=get_download_stats_fn,
            logger=logger,
            update_interval=update_interval,
        )

    return _build_snapshot_tqdm_proxy(
        base_tqdm_class,
        progress_bar_name=progress_bar_name,
        reporter_factory=reporter_factory,
    )


def wrap_snapshot_tqdm_class_auto(
    base_tqdm_class: Any,
    *,
    progress_bar_name: str,
    extract_snapshot_context_from_stack_fn: Callable[[], Optional[dict]],
    get_download_stats_fn: Callable[..., dict],
    logger,
    update_interval: float = 0.25,
) -> Any:
    """Wrap hf_tqdm so older snapshot_download refs still get live P2P stats."""

    def reporter_factory(progress_bar: Any) -> Optional[SnapshotProgressReporter]:
        ctx = extract_snapshot_context_from_stack_fn()
        if ctx is None:
            return None
        return SnapshotProgressReporter(
            progress_bar=progress_bar,
            repo_id=ctx["repo_id"],
            revision=ctx["revision"],
            repo_type=ctx["repo_type"],
            cache_dir=ctx.get("cache_dir"),
            local_dir=ctx.get("local_dir"),
            get_download_stats_fn=get_download_stats_fn,
            logger=logger,
            update_interval=update_interval,
        )

    proxy = _build_snapshot_tqdm_proxy(
        base_tqdm_class,
        progress_bar_name=progress_bar_name,
        reporter_factory=reporter_factory,
    )
    proxy.__name__ = "SnapshotAutoTqdmProxy"
    return proxy


def format_bytes(n: int) -> str:
    """Format byte count to a human-readable string."""
    try:
        from tqdm import tqdm as _tqdm

        formatted = _tqdm.format_sizeof(float(n), divisor=1000)
    except Exception:
        formatted = f"{n}B"

    if formatted == "0.00":
        return "0 B"
    if formatted.endswith("B"):
        return formatted
    return f"{formatted}B"


def print_p2p_summary(
    *,
    stats: dict,
    elapsed: float,
    repo_id: str,
    resolved_revision: str,
    repo_type: str,
) -> None:
    """Print a P2P acceleration summary after snapshot_download completes."""
    p2p_files = stats.get("p2p", set())
    http_files = stats.get("http", set())
    total_files = len(p2p_files) + len(http_files)

    if total_files == 0:
        return

    try:
        from .p2p_batch import P2PBatchManager

        session_stats = P2PBatchManager().get_repo_p2p_stats(
            repo_id,
            resolved_revision,
            repo_type,
        )
    except Exception:
        session_stats = {}

    sep = "-" * 56
    lines = [
        "",
        sep,
        f"  P2P Download Summary: {repo_id}",
        sep,
        f"  Files:     {len(p2p_files)}/{total_files} via P2P, "
        f"{len(http_files)}/{total_files} via HTTP fallback",
        f"  Time:      {elapsed:.1f}s",
    ]

    peer_bytes = session_stats.get("peer_download", 0)
    webseed_bytes = session_stats.get("webseed_download", 0)
    total_bytes = peer_bytes + webseed_bytes

    if total_bytes > 0:
        lines.append("")
        lines.append("  Data Sources:")
        if peer_bytes > 0:
            ratio = peer_bytes / total_bytes * 100
            lines.append(
                f"    From P2P peers:  {format_bytes(peer_bytes)} ({ratio:.1f}%)"
            )
        if webseed_bytes > 0:
            ratio = webseed_bytes / total_bytes * 100
            lines.append(
                f"    From WebSeed:    {format_bytes(webseed_bytes)} ({ratio:.1f}%)"
            )
        if peer_bytes > 0:
            lines.append("")
            lines.append(
                f"  Bandwidth saved: {format_bytes(peer_bytes)} from server"
            )

    max_peers = session_stats.get("max_p2p_peers", 0)
    if max_peers > 0:
        lines.append(f"  Peak peers: {max_peers}")

    lines.append(sep)
    lines.append("")

    output = "\n".join(lines)
    try:
        from tqdm import tqdm as _tqdm

        _tqdm.write(output)
    except ImportError:
        print(output)
