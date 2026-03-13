"""
Child-process harness for breakpoint-resume E2E tests.

The parent pytest process launches this script, waits for partial progress,
interrupts it, and then launches it again with the same storage roots.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
import llmpt
import llmpt.p2p_batch as p2p_batch_module
import llmpt.patch as patch_module
import llmpt.session_context as session_context_module
import llmpt.torrent_init as torrent_init
from huggingface_hub import snapshot_download
import huggingface_hub.utils as hf_utils
from tqdm.auto import tqdm as _BaseTqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume E2E download harness")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--events-file", required=True)
    parser.add_argument("--tracker-url", required=True)
    parser.add_argument("--allow-pattern", default="pytorch_model.bin")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--partial-threshold", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--local-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--webseed", action="store_true")
    parser.add_argument("--interrupt-on-partial", action="store_true")
    parser.add_argument("--p2p-download-limit", type=int, default=0)
    return parser.parse_args()


def _make_emitter(events_file: str):
    path = Path(events_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(event: str, **payload) -> None:
        record = {"event": event, "ts": time.time(), **payload}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            f.flush()

    return _emit


def _dir_size(root: str) -> int:
    total = 0
    if not os.path.isdir(root):
        return 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, filename))
            except OSError:
                pass
    return total


class _RecordingTqdm(_BaseTqdm):
    emitter = None
    interrupt_on_partial = False
    partial_threshold = 0
    _interrupted_descs: set[str] = set()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        desc = kwargs.get("desc") or getattr(self, "desc", "")
        if (
            self.emitter is not None
            and isinstance(desc, str)
            and desc.endswith("(P2P)")
        ):
            self.emitter(
                "p2p_bar_created",
                desc=desc,
                initial=int(kwargs.get("initial", 0) or 0),
                total=int(kwargs.get("total", 0) or 0),
            )

    def update(self, n=1):
        result = super().update(n)
        desc = getattr(self, "desc", "")
        if (
            self.emitter is not None
            and self.interrupt_on_partial
            and isinstance(desc, str)
            and desc.endswith("(P2P)")
            and desc not in self._interrupted_descs
            and int(getattr(self, "n", 0) or 0) >= int(self.partial_threshold or 0)
        ):
            current = int(getattr(self, "n", 0) or 0)
            self._interrupted_descs.add(desc)
            self.emitter("p2p_partial_observed", desc=desc, size=current)
            self.emitter("self_interrupt_requested", mode="p2p", size=current)
            raise KeyboardInterrupt("p2p partial threshold reached")
        return result


class _PartialStateMonitor(threading.Thread):
    def __init__(
        self,
        *,
        emitter,
        p2p_root: str,
        fastresume_dir: str,
        partial_threshold: int,
        interrupt_on_partial: bool,
        repo_id: str,
        revision: str,
        cache_dir: str | None = None,
        local_dir: str | None = None,
    ) -> None:
        super().__init__(daemon=True, name="resume-partial-monitor")
        self._emit = emitter
        self._p2p_root = p2p_root
        self._fastresume_dir = fastresume_dir
        self._partial_threshold = partial_threshold
        self._interrupt_on_partial = interrupt_on_partial
        self._repo_id = repo_id
        self._revision = revision
        self._cache_dir = self._normalize_root(cache_dir)
        self._local_dir = self._normalize_root(local_dir)
        self._stop_event = threading.Event()
        self._partial_emitted = False
        self._fastresume_seen: set[str] = set()
        self._interrupt_sent = False
        self._seen_sessions: set[tuple[str, str | None, str | None]] = set()

    @staticmethod
    def _normalize_root(root: str | None) -> str | None:
        if not root:
            return None
        return os.path.realpath(os.path.abspath(os.path.expanduser(root)))

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._poll_once()
            self._stop_event.wait(0.2)

    def _poll_once(self) -> None:
        contexts = self._get_matching_contexts()
        for ctx in contexts:
            session_key = (
                getattr(ctx, "revision", ""),
                self._normalize_root(getattr(ctx, "cache_dir", None)),
                self._normalize_root(getattr(ctx, "local_dir", None)),
            )
            if session_key not in self._seen_sessions:
                self._seen_sessions.add(session_key)
                self._emit(
                    "p2p_session_observed",
                    repo_id=getattr(ctx, "repo_id", None),
                    revision=getattr(ctx, "revision", None),
                    cache_dir=getattr(ctx, "cache_dir", None),
                    local_dir=getattr(ctx, "local_dir", None),
                )

        if not self._partial_emitted:
            size = self._current_p2p_progress(contexts)
            if size >= self._partial_threshold:
                self._partial_emitted = True
                self._emit(
                    "p2p_partial_observed",
                    root=self._p2p_root,
                    size=size,
                )
                if self._interrupt_on_partial and not self._interrupt_sent:
                    self._interrupt_sent = True
                    self._emit("self_interrupt_requested", mode="p2p", size=size)
                    os.kill(os.getpid(), signal.SIGINT)

        if os.path.isdir(self._fastresume_dir):
            for path in glob.glob(os.path.join(self._fastresume_dir, "*.fastresume")):
                if path in self._fastresume_seen:
                    continue
                self._fastresume_seen.add(path)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    size = 0
                self._emit("fastresume_observed", path=path, size=size)

    def _get_matching_contexts(self):
        try:
            from llmpt.p2p_batch import P2PBatchManager

            manager = P2PBatchManager()
            with manager._lock:
                contexts = [
                    ctx
                    for ctx in manager.sessions.values()
                    if self._context_matches(ctx)
                ]
        except Exception:
            contexts = []
        return contexts

    def _context_matches(self, ctx) -> bool:
        if getattr(ctx, "repo_id", None) != self._repo_id:
            return False
        if getattr(ctx, "session_mode", None) != "on_demand":
            return False

        ctx_cache_dir = self._normalize_root(getattr(ctx, "cache_dir", None))
        ctx_local_dir = self._normalize_root(getattr(ctx, "local_dir", None))

        if self._local_dir is not None:
            return ctx_local_dir == self._local_dir
        if self._cache_dir is not None:
            return ctx_cache_dir == self._cache_dir and ctx_local_dir is None
        return ctx_local_dir is None

    def _current_p2p_progress(self, contexts) -> int:
        best_progress = 0
        for ctx in contexts:
            try:
                progresses = ctx.get_file_progress(verified_only=False)
                if progresses:
                    best_progress = max(best_progress, max(int(v or 0) for v in progresses))
            except Exception:
                pass
            try:
                stats = ctx.get_p2p_stats()
                best_progress = max(best_progress, int(stats.get("total_payload_download", 0) or 0))
            except Exception:
                pass
            try:
                with ctx.lock:
                    handle = ctx.handle
                if handle and handle.is_valid():
                    status = handle.status()
                    best_progress = max(best_progress, int(getattr(status, "total_payload_download", 0) or 0))
            except Exception:
                pass

        if best_progress > 0:
            return best_progress
        return _dir_size(self._p2p_root)


def _install_interrupt_handlers() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _compute_p2p_root(args: argparse.Namespace) -> str:
    if args.local_dir:
        return os.path.join(args.local_dir, ".cache", "huggingface", "p2p_root")
    if args.cache_dir:
        return os.path.join(args.cache_dir, "p2p_root")
    return os.path.expanduser("~/.cache/huggingface/hub/p2p_root")


def _patch_http_get(emitter, *, partial_threshold: int, interrupt_on_partial: bool) -> None:
    original_http_get = patch_module._original_http_get

    class _InterruptingTempFile:
        def __init__(self, raw_file) -> None:
            self._raw_file = raw_file
            self._partial_emitted = False
            self._interrupt_sent = False

        @property
        def name(self):
            return self._raw_file.name

        def write(self, data):
            try:
                current_size = self._raw_file.tell()
            except OSError:
                current_size = os.path.getsize(self._raw_file.name)

            if (
                interrupt_on_partial
                and not self._interrupt_sent
                and current_size < partial_threshold
                and current_size + len(data) >= partial_threshold
            ):
                cutoff = partial_threshold - current_size
                if cutoff > 0:
                    self._raw_file.write(data[:cutoff])
                self._raw_file.flush()
                self._partial_emitted = True
                self._interrupt_sent = True
                emitter(
                    "http_partial_observed",
                    temp_file=self._raw_file.name,
                    size=partial_threshold,
                )
                emitter(
                    "self_interrupt_requested",
                    mode="http",
                    temp_file=self._raw_file.name,
                    size=partial_threshold,
                )
                os.kill(os.getpid(), signal.SIGINT)
                return cutoff

            written = self._raw_file.write(data)
            if not self._partial_emitted:
                try:
                    current_size = self._raw_file.tell()
                except OSError:
                    current_size = os.path.getsize(self._raw_file.name)
                if current_size >= partial_threshold:
                    self._partial_emitted = True
                    emitter(
                        "http_partial_observed",
                        temp_file=self._raw_file.name,
                        size=current_size,
                    )
            return written

        def __getattr__(self, name):
            return getattr(self._raw_file, name)

    def _recording_http_get(url: str, temp_file, **kwargs):
        emitter(
            "http_get",
            url=url,
            temp_file=getattr(temp_file, "name", None),
            resume_size=int(kwargs.get("resume_size", 0) or 0),
        )
        wrapped_temp_file = _InterruptingTempFile(temp_file)
        return original_http_get(url, wrapped_temp_file, **kwargs)

    patch_module._original_http_get = _recording_http_get


def _patch_fastresume_loading(emitter) -> None:
    original_loader = torrent_init._load_fastresume

    def _wrapped_loader(params, info, save_path: str, path: str, repo_id: str):
        if os.path.exists(path):
            emitter("fastresume_load_attempt", path=path, repo_id=repo_id)
        result = original_loader(params, info, save_path, path, repo_id)
        if os.path.exists(path):
            emitter("fastresume_loaded", path=path, repo_id=repo_id)
        return result

    torrent_init._load_fastresume = _wrapped_loader


def _patch_final_checkpoint(emitter) -> None:
    original_checkpoint = p2p_batch_module.P2PBatchManager._checkpoint_on_demand_session

    def _wrapped_checkpoint(self, ctx, *args, **kwargs):
        fastresume_path = getattr(ctx, "fastresume_path", None)
        baseline_mtime_ns = None
        if fastresume_path and os.path.exists(fastresume_path):
            try:
                baseline_mtime_ns = os.stat(fastresume_path).st_mtime_ns
            except OSError:
                baseline_mtime_ns = None

        emitter(
            "final_checkpoint_attempt",
            repo_id=getattr(ctx, "repo_id", None),
            revision=getattr(ctx, "revision", None),
            cache_dir=getattr(ctx, "cache_dir", None),
            local_dir=getattr(ctx, "local_dir", None),
            fastresume_path=fastresume_path,
            baseline_mtime_ns=baseline_mtime_ns,
        )

        try:
            return original_checkpoint(self, ctx, *args, **kwargs)
        finally:
            current_mtime_ns = None
            current_exists = False
            if fastresume_path and os.path.exists(fastresume_path):
                current_exists = True
                try:
                    current_mtime_ns = os.stat(fastresume_path).st_mtime_ns
                except OSError:
                    current_mtime_ns = None
            emitter(
                "final_checkpoint_done",
                repo_id=getattr(ctx, "repo_id", None),
                revision=getattr(ctx, "revision", None),
                cache_dir=getattr(ctx, "cache_dir", None),
                local_dir=getattr(ctx, "local_dir", None),
                fastresume_path=fastresume_path,
                baseline_mtime_ns=baseline_mtime_ns,
                current_mtime_ns=current_mtime_ns,
                current_exists=current_exists,
            )

    p2p_batch_module.P2PBatchManager._checkpoint_on_demand_session = _wrapped_checkpoint


def _patch_p2p_init_progress(emitter, *, interrupt_on_partial: bool, p2p_download_limit: int) -> None:
    original_init = session_context_module.SessionContext._init_torrent

    def _wrapped_init(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        if result and getattr(self, "session_mode", None) == "on_demand":
            if p2p_download_limit > 0:
                try:
                    self.handle.set_download_limit(p2p_download_limit)
                except Exception:
                    pass
            progress = 0
            try:
                progresses = self.get_file_progress(verified_only=False)
                if progresses:
                    progress = max(int(v or 0) for v in progresses)
            except Exception:
                progress = 0
            emitter(
                "p2p_init_progress",
                repo_id=self.repo_id,
                revision=self.revision,
                cache_dir=self.cache_dir,
                local_dir=self.local_dir,
                size=progress,
            )
        return result

    session_context_module.SessionContext._init_torrent = _wrapped_init


def _patch_graceful_p2p_interrupt(emitter, *, interrupt_on_partial: bool, partial_threshold: int) -> None:
    if not interrupt_on_partial:
        return

    original_formatter = session_context_module._format_live_transfer_postfix
    interrupted = False

    def _wrapped_formatter(stats):
        nonlocal interrupted
        total_payload_download = 0
        if stats:
            try:
                total_payload_download = int(stats.get("total_payload_download", 0) or 0)
            except Exception:
                total_payload_download = 0
        if not interrupted and total_payload_download >= partial_threshold:
            interrupted = True
            emitter("p2p_partial_observed", size=total_payload_download, source="main_thread")
            emitter(
                "self_interrupt_requested",
                mode="p2p",
                size=total_payload_download,
                source="main_thread",
            )
            raise KeyboardInterrupt("p2p partial threshold reached")
        return original_formatter(stats)

    session_context_module._format_live_transfer_postfix = _wrapped_formatter


def _force_enable_hf_progress_bars() -> None:
    hf_utils.are_progress_bars_disabled = lambda: False


def main() -> int:
    args = _parse_args()
    emit = _make_emitter(args.events_file)
    _install_interrupt_handlers()

    fastresume_dir = os.path.expanduser("~/.cache/llmpt/p2p_resume")
    p2p_root = _compute_p2p_root(args)

    emit(
        "start",
        repo_id=args.repo_id,
        revision=args.revision,
        repo_type=args.repo_type,
        allow_pattern=args.allow_pattern,
        tracker_url=args.tracker_url,
        fastresume_dir=fastresume_dir,
        p2p_root=p2p_root,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
    )

    monitor = _PartialStateMonitor(
        emitter=emit,
        p2p_root=p2p_root,
        fastresume_dir=fastresume_dir,
        partial_threshold=args.partial_threshold,
        interrupt_on_partial=False,
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
    )

    try:
        llmpt.enable_p2p(
            tracker_url=args.tracker_url,
            timeout=args.timeout,
            webseed=args.webseed,
            verbose=False,
        )
        _patch_http_get(
            emit,
            partial_threshold=args.partial_threshold,
            interrupt_on_partial=args.interrupt_on_partial,
        )
        _force_enable_hf_progress_bars()
        _patch_fastresume_loading(emit)
        _patch_final_checkpoint(emit)
        _patch_p2p_init_progress(
            emit,
            interrupt_on_partial=args.interrupt_on_partial,
            p2p_download_limit=args.p2p_download_limit,
        )
        _patch_graceful_p2p_interrupt(
            emit,
            interrupt_on_partial=args.interrupt_on_partial,
            partial_threshold=args.partial_threshold,
        )
        _RecordingTqdm.emitter = emit
        _RecordingTqdm.interrupt_on_partial = args.interrupt_on_partial
        _RecordingTqdm.partial_threshold = args.partial_threshold
        _RecordingTqdm._interrupted_descs = set()
        monitor.start()

        kwargs: dict[str, object] = {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "allow_patterns": [args.allow_pattern],
            "local_files_only": False,
            "tqdm_class": _RecordingTqdm,
        }
        if args.repo_type != "model":
            kwargs["repo_type"] = args.repo_type
        if args.local_dir:
            kwargs["local_dir"] = args.local_dir
        if args.cache_dir:
            kwargs["cache_dir"] = args.cache_dir

        emit("download_invoked", kwargs=kwargs)
        result = snapshot_download(**kwargs)
        emit("completed", path=result)
        llmpt.shutdown()
        return 0
    except KeyboardInterrupt as exc:
        emit("interrupted", reason=str(exc))
        llmpt.shutdown()
        return 130
    except BaseException as exc:  # noqa: BLE001
        emit("error", error=repr(exc))
        llmpt.shutdown()
        raise
    finally:
        monitor.stop()
        monitor.join(timeout=1.0)


if __name__ == "__main__":
    sys.exit(main())
