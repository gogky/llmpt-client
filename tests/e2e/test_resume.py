"""
End-to-end tests for breakpoint resume across HTTP fallback and P2P transfers.

These tests launch a child Python process that performs a real download,
interrupt it after partial progress, then start the same download again with
the same storage roots and verify resume semantics.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import pytest


REPO_ID = os.environ.get("RESUME_REPO_ID", "prajjwal1/bert-tiny")
ALLOW_PATTERN = os.environ.get("RESUME_ALLOW_PATTERN", "pytorch_model.bin")
TRACKER_URL = os.environ.get("TRACKER_URL", "http://118.195.159.242")
INVALID_TRACKER_URL = os.environ.get("INVALID_TRACKER_URL", "http://127.0.0.1:9")

SEEDER_READY_SIGNAL = "/app/.resume_seeder_ready"
SEEDER_DONE_SIGNAL = "/app/.resume_e2e_done"
PARTIAL_THRESHOLD = int(os.environ.get("RESUME_PARTIAL_THRESHOLD", str(128 * 1024)))


class _EventStream:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.pending: list[dict] = []

    def read_new(self) -> list[dict]:
        if self.pending:
            events = self.pending
            self.pending = []
            return events
        if not self.path.exists():
            return []
        events = []
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self.offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
            self.offset = f.tell()
        return events

    def push_back(self, events: list[dict]) -> None:
        if not events:
            return
        self.pending = events + self.pending


def _wait_for_event(
    stream: _EventStream,
    predicate: Callable[[dict], bool],
    *,
    timeout: int,
    label: str,
) -> dict:
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        events = stream.read_new()
        for idx, event in enumerate(events):
            seen.append(event)
            if predicate(event):
                stream.push_back(events[idx + 1 :])
                return event
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for event: {label}. Seen tail={seen[-5:]}")


def _wait_for_signal(path: str, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return
        time.sleep(1)
    raise AssertionError(f"Timed out waiting for signal: {path}")


def _wait_for_file_growth(path: str, *, min_bytes: int, timeout: int) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            if size >= min_bytes:
                return size
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for file growth: {path}")


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


def _cleanup_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _prepare_layout(tmp_path: Path, name: str) -> dict[str, Path]:
    base = tmp_path / name
    home = base / "home"
    hf_home = base / "hf_home"
    local_dir = base / "local_dir"
    cache_dir = base / "cache_dir"
    events = base / "events.jsonl"
    log = base / "harness.log"

    for path in (base, home, hf_home):
        path.mkdir(parents=True, exist_ok=True)
    if events.exists():
        events.unlink()

    return {
        "base": base,
        "home": home,
        "hf_home": hf_home,
        "local_dir": local_dir,
        "cache_dir": cache_dir,
        "events": events,
        "log": log,
        "fastresume_dir": home / ".cache" / "llmpt" / "p2p_resume",
        "default_p2p_root": home / ".cache" / "huggingface" / "hub" / "p2p_root",
    }


def _run_paths(layout: dict[str, Path], name: str) -> tuple[Path, Path]:
    return layout["base"] / f"{name}.events.jsonl", layout["base"] / f"{name}.harness.log"


def _spawn_harness(
    layout: dict[str, Path],
    *,
    tracker_url: str,
    local_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    webseed: bool = False,
    interrupt_on_partial: bool = False,
    p2p_download_limit: int = 0,
    events_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    partial_threshold: Optional[int] = None,
) -> tuple[subprocess.Popen, _EventStream]:
    env = os.environ.copy()
    env["HOME"] = str(layout["home"])
    env["HF_HOME"] = str(layout["hf_home"])
    env["PYTHONPATH"] = os.getcwd()
    events_path = events_path or layout["events"]
    log_path = log_path or layout["log"]
    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if events_path.exists():
        events_path.unlink()
    if log_path.exists():
        log_path.unlink()

    cmd = [
        sys.executable,
        os.path.join(os.getcwd(), "tests/e2e/resume_harness.py"),
        "--repo-id",
        REPO_ID,
        "--allow-pattern",
        ALLOW_PATTERN,
        "--tracker-url",
        tracker_url,
        "--events-file",
        str(events_path),
        "--partial-threshold",
        str(partial_threshold or PARTIAL_THRESHOLD),
        "--timeout",
        "180",
    ]
    if local_dir is not None:
        cmd.extend(["--local-dir", str(local_dir)])
    if cache_dir is not None:
        cmd.extend(["--cache-dir", str(cache_dir)])
    if webseed:
        cmd.append("--webseed")
    if interrupt_on_partial:
        cmd.append("--interrupt-on-partial")
    if p2p_download_limit > 0:
        cmd.extend(["--p2p-download-limit", str(p2p_download_limit)])

    log_handle = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._llmpt_log_handle = log_handle  # type: ignore[attr-defined]
    proc._llmpt_log_path = str(log_path)  # type: ignore[attr-defined]
    return proc, _EventStream(events_path)


def _read_all_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _run_graceful_p2p_resume(
    tmp_path: Path,
    *,
    name: str,
    local_dir: bool = False,
) -> None:
    _wait_for_signal(SEEDER_READY_SIGNAL)
    layout = _prepare_layout(tmp_path, name)
    events1, log1 = _run_paths(layout, f"{name}_run1")
    events2, log2 = _run_paths(layout, f"{name}_run2")
    run_local_dir = layout["local_dir"] if local_dir else None

    proc1, stream1 = _spawn_harness(
        layout,
        tracker_url=TRACKER_URL,
        local_dir=run_local_dir,
        webseed=True,
        p2p_download_limit=8 * 1024,
        partial_threshold=64 * 1024,
        interrupt_on_partial=True,
        events_path=events1,
        log_path=log1,
    )
    _wait_for_event(
        stream1,
        lambda e: e["event"] == "self_interrupt_requested" and e.get("mode") == "p2p",
        timeout=300,
        label=f"{name} self interrupt requested",
    )
    first_rc = _wait_for_exit(proc1, timeout=180)
    assert first_rc == 130, f"Expected graceful interruption exit code 130 for {name}"

    first_run_events = _read_all_events(events1)
    assert any(e["event"] == "interrupted" for e in first_run_events), (
        f"Expected graceful interruption event in first run for {name}"
    )
    checkpoint_events = [e for e in first_run_events if e["event"] == "final_checkpoint_done"]
    assert checkpoint_events, f"Expected final checkpoint during graceful interruption for {name}"
    assert checkpoint_events[-1]["current_exists"], (
        f"Expected fastresume file to remain after graceful interruption for {name}"
    )
    fastresume_path = checkpoint_events[-1]["fastresume_path"]
    assert fastresume_path and os.path.exists(fastresume_path)

    proc2, stream2 = _spawn_harness(
        layout,
        tracker_url=TRACKER_URL,
        local_dir=run_local_dir,
        webseed=True,
        events_path=events2,
        log_path=log2,
    )
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "fastresume_loaded",
        timeout=180,
        label=f"{name} fastresume load",
    )
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "completed",
        timeout=1800,
        label=f"{name} p2p completion",
    )
    _assert_success(proc2)


def _interrupt_process(proc: subprocess.Popen, *, sig: int = signal.SIGINT) -> int:
    if proc.poll() is None:
        proc.send_signal(sig)
    try:
        rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait(timeout=30)
    log_handle = getattr(proc, "_llmpt_log_handle", None)
    if log_handle is not None and not log_handle.closed:
        log_handle.close()
    return rc


def _wait_for_exit(proc: subprocess.Popen, *, timeout: int = 120) -> int:
    rc = proc.wait(timeout=timeout)
    log_handle = getattr(proc, "_llmpt_log_handle", None)
    if log_handle is not None and not log_handle.closed:
        log_handle.close()
    return rc


def _assert_success(proc: subprocess.Popen, *, timeout: int = 1800) -> None:
    proc.wait(timeout=timeout)
    log_handle = getattr(proc, "_llmpt_log_handle", None)
    if log_handle is not None and not log_handle.closed:
        log_handle.close()
    output = ""
    log_path = getattr(proc, "_llmpt_log_path", None)
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            output = f.read()
    if proc.returncode != 0:
        raise AssertionError(
            f"Harness failed with exit code {proc.returncode}\n--- output ---\n{output}"
        )


@pytest.fixture(scope="session", autouse=True)
def _signal_seeder_completion():
    if os.path.exists(SEEDER_DONE_SIGNAL):
        os.unlink(SEEDER_DONE_SIGNAL)
    yield
    with open(SEEDER_DONE_SIGNAL, "w", encoding="utf-8") as f:
        f.write("done")


def test_http_resume_default_cache_no_torrent(tmp_path):
    layout = _prepare_layout(tmp_path, "http_default")

    proc1, stream1 = _spawn_harness(
        layout,
        tracker_url=INVALID_TRACKER_URL,
        interrupt_on_partial=True,
    )
    _wait_for_event(
        stream1,
        lambda e: e["event"] == "http_get" and e.get("temp_file"),
        timeout=180,
        label="initial http_get",
    )
    partial = _wait_for_event(
        stream1,
        lambda e: e["event"] == "http_partial_observed",
        timeout=180,
        label="http partial observed",
    )
    temp_file = partial["temp_file"]
    first_rc = _wait_for_exit(proc1)
    assert first_rc != 0, "First run should be interrupted, not complete successfully"
    assert os.path.exists(temp_file), "HTTP partial file vanished after interruption"
    assert os.path.getsize(temp_file) >= int(partial["size"])

    proc2, stream2 = _spawn_harness(layout, tracker_url=INVALID_TRACKER_URL)
    resumed = _wait_for_event(
        stream2,
        lambda e: e["event"] == "http_get" and int(e.get("resume_size", 0)) > 0,
        timeout=180,
        label="resumed http_get",
    )
    assert resumed["resume_size"] > 0
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "completed",
        timeout=1800,
        label="http completion",
    )
    _assert_success(proc2)


def test_http_resume_local_dir_no_torrent(tmp_path):
    layout = _prepare_layout(tmp_path, "http_local_dir")

    proc1, stream1 = _spawn_harness(
        layout,
        tracker_url=INVALID_TRACKER_URL,
        local_dir=layout["local_dir"],
        interrupt_on_partial=True,
    )
    _wait_for_event(
        stream1,
        lambda e: e["event"] == "http_get" and e.get("temp_file"),
        timeout=180,
        label="initial local_dir http_get",
    )
    partial = _wait_for_event(
        stream1,
        lambda e: e["event"] == "http_partial_observed",
        timeout=180,
        label="local_dir http partial observed",
    )
    temp_file = partial["temp_file"]
    first_rc = _wait_for_exit(proc1)
    assert first_rc != 0
    assert os.path.exists(temp_file)
    assert os.path.getsize(temp_file) >= int(partial["size"])

    proc2, stream2 = _spawn_harness(
        layout,
        tracker_url=INVALID_TRACKER_URL,
        local_dir=layout["local_dir"],
    )
    resumed = _wait_for_event(
        stream2,
        lambda e: e["event"] == "http_get" and int(e.get("resume_size", 0)) > 0,
        timeout=180,
        label="local_dir resumed http_get",
    )
    assert resumed["resume_size"] > 0
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "completed",
        timeout=1800,
        label="local_dir http completion",
    )
    _assert_success(proc2)


def test_http_resume_cache_dir_no_torrent(tmp_path):
    layout = _prepare_layout(tmp_path, "http_cache_dir")

    proc1, stream1 = _spawn_harness(
        layout,
        tracker_url=INVALID_TRACKER_URL,
        cache_dir=layout["cache_dir"],
        interrupt_on_partial=True,
    )
    _wait_for_event(
        stream1,
        lambda e: e["event"] == "http_get" and e.get("temp_file"),
        timeout=180,
        label="initial cache_dir http_get",
    )
    partial = _wait_for_event(
        stream1,
        lambda e: e["event"] == "http_partial_observed",
        timeout=180,
        label="cache_dir http partial observed",
    )
    temp_file = partial["temp_file"]
    first_rc = _wait_for_exit(proc1)
    assert first_rc != 0
    assert os.path.exists(temp_file)
    assert os.path.getsize(temp_file) >= int(partial["size"])

    proc2, stream2 = _spawn_harness(
        layout,
        tracker_url=INVALID_TRACKER_URL,
        cache_dir=layout["cache_dir"],
    )
    resumed = _wait_for_event(
        stream2,
        lambda e: e["event"] == "http_get" and int(e.get("resume_size", 0)) > 0,
        timeout=180,
        label="cache_dir resumed http_get",
    )
    assert resumed["resume_size"] > 0
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "completed",
        timeout=1800,
        label="cache_dir http completion",
    )
    _assert_success(proc2)


def test_p2p_resume_default_cache_after_interrupt(tmp_path):
    _wait_for_signal(SEEDER_READY_SIGNAL)
    layout = _prepare_layout(tmp_path, "p2p_default")

    proc1, stream1 = _spawn_harness(
        layout,
        tracker_url=TRACKER_URL,
        webseed=True,
        p2p_download_limit=32 * 1024,
    )
    fastresume = _wait_for_event(
        stream1,
        lambda e: e["event"] == "fastresume_observed",
        timeout=180,
        label="initial fastresume observed",
    )
    partial = _wait_for_event(
        stream1,
        lambda e: e["event"] == "p2p_partial_observed",
        timeout=300,
        label="initial p2p partial",
    )
    first_rc = _interrupt_process(proc1, sig=signal.SIGKILL)
    assert first_rc != 0

    assert os.path.exists(fastresume["path"]), "Expected fastresume file after interrupted P2P download"
    assert int(partial["size"]) >= PARTIAL_THRESHOLD

    proc2, stream2 = _spawn_harness(layout, tracker_url=TRACKER_URL, webseed=True)
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "fastresume_loaded",
        timeout=180,
        label="fastresume load",
    )
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "completed",
        timeout=1800,
        label="p2p completion",
    )
    _assert_success(proc2)


def test_p2p_resume_local_dir_after_interrupt(tmp_path):
    _wait_for_signal(SEEDER_READY_SIGNAL)
    layout = _prepare_layout(tmp_path, "p2p_local_dir")

    proc1, stream1 = _spawn_harness(
        layout,
        tracker_url=TRACKER_URL,
        local_dir=layout["local_dir"],
        webseed=True,
        p2p_download_limit=32 * 1024,
    )
    fastresume = _wait_for_event(
        stream1,
        lambda e: e["event"] == "fastresume_observed",
        timeout=180,
        label="initial local_dir fastresume observed",
    )
    partial = _wait_for_event(
        stream1,
        lambda e: e["event"] == "p2p_partial_observed",
        timeout=300,
        label="initial local_dir p2p partial",
    )
    first_rc = _interrupt_process(proc1, sig=signal.SIGKILL)
    assert first_rc != 0

    assert os.path.exists(fastresume["path"]), "Expected local_dir fastresume after interruption"
    assert _dir_size(str(layout["local_dir"] / ".cache" / "huggingface" / "p2p_root")) > 0, (
        "Expected preserved local_dir partial P2P data after interruption"
    )
    assert int(partial["size"]) >= PARTIAL_THRESHOLD

    proc2, stream2 = _spawn_harness(
        layout,
        tracker_url=TRACKER_URL,
        local_dir=layout["local_dir"],
        webseed=True,
    )
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "fastresume_loaded",
        timeout=180,
        label="local_dir fastresume load",
    )
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "completed",
        timeout=1800,
        label="local_dir p2p completion",
    )
    _assert_success(proc2)


def test_p2p_resume_cache_dir_after_interrupt(tmp_path):
    _wait_for_signal(SEEDER_READY_SIGNAL)
    layout = _prepare_layout(tmp_path, "p2p_cache_dir")

    proc1, stream1 = _spawn_harness(
        layout,
        tracker_url=TRACKER_URL,
        cache_dir=layout["cache_dir"],
        webseed=True,
        p2p_download_limit=32 * 1024,
    )
    fastresume = _wait_for_event(
        stream1,
        lambda e: e["event"] == "fastresume_observed",
        timeout=180,
        label="initial cache_dir fastresume observed",
    )
    partial = _wait_for_event(
        stream1,
        lambda e: e["event"] == "p2p_partial_observed",
        timeout=300,
        label="initial cache_dir p2p partial",
    )
    first_rc = _interrupt_process(proc1, sig=signal.SIGKILL)
    assert first_rc != 0

    assert os.path.exists(fastresume["path"]), "Expected cache_dir fastresume after interruption"
    assert _dir_size(str(layout["cache_dir"] / "p2p_root")) > 0, (
        "Expected preserved cache_dir partial P2P data after interruption"
    )
    assert int(partial["size"]) >= PARTIAL_THRESHOLD

    proc2, stream2 = _spawn_harness(
        layout,
        tracker_url=TRACKER_URL,
        cache_dir=layout["cache_dir"],
        webseed=True,
    )
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "fastresume_loaded",
        timeout=180,
        label="cache_dir fastresume load",
    )
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "completed",
        timeout=1800,
        label="cache_dir p2p completion",
    )
    _assert_success(proc2)


def test_p2p_resume_state_isolated_between_local_dirs(tmp_path):
    _wait_for_signal(SEEDER_READY_SIGNAL)
    layout = _prepare_layout(tmp_path, "p2p_local_dir_isolation")
    local_dir_a = layout["base"] / "local_dir_a"
    local_dir_b = layout["base"] / "local_dir_b"
    events1, log1 = _run_paths(layout, "local_dir_isolation_run1")
    events2, log2 = _run_paths(layout, "local_dir_isolation_run2")

    proc1, stream1 = _spawn_harness(
        layout,
        tracker_url=TRACKER_URL,
        local_dir=local_dir_a,
        webseed=True,
        p2p_download_limit=32 * 1024,
        events_path=events1,
        log_path=log1,
    )
    fastresume = _wait_for_event(
        stream1,
        lambda e: e["event"] == "fastresume_observed",
        timeout=180,
        label="local_dir_a fastresume observed",
    )
    _wait_for_event(
        stream1,
        lambda e: e["event"] == "p2p_partial_observed",
        timeout=300,
        label="local_dir_a partial observed",
    )
    first_rc = _interrupt_process(proc1, sig=signal.SIGKILL)
    assert first_rc != 0
    assert os.path.exists(fastresume["path"])
    assert _dir_size(str(local_dir_a / ".cache" / "huggingface" / "p2p_root")) > 0

    proc2, stream2 = _spawn_harness(
        layout,
        tracker_url=TRACKER_URL,
        local_dir=local_dir_b,
        webseed=True,
        events_path=events2,
        log_path=log2,
    )
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "completed",
        timeout=1800,
        label="local_dir_b completion",
    )
    _assert_success(proc2)

    second_run_events = _read_all_events(events2)
    assert not any(e["event"] == "fastresume_loaded" for e in second_run_events), (
        "A different local_dir should not reuse fastresume state from the previous run"
    )


def test_p2p_resume_state_isolated_between_cache_dirs(tmp_path):
    _wait_for_signal(SEEDER_READY_SIGNAL)
    layout = _prepare_layout(tmp_path, "p2p_cache_dir_isolation")
    cache_dir_a = layout["base"] / "cache_dir_a"
    cache_dir_b = layout["base"] / "cache_dir_b"
    events1, log1 = _run_paths(layout, "cache_dir_isolation_run1")
    events2, log2 = _run_paths(layout, "cache_dir_isolation_run2")

    proc1, stream1 = _spawn_harness(
        layout,
        tracker_url=TRACKER_URL,
        cache_dir=cache_dir_a,
        webseed=True,
        p2p_download_limit=32 * 1024,
        events_path=events1,
        log_path=log1,
    )
    fastresume = _wait_for_event(
        stream1,
        lambda e: e["event"] == "fastresume_observed",
        timeout=180,
        label="cache_dir_a fastresume observed",
    )
    _wait_for_event(
        stream1,
        lambda e: e["event"] == "p2p_partial_observed",
        timeout=300,
        label="cache_dir_a partial observed",
    )
    first_rc = _interrupt_process(proc1, sig=signal.SIGKILL)
    assert first_rc != 0
    assert os.path.exists(fastresume["path"])
    assert _dir_size(str(cache_dir_a / "p2p_root")) > 0

    proc2, stream2 = _spawn_harness(
        layout,
        tracker_url=TRACKER_URL,
        cache_dir=cache_dir_b,
        webseed=True,
        events_path=events2,
        log_path=log2,
    )
    _wait_for_event(
        stream2,
        lambda e: e["event"] == "completed",
        timeout=1800,
        label="cache_dir_b completion",
    )
    _assert_success(proc2)

    second_run_events = _read_all_events(events2)
    assert not any(e["event"] == "fastresume_loaded" for e in second_run_events), (
        "A different cache_dir should not reuse fastresume state from the previous run"
    )


def test_p2p_completed_download_cleans_resumable_state(tmp_path):
    _wait_for_signal(SEEDER_READY_SIGNAL)
    layout = _prepare_layout(tmp_path, "p2p_cleanup")

    proc, stream = _spawn_harness(
        layout,
        tracker_url=TRACKER_URL,
        webseed=True,
    )
    _wait_for_event(
        stream,
        lambda e: e["event"] == "completed",
        timeout=1800,
        label="cleanup completion",
    )
    _assert_success(proc)

    assert not list(layout["fastresume_dir"].glob("*.fastresume")), (
        "Completed downloads should remove persisted fastresume state"
    )
    assert _dir_size(str(layout["default_p2p_root"])) == 0, (
        "Completed downloads should clean up preserved P2P partial data"
    )


def test_p2p_resume_default_cache_after_graceful_interrupt(tmp_path):
    _run_graceful_p2p_resume(
        tmp_path,
        name="p2p_default_graceful",
    )


def test_p2p_resume_local_dir_after_graceful_interrupt(tmp_path):
    _run_graceful_p2p_resume(
        tmp_path,
        name="p2p_local_dir_graceful",
        local_dir=True,
    )
