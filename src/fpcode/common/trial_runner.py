"""Bounded, persistent spawned workers for caller-defined empirical trials.

The parent owns scheduling and observes deadlines independently of numerical
code in workers. Initializers and evaluators must be importable, module-level
callables. Importing this module does not launch processes or import a backend.
RSS limits use sampled Linux resident memory, not an OS-enforced allocation cap;
use an outer cgroup/service limit when a strict aggregate RAM cap is required.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import signal
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrialOutcome:
    """One terminal result; ``peak_rss_bytes`` is the parent's sampled peak."""

    job: Any
    status: str
    elapsed_s: float
    result: dict[str, Any] | None
    error: str
    peak_rss_bytes: int | None
    worker_id: int
    pid: int


@dataclass
class _Worker:
    worker_id: int
    process: Any
    connection: Any
    initialized_at: float
    ready: bool = False
    job: Any = None
    busy: bool = False
    started_at: float = 0.0
    peak_rss_bytes: int | None = None


def _worker_main(connection, initializer, initargs, evaluator):
    if os.name == "posix":
        os.setsid()
    try:
        state = initializer(*initargs) if initializer is not None else None
    except BaseException:
        connection.send(("initializer_failed", traceback.format_exc()[-6000:]))
        connection.close()
        return
    connection.send(("ready", None))
    try:
        while True:
            command, job = connection.recv()
            if command == "close":
                return
            try:
                result = evaluator(state, job)
                if not isinstance(result, dict):
                    raise TypeError("trial evaluator must return a JSON-safe dict")
                # Reject NaN/inf and unsupported objects before using the pipe.
                encoded = json.dumps(result, allow_nan=False)
                connection.send(("result", json.loads(encoded)))
            except Exception:
                connection.send(("failed", traceback.format_exc()[-6000:]))
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


def _rss_bytes(pid):
    """Read Linux process RSS without introducing a numerical dependency."""
    try:
        resident = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
        return resident * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, ProcessLookupError):
        return None


def _terminate(worker, grace_s):
    process = worker.process
    if process.is_alive():
        try:
            if os.name == "posix" and os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        process.join(grace_s)
        if process.is_alive():
            try:
                if os.name == "posix" and os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.join(max(1.0, grace_s))
    else:
        process.join()
    worker.connection.close()
    if process.is_alive():
        raise RuntimeError(f"worker {process.pid} could not be terminated")
    process.close()


def _positive_real(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def run_trials(
    jobs: Iterable[Any],
    evaluator: Callable[[Any, Any], dict[str, Any]],
    *,
    initializer: Callable[..., Any] | None = None,
    initargs: tuple[Any, ...] = (),
    workers: int = 1,
    timeout_s: float = 600.0,
    initializer_timeout_s: float = 120.0,
    max_worker_rss_bytes: int | None = None,
    stop_file: str | Path | None = None,
    stop_requested: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
    poll_interval_s: float = 0.1,
    termination_grace_s: float = 0.25,
    on_start: Callable[[Any, int, int], None] | None = None,
    can_schedule: Callable[[], bool] | None = None,
) -> Iterator[TrialOutcome]:
    """Yield outcomes from persistent workers, replacing crashed/timed-out ones.

    Jobs are consumed lazily only when a worker is ready. ``on_start`` runs in
    the parent immediately after dispatch, allowing the caller to persist job
    identity and worker PID. Outcomes are unordered. A trial timeout includes
    dispatch/transport; initialization has its own timeout. STOP/deadline checks
    abort active jobs with ``STOPPED`` outcomes and consume no further jobs.
    Closing the iterator (including an exception in a parent callback) terminates
    workers. Callers stopping iteration early should explicitly ``close()`` it.
    A false ``can_schedule`` pauses new dispatch without consuming jobs; active
    workers, timeouts, STOP and the deadline remain monitored. The callback must
    return promptly rather than sleeping or waiting for resources itself.

    Initializer failure/timeout raises ``RuntimeError`` after cleanup. Trial
    outcomes use ``SUCCESS``, ``FAILED``, ``TIMEOUT``, ``CRASHED``,
    ``MEMORY_LIMIT`` or ``STOPPED``. RSS is sampled once per poll; a hard service
    memory limit should cover workers and parent together. Numerical thread
    limits must be configured by the caller before numerical imports/spawning.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    timeout_s = _positive_real("timeout_s", timeout_s)
    initializer_timeout_s = _positive_real("initializer_timeout_s", initializer_timeout_s)
    poll_interval_s = _positive_real("poll_interval_s", poll_interval_s)
    termination_grace_s = _positive_real("termination_grace_s", termination_grace_s)
    if max_worker_rss_bytes is not None:
        if isinstance(max_worker_rss_bytes, bool) or not isinstance(max_worker_rss_bytes, int) or max_worker_rss_bytes < 1:
            raise ValueError("max_worker_rss_bytes must be a positive integer")
        if not Path("/proc/self/statm").exists():
            raise RuntimeError("sampled RSS limits require Linux /proc")
    if deadline_monotonic is not None:
        if isinstance(deadline_monotonic, bool) or not math.isfinite(deadline_monotonic):
            raise ValueError("deadline_monotonic must be finite")
    stop_path = Path(stop_file) if stop_file is not None else None

    def stopping():
        return ((stop_path is not None and stop_path.exists())
                or (stop_requested is not None and stop_requested())
                or (deadline_monotonic is not None and time.monotonic() >= deadline_monotonic))

    if stopping():
        return
    context = mp.get_context("spawn")
    active: list[_Worker] = []
    iterator = iter(jobs)
    exhausted = False

    def start(worker_id):
        parent, child = context.Pipe()
        process = context.Process(target=_worker_main, args=(child, initializer, initargs, evaluator))
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        return _Worker(worker_id, process, parent, time.monotonic())

    def outcome(worker, status, error="", result=None):
        return TrialOutcome(worker.job, status, time.monotonic() - worker.started_at,
                            result, error, worker.peak_rss_bytes, worker.worker_id, worker.process.pid)

    try:
        for worker_id in range(workers):
            active.append(start(worker_id))
        while active:
            if stopping():
                stopped = [outcome(worker, "STOPPED", "stop requested or run deadline reached")
                           for worker in active if worker.busy]
                for worker in active:
                    _terminate(worker, termination_grace_s)
                active.clear()
                yield from stopped
                return
            for worker in list(active):
                now = time.monotonic()
                rss = _rss_bytes(worker.process.pid) if Path("/proc/self/statm").exists() else None
                if rss is not None:
                    worker.peak_rss_bytes = max(worker.peak_rss_bytes or 0, rss)
                status, error, result = None, "", None
                if max_worker_rss_bytes is not None and rss is not None and rss > max_worker_rss_bytes:
                    if not worker.ready:
                        raise RuntimeError(f"worker {worker.worker_id} initializer exceeded RSS limit")
                    status, error = "MEMORY_LIMIT", f"sampled worker RSS {rss} exceeds {max_worker_rss_bytes} bytes"
                elif worker.busy and now - worker.started_at >= timeout_s:
                    status, error = "TIMEOUT", f"trial exceeded {timeout_s:g} seconds"
                elif worker.connection.poll():
                    try:
                        kind, payload = worker.connection.recv()
                    except (EOFError, ConnectionResetError):
                        kind, payload = "crashed", f"worker exited with code {worker.process.exitcode}"
                    if kind == "initializer_failed":
                        raise RuntimeError(f"worker {worker.worker_id} initializer failed:\n{payload}")
                    if kind == "ready":
                        worker.ready = True
                    elif kind == "result":
                        status, result = "SUCCESS", payload
                    elif kind == "failed":
                        status, error = "FAILED", payload
                    elif kind == "crashed":
                        status, error = "CRASHED", payload
                    else:
                        raise RuntimeError(f"unknown worker message: {kind}")
                elif not worker.process.is_alive():
                    status, error = "CRASHED", f"worker exited with code {worker.process.exitcode}"
                elif not worker.ready and now - worker.initialized_at >= initializer_timeout_s:
                    raise RuntimeError(f"worker {worker.worker_id} initializer exceeded {initializer_timeout_s:g} seconds")

                if status is not None:
                    if not worker.ready:
                        raise RuntimeError(f"worker {worker.worker_id} exited during initialization: {error}")
                    completed = outcome(worker, status, error, result) if worker.busy else None
                    worker.busy, worker.job = False, None
                    if status not in ("SUCCESS", "FAILED"):
                        _terminate(worker, termination_grace_s)
                        active.remove(worker)
                        if not exhausted and not stopping():
                            active.append(start(worker.worker_id))
                    if completed is not None:
                        yield completed
                    continue
                if (worker.ready and not worker.busy and not exhausted and not stopping()
                        and (can_schedule is None or can_schedule())):
                    try:
                        job = next(iterator)
                    except StopIteration:
                        exhausted = True
                    else:
                        worker.job, worker.busy, worker.started_at = job, True, time.monotonic()
                        worker.peak_rss_bytes = rss
                        worker.connection.send(("trial", job))
                        if on_start is not None:
                            on_start(job, worker.worker_id, worker.process.pid)
            if exhausted and not any(worker.busy for worker in active):
                return
            time.sleep(poll_interval_s)
    finally:
        for worker in active:
            _terminate(worker, termination_grace_s)
