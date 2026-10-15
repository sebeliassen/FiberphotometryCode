"""Independent process-supervision checks; no empirical models or datasets."""

import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from fpcode.common.trial_runner import run_trials


def _initialize(offset=0):
    return {"offset": offset, "calls": 0}


def _evaluate(state, job):
    state["calls"] += 1
    if job.get("crash"):
        os._exit(17)
    if job.get("sleep"):
        time.sleep(job["sleep"])
    if job.get("fail"):
        raise ValueError("deliberate trial failure")
    if job.get("stop"):
        Path(job["stop"]).touch()
        time.sleep(10)
    return {"value": job["value"] + state["offset"], "calls": state["calls"]}


def _bad_initializer():
    raise ValueError("deliberate initializer failure")


def _event_initializer(event):
    return event


def _barrier_evaluate(event, job):
    if not event.wait(5):
        raise TimeoutError("all four workers should receive a job concurrently")
    return {"value": job}


def _options(**overrides):
    return dict(initializer=_initialize, initargs=(7,), workers=1,
                timeout_s=2, initializer_timeout_s=10, poll_interval_s=.02,
                termination_grace_s=.05, **overrides)


def test_successful_jobs_reuse_worker_state_and_parent_callback():
    starts = []
    jobs = [{"value": 2}, {"value": 5}]
    results = list(run_trials(jobs, _evaluate, on_start=lambda job, wid, pid: starts.append((job, wid, pid)),
                              **_options()))
    assert [r.status for r in results] == ["SUCCESS", "SUCCESS"]
    assert [r.result for r in results] == [{"value": 9, "calls": 1}, {"value": 12, "calls": 2}]
    assert [r.job for r in results] == jobs
    assert results[0].pid == results[1].pid == starts[0][2] == starts[1][2]
    assert all(r.elapsed_s >= 0 and r.error == "" for r in results)
    if Path("/proc/self/statm").exists():
        assert all(r.peak_rss_bytes > 0 for r in results)


@pytest.mark.parametrize("bad_job,expected", [({"value": 0, "sleep": 10}, "TIMEOUT"),
                                            ({"value": 0, "crash": True}, "CRASHED")])
def test_timeout_or_crash_replaces_worker_and_next_job_succeeds(bad_job, expected):
    options = _options()
    options["timeout_s"] = .15
    results = list(run_trials([bad_job, {"value": 3}], _evaluate, **options))
    assert [r.status for r in results] == [expected, "SUCCESS"]
    assert results[1].result == {"value": 10, "calls": 1}
    assert results[0].pid != results[1].pid
    assert results[0].error
    if expected == "TIMEOUT":
        assert results[0].elapsed_s < 1.5


def test_trial_failure_is_recorded_without_losing_next_job():
    results = list(run_trials([{"value": 0, "fail": True}, {"value": 2}], _evaluate, **_options()))
    assert [r.status for r in results] == ["FAILED", "SUCCESS"]
    assert "deliberate trial failure" in results[0].error
    assert results[1].result == {"value": 9, "calls": 2}


def test_initializer_failure_is_visible_and_workers_cleaned_up():
    options = _options()
    options.update(initializer=_bad_initializer, initargs=())
    with pytest.raises(RuntimeError, match="deliberate initializer failure"):
        list(run_trials([{"value": 0}], _evaluate, **options))


def test_stop_file_aborts_active_job_and_does_not_consume_more(tmp_path):
    consumed = []
    stop = tmp_path / "STOP"

    def jobs():
        for value in range(3):
            consumed.append(value)
            yield {"value": value, "stop": str(stop)}

    results = list(run_trials(jobs(), _evaluate, stop_file=stop, **_options()))
    assert consumed == [0]
    assert len(results) == 1 and results[0].status == "STOPPED"
    assert results[0].elapsed_s < 1.5
    with pytest.raises(ProcessLookupError):
        os.kill(results[0].pid, 0)


def test_expired_deadline_consumes_nothing():
    def unexpected():
        raise AssertionError("expired run must not consume jobs")
        yield

    assert list(run_trials(unexpected(), _evaluate, deadline_monotonic=time.monotonic() - 1, **_options())) == []


def test_four_workers_are_bounded_and_concurrent():
    event = mp.get_context("spawn").Event()
    starts = []

    def on_start(job, worker_id, pid):
        starts.append((worker_id, pid))
        if len(starts) == 4:
            event.set()

    results = list(run_trials(range(4), _barrier_evaluate, initializer=_event_initializer,
                              initargs=(event,), workers=4, timeout_s=10, on_start=on_start,
                              poll_interval_s=.02, termination_grace_s=.05))
    assert len(results) == 4 and all(result.status == "SUCCESS" for result in results)
    assert {result.result["value"] for result in results} == set(range(4))
    assert len({pid for _, pid in starts}) == 4
    assert {worker_id for worker_id, _ in starts} == {0, 1, 2, 3}


def test_early_iterator_close_terminates_persistent_worker():
    results = run_trials([{"value": 2}, {"value": 5, "sleep": 10}], _evaluate, **_options())
    first = next(results)
    assert first.status == "SUCCESS"
    results.close()
    with pytest.raises(ProcessLookupError):
        os.kill(first.pid, 0)


def test_schedule_gate_pauses_without_consuming_jobs_then_resumes():
    consumed, checks = [], []

    def jobs():
        consumed.append(2)
        yield {"value": 2}

    def can_schedule():
        checks.append(len(consumed))
        return len(checks) > 3

    results = list(run_trials(jobs(), _evaluate, can_schedule=can_schedule, **_options()))
    assert checks[:3] == [0, 0, 0]
    assert consumed == [2]
    assert len(results) == 1 and results[0].status == "SUCCESS"


def test_schedule_pause_does_not_disable_active_trial_watchdog():
    consumed, checks = [], []

    def jobs():
        for value in range(2):
            consumed.append(value)
            yield {"value": value, "sleep": 10}

    def can_schedule():
        checks.append(True)
        return len(checks) == 1

    options = _options()
    options["timeout_s"] = .15
    runner = run_trials(jobs(), _evaluate, can_schedule=can_schedule, **options)
    try:
        result = next(runner)
        assert result.status == "TIMEOUT" and result.elapsed_s < 1.5
        assert consumed == [0]
    finally:
        runner.close()
