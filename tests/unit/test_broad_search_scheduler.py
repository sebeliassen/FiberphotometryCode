"""Durable finite-search semantics on tiny non-numerical supervised jobs."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.monitoring.launch_behavior_search import build_broad_plan, freeze_broad_runtime
from studies.sequence_viz.run_broad_search import (
    ResourceGuard,
    SearchLedger,
    build_search_plan,
    digest,
    exclusive_run,
    execute_search,
    verify_snapshot,
)


def _fixture(tmp_path, *, recipes=3, folds=1):
    root = tmp_path / "code with spaces"
    source = root / "studies/sequence_viz/run_broad_search.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen source\n")
    library = root / "src/fpcode/__init__.py"
    library.parent.mkdir(parents=True)
    library.write_text("# library\n")
    npz = tmp_path / "prepared.npz"
    npz.write_bytes(b"the supervisor never reads numerical arrays")
    stat = npz.stat()
    manifest = tmp_path / "input_manifest.json"
    manifest.write_text(json.dumps(dict(schema_version=1, cohorts=[dict(
        cohort_id="fixture", folds=[[f"held_{i}"] for i in range(folds)], records=[dict(
            npz_path=str(npz), sha256=digest(npz), size=stat.st_size, mtime_ns=stat.st_mtime_ns)])])))
    output = tmp_path / "new output"
    plan = build_search_plan(manifest=manifest, output=output, code_root=root, recipes_per_cohort=recipes,
        folds=folds, replicates=1, workers=1, threads=1, per_animal=10, held_per_animal=5,
        trial_timeout_s=.2, max_worker_rss_gb=.5, min_free_disk_gb=0, min_free_ram_gb=0,
        cache_quota_gb=1, recipes=[dict(method="fixture", creator_seed=7, knob=i) for i in range(recipes)])
    return plan, root, manifest, output


def _initialize(plan):
    return plan


def _evaluate_timeout(_state, job):
    if job["base_recipe"] == 1:
        time.sleep(5)
    return {"status": "EMPTY" if job["base_recipe"] == 0 else "SUCCESS", "value": job["base_recipe"]}


def _evaluate_stop(state, _job):
    (Path(state["output"]) / "STOP").touch()
    time.sleep(5)
    return {"status": "SUCCESS"}


def _evaluate_success(_state, job):
    return {"status": "SUCCESS", "value": job["base_recipe"]}


def test_plan_is_finite_read_only_reproducible_and_uses_fixed_creator_seed(tmp_path):
    plan, root, manifest, output = _fixture(tmp_path, folds=3)
    assert not output.exists()
    assert plan["total_jobs"] == 9
    assert len({job["job_id"] for job in plan["jobs"]}) == 9
    assert {job["recipe"]["creator_seed"] for job in plan["jobs"]} == {7}
    repeated = build_search_plan(manifest=manifest, output=output, code_root=root, recipes_per_cohort=3,
        folds=3, replicates=1, workers=1, threads=1, per_animal=10, held_per_animal=5,
        trial_timeout_s=.2, max_worker_rss_gb=.5, min_free_disk_gb=0, min_free_ram_gb=0,
        cache_quota_gb=1, recipes=[dict(method="fixture", creator_seed=7, knob=i) for i in range(3)])
    assert repeated == plan
    assert not output.exists()
    assert plan["window_seed"] == 73


def test_help_and_real_cli_plan_do_not_import_numerics_or_write(tmp_path):
    _, root, manifest, output = _fixture(tmp_path)
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPATH=os.pathsep.join((str(repository / "src"), str(repository))))
    help_result = subprocess.run([sys.executable, "-B", "-m", "studies.sequence_viz.run_broad_search", "--help"],
                                 env=environment, capture_output=True, text=True, check=True)
    assert "--resume" in help_result.stdout
    script = ("import sys; names=('numpy','pandas','jax','torch','scipy'); "
              "before={name for name in names if name in sys.modules}; "
              "from studies.sequence_viz.run_broad_search import main; main(sys.argv[1:]); "
              "assert {name for name in names if name in sys.modules} == before")
    command = [sys.executable, "-B", "-c", script, "--manifest", str(manifest), "--output", str(output),
               "--code-root", str(root), "--recipes-per-cohort", "30", "--folds", "1", "--replicates", "1",
               "--workers", "1", "--threads", "1", "--min-free-disk-gb", "0", "--min-free-ram-gb", "0"]
    planned = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    assert planned.returncode == 0, planned.stderr
    assert json.loads(planned.stdout)["total_jobs"] == 30
    assert not output.exists()


def test_exclusive_writer_and_resume_never_retry_terminal_jobs(tmp_path):
    plan, _, _, output = _fixture(tmp_path)
    with exclusive_run(output):
        with pytest.raises(RuntimeError, match="another parent"):
            with exclusive_run(output):
                pass
        (output / "jobs").mkdir()
        ledger = SearchLedger(output, plan)
        first, second, _ = ledger.pending()
        ledger.start(first, 0, 12345)
        ledger.finish(SimpleNamespace(job=first, status="SUCCESS", result={"status": "EMPTY", "calls": []},
                                     elapsed_s=.5, peak_rss_bytes=123, error=""))
        ledger.start(second, 0, 12345)
        ledger.finish(SimpleNamespace(job=second, status="TIMEOUT", result=None, elapsed_s=1,
                                     peak_rss_bytes=123, error="timeout"))
        ledger.close()
        with pytest.raises(ValueError, match="explicit --resume"):
            SearchLedger(output, plan)
        resumed = SearchLedger(output, plan, resume=True)
        assert [job["base_recipe"] for job in resumed.pending()] == [2]
        assert resumed.counts() == {"EMPTY": 1, "PENDING": 1, "TIMEOUT": 1}
        resumed.close()
        with pytest.raises(ValueError, match="resume scope"):
            SearchLedger(output, dict(plan, per_animal=11), resume=True)


def test_supervised_timeout_empty_and_success_are_distinct_and_durable(tmp_path):
    plan, _, _, output = _fixture(tmp_path)
    result = execute_search(plan, evaluator=_evaluate_timeout, initializer=_initialize)
    assert result == {"state": "COMPLETED", "counts": {"EMPTY": 1, "SUCCESS": 1, "TIMEOUT": 1}}
    assert len(list((output / "jobs").glob("*.json"))) == 2
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "COMPLETED"
    assert execute_search(plan, resume=True, evaluator=_evaluate_success, initializer=_initialize) == result


def test_stop_requires_explicit_resume_and_preserves_successful_scope(tmp_path):
    plan, _, _, output = _fixture(tmp_path)
    stopped = execute_search(plan, evaluator=_evaluate_stop, initializer=_initialize)
    assert stopped["state"] == "STOPPED"
    assert stopped["counts"] == {"PENDING": 2, "STOPPED": 1}
    with pytest.raises(ValueError, match="STOP exists"):
        execute_search(plan, evaluator=_evaluate_success, initializer=_initialize)
    with pytest.raises(ValueError, match="resume scope"):
        execute_search(dict(plan, per_animal=11), resume=True,
                       evaluator=_evaluate_success, initializer=_initialize)
    assert (output / "STOP").exists(), "failed scope validation must preserve user stop"
    completed = execute_search(plan, resume=True, evaluator=_evaluate_success, initializer=_initialize)
    assert completed == {"state": "COMPLETED", "counts": {"SUCCESS": 3}}
    assert not (output / "STOP").exists()


def test_fingerprint_and_disk_cache_guards_stop_before_dispatch(tmp_path, monkeypatch):
    plan, _, _, output = _fixture(tmp_path)
    output.mkdir()
    monkeypatch.setattr("studies.sequence_viz.run_broad_search.available_ram_bytes", lambda: 0)
    ram_plan = dict(plan, min_free_ram_gb=1)
    guard = ResourceGuard(ram_plan)
    assert not guard.can_schedule() and guard.pause_reason and not guard.reason
    cache = output / "cache"
    cache.mkdir()
    (cache / "component.npz").write_bytes(b"abcd")
    quota = ResourceGuard(dict(plan, cache_quota_gb=3 / 1024 ** 3))
    assert quota.stopping() and "quota" in quota.reason
    frozen_path = Path(plan["input_snapshot"][0]["path"])
    frozen_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        verify_snapshot(plan["input_snapshot"])


def test_broad_launcher_freezes_source_and_manifest_and_preserves_old_service(tmp_path):
    _, root, manifest, output = _fixture(tmp_path)
    options = dict(code_root=root, python=sys.executable, manifest=manifest, output=output)
    plan = build_broad_plan(**options)
    assert not output.exists()
    assert "--property=Restart=no" in plan["argv"]
    assert "--property=StandardInput=null" in plan["argv"]
    assert plan["argv"][plan["argv"].index("--manifest") + 1] == str(output / "frozen_input_manifest.json")
    python_index = plan["argv"].index(str(Path(sys.executable).resolve()))
    assert Path(plan["argv"][python_index - 1]).name == "nohup"
    assert plan["environment"]["PYTHONPATH"] == f"{output / 'runtime/src'}{os.pathsep}{output / 'runtime'}"
    with pytest.raises(ValueError, match="previously stopped Karen"):
        build_broad_plan(**options, unit="karen-behavior-search-20261013")
    raw = root / "data/untouched.pkl"
    raw.parent.mkdir()
    raw.write_bytes(b"empirical data stays outside frozen source")
    output.mkdir()
    frozen = freeze_broad_runtime(plan)
    assert frozen["files"]
    assert not (output / "runtime/data").exists()
    original = root / "src/fpcode/__init__.py"
    original.write_text("# later source edit\n")
    assert (output / "runtime/src/fpcode/__init__.py").read_text() == "# library\n"
    resumed = build_broad_plan(**options, resume=True)
    assert freeze_broad_runtime(resumed) == frozen
    (output / "runtime/src/fpcode/__init__.py").write_text("# corrupted\n")
    with pytest.raises(ValueError, match="frozen runtime differs"):
        freeze_broad_runtime(resumed)
