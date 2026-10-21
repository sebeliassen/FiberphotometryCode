"""Plan or explicitly launch a bounded Karen search through user systemd.

The default is read-only: no directory, log, manifest or service is created.
With --execute, systemd owns the job independently of this terminal or IDE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _regular_or_absent(path):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"expected a regular file or absent path: {path}")


def build_plan(
    *, code_root, python, atlas, output, unit, workers=4, threads=4, hours=0,
    trial_timeout_s=120, worker_rss_gb=4, memory_high_gb=16, memory_max_gb=24,
    min_free_disk_gb=80, min_free_ram_gb=8,
):
    """Validate selected locations and return shell-free launch/status arguments."""
    if not isinstance(unit, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}(?:\.service)?", unit):
        raise ValueError("unit must contain only letters, digits, underscores and hyphens, optionally .service")
    if not unit.endswith(".service"):
        unit += ".service"
    for name, value in (("workers", workers), ("threads", threads)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours < 0:
        raise ValueError("hours must be finite and nonnegative; 0 means run until stopped")
    for name, value in (("trial_timeout_s", trial_timeout_s), ("worker_rss_gb", worker_rss_gb),
                        ("memory_high_gb", memory_high_gb), ("memory_max_gb", memory_max_gb)):
        _positive(name, value)
    if memory_high_gb > memory_max_gb:
        raise ValueError("memory_high_gb must not exceed memory_max_gb")
    if memory_max_gb <= workers * worker_rss_gb:
        raise ValueError("memory_max_gb must leave headroom above the aggregate worker RSS caps")
    for name, value in (("min_free_disk_gb", min_free_disk_gb), ("min_free_ram_gb", min_free_ram_gb)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    code_root, python, atlas = (Path(path).expanduser().resolve() for path in (code_root, python, atlas))
    selected_output = Path(output).expanduser().absolute()
    if selected_output.is_symlink():
        raise ValueError("output directory must not be a symlink")
    output = selected_output.resolve()
    if not code_root.is_dir() or not (code_root / "studies/karen_pff_vs_monomer/analysis/run_overnight_random_search.py").is_file():
        raise ValueError("code_root must contain the maintained Karen search entry point")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("python must be an existing executable file")
    nohup = shutil.which("nohup")
    if nohup is None:
        raise ValueError("nohup executable is required")
    if not atlas.is_dir() or not (atlas / "sessions.csv").is_file():
        raise ValueError("atlas must contain sessions.csv")
    if output.exists() and not output.is_dir():
        raise ValueError("output must be a directory or absent path")
    log, record = output / "search.log", output / "service_launch.json"
    _regular_or_absent(log)
    _regular_or_absent(record)

    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": os.pathsep.join((str(code_root / "src"), str(code_root)))}
    env.update({name: str(threads) for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    argv = ["systemd-run", "--user", "--unit", unit, "--description", "Bounded Karen behavioural-unit search",
            f"--property=WorkingDirectory={code_root}",
            f"--property=MemoryHigh={int(memory_high_gb * 1024 ** 3)}",
            f"--property=MemoryMax={int(memory_max_gb * 1024 ** 3)}",
            "--property=KillMode=control-group", "--property=TimeoutStopSec=20s", "--property=StandardInput=null",
            f"--property=RuntimeMaxSec={math.ceil(hours * 3600 + 900)}s" if hours else "--property=RuntimeMaxSec=infinity",
            "--property=Restart=on-failure" if hours == 0 else "--property=Restart=no",
            "--property=RestartSec=30s", "--property=StartLimitIntervalSec=600s", "--property=StartLimitBurst=3",
            f"--property=StandardOutput=append:{log}", f"--property=StandardError=append:{log}"]
    argv.extend(f"--setenv={key}={value}" for key, value in env.items())
    argv.extend([str(Path(nohup).resolve()), str(python), "-B", "-m",
                 "studies.karen_pff_vs_monomer.analysis.run_overnight_random_search",
                 "--atlas", str(atlas), "--output", str(output), "--workers", str(workers), "--threads", str(threads),
                 "--hours", str(hours), "--trial-timeout-s", str(trial_timeout_s),
                 "--max-worker-rss-gb", str(worker_rss_gb), "--min-free-disk-gb", str(min_free_disk_gb),
                 "--min-free-ram-gb", str(min_free_ram_gb), "--execute"])
    if record.exists():
        previous = json.loads(record.read_text())
        if previous.get("argv") != argv:
            raise ValueError("existing service_launch.json describes different launch arguments; select a new output")
    elif log.exists() and log.stat().st_size:
        raise ValueError("existing nonempty search.log has no matching launch record; select a new output")
    return {"argv": argv, "unit": unit, "code_root": str(code_root), "python": str(python), "atlas": str(atlas),
            "output": str(output), "log": str(log), "launch_record": str(record), "environment": env,
            "min_free_disk_gb": min_free_disk_gb, "min_free_ram_gb": min_free_ram_gb,
            "memory_limit_units": "GiB (2**30 bytes)",
            "status_argv": ["systemctl", "--user", "status", unit],
            "stop_argv": ["systemctl", "--user", "stop", unit]}


def _write_record(path, payload):
    _regular_or_absent(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".service-launch-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def build_broad_plan(
    *, code_root, python, manifest, output, unit="broad-behaviour-search-20261013",
    workers=4, threads=2, recipes_per_cohort=120, folds=3, replicates=2, seed=101,
    window_seed=73, per_animal=600, held_per_animal=180, null_repeats=2,
    trial_timeout_s=600, worker_rss_gb=2, memory_high_gb=8, memory_max_gb=12,
    min_free_disk_gb=80, min_free_ram_gb=8, cache_quota_gb=16, resume=False,
    continuous=False, max_output_gb=20, write_reserve_gb=2, worker_address_space_gb=0,
    reference_output=None,
):
    """Plan a separate finite search with frozen source and no automatic restart."""
    if not isinstance(unit, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}(?:\.service)?", unit):
        raise ValueError("unit must contain only letters, digits, underscores and hyphens, optionally .service")
    if not unit.endswith(".service"):
        unit += ".service"
    if unit == "karen-behavior-search-20261013.service":
        raise ValueError("the previously stopped Karen service must not be reused for the broad search")
    for name, value in (("workers", workers), ("threads", threads), ("recipes_per_cohort", recipes_per_cohort),
                        ("folds", folds), ("replicates", replicates), ("per_animal", per_animal),
                        ("held_per_animal", held_per_animal)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("seed", seed), ("window_seed", window_seed), ("null_repeats", null_repeats)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name, value in (("trial_timeout_s", trial_timeout_s), ("worker_rss_gb", worker_rss_gb),
                        ("memory_high_gb", memory_high_gb), ("memory_max_gb", memory_max_gb),
                        ("cache_quota_gb", cache_quota_gb), ("max_output_gb", max_output_gb)):
        _positive(name, value)
    if not isinstance(continuous, bool):
        raise ValueError("continuous must be boolean")
    if worker_address_space_gb < 0 or write_reserve_gb < 0:
        raise ValueError("worker allocation limit and write reserve must be nonnegative")
    if continuous and (write_reserve_gb < 2 or write_reserve_gb >= min(cache_quota_gb, max_output_gb)):
        raise ValueError("continuous quotas need at least2GiB write headroom below both caps")
    if memory_high_gb > memory_max_gb or memory_max_gb <= workers * worker_rss_gb:
        raise ValueError("memory limits must leave headroom above aggregate worker RSS caps")
    for name, value in (("min_free_disk_gb", min_free_disk_gb), ("min_free_ram_gb", min_free_ram_gb)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    code_root, python, manifest = (Path(path).expanduser().resolve() for path in (code_root, python, manifest))
    selected_output = Path(output).expanduser().absolute()
    if selected_output.is_symlink():
        raise ValueError("output directory must not be a symlink")
    output = selected_output.resolve()
    if not (code_root / "studies/sequence_viz/run_broad_search.py").is_file():
        raise ValueError("code_root must contain the maintained broad-search entry point")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("python must be an existing executable file")
    if output.exists() and not output.is_dir():
        raise ValueError("output must be a directory or absent path")
    _regular_or_absent(manifest)
    if not manifest.is_file():
        raise ValueError("prepared input manifest must be an existing regular JSON file")
    prepared = json.loads(manifest.read_text())
    if prepared.get("schema_version") != 1 or not prepared.get("cohorts"):
        raise ValueError("prepared input manifest schema_version=1 and explicit cohorts required")
    nohup = shutil.which("nohup")
    if nohup is None:
        raise ValueError("nohup executable is required")
    log, record, runtime = output / "search.log", output / "service_launch.json", output / "runtime"
    frozen_manifest = output / "frozen_input_manifest.json"
    for path in (log, record, output / "runtime_manifest.json", frozen_manifest):
        _regular_or_absent(path)
    if runtime.is_symlink() or (runtime.exists() and not runtime.is_dir()):
        raise ValueError("frozen runtime must be an ordinary directory or absent path")
    environment = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                   "FIBERPHOTOMETRY_ROOT": str(code_root),
                   "PYTHONPATH": os.pathsep.join((str(runtime / "src"), str(runtime)))}
    environment.update({name: str(threads) for name in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    argv = ["systemd-run", "--user", "--unit", unit, "--description", "Finite broad behavioural recipe search",
            f"--property=WorkingDirectory={runtime}", f"--property=MemoryHigh={int(memory_high_gb * 1024 ** 3)}",
            f"--property=MemoryMax={int(memory_max_gb * 1024 ** 3)}", "--property=KillMode=control-group",
            "--property=TimeoutStopSec=20s", "--property=StandardInput=null", "--property=RuntimeMaxSec=infinity",
            "--property=Restart=no", f"--property=StandardOutput=append:{log}",
            f"--property=StandardError=append:{log}"]
    if continuous:
        argv.extend(["--property=MemorySwapMax=0", "--property=OOMPolicy=stop"])
    argv.extend(f"--setenv={key}={value}" for key, value in environment.items())
    argv.extend([str(Path(nohup).resolve()), str(python), "-B", "-m", "studies.sequence_viz.run_broad_search",
                 "--manifest", str(frozen_manifest), "--output", str(output), "--code-root", str(runtime)])
    options = dict(workers=workers, threads=threads, recipes_per_cohort=recipes_per_cohort, folds=folds,
                   replicates=replicates, seed=seed, window_seed=window_seed, per_animal=per_animal,
                   held_per_animal=held_per_animal, null_repeats=null_repeats, trial_timeout_s=trial_timeout_s,
                   max_worker_rss_gb=worker_rss_gb, min_free_disk_gb=min_free_disk_gb,
                   min_free_ram_gb=min_free_ram_gb, cache_quota_gb=cache_quota_gb,
                   max_output_gb=max_output_gb, write_reserve_gb=write_reserve_gb,
                   worker_address_space_gb=worker_address_space_gb)
    for key, value in options.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    if continuous:
        argv.append("--continuous")
    if reference_output is not None:
        argv.extend(["--reference-output", str(Path(reference_output).resolve())])
    if resume:
        argv.append("--resume")
    argv.append("--execute")
    if record.exists() and not resume:
        raise ValueError("an existing broad launch requires explicit resume=True")
    if record.exists():
        previous = json.loads(record.read_text())
        previous_argv = [part for part in previous.get("argv", []) if part != "--resume"]
        if previous_argv != [part for part in argv if part != "--resume"]:
            raise ValueError("existing broad launch has different scope or service arguments")
        if previous.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError("prepared manifest changed since the original broad launch")
    if not record.exists() and log.exists() and log.stat().st_size:
        raise ValueError("existing nonempty log has no matching launch record")
    return dict(mode="broad", argv=argv, unit=unit, code_root=str(code_root), python=str(python),
                manifest=str(manifest), frozen_manifest=str(frozen_manifest),
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                output=str(output), log=str(log), launch_record=str(record), environment=environment,
                runtime=str(runtime), resume=resume, options=options, continuous=continuous,
                reference_output=str(Path(reference_output).resolve()) if reference_output else None,
                min_free_disk_gb=min_free_disk_gb, min_free_ram_gb=min_free_ram_gb,
                memory_limit_units="GiB (2**30 bytes)",
                status_argv=["systemctl", "--user", "status", unit],
                stop_argv=["systemctl", "--user", "stop", unit], stop_file=str(output / "STOP"))


def freeze_broad_runtime(plan):
    """Atomically copy only source, preserving it across later workspace edits."""
    runtime, root = Path(plan["runtime"]), Path(plan["code_root"])
    record = Path(plan["output"]) / "runtime_manifest.json"
    if runtime.exists():
        if not plan["resume"] or not record.is_file():
            raise ValueError("existing frozen runtime needs explicit resume and its runtime manifest")
        previous = json.loads(record.read_text())
        frozen_manifest = Path(plan["frozen_manifest"])
        if not frozen_manifest.is_file() or hashlib.sha256(frozen_manifest.read_bytes()).hexdigest() != previous["frozen_manifest_sha256"]:
            raise ValueError("frozen prepared manifest differs from its original fingerprint")
        for row in previous["files"]:
            path = runtime / row["relative_path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("existing frozen runtime differs from its fingerprint")
        return previous
    if plan["resume"]:
        raise ValueError("cannot resume without the original frozen runtime")
    if hashlib.sha256(Path(plan["manifest"]).read_bytes()).hexdigest() != plan["manifest_sha256"]:
        raise ValueError("prepared input manifest changed after launch planning")
    staging = Path(tempfile.mkdtemp(prefix=".runtime-", dir=runtime.parent))
    try:
        files = []
        for directory in (root / "src/fpcode", root / "studies/sequence_viz"):
            paths = directory.rglob("*") if directory.name == "fpcode" else directory.glob("*.py")
            for path in sorted(paths):
                if not path.is_file() or path.suffix not in (".py", ".json", ".yaml", ".yml", ".typed"):
                    continue
                if path.is_symlink():
                    raise ValueError(f"runtime source symlinks are unsupported: {path}")
                relative = path.relative_to(root)
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                files.append(dict(relative_path=str(relative), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                  size=target.stat().st_size))
        for relative in ("studies/__init__.py",):
            path = root / relative
            if path.is_file():
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                files.append(dict(relative_path=relative, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                  size=target.stat().st_size))
        prepared = json.loads(Path(plan["manifest"]).read_text())
        for cohort in prepared["cohorts"]:
            for row in cohort.get("records", []):
                source = Path(row["npz_path"])
                row["npz_path"] = str((source if source.is_absolute() else Path(plan["manifest"]).parent / source).resolve())
        _write_record(Path(plan["frozen_manifest"]), prepared)
        frozen = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(), source_root=str(root), files=files,
                      prepared_manifest=plan["manifest"], prepared_manifest_sha256=plan["manifest_sha256"])
        frozen["frozen_manifest_sha256"] = hashlib.sha256(Path(plan["frozen_manifest"]).read_bytes()).hexdigest()
        os.replace(staging, runtime)
        _write_record(record, frozen)
        return frozen
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def execute_plan(plan):
    """Launch the already-reviewed plan and persist its attempt/result."""
    output, log, record = (Path(plan[key]) for key in ("output", "log", "launch_record"))
    output.mkdir(parents=True, exist_ok=True)
    _regular_or_absent(log)
    _regular_or_absent(record)
    if record.exists() and json.loads(record.read_text()).get("argv") != plan["argv"] and not plan.get("resume"):
        raise ValueError("launch record changed scope after planning")
    if not record.exists() and log.exists() and log.stat().st_size:
        raise ValueError("existing nonempty log has no matching launch record")
    flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(log, flags, 0o600)
    os.close(descriptor)
    if plan.get("mode") == "broad":
        freeze_broad_runtime(plan)
    payload = dict(plan, requested_at_utc=datetime.now(timezone.utc).isoformat(), status="LAUNCH_REQUESTED")
    _write_record(record, payload)
    try:
        completed = subprocess.run(plan["argv"], capture_output=True, text=True, timeout=30, check=False)
    except Exception as exc:
        payload.update(status="LAUNCH_FAILED", error=f"{type(exc).__name__}: {exc}")
        _write_record(record, payload)
        raise
    payload.update(status="LAUNCHED" if completed.returncode == 0 else "LAUNCH_FAILED",
                   returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    _write_record(record, payload)
    if completed.returncode:
        raise RuntimeError(f"systemd launch failed: {completed.stderr.strip()}")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("karen", "broad"), default="karen")
    for name in ("code-root", "python", "output", "unit"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--atlas")
    parser.add_argument("--manifest")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--hours", type=float, default=0, help="0 = run until stopped; positive = bounded hours")
    parser.add_argument("--trial-timeout-s", type=float)
    parser.add_argument("--worker-rss-gb", type=float)
    parser.add_argument("--memory-high-gb", type=float)
    parser.add_argument("--memory-max-gb", type=float)
    parser.add_argument("--min-free-disk-gb", type=float, default=80)
    parser.add_argument("--min-free-ram-gb", type=float, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    for name, default in (("recipes-per-cohort", 120), ("folds", 3), ("replicates", 2), ("seed", 101),
                          ("window-seed", 73), ("per-animal", 600), ("held-per-animal", 180), ("null-repeats", 2)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--cache-quota-gb", type=float, default=16)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--reference-output")
    parser.add_argument("--max-output-gb", type=float, default=20)
    parser.add_argument("--write-reserve-gb", type=float, default=2)
    parser.add_argument("--worker-address-space-gb", type=float, default=0)
    args = vars(parser.parse_args(argv))
    execute, mode = args.pop("execute"), args.pop("mode")
    defaults = dict(threads=2, trial_timeout_s=600, worker_rss_gb=2, memory_high_gb=8, memory_max_gb=12)
    if mode == "karen":
        defaults = dict(threads=4, trial_timeout_s=120, worker_rss_gb=4, memory_high_gb=16, memory_max_gb=24)
    for name, value in defaults.items():
        if args[name] is None:
            args[name] = value
    if mode == "broad":
        args.pop("atlas")
        args.pop("hours")
        if args["manifest"] is None:
            parser.error("--mode broad requires --manifest")
        plan = build_broad_plan(**args)
    else:
        args.pop("manifest")
        for key in ("resume", "recipes_per_cohort", "folds", "replicates", "seed", "window_seed",
                    "per_animal", "held_per_animal", "null_repeats", "cache_quota_gb", "continuous",
                    "reference_output", "max_output_gb", "write_reserve_gb", "worker_address_space_gb"):
            args.pop(key)
        if args["atlas"] is None:
            parser.error("--mode karen requires --atlas")
        plan = build_plan(**args)
    print(json.dumps(execute_plan(plan) if execute else dict(plan, execute=False), indent=2))


if __name__ == "__main__":
    main()
