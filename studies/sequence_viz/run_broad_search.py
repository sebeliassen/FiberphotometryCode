"""Finite quantitative behavioural recipe search with durable supervised workers.

Import, help and the default plan are read-only and do not import numerical
libraries. ``--execute`` fits only the explicitly prepared manifest. A stopped
run requires ``--resume``; completed, failed and timed-out jobs are retained.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import sqlite3
import tempfile
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
TERMINAL = ("SUCCESS", "EMPTY", "FAILED", "TIMEOUT", "CRASHED", "MEMORY_LIMIT")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def regular_or_absent(path):
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"expected a regular file or absent path: {path}")


def atomic_json(path, value):
    path = Path(path)
    regular_or_absent(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".broad-search-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _number(name, value, *, integer=False, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (value == 0 and not allow_zero) or (integer and not isinstance(value, int)):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}{' integer' if integer else ''}")


def source_snapshot(code_root):
    root = Path(code_root)
    files = sorted((root / "src/fpcode").rglob("*.py"))
    files += sorted((root / "studies/sequence_viz").glob("*.py"))
    return [{"path": str(path), "size": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns, "sha256": digest(path)} for path in files]


def verify_snapshot(rows, *, full=False):
    """Fast repeated immutable-input check; SHA validation is used once on entry."""
    for row in rows:
        path = Path(row["path"])
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"frozen input disappeared or changed file type: {path}")
        stat = path.stat()
        if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError(f"frozen input changed: {path}")
        if full and digest(path) != row["sha256"]:
            raise ValueError(f"frozen input fingerprint differs: {path}")


def build_search_plan(
    *, manifest, output, code_root=None, recipes_per_cohort=120, folds=3, replicates=2,
    seed=101, window_seed=73, per_animal=600, held_per_animal=180, workers=4, threads=2,
    trial_timeout_s=600, max_worker_rss_gb=2, min_free_disk_gb=80, min_free_ram_gb=8,
    cache_quota_gb=16, null_repeats=2, recipes=None, continuous=False,
    max_output_gb=20, write_reserve_gb=2, worker_address_space_gb=0, reference_output=None,
):
    """Freeze finite recipe/job identities without reading empirical arrays or writing."""
    for name, value in (("recipes_per_cohort", recipes_per_cohort), ("folds", folds), ("replicates", replicates),
                        ("per_animal", per_animal), ("held_per_animal", held_per_animal),
                        ("workers", workers), ("threads", threads)):
        _number(name, value, integer=True)
    for name, value in (("seed", seed), ("window_seed", window_seed), ("null_repeats", null_repeats)):
        _number(name, value, integer=True, allow_zero=True)
    for name, value in (("trial_timeout_s", trial_timeout_s), ("max_worker_rss_gb", max_worker_rss_gb),
                        ("cache_quota_gb", cache_quota_gb), ("max_output_gb", max_output_gb)):
        _number(name, value)
    if not isinstance(continuous, bool):
        raise ValueError("continuous must be boolean")
    _number("write_reserve_gb", write_reserve_gb, allow_zero=True)
    _number("worker_address_space_gb", worker_address_space_gb, allow_zero=True)
    if continuous and (write_reserve_gb >= min(cache_quota_gb, max_output_gb) or write_reserve_gb < 2):
        raise ValueError("continuous quotas require at least2GiB write headroom below both caps")
    for name, value in (("min_free_disk_gb", min_free_disk_gb), ("min_free_ram_gb", min_free_ram_gb)):
        _number(name, value, allow_zero=True)
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if available_cpus is not None and workers * threads > available_cpus:
        raise ValueError("workers times threads exceeds CPUs available to this process")
    manifest_path = Path(manifest).expanduser().resolve()
    regular_or_absent(manifest_path)
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("cohorts"), list):
        raise ValueError("prepared manifest schema_version=1 and cohorts list required")
    selected_output = Path(output).expanduser().absolute()
    if selected_output.is_symlink() or (selected_output.exists() and not selected_output.is_dir()):
        raise ValueError("output must be a directory or absent path, never a symlink")
    output_path = selected_output.resolve()
    root = Path(code_root or Path(__file__).resolve().parents[2]).resolve()
    if recipes is None:
        from studies.sequence_viz.broad_search_inputs import generate_recipes

        recipes = _batch_recipes(seed, recipes_per_cohort) if continuous else generate_recipes(seed, recipes_per_cohort)
    if len(recipes) != recipes_per_cohort:
        raise ValueError("recipe generator did not return the requested finite recipe count")
    cohorts, inputs = [], []
    for cohort in payload["cohorts"]:
        name = cohort["cohort_id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
            raise ValueError("cohort_id must contain only letters, digits, underscores and hyphens")
        if name in cohorts or len(cohort["folds"]) != folds:
            raise ValueError("cohort names must be unique and fold counts must match prepared manifest")
        cohorts.append(name)
        for row in cohort["records"]:
            path = Path(row["npz_path"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            inputs.append({"path": str(path.resolve()), "size": row["size"],
                           "mtime_ns": row["mtime_ns"], "sha256": row["sha256"]})
    if not cohorts:
        raise ValueError("at least one explicit cohort is required")
    manifest_stat = manifest_path.stat()
    inputs.append({"path": str(manifest_path), "size": manifest_stat.st_size,
                   "mtime_ns": manifest_stat.st_mtime_ns, "sha256": digest(manifest_path)})
    verify_snapshot(inputs)
    if reference_output:
        reference = Path(reference_output).resolve()
        if reference == output_path:
            raise ValueError("reference output must be a separate completed run")
        status_path, reference_manifest = reference/"status.json", reference/"run_manifest.json"
        if json.loads(status_path.read_text()).get("state") != "COMPLETED":
            raise ValueError("reference output must have completed successfully")
        for path in (status_path, reference_manifest):
            stat = path.stat()
            inputs.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest(path)))
    jobs = []
    # Interleave cohorts, folds and replicates within each base recipe so an
    # early stop leaves the most even finite coverage possible.
    for ordinal, base_recipe in enumerate(recipes):
        for replicate in range(replicates):
            for fold in range(folds):
                for cohort in cohorts:
                    recipe = dict(base_recipe, seed=seed + 1000003 * replicate)
                    job = dict(cohort=cohort, fold=fold, replicate=replicate,
                               base_recipe=ordinal, recipe=recipe)
                    if continuous:
                        if not _stochastic_fit(base_recipe) and replicate > 0:
                            continue
                        job["batch_index"] = 0
                    job["job_id"] = hashlib.sha256(canonical(job).encode()).hexdigest()[:24]
                    jobs.append(job)
    if continuous:
        seen = set()
        if reference_output:
            previous = json.loads((Path(reference_output)/"run_manifest.json").read_text())
            seen.update(_recipe_signature(j["recipe"]) for j in previous["jobs"])
        decisions, fresh = {}, []
        for job in jobs:
            base = job["base_recipe"]
            if base not in decisions:
                signature = _recipe_signature(job["recipe"])
                decisions[base] = signature not in seen
                seen.add(signature)
            if decisions[base]:
                fresh.append(job)
        jobs = fresh
    return dict(schema_version=1, manifest=str(manifest_path), manifest_hash=digest(manifest_path),
                output=str(output_path), code_root=str(root), cohorts=cohorts,
                recipes_per_cohort=recipes_per_cohort, folds=folds, replicates=replicates,
                seed=seed, window_seed=window_seed, per_animal=per_animal, held_per_animal=held_per_animal,
                workers=workers, threads=threads, trial_timeout_s=trial_timeout_s,
                max_worker_rss_gb=max_worker_rss_gb, min_free_disk_gb=min_free_disk_gb,
                min_free_ram_gb=min_free_ram_gb, cache_quota_gb=cache_quota_gb,
                null_repeats=null_repeats, jobs=jobs, total_jobs=len(jobs),
                available_cpus=available_cpus,
                input_snapshot=inputs, source_snapshot=source_snapshot(root),
                units={"RAM_and_disk_limits": "GiB (2**30 bytes)", "windows": "complete source bouts"},
                selection_policy="Quantitative diagnostic vector; no phenotype, drug or DA objective",
                replicate_policy="Detector seed changes; creator seed and input windows remain fixed",
                continuous=continuous, max_output_gb=max_output_gb,
                write_reserve_gb=write_reserve_gb if continuous else 0,
                worker_address_space_gb=worker_address_space_gb,
                reference_output=str(Path(reference_output).resolve()) if reference_output else None)


def _batch_recipes(seed, count):
    from studies.sequence_viz.broad_search_inputs import generate_recipes

    rng = random.Random(seed ^ 0x6A09E667)
    recipes = generate_recipes(seed, count)
    for recipe in recipes:
        if recipe["method"] == "hsmm":
            recipe["iterations"] = rng.choice((8, 16, 32, 64))
        elif recipe["method"] == "seqnmf":
            recipe["iterations"] = rng.choice((40, 80, 120))
    return recipes


def _recipe_signature(recipe):
    names = set(recipe.get("active_parameters", recipe.keys())) | {"method"}
    names -= {"base_id", "representation_cell", "active_parameters"}
    if not _stochastic_fit(recipe):
        names.discard("seed")
    return hashlib.sha256(canonical({k: recipe[k] for k in sorted(names)}).encode()).hexdigest()


def _stochastic_fit(recipe):
    return recipe["method"] == "seqnmf" or (recipe["method"] == "hsmm" and recipe.get("initialization") == "dirichlet")


def continuation_jobs(plan, batch_index):
    """Deterministic fresh balanced batches, with real seeds and budgets persisted."""
    seed = plan["seed"] + 104729*batch_index
    recipes = _batch_recipes(seed, plan["recipes_per_cohort"])
    jobs = []
    for ordinal, base in enumerate(recipes):
        for replicate in range(plan["replicates"]):
            if not _stochastic_fit(base) and replicate:
                continue
            for fold in range(plan["folds"]):
                for cohort in plan["cohorts"]:
                    job = dict(cohort=cohort, fold=fold, replicate=replicate, batch_index=batch_index,
                        base_recipe=batch_index*plan["recipes_per_cohort"]+ordinal,
                        recipe=dict(base, seed=seed+1000003*replicate))
                    job["job_id"] = hashlib.sha256(canonical(job).encode()).hexdigest()[:24]
                    jobs.append(job)
    return jobs


@contextmanager
def exclusive_run(output):
    """Kernel lock prevents competing parent writers; process death releases it."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".run.lock"
    regular_or_absent(lock_path)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another parent already owns this search output") from exc
        yield
    finally:
        os.close(descriptor)


class SearchLedger:
    """Single-parent SQLite state; immutable configs and explicit interrupted resume."""

    def __init__(self, output, plan, *, resume=False):
        self.output, self.plan = Path(output), plan
        manifest = self.output / "run_manifest.json"
        regular_or_absent(manifest)
        database = self.output / "trials.sqlite"
        regular_or_absent(database)
        if manifest.exists():
            if canonical(json.loads(manifest.read_text())) != canonical(plan):
                raise ValueError("resume scope or frozen fingerprints differ from run_manifest.json")
            if not resume:
                raise ValueError("existing run requires explicit --resume")
        elif database.exists():
            raise ValueError("trial database has no matching immutable run manifest")
        else:
            atomic_json(manifest, plan)
        self.connection = sqlite3.connect(database)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS jobs(
            job_id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL, config_json TEXT NOT NULL,
            status TEXT NOT NULL, worker_id INTEGER, pid INTEGER, started_at TEXT, finished_at TEXT,
            elapsed_s REAL, peak_rss_bytes INTEGER, error TEXT, artifact TEXT, attempts INTEGER NOT NULL DEFAULT 0)""")
        self.connection.execute("CREATE TABLE IF NOT EXISTS batches(batch_index INTEGER PRIMARY KEY, jobs_json TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS recipe_signatures(signature TEXT PRIMARY KEY)")
        self.connection.executemany(
            "INSERT OR IGNORE INTO jobs(job_id,ordinal,config_json,status) VALUES(?,?,?,'PENDING')",
            [(job["job_id"], index, canonical(job)) for index, job in enumerate(plan["jobs"])])
        rows = self.connection.execute("SELECT config_json FROM jobs ORDER BY ordinal").fetchall()
        expected = [canonical(job) for job in plan["jobs"]]
        actual = [row[0] for row in rows]
        if (actual[:len(expected)] != expected or (not plan.get("continuous") and len(actual) != len(expected))):
            self.connection.close()
            raise ValueError("persisted job configurations differ from immutable plan")
        if resume:
            self.connection.execute("UPDATE jobs SET status='PENDING', error='explicitly resumed interrupted job' "
                                    "WHERE status IN ('RUNNING','STOPPED')")
        self.connection.commit()
        if plan.get("continuous"):
            self.connection.execute("INSERT OR IGNORE INTO batches VALUES(0,?)", (canonical(plan["jobs"]),))
            for job in plan["jobs"]:
                self.connection.execute("INSERT OR IGNORE INTO recipe_signatures VALUES(?)", (_recipe_signature(job["recipe"]),))
            saved = [canonical(job) for _, payload in self.connection.execute("SELECT batch_index,jobs_json FROM batches ORDER BY batch_index")
                     for job in json.loads(payload)]
            if saved != actual:
                self.connection.close()
                raise ValueError("continuous job ledger differs from immutable batch records")
            self.connection.commit()

    def next_batch(self):
        return self.connection.execute("SELECT COALESCE(MAX(batch_index),0)+1 FROM batches").fetchone()[0]

    def append_batch(self, batch_index, jobs):
        seen = {row[0] for row in self.connection.execute("SELECT signature FROM recipe_signatures")}
        keep, accepted, rejected, new_signatures = [], set(), set(), set()
        for job in jobs:
            signature = _recipe_signature(job["recipe"])
            # All cohort/fold/seed evaluations of an accepted base recipe remain.
            base = job["base_recipe"]
            if base not in accepted and base not in rejected:
                (rejected if signature in seen else accepted).add(base)
                if base in accepted:
                    seen.add(signature)
                    new_signatures.add(signature)
            if base in accepted:
                keep.append(job)
        first = self.connection.execute("SELECT COALESCE(MAX(ordinal),-1)+1 FROM jobs").fetchone()[0]
        with self.connection:
            self.connection.execute("INSERT INTO batches VALUES(?,?)", (batch_index, canonical(keep)))
            self.connection.executemany("INSERT INTO jobs(job_id,ordinal,config_json,status) VALUES(?,?,?,'PENDING')",
                [(job["job_id"], first+i, canonical(job)) for i, job in enumerate(keep)])
            self.connection.executemany("INSERT OR IGNORE INTO recipe_signatures VALUES(?)", [(s,) for s in new_signatures])
        return keep

    def pending(self):
        # Snapshot IDs only; worker results never share this writer connection.
        rows = self.connection.execute("SELECT config_json FROM jobs WHERE status='PENDING' ORDER BY ordinal").fetchall()
        return [json.loads(row[0]) for row in rows]

    def start(self, job, worker_id, pid):
        cursor = self.connection.execute(
            "UPDATE jobs SET status='RUNNING',worker_id=?,pid=?,started_at=?,finished_at=NULL,"
            "elapsed_s=NULL,peak_rss_bytes=NULL,error=NULL,artifact=NULL,attempts=attempts+1 "
            "WHERE job_id=? AND status='PENDING'", (worker_id, pid, now_utc(), job["job_id"]))
        if cursor.rowcount != 1:
            raise RuntimeError("job was not pending at dispatch")
        self.connection.commit()

    def finish(self, outcome):
        status, artifact = outcome.status, None
        if status == "FAILED" and ("MemoryError" in outcome.error or "_ArrayMemoryError" in outcome.error):
            status = "MEMORY_LIMIT"
        if status == "SUCCESS":
            status = outcome.result.get("status", "SUCCESS")
            if status not in ("SUCCESS", "EMPTY"):
                raise ValueError(f"evaluator returned unsupported success status: {status}")
            artifact_path = self.output / "jobs" / f"{outcome.job['job_id']}.json"
            artifact = str(artifact_path)
            atomic_json(artifact_path, dict(job=outcome.job, elapsed_s=outcome.elapsed_s,
                                          peak_rss_bytes=outcome.peak_rss_bytes, result=outcome.result))
        self.connection.execute("UPDATE jobs SET status=?,finished_at=?,elapsed_s=?,peak_rss_bytes=?,error=?,artifact=? "
                                "WHERE job_id=? AND status='RUNNING'",
                                (status, now_utc(), outcome.elapsed_s, outcome.peak_rss_bytes,
                                 outcome.error, artifact, outcome.job["job_id"]))
        self.connection.commit()
        return status

    def counts(self):
        return dict(self.connection.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))

    def interrupt(self, reason):
        self.connection.execute("UPDATE jobs SET status='STOPPED',finished_at=?,error=? WHERE status='RUNNING'",
                                (now_utc(), reason))
        self.connection.commit()

    def seed_pairs(self, completed):
        jobs = [json.loads(row[0]) for row in self.connection.execute(
            "SELECT config_json FROM jobs WHERE json_extract(config_json,'$.cohort')=? "
            "AND json_extract(config_json,'$.fold')=? AND json_extract(config_json,'$.base_recipe')=?",
            (completed["cohort"], completed["fold"], completed["base_recipe"]))]
        reference = next((job for job in jobs if job["replicate"] == 0), None)
        if reference is None:
            return []
        pairs = []
        for alternative in jobs:
            if alternative["replicate"] == 0:
                continue
            rows = []
            for job in (reference, alternative):
                row = self.connection.execute("SELECT status,artifact,error FROM jobs WHERE job_id=?",
                                              (job["job_id"],)).fetchone()
                rows.append(row)
            if all(row[0] in TERMINAL for row in rows):
                pairs.append((reference, alternative, rows))
        return pairs

    def close(self):
        self.connection.close()


def available_ram_bytes():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    return values["MemAvailable"]


def tree_size(path):
    return sum(entry.stat().st_size for entry in Path(path).rglob("*") if entry.is_file() and not entry.is_symlink())


class ResourceGuard:
    """Prompt polling: low RAM pauses dispatch; disk/quota/identity stop the run."""

    def __init__(self, plan, *, interval_s=10):
        self.plan, self.interval_s = plan, interval_s
        self.checked_at, self.reason, self.pause_reason = -math.inf, "", ""
        self.resources = {}

    def check(self, *, force=False):
        if not force and time.monotonic() - self.checked_at < self.interval_s:
            return
        self.checked_at = time.monotonic()
        try:
            verify_snapshot(self.plan["input_snapshot"])
            verify_snapshot(self.plan["source_snapshot"])
            free_disk = shutil.disk_usage(self.plan["output"]).free
            free_ram = available_ram_bytes()
            cache_bytes = tree_size(Path(self.plan["output"]) / "cache")
            output_bytes = tree_size(Path(self.plan["output"])) if (self.plan.get("continuous")
                or self.plan.get("enforce_output_quota")) else None
            self.resources = dict(free_disk_bytes=free_disk, available_ram_bytes=free_ram,
                                  cache_bytes=cache_bytes, output_bytes=output_bytes, checked_at_utc=now_utc())
            reserve = self.plan.get("write_reserve_gb", 0)
            if free_disk < (self.plan["min_free_disk_gb"]+reserve) * 1024 ** 3:
                self.reason = "free disk is below configured threshold"
            elif cache_bytes >= (self.plan["cache_quota_gb"]-reserve) * 1024 ** 3:
                self.reason = "component cache reached configured quota"
            elif output_bytes is not None and output_bytes >= (self.plan["max_output_gb"]-reserve)*1024**3:
                self.reason = "total output reached configured quota with write headroom"
            self.pause_reason = ("available RAM is below configured threshold"
                                 if free_ram < self.plan["min_free_ram_gb"] * 1024 ** 3 else "")
        except (OSError, ValueError) as exc:
            self.reason = str(exc)

    def stopping(self):
        self.check()
        return bool(self.reason)

    def can_schedule(self):
        self.check()
        return not self.reason and not self.pause_reason


def set_worker_allocation_cap(gib):
    """Allocation failure becomes MemoryError before unbounded virtual growth."""
    if not gib:
        return 0
    import resource

    cap = int(gib*1024**3)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    return cap


def _worker_initialize(plan):
    # Called only by explicit execution, after the parent set thread limits.
    address_space_cap = set_worker_allocation_cap(plan.get("worker_address_space_gb", 0))
    from fpcode.workflows.sequence_search import evaluate_recipe
    from studies.sequence_viz.broad_search_inputs import job_identity, load_job_records

    code_paths = dict(evaluator=str(Path(evaluate_recipe.__code__.co_filename).resolve()),
                      loader=str(Path(load_job_records.__code__.co_filename).resolve()))
    expected = dict(evaluator=str(Path(plan["code_root"]) / "src/fpcode/workflows/sequence_search.py"),
                    loader=str(Path(plan["code_root"]) / "studies/sequence_viz/broad_search_inputs.py"))
    if code_paths != expected:
        raise RuntimeError("worker imported code outside the selected frozen runtime")
    try:
        from threadpoolctl import threadpool_info

        threadpools = threadpool_info()
    except ImportError:
        threadpools = [{"status": "unavailable", "reason": "optional threadpoolctl is not installed"}]
    return {"plan": plan, "load": load_job_records, "identity": job_identity,
            "evaluate": evaluate_recipe, "threadpools": threadpools,
            "record_cache": OrderedDict(), "record_cache_bytes": 0,
        "record_cache_cap_bytes": 64 * 1024 ** 2, "record_cache_max_entries": 2, "code_paths": code_paths,
        "address_space_cap": address_space_cap}


def _array_bytes(value):
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_array_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_bytes(item) for item in value)
    return 0


def _worker_evaluate(state, job):
    plan = state["plan"]
    key = (job["cohort"], job["fold"])
    cached = state["record_cache"].pop(key, None)
    if cached is None:
        records = state["load"](plan["manifest"], job["cohort"], job["fold"],
                                per_animal=plan["per_animal"], held_per_animal=plan["held_per_animal"],
                                window_seed=plan["window_seed"])
        identity = state["identity"](plan["manifest"], job["cohort"], job["fold"], records,
                                      window_seed=plan["window_seed"], per_animal=plan["per_animal"],
                                      held_per_animal=plan["held_per_animal"])
        size = _array_bytes(records) + len(canonical(identity).encode())
        if size <= state["record_cache_cap_bytes"]:
            while state["record_cache"] and (state["record_cache_bytes"] + size > state["record_cache_cap_bytes"]
                                              or len(state["record_cache"]) >= state["record_cache_max_entries"]):
                _, (_, _, previous_size) = state["record_cache"].popitem(last=False)
                state["record_cache_bytes"] -= previous_size
            state["record_cache"][key] = (records, identity, size)
            state["record_cache_bytes"] += size
    else:
        records, identity, size = cached
        state["record_cache"][key] = cached
    result = state["evaluate"](records, job["recipe"], identity, Path(plan["output"]) / "cache",
                                null_repeats=plan["null_repeats"])
    result["operational_conditions"] = dict(pid=os.getpid(), thread_environment={
        name: os.environ.get(name) for name in THREAD_VARIABLES}, threadpools=state["threadpools"],
        worker_rss_cap_bytes=int(plan["max_worker_rss_gb"] * 1024 ** 3),
        worker_address_space_cap_bytes=state["address_space_cap"],
        trial_timeout_s=plan["trial_timeout_s"], record_cache_hit=cached is not None,
        cached_record_array_and_identity_bytes=state["record_cache_bytes"],
        record_cache_cap_bytes=state["record_cache_cap_bytes"], code_paths=state["code_paths"])
    return result


def execute_search(plan, *, resume=False, evaluator=None, initializer=None, batch_generator=None):
    """Run a finite frozen plan. Optional callbacks support isolated supervisor tests."""
    from fpcode.common.trial_runner import run_trials

    output = Path(plan["output"])
    with exclusive_run(output):
        for name in ("jobs", "cache", "seed_stability"):
            selected = output / name
            if selected.is_symlink() or (selected.exists() and not selected.is_dir()):
                raise ValueError(f"expected ordinary output directory: {selected}")
            selected.mkdir(exist_ok=True)
        verify_snapshot(plan["input_snapshot"], full=True)
        verify_snapshot(plan["source_snapshot"], full=True)
        stop_path = output / "STOP"
        regular_or_absent(stop_path)
        if stop_path.exists() and not resume:
            raise ValueError("STOP exists; use explicit --resume to acknowledge and remove it")
        ledger = SearchLedger(output, plan, resume=resume)
        tracker = None
        if plan.get("continuous"):
            from studies.sequence_viz.broad_search_tracking import CandidateTracker

            tracker = CandidateTracker(ledger.connection, output, atomic_json)
            tracker.rebuild([plan["reference_output"]] if plan.get("reference_output") else [])
            tracker.export(force=True)
            reference = Path(plan["reference_output"]) if plan.get("reference_output") else None
            if reference:
                for path in (reference/"jobs").glob("*.json"):
                    job = json.loads(path.read_text())["job"]
                    ledger.connection.execute("INSERT OR IGNORE INTO recipe_signatures VALUES(?)", (_recipe_signature(job["recipe"]),))
                ledger.connection.commit()
        if stop_path.exists():
            stop_path.unlink()
        guard = ResourceGuard(plan)
        requested = {"reason": ""}
        old_handlers = {}
        iterator = None

        def handle_signal(signum, _frame):
            requested["reason"] = f"received signal {signum}"

        def write_status(state, error=""):
            atomic_json(output / "status.json", dict(state=state, updated_at_utc=now_utc(),
                parent_pid=os.getpid(), counts=ledger.counts(), total_jobs=None if plan.get("continuous") else plan["total_jobs"],
                generated_jobs=sum(ledger.counts().values()), continuous=plan.get("continuous", False),
                stop_reason=requested["reason"] or guard.reason, pause_reason=guard.pause_reason,
                resources=guard.resources, error=error))

        def started(job, worker_id, pid):
            ledger.start(job, worker_id, pid)
            print(f"START {job['job_id']} {job['cohort']} {job['recipe']['method']} "
                  f"fold={job['fold']} replicate={job['replicate']} worker={worker_id} pid={pid}", flush=True)
            write_status("RUNNING")

        def stopping():
            return bool(requested["reason"]) or guard.stopping()

        def export_pairs(job):
            if not plan.get("compare_seed_catalogues", True):
                return
            for reference, alternative, rows in ledger.seed_pairs(job):
                pair_path = output / "seed_stability" / (
                    f"{reference['cohort']}_fold{reference['fold']}_base{reference['base_recipe']}"
                    f"_replicate{alternative['replicate']}.json")
                if pair_path.exists():
                    continue
                pair = dict(reference=reference, alternative=alternative)
                if all(row[0] in ("SUCCESS", "EMPTY") and row[1] for row in rows):
                    try:
                        from fpcode.workflows.sequence_search import compare_recipe_catalogues

                        results = [json.loads(Path(row[1]).read_text())["result"] for row in rows]
                        pair.update(status="AVAILABLE", comparison=compare_recipe_catalogues(*results))
                    except Exception as exc:
                        pair.update(status="UNAVAILABLE", reason=f"{type(exc).__name__}: {exc}")
                else:
                    pair.update(status="UNAVAILABLE", reason="one or both detector seeds did not complete successfully",
                                outcomes=[dict(status=row[0], error=row[2]) for row in rows])
                atomic_json(pair_path, pair)
                if tracker is not None:
                    tracker.add_pair(pair)

        def job_stream():
            yield from ledger.pending()
            while plan.get("continuous") and not stopping() and not stop_path.exists():
                batch = ledger.next_batch()
                jobs = (batch_generator or continuation_jobs)(plan, batch)
                fresh = ledger.append_batch(batch, jobs)
                print(f"BATCH {batch} generated={len(jobs)} fresh={len(fresh)}", flush=True)
                yield from fresh

        try:
            for name in THREAD_VARIABLES:
                os.environ[name] = str(plan["threads"])
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
            for signum in (signal.SIGTERM, signal.SIGINT):
                old_handlers[signum] = signal.signal(signum, handle_signal)
            guard.check(force=True)
            write_status("RUNNING")
            print(f"READY parent={os.getpid()} total_jobs={plan['total_jobs']} "
                  f"pending={len(ledger.pending())} workers={plan['workers']} threads={plan['threads']}", flush=True)
            iterator = run_trials(job_stream(), evaluator or _worker_evaluate,
                initializer=initializer or _worker_initialize, initargs=(plan,), workers=plan["workers"],
                timeout_s=plan["trial_timeout_s"], initializer_timeout_s=120,
                max_worker_rss_bytes=int(plan["max_worker_rss_gb"] * 1024 ** 3),
                stop_file=stop_path, stop_requested=stopping, on_start=started,
                can_schedule=guard.can_schedule, poll_interval_s=.1, termination_grace_s=.5)
            for outcome in iterator:
                status = ledger.finish(outcome)
                if tracker is not None and outcome.status == "SUCCESS":
                    tracker.add(outcome.job, outcome.result, output/"jobs"/f"{outcome.job['job_id']}.json")
                export_pairs(outcome.job)
                if tracker is not None:
                    tracker.export()
                print(f"{status} {outcome.job['job_id']} elapsed_s={outcome.elapsed_s:.3f} "
                      f"peak_rss_bytes={outcome.peak_rss_bytes} {outcome.error[-500:]}", flush=True)
                write_status("RUNNING")
            counts = ledger.counts()
            state = "STOPPED" if plan.get("continuous") or counts.get("PENDING", 0) or counts.get("STOPPED", 0) else "COMPLETED"
            if state == "STOPPED" and not requested["reason"] and not guard.reason:
                requested["reason"] = "STOP file requested" if stop_path.exists() else "interrupted jobs remain"
            write_status(state)
            if tracker is not None:
                tracker.export(force=True)
            print(f"{state} {canonical(counts)}", flush=True)
            return dict(state=state, counts=counts)
        except BaseException as exc:
            ledger.interrupt(f"parent interrupted: {type(exc).__name__}: {exc}")
            write_status("FAILED", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if iterator is not None:
                iterator.close()
            for signum, old_handler in old_handlers.items():
                signal.signal(signum, old_handler)
            ledger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-root")
    for name, default in (("recipes-per-cohort", 120), ("folds", 3), ("replicates", 2), ("seed", 101),
                          ("window-seed", 73), ("per-animal", 600), ("held-per-animal", 180),
                          ("workers", 4), ("threads", 2), ("null-repeats", 2)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("trial-timeout-s", 600), ("max-worker-rss-gb", 2), ("min-free-disk-gb", 80),
                          ("min-free-ram-gb", 8), ("cache-quota-gb", 16), ("max-output-gb", 20),
                          ("write-reserve-gb", 2), ("worker-address-space-gb", 0)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continuous", action="store_true", help="append fresh balanced batches until STOP or a resource guard")
    parser.add_argument("--reference-output", help="completed baseline artifacts for provenance and provisional leaders")
    options = vars(parser.parse_args(argv))
    execute, resume = options.pop("execute"), options.pop("resume")
    plan = build_search_plan(**options)
    if execute:
        result = execute_search(plan, resume=resume)
    else:
        result = {key: value for key, value in plan.items() if key not in ("jobs", "input_snapshot", "source_snapshot")}
        result.update(execute=False, resume=resume, recipe_preview=[job for job in plan["jobs"][:6]],
                      stop_file=str(Path(plan["output"]) / "STOP"))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
