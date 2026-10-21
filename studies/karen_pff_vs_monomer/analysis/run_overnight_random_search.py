"""Karen routine-quality search with real animal-held-out evidence and checkpoints.

Rank only within (resolution, method). Disease labels balance folds but never
enter fitness; phenotype/L-DOPA analyses and first-order-null checks follow after
shortlisting. Five animals are reserved from THIS search, not historically unseen.
Default invocation is a read-only plan; --execute performs the authorised fits.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import os
import shutil
import signal
import sqlite3
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

LEVELS = ("L1", "L2", "L3", "L4")
METHODS = ("exact", "seqnmf")
COMMON_SPACE = {"min_bout_ms": (0, 50, 100, 150), "chain_len": (3, 4, 5, 6), "n_units": (6, 10, 14, 18, 22)}
METHOD_SPACE = {"exact": ("min_prevalence", (4, 6, 8, 10)), "seqnmf": ("penalty", (0.005, 0.01, 0.05, 0.1))}
METRICS = (
    "order_gain",
    "order_gain_sd",
    "stability",
    "same_data_seed_stability",
    "crossfold_stability",
    "explained",
    "dead_fraction",
    "redundancy",
    "fitness",
    "null_sd_mean",
)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".search-", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def configuration_buckets(seed):
    """Deterministic balanced coverage, with active knobs only and no time RNG."""
    import numpy as np

    buckets = []
    for level in LEVELS:
        for method in METHODS:
            knob, choices = METHOD_SPACE[method]
            configs = [
                dict(level=level, method=method, min_bout_ms=ms, chain_len=length, n_units=units, **{knob: value})
                for ms, length, units, value in itertools.product(*COMMON_SPACE.values(), choices)
            ]
            baseline = dict(
                level=level,
                method=method,
                min_bout_ms=0,
                chain_len=3,
                n_units=6,
                **{knob: 4 if method == "exact" else 0.01},
            )
            configs.remove(baseline)
            stratum_seed = int(hashlib.sha256(f"{seed}|{level}|{method}".encode()).hexdigest()[:8], 16)
            order = np.random.default_rng(stratum_seed).permutation(len(configs))
            buckets.append([baseline] + [configs[int(index)] for index in order])
    return buckets


def configuration_at(ordinal, buckets, replicates):
    strata = len(buckets)
    round_index, position = divmod(ordinal, strata * replicates)
    replicate_position, stratum = divmod(position, strata)
    cycle, index = divmod(round_index, len(buckets[stratum]))
    return buckets[stratum][index], cycle * replicates + replicate_position


def source_identity(code_root):
    files = sorted((code_root / "src/fpcode").rglob("*.py"))
    files.append(Path(__file__).resolve())
    return {str(path): digest(path) for path in files}


def load_metadata(atlas, seed):
    import numpy as np
    import pandas as pd

    from fpcode.workflows.sequence_atlas import validate_animal_splits

    sessions = pd.read_csv(atlas / "sessions.csv", dtype={"animal_id": str})
    if sessions.session_id.duplicated().any() or sessions.groupby("animal_id").group.nunique().max() != 1:
        raise ValueError("unique recordings and consistent animal groups required")
    splits = json.loads((atlas / "chains/animal_splits.json").read_text())
    validate_animal_splits(sessions, splits)
    pool = list(map(str, splits["train"])) + list(map(str, splits["validation"]))
    reserved = sorted(map(str, splits["test"]))
    group = sessions.drop_duplicates("animal_id").set_index("animal_id").group
    rng, folds = np.random.default_rng(seed), [[], [], []]
    for name in sorted(group.unique()):
        animals = [animal for animal in pool if group[animal] == name]
        rng.shuffle(animals)
        for index, animal in enumerate(animals):
            folds[index % len(folds)].append(animal)
    annotation = pd.read_csv(atlas / "syllable_annotations.csv")
    vocab = {"L1": int(annotation.syllable_id.max()) + 1}
    metadata_paths = [atlas / "sessions.csv", atlas / "chains/animal_splits.json", atlas / "syllable_annotations.csv"]
    for level in LEVELS[1:]:
        path = atlas / "clustering" / level / "mapping.json"
        mapping = json.loads(path.read_text())
        if len(mapping) != vocab["L1"] or any(not isinstance(value, int) or value < 0 for value in mapping):
            raise ValueError("declared resolution mapping differs from raw vocabulary")
        vocab[level] = max(mapping) + 1
        metadata_paths.append(path)
    return sessions, pool, reserved, folds, vocab, metadata_paths


class SearchData:
    """Load only search-animal bout tables, never reserved-test observations."""

    def __init__(self, atlas, seed):
        import pandas as pd

        self.atlas, self.base_seed = Path(atlas), seed
        sessions, self.pool, self.test, self.folds, self.vocab, _ = load_metadata(self.atlas, seed)
        self.sessions = sessions[sessions.animal_id.isin(self.pool)].copy()
        self.sessions_of = {animal: sub.session_id.tolist() for animal, sub in self.sessions.groupby("animal_id")}
        self.raw, self.memo = {}, OrderedDict()
        columns = ["syllable_id", "segment", "duration_s", "start_frame", "end_frame_exclusive"]
        for level in LEVELS:
            folder = (
                self.atlas / "measurements" if level == "L1" else self.atlas / "clustering" / level / "measurements"
            )
            self.raw[level] = {
                sid: pd.read_csv(folder / "bouts_by_session" / f"{sid}.csv", usecols=columns)
                for sid in self.sessions.session_id
            }
        print(
            f"worker loaded {len(self.pool)} search mice / {len(self.sessions)} recordings; "
            f"{len(self.test)} mice reserved from this search",
            flush=True,
        )

    def bouts(self, level, min_ms):
        from fpcode.workflows.sequence_search import filter_short_bouts

        key = (level, min_ms)
        if key not in self.memo:
            if len(self.memo) >= 8:
                self.memo.popitem(last=False)
            self.memo[key] = {sid: filter_short_bouts(frame, min_ms / 1000.0) for sid, frame in self.raw[level].items()}
        self.memo.move_to_end(key)
        return self.memo[key]

    def fold_split(self, fold):
        held = self.folds[fold]
        return [animal for animal in self.pool if animal not in held], list(held)


def initialize_worker(atlas, seed, expected_files):
    if any(digest(path) != value for path, value in expected_files.items()):
        raise RuntimeError("source/input changed after run identity was recorded")
    return SearchData(atlas, seed)


def evaluate_job(data, job):
    from fpcode.workflows.sequence_search import evaluate_configuration

    result = evaluate_configuration(data, job["configuration"], job["replicate"], **job["settings"])
    level, min_ms = job["configuration"]["level"], job["configuration"]["min_bout_ms"]
    original = data.raw[level]
    filtered = data.bouts(level, min_ms)
    seconds = sum(frame.duration_s.sum() for frame in original.values())
    result["retained_duration_fraction"] = float(sum(frame.duration_s.sum() for frame in filtered.values()) / seconds)
    result["retained_bout_fraction"] = float(
        sum(len(frame) for frame in filtered.values()) / sum(map(len, original.values()))
    )
    return result


def init_db(output, identity):
    conn = sqlite3.connect(output / "trials.db", timeout=30)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if tables:
        saved = (
            conn.execute("SELECT value FROM metadata WHERE key='identity'").fetchone() if "metadata" in tables else None
        )
        if saved is None:
            conn.close()
            raise ValueError("existing database lacks v3 identity; choose a new output directory")
        if saved[0] != canonical(identity):
            conn.close()
            raise ValueError("source/input/scoring identity changed; choose a new output directory")
        required_columns = {
            "trial_id",
            "config_key",
            "replicate",
            "level",
            "method",
            "status",
            "created_utc",
            "started_utc",
            "finished_utc",
            "worker",
            "pid",
            "elapsed_s",
            "peak_rss_bytes",
            *METRICS,
            "result_json",
            "error",
        }
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(trials)")}
        if existing_columns != required_columns:
            conn.close()
            raise ValueError("existing database schema is incompatible with v3")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    saved = conn.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if saved and saved[0] != canonical(identity):
        conn.close()
        raise ValueError("source/input/scoring identity changed; choose a new output directory")
    conn.execute("""CREATE TABLE IF NOT EXISTS trials (
        trial_id INTEGER PRIMARY KEY, config_key TEXT NOT NULL, replicate INTEGER NOT NULL,
        level TEXT NOT NULL, method TEXT NOT NULL, status TEXT NOT NULL,
        created_utc TEXT, started_utc TEXT, finished_utc TEXT, worker INTEGER, pid INTEGER,
        elapsed_s REAL, peak_rss_bytes INTEGER, order_gain REAL, order_gain_sd REAL,
        stability REAL, same_data_seed_stability REAL, crossfold_stability REAL, explained REAL,
        dead_fraction REAL, redundancy REAL, fitness REAL, null_sd_mean REAL,
        result_json TEXT, error TEXT, UNIQUE(config_key, replicate))""")
    conn.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)", (canonical(identity),))
    conn.execute("INSERT OR IGNORE INTO metadata VALUES ('next_ordinal','0')")
    conn.execute(
        "UPDATE trials SET status='INTERRUPTED',error='supervisor ended before terminal checkpoint' "
        "WHERE status IN ('PENDING','RUNNING')"
    )
    conn.commit()
    atomic_json(output / "run_manifest.json", identity)
    return conn


def write_summaries(conn, output):
    import pandas as pd

    columns = "trial_id,level,method,config_key,fitness,order_gain,stability,explained,dead_fraction,redundancy"
    frame = pd.read_sql_query(f"SELECT {columns} FROM trials WHERE status='SUCCESS'", conn)
    if frame.empty:
        return
    grouped = frame.groupby(["level", "method", "config_key"], as_index=False).agg(
        fitness_mean=("fitness", "mean"),
        fitness_sd=("fitness", "std"),
        successful_replicates=("fitness", "size"),
        order_gain_mean=("order_gain", "mean"),
        stability_mean=("stability", "mean"),
        explained_mean=("explained", "mean"),
        dead_fraction_mean=("dead_fraction", "mean"),
        redundancy_mean=("redundancy", "mean"),
    )
    grouped = grouped.sort_values(
        ["level", "method", "fitness_mean", "config_key"], ascending=[True, True, False, True]
    )
    top = grouped.groupby(["level", "method"], sort=False).head(10).copy()
    top["rank_within_stratum"] = top.groupby(["level", "method"]).cumcount() + 1
    path = output / "leaderboard.csv"
    temporary = path.with_suffix(".csv.tmp")
    top.to_csv(temporary, index=False)
    os.replace(temporary, path)
    models = output / "top_models"
    models.mkdir(exist_ok=True)
    for (level, method), rows in top.groupby(["level", "method"]):
        folder = models / f"{level}_{method}"
        folder.mkdir(exist_ok=True)
        expected = set()
        for rank, row in enumerate(rows.head(5).itertuples(), 1):
            filename = f"rank_{rank:02}.json"
            expected.add(filename)
            candidates = frame[frame.config_key == row.config_key].copy()
            candidates["distance"] = (candidates.fitness - row.fitness_mean).abs()
            representative = candidates.sort_values(["distance", "trial_id"]).iloc[0]
            atomic_json(
                folder / filename,
                {
                    "config": json.loads(row.config_key),
                    "fitness_mean": row.fitness_mean,
                    "fitness_sd": None if pd.isna(row.fitness_sd) else row.fitness_sd,
                    "successful_replicates": row.successful_replicates,
                    "ranking_scope": [level, method],
                    "representative_trial": int(representative.trial_id),
                    "evidence": json.loads(
                        conn.execute(
                            "SELECT result_json FROM trials WHERE trial_id=?", (int(representative.trial_id),)
                        ).fetchone()[0]
                    ),
                    "confirmation_pending": [
                        "first_order_null",
                        "qualitative_review",
                        "reserved_animal_evaluation",
                        "posthoc_PFF_monomer_and_LDOPA_comparisons",
                    ],
                },
            )
        for old in folder.glob("rank_*.json"):
            if old.name not in expected:
                old.unlink()
    atomic_json(output / "status_counts.json", dict(conn.execute("SELECT status,COUNT(*) FROM trials GROUP BY status")))


def build_identity(args, code_root):
    sessions, pool, reserved, folds, vocab, paths = load_metadata(args.atlas, args.seed)
    selected = sessions[sessions.animal_id.isin(pool)]
    paths += [
        path
        for path in (
            args.atlas / "manifest.json",
            args.atlas / "reviewed_exclusions.csv",
            args.atlas / "validity_audit.csv",
        )
        if path.exists()
    ]
    for level in LEVELS:
        folder = args.atlas / "measurements" if level == "L1" else args.atlas / "clustering" / level / "measurements"
        paths += [folder / "bouts_by_session" / f"{sid}.csv" for sid in selected.session_id]
    return {
        "protocol": "karen-sequence-search-v3",
        "created_date_local": "2026-10-13",
        "atlas": str(args.atlas.resolve()),
        "input_sha256": {str(path.resolve()): digest(path) for path in paths},
        "source_sha256": source_identity(code_root),
        "search_animals": pool,
        "reserved_animals": reserved,
        "folds": folds,
        "vocab": vocab,
        "seed": args.seed,
        "evaluation": {"iterations": args.iterations, "per_animal": args.per_animal, "null_repeats": args.null_repeats},
        "replicates_per_configuration": args.replicates,
        "threads_per_worker": args.threads,
        "space": {"common": COMMON_SPACE, "method": METHOD_SPACE},
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas")},
        },
        "ranking": "within (resolution,method), mean fitness over completed deterministic replicates",
        "fitness": (".4*positive_order_gain+.25*stability+.2*clipped_coverage_or_uncentered_skill"
                    "+.15*alive-.5*max(redundancy-.7,0)"),
        "phenotype_in_selection": False,
        "history_lag_in_discovery_search": False,
        "limits": "shuffle null does not preserve first-order transitions; scores are screening proxies; "
        "reserved animals were examined in prior atlas reports; no causal LDOPA inference",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hours", type=float, default=0.0, help="0 runs until STOP or user termination")
    parser.add_argument("--trial-timeout-s", type=float, default=120.0)
    parser.add_argument("--max-worker-rss-gb", type=float, default=4.0)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--per-animal", type=int, default=700)
    parser.add_argument("--null-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--max-consecutive-failures", type=int, default=20)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.0)
    parser.add_argument("--min-free-ram-gb", type=float, default=8.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(args.workers, args.threads, args.iterations, args.per_animal, args.null_repeats, args.replicates) < 1
        or args.workers * args.threads > len(os.sched_getaffinity(0))
        or args.max_trials < 0
        or args.hours < 0
        or min(args.trial_timeout_s, args.max_worker_rss_gb) <= 0
    ):
        raise ValueError("positive bounded resources required; thread budget exceeds available CPUs")
    for name in THREAD_VARIABLES:
        os.environ[name] = str(args.threads)
    print(
        canonical(
            {
                "status": "execute" if args.execute else "plan",
                "workers": args.workers,
                "threads": args.threads,
                "hours": args.hours,
                "atlas": str(args.atlas),
                "output": str(args.output),
            }
        ),
        flush=True,
    )
    if not args.execute:
        return
    from fpcode.common.trial_runner import run_trials

    code_root = Path(__file__).resolve().parents[3]
    args.atlas, args.output = args.atlas.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    stop = {"requested": False}
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.update(requested=True))
    with (args.output / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = build_identity(args, code_root)
        conn = init_db(args.output, identity)
        buckets = configuration_buckets(args.seed)
        expected_files = {**identity["source_sha256"], **identity["input_sha256"]}
        source_stats = {path: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in expected_files}
        scheduled = 0
        deadline = time.monotonic() + args.hours * 3600 if args.hours else None
        resources = {"next_check": 0.0, "can_schedule": True}

        def refresh_resources():
            if time.monotonic() < resources["next_check"]:
                return
            resources["next_check"] = time.monotonic() + 1.0
            if shutil.disk_usage(args.output).free < args.min_free_disk_gb * 1024**3:
                if not stop["requested"]:
                    print("STOP: free disk below configured safety threshold; checkpointing", flush=True)
                stop["requested"] = True
                stop["reason"] = "disk_safety"
            import psutil

            available = psutil.virtual_memory().available >= args.min_free_ram_gb * 1024**3
            if available != resources["can_schedule"]:
                print("RAM recovered: resuming dispatch" if available else "Low RAM: pausing new trials", flush=True)
            resources["can_schedule"] = available

        def should_stop():
            refresh_resources()
            return stop["requested"]

        def can_schedule():
            refresh_resources()
            return resources["can_schedule"] and not stop["requested"]

        def jobs():
            nonlocal scheduled
            while not stop["requested"] and (args.max_trials == 0 or scheduled < args.max_trials):
                if any(
                    (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) != saved
                    for path, saved in source_stats.items()
                ):
                    raise RuntimeError("source/input changed during search; stop and use a new run identity")
                ordinal = int(conn.execute("SELECT value FROM metadata WHERE key='next_ordinal'").fetchone()[0])
                cfg, replicate = configuration_at(ordinal, buckets, args.replicates)
                trial_id = ordinal + 1
                conn.execute(
                    "INSERT INTO trials (trial_id,config_key,replicate,level,method,status,created_utc) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        trial_id,
                        canonical(cfg),
                        replicate,
                        cfg["level"],
                        cfg["method"],
                        "PENDING",
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.execute("UPDATE metadata SET value=? WHERE key='next_ordinal'", (str(ordinal + 1),))
                conn.commit()
                scheduled += 1
                yield {
                    "trial_id": trial_id,
                    "configuration": cfg,
                    "replicate": replicate,
                    "settings": identity["evaluation"],
                }

        def on_start(job, worker_id, pid):
            conn.execute(
                "UPDATE trials SET status='RUNNING',started_utc=?,worker=?,pid=? WHERE trial_id=?",
                (datetime.now(timezone.utc).isoformat(), worker_id, pid, job["trial_id"]),
            )
            conn.commit()
            print(
                f"START #{job['trial_id']} w{worker_id} pid{pid} "
                f"{canonical(job['configuration'])} rep{job['replicate']}",
                flush=True,
            )

        runner = run_trials(
            jobs(),
            evaluate_job,
            initializer=initialize_worker,
            initargs=(str(args.atlas), args.seed, expected_files),
            workers=args.workers,
            timeout_s=args.trial_timeout_s,
            initializer_timeout_s=120.0,
            max_worker_rss_bytes=int(args.max_worker_rss_gb * 1024**3),
            stop_file=args.output / "STOP",
            stop_requested=should_stop,
            can_schedule=can_schedule,
            deadline_monotonic=deadline,
            on_start=on_start,
        )
        completed, failures = 0, 0
        try:
            for outcome in runner:
                result = outcome.result or {}
                columns = ",".join(f"{name}=?" for name in METRICS)
                conn.execute(
                    f"UPDATE trials SET status=?,finished_utc=?,elapsed_s=?,peak_rss_bytes=?,{columns},"
                    "result_json=?,error=? WHERE trial_id=?",
                    (
                        outcome.status,
                        datetime.now(timezone.utc).isoformat(),
                        outcome.elapsed_s,
                        outcome.peak_rss_bytes,
                        *[result.get(name) for name in METRICS],
                        canonical(result) if result else None,
                        outcome.error,
                        outcome.job["trial_id"],
                    ),
                )
                conn.commit()
                completed += 1
                failures = 0 if outcome.status == "SUCCESS" else failures + 1
                score = (
                    f"fitness={result['fitness']:.4f} gain={result['order_gain']:.4f}"
                    if result
                    else outcome.error[-300:]
                )
                print(
                    f"DONE #{outcome.job['trial_id']} {outcome.status} {outcome.elapsed_s:.2f}s "
                    f"RSS={outcome.peak_rss_bytes} {score}",
                    flush=True,
                )
                if completed % 8 == 0:
                    write_summaries(conn, args.output)
                if failures >= args.max_consecutive_failures:
                    raise RuntimeError(f"{failures} consecutive unsuccessful trials; refusing an unproductive run")
        finally:
            runner.close()
            conn.execute(
                "UPDATE trials SET status='INTERRUPTED',error='supervisor cleanup' "
                "WHERE status IN ('PENDING','RUNNING')"
            )
            conn.commit()
            write_summaries(conn, args.output)
            atomic_json(
                args.output / "completion.json",
                {
                    "completed_this_invocation": completed,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "stop_requested": stop["requested"],
                    "stop_reason": stop.get("reason", "user_or_signal" if stop["requested"] else ""),
                    "stop_file": (args.output / "STOP").exists(),
                    "deadline_reached": deadline is not None and time.monotonic() >= deadline,
                },
            )
            conn.close()
        print(f"Search supervisor finished; {completed} terminal checkpoints persisted", flush=True)


if __name__ == "__main__":
    main()
