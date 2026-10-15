"""Prepare and explicitly run the bounded Karen September-reference seqNMF panel.

Every candidate is a single pooled fit over all99recordings/all21mice/all waves.
No KPMS fitting, notifications, held-out claims or group-effect selection.
Import/help/plan do not load inputs or initialize an accelerator.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

WAVES = ("3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa")
LAMBDAS = (0., .1, .3, 1., 3.)
SEEDS = (42, 43, 44)


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard(args):
    import psutil

    rss = psutil.Process().memory_info().rss
    if rss > args.memory_gb * 1024**3:
        raise MemoryError("RSS guard exceeded")
    if shutil.disk_usage(args.output).free < args.min_free_disk_gb * 1024**3:
        raise RuntimeError("free disk guard reached")
    size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
    if size > args.max_output_gb * 1024**3:
        raise RuntimeError("output-size guard reached")


def frame_bouts(frames, labels, excluded):
    """Independent RLE preserving invalid observations and dropped native frames."""
    import numpy as np
    import pandas as pd

    if len(frames) != len(labels) or len(labels) != len(excluded) or np.any(np.diff(frames) <= 0):
        raise ValueError("invalid native frame/label arrays")
    valid = np.isfinite(labels) & (labels >= 0) & (labels < 100) & ~excluded
    token = np.where(valid, labels, -1).astype(int)
    adjacency = (np.diff(frames) == 1) & valid[:-1] & valid[1:]
    starts = np.flatnonzero(np.r_[True, (token[1:] != token[:-1]) | ~adjacency])
    ends = np.r_[starts[1:], len(token)]
    keep = valid[starts]
    starts, ends = starts[keep], ends[keep]
    # Blocks represent uninterrupted valid support, independent of token changes.
    blocks = np.cumsum(np.r_[True, ~adjacency])
    return pd.DataFrame({"token": token[starts], "start_frame": frames[starts],
                         "end_frame_exclusive": frames[ends - 1] + 1, "block": blocks[starts]})


def merge_bouts(table, mapping):
    import numpy as np
    import pandas as pd

    tokens = mapping[table.token.to_numpy(int)]
    same = (tokens[1:] == tokens[:-1]) & (table.block.to_numpy()[1:] == table.block.to_numpy()[:-1])
    same &= table.start_frame.to_numpy()[1:] == table.end_frame_exclusive.to_numpy()[:-1]
    starts = np.flatnonzero(np.r_[True, ~same])
    ends = np.r_[starts[1:], len(table)]
    out = table.iloc[starts].copy().reset_index(drop=True)
    out["token"] = tokens[starts]
    out["end_frame_exclusive"] = table.end_frame_exclusive.to_numpy()[ends - 1]
    return pd.DataFrame(out)


def pack_chunks(chunks, mapping, vocabulary, length):
    import numpy as np
    import pandas as pd

    tokens, valid, references = [], [], []
    for chunk in chunks:
        chunk = merge_bouts(chunk, mapping)
        count = len(chunk)
        tokens.extend(chunk.token.to_numpy(int))
        valid.extend([True] * max(0, count - length + 1) + [False] * min(count, length - 1))
        references.extend(chunk.to_dict("records"))
        tokens.extend([-1] * length)
        valid.extend([False] * length)
        references.extend([{} for _ in range(length)])
    tokens, valid = np.asarray(tokens, int), np.asarray(valid, bool)
    x = np.zeros((vocabulary, len(tokens)), np.float32)
    good = np.flatnonzero(tokens >= 0)
    x[tokens[good], good] = 1
    if not valid.any():
        raise ValueError("no supported motif onsets")
    return x, valid, tokens, pd.DataFrame(references)


def prepare(args):
    import numpy as np
    import pandas as pd

    from fpcode.workflows.sequence_atlas import validate_animal_splits

    sessions_path = args.view / "sessions.csv"
    split_path = args.view / "chains/animal_splits.json"
    mask_path = args.view / "reviewed_exclusions.csv"
    sessions = pd.read_csv(sessions_path, dtype={"animal_id": str})
    splits = json.loads(split_path.read_text())
    validate_animal_splits(sessions, splits)
    if [len(splits[k]) for k in ("train", "validation", "test")] != [12, 4, 5]:
        raise ValueError("unexpected Karen animal split")
    sessions["split"] = "pooled_fit"
    selected = sessions[sessions.condition.isin(WAVES)].copy()
    if len(selected) != 99 or selected.animal_id.nunique() != 21:
        raise ValueError("pooled fit must include all99recordings/all21mice")
    exclusions = pd.read_csv(mask_path)
    source_rows, tables = [], {}
    for meta in selected.itertuples():
        guard(args)
        path = args.view / "frame_cache" / (meta.session_id + ".npz")
        with np.load(path, allow_pickle=False) as cache:
            frames, labels = cache["frames"], cache["labels"]
            if float(cache["fps"]) != 60.:
                raise ValueError("Karen native clock changed")
            expected = np.zeros(len(frames), bool)
            for interval in exclusions[exclusions.session_id == meta.session_id].itertuples():
                expected |= (frames >= interval.start_frame) & (frames < interval.end_frame_exclusive)
            actual = cache["analysis_excluded"] if "analysis_excluded" in cache.files else np.zeros(len(frames), bool)
            if not np.array_equal(expected, actual):
                raise ValueError(f"reviewed exclusion mask mismatch: {meta.session_id}")
        table = frame_bouts(frames, labels, expected)
        for name in ("session_id", "animal_id", "condition", "group", "split"):
            table[name] = getattr(meta, name)
        tables[meta.session_id] = table
        identity = hashlib.sha256()
        for array in (frames, labels, expected):
            identity.update(str((array.shape, str(array.dtype))).encode())
            identity.update(np.ascontiguousarray(array).tobytes())
        source_rows.append({"session_id": meta.session_id, "animal_id": meta.animal_id, "split": meta.split,
                            "path": str(path), "bytes": path.stat().st_size,
                            "relevant_frame_label_mask_sha256": identity.hexdigest(), "bouts": len(table)})
        log(f"PREPARE {len(source_rows)}/{len(selected)} {meta.session_id}: {len(table)} valid native bouts")
    chunks, audit = [], []
    for session_id, table in tables.items():
        for _, block in table.groupby("block", sort=False):
            block = block.copy()
            block["chunk_id"] = len(chunks)
            chunks.append(block)
        audit.append({"session_id": session_id, "animal_id": str(table.animal_id.iloc[0]),
                      "condition": str(table.condition.iloc[0]), "included_bouts": len(table)})
    mapping = np.asarray(json.loads(args.mapping.read_text()), dtype=int)
    if mapping.shape != (100,) or mapping.min() < 0:
        raise ValueError("invalid existing grouped mapping")
    # Unknown/dump IDs must remain individually distinct in this existing variant.
    prepared = {}
    for level, current in (("L1", np.arange(100)), ("L3_unresolved_separate", mapping)):
        folder = args.output / "prepared" / level
        folder.mkdir(parents=True)
        vocabulary = int(current.max() + 1)
        xt, vt, tt, rt = pack_chunks(chunks, current, vocabulary, args.length)
        np.savez_compressed(folder / "matrices.npz", fit_x=xt, fit_valid=vt, fit_tokens=tt)
        rt.to_csv(folder / "fit_references.csv.gz", index=False)
        write_json(folder / "mapping.json", current.tolist())
        prepared[level] = {"vocabulary": vocabulary, "fit_columns": len(tt), "fit_valid_starts": int(vt.sum()),
                           "matrix_sha256": digest(folder / "matrices.npz"),
                           "fit_reference_sha256": digest(folder / "fit_references.csv.gz")}
    pd.DataFrame(audit).to_csv(args.output / "included_recordings.csv", index=False)
    recipe = {"factors": args.factors, "length_bouts": args.length, "iterations": args.iterations,
              "lambdas": LAMBDAS, "seeds": SEEDS,
              "training_threshold": "per factor 90th percentile of supported pooled-fit activations >1e-4",
              "detector": {"max_error": .68, "min_stage_support": .5, "stage_weight": .15, "nms_radius_bouts": 2},
              "historical_split_unused_for_this_fit": splits, "waves": WAVES, "all21mice_included": True,
              "fit_design": "one pooled generator per candidate over all99recordings; includes L-DOPA; no held-out evaluation",
              "sampling": "all eligible native bouts; no bout cap or animal sampling; actual support blocks separated by masked padding",
              "input_metadata": [{"path": str(p), "sha256": digest(p)} for p in (sessions_path, split_path, mask_path, args.mapping)],
              "source_arrays": source_rows, "prepared": prepared,
              "code": {str(p): digest(p) for p in (Path(__file__), Path(__file__).with_name("reference_seqnmf.py"))}}
    write_json(args.output / "recipe.json", recipe)
    log(f"PREPARED {len(source_rows)} recordings/all21mice/all5waves for a pooled fit; no held-out claim")
    return recipe


def evaluate(w, h, tokens, valid, references, thresholds):
    import numpy as np
    import pandas as pd

    from .reference_seqnmf import reconstruction, retrieve

    x = np.zeros((w.shape[0], len(tokens)), np.float32)
    good = np.flatnonzero(tokens >= 0)
    x[tokens[good], good] = 1
    residual = reconstruction(w, h) - x
    errors = []
    for animal in sorted(references.animal_id.dropna().unique()):
        mask = references.animal_id.eq(animal).to_numpy()
        errors.append(float(np.sqrt(np.sum(residual[:, mask]**2) / mask.sum())))
    detected = retrieve(tokens, valid, w, h, thresholds)
    rows, covered = [], np.zeros(len(tokens), bool)
    for event in detected:
        a, b = event["start"], event["end_exclusive"]
        first, last = references.iloc[a], references.iloc[b - 1]
        window = references.iloc[a:b]
        if window.chunk_id.nunique() != 1 or window.session_id.nunique() != 1:
            raise AssertionError("detector crossed a chunk/recording barrier")
        starts, ends = window.start_frame.to_numpy(), window.end_frame_exclusive.to_numpy()
        if not np.array_equal(starts[1:], ends[:-1]):
            raise AssertionError("detector crossed excluded/native missing frames")
        covered[a:b] = True
        rows.append({**event, "animal_id": str(first.animal_id), "session_id": first.session_id,
                     "condition": first.condition, "start_frame": int(first.start_frame),
                     "end_frame_exclusive": int(last.end_frame_exclusive),
                     "duration_sec": float((last.end_frame_exclusive - first.start_frame) / 60.)})
    # Coverage is union physical time over all frozen native support blocks.
    duration = (references.end_frame_exclusive - references.start_frame).fillna(0).to_numpy() / 60.
    count_by_factor = {k: [row for row in rows if row["factor"] == k] for k in range(w.shape[1])}
    flat = w.transpose(1, 0, 2).reshape(w.shape[1], -1)
    flat = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)
    similarity = flat @ flat.T
    np.fill_diagonal(similarity, 0)
    metrics = {"fit_animal_mean_relative_error": float(np.mean(errors)),
               "fit_union_time_coverage": float(duration[covered].sum() / duration.sum()),
               "fit_union_bout_coverage": float(covered[good].mean()), "fit_calls": len(rows),
               "fit_mean_template_error": float(np.mean([r["template_error"] for r in rows])) if rows else None,
               "template_max_redundancy": float(similarity.max()),
               "factors_with_5_calls_2_mice": sum(len(v) >= 5 and len({r["animal_id"] for r in v}) >= 2 for v in count_by_factor.values())}
    columns = ["factor", "start", "end_exclusive", "activation", "template_error", "stage_support", "observed_tokens",
               "animal_id", "session_id", "condition", "start_frame", "end_frame_exclusive", "duration_sec"]
    return metrics, pd.DataFrame(rows, columns=columns)


def preflight(args, recipe):
    import numpy as np

    from .reference_seqnmf import backend, fit

    with np.load(args.output / "prepared/L1/matrices.npz", allow_pickle=False) as saved:
        x, valid = saved["fit_x"], saved["fit_valid"]
    records, arrays = {}, {}
    for device in ("cpu", "gpu"):
        runtime = backend(device)
        # First invocation includes compilation; second checks sustained cost.
        for phase in ("warmup", "timed"):
            start = time.monotonic()
            w, h, _ = fit(x, valid, factors=args.factors, length=args.length, penalty=1., iterations=8,
                           seed=42, runtime=runtime)
            records[f"{device}_{phase}_sec"] = time.monotonic() - start
        arrays[device] = (w, h)
        records[device + "_backend"] = runtime[2]
    errors = {name: float(np.max(np.abs(a - b))) for name, a, b in
              zip(("W_max_abs", "H_max_abs"), arrays["cpu"], arrays["gpu"])}
    for a, b in zip(arrays["cpu"], arrays["gpu"]):
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-4)
    chosen = "gpu" if records["gpu_timed_sec"] < records["cpu_timed_sec"] else "cpu"
    result = {"status": "PASS", "iterations_per_check": 8, "dtype": "float32",
              "atol": 3e-5, "rtol": 3e-4, **records, **errors, "recommended_device": chosen,
              "limits": "Bounded numerical/runtime check, not full-fit backend equivalence or convergence."}
    write_json(args.output / "backend_preflight.json", result)
    log(f"BACKEND_PREFLIGHT {json.dumps(result)}")


def run_panel(args, recipe):
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    from .reference_seqnmf import backend, fit

    for path, expected in recipe["code"].items():
        if digest(path) != expected:
            raise ValueError("fitting source changed after input/recipe freeze")
    for key in ("factors", "iterations"):
        if recipe[key] != getattr(args, key):
            raise ValueError("requested fit differs from frozen recipe")
    if recipe["length_bouts"] != args.length:
        raise ValueError("requested length differs from frozen recipe")
    runtime = backend(args.device)
    write_json(args.output / "runtime_backend.json", runtime[2])
    log(f"FIT_START device={runtime[2]} trials=30 K={args.factors} L={args.length} iterations={args.iterations}")
    results, saved_templates = [], {}
    for level, identity in recipe["prepared"].items():
        prepared = args.output / "prepared" / level
        if digest(prepared / "matrices.npz") != identity["matrix_sha256"]:
            raise ValueError("frozen prepared matrix changed")
        if digest(prepared / "fit_references.csv.gz") != identity["fit_reference_sha256"]:
            raise ValueError("frozen reference table changed")
        with np.load(prepared / "matrices.npz", allow_pickle=False) as p:
            xt, vt, tokens = p["fit_x"], p["fit_valid"], p["fit_tokens"]
        references = pd.read_csv(prepared / "fit_references.csv.gz", dtype={"animal_id": str})
        for penalty in LAMBDAS:
            for seed in SEEDS:
                guard(args)
                trial = f"{level}_lambda{penalty:g}_seed{seed}"
                folder = args.output / "trials" / trial
                folder.mkdir(parents=True)
                start = time.monotonic()
                log(f"TRIAL {len(results)+1}/30 {trial} START")

                def progress(iteration, error):
                    guard(args)
                    log(f"TRIAL {trial} iteration={iteration}/{args.iterations} training_relative_error={error:.6f}")

                w, h, history = fit(xt, vt, factors=args.factors, length=args.length, penalty=penalty,
                                    iterations=args.iterations, seed=seed, runtime=runtime, progress=progress)
                thresholds = np.array([np.percentile(row[vt & (row > 1e-4)], 90)
                                       if np.any(vt & (row > 1e-4)) else np.inf for row in h])
                if np.any(h[:, ~vt] != 0):
                    raise AssertionError("unsupported onset acquired activation")
                metrics, calls = evaluate(w, h, tokens, vt, references, thresholds)
                np.savez_compressed(folder / "model.npz", W=w, pooled_H=h,
                                    thresholds=thresholds, factors=args.factors, length=args.length,
                                    penalty=penalty, seed=seed)
                calls.to_csv(folder / "pooled_calls.csv.gz", index=False)
                row = {"trial": trial, "level": level, "penalty": penalty, "seed": seed, **metrics,
                       "elapsed_sec": time.monotonic() - start, "directory": str(folder)}
                write_json(folder / "result.json", {**row, "history": history, "recipe_sha256": digest(args.output / "recipe.json")})
                results.append(row)
                saved_templates[(level, penalty, seed)] = w.copy()
                pd.DataFrame(results).to_csv(args.output / "results.csv", index=False)
                write_json(args.output / "status.json", {"state": "RUNNING", "completed_trials": len(results), "total_trials": 30, "last_trial": trial})
                log(f"TRIAL {trial} COMPLETE {json.dumps(row)}")
    stability = []
    for level in recipe["prepared"]:
        for penalty in LAMBDAS:
            for i, first in enumerate(SEEDS):
                for second in SEEDS[i+1:]:
                    a, b = [saved_templates[(level, penalty, seed)].transpose(1, 0, 2).reshape(args.factors, -1)
                            for seed in (first, second)]
                    a, b = [value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8) for value in (a, b)]
                    sim = a @ b.T
                    ii, jj = linear_sum_assignment(-sim)
                    stability.append({"level": level, "penalty": penalty, "first_seed": first,
                                      "second_seed": second, "matched_template_cosine": float(sim[ii, jj].mean())})
    pd.DataFrame(stability).to_csv(args.output / "seed_stability.csv", index=False)
    write_json(args.output / "status.json", {"state": "COMPLETE", "completed_trials": 30, "total_trials": 30,
                                              "limits": "Pooled in-sample fit diagnostics and seed stability, not held-out validation. Reconstruction errors across vocabularies have different targets. No winner chosen by group effects or qualitative acceptance."})
    log("PANEL COMPLETE: 30/30 pooled fits; parameter/result pairs, weights, inputs and calls saved; no held-out claims")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "preflight", "fit"), required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--factors", type=int, default=10)
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-gb", type=float, default=8.)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.)
    parser.add_argument("--max-output-gb", type=float, default=4.)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"stage": args.stage, "trials": 30, "lambdas": LAMBDAS, "seeds": SEEDS,
                          "factors": args.factors, "length": args.length, "output": str(args.output), "execute": False}, indent=2))
        return
    if args.factors < 1 or args.length < 2 or args.iterations < 1 or args.threads < 1:
        parser.error("invalid positive fitting settings")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".25")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    args.view, args.mapping, args.output = args.view.resolve(), args.mapping.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        guard(args)
        if args.stage == "prepare":
            if (args.output / "recipe.json").exists():
                raise ValueError("refusing to overwrite frozen recipe")
            prepare(args)
        else:
            recipe = json.loads((args.output / "recipe.json").read_text())
            if args.stage == "preflight":
                preflight(args, recipe)
            else:
                run_panel(args, recipe)
    except BaseException as error:
        write_json(args.output / "status.json", {"state": "FAILED", "error": repr(error)})
        log(f"FAILED {error!r}")
        raise


if __name__ == "__main__":
    main()
