"""Execute the pooled Karen K/lambda sweep and quantitative family diagnostics.

Explicit execution only. Reuses immutable all99-recording inputs and K10 fits;
all measures are descriptive robustness diagnostics, not held-out validation.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

KS = (6, 10, 14, 18, 22)
PENALTIES = (0.3, 3.0)
SEEDS = (42, 43, 44)


def levels(args):
    return json.loads((args.base / "recipe.json").read_text())["prepared"]


def inputs(args, level):
    import numpy as np
    import pandas as pd

    original = levels(args)[level]
    folder = args.base / "prepared" / level
    for filename, key in (("matrices.npz", "matrix_sha256"), ("fit_references.csv.gz", "fit_reference_sha256")):
        if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != original[key]:
            raise ValueError("frozen pooled input changed")
    with np.load(folder / "matrices.npz", allow_pickle=False) as p:
        x, valid, tokens = p["fit_x"], p["fit_valid"], p["fit_tokens"]
    refs = pd.read_csv(folder / "fit_references.csv.gz", dtype={"animal_id": str})
    duration = ((refs.end_frame_exclusive - refs.start_frame) / 60.0).fillna(0).to_numpy()
    return x, valid, tokens, refs, duration


def fit_sweep(args):
    import numpy as np
    import pandas as pd

    from .reference_seqnmf import backend, fit
    from .run_reference_seqnmf import digest, evaluate, guard, log, write_json

    runtime = backend(args.device)
    write_json(args.output / "backend.json", runtime[2])
    old = pd.read_csv(args.base / "results.csv")
    results = []
    for level in levels(args):
        x, valid, tokens, refs, _ = inputs(args, level)
        for k in KS:
            for penalty in PENALTIES:
                for seed in SEEDS:
                    guard(args)
                    name = f"{level}_K{k}_lambda{penalty:g}_seed{seed}"
                    if k == 10:
                        row = (
                            old[(old.level == level) & (old.penalty == penalty) & (old.seed == seed)].iloc[0].to_dict()
                        )
                        folder = Path(row["directory"])
                        saved = json.loads((folder / "result.json").read_text())
                        row.update(
                            trial=name,
                            components=k,
                            source="reused_K10",
                            global_relative_reconstruction_error=saved["history"][-1]["relative_error"],
                        )
                        log(f"REUSE {len(results) + 1}/60 {name}")
                    else:
                        folder = args.output / "trials" / name
                        folder.mkdir(parents=True)
                        start = time.monotonic()
                        log(f"FIT {len(results) + 1}/60 {name} START")

                        def progress(iteration, error):
                            guard(args)
                            log(f"FIT {name} iteration={iteration}/80 error={error:.6f}")

                        w, h, history = fit(
                            x,
                            valid,
                            factors=k,
                            length=3,
                            penalty=penalty,
                            iterations=80,
                            seed=seed,
                            runtime=runtime,
                            progress=progress,
                        )
                        if not np.isfinite(w).all() or not np.isfinite(h).all() or np.any(h[:, ~valid] != 0):
                            raise AssertionError("nonfinite model or unsupported activation")
                        thresholds = np.array(
                            [
                                np.percentile(row[valid & (row > 1e-4)], 90) if np.any(valid & (row > 1e-4)) else np.inf
                                for row in h
                            ]
                        )
                        metrics, calls = evaluate(w, h, tokens, valid, refs, thresholds)
                        np.savez_compressed(
                            folder / "model.npz",
                            W=w,
                            pooled_H=h,
                            thresholds=thresholds,
                            factors=k,
                            length=3,
                            penalty=penalty,
                            seed=seed,
                        )
                        calls.to_csv(folder / "pooled_calls.csv.gz", index=False)
                        row = dict(
                            trial=name,
                            level=level,
                            components=k,
                            penalty=penalty,
                            seed=seed,
                            source="new_fit",
                            directory=str(folder),
                            **metrics,
                            global_relative_reconstruction_error=history[-1]["relative_error"],
                            elapsed_sec=time.monotonic() - start,
                        )
                        write_json(
                            folder / "result.json",
                            {
                                **row,
                                "history": history,
                                "capacity_recipe_sha256": digest(args.output / "capacity_recipe.json"),
                            },
                        )
                        log(f"FIT {name} COMPLETE coverage={row['fit_union_time_coverage']:.4f}")
                    row["uncentred_reconstruction_score"] = 1 - row["global_relative_reconstruction_error"] ** 2
                    row["model_sha256"] = digest(folder / "model.npz")
                    row["calls_sha256"] = digest(folder / "pooled_calls.csv.gz")
                    results.append(row)
                    pd.DataFrame(results).to_csv(args.output / "results.csv", index=False)
                    write_json(
                        args.output / "status.json",
                        {
                            "state": "RUNNING",
                            "stage": "fit",
                            "indexed_configurations": len(results),
                            "total_configurations": 60,
                        },
                    )
    log("SWEEP FIT COMPLETE: 60 configurations,48new and12reused")


def diagnose(args):
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    from .reference_capacity import execution_masks, execution_overlap, template_cosines
    from .reference_seqnmf import retrieve
    from .run_reference_seqnmf import guard, log

    results = pd.read_csv(args.output / "results.csv")
    family_rows, pair_rows, stable_rows, split_rows = [], [], [], []
    for level in levels(args):
        _, valid, tokens, refs, duration = inputs(args, level)
        models = {}
        for row in results[results.level == level].to_dict("records"):
            guard(args)
            folder = Path(row["directory"])
            with np.load(folder / "model.npz", allow_pickle=False) as p:
                w, h, thresholds = p["W"], p["pooled_H"], p["thresholds"]
            calls = pd.read_csv(folder / "pooled_calls.csv.gz", dtype={"animal_id": str})
            a, b = calls.start.to_numpy(int), calls.end_exclusive.to_numpy(int)
            if not (np.all(valid[a]) and np.all(b - a == 3)):
                raise AssertionError("invalid saved call windows")
            for lag in (1, 2):
                if not np.array_equal(
                    refs.start_frame.to_numpy()[a + lag], refs.end_frame_exclusive.to_numpy()[a + lag - 1]
                ):
                    raise AssertionError("persisted call crosses native gap")
            if not (
                np.array_equal(calls.start_frame.to_numpy(), refs.start_frame.to_numpy()[a])
                and np.array_equal(calls.end_frame_exclusive.to_numpy(), refs.end_frame_exclusive.to_numpy()[b - 1])
            ):
                raise AssertionError("persisted native endpoints differ")
            # Include all activation/NMS candidates to show error distributions
            # before the acceptance cutoff, not just the truncated accepted tail.
            candidates = retrieve(tokens, valid, w, h, thresholds, max_error=1.00001, min_support=0.0)
            by_factor = {k: [] for k in range(w.shape[1])}
            for event in candidates:
                by_factor[event["factor"]].append(event["template_error"])
            for k in range(w.shape[1]):
                sub = calls[calls.factor == k]
                counts = sub.animal_id.value_counts().to_numpy(float)
                effective = 1 / np.sum((counts / counts.sum()) ** 2) if len(counts) else 0.0
                entry = dict(
                    trial=row["trial"],
                    level=level,
                    components=row["components"],
                    penalty=row["penalty"],
                    seed=row["seed"],
                    factor=k,
                    calls=len(sub),
                    mice=sub.animal_id.nunique(),
                    effective_mice=effective,
                    activation_candidates=len(by_factor[k]),
                    acceptance_fraction=len(sub) / max(1, len(by_factor[k])),
                )
                for label, values in (
                    ("accepted_error", sub.template_error.to_numpy()),
                    ("candidate_error", np.asarray(by_factor[k])),
                ):
                    for q in (0.1, 0.5, 0.9):
                        entry[f"{label}_p{int(100 * q)}"] = float(np.quantile(values, q)) if len(values) else np.nan
                family_rows.append(entry)
            masks = execution_masks(calls, int(row["components"]), len(tokens))
            locked, shifted = template_cosines(w, w)
            jac, _, _ = execution_overlap(masks, masks, duration)
            for i in range(len(w[0])):
                for j in range(i + 1, len(w[0])):
                    pair_rows.append(
                        dict(
                            trial=row["trial"],
                            level=level,
                            components=row["components"],
                            penalty=row["penalty"],
                            seed=row["seed"],
                            first_factor=i,
                            second_factor=j,
                            template_cosine=locked[i, j],
                            shifted_template_cosine=shifted[i, j],
                            execution_time_jaccard=jac[i, j],
                        )
                    )
            models[(int(row["components"]), row["penalty"], int(row["seed"]))] = (row["trial"], w, masks)
            log(f"DIAGNOSTICS {row['trial']} calls={len(calls)} families={w.shape[1]}")
        for k in KS:
            for penalty in PENALTIES:
                for index, first in enumerate(SEEDS):
                    for second in SEEDS[index + 1 :]:
                        id_a, wa, ma = models[(k, penalty, first)]
                        id_b, wb, mb = models[(k, penalty, second)]
                        locked, shifted = template_cosines(wa, wb)
                        overlap, _, _ = execution_overlap(ma, mb, duration)
                        ii, jj = linear_sum_assignment(-shifted)
                        for i, j in zip(ii, jj):
                            stable_rows.append(
                                dict(
                                    level=level,
                                    components=k,
                                    penalty=penalty,
                                    trial_a=id_a,
                                    trial_b=id_b,
                                    first_factor=int(i),
                                    second_factor=int(j),
                                    template_cosine=locked[i, j],
                                    shifted_template_cosine=shifted[i, j],
                                    execution_time_jaccard=overlap[i, j],
                                )
                            )
        # No one-to-one assignment across K: retain all edges for split/merge review.
        for previous, following in zip(KS[:-1], KS[1:]):
            for penalty in PENALTIES:
                for seed in SEEDS:
                    id_a, wa, ma = models[(previous, penalty, seed)]
                    id_b, wb, mb = models[(following, penalty, seed)]
                    locked, shifted = template_cosines(wa, wb)
                    overlap, in_a, in_b = execution_overlap(ma, mb, duration)
                    for i in range(previous):
                        for j in range(following):
                            split_rows.append(
                                dict(
                                    level=level,
                                    penalty=penalty,
                                    seed=seed,
                                    from_K=previous,
                                    to_K=following,
                                    trial_a=id_a,
                                    trial_b=id_b,
                                    parent_factor=i,
                                    child_factor=j,
                                    template_cosine=locked[i, j],
                                    shifted_template_cosine=shifted[i, j],
                                    execution_time_jaccard=overlap[i, j],
                                    fraction_parent_time_in_child=in_a[i, j],
                                    fraction_child_time_in_parent=in_b[i, j],
                                )
                            )
        del models
    families, stability = pd.DataFrame(family_rows), pd.DataFrame(stable_rows)
    pd.DataFrame(pair_rows).to_csv(args.output / "redundancy_pairs.csv", index=False)
    stability.to_csv(args.output / "seed_family_matching.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.output / "split_merge_edges.csv.gz", index=False)
    for position, row in results.iterrows():
        subset = families[families.trial == row.trial]
        results.loc[position, "supported_families_20calls_5mice"] = int(
            ((subset.calls >= 20) & (subset.mice >= 5)).sum()
        )
        left = stability[stability.trial_a == row.trial].rename(columns={"first_factor": "own_factor"})
        right = stability[stability.trial_b == row.trial].rename(columns={"second_factor": "own_factor"})
        pair_values = pd.concat([left, right]).groupby("own_factor").shifted_template_cosine.mean()
        results.loc[position, "matched_seed_template_median"] = pair_values.median()
        results.loc[position, "best5_matched_seed_template_mean"] = pair_values.nlargest(5).mean()
        results.loc[position, "median_family_accepted_error"] = subset.accepted_error_p50.median()
        for factor, value in pair_values.items():
            families.loc[(families.trial == row.trial) & (families.factor == factor), "matched_seed_template_mean"] = (
                value
            )
    families.to_csv(args.output / "family_metrics.csv", index=False)
    results.to_csv(args.output / "capacity_metrics.csv", index=False)
    shortlist = []
    for level, subset in results.groupby("level", sort=False):
        # Three transparent views, not an arbitrary combined quality score.
        winners = [
            subset.sort_values("fit_union_time_coverage", ascending=False).iloc[0],
            subset.sort_values("best5_matched_seed_template_mean", ascending=False).iloc[0],
            subset[subset.components == 10].sort_values("fit_union_time_coverage", ascending=False).iloc[0],
        ]
        unique = {}
        for role, row in zip(("coverage", "stable_families", "compact_K10_reference"), winners):
            if row.trial not in unique:
                unique[row.trial] = {**row.to_dict(), "shortlist_role": role}
        shortlist.extend(unique.values())
    pd.DataFrame(shortlist).to_csv(args.output / "null_shortlist.csv", index=False)
    log(f"DIAGNOSTICS COMPLETE: {len(families)} per-family rows; {len(shortlist)} models for paired order-null checks")


def run_nulls(args):
    import numpy as np
    import pandas as pd

    from .reference_capacity import order_null
    from .reference_seqnmf import backend, retrieve, update
    from .run_reference_seqnmf import guard, log

    xp, _, _ = backend(args.device)
    if args.device == "gpu":
        import jax

        step = jax.jit(lambda w, h, x, v, p: update(w, h, x, v, p, templates=False, xp=xp))
    else:

        def step(w, h, x, v, p):
            return update(w, h, x, v, p, templates=False)

    shortlist = pd.read_csv(args.output / "null_shortlist.csv")
    rows, audits = [], []
    for level, group in shortlist.groupby("level", sort=False):
        x, valid, tokens, _, duration = inputs(args, level)
        models = []
        for row in group.to_dict("records"):
            with np.load(Path(row["directory"]) / "model.npz", allow_pickle=False) as p:
                models.append((row, p["W"], p["thresholds"]))
        vv = xp.asarray(valid)
        # Paired surrogates are shared across all candidate K/lambdas in this vocabulary.
        for kind in ("real", "frequency", "first_order"):
            for replicate in range(1 if kind == "real" else args.null_replicates):
                guard(args)
                if kind == "real":
                    z = tokens
                else:
                    z, audit = order_null(tokens, kind, seed=10000 + replicate, vocabulary=x.shape[0])
                    audits.append({"level": level, "replicate": replicate, **audit})
                matrix = np.zeros_like(x)
                good = np.flatnonzero(z >= 0)
                matrix[z[good], good] = 1
                xx = xp.asarray(matrix)
                for row, w, thresholds in models:
                    ww = xp.asarray(w)
                    h = np.random.RandomState(101).uniform(0.1, 1.0, (w.shape[1], len(tokens))).astype(np.float32)
                    h[:, ~valid] = 0
                    hh = xp.asarray(h)
                    for _ in range(80):
                        _, hh = step(ww, hh, xx, vv, row["penalty"])
                    calls = retrieve(z, valid, w, np.asarray(hh), thresholds)
                    covered = np.zeros(len(tokens), bool)
                    for call in calls:
                        covered[call["start"] : call["end_exclusive"]] = True
                    counts = np.bincount([c["factor"] for c in calls], minlength=w.shape[1])
                    for factor in range(w.shape[1]):
                        rows.append(
                            dict(
                                trial=row["trial"],
                                level=level,
                                components=row["components"],
                                penalty=row["penalty"],
                                seed=row["seed"],
                                kind=kind,
                                replicate=replicate,
                                factor=factor,
                                calls=int(counts[factor]),
                                model_union_bout_coverage=float(covered[good].mean()),
                                model_union_time_coverage=float(duration[covered].sum() / duration.sum()),
                            )
                        )
                    log(f"NULL {level} {kind} rep={replicate + 1} {row['trial']} calls={len(calls)}")
                    pd.DataFrame(rows).to_csv(args.output / "order_null_results.csv", index=False)
                if kind != "real":
                    pd.DataFrame(audits).to_csv(args.output / "order_null_preservation.csv", index=False)
    log("ORDER NULLS COMPLETE: frozen W/thresholds, same80-step real/null H inference; descriptive comparisons only")


def report(args):
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .run_reference_seqnmf import write_json

    data = pd.read_csv(args.output / "capacity_metrics.csv")
    metrics = [
        ("global_relative_reconstruction_error", "Reconstruction error (lower)"),
        ("fit_union_time_coverage", "Accepted union time coverage"),
        ("supported_families_20calls_5mice", "Families: >=20calls, >=5mice"),
        ("template_max_redundancy", "Max template redundancy (lower)"),
        ("median_family_accepted_error", "Median family matching error"),
        ("matched_seed_template_median", "Matched seed cosine (shift +/-1)"),
    ]
    for level, subset in data.groupby("level", sort=False):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, (metric, title) in zip(axes.flat, metrics):
            for penalty, color in zip(PENALTIES, ("#2377a4", "#d17b27")):
                for seed, style in zip(SEEDS, ("-", "--", ":")):
                    values = subset[(subset.penalty == penalty) & (subset.seed == seed)].sort_values("components")
                    ax.plot(
                        values.components,
                        values[metric],
                        style,
                        marker="o",
                        color=color,
                        label=f"lambda{penalty:g},seed{seed}",
                        alpha=0.85,
                    )
            ax.set(title=title, xlabel="K", xticks=KS)
            ax.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"Pooled Karen: {level}; all99recordings; individual seeds")
        fig.tight_layout()
        fig.savefig(args.output / f"capacity_curves_{level}.png", dpi=170)
        fig.savefig(args.output / f"capacity_curves_{level}.pdf")
        plt.close(fig)
    nulls = pd.read_csv(args.output / "order_null_results.csv")
    real = nulls[nulls.kind == "real"].set_index(["trial", "factor"]).calls
    comparison = []
    for (trial, factor, kind), sub in nulls[nulls.kind != "real"].groupby(["trial", "factor", "kind"]):
        comparison.append(
            dict(
                trial=trial,
                factor=factor,
                kind=kind,
                real_calls=int(real.loc[(trial, factor)]),
                null_median_calls=float(sub.calls.median()),
                null_min_calls=int(sub.calls.min()),
                null_max_calls=int(sub.calls.max()),
                replicates=len(sub),
            )
        )
    pd.DataFrame(comparison).to_csv(args.output / "order_null_family_comparison.csv", index=False)
    write_json(
        args.output / "status.json",
        {
            "state": "COMPLETE",
            "configurations": 60,
            "new_fits": 48,
            "reused": 12,
            "null_models": int(nulls.trial.nunique()),
            "null_replicates_per_type": args.null_replicates,
            "limits": "Pooled descriptive diagnostics, no held-out claims or biological acceptance; no single combined winner score.",
        },
    )
    (args.output / "READ_ME.md").write_text(
        "# Pooled Karen capacity sweep\n\nAll99recordings/all21mice/all5waves; K6/10/14/18/22,lambda.3/3,seeds42/43/44.\n"
        "60configurations,12reusedK10,48new. Each is one shared pooled generator.\n\n"
        "capacity_metrics.csv and capacity_curves_* show individual seeds. family_metrics.csv includes pre-cutoff candidate and accepted errors.\n"
        "redundancy_pairs.csv combines templates and detected-time overlap. seed_family_matching.csv retains individual matches, allowing +/-1stage shifts with full-norm penalty.\n"
        "split_merge_edges.csv.gz retains all adjacent-K links, not a one-to-one clustering or accepted family genealogy.\n\n"
        "Order nulls freeze W and the original thresholds; BOTH real/null H are re-inferred with identical settings.\n"
        "Frequency null preserves exact bout counts but can create repeated neighbours; no RLE is applied again.\n"
        "First-order null uses random Euler trails preserving exact directed transition counts, endpoints and token frequencies per support block. It is not uniform over trails.\n"
        "Padding/native support and fixed duration slots stay unchanged. Null time coverage is conditional on those slots, not a duration/kinematic null.\n"
        "Reported null counts/ranges are descriptive, not independent test p-values. Models were learned on these same data.\n\n"
        "Shortlist contains coverage leader, stable-family leader and compact K10 reference per vocabulary where distinct; it is not final qualitative acceptance.\n"
        "Group effects and DA did not choose K/lambda. Next: inspect the tradeoff, retain compact/finer alternatives and generate p100/P80/random movies.\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--null-replicates", type=int, default=19)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-gb", type=float, default=8.0)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.0)
    parser.add_argument("--max-output-gb", type=float, default=4.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "K": KS,
                    "lambdas": PENALTIES,
                    "seeds": SEEDS,
                    "configurations": 60,
                    "new_fits": 48,
                    "null_replicates": args.null_replicates,
                    "execute": False,
                },
                indent=2,
            )
        )
        return
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".25")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    args.base, args.output = args.base.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "capacity_recipe.json").exists():
        raise ValueError("refusing to overwrite existing capacity run")
    from .run_reference_seqnmf import digest, guard, log, write_json

    guard(args)
    original = json.loads((args.base / "recipe.json").read_text())
    if len(original["source_arrays"]) != 99 or not original["all21mice_included"]:
        raise ValueError("base is not the authorised all-data pooled input")
    for path, expected in original["code"].items():
        if digest(path) != expected:
            raise ValueError("original fitting source changed")
    write_json(
        args.output / "capacity_recipe.json",
        {
            "base": str(args.base),
            "base_recipe_sha256": digest(args.base / "recipe.json"),
            "K": KS,
            "lambdas": PENALTIES,
            "lambda_choice": ".3 coverage-performing reference;3 stronger competition contrast, fixed before K sweep",
            "seeds": SEEDS,
            "iterations": 80,
            "length_bouts": 3,
            "null_replicates": args.null_replicates,
            "null_shortlist": "per vocabulary coverage leader, stable-family leader, compactK10 reference, deduplicated",
            "code": {str(p): digest(p) for p in (Path(__file__), Path(__file__).with_name("reference_capacity.py"))},
        },
    )
    try:
        log("CAPACITY START:60configurations;48new+12reused;all99recordings")
        fit_sweep(args)
        diagnose(args)
        run_nulls(args)
        report(args)
        log("CAPACITY COMPLETE:fit sweep,per-family diagnostics,split/merge,paired nulls and curves saved")
    except BaseException as error:
        write_json(args.output / "status.json", {"state": "FAILED", "error": repr(error)})
        log(f"CAPACITY FAILED {error!r}")
        raise


if __name__ == "__main__":
    main()
