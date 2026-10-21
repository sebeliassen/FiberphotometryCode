"""Explicit bounded K10/K18 repair: two pooled refits, fixed execution views.

Original inputs/media stay immutable. Execution requires --execute; metadata/help
does not initialize numerical backends. Outputs retain all parameter/result links.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def refit(args):
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    from .reference_capacity import template_cosines
    from .reference_seqnmf import backend, fit
    from .run_reference_seqnmf import digest, evaluate, guard, log, prepare, write_json

    fitroot = args.output / "fits"
    fitroot.mkdir(exist_ok=True)
    prep = argparse.Namespace(**vars(args))
    prep.output, prep.factors, prep.length, prep.iterations = fitroot, 10, 3, 80
    if not (fitroot / "recipe.json").exists():
        recipe = prepare(prep)
        recipe.update(lambdas=[.3], seeds=[43], component_choices=[10, 18],
                      execution_scope="two pooled grouped69 fits, K10/K18, lambda.3, seed43,80iterations")
        write_json(fitroot / "recipe.json", recipe)
    recipe = json.loads((fitroot / "recipe.json").read_text())
    with np.load(fitroot / "prepared/L3_unresolved_separate/matrices.npz", allow_pickle=False) as p:
        x, valid, tokens = p["fit_x"], p["fit_valid"], p["fit_tokens"]
    refs = pd.read_csv(fitroot / "prepared/L3_unresolved_separate/fit_references.csv.gz", dtype={"animal_id": str})
    runtime = backend(args.device)
    write_json(fitroot / "backend.json", runtime[2])
    original = json.loads((args.source / "manifest.json").read_text())
    models, matches = [], []
    for k in (10, 18):
        folder = fitroot / f"K{k}"
        folder.mkdir(exist_ok=True)
        result_path = folder / "result.json"
        if result_path.exists():
            row = json.loads(result_path.read_text())
            if digest(folder / "model.npz") != row["model_sha256"] or digest(folder / "pooled_calls.csv.gz") != row["calls_sha256"]:
                raise ValueError("completed repaired fit changed")
        else:
            log(f"CLEAN FIT K{k} START; pooled99recordings/all21mice/allwaves; seed43/lambda.3/80iterations")

            def progress(iteration, error):
                guard(args)
                log(f"CLEAN FIT K{k} iteration={iteration}/80 reconstruction={error:.6f}")

            w, h, history = fit(x, valid, factors=k, length=3, penalty=.3, iterations=80,
                                seed=43, runtime=runtime, progress=progress)
            if np.any(h[:, ~valid] != 0) or not np.isfinite(w).all() or not np.isfinite(h).all():
                raise AssertionError("unsupported or nonfinite repaired model")
            thresholds = np.array([np.percentile(v[valid & (v > 1e-4)], 90)
                                   if np.any(valid & (v > 1e-4)) else np.inf for v in h])
            metrics, calls = evaluate(w, h, tokens, valid, refs, thresholds)
            np.savez_compressed(folder / "model.npz", W=w, pooled_H=h, thresholds=thresholds,
                                factors=k, length=3, penalty=.3, seed=43)
            calls.to_csv(folder / "pooled_calls.csv.gz", index=False)
            row = dict(K=k, directory=str(folder.resolve()), penalty=.3, seed=43, iterations=80,
                       recipe_sha256=digest(fitroot / "recipe.json"), **metrics, history=history,
                       model_sha256=digest(folder / "model.npz"), calls_sha256=digest(folder / "pooled_calls.csv.gz"))
            write_json(result_path, row)
        old = original["inputs"][f"L3_unresolved_separate_K{k}_lambda0.3_seed43"]
        with np.load(Path(old["directory"]) / "model.npz", allow_pickle=False) as p:
            old_w = p["W"]
        with np.load(folder / "model.npz", allow_pickle=False) as p:
            new_w = p["W"]
        locked, _ = template_cosines(old_w, new_w)
        a, b = linear_sum_assignment(-locked)
        matches.extend(dict(K=k, original_factor=int(i), repaired_factor=int(j),
                            template_cosine=float(locked[i, j]),
                            limit="descriptive Hungarian alignment; no acceptance or exact factor identity") for i, j in zip(a, b))
        models.append(row)
        log(f"CLEAN FIT K{k} COMPLETE calls={row['fit_calls']} coverage={row['fit_union_time_coverage']:.4f}")
    pd.DataFrame(matches).to_csv(args.output / "old_to_repaired_factors.csv", index=False)
    write_json(args.output / "refits.json", dict(models=models, fit_design=recipe["fit_design"]))


def extract(args):
    import math

    import numpy as np
    import pandas as pd

    from studies.sequence_viz.recreate_generator_videos import execution_descriptors

    from .execution_similarity import fit_scaling, kinematics, phase_blocks, scaled_blocks
    from .render_reference_review import balance_sample, verify_call
    from .run_reference_seqnmf import digest, guard, log, write_json

    original = json.loads((args.source / "manifest.json").read_text())
    recipe = json.loads((args.base / "recipe.json").read_text())
    mapping = np.asarray(json.loads(Path(recipe["input_metadata"][3]["path"]).read_text()), int)
    annotations = pd.read_csv(args.view / "syllable_annotations.csv").fillna("")
    categories = sorted(set(annotations.loc[~annotations.description.isin(["", "x", "unknown"]), "category"]) - {"", "Unassigned"})
    semantic_codes = np.zeros(100, int)
    for r in annotations.itertuples():
        if r.description not in ("", "x", "unknown") and r.category in categories:
            semantic_codes[r.syllable_id] = categories.index(r.category) + 1
    refs = {
        "frozen": pd.read_csv(args.base / "prepared/L3_unresolved_separate/fit_references.csv.gz"),
        "clean": pd.read_csv(args.output / "fits/prepared/L3_unresolved_separate/fit_references.csv.gz"),
    }
    tables, source_paths = {}, {}
    for cohort in ("frozen", "clean"):
        for k in (10, 18):
            key = f"{cohort}_K{k}"
            folder = (Path(original["inputs"][f"L3_unresolved_separate_K{k}_lambda0.3_seed43"]["directory"])
                      if cohort == "frozen" else args.output / f"fits/K{k}")
            table = pd.read_csv(folder / "pooled_calls.csv.gz", dtype={"animal_id": str})
            table["case"], table["K"], table["source_index"] = key, k, np.arange(len(table))
            tables[key], source_paths[key] = table, folder
    selected, feature_rows, rejected = [], [], []
    for number, sid in enumerate(original["bindings"], 1):
        guard(args)
        binding = original["bindings"][sid]
        for cohort in ("frozen", "clean"):
            path = (Path(original["track_rows"][sid]["source_provenance"]["source"]["path"])
                    if cohort == "frozen" else args.view / "frame_cache" / f"{sid}.npz")
            with np.load(path, allow_pickle=False) as p:
                excluded = p["analysis_excluded"] if "analysis_excluded" in p.files else np.zeros(len(p["frames"]), bool)
                valid = (p["labels"] >= 0) & ~excluded
                track = {key: p[key][valid] for key in ("frames", "labels", "latent", "centroid", "heading")}
            motion = kinematics(track["frames"], track["centroid"], track["heading"])
            clock = np.arange(binding["video_frames"]) / 60.
            for k in (10, 18):
                key = f"{cohort}_K{k}"
                sub = tables[key][tables[key].session_id == sid]
                if sub.empty:
                    continue
                calls = []
                for call in sub.to_dict("records"):
                    call.update(recording=sid, subject=str(call["animal_id"]), stop_frame=int(call["end_frame_exclusive"]))
                    if call["stop_frame"] >= len(clock):
                        rejected.append(dict(case=key, source_index=call["source_index"], reason="encoded video endpoint unavailable"))
                        continue
                    verify_call(track, call, mapping)
                    calls.append(call)
                if not calls:
                    continue
                starts = np.asarray([c["start"] for c in calls], int)[:, None] + np.arange(3)
                start_frames = refs[cohort].start_frame.to_numpy()[starts].astype(int)
                stop_frames = refs[cohort].end_frame_exclusive.to_numpy()[starts].astype(int)
                blocks = phase_blocks(track, start_frames, stop_frames, semantic_codes, motion)
                legacy, okay, _ = execution_descriptors(track, clock, calls, dict(latent=np.ones(10), path=1.))
                if not okay.all():
                    raise ValueError("descriptor support unavailable")
                s = original["ranking"]["scales"]
                legacy[:, :320] /= np.tile(np.asarray(s["latent_iqr"]) * math.sqrt(4 * 320), 32)
                legacy[:, 320:384] /= s["displacement_px_p75"]
                legacy[:, -1] /= s["log_duration_iqr"]
                selected.extend(calls)
                feature_rows.append({**blocks, "legacy": legacy})
        log(f"FEATURES {number}/99 {sid}")
    table = pd.DataFrame(selected)
    arrays = {key: np.concatenate([r[key] for r in feature_rows]) for key in feature_rows[0]}
    frozen = np.flatnonzero(table.case.str.startswith("frozen").to_numpy())
    calibration = balance_sample(frozen[:, None], table.iloc[frozen].animal_id.to_numpy()).ravel()
    scale = fit_scaling(arrays, calibration, subjects=table.animal_id.to_numpy())
    normalized = scaled_blocks(arrays, scale)
    np.savez_compressed(args.output / "descriptors.npz", **normalized, legacy=arrays["legacy"])
    table.to_csv(args.output / "eligible_calls.csv.gz", index=False)
    pd.DataFrame(rejected, columns=["case", "source_index", "reason"]).to_csv(args.output / "unavailable_calls.csv", index=False)
    write_json(args.output / "feature_recipe.json", dict(
        phase_scale=scale, calibration="equal-mouse512calls; frozen K10/K18 pool; deterministic seed101; shared across cases",
        semantic_categories=categories, semantics="human category agreement at mutually known ordered landmarks; x/unassigned are missing, never a shared category",
        annotated_source=dict(path=str(args.view / "syllable_annotations.csv"), sha256=digest(args.view / "syllable_annotations.csv")),
        corrected_legacy="original frozen scale with latent native-unit correction; same32 elapsed-time phase samples",
        datasets={key: dict(folder=str(path.resolve()), calls_sha256=digest(path / "pooled_calls.csv.gz"),
                            model_sha256=digest(path / "model.npz")) for key, path in source_paths.items()},
        no_location_weight="Calibrated arena boundaries unavailable. Screen position is not substituted for wall/corner distance.",
        no_phenotype_selection=True, dtype="float64", memory_cap_gb=args.memory_gb))


def neighbour_matrix(blocks, profile):
    import numpy as np
    from scipy.spatial.distance import cdist

    d = sum(weight * cdist(blocks[key], blocks[key], metric="sqeuclidean") / blocks[key].shape[1]
            for key, weight in profile.items() if key != "semantic")
    denom = np.full(d.shape, sum(w for k, w in profile.items() if k != "semantic"))
    if "semantic" in profile:
        labels = blocks["semantic"]
        known, mismatch = np.zeros(d.shape), np.zeros(d.shape)
        for column in labels.T:
            common = (column[:, None] > 0) & (column[None, :] > 0)
            known += common
            mismatch += common & (column[:, None] != column[None, :])
        available = known > 0
        d[available] += profile["semantic"] * mismatch[available] / labels.shape[1]
        denom += profile["semantic"] * known / labels.shape[1]
    return d / denom


def diagnose(args):
    import numpy as np
    import pandas as pd

    from studies.sequence_viz.recreate_generator_videos import rank_exemplar_neighbors

    from .execution_similarity import PROFILES, distinct_execution_neighbours, exemplar_distances, percentiles
    from .reference_capacity import execution_masks, execution_overlap
    from .run_reference_seqnmf import guard, log, write_json

    calls = pd.read_csv(args.output / "eligible_calls.csv.gz", dtype={"animal_id": str, "subject": str})
    with np.load(args.output / "descriptors.npz", allow_pickle=False) as p:
        arrays = {k: p[k] for k in p.files}
    rows, ranks, neighbour_rows, detector, overlaps = [], [], [], [], []
    rng = np.random.default_rng(101)
    for case, own in calls.groupby("case", sort=False):
        guard(args)
        balanced = []
        for _, group in own.groupby("factor", sort=True):
            target = min(64, len(group))
            per_mouse = max(1, target // group.animal_id.nunique())
            first = np.concatenate([rng.choice(g.index.to_numpy(), min(per_mouse, len(g)), replace=False)
                                    for _, g in group.groupby("animal_id", sort=True)])
            remaining = np.setdiff1d(group.index.to_numpy(), first)
            if len(first) < target:
                first = np.r_[first, rng.choice(remaining, target - len(first), replace=False)]
            balanced.append(first[:target])
        sampled = np.concatenate(balanced)
        all_blocks = {key: value[sampled] for key, value in arrays.items()}
        for profile in ("corrected_legacy", *PROFILES):
            if profile == "corrected_legacy":
                from scipy.spatial.distance import cdist
                matrix = cdist(all_blocks["legacy"], all_blocks["legacy"], metric="sqeuclidean")
            else:
                matrix = neighbour_matrix(all_blocks, PROFILES[profile])
            animals = calls.loc[sampled, "animal_id"].to_numpy()
            factors = calls.loc[sampled, "factor"].to_numpy()
            matrix[animals[:, None] == animals[None, :]] = np.inf
            sampled_calls = calls.loc[sampled]
            nearest, valid_neighbours = distinct_execution_neighbours(
                matrix, sampled_calls.recording.to_numpy(), sampled_calls.start_frame.to_numpy(),
                sampled_calls.stop_frame.to_numpy())
            agreement = np.divide(((factors[nearest] == factors[:, None]) & valid_neighbours).sum(axis=1),
                                  valid_neighbours.sum(axis=1), out=np.full(len(sampled), np.nan),
                                  where=valid_neighbours.sum(axis=1) > 0)
            neighbour_rows.extend(dict(case=case, profile=profile, factor=int(factor), animal_id=animal,
                                       source_index=int(calls.loc[idx, "source_index"]), cross_mouse_knn5_agreement=float(score))
                                  for idx, factor, animal, score in zip(sampled, factors, animals, agreement))
            for factor, sub in own.groupby("factor", sort=True):
                indexes = sub.index.to_numpy()
                blocks = {key: value[indexes] for key, value in arrays.items()}
                subjects = sub.animal_id.to_numpy()
                if profile == "corrected_legacy":
                    distance, pct, exemplar = rank_exemplar_neighbors(blocks["legacy"], subjects)
                    delta = (blocks["legacy"] - blocks["legacy"][exemplar]) ** 2
                    parts = {"posture": delta[:, :320].sum(axis=1), "displacement": delta[:, 320:384].sum(axis=1),
                             "heading": delta[:, 384:448].sum(axis=1), "tempo": delta[:, -1]}
                    weighted = parts
                else:
                    distance, exemplar, parts = exemplar_distances(blocks, subjects, PROFILES[profile])
                    pct = percentiles(distance)
                    weighted = {key: PROFILES[profile][key] * value for key, value in parts.items() if key in PROFILES[profile]}
                    denominator = np.full(len(distance), sum(v for k, v in PROFILES[profile].items() if k != "semantic"))
                    if "semantic" in PROFILES[profile]:
                        denominator += PROFILES[profile]["semantic"] * parts["semantic_shared_fraction"]
                        weighted["semantic"] *= parts["semantic_shared_fraction"]
                    weighted = {key: value / denominator for key, value in weighted.items()}
                    np.testing.assert_allclose(sum(np.nan_to_num(v) for v in weighted.values()), distance,
                                               rtol=1e-12, atol=1e-12)
                row = dict(case=case, K=int(sub.K.iloc[0]), profile=profile, factor=int(factor), calls=len(sub),
                           mice=sub.animal_id.nunique(), supported=len(sub) >= 20 and sub.animal_id.nunique() >= 5,
                           exemplar_source_index=int(sub.iloc[exemplar].source_index),
                           distance_p50=float(np.median(distance)), distance_p80=float(np.quantile(distance, .8)),
                           duration_p10=float(sub.duration_sec.quantile(.1)), duration_p90=float(sub.duration_sec.quantile(.9)),
                           knn5_agreement=float(pd.DataFrame(dict(animal=animals[factors == factor],
                                                                 score=agreement[factors == factor])).groupby("animal").score.mean().mean()))
                total = sum(np.nansum(v) for v in weighted.values())
                row.update({key + "_contribution": float(np.nansum(v) / total) if total else 0 for key, v in weighted.items()})
                rows.append(row)
                for ix, d, p in zip(indexes, distance, pct):
                    call = calls.loc[ix]
                    ranks.append(dict(case=case, profile=profile, factor=int(factor), source_index=int(call.source_index),
                                      recording=call.recording, subject=str(call.subject), start_frame=int(call.start_frame),
                                      stop_frame=int(call.stop_frame), distance=float(d), percentile=float(p)))
        for factor, sub in own.groupby("factor", sort=True):
            for cutoff in (.5, .6, .68):
                retained = sub[sub.template_error <= cutoff]
                detector.append(dict(case=case, factor=int(factor), max_error=cutoff, original_calls=len(sub),
                                     retained_calls=len(retained), mice=retained.animal_id.nunique(),
                                     note="fixed calls/NMS; acceptance sensitivity only, not a new fit or behavioural validation"))
        refs_path = (args.base if case.startswith("frozen") else args.output / "fits") / "prepared/L3_unresolved_separate/fit_references.csv.gz"
        refs = pd.read_csv(refs_path)
        duration = ((refs.end_frame_exclusive - refs.start_frame) / 60.).fillna(0).to_numpy()
        masks = execution_masks(own, int(own.K.iloc[0]), len(refs))
        jaccard, _, _ = execution_overlap(masks, masks, duration)
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                overlaps.append(dict(case=case, first_factor=i, second_factor=j,
                                     physical_time_jaccard=float(jaccard[i, j])))
        log(f"DIAGNOSTICS {case}: three ranking profiles; cross-mouse sample={len(sampled)}")
    pd.DataFrame(rows).to_csv(args.output / "family_comparison.csv", index=False)
    pd.DataFrame(ranks).to_csv(args.output / "execution_rankings.csv.gz", index=False)
    pd.DataFrame(neighbour_rows).to_csv(args.output / "cross_mouse_neighbours.csv.gz", index=False)
    pd.DataFrame(detector).to_csv(args.output / "detector_sensitivity.csv", index=False)
    pd.DataFrame(overlaps).to_csv(args.output / "family_time_overlaps.csv", index=False)
    write_json(args.output / "status.json", dict(state="QUANTITATIVE_COMPLETE", fitted_models=2,
                                                cases=4, profiles=3, profile_weights=PROFILES,
                                                no_location_claim=True, no_coherence_acceptance=True,
                                                next="small before/after review then refreshed catalogue pass"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "source", "view", "mapping", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "fit", "features", "diagnose"), default="all")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-gb", type=float, default=8.)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.)
    parser.add_argument("--max-output-gb", type=float, default=6.)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(stage=args.stage, K=[10, 18], seeds=[43], penalty=.3, iterations=80,
                              device=args.device, status="PLAN: pass --execute for requested empirical scope")))
        return
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".25")
    for key in ("base", "source", "view", "mapping", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    from .run_reference_seqnmf import write_json
    sources = [Path(__file__), Path(__file__).with_name("execution_similarity.py"),
               Path(__file__).with_name("render_reference_review.py")]
    launch = args.output / "launch_recipe.json"
    if launch.exists():
        launch = args.output / ("launch_" + args.stage + ".json")
    write_json(launch, dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                                       source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                                                       actual_affinity_cpus=len(os.sched_getaffinity(0))))
    write_json(args.output / "status.json", dict(state="RUNNING", stage=args.stage))
    try:
        if args.stage in ("all", "fit"):
            refit(args)
        if args.stage in ("all", "features"):
            extract(args)
        if args.stage in ("all", "diagnose"):
            diagnose(args)
    except BaseException as error:
        write_json(args.output / "status.json", dict(state="FAILED", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
