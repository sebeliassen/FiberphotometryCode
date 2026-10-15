"""Post-search study diagnostics from frozen calls; no detector or KPMS fitting.

Selection is within cohort/method, on complete configurations rather than trial
or method averages. Physical distances use analytic cross-animal moments, with
training-only scales and KPMS-conditioned residuals. No phenotype/DA inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import resource
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(x)]
    return statistics.mean(values) if values else None


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def complete_configurations(connection):
    grouped = defaultdict(list)
    for config, metrics, artifact, stability, status in connection.execute(
            "SELECT config_json,metrics_json,artifact,seed_stability,status FROM candidate_metrics"):
        job, metrics = json.loads(config), json.loads(metrics)
        origin = str(Path(artifact).parent.parent)
        grouped[(origin, job["cohort"], job["base_recipe"])].append(
            dict(job=job, metrics=metrics, artifact=artifact, seed_stability=stability, status=status))
    fields = ("coverage", "agreement", "separation", "animal_support", "singleton_fraction",
              "selected_calls", "observed_types", "feature_support", "mean_distinct_constituents",
              "hsmm_convergence_fraction")
    complete, incomplete = [], []
    for (origin, cohort, base), trials in sorted(grouped.items()):
        recipe = trials[0]["job"]["recipe"]
        expected = 2 if recipe["method"] == "seqnmf" or (
            recipe["method"] == "hsmm" and recipe["initialization"] == "dirichlet") else 1
        folds = {fold: [x for x in trials if x["job"]["fold"] == fold] for fold in range(3)}
        if any(len({x["job"]["replicate"] for x in part}) < expected for part in folds.values()):
            incomplete.append(dict(origin=origin, cohort=cohort, base_recipe=base, completed_jobs=len(trials)))
            continue
        metrics = {}
        for field in fields:
            values = [mean(x["metrics"].get(field) for x in folds[fold]) for fold in range(3)]
            metrics[field], metrics[field + "_folds"] = mean(values), values
        seeds = [mean(x["seed_stability"] for x in folds[f]) for f in range(3)]
        # Deterministic fits do not acquire stability evidence from repeated execution.
        if expected == 1:
            seeds = [None] * 3
        metrics.update(seed_stability=mean(seeds), seed_stability_folds=seeds,
                       seed_stability_min=min((v for v in seeds if v is not None), default=None),
                       empty_folds=sum(any(x["status"] == "EMPTY" for x in folds[f]) for f in range(3)))
        complete.append(dict(origin=origin, cohort=cohort, base_recipe=base,
                             recipe=recipe, metrics=metrics, trials=trials))
    return complete, incomplete


def eligible(row):
    m = row["metrics"]
    return (not m["empty_folds"] and all(v is not None and v >= .03 for v in m["coverage_folds"])
            and m["selected_calls"] >= 10 and m["animal_support"] >= .4
            and m["feature_support"] >= .75 and all(v is not None for v in m["agreement_folds"]))


def retain_candidates(rows):
    """Full within-family frontiers, plus bounded diverse review candidates."""
    import numpy as np

    groups = defaultdict(list)
    for row in rows:
        if eligible(row):
            groups[(row["cohort"], row["recipe"]["method"])].append(row)
    selected, frontiers, counts = {}, [], []
    def retain(row, reason):
        key = (row["origin"], row["cohort"], row["base_recipe"])
        if key not in selected:
            selected[key] = dict(row, reasons=[])
        selected[key]["reasons"].append(reason)
    for (cohort, method), group in sorted(groups.items()):
        counts.append(dict(cohort=cohort, method=method, complete_supported_configs=len(group)))
        for stochastic in (False, True):
            panel = [r for r in group if (r["metrics"]["seed_stability"] is not None) == stochastic]
            if not panel:
                continue
            axes = ["agreement", "coverage", "animal_support"]
            if stochastic:
                axes.append("seed_stability")
            vectors = np.array([[r["metrics"][k] for k in axes] + [1-r["metrics"]["singleton_fraction"]]
                                for r in panel], dtype=np.float64)
            front = []
            for i, vector in enumerate(vectors):
                dominated = np.all(vectors >= vector, axis=1) & np.any(vectors > vector, axis=1)
                if not dominated.any():
                    front.append(panel[i])
            frontiers.append(dict(cohort=cohort, method=method, stochastic=stochastic,
                                  axes=axes+["1-singleton_fraction"],
                                  candidates=[dict(origin=r["origin"], base_recipe=r["base_recipe"], metrics=r["metrics"])
                                              for r in front]))
            retain(max(front, key=lambda r: r["metrics"]["agreement"]), "coherence leader")
            # Coverage alone can select indiscriminate detection; retain coherence floor.
            broader = [r for r in front if r["metrics"]["agreement"] >= .35]
            if broader:
                retain(max(broader, key=lambda r: r["metrics"]["coverage"]), "coverage leader with agreement>=0.35")
            low_frag = [r for r in front if r["metrics"]["singleton_fraction"] <= .1]
            if low_frag:
                retain(max(low_frag, key=lambda r: r["metrics"]["agreement"]), "coherence with singleton fraction<=0.1")
            if stochastic:
                stable = [r for r in front if r["metrics"]["agreement"] >= .25]
                if stable:
                    retain(max(stable, key=lambda r: r["metrics"]["seed_stability"]), "stability leader with agreement>=0.25")
        # Do not erase alternative feature routes just because source strings differ.
        for route in ("feature_only", "joint"):
            challengers = [r for r in group if r["recipe"]["route"] == route]
            if challengers:
                retain(max(challengers, key=lambda r: r["metrics"]["agreement"]), route+" physical challenger")
    return list(selected.values()), frontiers, counts


def moments(values):
    import numpy as np

    valid = np.isfinite(values)
    n = valid.sum(axis=0)
    sums = np.where(valid, values, 0).sum(axis=0)
    center = np.divide(sums, n, out=np.full(values.shape[1], np.nan), where=n > 0)
    squared = np.where(valid, (values-center)**2, 0).sum(axis=0)
    variance = np.divide(squared, n, out=np.full(values.shape[1], np.nan), where=n > 0)
    return center, variance


def physical_distances(profiles, labels, subjects, columns):
    """Exact independent cross-animal pair MSE from population moments.

    Equal feature, animal-pair and type weights; no random pair subsampling.
    At least two calls per contributing animal and two animals per own type.
    """
    if not columns or not len(profiles):
        return dict(within=None, between=None, separation=None, eligible_types=0)
    groups = {}
    for token in sorted(set(labels)):
        for subject in sorted(set(subjects[labels == token])):
            part = profiles[(labels == token) & (subjects == subject)][:, columns]
            if len(part) >= 2:
                groups[(token, subject)] = moments(part)
    def distance(a, b):
        value = (a[0]-b[0])**2 + a[1] + b[1]
        return mean(value.tolist())
    types = []
    for token in sorted(set(labels)):
        own = [(s, v) for (t, s), v in groups.items() if t == token]
        if len(own) < 2:
            continue
        within = mean(distance(a[1], b[1]) for a, b in itertools.combinations(own, 2))
        between = mean(distance(v, other) for subject, v in own
                       for (other_token, other_subject), other in groups.items()
                       if other_token != token and other_subject != subject)
        types.append(dict(type_id=token, within=within, between=between,
                          separation=1-within/between if within is not None and between is not None and between > 1e-12 else None))
    return dict(within=mean(t["within"] for t in types), between=mean(t["between"] for t in types),
                separation=mean(t["separation"] for t in types), eligible_types=len(types), types=types)


def training_context(records):
    import numpy as np

    train = [r for r in records if r["split"] == "train"]
    names = list(train[0]["feature_names"])
    values = np.concatenate([r["values"] for r in train])
    parents = np.concatenate([r["parent_ids"] for r in train])
    location, scale = np.zeros(len(names)), np.ones(len(names))
    for i in range(len(names)):
        finite = values[np.isfinite(values[:, i]), i]
        if len(finite):
            location[i] = np.median(finite)
            scale[i] = float(np.quantile(finite, .75)-np.quantile(finite, .25))
            if scale[i] < 1e-8:
                scale[i] = 1.
    z = (values-location)/scale
    parent_expected = {}
    for token in sorted(set(parents)):
        v = z[parents == token]
        parent_expected[token] = np.array([np.median(v[np.isfinite(v[:, i]), i])
                                          if np.isfinite(v[:, i]).any() else np.nan for i in range(len(names))])
    return dict(records={r["recording"]: r for r in records}, names=names, location=location,
                scale=scale, parent_expected=parent_expected)


def call_profiles(record, calls, context):
    """Three native-time phases of piecewise-constant source-bout features."""
    import numpy as np

    z = (record["values"]-context["location"])/context["scale"]
    expected = np.array([context["parent_expected"].get(t, np.full(z.shape[1], np.nan)) for t in record["parent_ids"]])
    raw, residual = np.clip(z, -8, 8), np.clip(z-expected, -8, 8)
    rows = []
    for call in calls:
        start, stop = call["start_frame"], call["stop_frame"]
        a = int(np.searchsorted(record["start_frames"], start))
        b = int(np.searchsorted(record["stop_frames"], stop, side="right"))
        assert b > a and record["start_frames"][a] == start and record["stop_frames"][b-1] == stop
        assert np.all(record["stop_frames"][a:b-1] == record["start_frames"][a+1:b])
        bounds = np.linspace(start, stop, 4)
        arrays = []
        for values in (raw[a:b], residual[a:b]):
            phase = []
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                w = np.maximum(0, np.minimum(record["stop_frames"][a:b], hi)-np.maximum(record["start_frames"][a:b], lo))
                valid = np.isfinite(values)
                weights = w[:, None]*valid
                phase.append(np.divide((np.where(valid, values, 0)*weights).sum(axis=0), weights.sum(axis=0),
                                       out=np.full(values.shape[1], np.nan), where=weights.sum(axis=0) > 0))
            arrays.append(np.concatenate(phase))
        tokens = record["parent_ids"][a:b].tolist()
        counts = Counter(tokens)
        period = next((p for p in range(1, len(tokens)+1) if len(tokens) % p == 0
                       and tokens == tokens[:p]*(len(tokens)//p)), len(tokens))
        rows.append(dict(type_id=call["type_id"], subject=record["subject"], recording=record["recording"],
                         start_frame=start, stop_frame=stop, duration_s=(stop-start)/record["fps"],
                         events=len(tokens), distinct=len(counts), dominant_fraction=max(counts.values())/len(tokens),
                         exact_repeat_copies=len(tokens)//period, profile=arrays[0], residual=arrays[1]))
    return rows


def assess_trial(trial, context):
    import numpy as np

    payload = json.loads(Path(trial["artifact"]).read_text())
    result = payload["result"]
    consumed = set(result["quality"]["physical"]["features_used_in_construction"])
    calls, adjacent, raw_count, selected_count = [], 0, 0, 0
    for rec in result["records"]:
        record = context["records"][rec["recording"]]
        assert record["split"] == "held" and record["subject"] == rec["subject"]
        ordered = sorted(rec["selected_calls"], key=lambda c: c["start_frame"])
        assert all(a["stop_frame"] <= b["start_frame"] for a, b in zip(ordered[:-1], ordered[1:]))
        adjacent += sum(a["stop_frame"] == b["start_frame"] and a["type_id"] == b["type_id"]
                        for a, b in zip(ordered[:-1], ordered[1:]))
        calls.extend(call_profiles(record, ordered, context))
        raw_count += len(rec["raw_calls"])
        selected_count += len(ordered)
    assert selected_count == result["summary"]["selected_calls"]
    names = context["names"]
    labels = np.array([c["type_id"] for c in calls])
    subjects = np.array([c["subject"] for c in calls])
    profiles = np.array([c["profile"] for c in calls]).reshape(-1, 3*len(names))
    residual = np.array([c["residual"] for c in calls]).reshape(-1, 3*len(names))
    groups = dict(execution=[n for n in names if not n.startswith("centroid_")],
                  unused_execution=[n for n in names if n not in consumed and not n.startswith("centroid_")],
                  movement=[n for n in names if "speed" in n or "heading_change" in n or n == "path_length_px"],
                  posture=[n for n in names if n.startswith("latent_")], duration=["duration_s"])
    physical = {}
    for group, features in groups.items():
        columns = [phase*len(names)+names.index(name) for phase in range(3) for name in features]
        physical[group] = physical_distances(profiles, labels, subjects, columns)
        physical[group]["feature_names"] = features
    execution_cols = [phase*len(names)+i for phase in range(3) for i, n in enumerate(names) if not n.startswith("centroid_")]
    physical["source_conditioned_execution"] = physical_distances(residual, labels, subjects, execution_cols)
    types = []
    for token in sorted(set(labels)):
        own = [c for c in calls if c["type_id"] == token]
        durations = [c["duration_s"] for c in own]
        animals = len({c["subject"] for c in own})
        types.append(dict(type_id=token, occurrences=len(own), animals=animals,
                          duration_cv=statistics.stdev(durations)/mean(durations) if len(own) > 1 else None,
                          mean_distinct=mean(c["distinct"] for c in own)))
    counts = np.array([t["occurrences"] for t in types], dtype=float)
    effective = 1/np.sum((counts/counts.sum())**2) if counts.sum() else None
    metrics = dict(
        single_animal_type_fraction=mean(t["animals"] == 1 for t in types),
        fragile_type_fraction=mean(t["animals"] < 2 or t["occurrences"] < 4 for t in types),
        effective_types=float(effective) if effective is not None else None,
        effective_type_fraction=float(effective)/len(types) if types else None,
        adjacent_same_type_fraction=adjacent/max(1, selected_count),
        overlap_rejection_fraction=1-selected_count/raw_count if raw_count else None,
        median_duration_s=statistics.median(c["duration_s"] for c in calls) if calls else None,
        mean_type_duration_cv=mean(t["duration_cv"] for t in types),
        one_constituent_fraction=mean(c["distinct"] == 1 for c in calls),
        two_constituent_fraction=mean(c["distinct"] <= 2 for c in calls),
        exact_repeat_fraction=mean(c["exact_repeat_copies"] >= 2 for c in calls),
        mean_distinct=mean(c["distinct"] for c in calls),
        mean_events=mean(c["events"] for c in calls),
        mean_dominant_fraction=mean(c["dominant_fraction"] for c in calls))
    for group, values in physical.items():
        metrics[group + "_separation"] = values["separation"]
        metrics[group + "_eligible_types"] = values["eligible_types"]
    event_fraction = sum(c["events"] for c in calls)/result["summary"]["held_events"]
    metrics["real_event_fraction"] = event_fraction
    for kind in ("shuffle", "first_order"):
        control = mean(c.get("selected_event_fraction") for c in result["controls"] if c["kind"] == kind)
        metrics[kind + "_event_gain"] = event_fraction-control if control is not None else None
    audit_calls = [{k: v for k, v in c.items() if k not in {"profile", "residual"}} for c in calls]
    return dict(job=payload["job"], metrics=metrics, physical=physical, types=types,
                consumed_features=sorted(consumed), calls=audit_calls, conditions=result["conditions"])


def write_csv(path, rows):
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-ram-gib", type=float, default=4.)
    args = parser.parse_args()
    if not math.isfinite(args.max_ram_gib) or args.max_ram_gib <= 0:
        parser.error("--max-ram-gib must be finite and positive")
    allocation_cap = int(args.max_ram_gib * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (allocation_cap, allocation_cap))
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    status = json.loads((args.run / "status.json").read_text())
    if status["state"] != "STOPPED":
        raise ValueError("Assessment requires the explicitly stopped search")
    dump(args.output / "stop_status.json", status)
    source = sqlite3.connect(f"file:{args.run / 'trials.sqlite'}?mode=ro", uri=True)
    snapshot = sqlite3.connect(args.output / "trials_snapshot.sqlite")
    source.backup(snapshot)
    source.close()
    rows, incomplete = complete_configurations(snapshot)
    snapshot.close()
    dump(args.output / "configurations.json", rows)
    dump(args.output / "incomplete_configurations.json", incomplete)
    selected, frontiers, counts = retain_candidates(rows)
    # Preserve earlier discussed alternatives even when a different frontier
    # extreme dominates their original-string diagnostics. Physical consistency
    # has not yet been used at the selection stage.
    reference_choices = {("karen", 2967), ("jonathan", 2368), ("jonathan", 2729)}
    retained_keys = {(r["origin"], r["cohort"], r["base_recipe"]) for r in selected}
    for row in rows:
        key = (row["origin"], row["cohort"], row["base_recipe"])
        if "overnight" in row["origin"] and (row["cohort"], row["base_recipe"]) in reference_choices and key not in retained_keys:
            selected.append(dict(row, reasons=["previously discussed coverage/stability alternative"]))
    dump(args.output / "pareto_frontiers.json", frontiers)
    dump(args.output / "search_conditions.json", counts)
    print(f"Snapshot: {len(rows)} complete, {len(incomplete)} incomplete; {len(selected)} retained", flush=True)
    # No phenotype/drug/DA metadata enters the record loader or metrics.
    from studies.sequence_viz.broad_search_inputs import load_job_records
    manifest = args.run / "frozen_input_manifest.json"
    settings = json.loads((args.run / "run_manifest.json").read_text())
    contexts = {}
    for cohort in settings["cohorts"]:
        for fold in range(3):
            records = load_job_records(manifest, cohort, fold, per_animal=settings["per_animal"],
                                       held_per_animal=settings["held_per_animal"], window_seed=settings["window_seed"])
            contexts[(cohort, fold)] = training_context(records)
    summaries = []
    for i, row in enumerate(selected):
        details = [assess_trial(t, contexts[(row["cohort"], t["job"]["fold"])]) for t in row["trials"]]
        fields = sorted({k for d in details for k in d["metrics"]})
        assessment = {}
        for field in fields:
            values = [mean(d["metrics"].get(field) for d in details if d["job"]["fold"] == fold) for fold in range(3)]
            assessment[field] = mean(values)
            assessment[field + "_folds"] = values
        row["assessment"] = assessment
        name = f"{row['cohort']}_{Path(row['origin']).name}_r{row['base_recipe']}"
        dump(args.output / (name + ".json"), dict(configuration=row, trial_assessments=details))
        summaries.append(dict(cohort=row["cohort"], method=row["recipe"]["method"], route=row["recipe"]["route"],
                              recipe=row["base_recipe"], origin=row["origin"], reasons="; ".join(row["reasons"]),
                              **{k: v for k, v in row["metrics"].items() if not k.endswith("_folds")},
                              **{k: v for k, v in assessment.items() if not k.endswith("_folds")}, artifact=name + ".json"))
        print(f"Assessed {i+1}/{len(selected)}: {row['cohort']} {row['recipe']['method']} {row['base_recipe']}", flush=True)
    dump(args.output / "shortlist.json", selected)
    write_csv(args.output / "shortlist.csv", summaries)
    from threadpoolctl import threadpool_info
    dump(args.output / "provenance.json", dict(
        source_script=str(Path(__file__).resolve()), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        run=str(args.run.resolve()), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        wall_seconds=time.monotonic()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        dtype="float64", workers=1, address_space_cap_bytes=allocation_cap,
        numerical_pools=threadpool_info(),
        physical_pair_sampling="exact analytical moments, no simulation",
        profile="3 native-time phases, source-bout features piecewise constant; frames never resampled",
        normalization="training-only median/IQR, fallback1 for constant channel, standardized values clipped at8",
        source_conditioning="subtract training-only per-KPMS-state median before clipping; unseen state residual unavailable",
        metric="1 - within/between cross-animal squared feature distance; equal feature/animal-pair/type weights",
        eligible_physical="at least two calls per animal and two animals per own type; unavailable remains missing",
        scope="exploratory held windows only; no fits, video review, phenotype/DA or reserved-animal access"))


if __name__ == "__main__":
    main()
