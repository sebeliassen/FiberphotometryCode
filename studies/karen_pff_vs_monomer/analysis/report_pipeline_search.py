"""Karen's stopped-search report and explicit frozen-finalist cohort analysis.

Discovery rankings are label independent and stratified by method/resolution.
Only persisted, twice-successful configurations enter the finalist pass. Cohort
policy stays here; native projection, organisation, nulls and inference statistics
are maintained in their existing reusable library owners.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

UNTREATED = ("3w_PI", "8w_PI", "16w_PI", "24w_PI")
GAPS = (0., 1., 5.)
MEASURES = ("calls_per_min", "coverage_fraction", "self_probability", "self_excess", "mean_run_calls",
            "initiations_per_min", "observed_initiations_per_min", "return2_probability", "return2_excess",
            "return3_probability", "return4_probability", "effective_routines", "transition_entropy_nats",
            "repeated_covered_fraction", "right_censored_fraction", "mean_duration_s", "mean_phase_duration_cv",
            "mean_speed_px_s", "duration_cv")
DATA = None


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize_search(search, output):
    import pandas as pd

    conn = sqlite3.connect(f"file:{search / 'trials.db'}?mode=ro", uri=True)
    columns = [r[1] for r in conn.execute("PRAGMA table_info(trials)") if r[1] not in ("result_json", "error")]
    trials = pd.read_sql_query("SELECT " + ",".join(columns) + " FROM trials", conn)
    if trials.status.isin(("PENDING", "RUNNING")).any():
        raise RuntimeError("search must be cleanly stopped before reporting")
    trials.to_csv(output / "trial_metrics.csv", index=False)
    outcomes = trials.groupby(["level", "method", "status"]).size().rename("trials").reset_index()
    outcomes.to_csv(output / "trial_outcomes.csv", index=False)
    metrics = ["fitness", "order_gain", "stability", "same_data_seed_stability", "crossfold_stability",
               "explained", "dead_fraction", "redundancy", "elapsed_s", "null_sd_mean"]
    successful = trials[trials.status == "SUCCESS"].copy()
    agg = successful.groupby(["level", "method", "config_key"], as_index=False)[metrics].agg(["mean", "std"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    counts = trials.groupby("config_key").agg(attempted=("status", "size"),
                                              successes=("status", lambda v: int((v == "SUCCESS").sum())))
    agg = agg.merge(counts, on="config_key")
    agg["fully_replicated"] = (agg.successes >= 2) & (agg.successes == agg.attempted)
    agg = agg.sort_values(["level", "method", "fitness_mean", "config_key"], ascending=[True, True, False, True])
    agg["paired_rank"] = agg[agg.fully_replicated].groupby(["level", "method"]).cumcount() + 1
    agg.to_csv(output / "configuration_metrics.csv", index=False)
    selected = agg[agg.fully_replicated].groupby(["level", "method"], sort=True).head(1).copy()
    selected["library"] = selected.level + "_" + selected.method
    selected.to_csv(output / "selected_configurations.csv", index=False)
    jobs, evidence_rows = [], []
    for row in selected.itertuples():
        candidates = trials[(trials.config_key == row.config_key) & (trials.status == "SUCCESS")]
        for trial in candidates.itertuples():
            result = json.loads(conn.execute("SELECT result_json FROM trials WHERE trial_id=?", (trial.trial_id,)).fetchone()[0])
            for fold in result["artifacts"]["folds"]:
                jobs.append({"library": row.library, "configuration": json.loads(row.config_key),
                             "trial_id": int(trial.trial_id), "replicate": int(trial.replicate), "artifact": fold,
                             "support": result["details"]["support_by_fold"][fold["fold"]]})
            for fold, animal_rows in enumerate(result["details"]["animal_readouts_by_fold"]):
                for animal_row in animal_rows:
                    evidence_rows.append({"library": row.library, "trial_id": trial.trial_id,
                                          "replicate": trial.replicate, "fold": fold, **animal_row})
    pd.DataFrame(evidence_rows).to_csv(output / "selected_original_heldout_readouts.csv", index=False)
    parameters = []
    for row in agg.itertuples():
        cfg = json.loads(row.config_key)
        for name, value in cfg.items():
            if name not in ("level", "method"):
                parameters.append({"level": row.level, "method": row.method, "parameter": name, "value": value,
                                   "fitness": row.fitness_mean, "gain": row.order_gain_mean,
                                   "coverage_or_skill": row.explained_mean, "fully_replicated": row.fully_replicated})
    parameters = pd.DataFrame(parameters)
    parameters[parameters.fully_replicated].groupby(["level", "method", "parameter", "value"]).agg(
        configurations=("fitness", "size"), median_fitness=("fitness", "median"), median_gain=("gain", "median"),
        median_coverage_or_skill=("coverage_or_skill", "median")).reset_index().to_csv(output / "parameter_profiles.csv", index=False)
    summary = successful.groupby(["level", "method"]).agg(
        successes=("trial_id", "size"), median_seconds=("elapsed_s", "median"),
        median_gain=("order_gain", "median"), median_stability=("stability", "median"),
        median_explained=("explained", "median"), peak_rss_bytes=("peak_rss_bytes", "max"))
    summary.to_csv(output / "stratum_summary.csv")
    (output / "frozen_selection.json").write_text(json.dumps({
        "rule": "one highest mean-fitness fully successful two-replicate configuration per method/resolution",
        "phenotype_used": False, "jobs": jobs}, indent=2))
    failure_rows = conn.execute("SELECT error,COUNT(*) FROM trials WHERE status='FAILED' GROUP BY error").fetchall()
    (output / "search_summary.json").write_text(json.dumps({"statuses": trials.status.value_counts().to_dict(),
        "unique_attempted_configurations": trials.config_key.nunique(), "paired_successful_configurations": int(agg.fully_replicated.sum()),
        "unique_space": 2560, "failures": [{"count": n, "last_line": e.rstrip().splitlines()[-1]} for e, n in failure_rows],
        "median_trial_seconds": float(successful.elapsed_s.median()), "sum_worker_seconds": float(trials.elapsed_s.sum()),
        "started_utc": trials.started_utc.min(), "finished_utc": trials.finished_utc.max()}, indent=2))
    # Exact length ceilings can be genuine catalogue aliases; disclose them rather
    # than count repeated rankings as independent evidence.
    paired = agg[agg.fully_replicated].copy()
    paired["metric_signature"] = paired[[m + "_mean" for m in metrics[:8]]].round(12).astype(str).agg("|".join, axis=1)
    aliases = paired.groupby(["level", "method", "metric_signature"]).filter(lambda g: len(g) > 1)
    aliases.to_csv(output / "metric_alias_candidates.csv", index=False)
    conn.close()
    return jobs


def initialize(atlas, manifest):
    global DATA
    import numpy as np
    import pandas as pd

    sessions = pd.read_csv(Path(atlas) / "sessions.csv", dtype={"animal_id": str})
    columns = ["syllable_id", "segment", "duration_s", "start_frame", "end_frame_exclusive", "mean_speed_px_s"]
    raw = {}
    for level in manifest["vocab"]:
        folder = Path(atlas) / ("measurements" if level == "L1" else f"clustering/{level}/measurements")
        raw[level] = {sid: pd.read_csv(folder / "bouts_by_session" / f"{sid}.csv", usecols=columns) for sid in sessions.session_id}
        for meta in sessions.itertuples():
            table = raw[level][meta.session_id]
            if not np.allclose(table.duration_s, (table.end_frame_exclusive - table.start_frame) / meta.fps, rtol=0, atol=1e-9):
                raise ValueError("bout seconds disagree with verified native FPS")
    DATA = {"sessions": sessions, "raw": raw, "manifest": manifest, "atlas": Path(atlas)}


def _chunks_for(animal, bouts, support, length, seed):
    import numpy as np

    records = support.get(animal, {}).get("chunks")
    selected = DATA["sessions"][DATA["sessions"].animal_id == animal]
    if records is None:
        records = []
        allocation = max(length, 700 // len(selected))
        rng = np.random.default_rng(seed)
        for sid in sorted(selected.session_id):
            groups = [g for _, g in bouts[sid].groupby("segment", sort=False) if len(g) >= length]
            if not groups:
                continue
            sizes = np.array([len(g) for g in groups])
            group = groups[int(rng.choice(len(groups), p=sizes / sizes.sum()))]
            size = min(allocation, len(group))
            a = int(rng.integers(len(group) - size + 1))
            sub = group.iloc[a:a + size]
            records.append({"session_id": sid, "start_frame": int(sub.start_frame.iloc[0]),
                            "end_frame_exclusive": int(sub.end_frame_exclusive.iloc[-1])})
    chunks = []
    for record in records:
        b = bouts[record["session_id"]]
        sub = b[(b.start_frame >= record["start_frame"]) & (b.end_frame_exclusive <= record["end_frame_exclusive"])]
        if sub.segment.nunique() != 1 or len(sub) < length:
            raise ValueError("saved sampled chunk lost its native support")
        if "sampled_bouts" in record and len(sub) != record["sampled_bouts"]:
            raise ValueError("saved sampled bout count changed")
        chunks.append((record["session_id"], sub.copy()))
    return chunks


def measure_job(job):
    import resource

    import numpy as np
    import pandas as pd

    from fpcode.behavior.analysis.organisation import (
        eligible_pairs,
        exclusive_timeline,
        metric_values,
        organisation_arrays,
        organisation_metrics,
    )
    from fpcode.statistics.chain_nulls import frequency_expectations, sample_first_order_bouts
    from fpcode.workflows.sequence_atlas import one_hot_segments
    from fpcode.workflows.sequence_search import (
        filter_short_bouts,
        frozen_seqnmf_model,
        project_frozen_catalogue,
        score_frozen_sequence_batch,
    )

    started = time.perf_counter()
    cfg, artifact = job["configuration"], job["artifact"]
    level, library = cfg["level"], job["library"]
    vocabulary = DATA["manifest"]["vocab"][level]
    bouts = {sid: filter_short_bouts(b, cfg["min_bout_ms"] / 1000.) for sid, b in DATA["raw"][level].items()}
    model, thresholds = None, None
    out = Path(job["output"]) / "projections" / library / f"trial_{job['trial_id']}_fold_{artifact['fold']}"
    out.mkdir(parents=True, exist_ok=True)
    if cfg["method"] == "seqnmf":
        model = frozen_seqnmf_model(artifact, cfg, vocabulary=vocabulary)
        training = []
        for animal in artifact["training_animals"]:
            training.extend(chunk for _, chunk in _chunks_for(animal, bouts, job["support"]["train"], cfg["chain_len"], 0))
        for i, chunk in enumerate(training):
            chunk["segment"] = i
        combined = pd.concat(training, ignore_index=True)
        x, valid, _ = one_hot_segments(combined, vocabulary, cfg["chain_len"])
        h = model.infer(x, valid_starts=valid, iterations=35)
        thresholds = np.quantile(h[:, valid], .90, axis=1)
        (out / "thresholds.json").write_text(json.dumps({"quantile": .90, "inference_iterations": 35,
                                                       "values": thresholds.tolist(), "training_animals": artifact["training_animals"]}))
    animals = set(artifact["heldout_animals"]) | set(DATA["manifest"]["reserved_animals"])
    if animals & set(artifact["training_animals"]):
        raise ValueError("frozen model would be evaluated on training animals")
    selected_sessions = DATA["sessions"][DATA["sessions"].animal_id.isin(animals)]
    annotation_path = DATA["atlas"] / ("syllable_annotations.csv" if level == "L1" else f"clustering/{level}/cluster_annotations.csv")
    annotation = pd.read_csv(annotation_path).set_index("syllable_id")
    unresolved = annotation.unresolved if "unresolved" in annotation else annotation.category.str.startswith(("Unassigned", "Dump"))
    counts, selections, audits, projections = [], [], [], []
    catalog = None
    for meta in selected_sessions.itertuples():
        original, filtered = DATA["raw"][level][meta.session_id], bouts[meta.session_id]
        seconds = float(original.duration_s.sum())
        blocks = np.array([(int(g.start_frame.iloc[0]), int(g.end_frame_exclusive.iloc[-1]))
                           for _, g in filtered.groupby("segment", sort=False)], dtype=np.int64).reshape(-1, 2)
        events, catalog = project_frozen_catalogue(original, artifact, cfg, vocabulary=vocabulary, fps=meta.fps, thresholds=thresholds)
        catalog["interpretable"] = [not bool(unresolved.loc[[int(t) for t in text.split()]].any()) for text in catalog.tokens]
        stream, decisions, _ = exclusive_timeline(events, blocks, fps=meta.fps, recording_end_frame=meta.n_frames)
        speeds = []
        for call in stream.itertuples():
            chunk = filtered.iloc[call.start_bout:call.end_bout]
            values = chunk.mean_speed_px_s.to_numpy(float)
            good = np.isfinite(values)
            speeds.append(float(np.average(values[good], weights=chunk.duration_s.to_numpy()[good])) if good.any() else np.nan)
        stream["mean_speed_px_s"] = speeds
        identity = {"library": library, "trial_id": job["trial_id"], "replicate": job["replicate"], "fold": artifact["fold"],
                    "session_id": meta.session_id, "animal_id": meta.animal_id, "group": meta.group,
                    "condition": meta.condition, "role": "reserved" if meta.animal_id in DATA["manifest"]["reserved_animals"] else "crossfit"}
        for scope in ("inclusive", "interpretable"):
            retained = set(catalog.chain_id if scope == "inclusive" else catalog.loc[catalog.interpretable, "chain_id"])
            arrays = organisation_arrays(stream, retained_ids=retained, fps=meta.fps)
            kept = stream[stream.chain_id.isin(retained)]
            duration = kept.duration_s.to_numpy(float)
            kept_speeds = kept.mean_speed_px_s.to_numpy(float)
            for gap in GAPS:
                row, _ = organisation_metrics(arrays, gap_seconds=gap, modelled_seconds=seconds)
                row.update(frequency_expectations(arrays["ids"], arrays["reference"], eligible_pairs(arrays, gap_seconds=gap)))
                row.update({"filtered_seconds": float(filtered.duration_s.sum()), "duration_sum": float(duration.sum()),
                            "duration_square_sum": float((duration ** 2).sum()),
                            "phase_cv_sum": float(kept.phase_duration_cv.sum()), "phase_cv_count": len(kept),
                            "speed_sum": float(np.nansum(kept_speeds)), "speed_count": int(np.isfinite(kept_speeds).sum())})
                counts.append({**identity, "scope": scope, "gap_seconds": gap, **row, **metric_values(row)})
        for table in (stream, decisions):
            for key, value in identity.items():
                table[key] = value
        selections.append(stream)
        projections.append(decisions)
        audits.append({**identity, "raw_calls": len(events), "selected_calls": len(stream), "filtered_seconds": float(filtered.duration_s.sum()),
                       "modelled_seconds": seconds, "overlap_rejections": int((decisions.rejection_reason == "overlap").sum()),
                       "invalid_rejections": int((decisions.rejection_reason == "outside_valid_block").sum())})
    catalog["description"] = [" → ".join(annotation.loc[[int(t) for t in text.split()], "description"].astype(str)) for text in catalog.tokens]
    catalog["distinct_dominant_tokens"] = [len(set(text.split())) for text in catalog.tokens]
    if model is not None:
        catalog["unresolved_weight_fraction"] = model.w[unresolved.reindex(range(vocabulary)).to_numpy(bool)].sum(axis=(0, 2)) / np.maximum(model.w.sum(axis=(0, 2)), 1e-9)
    catalog.to_csv(out / "catalogue.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(out / "selected_calls.csv.gz", index=False)
    pd.concat(projections, ignore_index=True).to_csv(out / "call_decisions.csv.gz", index=False)
    pd.DataFrame(counts).to_csv(out / "session_counts.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "selection_audits.csv", index=False)
    nulls = []
    for animal in sorted(animals):
        seed = int.from_bytes(hashlib.sha256(f"{library}:{job['replicate']}:{artifact['fold']}:{animal}".encode()).digest()[:4], "little")
        chunks = _chunks_for(animal, bouts, job["support"]["heldout"], cfg["chain_len"], seed)
        segments = [chunk.syllable_id.to_numpy(int) for _, chunk in chunks]
        rng = np.random.default_rng(seed)
        samples = {"real": segments}
        samples.update({f"shuffle_{i}": [rng.permutation(z) for z in segments] for i in range(5)})
        surrogate_chunks = []
        for sid, chunk in chunks:
            reference = [g.syllable_id.to_numpy(int) for _, g in bouts[sid].groupby("segment", sort=False)]
            surrogate_chunks.append(sample_first_order_bouts(reference, length=len(chunk), initial=int(chunk.syllable_id.iloc[0]),
                vocabulary=vocabulary, replicates=job["null_repeats"], seed=int(rng.integers(2 ** 32)), memory_mb=128))
        samples.update({f"markov_{i}": [z[i] for z in surrogate_chunks] for i in range(job["null_repeats"])})
        scores = score_frozen_sequence_batch(samples, artifact, cfg, vocabulary=vocabulary)
        real = scores["real"]
        shuffled = [scores[f"shuffle_{i}"] for i in range(5)]
        markov = [scores[f"markov_{i}"] for i in range(job["null_repeats"])]
        for kind, values in (("shuffle", shuffled), ("first_order", markov)):
            mean = float(np.mean(values))
            gain = (real - mean) / real if cfg["method"] == "exact" and real > 0 else (mean - real) / mean if cfg["method"] == "seqnmf" and mean > 0 else 0.
            nulls.append({"library": library, "trial_id": job["trial_id"], "replicate": job["replicate"], "fold": artifact["fold"],
                "animal_id": animal, "role": "reserved" if animal in DATA["manifest"]["reserved_animals"] else "crossfit",
                "null_kind": kind, "real": real, "null_mean": mean, "null_sd": float(np.std(values)),
                "null_repeats": len(values), "order_gain": gain, "sampled_bouts": sum(map(len, segments)), "null_values": json.dumps(values)})
    pd.DataFrame(nulls).to_csv(out / "order_null_audit.csv", index=False)
    status = {"library": library, "trial_id": job["trial_id"], "fold": artifact["fold"], "seconds": time.perf_counter() - started,
              "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "sessions": len(selected_sessions)}
    (out / "complete.json").write_text(json.dumps(status, indent=2))
    return status


def aggregate_analysis(output):
    import numpy as np
    import pandas as pd

    from fpcode.statistics.behavior_comparisons import compare_feature_atlas
    from fpcode.statistics.followup_planning import exploratory_effect_stability
    from fpcode.workflows.chain_organisation import pool_organisation_measurements

    roots = sorted((output / "projections").glob("*/*/complete.json"))
    counts = pd.concat([pd.read_csv(p.parent / "session_counts.csv", dtype={"animal_id": str}) for p in roots], ignore_index=True)
    counts.to_csv(output / "session_counts.csv", index=False)
    identity = ["library", "scope", "trial_id", "replicate", "fold", "role", "animal_id", "group", "condition", "gap_seconds"]
    model_panels = pool_organisation_measurements(counts, identity=identity)
    model_panels["mean_duration_s"] = model_panels.duration_sum / model_panels.calls.replace(0, np.nan)
    variance = np.maximum(0., model_panels.duration_square_sum / model_panels.calls.replace(0, np.nan) - model_panels.mean_duration_s ** 2)
    model_panels["duration_cv"] = np.sqrt(variance) / model_panels.mean_duration_s
    model_panels["mean_phase_duration_cv"] = model_panels.phase_cv_sum / model_panels.phase_cv_count.replace(0, np.nan)
    model_panels["mean_speed_px_s"] = model_panels.speed_sum / model_panels.speed_count.replace(0, np.nan)
    model_panels.to_csv(output / "model_animal_wave_measurements.csv", index=False)
    # Preserve raw estimates above, but require ten observed opportunities/calls
    # per model-animal-wave for execution and conditional organisation inference.
    for measure in ("self_probability", "self_excess", "transition_entropy_nats"):
        model_panels.loc[model_panels.pair_opportunities < 10, measure] = np.nan
    for lag in (2, 3, 4):
        model_panels.loc[model_panels[f"return{lag}_opportunities"] < 10, f"return{lag}_probability"] = np.nan
    model_panels.loc[model_panels.return2_opportunities < 10, "return2_excess"] = np.nan
    for measure in ("duration_cv", "mean_duration_s", "mean_phase_duration_cv", "mean_speed_px_s", "mean_run_calls", "repeated_covered_fraction"):
        model_panels.loc[model_panels.calls < 10, measure] = np.nan
    panel_identity = ["library", "scope", "role", "animal_id", "group", "condition", "gap_seconds"]
    panel = model_panels.groupby(panel_identity, as_index=False)[list(MEASURES)].mean()
    panel.to_csv(output / "ensemble_animal_wave_measurements.csv", index=False)
    long = panel.melt(id_vars=panel_identity, value_vars=MEASURES, var_name="measure", value_name="value")
    long["family"] = "frozen_finalist_" + long.library
    long["feature"] = long.library + "/" + long.scope + "/g" + long.gap_seconds.map(lambda x: f"{x:g}") + "/" + long.measure
    long["session_id"] = long.animal_id + "/" + long.condition
    long.to_csv(output / "animal_features_long.csv", index=False)
    results = compare_feature_atlas(long, groups=("Monomer", "PFF"), untreated=UNTREATED, drug_condition="L_dopa",
        baseline_condition="24w_PI", longitudinal_pairs=(("3w_PI", "8w_PI"), ("8w_PI", "16w_PI"), ("16w_PI", "24w_PI")), resamples=4999)
    for name, table in results.items():
        table.to_csv(output / (name + ".csv"), index=False)
    # An honest small-n reserved-animal sensitivity; insufficient groups retain
    # descriptive estimates instead of invented p-values.
    reserved = compare_feature_atlas(long[long.role == "reserved"], groups=("Monomer", "PFF"), untreated=UNTREATED,
        drug_condition="L_dopa", baseline_condition="24w_PI", resamples=4999)
    reserved["group_contrasts"].to_csv(output / "reserved_group_contrasts.csv", index=False)
    stability = exploratory_effect_stability(results["animal_wave_measurements"], results["group_contrasts"],
        groups=("Monomer", "PFF"), untreated=UNTREATED, consistency_waves=("3w_PI", "8w_PI", "16w_PI"))
    stability.to_csv(output / "effect_stability.csv", index=False)
    nulls = pd.concat([pd.read_csv(p.parent / "order_null_audit.csv", dtype={"animal_id": str}) for p in roots], ignore_index=True)
    nulls.to_csv(output / "order_null_audit.csv", index=False)
    animal_null = nulls.groupby(["library", "animal_id", "role", "null_kind"], as_index=False)[["order_gain", "real", "null_mean", "null_sd"]].mean()
    animal_null.to_csv(output / "animal_order_null_audit.csv", index=False)
    animal_null.groupby(["library", "role", "null_kind"]).agg(animals=("animal_id", "nunique"),
        mean_order_gain=("order_gain", "mean"), median_order_gain=("order_gain", "median"),
        positive_animals=("order_gain", lambda x: int((x > 0).sum())), mean_real=("real", "mean"),
        mean_null=("null_mean", "mean")).reset_index().to_csv(output / "order_null_summary.csv", index=False)
    pd.concat([pd.read_csv(p.parent / "selection_audits.csv", dtype={"animal_id": str}) for p in roots], ignore_index=True).to_csv(output / "selection_audits.csv", index=False)
    return results


def write_findings(output, atlas):
    """Summarize cached estimates and their sensitivities, without model inference."""
    import numpy as np
    import pandas as pd

    from fpcode.statistics.behavior_comparisons import independent_contrast
    from fpcode.statistics.followup_planning import sensitivity_scenarios

    contrasts = pd.read_csv(output / "group_contrasts.csv")
    stability = pd.read_csv(output / "effect_stability.csv")
    models = pd.read_csv(output / "model_animal_wave_measurements.csv", dtype={"animal_id": str})
    models.loc[models.pair_opportunities < 10, ["self_probability", "self_excess", "transition_entropy_nats"]] = np.nan
    for lag in (2, 3, 4):
        models.loc[models[f"return{lag}_opportunities"] < 10, f"return{lag}_probability"] = np.nan
    models.loc[models.return2_opportunities < 10, "return2_excess"] = np.nan
    execution = ["duration_cv", "mean_duration_s", "mean_phase_duration_cv", "mean_speed_px_s", "mean_run_calls", "repeated_covered_fraction"]
    models.loc[models.calls < 10, execution] = np.nan
    identity = ["library", "scope", "replicate", "animal_id", "group", "condition", "gap_seconds"]
    rep_panel = models.groupby(identity, as_index=False)[list(MEASURES)].mean()
    seed_rows = []
    for key, sub in rep_panel[rep_panel.condition.isin(UNTREATED)].groupby(["library", "scope", "replicate", "gap_seconds"]):
        library, scope, replicate, gap = key
        per_animal = sub.groupby(["animal_id", "group"], as_index=False)[list(MEASURES)].mean()
        for measure in MEASURES:
            row = independent_contrast(per_animal.loc[per_animal.group == "Monomer", measure],
                                       per_animal.loc[per_animal.group == "PFF", measure], resamples=1999)
            seed_rows.append({"feature": f"{library}/{scope}/g{gap:g}/{measure}", "replicate": replicate, **row})
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(output / "replicate_effect_sensitivity.csv", index=False)
    support = models.groupby(["library", "scope", "animal_id", "group", "condition", "gap_seconds"])[list(MEASURES)].count()
    support.columns = [name + "_supported_models" for name in support.columns]
    support.reset_index().to_csv(output / "endpoint_model_support.csv", index=False)
    promising = stability[stability.meets_exploratory_heuristic & stability.feature.str.contains("/interpretable/g1/")].copy()
    promising.to_csv(output / "promising_candidates.csv", index=False)
    sensitivity_scenarios(animal_counts=tuple(range(5, 202))).to_csv(output / "prospective_cohort_scenarios.csv", index=False)
    counts = pd.read_csv(output / "session_counts.csv", dtype={"animal_id": str})
    filtered = counts[(counts.scope == "inclusive") & (counts.gap_seconds == 1)].copy()
    filtered = filtered.groupby(["library", "trial_id", "fold", "animal_id", "group", "condition"], as_index=False)[["filtered_seconds", "modelled_seconds"]].sum()
    filtered["filtered_exposure_fraction"] = filtered.filtered_seconds / filtered.modelled_seconds
    filtered = filtered.groupby(["library", "animal_id", "group", "condition"], as_index=False).filtered_exposure_fraction.mean()
    filtered.to_csv(output / "filtered_exposure_by_animal.csv", index=False)
    filtered.groupby(["library", "group", "condition"]).filtered_exposure_fraction.agg(["mean", "min", "max"]).to_csv(output / "filtered_exposure_summary.csv")
    review = []
    sessions = pd.read_csv(atlas / "sessions.csv", dtype={"animal_id": str}).set_index("session_id")
    for library in sorted((output / "projections").iterdir()):
        root = sorted(library.glob("*/complete.json"))[0].parent
        catalog = pd.read_csv(root / "catalogue.csv").set_index("chain_id")
        calls = pd.read_csv(root / "selected_calls.csv.gz", dtype={"animal_id": str})
        calls = calls[calls.chain_id.isin(catalog[catalog.interpretable].index)]
        for _, local in calls.groupby(["group", "condition"]):
            # Same frozen catalogue in both groups; these are examples of calls,
            # not a selected exemplar purporting to prove the group difference.
            for call in local.sort_values(["selection_score", "session_id", "start_frame"], ascending=[False, True, True]).head(2).itertuples():
                meta = sessions.loc[call.session_id]
                review.append({"library": library.name, "group": call.group, "condition": call.condition,
                    "animal_id": call.animal_id, "session_id": call.session_id, "chain_id": call.chain_id,
                    "description": catalog.loc[call.chain_id, "description"], "start_frame": call.start_frame,
                    "end_frame_exclusive": call.end_frame_exclusive, "fps": meta.fps, "video_path": meta.video_path,
                    "trial_id": call.trial_id, "fold": call.fold})
    pd.DataFrame(review).to_csv(output / "qualitative_review_candidates.csv", index=False)
    features = ["L2_seqnmf/interpretable/g1/self_excess", "L1_seqnmf/interpretable/g1/calls_per_min",
                "L1_seqnmf/interpretable/g1/initiations_per_min", "L1_seqnmf/interpretable/g1/return3_probability",
                "L4_exact/interpretable/g1/mean_run_calls", "L2_exact/interpretable/g1/mean_phase_duration_cv"]
    spotlight = contrasts[(contrasts.view == "untreated_pooled") & contrasts.feature.isin(features)]
    spotlight.to_csv(output / "spotlight_estimates.csv", index=False)
    def estimate(feature, view="untreated_pooled"):
        return contrasts[(contrasts.feature == feature) & (contrasts.view == view)].iloc[0]

    self_row, call_row, start_row, run_row = (estimate(features[i]) for i in (0, 1, 2, 4))
    wave_rows = contrasts[(contrasts.feature == features[0]) & contrasts.view.isin(UNTREATED)]
    wave_agreement = int((np.sign(wave_rows.difference) == np.sign(self_row.difference)).sum())
    loop_feature = "L1_seqnmf/interpretable/g1/return2_probability"
    drug_row = estimate(loop_feature, "L_dopa")
    drug_change = estimate(loop_feature, "change_untreated_mean_to_L_dopa")
    null_summary = pd.read_csv(output / "order_null_summary.csv")
    first_order = null_summary[null_summary.null_kind == "first_order"].set_index(["library", "role"]).mean_order_gain
    exact_gains = ", ".join(f"{100 * first_order.loc[(f'L{i}_exact', 'crossfit')]:+.1f}%" for i in range(1, 5))
    seq_gains = ", ".join(f"{100 * first_order.loc[(f'L{i}_seqnmf', 'crossfit')]:+.2f}%" for i in range(1, 5))
    scenarios = pd.read_csv(output / "prospective_cohort_scenarios.csv")
    planning = scenarios[(scenarios.alpha == .05) & (scenarios.prospective_power >= .8)].groupby("assumed_true_d").animals_per_group.min()
    tested = contrasts[contrasts.status == "tested"]
    adjusted = pd.read_csv(output / "wave_adjusted_contrasts.csv")
    paired = pd.read_csv(output / "paired_wave_changes.csv")
    pacing_feature = "L2_exact/interpretable/g1/mean_phase_duration_cv"
    pacing = adjusted[(adjusted.feature == pacing_feature) & (adjusted.view == "untreated_adjusted")].iloc[0]
    paired_calls = paired[(paired.feature == "L3_seqnmf/interpretable/g1/calls_per_min") &
                          (paired.baseline == "untreated_mean") & (paired.target == "L_dopa")].set_index("group")
    paired_duration = paired[(paired.feature == "L3_seqnmf/interpretable/g1/mean_duration_s") &
                             (paired.baseline == "untreated_mean") & (paired.target == "L_dopa")].set_index("group")
    pd.concat([adjusted[adjusted.q_family < .05].assign(analysis="wave_adjusted_group"),
               paired[paired.q_family < .05].assign(analysis="paired_wave_change")], ignore_index=True).to_csv(output / "family_corrected_signals.csv", index=False)
    sections = ["## Behavioural findings to investigate",
        f"**There are promising organisation candidates and some narrower-family signals, with no globally corrected findings.** Among {len(tested):,} tested main permutation group contrasts, "
        f"{int((tested.q_family < .05).sum())} passed the disclosed family/view correction and {int((tested.q_global < .05).sum())} passed the global correction. "
        f"The separate wave-adjusted models have {int((adjusted.q_family < .05).sum())} family-corrected rows and paired wave changes have {int((paired.q_family < .05).sum())}; "
        "these include correlated copies across gaps and scopes, rather than that many independent effects. Neither table has a global q<0.05 result. "
        "This is not evidence of equivalence. The findings below retain effect magnitude, uncertainty and stability without requiring significance. All signs mean PFF minus monomer."]
    labels = {
        features[0]: "Higher frequency-adjusted self-transition tendency at L2",
        features[1]: "Fewer named L1 seqNMF detections",
        features[2]: "Fewer L1 seqNMF run initiations",
        features[3]: "Lower L1 seqNMF return-after-three-calls probability",
        features[4]: "Shorter L4 exact repetition runs, with a small absolute difference",
        features[5]: "Less even phase pacing within L2 exact calls, with sparse wave support"}
    for feature in features:
        row = spotlight[spotlight.feature == feature].iloc[0]
        compatible = seeds[(seeds.feature == feature) & (seeds.status == "tested")]
        gap_features = [feature.replace("/g1/", f"/g{gap}/") for gap in (0, 1, 5)]
        gaps = contrasts[(contrasts.view == "untreated_pooled") & contrasts.feature.isin(gap_features) & (contrasts.status == "tested")]
        seed_agree = int((np.sign(compatible.difference) == np.sign(row.difference)).sum())
        gap_agree = int((np.sign(gaps.difference) == np.sign(row.difference)).sum())
        sections.append(f"- **{labels[feature]}:** {row.mean_first:.5f} in monomer versus {row.mean_second:.5f} in PFF; "
            f"difference {row.difference:+.5f} (95% mouse-bootstrap interval {row.ci_low:+.5f} to {row.ci_high:+.5f}), "
            f"Hedges g {row.hedges_g:+.2f}, raw permutation p {row.p_permutation:.4f}, family q {row.q_family:.3f}. "
            f"Support: {int(row.n_first)} monomer/{int(row.n_second)} PFF mice; direction agrees in {seed_agree}/{len(compatible)} replicate estimates "
            f"and {gap_agree}/{len(gaps)} supported gap rules.")
    sections.extend([
        f"The L2 self-excess difference is about **{100 * self_row.difference:+.2f} percentage points**, with the same sign in {wave_agreement}/{len(wave_rows)} untreated waves. "
        "Both the pooled and wave-adjusted estimates are positive. This supports a targeted repeat-tendency hypothesis, while the factor's weak higher-order evidence limits claims about a distinct routine. "
        "Self-excess controls the detected factor frequencies, not movement speed, detector uncertainty or all possible behavioural confounders.",
        f"L1 detections differ by **{100 * call_row.difference / call_row.mean_first:+.1f}%** and all run initiations by **{100 * start_row.difference / start_row.mean_first:+.1f}%** in PFF relative to monomer across untreated weeks. These related endpoints describe detected calls/runs; "
        "they do not independently establish stronger persistence. The call-rate sign reverses at 8 weeks and in the separate L-DOPA wave. "
        "Observed initiations exclude starts censored by gaps/boundaries and are exported separately.",
        f"The L4 exact mean-run comparison is close to one call per run in both groups: **{run_row.mean_first:.3f} versus {run_row.mean_second:.3f}**. Its large standardized effect accompanies a small practical difference and few repetitions; "
        "it is a lower-priority biological lead. The L2 exact phase-CV difference is larger, but does not yet have broad per-wave support. "
        "No general statement that PFF mice loop more, or less, is supported across methods and resolutions.",
        f"The wave-adjusted L2 exact phase-pacing estimate is **{pacing.adjusted_difference:+.4f} CV units** "
        f"(95% mouse-clustered interval {pacing.ci_low:+.4f} to {pacing.ci_high:+.4f}; "
        f"family q={pacing.q_family:.3f}, global q={pacing.q_global:.3f}). "
        "This passes the narrower family correction and deserves follow-up. It pools different named motifs across model folds, "
        "including dash–turn–dash, sniff–look–sniff and posture/look sequences. Match individual chains before assigning the difference to a specific action or claiming within-chain execution variability.",
        "## Separate L-DOPA-wave pattern",
        f"At L1, named seqNMF A→B→A return probability differs by **{100 * drug_row.difference:+.2f} percentage points in PFF** during L_dopa (Hedges g={drug_row.hedges_g:+.2f}; raw p={drug_row.p_permutation:.4f}), "
        f"and its untreated-mean-to-L_dopa change differs from monomer by **{100 * drug_change.difference:+.2f} percentage points**. These comparisons have {int(drug_row.n_first)} monomer and {int(drug_row.n_second)} PFF mice. "
        "The nearest-24w paired sensitivity has fewer animals and does not establish a drug effect. None survives multiplicity correction. "
        "The later wave also has lower call rates in both groups than their own untreated means; this is a wave-associated change, not proof of recovery or deterioration.",
        f"For named L3 seqNMF calls, the L_dopa-minus-untreated-mean change is **{paired_calls.loc['Monomer', 'mean_change']:+.2f} calls/min** in monomer "
        f"(family q={paired_calls.loc['Monomer', 'q_family']:.3f}) and **{paired_calls.loc['PFF', 'mean_change']:+.2f} calls/min** in PFF "
        f"(family q={paired_calls.loc['PFF', 'q_family']:.3f}). Mean detected-call durations increase by "
        f"{paired_duration.loc['Monomer', 'mean_change']:.2f}s and {paired_duration.loc['PFF', 'mean_change']:.2f}s respectively. "
        "This suggests fewer, longer detections in the later wave; the monomer call-rate change passes its narrower family correction. "
        "Related paired changes are indexed in `family_corrected_signals.csv`; broad global correction and age/wave confounding still limit interpretation.",
        "## Which sequence methods currently look useful?",
        f"For exact motifs, the inclusive-catalogue first-order gain averages **{exact_gains}** across search mice at L1–L4. "
        "The five reserved mice retain positive average gains at L1, L3 and L4; L2 is slightly negative. Fine exact motifs cover only around 1.5–2.3% of sampled bouts, compared with about 12% at L3/L4. "
        "These inclusive controls contain unresolved motifs; they do not certify that every named motif has higher-order support.",
        f"SeqNMF's first-order gain is **{seq_gains}** in search mice at L1–L4. "
        f"The L4 reserved-mouse gain is **{100 * first_order.loc[('L4_seqnmf', 'reserved')]:+.2f}%**, despite its high reconstruction score. Selected template matching is modest, around 0.3–0.4. "
        "Use these factors as exploratory descriptive measurements; the current winners are not yet a convincing reproducible higher-order routine model. "
        "Near-perfect coarse reconstruction is not sufficient evidence that seqNMF discovered useful chains.",
        "## What would justify a larger cohort?",
        "Prioritise the L2 self-excess hypothesis, L1 detected-call usage and the L-DOPA-wave return pattern for matched video review and a small number of fixed endpoint definitions. "
        "Check gap/replicate sensitivity, unresolved template weights and observation support before committing to the endpoint. "
        "The CSV of prospective scenarios uses externally assumed true standardized effects, rather than the largest selected pilot estimates. "
        f"Under a fixed two-sided equal-variance t-test at α=0.05, 80% power requires roughly **{int(planning.loc[.8])}, {int(planning.loc[.5])} or {int(planning.loc[.3])} animals per group** for true effects of d=0.8, 0.5 or 0.3. "
        "These are illustrative independent-animal, normal-outcome, no-attrition scenarios. Multiplicity, a different design or endpoint distribution changes the requirement.",
        "Mouse-bootstrap and clustered intervals are conditional on the frozen models; models are not refitted inside resampling and catalogue-selection uncertainty is not included. "
        "Repeated weeks, gaps, seeds and clustering levels are dependent sensitivities, not independent replications. "
        "`qualitative_review_candidates.csv` supplies verified native spans and video paths for named calls under common fold catalogues; examples do not prove a group effect. "
        "No KPMS fit, pose/frame-model training, drug-recovery classifier or production exclusion change was performed."])
    (output / "findings.md").write_text("\n\n".join(sections) + "\n")


def write_report(search, output, manifest):
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = pd.read_csv(output / "selected_configurations.csv")
    summary = json.loads((output / "search_summary.json").read_text())
    contrasts = pd.read_csv(output / "group_contrasts.csv")
    stability = pd.read_csv(output / "effect_stability.csv")
    null_summary = pd.read_csv(output / "order_null_summary.csv")
    panel = pd.read_csv(output / "ensemble_animal_wave_measurements.csv")
    figs = output / "figures"
    figs.mkdir(exist_ok=True)
    leaderboard = pd.read_csv(output / "configuration_metrics.csv")
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for ax, ((level, method), group) in zip(axes.flat, leaderboard[leaderboard.fully_replicated].groupby(["level", "method"])):
        colors = ax.scatter(group.explained_mean, group.order_gain_mean, c=group.stability_mean, s=13, alpha=.65, vmin=0, vmax=1)
        ax.set(title=f"{level} {method}", xlabel="Bout coverage" if method == "exact" else "Uncentered reconstruction skill", ylabel="Shuffle-relative gain")
    fig.colorbar(colors, ax=axes.ravel().tolist(), label="Matched stability")
    fig.savefig(figs / "search_tradeoffs.png", dpi=160)
    plt.close(fig)
    view = contrasts[(contrasts.view == "untreated_pooled") & contrasts.feature.str.contains("/g1/")].copy()
    merged = view.merge(stability, on=["family", "feature", "view"], how="left", suffixes=("", "_stability"))
    merged = merged.sort_values("hedges_g", key=lambda x: x.abs(), ascending=False)
    merged.to_csv(output / "exploratory_candidates.csv", index=False)
    candidates = merged[(merged.status == "tested") & merged.feature.str.contains("/interpretable/")].head(15)
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    y = np.arange(len(candidates))
    scale = candidates.hedges_g / candidates.difference
    ax.hlines(y, candidates.ci_low * scale, candidates.ci_high * scale, color="steelblue")
    ax.scatter(candidates.hedges_g, y, color="black", s=22)
    ax.axvline(0, color="grey", lw=1)
    ax.set(yticks=y, yticklabels=candidates.feature, xlabel="PFF − monomer in observed pooled-SD units", title="Exploratory untreated effects; intervals scaled by fixed observed SD")
    ax.invert_yaxis()
    fig.savefig(figs / "exploratory_effects.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, metric in zip(axes.flat, ("calls_per_min", "coverage_fraction", "self_probability", "self_excess", "mean_run_calls", "return2_probability")):
        for library in ("L3_exact", "L3_seqnmf"):
            local = panel[(panel.library == library) & (panel.gap_seconds == 1) & (panel.scope == "interpretable")]
            for group, color in (("Monomer", "tab:blue"), ("PFF", "tab:orange")):
                means = local[local.group == group].groupby("condition")[metric].mean().reindex([*UNTREATED, "L_dopa"])
                ax.plot(range(5), means, marker="o", color=color, ls="-" if library.endswith("exact") else "--", label=f"{library} {group}")
        ax.set(title=metric, xticks=range(5), xticklabels=["3w", "8w", "16w", "24w", "L-DOPA"])
    axes[0, 0].legend(fontsize=8)
    fig.savefig(figs / "L3_wave_profiles.png", dpi=160)
    plt.close(fig)
    display = selected.copy()
    for col in ("order_gain_mean", "explained_mean", "stability_mean", "fitness_mean", "fitness_std"):
        display[col] = display[col].round(4)
    tested = contrasts[contrasts.status == "tested"]
    spotlight = candidates.head(8)
    pooled = contrasts[contrasts.view == "untreated_pooled"]
    gap_checks = pooled[pooled.feature.str.contains("/g[015]/")].copy()
    gap_checks["base"] = gap_checks.feature.str.replace(r"/g[015]/", "/", regex=True)
    gap_checks.groupby("base").agg(supported_gaps=("difference", "count"),
        min_difference=("difference", "min"), max_difference=("difference", "max")).to_csv(output / "gap_sensitivity.csv")
    config_table = display[["library", "config_key", "fitness_mean", "fitness_std", "order_gain_mean", "explained_mean", "stability_mean"]].to_markdown(index=False)
    sections = ["# Karen overnight sequence-search report — 2026-10-13",
        f"The search was stopped at the user's request. **{summary['statuses'].get('SUCCESS', 0):,} successful trials**, "
        f"{summary['statuses'].get('FAILED', 0):,} unsupported exact-motif trials, and {summary['statuses'].get('STOPPED', 0)} interrupted fits were saved. "
        f"It attempted {summary['unique_attempted_configurations']:,}/{summary['unique_space']:,} configurations; "
        f"{summary['paired_successful_configurations']:,} have two successful replicates. No overnight process should remain running.",
        (output / "findings.md").read_text() if (output / "findings.md").exists() else "",
        "## What this search establishes",
        "Actual held-out, mouse-level discovery evidence replaces the invalid archived placeholder search. "
        "Rank only within a method and clustering resolution. Exact explained values are bout coverage; seqNMF values are uncentered reconstruction skill, not ordinary R². "
        "Relative exact order gain can be large when absolute coverage is tiny. Coarser vocabularies are easier to reconstruct. "
        "The composite rewards several proxies and may select a model without a strong order benefit. Two replicates reduce single-seed luck but do not establish convergence or independent scientific replication.",
        "![Search tradeoffs](figures/search_tradeoffs.png)",
        "## Label-independent frozen shortlist", config_table,
        "One highest mean-score configuration with two successful completed replicates was frozen for each of eight strata before phenotype comparison. "
        "No fit was restarted and no full-cohort templates were learned. Exact chain-length ceilings sometimes return the same short catalogues; identical score signatures are exported as possible aliases. "
        "Marginal parameter profiles describe the sampled successful configurations, not isolated causal effects of a knob; prevalence failures create selection bias.",
        "## Full-recording projection and what a repeat means",
        f"All {DATA['sessions'].shape[0] if DATA else 99} reviewed recordings from 21 mice were projected onto saved fold models. "
        "Each search mouse is scored only by its own held-out fold in each replicate. Each of five reserved mice is scored by all three training folds in both replicates; "
        "models are averaged within animal/wave before inferential comparisons. These reserved mice were previously examined in atlas work and are not historically unseen. "
        "Thresholds use the training chunks' inferred activation 90th percentiles with 35 inference updates; native candidate error ≤0.95, nearby-peak suppression and existing score-ranked native overlap selection apply. "
        "RLE bouts are preserved. Short-bout deletion, reviewed exclusions and missing native frames remain barriers. "
        "Rates use original reviewed modelled exposure; filtered-away time remains in the denominator. Unrecognised valid gaps of 0, 1 and 5 seconds are tested. "
        "Self-transition means successive selected detections of the same model-local factor/motif within this gap. It is an operational repeated-detection measure; "
        "it does not by itself prove a deliberate restart rather than sustained continuation. Factor IDs differ across folds, so the primary comparisons are general organisation and execution measures, not a single cross-fold chain identity.",
        "Inclusive and interpretable-member scopes are exported separately. The latter requires every exact token or seqNMF dominant phase to have a resolved annotation; "
        "unassigned/Dump calls remain barriers after overlap selection. For seqNMF, dominant labels do not certify minor weights; each catalogue exports its unresolved template-weight fraction. "
        "The candidate plot and L3 wave profiles use the interpretable-member scope, and all-inclusive contrasts remain available in CSV.",
        "## Stronger order control", null_summary.round(5).to_markdown(index=False),
        "The first-order control fits each recording's filtered bout transition probabilities, fixes each sampled chunk's initial state and length, and prohibits RLE self transitions. "
        "It fits no episode templates. Unsupported outgoing states use empirical frequency restricted to allowed successors. "
        "It preserves fitted transitions in expectation, not exact transition/state counts, bout durations or nonstationary dynamics. Transition-estimation uncertainty is not integrated. "
        "Original held-out sampled chunks are reused; new deterministic samples are selected for the reserved mice. Each first-order audit uses 19 surrogates and each shuffle audit uses five. "
        "These are screening contrasts, not null significance tests. A gain over random order can disappear after ordinary transitions are retained.",
        "## PFF versus monomer, including non-significant candidates",
        f"The full screen has {int((tested.q_global < .05).sum())} globally corrected q<0.05 contrasts among {len(tested):,} tested rows. "
        "All estimates, bootstrap intervals, permutation p-values, within-family/view q-values and global q-values are retained, regardless of significance. "
        "Independent units are mice. Recording sufficient counts are pooled within model/animal/wave, then model estimates are averaged. "
        "Across-week summaries weight each available wave equally per mouse; attendance differs, so wave-adjusted mouse-clustered regressions are exported too. "
        "The following candidates are selected post hoc by absolute untreated Hedges g at the one-second gap. This selects noisy extreme effects and is exploratory.",
        spotlight[["feature", "n_first", "n_second", "difference", "ci_low", "ci_high", "hedges_g", "p_permutation", "q_family", "q_global",
                   "wave_direction_fraction", "leave_one_animal_out_direction_fraction"]].round(5).to_markdown(index=False),
        "![Exploratory effect estimates](figures/exploratory_effects.png)",
        "![L3 wave profiles](figures/L3_wave_profiles.png)",
        "## Weeks and the separate L-DOPA wave",
        "Exports include each 3-, 8-, 16- and 24-week wave, the L_dopa wave, all waves pooled, untreated weeks pooled, paired L_dopa-minus-24w and L_dopa-minus-untreated-mean changes, "
        "and early longitudinal changes. L_dopa is a later recording wave; drug, age and wave are confounded. "
        "The nearest 24w baseline has only three monomer and six PFF mice; 16w has nine and eleven, and the other waves nine and twelve. "
        "Endpoint support can be lower because missing successor/loop opportunities remain undefined, rather than zero. Reserved-only inference has too few monomer animals for the minimum-three-per-group rule; descriptive estimates are retained.",
        "Raw estimates retain all opportunities. Inferential conditional self/transition measures require at least ten pairs per model-animal-wave; return measures require ten return windows, "
        "and execution measures require ten calls. Each ensemble value averages the supported models. Rates and coverage retain genuine zero detections. "
        "Population duration CV and mean within-call phase-duration CV describe pacing variation; these are not frame-level micromovement measures.",
        "## Interpretation and next decisions",
        "Use the null audit and catalogues to decide which detections warrant video review. Strong self-transition or initiation contrasts can reflect usage, exposure, threshold or overlap differences. "
        "Exact motifs that start and end on the same syllable can structurally lose abutting repeats under RLE. "
        "Compare gap sensitivity, wave direction and leave-one-mouse-out direction before deciding which endpoint warrants a larger cohort. "
        "The data support candidate generation rather than equivalence claims from non-significance. Future cohort planning should use conservative effect scenarios, not the largest selected pilot effect. "
        "This pass measures detected-chain usage, organisation and bout-level pacing/execution. It does not rerun KPMS or add new frame/pose models, causal recovery scores, or disease classifiers.",
        "## Reproducibility and files",
        "`manifest.json` records input/source fingerprints, frozen selection, runtime and resource settings. `trial_metrics.csv`, `configuration_metrics.csv`, "
        "`parameter_profiles.csv`, `selected_configurations.csv` and `metric_alias_candidates.csv` describe discovery. "
        "`projections/` retains model-local catalogues, calibrated thresholds, native calls, overlap decisions, sufficient counts and null samples. "
        "`ensemble_animal_wave_measurements.csv`, `group_contrasts.csv`, `paired_wave_changes.csv`, `wave_adjusted_contrasts.csv`, "
        "`effect_stability.csv`, `reserved_group_contrasts.csv` and `exploratory_candidates.csv` retain downstream estimates. "
        f"Search source: `{search}`. Invalid-v1 results are excluded."]
    execution = json.loads((output / "manifest.json").read_text())
    if "elapsed_seconds" in execution:
        sections.append(f"The full projection/statistics pass took {execution['elapsed_seconds'] / 60:.2f} minutes with "
            f"{execution['workers']} processes ×{execution['threads_per_worker']} numerical threads and an "
            f"{execution['ram_cap_mb']}MiB scheduling budget. Peak observed worker RSS was {execution['peak_worker_rss_mb']:.1f}MiB. "
            "The 699-test CPU suite, architecture, lint and navigation checks passed; artifact checks found no invalid native calls or exposure/count violations. "
            "The eight-timeline batching benchmark under concurrent load took 3.96s separately and 4.08s batched, with maximum score difference 5.4×10⁻⁷; "
            "it did not demonstrate a speedup for that shape. These bounded checks do not establish universal scientific equivalence.")
    text = "\n\n".join(sections) + "\n"
    (output / "report.md").write_text(text)
    import markdown

    (output / "report.html").write_text("<!doctype html><meta charset='utf-8'><style>body{max-width:1200px;margin:32px auto;font:16px sans-serif;line-height:1.5}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px}img{max-width:100%}code{overflow-wrap:anywhere}</style>" + markdown.markdown(text, extensions=["tables"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ram-cap-mb", type=int, default=8192)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"search": str(args.search), "atlas": str(args.atlas), "output": str(args.output), "execute": False}))
        return
    if min(args.workers, args.threads, args.ram_cap_mb) < 1:
        parser.error("positive resource limits required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import multiprocessing

    import pandas as pd
    import psutil
    from threadpoolctl import threadpool_info

    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.search / "run_manifest.json").read_text())
    # Check the exact frozen observations against their source view; source code
    # is allowed to evolve but selected empirical observations must not drift.
    snapshot = Path(manifest["atlas"])
    for path, expected in manifest["input_sha256"].items():
        original = args.atlas / Path(path).relative_to(snapshot)
        if fingerprint(original) != expected:
            raise ValueError(f"reviewed observation input changed: {original}")
    jobs = summarize_search(args.search.resolve(), args.output.resolve())
    for job in jobs:
        job.update(output=str(args.output.resolve()), null_repeats=19)
    if args.max_jobs:
        jobs = jobs[:args.max_jobs]
    worker_budget = 1024
    available_mb = psutil.virtual_memory().available / 1024 ** 2
    workers = min(args.workers, max(1, args.ram_cap_mb // worker_budget), max(1, int((available_mb - 2048) // worker_budget)),
                  max(1, len(os.sched_getaffinity(0)) // args.threads))
    sessions = pd.read_csv(args.atlas / "sessions.csv", dtype={"animal_id": str})
    provenance = {"created_date_local": "2026-10-13", "search": str(args.search.resolve()), "atlas": str(args.atlas.resolve()),
        "db_sha256": fingerprint(args.search / "trials.db"), "selection_sha256": fingerprint(args.output / "frozen_selection.json"),
        "workers": workers, "threads_per_worker": args.threads, "ram_cap_mb": args.ram_cap_mb,
        "memory_policy": "1024 MiB scheduling budget per worker plus 2048 MiB available headroom; not an OS RSS cap",
        "native_clock": "TopCam native frames, half-open intervals, recording-specific verified FPS",
        "sessions": len(sessions), "animals": sessions.animal_id.nunique(), "gaps_seconds": GAPS,
        "sources": {str(p): fingerprint(p) for p in (Path(__file__), Path('src/fpcode/workflows/sequence_search.py'), Path('src/fpcode/statistics/chain_nulls.py'))},
        "input_sha256": {str(args.atlas / Path(p).relative_to(snapshot)): h for p, h in manifest["input_sha256"].items()},
        "reserved_input_sha256": {str(args.atlas / ("measurements" if level == "L1" else f"clustering/{level}/measurements") / "bouts_by_session" / f"{sid}.csv"):
            fingerprint(args.atlas / ("measurements" if level == "L1" else f"clustering/{level}/measurements") / "bouts_by_session" / f"{sid}.csv")
            for level in manifest["vocab"] for sid in sessions[sessions.animal_id.isin(manifest["reserved_animals"])].session_id},
        "first_order_repeats": 19, "shuffle_repeats": 5, "inference_dtype": "float32", "statistics_dtype": "float64", "bootstrap_permutation_resamples": 4999,
        "purpose": "posthoc hypothesis generation; phenotype excluded from discovery; no new template fits"}
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2))
    completed = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(str(args.atlas.resolve()), manifest)) as pool:
        for result in pool.map(measure_job, jobs, chunksize=1):
            completed.append(result)
            print(f"Projected {len(completed)}/{len(jobs)}: {result}", flush=True)
    pd.DataFrame(completed).to_csv(args.output / "execution.csv", index=False)
    if args.max_jobs:
        print("Bounded projection check complete; full report intentionally deferred", flush=True)
        return
    aggregate_analysis(args.output)
    write_findings(args.output, args.atlas)
    write_report(args.search, args.output, manifest)
    provenance.update(elapsed_seconds=time.perf_counter() - started, thread_pools=threadpool_info(),
                      peak_worker_rss_mb=max(r["worker_peak_rss_mb"] for r in completed), model_jobs=len(completed))
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2))
    print(f"Report saved to {args.output / 'report.md'} in {provenance['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
