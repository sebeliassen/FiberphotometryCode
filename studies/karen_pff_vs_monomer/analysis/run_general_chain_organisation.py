"""Execute Karen's frozen general-chain plan using verified CSV/NPZ products."""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

LEVELS = ("L1", "L2", "L3", "L4", "L3_unresolved_separate", "L4_unresolved_separate")
UNTREATED = ("3w_PI", "8w_PI", "16w_PI", "24w_PI")
PAIRS = (("3w_PI", "8w_PI"), ("8w_PI", "16w_PI"), ("16w_PI", "24w_PI"), ("3w_PI", "16w_PI"), ("3w_PI", "24w_PI"))
PRIMARY_LIBRARY = "L3_anchors"
PRIMARY_METRICS = ("self_excess", "mean_run_calls", "return2_excess")
GAPS = (0., .5, 1., 2., 5.)


def prepare_libraries(atlas, clustering, output):
    """Freeze all membership choices from annotations, before group inference."""
    import pandas as pd

    definitions, audits, sources = [], [], []
    libraries = output / "libraries"
    libraries.mkdir(parents=True, exist_ok=True)
    for level in LEVELS:
        level_root = clustering / level
        annotations = pd.read_csv(level_root / "cluster_annotations.csv").set_index("syllable_id")
        audit_path = level_root / "anchors/anchor_mapping_audit.csv"
        audit = pd.read_csv(audit_path)
        audit["level"] = level
        audits.append(audit)
        selected = audit[audit.retained].drop_duplicates("anchor_id")
        catalog = pd.DataFrame({"chain_id": selected.anchor_id, "tokens": selected.mapped_tokens,
                                "method": "anchors", "length_bouts": selected.mapped_length})
        sources.append((level + "_anchors", level, "anchors", catalog,
                        level_root / "anchors/anchor_occurrences.csv", audit_path))
        if level in LEVELS[:4]:
            root = atlas if level == "L1" else level_root
            all_catalog = pd.read_csv(root / "chains/chain_catalog.csv")
            for method in ("exact", "seqnmf"):
                sources.append((level + "_" + method, level, method, all_catalog[all_catalog.method == method].copy(),
                                root / "chains/chain_occurrences.csv", root / "chains/chain_catalog.csv"))
    for library, level, method, catalog, occurrences, catalog_input in sources:
        annotations = pd.read_csv(clustering / level / "cluster_annotations.csv").set_index("syllable_id")
        interpreted, names, members = [], [], []
        for row in catalog.itertuples():
            tokens = [int(t) for t in row.tokens.split()]
            interpreted.append(not bool(annotations.loc[tokens, "unresolved"].any()))
            names.append(" → ".join(annotations.loc[tokens, "description"]))
            members.append(" | ".join(annotations.loc[tokens, "raw_members"].astype(str)))
        catalog["interpretable"], catalog["description"], catalog["raw_members_by_phase"] = interpreted, names, members
        catalog["library"], catalog["level"] = library, level
        catalog.to_csv(libraries / (library + ".csv"), index=False)
        definitions.append({"library": library, "level": level, "method": method,
                            "catalog": str((libraries / (library + ".csv")).resolve()),
                            "catalog_input": str(catalog_input.resolve()), "occurrences": str(occurrences.resolve()),
                            "n_inclusive": len(catalog), "n_interpretable": int(catalog.interpretable.sum()),
                            "primary": library == PRIMARY_LIBRARY})
    definitions = pd.DataFrame(definitions)
    definitions.to_csv(output / "library_definitions.csv", index=False)
    pd.concat(audits).to_csv(output / "anchor_crosswalk.csv", index=False)
    policy = {"primary_library": PRIMARY_LIBRARY, "primary_scope": "interpretable", "primary_gap_seconds": 1.,
              "primary_metrics": PRIMARY_METRICS, "gaps_seconds": GAPS,
              "library_selection": "frozen training motifs; all mapped >=3-bout anchors, deduplicated by sequence",
              "interpretable": "all catalog-token annotations resolved; seqNMF tokens are dominant phases, minor weights may remain unresolved",
              "overlap_priority": "score descending, lexical chain ID, start/end ascending, original row; inclusive before filtering",
              "frequency_blocks": "original contiguous modelled native frames, split at excluded calls; long unmatched gaps do not split frequency pools",
              "quarter_null": "same reference blocks subdivided by recording-time quarters; cross-quarter opportunities omitted",
              "simulations": "1999 configurable; frequency intervals at g1 all libraries/scopes; first-order only L3 anchors g1",
              "first_order": "session MLE from eligible g1 transitions; unobserved rows use session frequency; disconnected observed starts fixed; fit uncertainty not integrated",
              "run_censoring": "left/right gaps, exclusions, missing or recording boundaries censor; only eligible different-chain successors are exits",
              "pooling": "within mouse/wave summed opportunities, counts, exposure and run sums; across waves equal wave weights per mouse",
              "independent_unit": "mouse", "group_order": ["Monomer", "PFF"],
              "drug_interpretation": "earlier weeks versus separate L_dopa wave; age, wave and drug confounded",
              "duration_null": "duration outcomes descriptive; categorical ID shuffles do not preserve duration associations"}
    (output / "policy.json").write_text(json.dumps(policy, indent=2))
    return definitions


def _measure_job(job):
    import resource

    import pandas as pd
    from threadpoolctl import threadpool_info

    from fpcode.workflows.chain_organisation import measure_library_session

    library, identity, events, catalog, blocks, fps, end, output, seed, replicates, memory_mb = job
    started = time.perf_counter()
    results = measure_library_session(events, catalog, blocks, fps=fps, recording_end_frame=end,
                                     output=output, gaps=GAPS, primary_gap=1., seed=seed, replicates=replicates,
                                     memory_mb=memory_mb, detailed_sensitivity=library == PRIMARY_LIBRARY)
    audit = results.pop("selection_audit")
    for table in results.values():
        if isinstance(table, pd.DataFrame) and len(table):
            for key, value in {"library": library, **identity}.items():
                table[key] = value
    results["selection_audit"] = pd.DataFrame([{**identity, "library": library, **audit,
                                               "elapsed_seconds": time.perf_counter() - started,
                                               "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                               "thread_pools": json.dumps([{key: p[key] for key in ("internal_api", "num_threads")}
                                                                            for p in threadpool_info()])}])
    # Compare the old bout-based resolver without changing its historical policy.
    if library == PRIMARY_LIBRARY:
        from fpcode.workflows.sequence_atlas import _exclusive_events
        old = _exclusive_events(events)
        new = pd.read_csv(Path(output) / "call_decisions.csv.gz")
        old_ids = set(zip(old.chain_id, old.start_frame, old.end_frame_exclusive))
        new = new[new.selected]
        new_ids = set(zip(new.chain_id, new.start_frame, new.end_frame_exclusive))
        results["selection_audit"]["old_resolver_selected"] = len(old_ids)
        results["selection_audit"]["old_new_symmetric_difference"] = len(old_ids ^ new_ids)
    for name, table in results.items():
        if len(table.columns):
            table.to_csv(Path(output) / ("summary_" + name + ".csv"), index=False)
    return results


def measure(args, definitions, sessions):
    import numpy as np
    import pandas as pd

    from fpcode.behavior.sequences.organisation import valid_frame_intervals

    blocks = {}
    for meta in sessions.itertuples():
        with np.load(args.atlas / "frame_cache" / (meta.session_id + ".npz"), allow_pickle=False) as cache:
            blocks[meta.session_id] = valid_frame_intervals(cache["frames"], cache["labels"])
    jobs = []
    for number, definition in enumerate(definitions.itertuples()):
        catalog = pd.read_csv(definition.catalog)
        all_events = pd.read_csv(definition.occurrences, dtype={"animal_id": str})
        if definition.method == "anchors":
            all_events = all_events.rename(columns={"anchor_id": "chain_id"})
            scores = catalog.set_index("chain_id").length_bouts
            all_events["selection_score"] = all_events.chain_id.map(scores)
        else:
            all_events = all_events[all_events.method == definition.method]
        grouped = dict(tuple(all_events.groupby("session_id", sort=False)))
        for position, meta in enumerate(sessions.itertuples()):
            events = grouped.get(meta.session_id, all_events.iloc[:0]).copy()
            # Original resolver comparison needs bout segments; native block starts
            # give an equivalent segment partition without re-reading bout files.
            events["segment"] = np.searchsorted(blocks[meta.session_id][:, 0], events.start_frame, side="right") - 1
            events["start_s"] = events.start_frame / float(meta.fps)
            events["end_s"] = events.end_frame_exclusive / float(meta.fps)
            identity = {key: getattr(meta, key) for key in ("session_id", "animal_id", "group", "condition")}
            jobs.append((definition.library, identity, events, catalog, blocks[meta.session_id], float(meta.fps), int(meta.n_frames),
                         args.output / "audit" / definition.library / meta.session_id,
                         args.seed + number * 1000 + position, args.replicates, args.memory_mb))
    collected = {}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, result in enumerate(pool.map(_measure_job, jobs, chunksize=1), start=1):
            for name, table in result.items():
                if len(table):
                    collected.setdefault(name, []).append(table)
            if done % 25 == 0 or done == len(jobs):
                print(f"Measured {done}/{len(jobs)} library-sessions; {time.perf_counter() - started:.1f}s", flush=True)
    for name, pieces in collected.items():
        pd.concat(pieces, ignore_index=True).to_csv(args.output / (name + ".csv"), index=False)
    support = pd.read_csv(args.output / "session_metrics.csv")
    support = support[support.gap_seconds == 1.]
    support.groupby(["library", "scope"], as_index=False).agg(
        sessions=("session_id", "size"), animals=("animal_id", "nunique"),
        median_coverage=("coverage_fraction", "median"), min_coverage=("coverage_fraction", "min"),
        median_calls=("calls", "median"), total_pairs=("pair_opportunities", "sum"),
        total_return2_opportunities=("return2_opportunities", "sum")).to_csv(args.output / "library_support.csv", index=False)


def statistics(args):
    import numpy as np
    import pandas as pd

    from fpcode.behavior.sequences.organisation import metric_values
    from fpcode.statistics.behavior_comparisons import adjust_fdr, compare_feature_atlas_parallel, independent_contrast
    from fpcode.statistics.followup_planning import exploratory_effect_stability, sensitivity_scenarios
    from fpcode.workflows.chain_organisation import pool_organisation_measurements

    output = args.output
    table = pd.read_csv(output / "session_metrics.csv", dtype={"animal_id": str})
    keys = ["animal_id", "group", "condition", "library", "scope", "gap_seconds"]
    panel = pool_organisation_measurements(table, identity=keys)
    panel.to_csv(output / "animal_wave_counts.csv", index=False)
    value_names = list(metric_values({}))
    # Quarter-null values live in their own table; no all-gap alias with no data.
    value_names = [v for v in value_names if not v.endswith("quarter_null")]
    long = panel.melt(id_vars=keys, value_vars=value_names, var_name="metric", value_name="value")
    primary = ((long.library == PRIMARY_LIBRARY) & (long.scope == "interpretable")
               & (long.gap_seconds == 1.) & long.metric.isin(PRIMARY_METRICS))
    long["family"] = np.where(primary, "general_chain_primary", "general_chain_exploratory")
    long["feature"] = long.library + "/" + long.scope + "/g" + long.gap_seconds.map(lambda g: f"{g:g}") + "/" + long.metric
    long["session_id"] = long.animal_id + "/" + long.condition
    nulls = pd.read_csv(output / "null_summaries.csv", dtype={"animal_id": str})
    surrogate = nulls[(nulls.null_kind == "first_order") | (
        (nulls.null_kind == "frequency") & nulls.metric.isin(["mean_run_calls", "longest_run", "return3_probability", "return4_probability"]))].copy()
    surrogate["family"] = "general_chain_exploratory"
    surrogate["feature"] = (surrogate.library + "/" + surrogate.scope + "/g1/" + surrogate.null_kind
                            + "_" + surrogate.metric + "_excess_equal_recording")
    surrogate["value"] = surrogate.excess
    long = pd.concat([long, surrogate], ignore_index=True)
    result = compare_feature_atlas_parallel(long, workers=args.workers, memory_mb=args.memory_mb,
                                            groups=("Monomer", "PFF"), untreated=UNTREATED, drug_condition="L_dopa",
                                            baseline_condition="24w_PI", longitudinal_pairs=PAIRS, seed=args.seed, resamples=args.replicates)
    for name, data in result.items():
        data.to_csv(output / (name + ".csv"), index=False)
    candidates = exploratory_effect_stability(result["animal_wave_measurements"], result["group_contrasts"],
                                               groups=("Monomer", "PFF"), untreated=UNTREATED, consistency_waves=UNTREATED[:3])
    candidates.to_csv(output / "exploratory_candidates.csv", index=False)
    # Equal-recording averaging sensitivity for the three frozen endpoints.
    selected = table[(table.library == PRIMARY_LIBRARY) & (table.scope == "interpretable") & (table.gap_seconds == 1.)]
    equal = selected.groupby(["animal_id", "group", "condition"], as_index=False)[list(PRIMARY_METRICS)].mean()
    equal.to_csv(output / "primary_equal_session_values.csv", index=False)
    primary_panel = panel[(panel.library == PRIMARY_LIBRARY) & (panel.scope == "interpretable") & (panel.gap_seconds == 1.)]
    balanced_ids = primary_panel[primary_panel.condition.isin(UNTREATED[:3])].groupby("animal_id").condition.nunique()
    balanced_ids = set(balanced_ids[balanced_ids == 3].index)
    sensitivity = []
    for weighting, data in (("equal_session", equal), ("balanced_early_attendance", primary_panel[primary_panel.animal_id.isin(balanced_ids)])):
        for metric in PRIMARY_METRICS:
            local = data[data.condition.isin(UNTREATED[:3] if weighting.startswith("balanced") else UNTREATED)]
            animal = local.groupby(["animal_id", "group"])[metric].mean().reset_index()
            contrast = independent_contrast(animal.loc[animal.group == "Monomer", metric], animal.loc[animal.group == "PFF", metric],
                                            seed=args.seed, resamples=args.replicates)
            sensitivity.append({"policy": weighting, "metric": metric, **contrast})
    loo = pd.read_csv(output / "leave_chain_out.csv", dtype={"animal_id": str})
    loo = pool_organisation_measurements(loo, identity=["animal_id", "group", "condition", "library", "scope", "removed_chain"])
    loo.to_csv(output / "leave_chain_out_animal_wave.csv", index=False)
    for (library, scope, removed), sub in loo.groupby(["library", "scope", "removed_chain"]):
        for metric in (*PRIMARY_METRICS, "initiations_per_min", "calls_per_min", "longest_run"):
            animals = sub[sub.condition.isin(UNTREATED)].groupby(["animal_id", "group"])[metric].mean().reset_index()
            contrast = independent_contrast(animals.loc[animals.group == "Monomer", metric], animals.loc[animals.group == "PFF", metric],
                                            seed=args.seed, resamples=args.replicates)
            sensitivity.append({"policy": "leave_chain_out", "library": library, "scope": scope,
                                "removed_chain": removed, "metric": metric, **contrast})
    quarters = pd.read_csv(output / "time_quarters.csv", dtype={"animal_id": str, "quarter": str})
    quarter_panel = pool_organisation_measurements(quarters, identity=["animal_id", "group", "condition", "scope", "quarter", "library"])
    quarter_panel.to_csv(output / "time_quarters_animal_wave.csv", index=False)
    for scope, sub in quarter_panel[quarter_panel.quarter == "stratified_null"].groupby("scope"):
        for metric in PRIMARY_METRICS:
            animals = sub[sub.condition.isin(UNTREATED)].groupby(["animal_id", "group"])[metric].mean().reset_index()
            sensitivity.append({"policy": "quarter_stratified_frequency", "scope": scope, "metric": metric,
                                **independent_contrast(animals.loc[animals.group == "Monomer", metric], animals.loc[animals.group == "PFF", metric],
                                                       seed=args.seed, resamples=args.replicates)})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity["q_policy"] = sensitivity.groupby("policy").p_permutation.transform(adjust_fdr)
    sensitivity.to_csv(output / "sensitivity_contrasts.csv", index=False)
    # Surrogate tail areas are session diagnostics, not group inference. Export
    # conservative equal-recording animal/wave means of higher-order excesses.
    nulls.groupby(["animal_id", "group", "condition", "library", "scope", "null_kind", "metric"], as_index=False)[
        ["observed", "null_mean", "excess", "mc_mean_se"]].mean().to_csv(output / "null_animal_wave_diagnostics.csv", index=False)
    sensitivity_scenarios(alphas=(.05, .05 / 3)).to_csv(output / "future_cohort_scenarios.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--clustering", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=("prepare", "measure", "statistics", "all"), default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=256, help="per-worker surrogate-array budget; input tables are additional")
    parser.add_argument("--replicates", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--limit-sessions", type=int, help="explicit small profiling scope; use a separate output")
    args = parser.parse_args(argv)
    if min(args.workers, args.threads, args.replicates, args.memory_mb) < 1:
        parser.error("positive resource and replicate settings required")
    # Configure before loading numerical libraries; help has no numerical side effects.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    from threadpoolctl import threadpool_limits

    # Also limit pools preloaded by interpreter/site startup. Restore them when
    # invoked in an existing process; environment-only limits are insufficient.
    with threadpool_limits(limits=args.threads):
        execute(args)


def execute(args):
    """Run explicit stages after resource configuration; no import-time work."""
    import pandas as pd
    from threadpoolctl import threadpool_info

    from fpcode.common.run_record import RunRecord

    args.output.mkdir(parents=True, exist_ok=True)
    sessions = pd.read_csv(args.atlas / "sessions.csv", dtype={"animal_id": str})
    if args.limit_sessions is not None:
        sessions = sessions.iloc[:args.limit_sessions]
    sessions.to_csv(args.output / "sessions.csv", index=False)
    record = RunRecord("karen-general-chain-" + args.stage, args.output / "provenance" / args.stage,
                       config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    root = Path(__file__).resolve().parents[3]
    record.extra["source_sha256_at_start"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in [*list((root / "src/fpcode").rglob("*.py")), Path(__file__)]}
    record.extra["thread_pools"] = threadpool_info()
    record.extra["cpu_count"] = os.cpu_count()
    record.extra["ram_cap_note"] = "simulation budget per worker plus bounded input tables; no GPU backend needed for categorical count workload"
    record.add_input_file(args.atlas / "sessions.csv")
    record.add_input_file(args.atlas / "chains/animal_splits.json")
    for meta in sessions.itertuples():
        record.add_input_file(args.atlas / "frame_cache" / (meta.session_id + ".npz"))
    if args.stage in ("prepare", "all"):
        definitions = prepare_libraries(args.atlas, args.clustering, args.output)
    else:
        definitions = pd.read_csv(args.output / "library_definitions.csv")
    for definition in definitions.itertuples():
        for path in (definition.occurrences, definition.catalog_input, definition.catalog):
            record.add_input_file(path)
        record.add_input_file(args.clustering / definition.level / "cluster_annotations.csv")
    if args.stage in ("measure", "all"):
        measure(args, definitions, sessions)
    if args.stage in ("statistics", "all"):
        statistics(args)
    for path in args.output.glob("*.csv"):
        record.add_output_file(path)
    record.extra["thread_pools_at_finish"] = threadpool_info()
    record.finish()


if __name__ == "__main__":
    main()
