"""Create comparable feature tables and statistics from explicit atlas products."""

from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.statistics.behavior_comparisons import compare_feature_atlas

IDENTITY = ["session_id", "animal_id", "group", "condition"]


def _long(table, family, *, key=None, measures=None):
    if measures is None:
        measures = list(table.select_dtypes(include="number").columns.difference(
            ["week", "fps", "n_frames", "verified_labels", "unmodelled_frames", "model_segments", "label_mismatches",
             "latent_dimensions", "syllable_id", "bout_count", "uncensored_bout_count"]))
    melted = table.melt(id_vars=IDENTITY + ([key] if key else []), value_vars=measures,
                        var_name="measure", value_name="value")
    prefix = melted[key].astype(str) + "/" if key else ""
    melted["feature"] = prefix + melted.measure
    melted["family"] = family
    return melted[IDENTITY + ["family", "feature", "value"]]


def build_feature_table(sessions, measurement_dir, *, annotations, chains_dir=None, transition_limit=150):
    """Keep absent occurrences as zero and undefined execution measures missing."""
    measurement_dir = Path(measurement_dir)
    pieces = []
    summary = pd.read_csv(measurement_dir / "session_metrics.csv", dtype={"animal_id": str})
    pieces.append(_long(summary, "session"))
    syllables = pd.read_csv(measurement_dir / "syllables.csv", dtype={"animal_id": str})
    pieces.append(_long(syllables, "syllable_usage", key="syllable_id",
                        measures=["time_fraction", "bouts_per_min"]))
    execution = syllables.copy()
    selected = ["duration_s", "duration_median_s", "duration_p90_s", "duration_cv", "mean_speed_px_s",
                "peak_speed_px_s", "speed_sd_px_s", "pause_fraction", "peak_latency_s", "peak_phase",
                "entry_speed_px_s", "exit_speed_px_s", "speed_change_px_s", "mean_abs_yaw_rad_s",
                "mean_signed_yaw_rad_s", "net_turn_rad", "displacement_px", "straightness",
                "latent_path_units", "latent_dispersion_units2", "mean_abs_acceleration_px_s2", "mean_abs_jerk_px_s3"]
    execution.loc[execution.bout_count < 10, selected] = np.nan
    pieces.append(_long(execution, "syllable_execution", key="syllable_id", measures=selected))
    category = syllables.merge(annotations[["syllable_id", "category"]], on="syllable_id", validate="many_to_one")
    category = category.groupby(IDENTITY + ["category"], as_index=False).agg(
        time_fraction=("time_fraction", "sum"), raw_bouts_per_min=("bouts_per_min", "sum"))
    category.to_csv(measurement_dir / "category_measurements.csv", index=False)
    pieces.append(_long(category, "named_category", key="category"))
    transitions = pd.read_csv(measurement_dir / "transitions.csv", dtype={"animal_id": str})
    ranked = transitions.groupby(["from_syllable", "to_syllable"]).agg(count=("count", "sum"),
                                                                         mice=("animal_id", "nunique"))
    chosen = ranked[ranked.mice >= 5].nlargest(transition_limit, "count").index
    dense = []
    exposure = summary.set_index("session_id").valid_duration_s
    for meta in sessions.itertuples():
        sub = transitions[transitions.session_id == meta.session_id]
        source_counts = sub.groupby("from_syllable")["count"].sum()
        counts = sub.set_index(["from_syllable", "to_syllable"])["count"]
        identity = {k: getattr(meta, k) for k in IDENTITY}
        for source, target in chosen:
            count = counts.get((source, target), 0)
            opportunities = source_counts.get(source, 0)
            dense.append({**identity, "transition": f"{source}>{target}", "count": count,
                          "probability": count / opportunities if opportunities else np.nan,
                          "rate_per_min": count / (exposure[meta.session_id] / 60)})
    dense = pd.DataFrame(dense)
    dense.to_csv(measurement_dir / "transition_screen.csv", index=False)
    pieces.append(_long(dense, "syllable_transitions", key="transition", measures=["probability", "rate_per_min"]))
    if chains_dir is not None:
        chains_dir = Path(chains_dir)
        chains = pd.read_csv(chains_dir / "chain_session_metrics.csv", dtype={"animal_id": str})
        pieces.append(_long(chains, "chain_usage", key="chain_id", measures=["occurrences_per_min", "coverage_fraction"]))
        selected = ["duration_s", "duration_cv", "mean_speed_px_s", "speed_cv", "speed_change_px_s", "mean_phase_duration_cv"]
        chains.loc[chains.occurrences < 10, selected] = np.nan
        pieces.append(_long(chains, "chain_execution", key="chain_id", measures=selected))
        grouping = pd.read_csv(chains_dir / "chain_grouping_sensitivity.csv", dtype={"animal_id": str})
        grouping["grouping_id"] = grouping.chain_id + "/gap=" + grouping.gap_threshold_s.astype(str)
        measures = ["repeat_fraction", "max_run_length", "mean_run_length", "initiations_per_min"]
        pieces.append(_long(grouping, "chain_grouping", key="grouping_id", measures=measures))
        events = pd.read_csv(chains_dir / "chain_transitions.csv", dtype={"animal_id": str})
        ranked_chains = events.groupby(["method", "from_chain", "to_chain"]).agg(
            count=("gap_s", "size"), mice=("animal_id", "nunique"))
        selected_chains = ranked_chains[ranked_chains.mice >= 5].nlargest(100, "count").index
        chain_dense = []
        for meta in sessions.itertuples():
            local = events[events.session_id == meta.session_id]
            counts = local.groupby(["method", "from_chain", "to_chain"]).size()
            sources = local.groupby(["method", "from_chain"]).size()
            identity = {k: getattr(meta, k) for k in IDENTITY}
            for method, source, target in selected_chains:
                count = counts.get((method, source, target), 0)
                opportunities = sources.get((method, source), 0)
                chain_dense.append({**identity, "transition": f"{method}/{source}>{target}",
                                    "rate_per_min": count / (exposure[meta.session_id] / 60),
                                    "probability": count / opportunities if opportunities >= 10 else np.nan,
                                    "source_opportunities": opportunities})
        chain_dense = pd.DataFrame(chain_dense)
        chain_dense.to_csv(chains_dir / "chain_transition_screen.csv", index=False)
        pieces.append(_long(chain_dense, "chain_transitions", key="transition", measures=["rate_per_min", "probability"]))
        prefix = pd.read_csv(chains_dir / "prefix_progression.csv", dtype={"animal_id": str})
        supported = prefix[~prefix.prefix_censored]
        progression = supported.groupby(IDENTITY + ["chain_id"], as_index=False).agg(
            prefix_opportunities=("exact_completion", "size"), exact_completion_fraction=("exact_completion", "mean"),
            phases_reached=("phases_reached", "mean"), edit_distance=("edit_distance", "mean"),
            insertions=("insertions", "mean"), omissions=("omissions", "mean"), substitutions=("substitutions", "mean"))
        measures = ["exact_completion_fraction", "phases_reached", "edit_distance", "insertions", "omissions", "substitutions"]
        progression.loc[progression.prefix_opportunities < 10, measures] = np.nan
        progression.to_csv(chains_dir / "prefix_session_metrics.csv", index=False)
        pieces.append(_long(progression, "chain_progression", key="chain_id", measures=measures))
    for path, family, key, measures in (
        (measurement_dir / "native_time_motif_usage.csv", "native_time_motifs", "native_factor", ["rate_per_min"]),
        (measurement_dir / "micro_motif_usage.csv", "frame_micro_motifs", "micro_factor", ["rate_per_syllable_second"]),
        (measurement_dir / "pose_syllable_measurements.csv", "frame_posture_paws", "pose_feature", ["mean", "sd"]),
        (measurement_dir / "variant_usage.csv", "syllable_variants", "variant_id", ["fraction", "rate_per_min"]),
        (measurement_dir / "history_prediction.csv", "behavioural_history", "history", ["cross_entropy_bits", "gain_vs_unigram_bits"]),
        (measurement_dir / "chain_context.csv", "syllable_chain_context", "context_feature", ["difference_inside_minus_outside"]),
    ):
        if path.exists():
            pieces.append(_long(pd.read_csv(path, dtype={"animal_id": str}), family, key=key, measures=measures))
    table = pd.concat(pieces, ignore_index=True)
    table["value"] = pd.to_numeric(table.value, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return table


def run_atlas_statistics(sessions, measurement_dir, output_dir, *, annotations, chains_dir,
                         groups, untreated, drug_condition, baseline_condition, longitudinal_pairs=(), seed=101):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_feature_table(sessions, measurement_dir, annotations=annotations, chains_dir=chains_dir)
    table.to_csv(output_dir / "session_features_long.csv", index=False)
    feature_count = table[["family", "feature"]].drop_duplicates().shape[0]
    print(f"Comparing {feature_count} features at the animal level", flush=True)
    results = compare_feature_atlas(table, groups=groups, untreated=untreated, drug_condition=drug_condition,
                                   baseline_condition=baseline_condition, longitudinal_pairs=longitudinal_pairs, seed=seed)
    for name, frame in results.items():
        frame.to_csv(output_dir / (name + ".csv"), index=False)
    return results
