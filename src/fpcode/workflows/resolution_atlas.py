"""Resolution sensitivity features from explicit raw-state membership policies."""

from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.behavior.primitives.resolutions import member_composition, standardised_member_execution

IDENTITY = ["session_id", "animal_id", "group", "condition"]


def resolution_composition_features(raw_table, mapping, output_dir, *, training_animals, untreated,
                                    execution_features=("mean_speed_px_s", "duration_s", "latent_dispersion_units2")):
    """Save conditional membership and fixed-mixture sensitivity on training weights."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    composition = member_composition(raw_table, mapping, identity=IDENTITY)
    composition.to_csv(output_dir / "raw_member_composition.csv", index=False)
    training = raw_table[raw_table.animal_id.isin(training_animals) & raw_table.condition.isin(untreated)]
    animal = training.groupby(["animal_id", "syllable_id"]).time_fraction.mean().reset_index()
    weights = animal.groupby("syllable_id").time_fraction.mean().reindex(range(len(mapping)), fill_value=0).to_numpy()
    pd.DataFrame({"syllable_id": np.arange(len(mapping)), "training_reference_weight": weights}).to_csv(
        output_dir / "mixture_reference_weights.csv", index=False)
    execution = raw_table.copy()
    execution.loc[execution.bout_count < 10, list(execution_features)] = np.nan
    standardised = standardised_member_execution(execution, mapping, weights, identity=IDENTITY,
                                                 features=list(execution_features))
    standardised.to_csv(output_dir / "fixed_mixture_execution.csv", index=False)
    composition["feature"] = composition.cluster_id.astype(str) + "/raw=" + composition.syllable_id.astype(str)
    composition["family"], composition["value"] = "raw_member_composition", composition.within_cluster_fraction
    # A singleton's conditional share is identically one and carries no new test.
    size = np.bincount(mapping)
    composition = composition[size[composition.cluster_id] > 1]
    long = standardised.melt(id_vars=IDENTITY + ["cluster_id", "measure"],
                             value_vars=["fixed_mixture_raw_execution", "observed_mixture_raw_execution"],
                             var_name="family", value_name="value")
    long["feature"] = long.cluster_id.astype(str) + "/" + long.measure
    long = long[size[long.cluster_id] > 1]
    return pd.concat([composition[IDENTITY + ["family", "feature", "value"]],
                      long[IDENTITY + ["family", "feature", "value"]]], ignore_index=True)


def raw_merging_audit(raw_table, membership, *, training_animals, untreated):
    """Describe member heterogeneity using discovery mice, without selecting on group effects."""
    selected = raw_table[raw_table.animal_id.isin(training_animals) & raw_table.condition.isin(untreated)]
    raw = selected.groupby(["animal_id", "syllable_id"], as_index=False)[
        ["duration_s", "mean_speed_px_s", "time_fraction"]].mean()
    raw = raw.groupby("syllable_id", as_index=False)[["duration_s", "mean_speed_px_s", "time_fraction"]].mean()
    merged = membership.merge(raw, left_on="raw_syllable_id", right_on="syllable_id", validate="many_to_one")
    return merged.groupby(["level", "cluster_id", "cluster_name"], as_index=False).agg(
        raw_members=("raw_syllable_id", "size"), unresolved=("unresolved", "max"),
        member_mean_speed_min_px_s=("mean_speed_px_s", "min"), member_mean_speed_max_px_s=("mean_speed_px_s", "max"),
        member_mean_duration_min_s=("duration_s", "min"), member_mean_duration_max_s=("duration_s", "max"),
        training_time_fraction=("time_fraction", "sum"))
