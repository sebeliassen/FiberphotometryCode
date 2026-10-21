"""Explicit state aggregation, retaining invalid labels and raw membership."""

import numpy as np
import pandas as pd


def validate_state_map(mapping):
    mapping = np.asarray(mapping)
    if mapping.ndim != 1 or not len(mapping) or not np.issubdtype(mapping.dtype, np.integer) or np.any(mapping < 0):
        raise ValueError("state map must be a nonempty nonnegative integer vector")
    if not np.array_equal(np.unique(mapping), np.arange(mapping.max() + 1)):
        raise ValueError("mapped IDs must be contiguous from zero")
    return mapping


def remap_states(labels, mapping):
    """Preserve invalid samples as -1; never map them to a real state."""
    mapping = validate_state_map(mapping)
    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels >= len(mapping)):
        raise ValueError("labels must be integer IDs within the source vocabulary")
    out = np.full(labels.shape, -1, dtype=int)
    valid = labels >= 0
    out[valid] = mapping[labels[valid]]
    return out


def member_composition(table, mapping, *, identity, state_column="syllable_id", usage_column="time_fraction"):
    """Conditional raw-state time share within each parent, undefined if absent."""
    mapping = validate_state_map(mapping)
    result = table[identity + [state_column, usage_column]].copy()
    result["cluster_id"] = remap_states(result[state_column].to_numpy(int), mapping)
    parent = result.groupby(identity + ["cluster_id"])[usage_column].transform("sum")
    result["cluster_time_fraction"] = parent
    result["within_cluster_fraction"] = result[usage_column] / parent.where(parent > 0)
    return result


def standardised_member_execution(table, mapping, reference_weights, *, identity, features,
                                  minimum_weight_coverage=.8):
    """Compare observed mixtures with fixed raw-member execution averages.

    Caller supplies outcome-independent reference weights per raw ID. These
    summaries use original raw-state execution definitions. They are not the
    execution statistics of newly merged bouts and not a causal decomposition.
    """
    composition = member_composition(table, mapping, identity=identity)
    source = composition.merge(table[identity + ["syllable_id"] + features], on=identity + ["syllable_id"], validate="one_to_one")
    weights = np.asarray(reference_weights, float)
    if weights.shape != np.asarray(mapping).shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("reference weights must be finite nonnegative source-state weights")
    source["reference_weight"] = weights[source.syllable_id]
    rows = []
    for keys, sub in source.groupby(identity + ["cluster_id"], sort=False):
        total = sub.reference_weight.sum()
        for feature in features:
            good = sub[feature].notna()
            covered = sub.loc[good, "reference_weight"].sum()
            fraction = covered / total if total > 0 else 0.
            reference = (np.average(sub.loc[good, feature], weights=sub.loc[good, "reference_weight"])
                         if covered > 0 and fraction >= minimum_weight_coverage else np.nan)
            observed_weights = sub.loc[good, "within_cluster_fraction"]
            observed = (np.average(sub.loc[good, feature], weights=observed_weights)
                        if observed_weights.sum() > 0 else np.nan)
            rows.append({**dict(zip(identity + ["cluster_id"], keys)), "measure": feature,
                         "fixed_mixture_raw_execution": reference,
                         "observed_mixture_raw_execution": observed, "reference_weight_coverage": fraction})
    return pd.DataFrame(rows)
