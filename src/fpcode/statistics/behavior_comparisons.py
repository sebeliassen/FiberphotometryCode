"""Mouse-level contrasts, explicit multiplicity and clustered panel regression.

Callers identify independent animals, groups, waves, feature families and
eligibility. No cohort assumptions or outcome-dependent feature selection.
"""

import numpy as np
import pandas as pd
from scipy import stats


def adjust_fdr(values):
    """Benjamini-Hochberg on finite tests; preserve untested entries as missing."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    where = np.flatnonzero(np.isfinite(values))
    if not len(where):
        return out
    order = np.argsort(values[where])
    p = values[where][order]
    q = np.minimum.accumulate((p * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out[where[order]] = np.minimum(q, 1)
    return out


def independent_contrast(first, second, *, seed=101, resamples=1999, min_animals=3):
    """Difference second-minus-first, with independent-animal uncertainty."""
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    result = {"n_first": len(a), "n_second": len(b),
              "mean_first": np.mean(a) if len(a) else np.nan, "mean_second": np.mean(b) if len(b) else np.nan,
              "difference": np.mean(b) - np.mean(a) if len(a) and len(b) else np.nan,
              "ci_low": np.nan, "ci_high": np.nan, "p_permutation": np.nan, "p_welch": np.nan,
              "hedges_g": np.nan, "status": "insufficient_animals"}
    if min(len(a), len(b)) < min_animals:
        return result
    pooled = np.r_[a, b]
    scale = max(np.max(abs(pooled)), 1.)
    if np.ptp(pooled) <= 1e-12 * scale:
        result.update(status="constant", p_permutation=1., p_welch=1., ci_low=0., ci_high=0.)
        return result
    rng = np.random.default_rng(seed)
    boot = (b[rng.integers(len(b), size=(resamples, len(b)))].mean(axis=1)
            - a[rng.integers(len(a), size=(resamples, len(a)))].mean(axis=1))
    result["ci_low"], result["ci_high"] = np.quantile(boot, [.025, .975])
    # Random ranks generate uniform assignments with fixed group sizes.
    permutation = np.argsort(rng.random((resamples, len(pooled))), axis=1)
    differences = (pooled[permutation[:, len(a):]].mean(axis=1)
                   - pooled[permutation[:, :len(a)]].mean(axis=1))
    result["p_permutation"] = (1 + np.sum(abs(differences) >= abs(result["difference"]) - 1e-12)) / (resamples + 1)
    if min(np.std(a), np.std(b)) > 1e-12 * scale:
        result["p_welch"] = float(stats.ttest_ind(b, a, equal_var=False).pvalue)
    variance = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2)
    if variance > 0:
        correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
        result["hedges_g"] = result["difference"] / np.sqrt(variance) * correction
    result["status"] = "tested"
    return result


def paired_contrast(changes, *, seed=101, resamples=1999, min_animals=3):
    """Sign-flip contrast for within-animal changes; never pair by row position."""
    changes = np.asarray(changes, dtype=float)
    changes = changes[np.isfinite(changes)]
    out = {"n_pairs": len(changes), "mean_change": changes.mean() if len(changes) else np.nan,
           "ci_low": np.nan, "ci_high": np.nan, "p_permutation": np.nan,
           "status": "insufficient_animals"}
    if len(changes) < min_animals:
        return out
    rng = np.random.default_rng(seed)
    boot = changes[rng.integers(len(changes), size=(resamples, len(changes)))].mean(axis=1)
    out["ci_low"], out["ci_high"] = np.quantile(boot, [.025, .975])
    if len(changes) <= 12:
        signs = 2 * ((np.arange(2 ** len(changes))[:, None] >> np.arange(len(changes))) & 1) - 1
        out["p_permutation"] = float(np.mean(abs((signs * changes).mean(axis=1)) >= abs(changes.mean()) - 1e-12))
    else:
        signs = rng.choice([-1, 1], size=(resamples, len(changes)))
        out["p_permutation"] = (1 + np.sum(abs((signs * changes).mean(axis=1)) >= abs(changes.mean()) - 1e-12)) / (resamples + 1)
    out["status"] = "tested"
    return out


def clustered_group_regression(panel, *, first_group, second_group):
    """Wave-adjusted group coefficient with CR1 mouse-clustered covariance.

    Expects one row per animal/wave, explicit value and identity columns. This
    additive model estimates a common group difference across included waves.
    Uses a t reference with independent-cluster-minus-one degrees of freedom.
    """
    panel = panel.dropna(subset=["value"]).copy()
    n_animals = panel.animal_id.nunique()
    out = {"n_animals": n_animals, "n_animal_waves": len(panel), "adjusted_difference": np.nan,
           "cluster_se": np.nan, "p_cluster": np.nan, "ci_low": np.nan, "ci_high": np.nan,
           "status": "insufficient_panel"}
    group_counts = panel.groupby("group").animal_id.nunique()
    if min(group_counts.get(first_group, 0), group_counts.get(second_group, 0)) < 3:
        return out
    waves = pd.get_dummies(panel.condition, drop_first=True, dtype=float).to_numpy()
    x = np.column_stack((np.ones(len(panel)), (panel.group == second_group).to_numpy(float), waves))
    y = panel.value.to_numpy(float)
    if np.linalg.matrix_rank(x) != x.shape[1] or len(y) <= x.shape[1]:
        return out
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros_like(bread)
    ids = panel.animal_id.to_numpy()
    for animal in np.unique(ids):
        select = ids == animal
        score = x[select].T @ residual[select]
        meat += np.outer(score, score)
    correction = n_animals / (n_animals - 1) * (len(y) - 1) / (len(y) - x.shape[1])
    covariance = correction * bread @ meat @ bread
    se = np.sqrt(max(0., covariance[1, 1]))
    critical = stats.t.ppf(.975, n_animals - 1)
    p = 2 * stats.t.sf(abs(beta[1] / se), n_animals - 1) if se > 1e-12 else (1. if abs(beta[1]) < 1e-12 else np.nan)
    out.update(adjusted_difference=beta[1], cluster_se=se, p_cluster=p,
               ci_low=beta[1] - critical * se, ci_high=beta[1] + critical * se, status="tested")
    return out


def compare_feature_atlas(long_table, *, groups, untreated, drug_condition, baseline_condition,
                          longitudinal_pairs=(), seed=101, resamples=1999):
    """Aggregate sessions within animal/wave before every inferential operation.

    Long table columns: session_id, animal_id, group, condition, family, feature,
    value. Missing execution measures remain missing; rate zeroes must be supplied
    explicitly. Pooled descriptive means use each animal's available waves;
    wave-adjusted panel estimates are exported separately.
    """
    panel = long_table.groupby(["animal_id", "group", "condition", "family", "feature"],
                              observed=True, as_index=False).value.mean()
    contrasts, changes, regressions = [], [], []
    first, second = groups
    for feature_index, ((family, feature), sub) in enumerate(panel.groupby(["family", "feature"], sort=True)):
        if feature_index and feature_index % 500 == 0:
            print(f"Compared {feature_index} features", flush=True)
        views = [(str(c), sub[sub.condition == c]) for c in sorted(sub.condition.unique())]
        views += [("untreated_pooled", sub[sub.condition.isin(untreated)]), ("all_waves_pooled", sub)]
        for view, selected in views:
            animals = selected.groupby(["animal_id", "group"]).value.mean().reset_index()
            result = independent_contrast(animals.loc[animals.group == first, "value"],
                                          animals.loc[animals.group == second, "value"],
                                          seed=seed, resamples=resamples)
            contrasts.append({"family": family, "feature": feature, "view": view, **result})
        for scope, selected in (("untreated_adjusted", sub[sub.condition.isin(untreated)]),
                                ("all_waves_adjusted", sub)):
            regressions.append({"family": family, "feature": feature, "view": scope,
                                **clustered_group_regression(selected, first_group=first, second_group=second)})
        pairs = [(baseline_condition, drug_condition), ("untreated_mean", drug_condition), *longitudinal_pairs]
        for baseline, target in pairs:
            pre = sub[sub.condition == baseline] if baseline != "untreated_mean" else sub[sub.condition.isin(untreated)]
            pre = pre.groupby(["animal_id", "group"]).value.mean().rename("pre")
            post = sub[sub.condition == target].set_index(["animal_id", "group"]).value.rename("post")
            paired = pd.concat([pre, post], axis=1).dropna().reset_index()
            paired["change"] = paired.post - paired.pre
            for group in groups:
                result = paired_contrast(paired.loc[paired.group == group, "change"], seed=seed, resamples=resamples)
                changes.append({"family": family, "feature": feature, "baseline": baseline, "target": target, "group": group, **result})
            result = independent_contrast(paired.loc[paired.group == first, "change"],
                                          paired.loc[paired.group == second, "change"], seed=seed, resamples=resamples)
            contrasts.append({"family": family, "feature": feature, "view": "change_" + baseline + "_to_" + target,
                              **result})
    outputs = {"animal_wave_measurements": panel, "group_contrasts": pd.DataFrame(contrasts),
               "paired_wave_changes": pd.DataFrame(changes), "wave_adjusted_contrasts": pd.DataFrame(regressions)}
    for name, table in outputs.items():
        if name == "animal_wave_measurements" or table.empty:
            continue
        pcol = "p_cluster" if name == "wave_adjusted_contrasts" else "p_permutation"
        scopes = ["family", "view"] if "view" in table else ["family", "baseline", "target", "group"]
        table["q_family"] = table.groupby(scopes, dropna=False)[pcol].transform(adjust_fdr)
        table["q_global"] = adjust_fdr(table[pcol])
    return outputs


def _compare_feature_chunk(job):
    table, settings = job
    return compare_feature_atlas(table, **settings)


def compare_feature_atlas_parallel(long_table, *, workers, chunk_features=128, memory_mb=256, **settings):
    """Bounded feature batches with identical mouse contrasts and joint FDR scopes.

    Thread limits must be set by the caller before numerical imports. The memory
    cap covers each serialized input batch (with a 3x allowance); result tables,
    worker interpreters and per-feature resampling buffers are additional.
    Individual feature seeds/settings are unchanged by batching. Recompute FDR
    after joining chunks; chunk-local correction is never the reported result.
    """
    from concurrent.futures import ProcessPoolExecutor

    if workers < 1 or chunk_features < 1 or memory_mb <= 0:
        raise ValueError("positive parallel resource settings required")
    if workers == 1:
        return compare_feature_atlas(long_table, **settings)
    columns = ["session_id", "animal_id", "group", "condition", "family", "feature", "value"]
    table = long_table[columns].copy()
    groups = table.groupby(["family", "feature"], sort=True).indices
    budget = memory_mb * 1024 ** 2
    bytes_per_row = max(1., table.memory_usage(deep=True).sum() / max(1, len(table))) * 3
    jobs, indices, n_rows = [], [], 0
    for rows in groups.values():
        if len(rows) * bytes_per_row > budget:
            raise MemoryError("one feature input exceeds the worker input budget")
        if indices and (len(indices) >= chunk_features or (n_rows + len(rows)) * bytes_per_row > budget):
            jobs.append((table.iloc[np.concatenate(indices)], settings))
            indices, n_rows = [], 0
        indices.append(rows)
        n_rows += len(rows)
    if indices:
        jobs.append((table.iloc[np.concatenate(indices)], settings))
    if not jobs:
        return compare_feature_atlas(table, **settings)
    pieces = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_compare_feature_chunk, jobs, chunksize=1):
            for name, sub in result.items():
                pieces.setdefault(name, []).append(sub)
    outputs = {name: pd.concat(parts, ignore_index=True) for name, parts in pieces.items()}
    for name, sub in outputs.items():
        if name == "animal_wave_measurements" or sub.empty:
            continue
        pcol = "p_cluster" if name == "wave_adjusted_contrasts" else "p_permutation"
        scopes = ["family", "view"] if "view" in sub else ["family", "baseline", "target", "group"]
        sub["q_family"] = sub.groupby(scopes, dropna=False)[pcol].transform(adjust_fdr)
        sub["q_global"] = adjust_fdr(sub[pcol])
    return outputs
