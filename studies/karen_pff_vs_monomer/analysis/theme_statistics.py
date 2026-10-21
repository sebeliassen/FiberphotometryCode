"""Bounded mouse-level exploratory inference for the deadline theme atlas.

No endpoint is selected by its effect or p-value. Inputs already encode frozen
theme definitions, eligible opportunities, native clocks and cohort metadata.
Importing this module does not load empirical inputs or initialize accelerators.
"""

import hashlib
import itertools
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fpcode.statistics.behavior_comparisons import adjust_fdr

UNTREATED = ("3w_PI", "8w_PI", "16w_PI", "24w_PI")
WAVES = (*UNTREATED, "Ldopa")
KEYS = ["family", "theme", "feature"]
REQUIRED = ["session_id", "animal_id", "group", "wave", *KEYS, "value"]
COUNTS = ["support", "opportunities"]


class _Assignments:
    """Cache bounded float64 contrast weights, sharing draws across endpoints."""

    def __init__(self, replicates, seed, max_batch_bytes):
        self.replicates = int(replicates)
        self.seed = int(seed)
        self.budget = int(max_batch_bytes)
        if self.replicates < 1 or self.budget < 1024 * 1024:
            raise ValueError("positive replicates and at least 1 MiB array budget required")
        self.cache = OrderedDict()
        self.cache_bytes = 0

    def get(self, animals, n_second=None):
        animals = tuple(str(x) for x in animals)
        key = (animals, n_second)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        n = len(animals)
        total = 2 ** n if n_second is None else math.comb(n, n_second)
        exact = total <= max(self.replicates, 20000)
        draws = total if exact else self.replicates
        # Reserve headroom for sums, encoding temporaries and caller arrays.
        if draws * n * 8 > self.budget // 3:
            raise MemoryError("permutation weights exceed one third of configured array budget")
        digest = hashlib.sha256(repr((self.seed, key)).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        if n_second is None:
            if exact:
                weights = np.empty((draws, n), dtype=np.float64)
                for j in range(n):
                    weights[:, j] = 2 * ((np.arange(draws, dtype=np.uint64) >> j) & 1) - 1.
            else:
                weights = rng.integers(0, 2, (draws, n), dtype=np.int8).astype(np.float64) * 2 - 1
            weights /= n
        else:
            n_first = n - n_second
            weights = np.full((draws, n), -1. / n_first, dtype=np.float64)
            offset = 1. / n_second + 1. / n_first
            if exact:
                for row, indices in enumerate(itertools.combinations(range(n), n_second)):
                    weights[row, list(indices)] += offset
            else:
                # Independent uniformly sampled fixed-size mouse partitions.
                order = np.argsort(rng.random((draws, n)), axis=1)[:, :n_second]
                weights[np.arange(draws)[:, None], order] += offset
        result = weights, exact, total
        while self.cache and self.cache_bytes + weights.nbytes > self.budget // 2:
            _, previous = self.cache.popitem(last=False)
            self.cache_bytes -= previous[0].nbytes
        self.cache[key] = result
        self.cache_bytes += weights.nbytes
        return result


def _permutation_p(weights, values, observed, exact):
    null = weights @ np.asarray(values, dtype=np.float64)
    tolerance = 32 * np.finfo(float).eps * max(1., float(np.max(np.abs(values))))
    extreme = np.count_nonzero(np.abs(null) >= abs(observed) - tolerance)
    return float(extreme / len(null) if exact else (extreme + 1) / (len(null) + 1))


def _independent(panel, assignments):
    panel = panel[np.isfinite(panel.value)].sort_values("animal_id")
    a = panel.loc[panel.group == "Monomer", "value"].to_numpy(float)
    b = panel.loc[panel.group == "PFF", "value"].to_numpy(float)
    out = {"n_monomer": len(a), "n_pff": len(b),
           "mean_monomer": float(np.mean(a)) if len(a) else np.nan,
           "mean_pff": float(np.mean(b)) if len(b) else np.nan,
           "difference": float(np.mean(b) - np.mean(a)) if len(a) and len(b) else np.nan,
           "ci_low": np.nan, "ci_high": np.nan, "hedges_g": np.nan,
           "p_raw": np.nan, "p_welch": np.nan, "permutation_exact": False,
           "permutation_draws": 0, "permutation_partitions": 0, "status": "insufficient_animals"}
    for column in [*COUNTS, *[f"{name}_{stage}" for name in COUNTS for stage in ["baseline", "followup"]]]:
        if column in panel:
            for group in ["Monomer", "PFF"]:
                out[f"{column}_{group.lower()}"] = panel.loc[panel.group == group, column].sum(min_count=1)
    if min(len(a), len(b)) < 3:
        return out
    weights, exact, total = assignments.get(panel.animal_id, len(b))
    out.update(p_raw=_permutation_p(weights, panel.value.to_numpy(float), out["difference"], exact),
               permutation_exact=exact, permutation_draws=len(weights), permutation_partitions=total,
               status="tested")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sa, sb = va / len(a), vb / len(b)
    se2 = sa + sb
    if se2 > 0:
        degrees = se2 ** 2 / (sa ** 2 / (len(a) - 1) + sb ** 2 / (len(b) - 1))
        margin = stats.t.ppf(.975, degrees) * np.sqrt(se2)
        out.update(ci_low=out["difference"] - margin, ci_high=out["difference"] + margin,
                   p_welch=float(2 * stats.t.sf(abs(out["difference"]) / np.sqrt(se2), degrees)))
    else:
        out.update(ci_low=out["difference"], ci_high=out["difference"],
                   p_welch=1. if out["difference"] == 0 else 0.)
    variance = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    if variance > 0:
        out["hedges_g"] = (1 - 3 / (4 * (len(a) + len(b)) - 9)) * out["difference"] / np.sqrt(variance)
    if np.ptp(np.r_[a, b]) == 0:
        out["status"] = "constant"
    return out


def _paired(panel, assignments):
    panel = panel[np.isfinite(panel.value)].sort_values("animal_id")
    changes = panel.value.to_numpy(float)
    n = len(changes)
    out = {"n_pairs": n, "mean_change": float(np.mean(changes)) if n else np.nan,
           "ci_low": np.nan, "ci_high": np.nan, "paired_dz": np.nan,
           "p_raw": np.nan, "p_paired_t": np.nan, "permutation_exact": False,
           "permutation_draws": 0, "permutation_partitions": 0, "status": "insufficient_animals"}
    for name in COUNTS:
        for stage in ["baseline", "followup"]:
            column = f"{name}_{stage}"
            if column in panel:
                out[column] = panel[column].sum(min_count=1)
    if n < 3:
        return out
    weights, exact, total = assignments.get(panel.animal_id)
    out.update(p_raw=_permutation_p(weights, changes, out["mean_change"], exact),
               permutation_exact=exact, permutation_draws=len(weights), permutation_partitions=total,
               status="tested")
    sd = np.std(changes, ddof=1)
    margin = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n)
    out.update(ci_low=out["mean_change"] - margin, ci_high=out["mean_change"] + margin)
    if sd > 0:
        out.update(paired_dz=out["mean_change"] / sd,
                   p_paired_t=float(2 * stats.t.sf(abs(out["mean_change"]) / (sd / np.sqrt(n)), n - 1)))
    else:
        out["p_paired_t"] = 1. if out["mean_change"] == 0 else 0.
    if np.ptp(changes) == 0:
        out["status"] = "constant_change"
    return out


def _validate(readouts):
    missing = sorted(set(REQUIRED) - set(readouts.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    table = readouts.copy()
    if table.empty:
        raise ValueError("no readouts supplied")
    if table[REQUIRED[:-1]].isna().any().any():
        raise ValueError("identity, endpoint and cohort fields cannot be missing")
    for column in REQUIRED[:-1]:
        table[column] = table[column].astype(str)
    if not set(table.group).issubset({"Monomer", "PFF"}) or not set(table.wave).issubset(WAVES):
        raise ValueError("groups must be Monomer/PFF and waves must use the fixed Karen wave names")
    if table.groupby("animal_id").group.nunique().max() != 1:
        raise ValueError("an animal cannot change experimental group")
    if (table.groupby("session_id")[["animal_id", "group", "wave"]].nunique() > 1).any().any():
        raise ValueError("one recording must identify one animal, group and wave")
    if table.duplicated(["session_id", *KEYS]).any():
        raise ValueError("one endpoint may occur only once per recording")
    table["value"] = pd.to_numeric(table.value, errors="raise")
    if np.isinf(table.value).any():
        raise ValueError("infinite endpoint values are invalid; unavailable conditional measures use NaN")
    if "aggregation" not in table:
        table["aggregation"] = "mean"
    table["aggregation"] = table.aggregation.fillna("mean")
    if not set(table.aggregation).issubset({"mean", "ratio"}):
        raise ValueError("aggregation must be mean or ratio")
    if "value_scale" not in table:
        table["value_scale"] = 1.
    table["value_scale"] = table.value_scale.fillna(1.)
    if not np.isfinite(table.value_scale).all() or (table.value_scale <= 0).any():
        raise ValueError("value_scale must be finite and positive")
    if (table.groupby(KEYS)[["aggregation", "value_scale"]].nunique() > 1).any().any():
        raise ValueError("aggregation and value_scale must be constant within each endpoint")
    if "units" in table and (table.groupby(KEYS).units.nunique() > 1).any():
        raise ValueError("units must be constant within each endpoint")
    ratio = table.aggregation == "ratio"
    if ratio.any():
        if not {"numerator", "denominator"}.issubset(table):
            raise ValueError("ratio aggregation requires numerator and denominator")
        for column in ["numerator", "denominator"]:
            table[column] = pd.to_numeric(table[column], errors="raise")
            if np.isinf(table.loc[ratio, column]).any():
                raise ValueError("ratio inputs cannot be infinite")
        if (table.loc[ratio, "denominator"] < 0).any():
            raise ValueError("ratio denominators must be nonnegative")
        calculated = table.numerator / table.denominator.where(table.denominator > 0) * table.value_scale
        eligible = ratio & calculated.notna() & table.value.notna()
        if not np.allclose(table.loc[eligible, "value"], calculated[eligible], rtol=1e-8, atol=1e-12):
            raise ValueError("session values disagree with numerator/denominator times value_scale")
        if (ratio & table.value.notna() & calculated.isna()).any():
            raise ValueError("zero or unavailable denominator must have a missing value")
    return table


def _mouse_wave(table):
    rows = []
    for key, sub in table.groupby(["animal_id", "group", "wave", *KEYS], sort=True):
        row = dict(zip(["animal_id", "group", "wave", *KEYS], key))
        row.update(n_recordings=len(sub), n_finite_recordings=int(sub.value.notna().sum()),
                   session_ids="|".join(sorted(sub.session_id)), aggregation=sub.aggregation.iloc[0],
                   value_scale=float(sub.value_scale.iloc[0]))
        finite = sub.value.notna()
        if row["aggregation"] == "ratio":
            complete = sub.numerator.notna() & sub.denominator.notna() & finite
            numerator = sub.loc[complete, "numerator"].sum(min_count=1)
            denominator = sub.loc[complete, "denominator"].sum(min_count=1)
            row.update(numerator=numerator, denominator=denominator,
                       value=numerator / denominator * row["value_scale"] if denominator > 0 else np.nan)
        else:
            row["value"] = sub.value.mean()
        for column in ["support", "opportunities"]:
            if column in sub:
                row[column] = sub[column].sum(min_count=1)
        if "units" in sub:
            if sub.units.dropna().nunique() > 1:
                raise ValueError("units cannot vary within a mouse-wave endpoint")
            row["units"] = next(iter(sub.units.dropna()), "")
        rows.append(row)
    return pd.DataFrame(rows)


def _pooled(sub, scope, complete=False):
    rows = []
    for key, mouse in sub.groupby(["animal_id", "group"], sort=True):
        finite = mouse[np.isfinite(mouse.value)]
        value = finite.value.mean()
        if complete and set(finite.wave) != set(UNTREATED):
            value = np.nan
        row = {"animal_id": key[0], "group": key[1], "value": value, "scope": scope,
               "waves_present": "|".join(wave for wave in WAVES if wave in set(finite.wave)),
               "n_waves": len(finite), "n_recordings": int(mouse.n_recordings.sum())}
        for column in ["support", "opportunities"]:
            if column in mouse:
                row[column] = finite[column].sum(min_count=1)
        rows.append(row)
    return pd.DataFrame(rows, columns=["animal_id", "group", "value", "scope", "waves_present",
                                       "n_waves", "n_recordings", *[column for column in
                                                                  ["support", "opportunities"] if column in sub]])


def _descriptions(panel, keys):
    rows = []
    for key, sub in panel.groupby(keys, dropna=False, sort=True):
        values = sub.value.to_numpy(float)
        values = values[np.isfinite(values)]
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update(n_animals=len(values), n_animals_total=sub.animal_id.nunique(),
                   mean=float(np.mean(values)) if len(values) else np.nan,
                   median=float(np.median(values)) if len(values) else np.nan,
                   sd=float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                   minimum=float(np.min(values)) if len(values) else np.nan,
                   maximum=float(np.max(values)) if len(values) else np.nan)
        for column in ["support", "opportunities"]:
            if column in sub:
                row[column] = sub[column].sum(min_count=1)
        rows.append(row)
    return pd.DataFrame(rows)


def _correct(tables):
    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined["q_global"] = adjust_fdr(combined.p_raw)
    combined["q_family"] = combined.groupby("correction_family", dropna=False).p_raw.transform(adjust_fdr)
    combined["family_test_count"] = combined.groupby("correction_family").p_raw.transform("count")
    combined["global_test_count"] = int(combined.p_raw.notna().sum())
    combined["nominal_005"] = combined.p_raw.lt(.05) & combined.p_raw.notna()
    combined["family_fdr_005"] = combined.q_family.lt(.05) & combined.q_family.notna()
    combined["global_fdr_005"] = combined.q_global.lt(.05) & combined.q_global.notna()
    return combined


def infer(readouts, output, *, replicates=9999, seed=20261021, max_batch_bytes=512 * 1024 ** 2):
    """Write all effects, mouse panels and prespecified-scope multiplicity tables.

    The independent unit is animal_id. Replicated recording ratios are aggregated
    with summed numerators/denominators; means use equal recording weights within
    mouse-wave. Untreated pooled effects give each available wave equal weight
    within a mouse, with a separate complete-four-wave sensitivity. All-wave
    pooled values are descriptive only. Missing conditional measures stay missing.

    Group differences are PFF minus Monomer. Longitudinal changes are follow-up
    minus 3w_PI, matched explicitly by mouse and reported separately per group;
    differences in changes also receive mouse-label permutation tests. Two-sided
    randomization p-values are primary; Welch/paired-t CIs and auxiliary p-values
    are named separately. Ldopa is age/treatment confounded, never a causal effect.
    """
    started = time.monotonic()
    table = _validate(readouts)
    assignments = _Assignments(replicates, seed, max_batch_bytes)
    panel = _mouse_wave(table)
    pooled_rows, contrast_rows, paired_rows, change_panels = [], [], [], []
    for endpoint, sub in panel.groupby(KEYS, sort=True):
        identity = dict(zip(KEYS, endpoint))
        views = [(wave, sub[sub.wave == wave]) for wave in WAVES]
        untreated = sub[sub.wave.isin(UNTREATED)]
        available = _pooled(untreated, "pooled_untreated")
        complete = _pooled(untreated, "pooled_untreated_complete", complete=True)
        allwaves = _pooled(sub, "all_waves_descriptive")
        for pooled in [available, complete, allwaves]:
            for key, value in identity.items():
                pooled[key] = value
            pooled_rows.append(pooled)
        views.extend([(available.scope.iloc[0], available)] if len(available) else [])
        views.extend([(complete.scope.iloc[0], complete)] if len(complete) else [])
        for scope, selected in views:
            result = _independent(selected, assignments)
            contrast_rows.append({**identity, "scope": scope, "contrast_type": "group_difference",
                                  "correction_family": f"group_difference|{identity['family']}|{scope}",
                                  "ldopa_age_treatment_confounded": scope == "Ldopa", **result})
        for target in WAVES[1:]:
            columns = ["value", *[column for column in COUNTS if column in sub]]
            baseline = sub[sub.wave == "3w_PI"].set_index(["animal_id", "group"])[columns]
            followup = sub[sub.wave == target].set_index(["animal_id", "group"])[columns]
            baseline = baseline.rename(columns={column: "baseline" if column == "value" else
                                                f"{column}_baseline" for column in columns})
            followup = followup.rename(columns={column: "followup" if column == "value" else
                                                f"{column}_followup" for column in columns})
            changed = pd.concat([baseline, followup], axis=1).dropna(subset=["baseline", "followup"]).reset_index()
            changed["value"] = changed.followup - changed.baseline
            changed["baseline_wave"], changed["target_wave"] = "3w_PI", target
            for key, value in identity.items():
                changed[key] = value
            change_panels.append(changed)
            for group in ["Monomer", "PFF"]:
                result = _paired(changed[changed.group == group], assignments)
                paired_rows.append({**identity, "scope": f"3w_PI_to_{target}", "group": group,
                                    "baseline_wave": "3w_PI", "target_wave": target,
                                    "contrast_type": "paired_change",
                                    "correction_family": f"paired_change|{identity['family']}|{target}|{group}",
                                    "ldopa_age_treatment_confounded": target == "Ldopa", **result})
            result = _independent(changed, assignments)
            contrast_rows.append({**identity, "scope": f"3w_PI_to_{target}",
                                  "contrast_type": "difference_in_changes",
                                  "correction_family": f"difference_in_changes|{identity['family']}|{target}",
                                  "ldopa_age_treatment_confounded": target == "Ldopa", **result})
    contrasts, paired = pd.DataFrame(contrast_rows), pd.DataFrame(paired_rows)
    corrected = _correct([contrasts, paired])
    contrasts = corrected[corrected.contrast_type != "paired_change"].reset_index(drop=True)
    paired = corrected[corrected.contrast_type == "paired_change"].reset_index(drop=True)
    pooled = pd.concat([frame for frame in pooled_rows if not frame.empty], ignore_index=True, sort=False)
    descriptions = pd.concat([
        _descriptions(panel.rename(columns={"wave": "scope"}), [*KEYS, "scope", "group"]),
        _descriptions(pooled, [*KEYS, "scope", "group"])], ignore_index=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = {"session_readouts.csv.gz": table, "mouse_readouts.csv": panel,
              "pooled_mouse_readouts.csv": pooled, "mouse_descriptives.csv": descriptions,
              "contrasts.csv": contrasts, "longitudinal.csv": paired,
              "paired_mouse_changes.csv": pd.concat(change_panels, ignore_index=True, sort=False),
              "all_inference.csv": corrected}
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    policy = {"independent_unit": "animal_id", "groups": ["Monomer", "PFF"], "waves": list(WAVES),
              "endpoint_count": int(table.groupby(KEYS).ngroups), "input_rows": len(table),
              "n_animals": table.animal_id.nunique(), "n_recordings": table.session_id.nunique(),
              "replicates": replicates, "seed": seed, "max_batch_bytes": max_batch_bytes,
              "dtype": "float64", "workers": 1,
              "thread_environment": {key: os.environ.get(key) for key in
                                     ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
              "assignment_cache_peak_limit_bytes": max_batch_bytes // 2,
              "permutation": "two-sided absolute mean difference; fixed finite-mouse roster and group sizes; "
                             "cached assignments shared for identical rosters; exact when partitions <= "
                             "max(replicates,20000), otherwise independent uniform partitions with plus-one p",
              "paired": "two-sided mouse sign flips; same exact/Monte Carlo threshold and plus-one policy",
              "ci": "95% Welch t for independent differences; Student t for paired mouse differences; "
                    "not multiplicity-adjusted or permutation confidence intervals",
              "ratio_aggregation": "sum complete numerators / sum complete denominators * value_scale "
                                   "within mouse-wave; zero denominator stays missing",
              "mean_aggregation": "equal recording weights within mouse-wave",
              "pooled_untreated": "equal available finite untreated wave values within each mouse; "
                                  "incomplete-wave mice included with waves_present exported",
              "pooled_untreated_complete": "same four untreated waves required finite, sensitivity only",
              "all_waves_descriptive": "equal available wave values within mouse, no pooled p-value",
              "support_and_opportunities": "exported totals describe contributing records; they never "
                                           "replace animal counts or weight group means/uncertainty",
              "correction": "Benjamini-Hochberg on primary finite permutation p-values: family keys "
                            "contrast_type × feature-family × scope (paired changes also group); all "
                            "themes/endpoints included. Global BH includes every finite primary group, "
                            "change-interaction and paired test across all scopes; missing tests excluded.",
              "interpretation": "exploratory descriptive associations in generator training cohort; "
                                "permutation exchangeability and paired symmetry are assumptions; "
                                "sparse conditional missingness may differ by group; Ldopa confounds age/treatment; "
                                "no held-out or causal claims",
              "elapsed_seconds": time.monotonic() - started}
    (output / "inference_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    return {"output": str(output.resolve()), "paths": {name: str((output / name).resolve()) for name in frames},
            "policy": policy, "n_tests": int(corrected.p_raw.notna().sum())}
