"""Exploratory effect stability and prospective sensitivity, independent of p cutoffs."""

import numpy as np
import pandas as pd
from scipy import stats


def two_group_power(effect, n_per_group, *, alpha=.05):
    """Prospective two-sided equal-variance t sensitivity for a specified true d.

    Independent animals, equal group sizes, normal outcomes, no attrition and a
    fixed endpoint are assumed. This does not estimate the probability that a
    pilot finding is true or calculate achieved power from its observed p value.
    """
    if n_per_group < 2 or not 0 < alpha < 1 or not np.isfinite(effect):
        raise ValueError("finite effect, n >= 2 and 0 < alpha < 1 required")
    df = 2 * n_per_group - 2
    critical = stats.t.ppf(1 - alpha / 2, df)
    nc = abs(effect) * np.sqrt(n_per_group / 2)
    return float(stats.nct.sf(critical, df, nc) + stats.nct.cdf(-critical, df, nc))


def sensitivity_scenarios(*, effects=(.3, .5, .8), animal_counts=(10, 15, 20, 30, 40, 60, 80, 100),
                          alphas=(.05, .005)):
    """Use externally specified effect sizes, rather than selected pilot estimates."""
    return pd.DataFrame([{"assumed_true_d": effect, "animals_per_group": n, "alpha": alpha,
                          "prospective_power": two_group_power(effect, n, alpha=alpha)}
                         for effect in effects for n in animal_counts for alpha in alphas])


def exploratory_effect_stability(panel, contrasts, *, groups, untreated, consistency_waves,
                                 min_animals=6, effect_threshold=.4):
    """Rank pooled untreated effects using magnitude, direction and mouse support.

    The rank is a transparent heuristic, not a probability or independent
    validation. Retain all eligible estimates, including nonsignificant ones.
    Direction agreement across repeated waves is not independent replication.
    """
    first, second = groups
    selected = contrasts[contrasts.view == "untreated_pooled"].copy()
    pieces = {(family, feature): sub for (family, feature), sub in
              panel[panel.condition.isin(untreated)].groupby(["family", "feature"], sort=False)}
    rows = []
    for row in selected.itertuples():
        if min(row.n_first, row.n_second) < min_animals or not np.isfinite(row.hedges_g):
            continue
        sub = pieces[(row.family, row.feature)].dropna(subset=["value"])
        animal = sub.groupby(["animal_id", "group"]).value.mean().reset_index()
        a = animal.loc[animal.group == first, "value"].to_numpy()
        b = animal.loc[animal.group == second, "value"].to_numpy()
        direction = np.sign(row.difference)
        loo = np.r_[b.mean() - (a.sum() - a) / (len(a) - 1),
                    (b.sum() - b) / (len(b) - 1) - a.mean()]
        supported, agrees = 0, 0
        wave_differences = {}
        for wave in consistency_waves:
            local = sub[sub.condition == wave]
            wa = local.loc[local.group == first, "value"].dropna()
            wb = local.loc[local.group == second, "value"].dropna()
            if min(len(wa), len(wb)) >= min_animals:
                delta = wb.mean() - wa.mean()
                wave_differences[wave] = float(delta)
                supported += 1
                agrees += int(np.sign(delta) == direction)
        median_delta = np.median(b) - np.median(a)
        direction_fraction = agrees / supported if supported else np.nan
        loo_fraction = float(np.mean(np.sign(loo) == direction))
        promising = (abs(row.hedges_g) >= effect_threshold and supported >= 2
                     and direction_fraction >= 2 / 3 and loo_fraction == 1)
        # Capping magnitude limits dominance by near-zero pooled variance.
        score = min(abs(row.hedges_g), 2) * (direction_fraction if supported else 0) * loo_fraction
        rows.append({**row._asdict(), "supported_consistency_waves": supported,
                     "agreeing_consistency_waves": agrees, "wave_direction_fraction": direction_fraction,
                     "leave_one_animal_out_direction_fraction": loo_fraction,
                     "loo_difference_min": float(loo.min()), "loo_difference_max": float(loo.max()),
                     "median_difference": median_delta, "median_direction_agrees": np.sign(median_delta) == direction,
                     "meets_exploratory_heuristic": promising, "priority_score": score,
                     **{"difference_" + wave: wave_differences.get(wave, np.nan) for wave in consistency_waves}})
    return pd.DataFrame(rows).sort_values(["meets_exploratory_heuristic", "priority_score"], ascending=False)


def pilot_effect_scenarios(candidates, *, max_features=30, shrinkages=(.5, 1.),
                           animal_counts=(10, 15, 20, 30, 40, 60, 80, 100), alphas=(.05, .005)):
    """Illustrate sensitivity to optimistic pilot d and an explicit 50% shrinkage.

    Neither assumption is a corrected or validated estimate of the true effect.
    Selected pilot magnitudes are usually optimistic; fixed-d scenarios should
    drive design until endpoint definition and quality are externally checked.
    """
    rows = []
    for row in candidates.head(max_features).itertuples():
        for shrinkage in shrinkages:
            d = abs(row.hedges_g) * shrinkage
            for alpha in alphas:
                required = next((n for n in range(2, 501) if two_group_power(d, n, alpha=alpha) >= .8), np.nan)
                for n in animal_counts:
                    rows.append({"family": row.family, "feature": row.feature, "pilot_hedges_g": row.hedges_g,
                                 "pilot_multiplier": shrinkage, "assumed_true_d": d, "alpha": alpha,
                                 "animals_per_group": n, "prospective_power": two_group_power(d, n, alpha=alpha),
                                 "illustrative_n_per_group_80pct_max500": required})
    return pd.DataFrame(rows)
