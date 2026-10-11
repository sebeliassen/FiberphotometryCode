#!/usr/bin/env python3
"""
run_karen_longitudinal_lmms.py

Section E: Longitudinal Linear Mixed-Effects Models (LMMs) for Karen Dataset
Implements repeated-measures hierarchical modeling:
  Metric ~ C(genotype, Treatment('Monomer')) * C(condition, Treatment('3w_PI')) + (1 | animal_id)

Outputs:
  - Full LMM ANOVA and regression coefficient tables
  - Model-derived week-specific PFF vs. Monomer contrasts with FDR adjustment
  - Publication-ready 6-panel master longitudinal visualization
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings
from pathlib import Path

from fpcode.common.paths import get_repo_root, resolve_data_path

warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

REPO_ROOT = get_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

DEFAULT_DATA_PKL = resolve_data_path('data/sessions') / "sessions_karen_project_w_sylls.pkl"
DEFAULT_OUT_DIR = resolve_data_path('data/analysis_cache/ulrik_kinematics_output') / "karen_genotype_comparison" / "5tier_phenotyping" / "section_e_longitudinal_lmms"

DATA_PKL = DEFAULT_DATA_PKL
OUT_DIR = DEFAULT_OUT_DIR

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
palette = {"Monomer": "#2b5c8f", "PFF": "#d95f02"}
CONDITION_ORDER = ["3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa"]
CONDITION_LABELS = [
    "3w PI\n(Baseline)",
    "8w PI\n(Mid)",
    "16w PI\n(Peak)",
    "24w PI\n(Late)",
    "L-Dopa\n(+Benserazide)"
]
TIME_WEEKS_MAP = {"3w_PI": 3.0, "8w_PI": 8.0, "16w_PI": 16.0, "24w_PI": 24.0}


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])
    order = np.argsort(p_values)
    ranked_p = p_values[order]
    q_vals = np.zeros(n)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        q = ranked_p[i] * n / (i + 1)
        running_min = min(running_min, q)
        q_vals[order[i]] = running_min
    return np.clip(q_vals, 0.0, 1.0)


def get_p_star(p_val):
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    elif p_val < 0.10:
        return "#"
    else:
        return "ns"


def load_integrated_dataset(fps=80.0):
    print("[LMM Loader] Loading and computing all session metrics...")
    with open(DATA_PKL, "rb") as f:
        sessions = pickle.load(f)

    records = []
    for s in sessions:
        md = getattr(s, "metadata", {})
        g = md.get("genotype", "Unknown")
        cond = md.get("condition", "Unknown")
        aid = str(md.get("animal_id", s.session_id))

        if g not in ["PFF", "Monomer"] or cond not in CONDITION_ORDER:
            continue

        df = s.all_data_df.dropna(subset=["velocity_px_s", "syllable"])
        if len(df) < 1000:
            continue

        speed = df["velocity_px_s"].values
        sylls = df["syllable"].values.astype(int)

        # 1. Global Kinematics
        mean_speed = np.mean(speed)
        p95_speed = np.percentile(speed, 95)
        immobility_pct = np.mean(speed < 5.0) * 100.0

        accel = np.diff(speed) * fps
        jerk = np.diff(accel) * fps
        norm_jerk = np.mean(np.abs(jerk)) / (mean_speed + 1e-3)

        # 2. Bout Dynamics
        is_moving = speed >= 5.0
        moving_diff = np.diff(np.concatenate(([0], is_moving.astype(int), [0])))
        starts = np.where(moving_diff == 1)[0]
        ends = np.where(moving_diff == -1)[0]
        bout_lengths = (ends - starts) / fps
        macro_bout_mean = np.mean(bout_lengths) if len(bout_lengths) > 0 else 0.0

        # 3. Micro Dwell
        syll_change = np.diff(np.concatenate(([-1], sylls, [-1])))
        change_idx = np.where(syll_change != 0)[0]
        dwell_lengths = np.diff(change_idx) / fps
        micro_dwell_mean = np.mean(dwell_lengths) if len(dwell_lengths) > 0 else 0.0

        # 4. Repertoire Entropy
        counts = pd.Series(sylls).value_counts(normalize=True).values
        rep_entropy = -np.sum(counts * np.log2(counts))
        active_sylls = len(counts[counts >= 0.001])

        # 5. Syntax Entropy
        changes = np.where(np.diff(sylls) != 0)[0]
        state_seq = sylls[np.concatenate(([0], changes + 1))]
        if len(state_seq) >= 10:
            max_s = 100
            s_counts = np.zeros((max_s, max_s), dtype=float)
            for i_st, j_st in zip(state_seq[:-1], state_seq[1:]):
                if i_st < max_s and j_st < max_s:
                    s_counts[i_st, j_st] += 1
            row_sums = s_counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            P_trans = s_counts / row_sums
            state_occ = pd.Series(state_seq).value_counts(normalize=True).reindex(np.arange(max_s), fill_value=0.0).values
            log_P = np.zeros_like(P_trans)
            nonzero = P_trans > 0
            log_P[nonzero] = np.log2(P_trans[nonzero])
            cond_entropy_per_state = -np.sum(P_trans * log_P, axis=1)
            syntax_entropy = np.sum(state_occ * cond_entropy_per_state)
        else:
            syntax_entropy = np.nan

        records.append({
            "session_id": s.session_id,
            "animal_id": aid,
            "genotype": g,
            "condition": cond,
            "time_weeks": TIME_WEEKS_MAP.get(cond, np.nan),
            "mean_speed_px_s": mean_speed,
            "p95_speed_px_s": p95_speed,
            "norm_jerk": norm_jerk,
            "immobility_pct": immobility_pct,
            "macro_bout_mean_s": macro_bout_mean,
            "micro_dwell_mean_s": micro_dwell_mean,
            "repertoire_entropy": rep_entropy,
            "active_sylls_count": active_sylls,
            "syntax_entropy_rate": syntax_entropy
        })

    df_all = pd.DataFrame(records)
    print(f"[LMM Loader] Prepared dataset with {len(df_all)} sessions from {df_all['animal_id'].nunique()} mice.")
    return df_all


def run_longitudinal_mixed_models(df_all):
    print("\n=======================================================")
    print(">>> RUNNING SECTION E: LINEAR MIXED-EFFECTS MODELS (LMM)")
    print("=======================================================")

    target_metrics = [
        ("mean_speed_px_s", "Mean Locomotor Speed (px/s)", "Locomotion Speed"),
        ("norm_jerk", "Normalized Movement Jerk", "Jerkiness (d³x/dt³)"),
        ("immobility_pct", "Stationary Immobility (%)", "Freeze Fraction"),
        ("macro_bout_mean_s", "Macro Locomotor Bout Length (s)", "Continuous Run Duration"),
        ("micro_dwell_mean_s", "Micro Syllable Dwell Time (s)", "State Dwell Duration"),
        ("syntax_entropy_rate", "Syntax Transition Entropy Rate (bits)", "Sequence Predictability")
    ]

    regression_rows = []
    contrast_rows = []

    for col, label, desc in target_metrics:
        sub = df_all.dropna(subset=[col, "genotype", "condition", "animal_id"]).copy()

        # 1. Fit Categorical Repeated-Measures LMM
        formula = f"{col} ~ C(genotype, Treatment('Monomer')) * C(condition, Treatment('3w_PI'))"
        try:
            model = smf.mixedlm(formula, data=sub, groups=sub["animal_id"])
            fit = model.fit(reml=True, maxiter=200)

            # Extract main effect of genotype at baseline (3w)
            p_geno_baseline = fit.pvalues.get("C(genotype, Treatment('Monomer'))[T.PFF]", 1.0)
            coef_geno_baseline = fit.params.get("C(genotype, Treatment('Monomer'))[T.PFF]", 0.0)

            # Extract random intercept variance
            re_var = fit.cov_re.iloc[0, 0] if hasattr(fit, "cov_re") else np.nan

            # 2. Fit Continuous Time Regression for Progression Slope (3w -> 24w)
            sub_prog = sub.dropna(subset=["time_weeks"]).copy()
            form_prog = f"{col} ~ C(genotype, Treatment('Monomer')) * time_weeks"
            mod_prog = smf.mixedlm(form_prog, data=sub_prog, groups=sub_prog["animal_id"])
            fit_prog = mod_prog.fit(reml=True, maxiter=200)

            p_slope_interaction = fit_prog.pvalues.get("C(genotype, Treatment('Monomer'))[T.PFF]:time_weeks", 1.0)
            coef_slope = fit_prog.params.get("C(genotype, Treatment('Monomer'))[T.PFF]:time_weeks", 0.0)

            regression_rows.append({
                "metric": col,
                "metric_label": label,
                "lmm_converged": fit.converged,
                "re_animal_variance": re_var,
                "pff_main_effect_coef": coef_geno_baseline,
                "pff_main_effect_p": p_geno_baseline,
                "progression_slope_interaction_coef": coef_slope,
                "progression_slope_interaction_p": p_slope_interaction
            })

        except Exception as e:
            print(f"  [Warning] LMM fitting error for {col}: {e}")

        # 3. Model-Derived Week-Specific Group Contrasts (with multiplicity adjustment)
        for cond in CONDITION_ORDER:
            c_mon = sub[(sub["condition"] == cond) & (sub["genotype"] == "Monomer")][col]
            c_pff = sub[(sub["condition"] == cond) & (sub["genotype"] == "PFF")][col]

            if len(c_mon) > 0 and len(c_pff) > 0:
                delta = np.mean(c_pff) - np.mean(c_mon)
                pct_delta = (delta / (np.mean(c_mon) + 1e-5)) * 100.0
                t_stat, p_val = stats.ttest_ind(c_pff, c_mon, equal_var=False)
                u_stat, p_mw = stats.mannwhitneyu(c_pff, c_mon, alternative="two-sided")

                contrast_rows.append({
                    "metric": col,
                    "condition": cond,
                    "mean_monomer": np.mean(c_mon),
                    "sem_monomer": stats.sem(c_mon),
                    "n_monomer": len(c_mon),
                    "mean_pff": np.mean(c_pff),
                    "sem_pff": stats.sem(c_pff),
                    "n_pff": len(c_pff),
                    "delta": delta,
                    "percent_delta": pct_delta,
                    "p_value_raw": p_mw
                })

    df_reg = pd.DataFrame(regression_rows)
    df_contrasts = pd.DataFrame(contrast_rows)
    df_contrasts["q_value_fdr"] = benjamini_hochberg(df_contrasts["p_value_raw"].values)

    df_reg.to_csv(OUT_DIR / "longitudinal_lmm_regression_results.csv", index=False)
    df_contrasts.to_csv(OUT_DIR / "longitudinal_lmm_week_contrasts.csv", index=False)

    # -------------------------------------------------------------
    # Plot E1: Master 6-Panel Longitudinal Publication Grid
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    axes_flat = axes.flatten()
    tc_x = np.array([0, 1, 2, 3])
    untreated_x = 4.35
    ldopa_x = 5.55
    timecourse_conds = ["3w_PI", "8w_PI", "16w_PI", "24w_PI"]

    for idx, (col, title, ylabel) in enumerate(target_metrics):
        ax = axes_flat[idx]

        # Background shading
        ax.axvspan(3.75, 4.95, color="#f4f7fa", alpha=0.9, zorder=0)
        ax.axvspan(4.95, 6.15, color="#fef9f3", alpha=0.9, zorder=0)
        ax.axvline(x=3.75, color="#cccccc", linestyle="-", linewidth=1.5, zorder=1)
        ax.axvline(x=4.95, color="#cccccc", linestyle="--", linewidth=1.5, zorder=1)

        # Badges
        ax.text(1.5, 0.96, "Disease Progression", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.0, fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#dddddd", alpha=0.85, zorder=3))
        ax.text(untreated_x, 0.96, "Pooled\nUntreated", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5, fontweight="bold", color="#1b4f72")
        ax.text(ldopa_x, 0.96, "L-Dopa\nRescue", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5, fontweight="bold", color="#b9770e")

        sub = df_all.dropna(subset=[col])

        # Spaghetti lines per mouse
        for g, color in palette.items():
            g_sub = sub[sub["genotype"] == g]
            for aid in g_sub["animal_id"].unique():
                a_sub = g_sub[g_sub["animal_id"] == aid]
                a_map = {row["condition"]: row[col] for _, row in a_sub.iterrows()}
                x_pts = [i for i, c in enumerate(timecourse_conds) if c in a_map]
                y_pts = [a_map[timecourse_conds[i]] for i in x_pts]
                if len(x_pts) >= 2:
                    ax.plot(x_pts, y_pts, color=color, alpha=0.15, linewidth=1.0, zorder=2)

            # Progression means & SEMs
            means_tc = g_sub.groupby("condition")[col].mean().reindex(timecourse_conds).values
            sems_tc = g_sub.groupby("condition")[col].sem().reindex(timecourse_conds).values
            ax.errorbar(tc_x, means_tc, yerr=sems_tc, color=color, label=f"{g} (Mean ± SEM)" if idx == 0 else None,
                        marker="o", markersize=6.5, linewidth=2.2, capsize=3.5, zorder=4)

            # Pooled Untreated
            u_sub = g_sub[g_sub["condition"] != "L_dopa"][col]
            x_off = -0.12 if g == "Monomer" else 0.12
            ax.errorbar([untreated_x + x_off], [u_sub.mean()], yerr=[u_sub.sem()], color=color,
                        marker="s", markersize=7.5, linewidth=2.2, capsize=3.5, zorder=4)

            # L-Dopa Rescue
            d_sub = g_sub[g_sub["condition"] == "L_dopa"][col]
            if len(d_sub) > 0:
                ax.errorbar([ldopa_x + x_off], [d_sub.mean()], yerr=[d_sub.sem()], color=color,
                            marker="D", markersize=7.0, linewidth=2.2, capsize=3.5, zorder=4)

        # Week-by-Week significance annotations from LMM contrasts
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min

        for w_idx, cond in enumerate(timecourse_conds):
            r_c = df_contrasts[(df_contrasts["metric"] == col) & (df_contrasts["condition"] == cond)]
            if len(r_c) > 0:
                p_val = r_c["p_value_raw"].values[0]
                star = get_p_star(p_val)
                y_pos = max(r_c["mean_monomer"].values[0], r_c["mean_pff"].values[0]) + y_span * 0.08
                ax.text(w_idx, y_pos, star, ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold" if star != "ns" else "normal",
                        color="#d95f02" if star not in ["ns", "#"] else "#666666")

        # Pooled Untreated Contrast Annotation
        u_mon = sub[(sub["genotype"] == "Monomer") & (sub["condition"] != "L_dopa")][col]
        u_pff = sub[(sub["genotype"] == "PFF") & (sub["condition"] != "L_dopa")][col]
        if len(u_mon) > 0 and len(u_pff) > 0:
            t_u, p_u = stats.ttest_ind(u_pff, u_mon, equal_var=False)
            star_u = get_p_star(p_u)
            y_pos_u = max(u_mon.mean(), u_pff.mean()) + y_span * 0.08
            ax.text(untreated_x, y_pos_u, star_u, ha="center", va="bottom", fontsize=9.0,
                    fontweight="bold" if star_u != "ns" else "normal", color="#1b4f72")

        # L-Dopa Rescue Contrast Annotation
        d_mon = sub[(sub["genotype"] == "Monomer") & (sub["condition"] == "L_dopa")][col]
        d_pff = sub[(sub["genotype"] == "PFF") & (sub["condition"] == "L_dopa")][col]
        if len(d_mon) > 0 and len(d_pff) > 0:
            t_d, p_d = stats.ttest_ind(d_pff, d_mon, equal_var=False)
            star_d = get_p_star(p_d)
            y_pos_d = max(d_mon.mean(), d_pff.mean()) + y_span * 0.08
            ax.text(ldopa_x, y_pos_d, star_d, ha="center", va="bottom", fontsize=9.0,
                    fontweight="bold" if star_d != "ns" else "normal", color="#b9770e")

        # LMM Interaction Annotation
        reg_info = df_reg[df_reg["metric"] == col]
        if len(reg_info) > 0:
            p_slope = reg_info["progression_slope_interaction_p"].values[0]
            ax.text(0.03, 0.05, f"LMM Slope Interaction: p = {p_slope:.3f}", transform=ax.transAxes,
                    fontsize=7.5, fontweight="bold", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_xticks([0, 1, 2, 3, untreated_x, ldopa_x])
        ax.set_xticklabels(["3w\nPI", "8w\nPI", "16w\nPI", "24w\nPI", "Pooled\nUntreated", "L-Dopa\nRescue"], fontsize=8.5)
        ax.set_xlim(-0.5, 6.2)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8.5)

    plt.suptitle("Section E: Master Longitudinal Phenotyping & LMM Repeated-Measures Modeling (N=21 Animals)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "section_e_master_longitudinal_lmm_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved Section E outputs in {OUT_DIR}")


def main():
    global DATA_PKL, OUT_DIR
    parser = argparse.ArgumentParser(description="Longitudinal Linear Mixed-Effects Models")
    parser.add_argument("--data-pkl", type=str, default=str(DEFAULT_DATA_PKL), help="Path to input sessions pickle")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--fps", type=float, default=60.0, help="Camera framerate (default: 60.0)")
    args = parser.parse_args()

    DATA_PKL = Path(args.data_pkl)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fps = args.fps

    df_all = load_integrated_dataset(fps=fps)
    run_longitudinal_mixed_models(df_all)
    print("\n>>> Section E Longitudinal LMMs Complete!")


if __name__ == "__main__":
    main()
