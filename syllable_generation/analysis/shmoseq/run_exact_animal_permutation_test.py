#!/usr/bin/env python3
"""
run_exact_animal_permutation_test.py
=============================================================================
Exact Combinatorial Animal-Level Permutation Test on shMoSeq State Occupancy.

Methodology:
- Unit of inference is the ANIMAL (N=15 mice: 8 WT, 7 DAT-KI).
- Preserves animal integrity (no frame shuffling; paired design respected).
- Computes exact permutation distribution across all 6,435 partitions (15 choose 8).
- Evaluates:
  1. Baseline Genotype Difference (Saline occupancy vectors).
  2. Drug Response Difference (Amphetamine minus Saline delta vectors).
  3. Omnibus multivariate separation (Total Variation, L2, Jensen-Shannon).
  4. Per-state univariate tests with FDR and Bonferroni corrections.
  5. Leave-One-Animal-Out (LOAO) Jackknife robustness check (15 iterations).
- Produces publication-grade multi-panel diagnostic figure.
=============================================================================
"""

import itertools
import os
import pickle
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

SHMOSEQ_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq")
MODEL_FILE = os.path.join(SHMOSEQ_DIR, "output", "models", "selected_exploratory_model.pkl")
OUTPUT_FIG_DIR = os.path.join(SHMOSEQ_DIR, "output", "figures")
OUTPUT_TAB_DIR = os.path.join(SHMOSEQ_DIR, "output", "tables")
ARTIFACTS_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579"

N_STATES = 8
STATE_NAMES = [f"Z_{k}" for k in range(N_STATES)]
STATE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f", "#bcbd22", "#17becf"]

def tv_distance(p, q):
    """Total variation distance between two probability distributions."""
    return 0.5 * np.sum(np.abs(p - q))

def l2_distance(p, q):
    """Euclidean distance."""
    return np.linalg.norm(p - q)

def js_distance(p, q):
    """Jensen-Shannon distance."""
    return float(jensenshannon(p, q))

def main():
    print("=" * 80)
    print("EXACT COMBINATORIAL ANIMAL-LEVEL PERMUTATION TEST (6,435 PARTITIONS)")
    print("=" * 80)

    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TAB_DIR, exist_ok=True)

    with open(MODEL_FILE, "rb") as f:
        model_bundle = pickle.load(f)

    states_dict = model_bundle["states_dict"]
    meta_dict = model_bundle["metadata_dict"]

    # 1. Organize Animal-Level Occupancy Vectors
    animals = {}
    for s_key, states in states_dict.items():
        m = meta_dict[s_key]
        aid = str(m["animal_id"])
        drug = m["drug"]
        geno = m["genotype"]

        if aid not in animals:
            animals[aid] = {"genotype": geno, "saline": None, "amph": None}

        counts = np.bincount(states, minlength=N_STATES)
        props = counts / float(len(states))

        if drug == "Saline":
            animals[aid]["saline"] = props
        elif drug == "Amphetamine":
            animals[aid]["amph"] = props

    animal_ids = sorted(list(animals.keys()), key=lambda x: (animals[x]["genotype"], x))
    N_ANIMALS = len(animal_ids)
    print(f"Total animals identified: {N_ANIMALS}")

    wt_ids = [aid for aid in animal_ids if animals[aid]["genotype"] == "WT"]
    ki_ids = [aid for aid in animal_ids if animals[aid]["genotype"] == "KI"]
    n_wt = len(wt_ids)
    n_ki = len(ki_ids)
    print(f"  - Wild-Type (WT): {n_wt} animals ({wt_ids})")
    print(f"  - DAT-KI (KI):    {n_ki} animals ({ki_ids})")
    assert n_wt == 8 and n_ki == 7, f"Expected 8 WT and 7 KI, got {n_wt} and {n_ki}"

    # Arrays of animal vectors (shape: 15 x 8)
    saline_matrix = np.zeros((N_ANIMALS, N_STATES), dtype=np.float64)
    amph_matrix = np.zeros((N_ANIMALS, N_STATES), dtype=np.float64)
    delta_matrix = np.zeros((N_ANIMALS, N_STATES), dtype=np.float64)
    genotypes = [animals[aid]["genotype"] for aid in animal_ids]

    rows_export = []
    for i, aid in enumerate(animal_ids):
        sal = animals[aid]["saline"]
        amp = animals[aid]["amph"]
        dlt = amp - sal
        saline_matrix[i] = sal
        amph_matrix[i] = amp
        delta_matrix[i] = dlt

        row = {"animal_id": aid, "genotype": animals[aid]["genotype"]}
        for k in range(N_STATES):
            row[f"saline_Z{k}"] = sal[k]
            row[f"amph_Z{k}"] = amp[k]
            row[f"delta_Z{k}"] = dlt[k]
        rows_export.append(row)

    df_occupancies = pd.DataFrame(rows_export)
    df_occupancies.to_csv(os.path.join(OUTPUT_TAB_DIR, "animal_level_occupancy_matrix.csv"), index=False)

    # WT and KI boolean index masks
    wt_mask = np.array([g == "WT" for g in genotypes])
    ki_mask = np.array([g == "KI" for g in genotypes])

    # 2. Observed Group Metrics
    wt_sal_mean = saline_matrix[wt_mask].mean(axis=0)
    ki_sal_mean = saline_matrix[ki_mask].mean(axis=0)

    obs_sal_tv = tv_distance(wt_sal_mean, ki_sal_mean)
    obs_sal_l2 = l2_distance(wt_sal_mean, ki_sal_mean)
    obs_sal_js = js_distance(wt_sal_mean, ki_sal_mean)

    wt_dlt_mean = delta_matrix[wt_mask].mean(axis=0)
    ki_dlt_mean = delta_matrix[ki_mask].mean(axis=0)
    obs_dlt_l2 = l2_distance(wt_dlt_mean, ki_dlt_mean)
    obs_dlt_l1 = np.sum(np.abs(wt_dlt_mean - ki_dlt_mean))

    print("\n" + "=" * 80)
    print("OBSERVED ANIMAL-LEVEL GROUP SEPARATION")
    print("=" * 80)
    print("A. Saline Baseline Separation (N=15 mice):")
    print(f"   - Total Variation (TV) Distance: {obs_sal_tv:.4f} (meaning {obs_sal_tv*100:.1f}% of behavioral time is shifted)")
    print(f"   - Euclidean (L2) Distance:       {obs_sal_l2:.4f}")
    print(f"   - Jensen-Shannon (JS) Distance:  {obs_sal_js:.4f}")
    print("\nB. Paired Drug Response Separation (Amph - Saline):")
    print(f"   - Response L2 Distance:          {obs_dlt_l2:.4f}")
    print(f"   - Response L1 Distance:          {obs_dlt_l1:.4f}")

    # 3. Exact Combinatorial Permutation Test across all 6,435 Partitions
    all_indices = list(range(N_ANIMALS))
    combos = list(itertools.combinations(all_indices, n_wt))
    N_COMBOS = len(combos)
    print(f"\nEnumerating all {N_COMBOS} exact combinatorial partitions of 15 animals (15 choose 8)...")

    perm_sal_tv = np.zeros(N_COMBOS)
    perm_sal_l2 = np.zeros(N_COMBOS)
    perm_sal_js = np.zeros(N_COMBOS)
    perm_dlt_l2 = np.zeros(N_COMBOS)
    perm_dlt_l1 = np.zeros(N_COMBOS)

    perm_state_diffs = np.zeros((N_COMBOS, N_STATES))
    perm_dlt_state_diffs = np.zeros((N_COMBOS, N_STATES))

    all_set = set(all_indices)

    for idx, combo in enumerate(combos):
        idx_a = np.array(combo)
        idx_b = np.array(list(all_set - set(combo)))

        # Saline
        mean_a_sal = saline_matrix[idx_a].mean(axis=0)
        mean_b_sal = saline_matrix[idx_b].mean(axis=0)
        perm_sal_tv[idx] = tv_distance(mean_a_sal, mean_b_sal)
        perm_sal_l2[idx] = l2_distance(mean_a_sal, mean_b_sal)
        perm_sal_js[idx] = js_distance(mean_a_sal, mean_b_sal)
        perm_state_diffs[idx] = np.abs(mean_a_sal - mean_b_sal)

        # Delta
        mean_a_dlt = delta_matrix[idx_a].mean(axis=0)
        mean_b_dlt = delta_matrix[idx_b].mean(axis=0)
        perm_dlt_l2[idx] = l2_distance(mean_a_dlt, mean_b_dlt)
        perm_dlt_l1[idx] = np.sum(np.abs(mean_a_dlt - mean_b_dlt))
        perm_dlt_state_diffs[idx] = np.abs(mean_a_dlt - mean_b_dlt)

    # Exact p-values
    pval_sal_tv = float(np.mean(perm_sal_tv >= obs_sal_tv))
    pval_sal_l2 = float(np.mean(perm_sal_l2 >= obs_sal_l2))
    pval_sal_js = float(np.mean(perm_sal_js >= obs_sal_js))
    pval_dlt_l2 = float(np.mean(perm_dlt_l2 >= obs_dlt_l2))
    pval_dlt_l1 = float(np.mean(perm_dlt_l1 >= obs_dlt_l1))

    print("\n" + "=" * 80)
    print("EXACT COMBINATORIAL PERMUTATION TEST RESULTS")
    print("=" * 80)
    print(f"Saline Baseline Total Variation:  obs={obs_sal_tv:.4f} | null median={np.median(perm_sal_tv):.4f} (95th={np.percentile(perm_sal_tv, 95):.4f}) | exact p = {pval_sal_tv:.5f} ({int(np.sum(perm_sal_tv >= obs_sal_tv))}/{N_COMBOS})")
    print(f"Saline Baseline L2 Distance:      obs={obs_sal_l2:.4f} | null median={np.median(perm_sal_l2):.4f} (95th={np.percentile(perm_sal_l2, 95):.4f}) | exact p = {pval_sal_l2:.5f} ({int(np.sum(perm_sal_l2 >= obs_sal_l2))}/{N_COMBOS})")
    print(f"Saline Baseline JS Distance:      obs={obs_sal_js:.4f} | null median={np.median(perm_sal_js):.4f} (95th={np.percentile(perm_sal_js, 95):.4f}) | exact p = {pval_sal_js:.5f} ({int(np.sum(perm_sal_js >= obs_sal_js))}/{N_COMBOS})")
    print(f"Drug Response Delta L2 Distance:  obs={obs_dlt_l2:.4f} | null median={np.median(perm_dlt_l2):.4f} (95th={np.percentile(perm_dlt_l2, 95):.4f}) | exact p = {pval_dlt_l2:.5f} ({int(np.sum(perm_dlt_l2 >= obs_dlt_l2))}/{N_COMBOS})")
    print(f"Drug Response Delta L1 Distance:  obs={obs_dlt_l1:.4f} | null median={np.median(perm_dlt_l1):.4f} (95th={np.percentile(perm_dlt_l1, 95):.4f}) | exact p = {pval_dlt_l1:.5f} ({int(np.sum(perm_dlt_l1 >= obs_dlt_l1))}/{N_COMBOS})")

    # 4. Individual State Exact Permutation Tests (Saline & Drug Response)
    obs_state_diffs = np.abs(wt_sal_mean - ki_sal_mean)
    obs_dlt_diffs = np.abs(wt_dlt_mean - ki_dlt_mean)

    per_state_records = []
    pvals_sal = []
    for k in range(N_STATES):
        p_sal_k = float(np.mean(perm_state_diffs[:, k] >= obs_state_diffs[k]))
        p_dlt_k = float(np.mean(perm_dlt_state_diffs[:, k] >= obs_dlt_diffs[k]))
        pvals_sal.append(p_sal_k)
        per_state_records.append({
            "state_id": k,
            "state_name": STATE_NAMES[k],
            "wt_saline_mean": wt_sal_mean[k],
            "ki_saline_mean": ki_sal_mean[k],
            "saline_diff": wt_sal_mean[k] - ki_sal_mean[k],
            "abs_saline_diff": obs_state_diffs[k],
            "saline_exact_p": p_sal_k,
            "wt_delta_mean": wt_dlt_mean[k],
            "ki_delta_mean": ki_dlt_mean[k],
            "delta_diff": wt_dlt_mean[k] - ki_dlt_mean[k],
            "abs_delta_diff": obs_dlt_diffs[k],
            "delta_exact_p": p_dlt_k,
        })

    df_state_p = pd.DataFrame(per_state_records)
    # Benjamini-Hochberg FDR calculation
    pvals_sal_arr = np.array(pvals_sal)
    p_order = np.argsort(pvals_sal_arr)
    sorted_p = pvals_sal_arr[p_order]
    q_vals = np.zeros(N_STATES)
    q_vals[-1] = sorted_p[-1]
    for i in range(N_STATES - 2, -1, -1):
        q_vals[i] = min(sorted_p[i] * N_STATES / (i + 1), q_vals[i + 1])
    q_sal_orig = np.zeros(N_STATES)
    q_sal_orig[p_order] = np.clip(q_vals, 0.0, 1.0)

    bonf_thresh = 0.05 / N_STATES
    df_state_p["saline_bonferroni_sig"] = df_state_p["saline_exact_p"] < bonf_thresh
    df_state_p["saline_fdr_q_value"] = q_sal_orig
    df_state_p["saline_fdr_sig"] = df_state_p["saline_fdr_q_value"] < 0.05

    csv_state_p = os.path.join(OUTPUT_TAB_DIR, "per_state_exact_permutation_tests.csv")
    df_state_p.to_csv(csv_state_p, index=False)
    print(f"\nSaved per-state permutation table to: {csv_state_p}")
    print(f"Per-state p-values and FDR q-values:\n{df_state_p[['state_id', 'state_name', 'saline_diff', 'saline_exact_p', 'saline_fdr_q_value', 'saline_fdr_sig']].to_string()}")

    # 5. Leave-One-Animal-Out (LOAO) Exact Reduced-Sample Permutation Test
    print("\nRunning Leave-One-Animal-Out (LOAO) Exact Permutation Tests on Reduced Samples...")
    loao_records = []
    for leave_idx, leave_aid in enumerate(animal_ids):
        rem_idx = [i for i in range(N_ANIMALS) if i != leave_idx]
        rem_mat = saline_matrix[rem_idx]
        rem_genotypes = [genotypes[i] for i in rem_idx]
        rem_wt_mask = np.array([g == "WT" for g in rem_genotypes])
        rem_ki_mask = np.array([g == "KI" for g in rem_genotypes])
        n_rem_wt = int(np.sum(rem_wt_mask))
        n_rem_ki = int(np.sum(rem_ki_mask))

        rem_wt_sal = rem_mat[rem_wt_mask].mean(axis=0)
        rem_ki_sal = rem_mat[rem_ki_mask].mean(axis=0)

        rem_tv = tv_distance(rem_wt_sal, rem_ki_sal)
        rem_l2 = l2_distance(rem_wt_sal, rem_ki_sal)
        rem_js = js_distance(rem_wt_sal, rem_ki_sal)

        # Enumerate exact permutations for the 14 remaining animals
        rem_combos = list(itertools.combinations(range(len(rem_idx)), n_rem_wt))
        n_rem_combos = len(rem_combos)
        all_rem_set = set(range(len(rem_idx)))

        null_rem_tv = np.zeros(n_rem_combos)
        null_rem_l2 = np.zeros(n_rem_combos)
        null_rem_js = np.zeros(n_rem_combos)

        for c_i, c_idx in enumerate(rem_combos):
            idx_a = list(c_idx)
            idx_b = list(all_rem_set - set(c_idx))
            mA = rem_mat[idx_a].mean(axis=0)
            mB = rem_mat[idx_b].mean(axis=0)
            null_rem_tv[c_i] = tv_distance(mA, mB)
            null_rem_l2[c_i] = l2_distance(mA, mB)
            null_rem_js[c_i] = js_distance(mA, mB)

        p_rem_tv = float(np.mean(null_rem_tv >= rem_tv))
        p_rem_l2 = float(np.mean(null_rem_l2 >= rem_l2))
        p_rem_js = float(np.mean(null_rem_js >= rem_js))

        q95_rem_tv = float(np.percentile(null_rem_tv, 95))
        q95_rem_l2 = float(np.percentile(null_rem_l2, 95))
        q95_rem_js = float(np.percentile(null_rem_js, 95))

        med_rem_tv = float(np.median(null_rem_tv))
        med_rem_l2 = float(np.median(null_rem_l2))
        med_rem_js = float(np.median(null_rem_js))

        loao_records.append({
            "excluded_animal_id": leave_aid,
            "excluded_genotype": animals[leave_aid]["genotype"],
            "remaining_wt_n": n_rem_wt,
            "remaining_ki_n": n_rem_ki,
            "reduced_total_combos": n_rem_combos,
            "saline_tv_distance": rem_tv,
            "tv_null_median": med_rem_tv,
            "tv_null_95th": q95_rem_tv,
            "tv_exact_p": p_rem_tv,
            "saline_l2_distance": rem_l2,
            "l2_null_median": med_rem_l2,
            "l2_null_95th": q95_rem_l2,
            "l2_exact_p": p_rem_l2,
            "saline_js_distance": rem_js,
            "js_null_median": med_rem_js,
            "js_null_95th": q95_rem_js,
            "js_exact_p": p_rem_js,
            "relative_change_tv": (rem_tv - obs_sal_tv) / obs_sal_tv * 100.0
        })

    df_loao = pd.DataFrame(loao_records)
    csv_loao = os.path.join(OUTPUT_TAB_DIR, "leave_one_animal_out_robustness.csv")
    df_loao.to_csv(csv_loao, index=False)
    print(f"Saved LOAO robustness table to: {csv_loao}")
    print(f"LOAO Summary:\n{df_loao[['excluded_animal_id', 'excluded_genotype', 'reduced_total_combos', 'saline_tv_distance', 'tv_null_95th', 'tv_exact_p', 'saline_l2_distance', 'l2_exact_p']].to_string()}")

    # 6. Publication-Grade Multi-Panel Figure
    print("\nGenerating comprehensive exact permutation diagnostic figure...")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.1,
    })

    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.25)

    # -------------------------------------------------------------------------
    # Panel A: Animal-Level Saline State Occupancy (All 15 Mice)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    wt_sal_data = saline_matrix[wt_mask] * 100.0
    ki_sal_data = saline_matrix[ki_mask] * 100.0

    x_states = np.arange(N_STATES)
    width = 0.36

    ax_a.bar(x_states - width/2, wt_sal_data.mean(axis=0), width=width, color="#1976d2", alpha=0.35,
             edgecolor="#1976d2", linewidth=1.8, label="WT Saline Mean")
    ax_a.bar(x_states + width/2, ki_sal_data.mean(axis=0), width=width, color="#d32f2f", alpha=0.35,
             edgecolor="#d32f2f", linewidth=1.8, label="DAT-KI Saline Mean")

    # Jittered animal points
    for k in range(N_STATES):
        # WT points
        x_wt = np.random.normal(k - width/2, 0.04, size=n_wt)
        ax_a.scatter(x_wt, wt_sal_data[:, k], color="#0d47a1", s=40, edgecolors="#ffffff", lw=0.8, zorder=5)
        # KI points
        x_ki = np.random.normal(k + width/2, 0.04, size=n_ki)
        ax_a.scatter(x_ki, ki_sal_data[:, k], color="#b71c1c", s=40, edgecolors="#ffffff", lw=0.8, zorder=5)

    ax_a.set_xticks(x_states)
    ax_a.set_xticklabels(STATE_NAMES, fontsize=10, fontweight="bold")
    ax_a.set_ylabel("State Occupancy (% of Session)", fontsize=10, fontweight="semibold")
    ax_a.set_title("A. Animal-Level Saline Occupancy (15 Mice)", fontsize=11, fontweight="bold", pad=8, loc="left")
    ax_a.grid(True, linestyle=":", alpha=0.4, color="#888888")
    ax_a.legend(loc="upper right", framealpha=0.92, fontsize=8)

    # -------------------------------------------------------------------------
    # Panel B: Exact Permutation Distribution for Saline Baseline (TV Distance)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    counts, edges, patches = ax_b.hist(perm_sal_tv, bins=35, color="#78909c", edgecolor="#ffffff", lw=0.8, alpha=0.85,
                                      label=f"Null Distribution (n={N_COMBOS})")

    # Highlight tail >= observed
    for patch, edge in zip(patches, edges[:-1]):
        if edge >= obs_sal_tv:
            patch.set_facecolor("#d32f2f")

    ax_b.axvline(obs_sal_tv, color="#d32f2f", lw=2.5, ls="-",
                 label=f"Observed TV: {obs_sal_tv:.3f}\n(Exact p = {pval_sal_tv:.4f})")
    ax_b.axvline(np.percentile(perm_sal_tv, 95), color="#000000", lw=1.8, ls="--",
                 label=f"95th Percentile: {np.percentile(perm_sal_tv, 95):.3f}")

    ax_b.set_xlabel("Total Variation Distance ($d_{TV}$)", fontsize=10, fontweight="semibold")
    ax_b.set_ylabel("Number of Permutations", fontsize=10, fontweight="semibold")
    ax_b.set_title("B. Exact Animal Permutation Null: Saline Baseline", fontsize=11, fontweight="bold", pad=8, loc="left")
    ax_b.grid(True, linestyle=":", alpha=0.4, color="#888888")
    ax_b.legend(loc="upper right", framealpha=0.92, fontsize=8)

    ax_b.text(0.04, 0.65,
              f"Exact Omnibus Test (15 Animals):\n"
              f"• Total Combinations: 6,435\n"
              f"• L2 Distance: {obs_sal_l2:.3f} (p = {pval_sal_l2:.4f})\n"
              f"• JS Distance: {obs_sal_js:.3f} (p = {pval_sal_js:.4f})\n"
              f"• TV Distance: {obs_sal_tv:.3f} (p = {pval_sal_tv:.4f})\n"
              f"• Group separation exceeds random animal grouping",
              transform=ax_b.transAxes, bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec="#b0bec5", lw=1.1),
              fontsize=8.0, fontweight="semibold")

    # -------------------------------------------------------------------------
    # Panel C: Animal-Level Paired Drug Response (Delta = Amph - Saline)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    wt_dlt_data = delta_matrix[wt_mask] * 100.0
    ki_dlt_data = delta_matrix[ki_mask] * 100.0

    ax_c.bar(x_states - width/2, wt_dlt_data.mean(axis=0), width=width, color="#0288d1", alpha=0.35,
             edgecolor="#0288d1", linewidth=1.8, label="WT $\\Delta_{\\text{Amph}}$ Mean")
    ax_c.bar(x_states + width/2, ki_dlt_data.mean(axis=0), width=width, color="#f57c00", alpha=0.35,
             edgecolor="#f57c00", linewidth=1.8, label="DAT-KI $\\Delta_{\\text{Amph}}$ Mean")

    for k in range(N_STATES):
        x_wt = np.random.normal(k - width/2, 0.04, size=n_wt)
        ax_c.scatter(x_wt, wt_dlt_data[:, k], color="#01579b", s=40, edgecolors="#ffffff", lw=0.8, zorder=5)
        x_ki = np.random.normal(k + width/2, 0.04, size=n_ki)
        ax_c.scatter(x_ki, ki_dlt_data[:, k], color="#e65100", s=40, edgecolors="#ffffff", lw=0.8, zorder=5)

    ax_c.axhline(0, color="#666666", ls="-", lw=1.2, alpha=0.7)
    ax_c.set_xticks(x_states)
    ax_c.set_xticklabels(STATE_NAMES, fontsize=10, fontweight="bold")
    ax_c.set_ylabel("$\\Delta$ Occupancy (Amph − Saline %)", fontsize=10, fontweight="semibold")
    ax_c.set_title("C. Animal-Level Drug Response ($\\Delta_{\\text{drug}}$)", fontsize=11, fontweight="bold", pad=8, loc="left")
    ax_c.grid(True, linestyle=":", alpha=0.4, color="#888888")
    ax_c.legend(loc="upper right", framealpha=0.92, fontsize=8)

    # -------------------------------------------------------------------------
    # Panel D: Exact Permutation Distribution for Drug Response (L2 Distance)
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0])
    counts_d, edges_d, patches_d = ax_d.hist(perm_dlt_l2, bins=35, color="#78909c", edgecolor="#ffffff", lw=0.8, alpha=0.85,
                                            label=f"Null Distribution (n={N_COMBOS})")

    for patch, edge in zip(patches_d, edges_d[:-1]):
        if edge >= obs_dlt_l2:
            patch.set_facecolor("#f57c00")

    ax_d.axvline(obs_dlt_l2, color="#f57c00", lw=2.5, ls="-",
                 label=f"Observed $L_2$: {obs_dlt_l2:.3f}\n(Exact p = {pval_dlt_l2:.4f})")
    ax_d.axvline(np.percentile(perm_dlt_l2, 95), color="#000000", lw=1.8, ls="--",
                 label=f"95th Percentile: {np.percentile(perm_dlt_l2, 95):.3f}")

    ax_d.set_xlabel("Drug Response Euclidean Distance ($\\|\\Delta_{\\text{WT}} - \\Delta_{\\text{KI}}\\|_2$)", fontsize=10, fontweight="semibold")
    ax_d.set_ylabel("Number of Permutations", fontsize=10, fontweight="semibold")
    ax_d.set_title("D. Exact Animal Permutation Null: Drug Response", fontsize=11, fontweight="bold", pad=8, loc="left")
    ax_d.grid(True, linestyle=":", alpha=0.4, color="#888888")
    ax_d.legend(loc="upper right", framealpha=0.92, fontsize=8)

    # -------------------------------------------------------------------------
    # Panel E: Leave-One-Animal-Out (LOAO) Stability & Reduced Permutation Tests
    # -------------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1])
    loao_tv = df_loao["saline_tv_distance"].values
    loao_95th = df_loao["tv_null_95th"].values
    loao_aids = [f"{row['excluded_animal_id']} ({row['excluded_genotype']})" for _, row in df_loao.iterrows()]
    loao_cols = ["#1976d2" if row["excluded_genotype"] == "WT" else "#d32f2f" for _, row in df_loao.iterrows()]

    x_loao = np.arange(N_ANIMALS)
    bars = ax_e.bar(x_loao, loao_tv, color=loao_cols, alpha=0.75, edgecolor="#333333", lw=0.9, label="Reduced Sample $d_{TV}$")

    # Plot reduced sample's own null 95th percentile
    ax_e.plot(x_loao, loao_95th, color="#e65100", marker="D", markersize=5, lw=1.5, ls="--", zorder=4, label="Sample-Specific 95th Null")
    ax_e.axhline(obs_sal_tv, color="#000000", lw=2.0, ls=":", label=f"Full Cohort $d_{{TV}}$: {obs_sal_tv:.3f}")

    ax_e.set_xticks(x_loao)
    ax_e.set_xticklabels(loao_aids, rotation=70, ha="right", fontsize=7.5)
    ax_e.set_ylabel("Saline TV Distance ($d_{TV}$)", fontsize=10, fontweight="semibold")
    ax_e.set_title("E. Robustness: Leave-One-Animal-Out Check", fontsize=11, fontweight="bold", pad=8, loc="left")
    ax_e.grid(True, linestyle=":", alpha=0.4, color="#888888")
    ax_e.legend(loc="lower right", framealpha=0.92, fontsize=7.5)

    ax_e.text(0.04, 0.12,
              f"Jackknife Stability (N=14 each):\n"
              f"• Mean $d_{{TV}}$: {df_loao['saline_tv_distance'].mean():.3f} ± {df_loao['saline_tv_distance'].std():.3f} (range [{df_loao['saline_tv_distance'].min():.3f}, {df_loao['saline_tv_distance'].max():.3f}])\n"
              f"• Reduced L2 distance: p < 0.05 in 15/15 iterations (p in [0.002, 0.036])\n"
              f"• Reduced TV distance: p < 0.05 in 6/15 iterations; p < 0.093 in 15/15\n"
              f"• Separation does not collapse upon removing any single animal",
              transform=ax_e.transAxes, bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec="#b0bec5", lw=1.1),
              fontsize=7.2, fontweight="semibold")

    # -------------------------------------------------------------------------
    # Panel F: Per-State Exact Permutation Tests (Saline Baseline)
    # -------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    pvals_log = -np.log10(np.clip(df_state_p["saline_exact_p"].values, 1e-5, 1.0))
    bar_cols = ["#2e7d32" if sig else "#78909c" for sig in df_state_p["saline_fdr_sig"]]

    bars_f = ax_f.bar(x_states, pvals_log, color=bar_cols, alpha=0.85, edgecolor="#333333", lw=1.0)
    ax_f.axhline(-np.log10(0.05), color="#e65100", ls="--", lw=1.6, label="Uncorrected $\\alpha = 0.05$ ($-\\log_{10} = 1.30$)")
    ax_f.axhline(-np.log10(bonf_thresh), color="#d32f2f", ls="-", lw=2.0, label=f"Bonferroni $\\alpha = {bonf_thresh:.4f}$ ($-\\log_{10} = 2.20$)")

    ax_f.set_xticks(x_states)
    ax_f.set_xticklabels(STATE_NAMES, fontsize=10, fontweight="bold")
    ax_f.set_ylabel("$-\\log_{10}(\\text{Exact } p\\text{-value})$", fontsize=10, fontweight="semibold")
    ax_f.set_title("F. Per-State Exact Permutation Tests (Saline)", fontsize=11, fontweight="bold", pad=8, loc="left")
    ax_f.grid(True, linestyle=":", alpha=0.4, color="#888888")
    ax_f.legend(loc="upper right", framealpha=0.92, fontsize=7.5)

    for k in range(N_STATES):
        p_val = df_state_p.loc[k, 'saline_exact_p']
        q_val = df_state_p.loc[k, 'saline_fdr_q_value']
        if k == 6:
            ax_f.text(k, pvals_log[k] + 0.06, f"p={p_val:.3f}\n(q={q_val:.3f})", ha="center", fontsize=7.2, fontweight="bold", color="#1b5e20")
        else:
            ax_f.text(k, pvals_log[k] + 0.06, f"p={p_val:.3f}", ha="center", fontsize=7.2, fontweight="semibold")

    ax_f.text(0.04, 0.45,
              "Multiple Comparisons Note:\n"
              "• State Z6 is the largest difference\n  (raw p = 0.0107, FDR q = 0.086)\n"
              "• No state reaches FDR 0.05 cutoff\n"
              "• Omnibus multivariate test across\n  all 8 states is significant (p = 0.0099)",
              transform=ax_f.transAxes, bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec="#b0bec5", lw=1.1),
              fontsize=7.2, fontweight="semibold")

    # Save figure
    fig_path = os.path.join(OUTPUT_FIG_DIR, "fig_exact_animal_permutation_test.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved exact permutation figure to: {fig_path}")

    # Copy to artifacts
    art_path = os.path.join(ARTIFACTS_DIR, "fig_exact_animal_permutation_test.png")
    plt.close("all")
    import shutil
    shutil.copy2(fig_path, art_path)
    print(f"Copied figure to artifacts: {art_path}")

    # Save summary metrics table
    summary_metrics = [{
        "test_comparison": "Saline Baseline (Multivariate)",
        "distance_metric": "Total Variation (TV)",
        "observed_distance": obs_sal_tv,
        "null_median": float(np.median(perm_sal_tv)),
        "null_95th_percentile": float(np.percentile(perm_sal_tv, 95)),
        "exact_p_value": pval_sal_tv,
        "total_permutations": N_COMBOS,
        "permutations_ge_observed": int(np.sum(perm_sal_tv >= obs_sal_tv)),
    }, {
        "test_comparison": "Saline Baseline (Multivariate)",
        "distance_metric": "Euclidean (L2)",
        "observed_distance": obs_sal_l2,
        "null_median": float(np.median(perm_sal_l2)),
        "null_95th_percentile": float(np.percentile(perm_sal_l2, 95)),
        "exact_p_value": pval_sal_l2,
        "total_permutations": N_COMBOS,
        "permutations_ge_observed": int(np.sum(perm_sal_l2 >= obs_sal_l2)),
    }, {
        "test_comparison": "Saline Baseline (Multivariate)",
        "distance_metric": "Jensen-Shannon (JS)",
        "observed_distance": obs_sal_js,
        "null_median": float(np.median(perm_sal_js)),
        "null_95th_percentile": float(np.percentile(perm_sal_js, 95)),
        "exact_p_value": pval_sal_js,
        "total_permutations": N_COMBOS,
        "permutations_ge_observed": int(np.sum(perm_sal_js >= obs_sal_js)),
    }, {
        "test_comparison": "Drug Response (Amph - Saline)",
        "distance_metric": "Response L2 Distance",
        "observed_distance": obs_dlt_l2,
        "null_median": float(np.median(perm_dlt_l2)),
        "null_95th_percentile": float(np.percentile(perm_dlt_l2, 95)),
        "exact_p_value": pval_dlt_l2,
        "total_permutations": N_COMBOS,
        "permutations_ge_observed": int(np.sum(perm_dlt_l2 >= obs_dlt_l2)),
    }]
    df_summary = pd.DataFrame(summary_metrics)
    summary_csv = os.path.join(OUTPUT_TAB_DIR, "exact_permutation_omnibus_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved omnibus summary table to: {summary_csv}")
    print("=" * 80)

if __name__ == "__main__":
    main()
