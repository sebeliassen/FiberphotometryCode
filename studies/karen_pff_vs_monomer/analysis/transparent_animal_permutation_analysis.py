#!/usr/bin/env python3
"""
transparent_animal_permutation_analysis.py

Transparent Animal-Level Permutation Analysis for Syllable Usage Results:
1. Shuffles genotype labels strictly between mice (21 unique animals: 12 PFF vs 9 Monomer).
   - Preserves all longitudinal sessions from the same mouse together.
   - Preserves original group sizes (12 PFF, 9 Monomer).
   - 25,000 permutations.

2. Evaluates 3 clean summary statistics across the 100 syllables:
   - Stat 1 (Star Count): Number of syllables with nominal p < 0.05 (two-tailed Welch t-test)
   - Stat 2 (Max Statistic): Largest absolute t-statistic across all 100 syllables (max_k |T_k|)
   - Stat 3 (Global Distance): Global Jensen-Shannon Distance between 100-D group mean occupancy vectors

3. Computes empirical permutation p-values:
   p = (1 + sum(shuffled >= observed)) / (N_perm + 1)

4. Generates a clear 3-panel publication figure showing each null distribution and the real value.
"""

import os
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from fpcode.behavior.sequences.counting import label_counts
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon

# Clean, professional styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

DATA_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls_k240k.pkl"
OUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "karen_syllable_analysis" / "output" / "permutation_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Loading dataset from {DATA_PKL}...", flush=True)
with open(DATA_PKL, "rb") as f:
    sessions = pickle.load(f)

# Filter for main cohort (exclude pilot)
main_sessions = [
    s for s in sessions
    if s.metadata.get("genotype") in ["PFF", "Monomer"] and not str(s.metadata.get("animal_id", "")).startswith("pilot")
]

print(f"[INFO] Total main cohort sessions: {len(main_sessions)}", flush=True)

# Group sessions by animal
animal_sessions = defaultdict(list)
animal_genotype = {}

for s in main_sessions:
    aid = str(s.metadata["animal_id"])
    geno = str(s.metadata["genotype"])
    animal_sessions[aid].append(s)
    animal_genotype[aid] = geno

animals = sorted(animal_sessions.keys(), key=lambda x: int(x) if x.isdigit() else x)
n_animals = len(animals)
genotypes = np.array([animal_genotype[aid] for aid in animals])

print(f"[INFO] Unique Animals ({n_animals}):", flush=True)
for aid in animals:
    print(f"  - Animal {aid:>3s}: {animal_genotype[aid]:<8s} ({len(animal_sessions[aid])} sessions)", flush=True)

n_pff = np.sum(genotypes == "PFF")
n_mono = np.sum(genotypes == "Monomer")
print(f"\n[INFO] Cohort composition: {n_pff} PFF animals, {n_mono} Monomer animals.", flush=True)

# Exactly 100 syllables: 0..99
K = 100
print(f"[INFO] Repertoire dimensions: K = {K} syllables (0 .. {K-1})", flush=True)

# Compute animal-level occupancy vectors
animal_occupancy = np.zeros((n_animals, K), dtype=np.float64)

for i, aid in enumerate(animals):
    sess_list = animal_sessions[aid]
    total_valid_frames = 0
    syll_counts = np.zeros(K, dtype=np.float64)

    for s in sess_list:
        if isinstance(s.syllables, pd.DataFrame):
            raw_s = s.syllables["syllable"].fillna(-5).to_numpy()
        else:
            raw_s = np.array(s.syllables)

        sylls = pd.to_numeric(raw_s, errors="coerce")
        sylls = np.nan_to_num(sylls, nan=-5).astype(np.int32)

        counts = label_counts(sylls, K, mode="frames")
        total_valid_frames += counts.sum()
        syll_counts += counts

    if total_valid_frames > 0:
        animal_occupancy[i, :] = syll_counts / total_valid_frames

# Normalize each mouse vector to sum to 1.0
animal_occupancy = animal_occupancy / animal_occupancy.sum(axis=1, keepdims=True)

# True group masks
real_pff_mask = (genotypes == "PFF")
real_mono_mask = (genotypes == "Monomer")

# Helper functions
def calc_summary_stats(data_matrix, mask1, mask2):
    """
    Computes for a given group partition:
    1. t-statistics across all 100 columns (Welch two-sample)
    2. Number of syllables with nominal p < 0.05
    3. Maximum absolute t-statistic (max_k |T_k|)
    4. Global Jensen-Shannon Distance between group mean distributions
    5. Global Euclidean L2 distance between group mean distributions
    """
    x1 = data_matrix[mask1] # (n1, K)
    x2 = data_matrix[mask2] # (n2, K)
    m1 = np.mean(x1, axis=0)
    m2 = np.mean(x2, axis=0)
    v1 = np.var(x1, axis=0, ddof=1)
    v2 = np.var(x2, axis=0, ddof=1)
    n1 = x1.shape[0]
    n2 = x2.shape[0]

    denom = np.sqrt(v1 / n1 + v2 / n2)
    denom = np.where(denom == 0, np.nan, denom)
    t_stats = (m1 - m2) / denom
    t_stats = np.nan_to_num(t_stats, nan=0.0)

    # Degrees of freedom for Welch t-test (Satterthwaite approximation)
    num_df = (v1 / n1 + v2 / n2) ** 2
    den_df = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df_welch = np.where(den_df == 0, n1 + n2 - 2, num_df / den_df)
    df_welch = np.nan_to_num(df_welch, nan=n1 + n2 - 2)

    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_welch))

    # 1. Star count (nominal p < 0.05)
    star_count = np.sum(p_vals < 0.05)

    # 2. Max |t| statistic
    max_t = np.max(np.abs(t_stats))

    # 3. Global distance (Jensen-Shannon Distance)
    mu1_norm = m1 / np.sum(m1)
    mu2_norm = m2 / np.sum(m2)
    js_dist = jensenshannon(mu1_norm, mu2_norm)

    # 4. Global Euclidean Distance (L2)
    l2_dist = np.linalg.norm(mu1_norm - mu2_norm)

    return {
        "t_stats": t_stats,
        "p_vals": p_vals,
        "star_count": star_count,
        "max_t": max_t,
        "js_dist": js_dist,
        "l2_dist": l2_dist,
    }

# ==============================================================================
# 1. OBSERVED REAL PFF vs MONOMER SPLIT
# ==============================================================================
obs = calc_summary_stats(animal_occupancy, real_pff_mask, real_mono_mask)

print("\n" + "="*80, flush=True)
print("📊 1. OBSERVED REAL PFF vs MONOMER SUMMARY STATISTICS", flush=True)
print("="*80, flush=True)
print(f"  • Syllables with nominal p < 0.05:  {obs['star_count']} / 100 syllables", flush=True)
print(f"  • Largest absolute t-statistic:     {obs['max_t']:.4f} (Top syllable index: {np.argmax(np.abs(obs['t_stats']))})", flush=True)
print(f"  • Global Jensen-Shannon Distance:   {obs['js_dist']:.4f}", flush=True)
print(f"  • Global Euclidean (L2) Distance:   {obs['l2_dist']:.4f}", flush=True)

# List the nominal p < 0.05 syllables in the real split
sig_sylls = np.where(obs['p_vals'] < 0.05)[0]
print(f"\nSyllables reaching nominal p < 0.05 in real data ({len(sig_sylls)} total):", flush=True)
for s in sig_sylls:
    m_pff = np.mean(animal_occupancy[real_pff_mask, s]) * 100
    m_mono = np.mean(animal_occupancy[real_mono_mask, s]) * 100
    fc = (m_pff - m_mono) / m_mono * 100 if m_mono > 0 else 0.0
    print(f"  - Syllable {s:2d}: t = {obs['t_stats'][s]:+7.3f} | p = {obs['p_vals'][s]:.4f} | PFF = {m_pff:.3f}%, Mono = {m_mono:.3f}% ({fc:+6.1f}%)", flush=True)

# ==============================================================================
# 2. RUN ANIMAL-LEVEL PERMUTATIONS
# ==============================================================================
N_PERMS = 25000
print("\n" + "="*80, flush=True)
print(f"🎲 2. RUNNING {N_PERMS:,} ANIMAL-LEVEL PERMUTATIONS", flush=True)
print("="*80, flush=True)

np.random.seed(42)

null_star_count = np.zeros(N_PERMS, dtype=np.int32)
null_max_t = np.zeros(N_PERMS, dtype=np.float64)
null_js_dist = np.zeros(N_PERMS, dtype=np.float64)
null_l2_dist = np.zeros(N_PERMS, dtype=np.float64)

for b in range(N_PERMS):
    # Randomly partition 21 animals into 12 vs 9
    perm_idx = np.random.permutation(n_animals)
    perm_pff_mask = np.zeros(n_animals, dtype=bool)
    perm_pff_mask[perm_idx[:n_pff]] = True
    perm_mono_mask = ~perm_pff_mask

    res = calc_summary_stats(animal_occupancy, perm_pff_mask, perm_mono_mask)
    null_star_count[b] = res["star_count"]
    null_max_t[b] = res["max_t"]
    null_js_dist[b] = res["js_dist"]
    null_l2_dist[b] = res["l2_dist"]

    if (b + 1) % 5000 == 0:
        print(f"  Completed {b+1:,} / {N_PERMS:,} permutations...", flush=True)

# ==============================================================================
# 3. EMPIRICAL PERMUTATION P-VALUES
# ==============================================================================
p_val_stars = (1 + np.sum(null_star_count >= obs["star_count"])) / (N_PERMS + 1)
p_val_max_t = (1 + np.sum(null_max_t >= obs["max_t"])) / (N_PERMS + 1)
p_val_js_dist = (1 + np.sum(null_js_dist >= obs["js_dist"])) / (N_PERMS + 1)
p_val_l2_dist = (1 + np.sum(null_l2_dist >= obs["l2_dist"])) / (N_PERMS + 1)

print("\n" + "="*80, flush=True)
print("🎯 3. EMPIRICAL PERMUTATION RESULTS & P-VALUES", flush=True)
print("="*80, flush=True)
print("1. STAR COUNT STATISTIC (Number of Syllables with nominal p < 0.05):", flush=True)
print(f"   • Real PFF vs Monomer:   {obs['star_count']} syllables", flush=True)
print(f"   • Shuffled Null Mean:    {np.mean(null_star_count):.2f} syllables (95% CI: [{np.percentile(null_star_count, 2.5):.0f}, {np.percentile(null_star_count, 97.5):.0f}])", flush=True)
print(f"   • Empirical p-value:     p = {p_val_stars:.5f} ({np.sum(null_star_count >= obs['star_count'])} / {N_PERMS} shuffles >= {obs['star_count']})", flush=True)

print("\n2. MAX-STATISTIC (Largest absolute t-statistic across all 100 syllables):", flush=True)
print(f"   • Real PFF vs Monomer:   max |t| = {obs['max_t']:.4f}", flush=True)
print(f"   • Shuffled Null Mean:    max |t| = {np.mean(null_max_t):.4f} (95% CI: [{np.percentile(null_max_t, 2.5):.4f}, {np.percentile(null_max_t, 97.5):.4f}])", flush=True)
print(f"   • Empirical p-value:     p = {p_val_max_t:.5f} ({np.sum(null_max_t >= obs['max_t'])} / {N_PERMS} shuffles >= {obs['max_t']:.4f})", flush=True)

print("\n3. GLOBAL REPERTOIRE DISTANCE (Jensen-Shannon Distance):", flush=True)
print(f"   • Real PFF vs Monomer:   JSD = {obs['js_dist']:.4f}", flush=True)
print(f"   • Shuffled Null Mean:    JSD = {np.mean(null_js_dist):.4f} (95% CI: [{np.percentile(null_js_dist, 2.5):.4f}, {np.percentile(null_js_dist, 97.5):.4f}])", flush=True)
print(f"   • Empirical p-value:     p = {p_val_js_dist:.5f} ({np.sum(null_js_dist >= obs['js_dist'])} / {N_PERMS} shuffles >= {obs['js_dist']:.4f})", flush=True)

# ==============================================================================
# 4. PLOT 3-PANEL VISUAL PERMUTATION DISTRIBUTION
# ==============================================================================
print("\n[INFO] Generating clean 3-panel publication figure...", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

# Panel 1: Star Count
ax1 = axes[0]
# Discrete histogram for counts
max_star_bin = max(np.max(null_star_count), obs['star_count']) + 2
bins1 = np.arange(-0.5, max_star_bin + 0.5, 1)
counts, _, _ = ax1.hist(null_star_count, bins=bins1, density=True, color="#4a90e2", edgecolor="white", alpha=0.75, rwidth=0.85)
ax1.axvline(obs['star_count'], color="#d9534f", linewidth=2.8, linestyle="--", label=f"Real Data: {obs['star_count']} stars")
ax1.set_title(f"A. Number of Syllables with Nominal $p < 0.05$\n(Empirical $p = {p_val_stars:.4f}$)", fontsize=12, fontweight="bold", pad=12)
ax1.set_xlabel("Count of Nominal $p < 0.05$ Syllables", fontsize=11)
ax1.set_ylabel("Null Probability Density", fontsize=11)
ax1.annotate(
    f"Observed = {obs['star_count']} stars\n$p = {p_val_stars:.4f}$",
    xy=(obs['star_count'], max(counts)*0.7),
    xytext=(obs['star_count'] + 1.2, max(counts)*0.85),
    arrowprops=dict(facecolor="#d9534f", shrink=0.08, width=1.5, headwidth=6),
    fontsize=10.5, fontweight="bold", color="#c9302c",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fdf7f7", ec="#d9534f", lw=1.2)
)
ax1.legend(loc="upper right", frameon=True)

# Panel 2: Max |t| statistic
ax2 = axes[1]
sns.histplot(null_max_t, kde=True, ax=ax2, color="#5cb85c", bins=45, stat="density", edgecolor="none", alpha=0.65)
ax2.axvline(obs['max_t'], color="#d9534f", linewidth=2.8, linestyle="--", label=f"Real Data: $\\max|t| = {obs['max_t']:.2f}$")
ax2.set_title(f"B. Strongest Individual Syllable Effect\n($\\max_k |T_k|$, Empirical $p = {p_val_max_t:.4f}$)", fontsize=12, fontweight="bold", pad=12)
ax2.set_xlabel(r"Maximum Absolute $t$-statistic ($\max_k |T_k|$)", fontsize=11)
ax2.set_ylabel("Null Probability Density", fontsize=11)
ax2.annotate(
    f"Observed $\\max|t| = {obs['max_t']:.2f}$\n$p = {p_val_max_t:.4f}$",
    xy=(obs['max_t'], ax2.get_ylim()[1]*0.65),
    xytext=(obs['max_t'] + 0.35, ax2.get_ylim()[1]*0.8),
    arrowprops=dict(facecolor="#d9534f", shrink=0.08, width=1.5, headwidth=6),
    fontsize=10.5, fontweight="bold", color="#c9302c",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fdf7f7", ec="#d9534f", lw=1.2)
)
ax2.legend(loc="upper right", frameon=True)

# Panel 3: Global Repertoire Distance (JSD)
ax3 = axes[2]
sns.histplot(null_js_dist, kde=True, ax=ax3, color="#f0ad4e", bins=45, stat="density", edgecolor="none", alpha=0.65)
ax3.axvline(obs['js_dist'], color="#d9534f", linewidth=2.8, linestyle="--", label=f"Real Data: $JSD = {obs['js_dist']:.3f}$")
ax3.set_title(f"C. Global 100-D Repertoire Divergence\n(Jensen-Shannon Dist, Empirical $p = {p_val_js_dist:.4f}$)", fontsize=12, fontweight="bold", pad=12)
ax3.set_xlabel("Repertoire Divergence (Jensen-Shannon Distance)", fontsize=11)
ax3.set_ylabel("Null Probability Density", fontsize=11)
ax3.annotate(
    f"Observed $JSD = {obs['js_dist']:.3f}$\n$p = {p_val_js_dist:.4f}$",
    xy=(obs['js_dist'], ax3.get_ylim()[1]*0.65),
    xytext=(obs['js_dist'] + 0.015, ax3.get_ylim()[1]*0.8),
    arrowprops=dict(facecolor="#d9534f", shrink=0.08, width=1.5, headwidth=6),
    fontsize=10.5, fontweight="bold", color="#c9302c",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fdf7f7", ec="#d9534f", lw=1.2)
)
ax3.legend(loc="upper right", frameon=True)

plt.tight_layout()
fig_save_path = OUT_DIR / "simple_animal_permutation_summary.png"
plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[INFO] Saved 3-panel figure to: {fig_save_path}", flush=True)

# Save results dictionary to text report
report_path = OUT_DIR / "permutation_test_report.txt"
with open(report_path, "w") as f:
    f.write("="*80 + "\n")
    f.write("TRANSPARENT ANIMAL-LEVEL PERMUTATION TEST REPORT (N_perm = 25,000)\n")
    f.write("="*80 + "\n\n")
    f.write("Cohort: 21 unique animals (12 PFF, 9 Monomer) across 99 total longitudinal sessions.\n")
    f.write("Syllable repertoire: 100 syllables (0 to 99).\n\n")
    f.write("1. STAR COUNT STATISTIC (Number of Syllables with nominal p < 0.05):\n")
    f.write(f"   - Observed Real Split:  {obs['star_count']} syllables\n")
    f.write(f"   - Null Distribution:    Mean = {np.mean(null_star_count):.2f}, Median = {np.median(null_star_count):.0f}, 95% CI: [{np.percentile(null_star_count, 2.5):.0f}, {np.percentile(null_star_count, 97.5):.0f}]\n")
    f.write(f"   - Empirical p-value:    p = {p_val_stars:.5f} ({np.sum(null_star_count >= obs['star_count'])} / {N_PERMS} shuffles >= {obs['star_count']})\n\n")
    f.write("2. MAX-STATISTIC (Largest absolute t-statistic across all 100 syllables):\n")
    f.write(f"   - Observed Real Split:  max |t| = {obs['max_t']:.4f}\n")
    f.write(f"   - Null Distribution:    Mean = {np.mean(null_max_t):.4f}, Median = {np.median(null_max_t):.4f}, 95% CI: [{np.percentile(null_max_t, 2.5):.4f}, {np.percentile(null_max_t, 97.5):.4f}]\n")
    f.write(f"   - Empirical p-value:    p = {p_val_max_t:.5f} ({np.sum(null_max_t >= obs['max_t'])} / {N_PERMS} shuffles >= {obs['max_t']:.4f})\n\n")
    f.write("3. GLOBAL REPERTOIRE DIVERGENCE (Jensen-Shannon Distance):\n")
    f.write(f"   - Observed Real Split:  JSD = {obs['js_dist']:.4f}\n")
    f.write(f"   - Null Distribution:    Mean = {np.mean(null_js_dist):.4f}, Median = {np.median(null_js_dist):.4f}, 95% CI: [{np.percentile(null_js_dist, 2.5):.4f}, {np.percentile(null_js_dist, 97.5):.4f}]\n")
    f.write(f"   - Empirical p-value:    p = {p_val_js_dist:.5f} ({np.sum(null_js_dist >= obs['js_dist'])} / {N_PERMS} shuffles >= {obs['js_dist']:.4f})\n\n")

print(f"[INFO] Saved text report to: {report_path}", flush=True)
print("\n" + "="*80, flush=True)
print("🎉 TRANSPARENT ANIMAL-LEVEL PERMUTATION ANALYSIS COMPLETE!", flush=True)
print("="*80, flush=True)
