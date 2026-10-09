#!/usr/bin/env python3
"""
run_animal_permutation_tests.py

Comprehensive Animal-Level Permutation Testing for Karen Project (kappa=240k):
1. Test A: Max-Statistic Animal-Level Permutation Test (max_k |T_k|)
   - Strictly controls Family-Wise Error Rate (FWER) across all 100 syllables.
   - Computes global p-value: p_global = P(max_k |T_k^perm| >= max_k |T_k^obs|).
   - Computes Westfall-Young / maxT-adjusted p-values for all individual syllables.

2. Test B: Global Repertoire Animal-Level Permutation Test
   - Multivariate behavioral divergence between PFF and Monomer repertoires.
   - Uses Jensen-Shannon Divergence (JSD), Aitchison Distance (CLR), and PERMANOVA Pseudo-F.

* CRITICAL: Preserves all repeated sessions/timepoints within each animal.
  Only permutes genotype labels between the 21 unique animals (12 PFF, 9 Monomer).
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
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA

# Set publication style
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

# Keypoint-MoSeq has exactly 100 syllables: 0..99
K = 100
print(f"[INFO] Repertoire dimensions: K = {K} syllables (0 .. {K-1})", flush=True)

# Compute animal-level occupancy vectors using fast vectorization
animal_occupancy = np.zeros((n_animals, K), dtype=np.float64)
animal_bout_rate = np.zeros((n_animals, K), dtype=np.float64)

for i, aid in enumerate(animals):
    sess_list = animal_sessions[aid]
    total_valid_frames = 0
    total_duration_min = 0.0

    syll_counts = np.zeros(K, dtype=np.float64)
    bout_counts = np.zeros(K, dtype=np.float64)

    for s in sess_list:
        if isinstance(s.syllables, pd.DataFrame):
            raw_s = s.syllables["syllable"].fillna(-5).to_numpy()
        else:
            raw_s = np.array(s.syllables)

        sylls = pd.to_numeric(raw_s, errors="coerce")
        sylls = np.nan_to_num(sylls, nan=-5).astype(np.int32)

        valid_mask = (sylls >= 0) & (sylls < K)
        total_valid_frames += np.sum(valid_mask)
        total_duration_min += len(sylls) / 80.0 / 60.0

        valid_sylls = sylls[valid_mask]
        if len(valid_sylls) > 0:
            counts = np.bincount(valid_sylls, minlength=K)[:K]
            syll_counts += counts

        diffs = np.diff(sylls, prepend=-999)
        bout_mask = (diffs != 0) & valid_mask
        bout_sylls = sylls[bout_mask]
        if len(bout_sylls) > 0:
            bcounts = np.bincount(bout_sylls, minlength=K)[:K]
            bout_counts += bcounts

    if total_valid_frames > 0:
        animal_occupancy[i, :] = syll_counts / total_valid_frames
    if total_duration_min > 0:
        animal_bout_rate[i, :] = bout_counts / total_duration_min

# Ensure occupancy normalized to 1
animal_occupancy = animal_occupancy / animal_occupancy.sum(axis=1, keepdims=True)

# Precompute observed group masks
pff_mask = (genotypes == "PFF")
mono_mask = (genotypes == "Monomer")

# Fast vectorized statistical functions
def compute_t_stats(data_matrix, mask1, mask2):
    """Compute Welch's two-sample t-statistic for each column."""
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
    return t_stats

def compute_jsd(p_mat, mask1, mask2):
    """Compute Jensen-Shannon Divergence between group mean distributions."""
    mu1 = np.mean(p_mat[mask1], axis=0)
    mu2 = np.mean(p_mat[mask2], axis=0)
    mu1 = mu1 / mu1.sum()
    mu2 = mu2 / mu2.sum()
    return jensenshannon(mu1, mu2)

def compute_aitchison_distance(p_mat, mask1, mask2, pseudocount=1e-5):
    """Compute Aitchison distance (Euclidean distance between CLR transforms)."""
    mu1 = np.mean(p_mat[mask1], axis=0) + pseudocount
    mu2 = np.mean(p_mat[mask2], axis=0) + pseudocount
    mu1 /= mu1.sum()
    mu2 /= mu2.sum()
    clr1 = np.log(mu1) - np.mean(np.log(mu1))
    clr2 = np.log(mu2) - np.mean(np.log(mu2))
    return np.linalg.norm(clr1 - clr2)

# Precompute Pairwise Distance Matrix for PERMANOVA
pairwise_D = np.zeros((n_animals, n_animals))
for i in range(n_animals):
    for j in range(i+1, n_animals):
        d = jensenshannon(animal_occupancy[i], animal_occupancy[j])
        pairwise_D[i, j] = d
        pairwise_D[j, i] = d

A_gower = -0.5 * (pairwise_D ** 2)
row_means = np.mean(A_gower, axis=1, keepdims=True)
col_means = np.mean(A_gower, axis=0, keepdims=True)
total_mean = np.mean(A_gower)
G_gower = A_gower - row_means - col_means + total_mean
SS_total = np.trace(G_gower)

def compute_permanova_f_fast(mask1, mask2):
    """Fast PERMANOVA computation using precomputed Gower centered matrix."""
    group_labels = np.where(mask1, 1, 0)
    n1 = np.sum(mask1)
    n2 = np.sum(mask2)
    H = np.zeros((n_animals, n_animals))
    for i in range(n_animals):
        for j in range(n_animals):
            if group_labels[i] == group_labels[j]:
                H[i, j] = 1.0 / (n1 if group_labels[i] == 1 else n2)
    H = H - 1.0 / n_animals

    SS_between = np.trace(H @ G_gower)
    SS_within = SS_total - SS_between
    if SS_within <= 0:
        return 0.0
    return (SS_between / 1.0) / (SS_within / (n_animals - 2))

# ==============================================================================
# 1. OBSERVED STATISTICS
# ==============================================================================
print("\n" + "="*80, flush=True)
print("📊 1. OBSERVED DATA STATISTICS (PFF vs Monomer)", flush=True)
print("="*80, flush=True)

t_obs_occ = compute_t_stats(animal_occupancy, pff_mask, mono_mask)
t_obs_rate = compute_t_stats(animal_bout_rate, pff_mask, mono_mask)

max_t_obs_occ = np.max(np.abs(t_obs_occ))
best_syll_occ = np.argmax(np.abs(t_obs_occ))

max_t_obs_rate = np.max(np.abs(t_obs_rate))
best_syll_rate = np.argmax(np.abs(t_obs_rate))

jsd_obs = compute_jsd(animal_occupancy, pff_mask, mono_mask)
aitchison_obs = compute_aitchison_distance(animal_occupancy, pff_mask, mono_mask)
permanova_f_obs = compute_permanova_f_fast(pff_mask, mono_mask)

print(f"Test A (Max-Statistic Occupancy):   M_obs = {max_t_obs_occ:.4f} (Top Syllable: {best_syll_occ}, t = {t_obs_occ[best_syll_occ]:+.4f})", flush=True)
print(f"Test A (Max-Statistic Bout Rate):   M_obs = {max_t_obs_rate:.4f} (Top Syllable: {best_syll_rate}, t = {t_obs_rate[best_syll_rate]:+.4f})", flush=True)
print(f"Test B (Global JSD Divergence):     JSD_obs = {jsd_obs:.4f}", flush=True)
print(f"Test B (Global Aitchison Distance): Aitchison_obs = {aitchison_obs:.4f}", flush=True)
print(f"Test B (Global PERMANOVA F-stat):   F_obs = {permanova_f_obs:.4f}", flush=True)

# ==============================================================================
# 2. RUN ANIMAL-LEVEL PERMUTATIONS (Preserving repeated sessions)
# ==============================================================================
N_PERMS = 25000
print("\n" + "="*80, flush=True)
print(f"🎲 2. RUNNING {N_PERMS:,} ANIMAL-LEVEL PERMUTATIONS", flush=True)
print("="*80, flush=True)

np.random.seed(42)

perm_max_t_occ = np.zeros(N_PERMS)
perm_max_t_rate = np.zeros(N_PERMS)
perm_jsd = np.zeros(N_PERMS)
perm_aitchison = np.zeros(N_PERMS)
perm_permanova_f = np.zeros(N_PERMS)

for b in range(N_PERMS):
    # Permute animal indices
    perm_idx = np.random.permutation(n_animals)
    perm_pff_mask = np.zeros(n_animals, dtype=bool)
    perm_pff_mask[perm_idx[:n_pff]] = True
    perm_mono_mask = ~perm_pff_mask

    # Test A: Max t-statistic across all 100 syllables
    t_perm_occ = compute_t_stats(animal_occupancy, perm_pff_mask, perm_mono_mask)
    t_perm_rate = compute_t_stats(animal_bout_rate, perm_pff_mask, perm_mono_mask)

    perm_max_t_occ[b] = np.max(np.abs(t_perm_occ))
    perm_max_t_rate[b] = np.max(np.abs(t_perm_rate))

    # Test B: Global divergence
    perm_jsd[b] = compute_jsd(animal_occupancy, perm_pff_mask, perm_mono_mask)
    perm_aitchison[b] = compute_aitchison_distance(animal_occupancy, perm_pff_mask, perm_mono_mask)
    perm_permanova_f[b] = compute_permanova_f_fast(perm_pff_mask, perm_mono_mask)

    if (b + 1) % 5000 == 0:
        print(f"  Completed {b+1:,} / {N_PERMS:,} permutations...", flush=True)

# ==============================================================================
# 3. COMPUTE GLOBAL PERMUTATION P-VALUES
# ==============================================================================
print("\n" + "="*80, flush=True)
print("🎯 3. GLOBAL PERMUTATION TEST RESULTS", flush=True)
print("="*80, flush=True)

p_global_max_t_occ = (np.sum(perm_max_t_occ >= max_t_obs_occ) + 1) / (N_PERMS + 1)
p_global_max_t_rate = (np.sum(perm_max_t_rate >= max_t_obs_rate) + 1) / (N_PERMS + 1)
p_global_jsd = (np.sum(perm_jsd >= jsd_obs) + 1) / (N_PERMS + 1)
p_global_aitchison = (np.sum(perm_aitchison >= aitchison_obs) + 1) / (N_PERMS + 1)
p_global_permanova = (np.sum(perm_permanova_f >= permanova_f_obs) + 1) / (N_PERMS + 1)

print("\n🅰️  TEST A: MAX-STATISTIC PERMUTATION TEST (Family-Wise Across All 100 Syllables):", flush=True)
print(f"   • Occupancy Max-|t|:     M_obs = {max_t_obs_occ:.4f}  |  p_global = {p_global_max_t_occ:.5f} ({np.sum(perm_max_t_occ >= max_t_obs_occ)} / {N_PERMS} null splits >= obs)", flush=True)
print(f"   • Bout Rate Max-|t|:     M_obs = {max_t_obs_rate:.4f}  |  p_global = {p_global_max_t_rate:.5f} ({np.sum(perm_max_t_rate >= max_t_obs_rate)} / {N_PERMS} null splits >= obs)", flush=True)

print("\n🅱️  TEST B: GLOBAL REPERTOIRE PERMUTATION TEST (Multivariate 100-D Behavioral Divergence):", flush=True)
print(f"   • Jensen-Shannon Div:    JSD_obs = {jsd_obs:.4f}     |  p_global = {p_global_jsd:.5f} ({np.sum(perm_jsd >= jsd_obs)} / {N_PERMS} null splits >= obs)", flush=True)
print(f"   • Aitchison (CLR) Dist:  Dist_obs = {aitchison_obs:.4f}    |  p_global = {p_global_aitchison:.5f} ({np.sum(perm_aitchison >= aitchison_obs)} / {N_PERMS} null splits >= obs)", flush=True)
print(f"   • PERMANOVA Pseudo-F:    F_obs = {permanova_f_obs:.4f}       |  p_global = {p_global_permanova:.5f} ({np.sum(perm_permanova_f >= permanova_f_obs)} / {N_PERMS} null splits >= obs)", flush=True)

# Westfall-Young / maxT-adjusted p-values for all syllables
maxt_adjusted_p = np.zeros(K)
for s in range(K):
    maxt_adjusted_p[s] = (np.sum(perm_max_t_occ >= np.abs(t_obs_occ[s])) + 1) / (N_PERMS + 1)

print("\n" + "="*80, flush=True)
print("📋 SUMMARY OF CANDIDATE HITS & TOP SYLLABLES (NOMINAL vs MAX-T ADJUSTED P)", flush=True)
print("="*80, flush=True)
print(f"{'Syllable':<12} {'Description':<26} {'t-stat':<10} {'PFF Mean':<12} {'Mono Mean':<12} {'% Change':<12} {'Nominal p':<12} {'Max-T Adj p':<12}", flush=True)
print("-" * 112, flush=True)

syll_names = {
    6: "Grooming/Sitting sniff",
    93: "Left turn",
    36: "Stand down/back",
    41: "Rightwards scrunch/groom",
}

sorted_sylls = np.argsort(np.abs(t_obs_occ))[::-1]
highlight_sylls = [6, 93, 36, 41]
all_show = sorted_sylls[:5].tolist()
for s in highlight_sylls:
    if s not in all_show:
        all_show.append(s)

for s in all_show:
    name = syll_names.get(s, "Top Ranked Syllable")
    m_pff = np.mean(animal_occupancy[pff_mask, s]) * 100
    m_mono = np.mean(animal_occupancy[mono_mask, s]) * 100
    fc = (m_pff - m_mono) / m_mono * 100 if m_mono > 0 else 0.0
    p_nom = 2 * (1 - stats.t.cdf(np.abs(t_obs_occ[s]), df=n_animals-2))
    p_adj = maxt_adjusted_p[s]
    print(f"Syllable {s:<3d} {name:<26} {t_obs_occ[s]:>+8.3f}   {m_pff:>8.3f}%   {m_mono:>8.3f}%   {fc:>+8.1f}%   {p_nom:>9.4f}   {p_adj:>9.4f}", flush=True)

# ==============================================================================
# 4. GENERATE HIGH-RES PUBLICATION FIGURES
# ==============================================================================
print("\n[INFO] Generating publication-quality permutation figures...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Plot 1: Test A (Max-statistic null distribution)
ax1 = axes[0]
sns.histplot(perm_max_t_occ, kde=True, ax=ax1, color="#3498db", bins=50, stat="density", edgecolor="none", alpha=0.6)
ax1.axvline(max_t_obs_occ, color="#e74c3c", linewidth=2.5, linestyle="--", label=f"Observed $M^{{obs}} = {max_t_obs_occ:.2f}$")
ax1.set_title(f"Test A: Max-Statistic Across All 100 Syllables\n($p_{{\\mathrm{{global}}}} = {p_global_max_t_occ:.4f}$)", fontsize=13, fontweight="bold", pad=12)
ax1.set_xlabel(r"Maximum Syllable Effect: $M = \max_k |T_k|$", fontsize=11)
ax1.set_ylabel("Null Probability Density", fontsize=11)
ax1.annotate(
    f"$M^{{obs}} = {max_t_obs_occ:.2f}$\n$p_{{\\mathrm{{global}}}} = {p_global_max_t_occ:.4f}$",
    xy=(max_t_obs_occ, ax1.get_ylim()[1]*0.6),
    xytext=(max_t_obs_occ + 0.3, ax1.get_ylim()[1]*0.7),
    arrowprops=dict(facecolor="#e74c3c", shrink=0.08, width=1.5, headwidth=7),
    fontsize=11, fontweight="bold", color="#c0392b",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#e74c3c", lw=1.5)
)
ax1.legend(loc="upper right", frameon=True)

# Plot 2: Test B (Global JSD Repertoire Divergence null distribution)
ax2 = axes[1]
sns.histplot(perm_jsd, kde=True, ax=ax2, color="#2ecc71", bins=50, stat="density", edgecolor="none", alpha=0.6)
ax2.axvline(jsd_obs, color="#e74c3c", linewidth=2.5, linestyle="--", label=f"Observed $JSD^{{obs}} = {jsd_obs:.3f}$")
ax2.set_title(f"Test B: Global Repertoire Divergence (JSD)\n($p_{{\\mathrm{{global}}}} = {p_global_jsd:.4f}$)", fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Repertoire Divergence (Jensen-Shannon Distance)", fontsize=11)
ax2.set_ylabel("Null Probability Density", fontsize=11)
ax2.annotate(
    f"$JSD^{{obs}} = {jsd_obs:.3f}$\n$p_{{\\mathrm{{global}}}} = {p_global_jsd:.4f}$",
    xy=(jsd_obs, ax2.get_ylim()[1]*0.6),
    xytext=(jsd_obs + 0.015, ax2.get_ylim()[1]*0.7),
    arrowprops=dict(facecolor="#e74c3c", shrink=0.08, width=1.5, headwidth=7),
    fontsize=11, fontweight="bold", color="#c0392b",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#e74c3c", lw=1.5)
)
ax2.legend(loc="upper right", frameon=True)

plt.tight_layout()
fig_path = OUT_DIR / "animal_permutation_tests.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[INFO] Saved primary figure to: {fig_path}", flush=True)

# Plot 3: 2D PCA ordination of all 21 animals
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(animal_occupancy)

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
for geno, col, marker in [("Monomer", "#2980b9", "o"), ("PFF", "#c0392b", "s")]:
    mask = (genotypes == geno)
    ax.scatter(
        coords_2d[mask, 0], coords_2d[mask, 1],
        c=col, marker=marker, s=120, edgecolors="white", linewidth=1.5,
        label=f"{geno} (n={np.sum(mask)})", alpha=0.9
    )
    c_x, c_y = np.mean(coords_2d[mask, 0]), np.mean(coords_2d[mask, 1])
    ax.scatter(c_x, c_y, c=col, marker="X", s=250, edgecolors="black", linewidth=1.5)

for i, aid in enumerate(animals):
    ax.annotate(f"{aid}", (coords_2d[i, 0]+0.003, coords_2d[i, 1]+0.003), fontsize=8, alpha=0.7)

ax.set_title(f"Animal-Level Behavioral Repertoire Ordination (PCA)\nPERMANOVA $F = {permanova_f_obs:.2f}$, $p = {p_global_permanova:.4f}$", fontsize=12, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", fontsize=11)
ax.legend(frameon=True, loc="upper right")
plt.tight_layout()
pca_fig_path = OUT_DIR / "animal_repertoire_pca_ordination.png"
plt.savefig(pca_fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[INFO] Saved PCA ordination figure to: {pca_fig_path}", flush=True)

# Save CSV
results_df = pd.DataFrame({
    "syllable": np.arange(K),
    "t_statistic": t_obs_occ,
    "pff_mean_pct": np.mean(animal_occupancy[pff_mask], axis=0) * 100,
    "mono_mean_pct": np.mean(animal_occupancy[mono_mask], axis=0) * 100,
    "fold_change_pct": np.where(np.mean(animal_occupancy[mono_mask], axis=0) > 0,
                                (np.mean(animal_occupancy[pff_mask], axis=0) - np.mean(animal_occupancy[mono_mask], axis=0)) / np.mean(animal_occupancy[mono_mask], axis=0) * 100, 0),
    "nominal_p": 2 * (1 - stats.t.cdf(np.abs(t_obs_occ), df=n_animals-2)),
    "maxt_adjusted_p": maxt_adjusted_p,
})
csv_path = OUT_DIR / "animal_permutation_syllable_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"[INFO] Saved summary table to: {csv_path}", flush=True)

print("\n" + "="*80, flush=True)
print("🎉 ALL PERMUTATION TESTS COMPLETED SUCCESSFULLY!", flush=True)
print("="*80, flush=True)
