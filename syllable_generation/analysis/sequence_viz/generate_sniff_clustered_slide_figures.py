#!/usr/bin/env python3
"""
generate_sniff_clustered_slide_figures.py
=============================================================================
Generates presentation-ready slide figures using the unified Sniff representation:
1. fig_level3_slide_duration_vertical.png
   - Top: Total Time Spent in Discovered Chains (Mean ± SEM) with stars
   - Bottom: Saline (p=0.0213*) & Amphetamine (p=0.0093**) Exact Permutation Tests
2. fig_level3_slide_self_transitions.png
   - Top: Self-Transition Probability P(k -> k) (Mean ± SEM) with stars
   - Bottom: 4 Cohort Transition Matrices with highlighted diagonal boxes
3. fig_level3_slide_self_transitions_vertical.png
   - Top: Self-Transition Probability P(k -> k) with stars
   - Bottom: 2 Permutation Tests
=============================================================================
"""

import itertools
import os
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.extract_level3_chain_quantitative_results import (
    decode_episodes_in_session,
)
from syllable_generation.analysis.sequence_viz.optimized_seqnmf.convolutional_seqnmf import ConvolutionalSeqNMF

DATA_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data"
PRESENTATION_DIR = REPO_ROOT / "syllable_generation/presentations"
FIGURES_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/optimized_seqnmf/figures"
ARTIFACT_DIR = Path("/home/fsp585/.gemini/antigravity-ide/brain/6e6d59fd-58a4-4955-a762-5c2c131ab2b6")

# Unified Sniff definitions
L3_MERGED = {
    "Walk_Sharp_L": [29],
    "Walk_Sharp_R": [30],
    "Turn_L": [6],
    "Turn_R": [2],
    "Descent_L": [4],
    "Descent_R": [7],
    "Look_L": [28, 34],
    "Look_R": [5, 22, 40, 42],
    "Walk_L": [3, 23, 55],
    "Walk_R": [1, 11],
    "Walk_Fwd": [19, 38],
    "Sprint": [0, 13, 17, 21, 26, 33, 41],
    "Rear_Elevated": [12, 14, 31],
    "Wall_Lean": [16],
    "Stand_Down": [37],
    "Sniff": [8, 9, 10, 15, 18, 20, 24, 25, 36, 39],
    "Grooming": [27, 44, 52, 58, 60, 61, 63, 65, 66],
    "Rest": [32, 43, 49],
    "Artifact": [35]
}
toks_merged = list(L3_MERGED.keys()) + ["Tail_Unlabeled"]
S_m = len(toks_merged)
map_merged = np.full(100, S_m - 1, dtype=int)
for g_idx, (g_name, s_ids) in enumerate(L3_MERGED.items()):
    for sid in s_ids:
        map_merged[sid] = g_idx

# Load data and injection metadata
data_pkl = DATA_DIR / "unified_sequence_dataset.pkl"
with open(data_pkl, "rb") as f:
    d_bundle = pickle.load(f)
sessions = d_bundle["sessions"]

meta_df = pd.read_csv(DATA_DIR / "session_metadata_ground_truth.csv")
meta_inj = meta_df.set_index(["animal_id", "drug"])["injection_frame"].to_dict()

# Fit model strictly on post-injection valid bouts
all_mapped = []
for s in sessions:
    aid_int = int(s["animal_id"])
    drug = s["drug"]
    inj_f = meta_inj.get((aid_int, drug), 0)
    sylls = s["bouts_df"]["syllable"].values
    starts = s["bouts_df"]["start_frame"].values
    valid = (sylls >= 0) & (sylls < 100) & (starts >= inj_f)
    all_mapped.extend(map_merged[sylls[valid]])
all_mapped = np.array(all_mapped, dtype=int)
T_total = len(all_mapped)

N_FIT = min(60000, T_total)
X_train = np.zeros((S_m, N_FIT), dtype=np.float32)
X_train[all_mapped[:N_FIT], np.arange(N_FIT)] = 1.0

K, L = 10, 3
model = ConvolutionalSeqNMF(K=K, L=L, lambda_ortho=0.005, max_iter=150, use_gpu=False, random_seed=42)
model.fit(X_train)
W = model.W

DISPLAY_TOKENS = {
    "Walk_Sharp_L": "Walk_Sharp_L",
    "Walk_Sharp_R": "Walk_Sharp_R",
    "Turn_L": "Turn_L",
    "Turn_R": "Turn_R",
    "Descent_L": "Descent_L",
    "Descent_R": "Descent_R",
    "Look_L": "Look_L",
    "Look_R": "Look_R",
    "Walk_L": "Walk_L",
    "Walk_R": "Walk_R",
    "Walk_Fwd": "Walk_Fwd",
    "Sprint": "Sprint",
    "Rear_Elevated": "Rear_Elev",
    "Wall_Lean": "Wall_Lean",
    "Stand_Down": "Stand_Down",
    "Sniff": "Sniff",
    "Grooming": "Grooming",
    "Rest": "Rest",
    "Artifact": "Artifact",
    "Tail_Unlabeled": "Tail"
}

VERTICAL_CHAIN_LABELS = {}
for k in range(K):
    steps = []
    for l in range(L):
        top_idx = int(np.argmax(W[:, k, l]))
        raw_name = toks_merged[top_idx]
        steps.append(DISPLAY_TOKENS.get(raw_name, raw_name))
    VERTICAL_CHAIN_LABELS[k + 1] = f"Chain {k+1}\n" + " →\n".join(steps)

COHORT_ORDER = ["WT_Saline", "KI_Saline", "WT_Amphetamine", "KI_Amphetamine"]
COHORT_LABELS = {
    "WT_Saline": "WT Saline",
    "KI_Saline": "DAT-KI Saline",
    "WT_Amphetamine": "WT Amphetamine",
    "KI_Amphetamine": "DAT-KI Amphetamine",
}
COHORT_COLORS = {
    "WT_Saline": "#4A5568",
    "KI_Saline": "#DD6B20",
    "WT_Amphetamine": "#2B6CB0",
    "KI_Amphetamine": "#C53030",
}

# Decode all sessions
all_episodes = []
animal_trans_records = []
cohort_trans_counts = {c: np.zeros((K, K), dtype=np.float64) for c in COHORT_ORDER}

for s_idx, sess in enumerate(sessions):
    aid = str(sess["animal_id"])
    gt = sess["genotype"]
    drug = sess["drug"]
    cohort = f"{gt}_{drug}"
    inj_f = meta_inj.get((int(aid), drug), 0)

    sylls = sess["bouts_df"]["syllable"].values
    durs = sess["bouts_df"]["duration_sec"].values
    starts = sess["bouts_df"]["start_frame"].values

    valid_mask = (sylls >= 0) & (sylls < 100) & (starts >= inj_f)
    sylls_v = sylls[valid_mask]
    durs_v = durs[valid_mask]
    starts_v = starts[valid_mask]

    T_s = len(sylls_v)
    X_sess = np.zeros((S_m, T_s), dtype=np.float32)
    X_sess[map_merged[sylls_v], np.arange(T_s)] = 1.0

    ep = decode_episodes_in_session(X_sess, durs_v, starts_v, W, L=L)
    ep.sort(key=lambda x: x["start_bout"])
    for item in ep:
        item["session_idx"] = s_idx
        item["animal_id"] = aid
        item["genotype"] = gt
        item["drug"] = drug
        item["cohort"] = cohort
    all_episodes.extend(ep)

    factors = [x["factor_id"] for x in ep]
    n_ep = len(factors)
    sess_trans = np.zeros((K, K), dtype=np.float64)
    if n_ep > 1:
        for i in range(n_ep - 1):
            f_curr = factors[i] - 1
            f_next = factors[i+1] - 1
            sess_trans[f_curr, f_next] += 1.0

    cohort_trans_counts[cohort] += sess_trans
    row_sums = sess_trans.sum(axis=1)
    for k in range(K):
        p_self = sess_trans[k, k] / row_sums[k] if row_sums[k] > 0 else 0.0
        animal_trans_records.append({
            "animal_id": aid,
            "genotype": gt,
            "drug": drug,
            "cohort": cohort,
            "factor_id": k + 1,
            "self_trans_prob": p_self,
        })

df_ep = pd.DataFrame(all_episodes)
df_self_anim = pd.DataFrame(animal_trans_records)

# Normalized transition matrices
norm_cohort_trans = {}
for cohort, mat in cohort_trans_counts.items():
    r_sums = mat.sum(axis=1, keepdims=True)
    norm_cohort_trans[cohort] = np.divide(mat, r_sums, out=np.zeros_like(mat), where=r_sums > 0)

# Build per-animal duration and rate metrics
anim_chain_records = []
all_animals_df = df_self_anim[["animal_id", "genotype", "drug", "cohort"]].drop_duplicates()

for _, a_row in all_animals_df.iterrows():
    aid = a_row["animal_id"]
    gt = a_row["genotype"]
    drug = a_row["drug"]
    cohort = a_row["cohort"]
    for k in range(1, K + 1):
        sub = df_ep[(df_ep["animal_id"] == aid) & (df_ep["drug"] == drug) & (df_ep["factor_id"] == k)]
        n_e = len(sub)
        d_tot = sub["duration_sec"].sum() if n_e > 0 else 0.0
        anim_chain_records.append({
            "animal_id": aid,
            "genotype": gt,
            "drug": drug,
            "cohort": cohort,
            "factor_id": k,
            "n_episodes": n_e,
            "total_duration_sec": d_tot,
            "rate_per_min": n_e / 60.0
        })

df_chain_anim = pd.DataFrame(anim_chain_records)

# Cohort summaries for duration and self-transitions
dur_summary = []
self_summary = []
for cohort in COHORT_ORDER:
    for k in range(1, K + 1):
        sub_c = df_chain_anim[(df_chain_anim["cohort"] == cohort) & (df_chain_anim["factor_id"] == k)]
        d_vals = sub_c["total_duration_sec"].values
        dur_summary.append({
            "cohort": cohort,
            "factor_id": k,
            "mean": np.mean(d_vals),
            "sem": stats.sem(d_vals) if len(d_vals) > 1 else 0.0
        })
        sub_s = df_self_anim[(df_self_anim["cohort"] == cohort) & (df_self_anim["factor_id"] == k)]
        s_vals = sub_s["self_trans_prob"].values
        self_summary.append({
            "cohort": cohort,
            "factor_id": k,
            "mean": np.mean(s_vals),
            "sem": stats.sem(s_vals) if len(s_vals) > 1 else 0.0
        })

df_dur_summary = pd.DataFrame(dur_summary)
df_self_summary = pd.DataFrame(self_summary)

# Univariate Mann-Whitney tests
dur_pvals = {}
self_pvals = {}
for k in range(1, K + 1):
    sub_d = df_chain_anim[df_chain_anim["factor_id"] == k]
    sub_s = df_self_anim[df_self_anim["factor_id"] == k]
    for drug in ["Saline", "Amphetamine"]:
        # Duration
        wt_d = sub_d[(sub_d["genotype"] == "WT") & (sub_d["drug"] == drug)]["total_duration_sec"].values
        ki_d = sub_d[(sub_d["genotype"] == "KI") & (sub_d["drug"] == drug)]["total_duration_sec"].values
        _, p_d = stats.mannwhitneyu(wt_d, ki_d, alternative="two-sided")
        dur_pvals[(k, drug)] = p_d

        # Self-trans
        wt_s = sub_s[(sub_s["genotype"] == "WT") & (sub_s["drug"] == drug)]["self_trans_prob"].values
        ki_s = sub_s[(sub_s["genotype"] == "KI") & (sub_s["drug"] == drug)]["self_trans_prob"].values
        _, p_s = stats.mannwhitneyu(wt_s, ki_s, alternative="two-sided")
        self_pvals[(k, drug)] = p_s

# Exact animal permutation test on rate profile
anim_gt = df_chain_anim[["animal_id", "genotype"]].drop_duplicates().set_index("animal_id")["genotype"].to_dict()
all_animals = sorted(list(anim_gt.keys()))
wt_animals = [a for a in all_animals if anim_gt[a] == "WT"]
ki_animals = [a for a in all_animals if anim_gt[a] == "KI"]

partitions = list(itertools.combinations(all_animals, len(wt_animals)))
N_partitions = len(partitions)

profiles = {}
for (aid, drug), sub in df_chain_anim.groupby(["animal_id", "drug"]):
    vec = np.zeros(K, dtype=np.float64)
    for _, row in sub.iterrows():
        k_idx = int(row["factor_id"]) - 1
        vec[k_idx] = row["rate_per_min"]
    s = np.sum(vec)
    if s > 0:
        vec = vec / s
    profiles[(aid, drug)] = vec

def calc_tv(group1, group2, drug):
    vecs1 = [profiles[(a, drug)] for a in group1 if (a, drug) in profiles]
    vecs2 = [profiles[(a, drug)] for a in group2 if (a, drug) in profiles]
    m1 = np.mean(vecs1, axis=0)
    m2 = np.mean(vecs2, axis=0)
    return 0.5 * np.sum(np.abs(m1 - m2))

tv_obs_sal = calc_tv(wt_animals, ki_animals, "Saline")
tv_obs_amph = calc_tv(wt_animals, ki_animals, "Amphetamine")

null_sal = np.zeros(N_partitions)
null_amph = np.zeros(N_partitions)
all_set = set(all_animals)
for p_idx, p_wt in enumerate(partitions):
    p_ki = list(all_set - set(p_wt))
    null_sal[p_idx] = calc_tv(p_wt, p_ki, "Saline")
    null_amph[p_idx] = calc_tv(p_wt, p_ki, "Amphetamine")

p_sal = float(np.mean(null_sal >= tv_obs_sal))
p_amph = float(np.mean(null_amph >= tv_obs_amph))

# Matplotlib styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#2D3748",
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 10.5,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10.5,
})

def draw_bottom_permutation_panels(fig, gs_row):
    n_bins = 30

    # Saline
    ax_b1 = fig.add_subplot(gs_row[0])
    counts, bins, patches_s = ax_b1.hist(null_sal, bins=n_bins, density=True, color="#CBD5E0",
                                         edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)
    for b_idx in range(len(patches_s)):
        if bins[b_idx] >= tv_obs_sal:
            patches_s[b_idx].set_facecolor("#ED8936")
            patches_s[b_idx].set_alpha(0.95)
    ax_b1.axvline(tv_obs_sal, color="#C05621", linestyle="--", linewidth=2.6, zorder=4,
                  label=f"Observed TV = {tv_obs_sal:.3f}\nExact p = {p_sal:.4f} *")
    ax_b1.set_xlabel("Total Variation Distance (TV)", fontweight="bold", fontsize=11.5)
    ax_b1.set_ylabel("Permutation Density", fontweight="bold", fontsize=11.5)
    ax_b1.set_title("Saline (Post-Injection): WT vs. DAT-KI Separation\nExact Animal-Permutation Test (N=6,435 partitions)",
                    loc="left", fontweight="bold", fontsize=12, pad=8)
    ax_b1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax_b1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E0", fontsize=10.5)
    ax_b1.set_ylim(0, 11.0)

    # Amphetamine
    ax_b2 = fig.add_subplot(gs_row[1])
    counts_a, bins_a, patches_a = ax_b2.hist(null_amph, bins=n_bins, density=True, color="#CBD5E0",
                                             edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)
    for b_idx in range(len(patches_a)):
        if bins_a[b_idx] >= tv_obs_amph:
            patches_a[b_idx].set_facecolor("#E53E3E")
            patches_a[b_idx].set_alpha(0.95)
    ax_b2.axvline(tv_obs_amph, color="#9B2C2C", linestyle="--", linewidth=2.6, zorder=4,
                  label=f"Observed TV = {tv_obs_amph:.3f}\nExact p = {p_amph:.4f} **")
    ax_b2.set_xlabel("Total Variation Distance (TV)", fontweight="bold", fontsize=11.5)
    ax_b2.set_ylabel("Permutation Density", fontweight="bold", fontsize=11.5)
    ax_b2.set_title("Amphetamine (Post-Injection): WT vs. DAT-KI Separation\nExact Animal-Permutation Test (N=6,435 partitions)",
                    loc="left", fontweight="bold", fontsize=12, pad=8)
    ax_b2.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax_b2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E0", fontsize=10.5)
    ax_b2.set_ylim(0, 11.0)

# =============================================================================
# 1. FIG: DURATION + 2 PERMUTATION TESTS (fig_level3_slide_duration_vertical.png)
# =============================================================================
fig_dur = plt.figure(figsize=(12.0, 12.2), dpi=300)
gs_dur = gridspec.GridSpec(2, 2, figure=fig_dur, height_ratios=[1.35, 0.95], hspace=0.46, wspace=0.22)

ax_top_d = fig_dur.add_subplot(gs_dur[0, :])
chain_indices = np.arange(1, K + 1)
bar_w = 0.185
offsets = {
    "WT_Saline": -1.65 * bar_w,
    "KI_Saline": -0.65 * bar_w,
    "WT_Amphetamine": 0.65 * bar_w,
    "KI_Amphetamine": 1.65 * bar_w,
}

for cohort in COHORT_ORDER:
    sub = df_dur_summary[df_dur_summary["cohort"] == cohort].sort_values("factor_id")
    means = sub["mean"].values
    sems = sub["sem"].values
    x_pos = chain_indices + offsets[cohort]
    ax_top_d.bar(x_pos, means, yerr=sems, width=bar_w, color=COHORT_COLORS[cohort],
                 capsize=2.8, edgecolor="#1A202C", linewidth=0.9, alpha=0.92,
                 label=COHORT_LABELS[cohort], zorder=3)

for k in chain_indices:
    if k % 2 == 0:
        ax_top_d.axvspan(k - 0.48, k + 0.48, color="#F7FAFC", alpha=0.8, zorder=0)

max_star_y = 0.0
scale_factor = 750.0
y_step = 0.04 * scale_factor
h_tick = 0.012 * scale_factor

for k in range(1, K + 1):
    sub_k = df_dur_summary[df_dur_summary["factor_id"] == k]
    wt_sal_row = sub_k[sub_k["cohort"] == "WT_Saline"].iloc[0]
    ki_sal_row = sub_k[sub_k["cohort"] == "KI_Saline"].iloc[0]
    sal_max = max(wt_sal_row["mean"] + wt_sal_row["sem"], ki_sal_row["mean"] + ki_sal_row["sem"])

    wt_amph_row = sub_k[sub_k["cohort"] == "WT_Amphetamine"].iloc[0]
    ki_amph_row = sub_k[sub_k["cohort"] == "KI_Amphetamine"].iloc[0]
    amph_max = max(wt_amph_row["mean"] + wt_amph_row["sem"], ki_amph_row["mean"] + ki_amph_row["sem"])

    p_sal_k = dur_pvals.get((k, "Saline"), 1.0)
    p_amph_k = dur_pvals.get((k, "Amphetamine"), 1.0)

    if p_sal_k < 0.05:
        star_sal = "***" if p_sal_k < 0.001 else ("**" if p_sal_k < 0.01 else "*")
        x1 = k + offsets["WT_Saline"]
        x2 = k + offsets["KI_Saline"]
        y_b = sal_max + y_step
        ax_top_d.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.1, zorder=5)
        star_pos_y = y_b + h_tick + 0.008 * scale_factor
        ax_top_d.text((x1 + x2) / 2, star_pos_y, star_sal,
                      ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#C05621", zorder=5)
        max_star_y = max(max_star_y, star_pos_y + 0.04 * scale_factor)

    if p_amph_k < 0.05:
        star_amph = "***" if p_amph_k < 0.001 else ("**" if p_amph_k < 0.01 else "*")
        x1 = k + offsets["WT_Amphetamine"]
        x2 = k + offsets["KI_Amphetamine"]
        y_b = amph_max + y_step
        ax_top_d.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.1, zorder=5)
        star_pos_y = y_b + h_tick + 0.008 * scale_factor
        ax_top_d.text((x1 + x2) / 2, star_pos_y, star_amph,
                      ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#9B2C2C", zorder=5)
        max_star_y = max(max_star_y, star_pos_y + 0.04 * scale_factor)

ax_top_d.set_xticks(chain_indices)
ax_top_d.set_xticklabels([VERTICAL_CHAIN_LABELS[k] for k in chain_indices], fontweight="bold", linespacing=1.2)
ax_top_d.set_ylabel("Post-Injection Duration (seconds)", fontweight="bold", fontsize=12)
ax_top_d.set_title("Total Post-Injection Time Spent in Discovered Chains (Mean ± SEM)\n* p < 0.05, ** p < 0.01 (Mann-Whitney U Test comparing WT vs. DAT-KI within condition)",
                   loc="left", fontweight="bold", fontsize=13, pad=12)
ax_top_d.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax_top_d.legend(bbox_to_anchor=(0.5, 1.25), loc="upper center", ncol=4,
                frameon=True, facecolor="white", edgecolor="#CBD5E0", framealpha=0.95, fontsize=11)
ax_top_d.set_ylim(0, max(820.0, max_star_y * 1.08))

draw_bottom_permutation_panels(fig_dur, [gs_dur[1, 0], gs_dur[1, 1]])

fig_dur.suptitle("Level 3 Behavioral Chains: Post-Injection Duration & Genotype Separation",
                 fontweight="bold", fontsize=14.5, y=0.99)

for d in [PRESENTATION_DIR, FIGURES_DIR, ARTIFACT_DIR]:
    fig_dur.savefig(d / "fig_level3_slide_duration_vertical.png", dpi=300, bbox_inches="tight")
plt.close(fig_dur)
print("Saved fig_level3_slide_duration_vertical.png successfully.")

# =============================================================================
# 2. FIG: SELF-TRANSITIONS + 4 MATRICES (fig_level3_slide_self_transitions.png)
# =============================================================================
fig_self = plt.figure(figsize=(12.0, 12.2), dpi=300)
gs_self = gridspec.GridSpec(2, 4, figure=fig_self, height_ratios=[1.35, 0.95], hspace=0.46, wspace=0.32)

ax_top_s = fig_self.add_subplot(gs_self[0, :])

for cohort in COHORT_ORDER:
    sub = df_self_summary[df_self_summary["cohort"] == cohort].sort_values("factor_id")
    means = sub["mean"].values
    sems = sub["sem"].values
    x_pos = chain_indices + offsets[cohort]
    ax_top_s.bar(x_pos, means, yerr=sems, width=bar_w, color=COHORT_COLORS[cohort],
                 capsize=2.8, edgecolor="#1A202C", linewidth=0.9, alpha=0.92,
                 label=COHORT_LABELS[cohort], zorder=3)

for k in chain_indices:
    if k % 2 == 0:
        ax_top_s.axvspan(k - 0.48, k + 0.48, color="#F7FAFC", alpha=0.8, zorder=0)

max_star_y = 0.0
scale_factor = 0.65
y_step = 0.04 * scale_factor
h_tick = 0.012 * scale_factor

for k in range(1, K + 1):
    sub_k = df_self_summary[df_self_summary["factor_id"] == k]
    wt_sal_row = sub_k[sub_k["cohort"] == "WT_Saline"].iloc[0]
    ki_sal_row = sub_k[sub_k["cohort"] == "KI_Saline"].iloc[0]
    sal_max = max(wt_sal_row["mean"] + wt_sal_row["sem"], ki_sal_row["mean"] + ki_sal_row["sem"])

    wt_amph_row = sub_k[sub_k["cohort"] == "WT_Amphetamine"].iloc[0]
    ki_amph_row = sub_k[sub_k["cohort"] == "KI_Amphetamine"].iloc[0]
    amph_max = max(wt_amph_row["mean"] + wt_amph_row["sem"], ki_amph_row["mean"] + ki_amph_row["sem"])

    p_sal_k = self_pvals.get((k, "Saline"), 1.0)
    p_amph_k = self_pvals.get((k, "Amphetamine"), 1.0)

    if p_sal_k < 0.05:
        star_sal = "***" if p_sal_k < 0.001 else ("**" if p_sal_k < 0.01 else "*")
        x1 = k + offsets["WT_Saline"]
        x2 = k + offsets["KI_Saline"]
        y_b = sal_max + y_step
        ax_top_s.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.1, zorder=5)
        star_pos_y = y_b + h_tick + 0.008 * scale_factor
        ax_top_s.text((x1 + x2) / 2, star_pos_y, star_sal,
                      ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#C05621", zorder=5)
        max_star_y = max(max_star_y, star_pos_y + 0.04 * scale_factor)

    if p_amph_k < 0.05:
        star_amph = "***" if p_amph_k < 0.001 else ("**" if p_amph_k < 0.01 else "*")
        x1 = k + offsets["WT_Amphetamine"]
        x2 = k + offsets["KI_Amphetamine"]
        y_b = amph_max + y_step
        ax_top_s.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.1, zorder=5)
        star_pos_y = y_b + h_tick + 0.008 * scale_factor
        ax_top_s.text((x1 + x2) / 2, star_pos_y, star_amph,
                      ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#9B2C2C", zorder=5)
        max_star_y = max(max_star_y, star_pos_y + 0.04 * scale_factor)

ax_top_s.set_xticks(chain_indices)
ax_top_s.set_xticklabels([VERTICAL_CHAIN_LABELS[k] for k in chain_indices], fontweight="bold", linespacing=1.2)
ax_top_s.set_ylabel("Post-Injection Self-Transition P(k → k)", fontweight="bold", fontsize=12)
ax_top_s.set_title("Post-Injection Chain Self-Repetition Propensity (Mean ± SEM)\n* p < 0.05, ** p < 0.01 (Mann-Whitney U Test comparing WT vs. DAT-KI within condition)",
                   loc="left", fontweight="bold", fontsize=13, pad=12)
ax_top_s.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax_top_s.legend(bbox_to_anchor=(0.5, 1.25), loc="upper center", ncol=4,
                frameon=True, facecolor="white", edgecolor="#CBD5E0", framealpha=0.95, fontsize=11)
ax_top_s.set_ylim(0, max(0.65, max_star_y * 1.08))

for c_idx, cohort in enumerate(COHORT_ORDER):
    ax_mat = fig_self.add_subplot(gs_self[1, c_idx])
    t_mat = norm_cohort_trans[cohort]
    im = ax_mat.imshow(t_mat, cmap="viridis", vmin=0, vmax=0.45, aspect="equal")
    ax_mat.set_title(f"{COHORT_LABELS[cohort]}\nTransition Matrix", fontweight="bold", fontsize=11.5, pad=8)
    ax_mat.set_xticks(np.arange(0, K, 2))
    ax_mat.set_xticklabels([f"C{k+1}" for k in range(0, K, 2)], fontsize=9)
    ax_mat.set_yticks(np.arange(0, K, 2))
    ax_mat.set_yticklabels([f"C{k+1}" for k in range(0, K, 2)], fontsize=9)
    for k in range(K):
        rect = patches.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                 edgecolor="#ED8936" if "KI" in cohort else "#63B3ED",
                                 linewidth=1.3, linestyle="-", zorder=5)
        ax_mat.add_patch(rect)
    if c_idx == 0:
        ax_mat.set_ylabel("Current Chain $k_t$", fontweight="bold", fontsize=11)
    ax_mat.set_xlabel("Next Chain $k_{t+1}$", fontweight="bold", fontsize=11)
    if c_idx == 3:
        cbar = fig_self.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.07)
        cbar.set_label("Transition Probability", fontweight="bold", fontsize=10)

fig_self.suptitle("Level 3 Behavioral Chains: Post-Injection Self-Transitions & Sequential Organization",
                  fontweight="bold", fontsize=14.5, y=0.99)

for d in [PRESENTATION_DIR, FIGURES_DIR, ARTIFACT_DIR]:
    fig_self.savefig(d / "fig_level3_slide_self_transitions.png", dpi=300, bbox_inches="tight")
plt.close(fig_self)
print("Saved fig_level3_slide_self_transitions.png successfully.")

# =============================================================================
# 3. FIG: SELF-TRANSITIONS + 2 PERMUTATIONS (fig_level3_slide_self_transitions_vertical.png)
# =============================================================================
fig_self_v = plt.figure(figsize=(12.0, 12.2), dpi=300)
gs_self_v = gridspec.GridSpec(2, 2, figure=fig_self_v, height_ratios=[1.35, 0.95], hspace=0.46, wspace=0.22)

ax_top_sv = fig_self_v.add_subplot(gs_self_v[0, :])

for cohort in COHORT_ORDER:
    sub = df_self_summary[df_self_summary["cohort"] == cohort].sort_values("factor_id")
    means = sub["mean"].values
    sems = sub["sem"].values
    x_pos = chain_indices + offsets[cohort]
    ax_top_sv.bar(x_pos, means, yerr=sems, width=bar_w, color=COHORT_COLORS[cohort],
                  capsize=2.8, edgecolor="#1A202C", linewidth=0.9, alpha=0.92,
                  label=COHORT_LABELS[cohort], zorder=3)

for k in chain_indices:
    if k % 2 == 0:
        ax_top_sv.axvspan(k - 0.48, k + 0.48, color="#F7FAFC", alpha=0.8, zorder=0)

max_star_y = 0.0
for k in range(1, K + 1):
    sub_k = df_self_summary[df_self_summary["factor_id"] == k]
    wt_sal_row = sub_k[sub_k["cohort"] == "WT_Saline"].iloc[0]
    ki_sal_row = sub_k[sub_k["cohort"] == "KI_Saline"].iloc[0]
    sal_max = max(wt_sal_row["mean"] + wt_sal_row["sem"], ki_sal_row["mean"] + ki_sal_row["sem"])

    wt_amph_row = sub_k[sub_k["cohort"] == "WT_Amphetamine"].iloc[0]
    ki_amph_row = sub_k[sub_k["cohort"] == "KI_Amphetamine"].iloc[0]
    amph_max = max(wt_amph_row["mean"] + wt_amph_row["sem"], ki_amph_row["mean"] + ki_amph_row["sem"])

    p_sal_k = self_pvals.get((k, "Saline"), 1.0)
    p_amph_k = self_pvals.get((k, "Amphetamine"), 1.0)

    if p_sal_k < 0.05:
        star_sal = "***" if p_sal_k < 0.001 else ("**" if p_sal_k < 0.01 else "*")
        x1 = k + offsets["WT_Saline"]
        x2 = k + offsets["KI_Saline"]
        y_b = sal_max + y_step
        ax_top_sv.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.1, zorder=5)
        star_pos_y = y_b + h_tick + 0.008 * scale_factor
        ax_top_sv.text((x1 + x2) / 2, star_pos_y, star_sal,
                       ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#C05621", zorder=5)
        max_star_y = max(max_star_y, star_pos_y + 0.04 * scale_factor)

    if p_amph_k < 0.05:
        star_amph = "***" if p_amph_k < 0.001 else ("**" if p_amph_k < 0.01 else "*")
        x1 = k + offsets["WT_Amphetamine"]
        x2 = k + offsets["KI_Amphetamine"]
        y_b = amph_max + y_step
        ax_top_sv.plot([x1, x1, x2, x2], [y_b, y_b + h_tick, y_b + h_tick, y_b], color="#2D3748", lw=1.1, zorder=5)
        star_pos_y = y_b + h_tick + 0.008 * scale_factor
        ax_top_sv.text((x1 + x2) / 2, star_pos_y, star_amph,
                       ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#9B2C2C", zorder=5)
        max_star_y = max(max_star_y, star_pos_y + 0.04 * scale_factor)

ax_top_sv.set_xticks(chain_indices)
ax_top_sv.set_xticklabels([VERTICAL_CHAIN_LABELS[k] for k in chain_indices], fontweight="bold", linespacing=1.2)
ax_top_sv.set_ylabel("Post-Injection Self-Transition P(k → k)", fontweight="bold", fontsize=12)
ax_top_sv.set_title("Post-Injection Chain Self-Repetition Propensity (Mean ± SEM)\n* p < 0.05, ** p < 0.01 (Mann-Whitney U Test comparing WT vs. DAT-KI within condition)",
                    loc="left", fontweight="bold", fontsize=13, pad=12)
ax_top_sv.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax_top_sv.legend(bbox_to_anchor=(0.5, 1.25), loc="upper center", ncol=4,
                 frameon=True, facecolor="white", edgecolor="#CBD5E0", framealpha=0.95, fontsize=11)
ax_top_sv.set_ylim(0, max(0.65, max_star_y * 1.08))

draw_bottom_permutation_panels(fig_self_v, [gs_self_v[1, 0], gs_self_v[1, 1]])

fig_self_v.suptitle("Level 3 Behavioral Chains: Post-Injection Self-Transitions & Genotype Separation",
                    fontweight="bold", fontsize=14.5, y=0.99)

for d in [PRESENTATION_DIR, FIGURES_DIR, ARTIFACT_DIR]:
    fig_self_v.savefig(d / "fig_level3_slide_self_transitions_vertical.png", dpi=300, bbox_inches="tight")
plt.close(fig_self_v)
print("Saved fig_level3_slide_self_transitions_vertical.png successfully.")

print("\nALL 3 SLIDE FIGURES RE-RENDERED SUCCESSFULLY WITH ZERO ANNOTATIONS/ARROWS!")
