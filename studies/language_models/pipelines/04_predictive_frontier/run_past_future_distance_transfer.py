#!/usr/bin/env python3
"""
Past-to-Future Distance Transfer & Geometry Benchmark
=====================================================
Scientific Objective:
Evaluates the continuous relationship between past-state behavioral difference
d_X(i, j) and future-state behavioral divergence d_Y(i, j) = D_JS(Y_i, Y_j).

Analyses:
1. E[d_Y | d_X]: Mean future divergence as a function of past distance.
2. 4-Regime Comparison:
   - Similar Pasts (k-NN cross-session matching)
   - Random Pasts (Unconditional reference baseline)
   - Dissimilar Pasts (Deliberately far-away behavioral states)
   - Shuffled Control (Permuted past-future correspondence)
3. 2D Phase Diagram / Heatmap:
   - Quantifies the 4 ethological quadrants (Predictability, Branching/Uncertainty,
     Convergence/Equifinality, Divergence).
4. Model Geometry Preservation:
   - Evaluates whether the trained predictive model reproduces E[d_Y_hat | d_X].
"""

import os
import pickle
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

LOG_FILE = os.path.join(LM_DIR, "output", "past_future_distance_transfer.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

with open(LOG_FILE, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INITIALIZED PAST-TO-FUTURE DISTANCE TRANSFER BENCHMARK\n")

cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
log(f"Loading continuous bout cache: {cache_path}")
with open(cache_path, "rb") as f:
    data = pickle.load(f)
session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

raw_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
with open(raw_path, "rb") as f:
    raw_sessions = pickle.load(f)

session_meta = []
for i, s in enumerate(raw_sessions):
    m = getattr(s, "metadata", {})
    session_meta.append({
        "sess_idx": i,
        "inj_frame": m.get("injection_frame", 0),
    })

FPS = 80.025
W_FRAMES = int(68.3 * FPS)
H_FRAMES = int(120.0 * FPS)
STRIDE_FRAMES = int(5.0 * FPS)
NUM_ACTIVE_SYLLS = 99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log("Extracting clean windows with direct bout onsets...")
sessions_data = []
for sess_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    N = bouts[-1]["end_frame"] + 1
    fs = np.full(N, -1, dtype=np.int32)
    sp = np.zeros((N, 6), dtype=np.float32)
    oh_bout = np.zeros((N, NUM_ACTIVE_SYLLS), dtype=np.float32)
    next_bout_ptrs = np.full(N, N, dtype=np.int32)

    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = b["syllable"] - 1
        if 0 <= s_id < NUM_ACTIVE_SYLLS:
            fs[sf:ef+1] = s_id
            oh_bout[sf, s_id] = 1.0
        vec = b.get("vector", None)
        if vec is not None and len(vec) >= 14:
            sp[sf:ef+1, :4] = vec[:4]; sp[sf:ef+1, 4] = vec[10]; sp[sf:ef+1, 5] = vec[12]

    b_starts = np.array([b["start_frame"] for b in bouts])
    for t_idx, b_st in enumerate(b_starts):
        prev_st = b_starts[t_idx-1] if t_idx > 0 else 0
        next_bout_ptrs[prev_st:b_st] = b_st
    if len(b_starts) > 0:
        next_bout_ptrs[b_starts[-1]:] = N

    cum_bout = np.vstack([np.zeros((1, NUM_ACTIVE_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])
    cum_spatial = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(sp, axis=0)])
    cum_spatial_sq = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(sp**2, axis=0)])

    sessions_data.append({
        "sess_idx": sess_idx, "N": N, "fs": fs, "cum_bout": cum_bout,
        "next_bout_ptrs": next_bout_ptrs, "cum_spatial": cum_spatial,
        "cum_spatial_sq": cum_spatial_sq, "inj_frame": session_meta[sess_idx]["inj_frame"]
    })

raw_windows = []
for s in sessions_data:
    N = s["N"]
    if N < W_FRAMES + H_FRAMES + 200: continue
    inj_f = s["inj_frame"]

    t_anchors = np.arange(W_FRAMES, N - H_FRAMES, STRIDE_FRAMES)
    t_tgt_start = s["next_bout_ptrs"][t_anchors]
    valid = (t_tgt_start + H_FRAMES < N)
    t_anchors = t_anchors[valid]
    t_tgt_start = t_tgt_start[valid]

    past_start = t_anchors - W_FRAMES
    future_end = t_tgt_start + H_FRAMES
    no_straddle = (future_end < inj_f) | (past_start > inj_f)
    t_anchors = t_anchors[no_straddle]
    t_tgt_start = t_tgt_start[no_straddle]
    M = len(t_anchors)
    if M == 0: continue

    # 1. Spatial context (14-D)
    t_norm = t_anchors / float(N)
    p1 = np.column_stack([t_norm, t_norm**2]).astype(np.float32)
    sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - W_FRAMES]
    sp_mean = sp_sum / float(W_FRAMES)
    sp_sq = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - W_FRAMES]
    sp_var = np.maximum(sp_sq / float(W_FRAMES) - sp_mean**2, 1e-6)
    feat_space = np.hstack([p1, sp_mean, np.sqrt(sp_var)]).astype(np.float32)

    # 2. Repertoire (99-D)
    b_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
    feat_rep = (b_cnt + 1e-4) / (np.sum(b_cnt, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    # 3. Target Future (99-D)
    f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
    target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    for i in range(M):
        raw_windows.append({
            "sess_idx": s["sess_idx"],
            "space": feat_space[i],
            "rep": feat_rep[i],
            "target": target[i]
        })

N_win = len(raw_windows)
log(f"Extracted {N_win:,} clean windows across {len(sessions_data)} sessions.")

sess_ids = np.array([w["sess_idx"] for w in raw_windows])
Y_targets = np.array([w["target"] for w in raw_windows], dtype=np.float32)
X_space_raw = np.array([w["space"] for w in raw_windows], dtype=np.float32)
X_rep_raw = np.array([w["rep"] for w in raw_windows], dtype=np.float32)

# Hybrid Representation
scaler_space = StandardScaler().fit(X_space_raw)
X_space_std = scaler_space.transform(X_space_raw).astype(np.float32)
X_space_norm = X_space_std / np.sqrt(X_space_std.shape[1])
X_rep_hellinger = np.sqrt(X_rep_raw)
X_hybrid = np.hstack([
    X_space_norm,
    X_rep_hellinger / np.sqrt(np.mean(np.sum(X_rep_hellinger**2, axis=1)))
]) / np.sqrt(2.0)

# Train 5-Fold Ridge Bag-of-Words Model to get Out-of-Fold Future Predictions Y_hat
log("Computing 5-fold cross-validated Ridge BoW predictions (Y_hat)...")
Y_hat_preds = np.zeros_like(Y_targets)
gkf = GroupKFold(n_splits=5)
for tr, te in gkf.split(X_hybrid, groups=sess_ids):
    p_hat = np.clip(Ridge(1.0).fit(X_hybrid[tr], Y_targets[tr]).predict(X_hybrid[te]), 1e-4, 1.0)
    p_hat /= np.sum(p_hat, axis=1, keepdims=True)
    Y_hat_preds[te] = p_hat

# Fast PyTorch Jensen-Shannon Divergence
def js_divergence_pt(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    P = torch.clamp(P, min=eps, max=1.0); Q = torch.clamp(Q, min=eps, max=1.0)
    P = P / P.sum(dim=-1, keepdim=True); Q = Q / Q.sum(dim=-1, keepdim=True)
    M = 0.5 * (P + Q)
    return 0.5 * (torch.sum(P * torch.log2(P / M), dim=-1) + torch.sum(Q * torch.log2(Q / M), dim=-1))

X_tensor = torch.tensor(X_hybrid, dtype=torch.float32, device=DEVICE)
Y_tensor = torch.tensor(Y_targets, dtype=torch.float32, device=DEVICE)
Y_hat_tensor = torch.tensor(Y_hat_preds, dtype=torch.float32, device=DEVICE)

# Sample 100,000 Cross-Session Window Pairs (i, j) with sess(i) != sess(j)
N_PAIRS = 100000
log(f"Sampling {N_PAIRS:,} cross-session pairs across the dataset...")
np.random.seed(42)

idx_i = np.random.choice(N_win, size=N_PAIRS * 2, replace=True)
idx_j = np.random.choice(N_win, size=N_PAIRS * 2, replace=True)

# Filter for strictly cross-session pairs
valid_pair = (sess_ids[idx_i] != sess_ids[idx_j])
idx_i = idx_i[valid_pair][:N_PAIRS]
idx_j = idx_j[valid_pair][:N_PAIRS]

# Shuffled control targets: randomly permute Y indices
perm_indices = np.random.permutation(N_win)
Y_shuffled = Y_tensor[perm_indices]

# Batch compute distances on GPU
BATCH_SIZE = 10000
d_X_list, d_Y_list, d_Y_hat_list, d_Y_shuff_list = [], [], [], []

for b in range(0, N_PAIRS, BATCH_SIZE):
    b_end = min(b + BATCH_SIZE, N_PAIRS)
    b_i = idx_i[b:b_end]
    b_j = idx_j[b:b_end]

    # Past distance
    diff_X = X_tensor[b_i] - X_tensor[b_j]
    d_X = torch.norm(diff_X, p=2, dim=1) # Euclidean in balanced hybrid space

    # Future divergence (Actual)
    d_Y = js_divergence_pt(Y_tensor[b_i], Y_tensor[b_j])

    # Future divergence (Model Predictions)
    d_Y_hat = js_divergence_pt(Y_hat_tensor[b_i], Y_hat_tensor[b_j])

    # Future divergence (Shuffled Control)
    d_Y_shuff = js_divergence_pt(Y_shuffled[b_i], Y_shuffled[b_j])

    d_X_list.append(d_X.cpu().numpy())
    d_Y_list.append(d_Y.cpu().numpy())
    d_Y_hat_list.append(d_Y_hat.cpu().numpy())
    d_Y_shuff_list.append(d_Y_shuff.cpu().numpy())

d_X_arr = np.concatenate(d_X_list)
d_Y_arr = np.concatenate(d_Y_list)
d_Y_hat_arr = np.concatenate(d_Y_hat_list)
d_Y_shuff_arr = np.concatenate(d_Y_shuff_list)

log(f"Computed {len(d_X_arr):,} cross-session pair distances.")

# Compute 4-Regime Statistics
# 1. Similar Pasts (Smallest 5% of d_X)
sim_mask = d_X_arr <= np.percentile(d_X_arr, 5)
d_Y_similar = float(np.mean(d_Y_arr[sim_mask]))
ci_similar = float(1.96 * np.std(d_Y_arr[sim_mask]) / np.sqrt(np.sum(sim_mask)))

# 2. Random Pasts (Unconditional reference level)
d_Y_random = float(np.mean(d_Y_arr))
ci_random = float(1.96 * np.std(d_Y_arr) / np.sqrt(len(d_Y_arr)))

# 3. Dissimilar Pasts (Largest 5% of d_X)
dissim_mask = d_X_arr >= np.percentile(d_X_arr, 95)
d_Y_dissimilar = float(np.mean(d_Y_arr[dissim_mask]))
ci_dissimilar = float(1.96 * np.std(d_Y_arr[dissim_mask]) / np.sqrt(np.sum(dissim_mask)))

# 4. Shuffled Control Level
d_Y_shuff_mean = float(np.mean(d_Y_shuff_arr))

constraint_pct = ((d_Y_random - d_Y_similar) / d_Y_random) * 100.0

log("\n" + "=" * 70)
log("FOUR-REGIME FUTURE DIVERGENCE COMPARISON:")
log("=" * 70)
log(f"  1. Similar Pasts (Top 5% closest):     d_Y = {d_Y_similar:.4f} ± {ci_similar:.4f} bits")
log(f"  2. Random Cross-Session Pairs:         d_Y = {d_Y_random:.4f} ± {ci_random:.4f} bits")
log(f"  3. Dissimilar Pasts (Top 5% farthest): d_Y = {d_Y_dissimilar:.4f} ± {ci_dissimilar:.4f} bits")
log(f"  4. Shuffled Control:                   d_Y = {d_Y_shuff_mean:.4f} bits (Flat Null)")
log(f"  --> Information Constraint Ratio:      +{constraint_pct:.2f}% dispersion removed by similar pasts")
log("=" * 70)

# Discretize d_X into 25 Bins to compute the continuous Transfer Curve E[d_Y | d_X]
bins = np.linspace(np.percentile(d_X_arr, 0.5), np.percentile(d_X_arr, 99.5), 26)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

mean_d_Y_bin = []
ci_d_Y_bin = []
mean_d_Y_hat_bin = []
mean_d_Y_shuff_bin = []

for b_idx in range(len(bins) - 1):
    m_bin = (d_X_arr >= bins[b_idx]) & (d_X_arr < bins[b_idx + 1])
    if np.sum(m_bin) > 10:
        sub_d_Y = d_Y_arr[m_bin]
        mean_d_Y_bin.append(np.mean(sub_d_Y))
        ci_d_Y_bin.append(1.96 * np.std(sub_d_Y) / np.sqrt(len(sub_d_Y)))
        mean_d_Y_hat_bin.append(np.mean(d_Y_hat_arr[m_bin]))
        mean_d_Y_shuff_bin.append(np.mean(d_Y_shuff_arr[m_bin]))
    else:
        mean_d_Y_bin.append(np.nan)
        ci_d_Y_bin.append(np.nan)
        mean_d_Y_hat_bin.append(np.nan)
        mean_d_Y_shuff_bin.append(np.nan)

mean_d_Y_bin = np.array(mean_d_Y_bin)
ci_d_Y_bin = np.array(ci_d_Y_bin)
mean_d_Y_hat_bin = np.array(mean_d_Y_hat_bin)
mean_d_Y_shuff_bin = np.array(mean_d_Y_shuff_bin)

# Quadrant Analysis (Median splits on d_X and d_Y)
med_X = np.median(d_X_arr)
med_Y = np.median(d_Y_arr)

q1 = np.mean((d_X_arr < med_X) & (d_Y_arr < med_Y)) * 100.0  # Similar past -> Similar future
q2 = np.mean((d_X_arr < med_X) & (d_Y_arr >= med_Y)) * 100.0 # Similar past -> Divergent future (Uncertainty)
q3 = np.mean((d_X_arr >= med_X) & (d_Y_arr < med_Y)) * 100.0 # Dissimilar past -> Similar future (Equifinality/Convergence!)
q4 = np.mean((d_X_arr >= med_X) & (d_Y_arr >= med_Y)) * 100.0# Dissimilar past -> Divergent future

log("\n" + "=" * 70)
log("ETHOLOGICAL QUADRANTS DISTRIBUTION:")
log(f"  Q1 (Predictable: Similar Past -> Similar Future):       {q1:.1f}%")
log(f"  Q2 (Uncertainty: Similar Past -> Divergent Future):     {q2:.1f}%")
log(f"  Q3 (Equifinality/Convergence: Distant Past -> Similar): {q3:.1f}%")
log(f"  Q4 (Divergence: Distant Past -> Distant Future):        {q4:.1f}%")
log("=" * 70)

# Save Transfer Curve Data Table
df_transfer = pd.DataFrame({
    "Bin_Center_Past_Distance": bin_centers,
    "Mean_Actual_Future_Divergence": mean_d_Y_bin,
    "CI_95_Actual": ci_d_Y_bin,
    "Mean_Model_Future_Divergence": mean_d_Y_hat_bin,
    "Mean_Shuffled_Future_Divergence": mean_d_Y_shuff_bin
})
csv_transfer_path = os.path.join(LM_DIR, "output", "04_predictive_frontier", "tables", "past_future_transfer_curve.csv")
df_transfer.to_csv(csv_transfer_path, index=False)
log(f"Saved transfer curve to: {csv_transfer_path}")

# Render Publication 3-Panel Figure
fig = plt.figure(figsize=(16, 5.2), dpi=300)
gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 0.85], wspace=0.28)

# Panel 1: E[d_Y | d_X] Transfer Curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(bin_centers, mean_d_Y_bin, color='#1f77b4', marker='o', linewidth=2.4, markersize=6, label='Empirical Animal Futures $\\mathbb{E}[d_Y \\mid d_X]$')
ax1.fill_between(bin_centers, mean_d_Y_bin - ci_d_Y_bin, mean_d_Y_bin + ci_d_Y_bin, color='#1f77b4', alpha=0.15)
ax1.plot(bin_centers, mean_d_Y_hat_bin, color='#2ca02c', linestyle='--', linewidth=2.0, label='Model Predictions $\\mathbb{E}[d_{\\hat{Y}} \\mid d_X]$')
ax1.plot(bin_centers, mean_d_Y_shuff_bin, color='#d62728', linestyle=':', linewidth=2.0, label='Shuffled Control (Permuted $X \\leftrightarrow Y$)')

ax1.axhline(d_Y_random, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label=f'Random Reference ({d_Y_random:.3f} bits)')
ax1.set_title("A. Past-to-Future Distance Transfer Curve\n(How Past Similarity Constrains Future Divergence)", fontsize=10.5, fontweight='bold', pad=10)
ax1.set_xlabel("Past Behavioral Distance $d_X(i, j)$", fontsize=10)
ax1.set_ylabel("Future Jensen-Shannon Divergence $d_Y(i, j)$ (bits)", fontsize=10)
ax1.legend(loc='lower right', frameon=True, fontsize=8)
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel 2: 2D Density Heatmap & 4 Quadrants
ax2 = fig.add_subplot(gs[0, 1])
# Subsample 15,000 points for clear hexbin density
sub_idx = np.random.choice(len(d_X_arr), size=20000, replace=False)
hb = ax2.hexbin(d_X_arr[sub_idx], d_Y_arr[sub_idx], gridsize=40, cmap='Blues', mincnt=1, bins='log')
cb = plt.colorbar(hb, ax=ax2)
cb.set_label("Log Pair Density", fontsize=9.5)

ax2.axvline(med_X, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
ax2.axhline(med_Y, color='red', linestyle='--', linewidth=1.2, alpha=0.7)

ax2.text(med_X * 0.75, med_Y * 0.5, f"Predictable\n({q1:.1f}%)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#08519c')
ax2.text(med_X * 0.75, med_Y * 1.5, f"Uncertainty\n({q2:.1f}%)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#08519c')
ax2.text(med_X * 1.25, med_Y * 0.5, f"Convergence\n({q3:.1f}%)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#08519c')
ax2.text(med_X * 1.25, med_Y * 1.5, f"Divergence\n({q4:.1f}%)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#08519c')

ax2.set_title("B. Joint Distance Phase Diagram $(d_X, d_Y)$\n[Revealing Equifinality & Branching Dynamics]", fontsize=10.5, fontweight='bold', pad=10)
ax2.set_xlabel("Past Behavioral Distance $d_X$", fontsize=10)
ax2.set_ylabel("Future Divergence $d_Y$ (bits)", fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.4)

# Panel 3: 4-Regime Comparison Bar Chart
ax3 = fig.add_subplot(gs[0, 2])
regimes = ["Similar Pasts\n(Top 5%)", "Random Pairs\n(Reference)", "Dissimilar\n(Top 5%)", "Shuffled\nControl"]
divergences = [d_Y_similar, d_Y_random, d_Y_dissimilar, d_Y_shuff_mean]
colors = ["#2ca02c", "#7f7f7f", "#9467bd", "#d62728"]

bars = ax3.bar(regimes, divergences, color=colors, alpha=0.85, width=0.55)
ax3.set_title("C. Four-Regime Comparison\n(Empirical Constraint)", fontsize=10.5, fontweight='bold', pad=10)
ax3.set_ylabel("Future Divergence (bits, $D_{JS}$)", fontsize=10)
ax3.set_xticks(range(len(regimes)))
ax3.set_xticklabels(regimes, rotation=15, ha='right', fontsize=8.5)
ax3.grid(axis='y', linestyle='--', alpha=0.5)

for bar, d_val in zip(bars, divergences):
    ax3.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.008, f"{d_val:.3f} b", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add bracket showing % constraint
ax3.annotate(f"-{constraint_pct:.1f}% Constraint", xy=(0.5, (d_Y_similar + d_Y_random)/2.0), xytext=(0.5, d_Y_random * 1.15),
             arrowprops=dict(arrowstyle="->", color="black", lw=1.2), ha='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
fig_out_path = os.path.join(LM_DIR, "output", "presentation", "figures", "fig9_past_future_distance_transfer.png")
fig.savefig(fig_out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
log(f"Saved Figure 9 to: {fig_out_path}")
log("BENCHMARK COMPLETE!")
