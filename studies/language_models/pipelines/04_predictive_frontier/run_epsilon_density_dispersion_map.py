#!/usr/bin/env python3
"""
Epsilon-Radius Conditional Dispersion & Density Quadrant Mapping
================================================================
Scientific Objective:
1. Decouples past similarity (epsilon) from local sample density n_i(epsilon).
2. Computes the empirical conditional dispersion D_i(epsilon) around the local JS barycenter.
3. Subtracts the bout-aware finite-window floor (R_window = 0.1376 bits at H=120s)
   to obtain the true excess conditional future unpredictability.
4. Maps every behavioral state X_i into the 2D plane:
      [Local Density n_i(epsilon)] x [Excess Dispersion D_i(epsilon) - R_window]
   isolating the High-Density + High-Dispersion "Neural Target" Branch Points.
"""

import os
import sys
import time

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation"))

import pickle

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

FIG_DIR = os.path.join(LM_DIR, "output", "presentation", "figures")
TABLE_DIR = os.path.join(LM_DIR, "output", "04_predictive_frontier", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

LOG_FILE = os.path.join(LM_DIR, "output", "epsilon_density_dispersion_map.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

with open(LOG_FILE, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INITIALIZED EPSILON DENSITY-DISPERSION QUADRANT MAPPER\n")

# Load Finite-Window Noise Floor from Step 1
R_WINDOW_120S = 0.13761 # Bout bootstrap floor from Step 1

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
    gt = m.get("genotype", "WT")
    drug = m.get("drug", "Saline")
    session_meta.append({
        "sess_idx": i,
        "cohort": f"{gt}_{drug}",
        "drug": drug,
        "genotype": gt,
        "inj_frame": m.get("injection_frame", 0),
        "entry_frame": m.get("entry_frame", 0)
    })

FPS = 80.025
W_SEC = 68.3
W_FRAMES = int(W_SEC * FPS)
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
NUM_ACTIVE_SYLLS = 99  # Syllables 1..99 mapped to 0..98

log("Preprocessing session timecourses with direct discrete bout onsets...")
sessions_data = []
for sess_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    N = bouts[-1]["end_frame"] + 1
    fs = np.full(N, -1, dtype=np.int32)
    sp = np.zeros((N, 6), dtype=np.float32)
    pose = np.zeros((N, 30), dtype=np.float32)
    oh_bout = np.zeros((N, NUM_ACTIVE_SYLLS), dtype=np.float32)
    next_bout_ptrs = np.full(N, N, dtype=np.int32)

    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_raw = b["syllable"]
        s_id = s_raw - 1 # 1..99 -> 0..98
        if 0 <= s_id < NUM_ACTIVE_SYLLS:
            fs[sf:ef+1] = s_id
            oh_bout[sf, s_id] = 1.0

        vec = b.get("vector", None)
        if vec is not None and len(vec) >= 14:
            sp[sf:ef+1, :4] = vec[:4]
            sp[sf:ef+1, 4] = vec[10]
            sp[sf:ef+1, 5] = vec[12]
            if len(vec) >= 30:
                pose[sf:ef+1] = vec[:30]
            else:
                pose[sf:ef+1, :len(vec)] = vec

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
        "sess_idx": sess_idx, "N": N, "fs": fs, "pose": pose, "cum_bout": cum_bout,
        "next_bout_ptrs": next_bout_ptrs, "cum_spatial": cum_spatial,
        "cum_spatial_sq": cum_spatial_sq, "meta": session_meta[sess_idx]
    })

raw_windows = []
for s in sessions_data:
    N = s["N"]
    if N < W_FRAMES + H_FRAMES + 200: continue
    inj_f = s["meta"]["inj_frame"]

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

    # 2. Repertoire probability distribution (99-D)
    b_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
    feat_rep = (b_cnt + 1e-4) / (np.sum(b_cnt, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    # 3. Future Target distribution over H=120s (99-D)
    f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
    target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    for i in range(M):
        raw_windows.append({
            "sess_idx": s["sess_idx"],
            "t_sec": t_anchors[i] / FPS,
            "space": feat_space[i],
            "rep": feat_rep[i],
            "target": target[i],
            "cohort": s["meta"]["cohort"],
            "drug": s["meta"]["drug"]
        })

N_win = len(raw_windows)
log(f"Extracted {N_win:,} clean windows across {len(sessions_data)} sessions.")

sess_ids = np.array([w["sess_idx"] for w in raw_windows])
Y_targets = np.array([w["target"] for w in raw_windows], dtype=np.float32)
X_space_raw = np.array([w["space"] for w in raw_windows], dtype=np.float32)
X_rep_raw = np.array([w["rep"] for w in raw_windows], dtype=np.float32)

# Hybrid representation: Space + Repertoire (Hellinger)
scaler_space = StandardScaler().fit(X_space_raw)
X_space_std = scaler_space.transform(X_space_raw).astype(np.float32)
X_space_norm = X_space_std / np.sqrt(X_space_std.shape[1])

X_rep_hellinger = np.sqrt(X_rep_raw)
X_rep_norm = X_rep_hellinger / np.sqrt(np.mean(np.sum(X_rep_hellinger**2, axis=1)))
X_hybrid = np.hstack([X_space_norm, X_rep_norm]) / np.sqrt(2.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Running GPU-accelerated epsilon-radius dispersion engine on: {device}")

def js_divergence_pt(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    P = torch.clamp(P, min=eps, max=1.0)
    Q = torch.clamp(Q, min=eps, max=1.0)
    P = P / P.sum(dim=-1, keepdim=True)
    Q = Q / Q.sum(dim=-1, keepdim=True)
    M = 0.5 * (P + Q)
    kl1 = torch.sum(P * torch.log2(P / M), dim=-1)
    kl2 = torch.sum(Q * torch.log2(Q / M), dim=-1)
    return 0.5 * (kl1 + kl2)

X_tensor = torch.tensor(X_hybrid, dtype=torch.float32, device=device)
Y_tensor = torch.tensor(Y_targets, dtype=torch.float32, device=device)

# Unconditional Global Null
global_prior = Y_tensor.mean(dim=0, keepdim=True)
js_null_per_window = js_divergence_pt(Y_tensor, global_prior.repeat(N_win, 1)).cpu().numpy()
R_null = float(np.mean(js_null_per_window))
log(f"Unconditional Random Pair Null D_JS: {R_null:.5f} bits")

EPSILON_VALS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
unique_sessions = np.unique(sess_ids)

sweep_results = []
per_window_primary = {
    "win_idx": np.arange(N_win),
    "sess_idx": sess_ids,
    "t_sec": np.array([w["t_sec"] for w in raw_windows]),
    "cohort": np.array([w["cohort"] for w in raw_windows]),
    "drug": np.array([w["drug"] for w in raw_windows])
}

PRIMARY_EPS = 0.25  # Standard calibrated radius (approx top 5-10% closest analogues)

log("\nSweeping epsilon radii and computing cross-session density & dispersion...")

for eps_r in EPSILON_VALS:
    disp_arr = np.full(N_win, np.nan, dtype=np.float32)
    risk_arr = np.full(N_win, np.nan, dtype=np.float32)
    density_arr = np.zeros(N_win, dtype=np.int32)
    mean_dist_arr = np.full(N_win, np.nan, dtype=np.float32)

    for s_idx in unique_sessions:
        anchor_mask = (sess_ids == s_idx)
        cand_mask = (sess_ids != s_idx)

        anchor_indices = np.where(anchor_mask)[0]
        cand_indices = np.where(cand_mask)[0]

        X_anchors = X_tensor[anchor_indices]
        X_cands = X_tensor[cand_indices]
        Y_anchors = Y_tensor[anchor_indices]
        Y_cands = Y_tensor[cand_indices]

        batch_size = 512
        for b_start in range(0, len(anchor_indices), batch_size):
            b_end = min(b_start + batch_size, len(anchor_indices))
            b_X = X_anchors[b_start:b_end]
            b_Y = Y_anchors[b_start:b_end]
            b_global_idx = anchor_indices[b_start:b_end]

            dists = torch.cdist(b_X, X_cands, p=2.0)  # [B, N_cands]
            within_eps = (dists < eps_r)
            counts = within_eps.sum(dim=1)  # [B]

            counts_cpu = counts.cpu().numpy()
            density_arr[b_global_idx] = counts_cpu

            # For windows with at least 2 cross-session analogues
            valid_b = np.where(counts_cpu >= 2)[0]
            for v_idx in valid_b:
                mask_j = within_eps[v_idx]
                matched_Y = Y_cands[mask_j]
                matched_dists = dists[v_idx, mask_j]

                # JS Barycenter / Centroid
                Y_bar = torch.mean(matched_Y, dim=0, keepdim=True)

                # Held-out prediction risk
                r_pred = js_divergence_pt(b_Y[v_idx:v_idx+1], Y_bar).item()

                # Within-neighborhood future dispersion
                disp_k = js_divergence_pt(matched_Y, Y_bar.repeat(len(matched_Y), 1)).mean().item()

                disp_arr[b_global_idx[v_idx]] = disp_k
                risk_arr[b_global_idx[v_idx]] = r_pred
                mean_dist_arr[b_global_idx[v_idx]] = matched_dists.mean().item()

    valid_mask = ~np.isnan(disp_arr)
    n_covered = np.sum(valid_mask)
    coverage_pct = (n_covered / N_win) * 100.0

    if n_covered > 0:
        mean_disp = float(np.mean(disp_arr[valid_mask]))
        mean_risk = float(np.mean(risk_arr[valid_mask]))
        mean_density = float(np.mean(density_arr[valid_mask]))
        mean_d = float(np.mean(mean_dist_arr[valid_mask]))

        # Empirical Predictability Index normalized against R_window
        resolved_fraction = ((R_null - mean_disp) / (R_null - R_WINDOW_120S)) * 100.0
    else:
        mean_disp, mean_risk, mean_density, mean_d, resolved_fraction = np.nan, np.nan, np.nan, np.nan, np.nan

    log(f"  [eps = {eps_r:4.2f}] Coverage: {coverage_pct:5.1f}% ({n_covered:,} wins) | Density: {mean_density:6.1f} | Mean d_X: {mean_d:.4f} | Disp D_i: {mean_disp:.4f} b | Resolved: {resolved_fraction:5.1f}%")

    sweep_results.append({
        "Epsilon_Radius": eps_r,
        "Coverage_pct": coverage_pct,
        "Covered_Windows": n_covered,
        "Mean_Density_n": mean_density,
        "Mean_Past_Distance": mean_d,
        "Conditional_Dispersion_bits": mean_disp,
        "Held_Out_Risk_bits": mean_risk,
        "Finite_Window_Floor_bits": R_WINDOW_120S,
        "Resolved_Fraction_pct": resolved_fraction
    })

    if eps_r == PRIMARY_EPS:
        per_window_primary["density_n"] = density_arr
        per_window_primary["dispersion_bits"] = disp_arr
        per_window_primary["excess_dispersion_bits"] = disp_arr - R_WINDOW_120S
        per_window_primary["held_out_risk_bits"] = risk_arr
        per_window_primary["mean_past_dist"] = mean_dist_arr

df_sweep = pd.DataFrame(sweep_results)
sweep_csv = os.path.join(TABLE_DIR, "epsilon_radius_dispersion_sweep.csv")
df_sweep.to_csv(sweep_csv, index=False)
log(f"\nSaved sweep results to: {sweep_csv}")

# -------------------------------------------------------------------------
# STEP 3: 2D DENSITY x EXCESS DISPERSION QUADRANT MAPPING
# -------------------------------------------------------------------------
df_win = pd.DataFrame(per_window_primary)
valid_win = df_win[~df_win["dispersion_bits"].isna()].copy()
log(f"\nAnalyzing 2D Quadrant Structure for {len(valid_win):,} windows at primary epsilon={PRIMARY_EPS}...")

# Compute Median Splits for Density and Excess Dispersion
med_density = valid_win["density_n"].median()
med_disp = valid_win["excess_dispersion_bits"].median()

def assign_quadrant(row):
    high_dens = (row["density_n"] >= med_density)
    high_disp = (row["excess_dispersion_bits"] >= med_disp)
    if high_dens and not high_disp:
        return "Q1: High Dens / Low Disp (Deterministic Routine)"
    elif high_dens and high_disp:
        return "Q2: High Dens / High Disp (★ Neural Target / Branch Point)"
    elif not high_dens and not high_disp:
        return "Q3: Low Dens / Low Disp (Rare Stereotyped Motif)"
    else:
        return "Q4: Low Dens / High Disp (Data-Starved Epistemic)"

valid_win["quadrant"] = valid_win.apply(assign_quadrant, axis=1)

q_counts = valid_win["quadrant"].value_counts()
log("\nQuadrant Distribution:")
for q_name, cnt in q_counts.items():
    log(f"  {q_name}: {cnt:,} windows ({cnt/len(valid_win)*100:.1f}%)")

win_csv = os.path.join(TABLE_DIR, "per_window_density_dispersion_quadrants.csv")
valid_win.to_csv(win_csv, index=False)
log(f"Saved per-window quadrant table to: {win_csv}")

# -------------------------------------------------------------------------
# RENDER 2D QUADRANT FIGURE
# -------------------------------------------------------------------------
log("Rendering 2D Density vs. Conditional Dispersion Quadrant Figure...")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

fig = plt.figure(figsize=(15, 6.5), dpi=300)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.28)

# Panel A: 2D Scatter / Hexbin Quadrant Map
ax1 = fig.add_subplot(gs[0])
x_vals = np.log10(valid_win["density_n"] + 1)
y_vals = valid_win["excess_dispersion_bits"]

q_colors = {
    "Q1: High Dens / Low Disp (Deterministic Routine)": "#2ca02c",
    "Q2: High Dens / High Disp (★ Neural Target / Branch Point)": "#d62728",
    "Q3: Low Dens / Low Disp (Rare Stereotyped Motif)": "#1f77b4",
    "Q4: Low Dens / High Disp (Data-Starved Epistemic)": "#ff7f0e"
}

for q_name, c in q_colors.items():
    sub = valid_win[valid_win["quadrant"] == q_name]
    ax1.scatter(np.log10(sub["density_n"] + 1), sub["excess_dispersion_bits"],
                color=c, alpha=0.35, s=12, label=f"{q_name.split(':')[0]} ({len(sub):,} wins)")

ax1.axvline(np.log10(med_density + 1), color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.axhline(med_disp, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

ax1.set_title("A. Behavioral State Space: Analogue Density vs. Excess Future Dispersion\n(Conditioned on Past Repertoire History $W = 68.3\\text{s}$, Radius $\\epsilon = 0.25$)", fontsize=11, fontweight='bold', pad=12)
ax1.set_xlabel("Local Analogue Density $\\log_{10}(n_i(\\epsilon) + 1)$ [Number of Cross-Session Twins]", fontsize=11)
ax1.set_ylabel("Excess Conditional Future Dispersion $D_i(\\epsilon) - R_{\\text{window}}$ (bits)", fontsize=11)
ax1.legend(loc='upper right', frameon=True, fontsize=8.5)
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel B: Calibrated Predictability Scale & Epsilon Sweep
ax2 = fig.add_subplot(gs[1])
ax2.plot(df_sweep["Epsilon_Radius"], df_sweep["Conditional_Dispersion_bits"],
         marker='o', linewidth=2.5, color='#1f77b4', label='Conditional Dispersion $D(\\epsilon)$')
ax2.plot(df_sweep["Epsilon_Radius"], df_sweep["Held_Out_Risk_bits"],
         marker='s', linewidth=2.0, linestyle='--', color='#9467bd', label='Held-Out Centroid Risk $\\text{JS}(Y_i, \\bar{Y})$')

ax2.axhline(R_null, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Random Pairs Null ({R_null:.3f} b)')
ax2.axhline(R_WINDOW_120S, color='#2ca02c', linestyle=':', linewidth=2.0, alpha=0.9, label=f'Finite-Window Floor $R_{{\\text{{window}}}}$ ({R_WINDOW_120S:.3f} b)')

ax2.set_title("B. Decoupling Similarity ($\\epsilon$) from Sample Density\n(Dispersion Extrapolation & Finite-Window Floor Calibration)", fontsize=11, fontweight='bold', pad=12)
ax2.set_xlabel("Analogue Matching Radius $\\epsilon$ (Past Behavioral Distance)", fontsize=11)
ax2.set_ylabel("Divergence / Dispersion (bits, $D_{\\text{JS}}$)", fontsize=11)
ax2.legend(loc='lower right', frameon=True, fontsize=8.5)
ax2.grid(True, linestyle='--', alpha=0.5)

fig_path = os.path.join(FIG_DIR, "fig13_density_dispersion_quadrant_map.png")
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)
log(f"Saved Quadrant Figure to: {fig_path}")

log("EPSILON DENSITY-DISPERSION QUADRANT MAPPER COMPLETE!")
