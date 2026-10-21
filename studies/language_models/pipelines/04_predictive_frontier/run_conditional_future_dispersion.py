#!/usr/bin/env python3
"""
Conditional Future Dispersion & Cross-Session k-NN Predictive Risk Benchmark
(Clean Direct Bout Onsets Version - Zero Inter-Bout Gap Artifact)
=============================================================================
"""

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

LOG_FILE = os.path.join(LM_DIR, "output", "conditional_future_dispersion.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

with open(LOG_FILE, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INITIALIZED CLEAN CONDITIONAL FUTURE DISPERSION BENCHMARK\n")

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

log("Preprocessing session timecourses with direct bout onsets...")
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
        s_id = s_raw - 1 # Map 1..99 -> 0..98
        if 0 <= s_id < NUM_ACTIVE_SYLLS:
            fs[sf:ef+1] = s_id
            oh_bout[sf, s_id] = 1.0 # Direct bout onset

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

log(f"Preprocessed {len(sessions_data)} sessions. Slicing clean windows...")

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

    # 2. Continuous pose (30-D)
    feat_pose_inst = s["pose"][t_anchors].astype(np.float32)

    # 3. Discrete syllable (99-D one-hot)
    curr_syll = s["fs"][t_anchors]
    feat_syll_oh = np.zeros((M, NUM_ACTIVE_SYLLS), dtype=np.float32)
    valid_syll = (curr_syll >= 0) & (curr_syll < NUM_ACTIVE_SYLLS)
    feat_syll_oh[np.arange(M)[valid_syll], curr_syll[valid_syll]] = 1.0

    # 4. Repertoire probability distribution (99-D)
    b_cnt = s["cum_bout"][t_anchors] - s["cum_bout"][t_anchors - W_FRAMES]
    feat_rep = (b_cnt + 1e-4) / (np.sum(b_cnt, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    # 5. Future Target distribution over H=120s (99-D)
    f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
    target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_ACTIVE_SYLLS)

    for i in range(M):
        raw_windows.append({
            "sess_idx": s["sess_idx"],
            "t_sec": t_anchors[i] / FPS,
            "space": feat_space[i],
            "pose": feat_pose_inst[i],
            "syll_oh": feat_syll_oh[i],
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
X_pose_raw = np.array([w["pose"] for w in raw_windows], dtype=np.float32)
X_syll_raw = np.array([w["syll_oh"] for w in raw_windows], dtype=np.float32)
X_rep_raw = np.array([w["rep"] for w in raw_windows], dtype=np.float32)

log("Standardizing representations & computing cross-session NMF...")
scaler_space = StandardScaler().fit(X_space_raw)
X_space_std = scaler_space.transform(X_space_raw).astype(np.float32)

scaler_pose = StandardScaler().fit(X_pose_raw)
X_pose_std = scaler_pose.transform(X_pose_raw).astype(np.float32)

X_space = X_space_std / np.sqrt(X_space_std.shape[1])
X_pose = np.hstack([X_space, X_pose_std / np.sqrt(X_pose_std.shape[1])]) / np.sqrt(2.0)
X_syll = np.hstack([X_space, X_syll_raw / np.sqrt(np.mean(np.sum(X_syll_raw**2, axis=1)))]) / np.sqrt(2.0)
X_rep_hellinger = np.sqrt(X_rep_raw)
X_hybrid = np.hstack([X_space, X_rep_hellinger / np.sqrt(np.mean(np.sum(X_rep_hellinger**2, axis=1)))]) / np.sqrt(2.0)

X_nmf = np.zeros((N_win, 4), dtype=np.float32)
unique_sessions = np.unique(sess_ids)
for test_s in unique_sessions:
    train_mask = (sess_ids != test_s)
    test_mask = (sess_ids == test_s)
    nmf_model = NMF(n_components=4, init='nndsvda', random_state=42, max_iter=200)
    nmf_model.fit(X_rep_raw[train_mask])
    W_test = nmf_model.transform(X_rep_raw[test_mask])
    W_test_norm = W_test / (np.sum(W_test, axis=1, keepdims=True) + 1e-6)
    X_nmf[test_mask] = W_test_norm

REPRESENTATIONS = {
    "1. Spatial Context": X_space,
    "2. Continuous Pose": X_pose,
    "3. Current Syllable": X_syll,
    "4. Repertoire (Hellinger)": X_rep_hellinger,
    "5. Repertoire + Space (Hybrid)": X_hybrid,
    "6. NMF K=4 (Cross-Validated)": X_nmf
}

K_NEIGHBORS = [100, 50, 20, 10, 5, 2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Running non-parametric k-NN sweep on device: {device}")

def js_divergence_pt(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    P = torch.clamp(P, min=eps, max=1.0)
    Q = torch.clamp(Q, min=eps, max=1.0)
    P = P / P.sum(dim=-1, keepdim=True)
    Q = Q / Q.sum(dim=-1, keepdim=True)
    M = 0.5 * (P + Q)
    kl1 = torch.sum(P * torch.log2(P / M), dim=-1)
    kl2 = torch.sum(Q * torch.log2(Q / M), dim=-1)
    return 0.5 * (kl1 + kl2)

Y_tensor = torch.tensor(Y_targets, dtype=torch.float32, device=device)
global_prior = Y_tensor.mean(dim=0, keepdim=True)
js_null_per_window = js_divergence_pt(Y_tensor, global_prior.repeat(N_win, 1)).cpu().numpy()
mean_null_js = float(np.mean(js_null_per_window))
log(f"Clean Global Blind Null D_JS: {mean_null_js:.5f} bits")

all_results = []
fanning_case_studies = {}

for rep_name, X_mat in REPRESENTATIONS.items():
    log("\n=================================================================")
    log(f"EVALUATING REPRESENTATION: {rep_name} (Dim={X_mat.shape[1]})")
    log("=================================================================")

    X_tensor = torch.tensor(X_mat, dtype=torch.float32, device=device)

    dispersion_by_k = {k: np.zeros(N_win, dtype=np.float32) for k in K_NEIGHBORS}
    pred_risk_by_k = {k: np.zeros(N_win, dtype=np.float32) for k in K_NEIGHBORS}
    mean_dist_by_k = {k: np.zeros(N_win, dtype=np.float32) for k in K_NEIGHBORS}

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

            dists = torch.cdist(b_X, X_cands, p=2.0)
            max_k = max(K_NEIGHBORS)
            topk_dists, topk_idx = torch.topk(dists, k=max_k, largest=False, dim=1)

            for k in K_NEIGHBORS:
                sub_dists = topk_dists[:, :k]
                sub_idx = topk_idx[:, :k]
                neigh_Y = Y_cands[sub_idx]

                Y_bar_N = torch.mean(neigh_Y, dim=1)
                risk_pred = js_divergence_pt(b_Y, Y_bar_N)

                Y_bar_expanded = Y_bar_N.unsqueeze(1).repeat(1, k, 1)
                disp_k = js_divergence_pt(neigh_Y.reshape(-1, NUM_ACTIVE_SYLLS), Y_bar_expanded.reshape(-1, NUM_ACTIVE_SYLLS))
                disp_k = disp_k.reshape(-1, k).mean(dim=1)

                pred_risk_by_k[k][b_global_idx] = risk_pred.cpu().numpy()
                dispersion_by_k[k][b_global_idx] = disp_k.cpu().numpy()
                mean_dist_by_k[k][b_global_idx] = sub_dists.mean(dim=1).cpu().numpy()

        if rep_name == "4. Repertoire (Hellinger)" and s_idx == unique_sessions[0]:
            sample_anchors = [150, 450]
            for a_i in sample_anchors:
                dists_sample = torch.cdist(X_anchors[a_i:a_i+1], X_cands, p=2.0)
                _, top5_idx = torch.topk(dists_sample, k=5, largest=False, dim=1)
                top5_cand_global = cand_indices[top5_idx[0].cpu().numpy()]
                fanning_case_studies[f"anchor_{a_i}"] = {
                    "anchor_idx": anchor_indices[a_i],
                    "anchor_sess": s_idx,
                    "anchor_past_rep": X_rep_raw[anchor_indices[a_i]],
                    "anchor_future": Y_targets[anchor_indices[a_i]],
                    "neighbor_indices": top5_cand_global,
                    "neighbor_sessions": sess_ids[top5_cand_global],
                    "neighbor_past_reps": X_rep_raw[top5_cand_global],
                    "neighbor_futures": Y_targets[top5_cand_global]
                }

    for k in K_NEIGHBORS:
        sess_gains, sess_disps, sess_dists = [], [], []

        for s_idx in unique_sessions:
            s_mask = (sess_ids == s_idx)
            null_s = np.mean(js_null_per_window[s_mask])
            risk_s = np.mean(pred_risk_by_k[k][s_mask])
            disp_s = np.mean(dispersion_by_k[k][s_mask])
            dist_s = np.mean(mean_dist_by_k[k][s_mask])

            sess_gains.append(((null_s - risk_s) / null_s) * 100.0)
            sess_disps.append(disp_s)
            sess_dists.append(dist_s)

        mean_gain = float(np.mean(sess_gains))
        ci_gain = float(1.96 * np.std(sess_gains) / np.sqrt(len(sess_gains)))
        mean_disp = float(np.mean(sess_disps))
        ci_disp = float(1.96 * np.std(sess_disps) / np.sqrt(len(sess_disps)))
        mean_d = float(np.mean(sess_dists))

        log(f"  [k={k:3d}] Past Dist: {mean_d:.4f} | Future Disp: {mean_disp:.4f} ± {ci_disp:.4f} bits | Pred Gain: +{mean_gain:5.2f}% ± {ci_gain:4.2f}%")

        all_results.append({
            "Representation": rep_name,
            "k_neighbors": k,
            "Mean_Past_Distance": mean_d,
            "Future_Dispersion_bits": mean_disp,
            "Dispersion_CI_95": ci_disp,
            "Predictive_Gain_pct": mean_gain,
            "Gain_CI_95": ci_gain
        })

df_results = pd.DataFrame(all_results)
csv_out_path = os.path.join(LM_DIR, "output", "04_predictive_frontier", "tables", "conditional_future_dispersion_results.csv")
df_results.to_csv(csv_out_path, index=False)
log(f"\nSaved clean results to: {csv_out_path}")

case_study_path = os.path.join(LM_DIR, "output", "04_predictive_frontier", "tables", "fanning_futures_exemplars.pkl")
with open(case_study_path, "wb") as f:
    pickle.dump(fanning_case_studies, f)

report_path = os.path.join(LM_DIR, "output", "04_predictive_frontier", "tables", "conditional_future_dispersion_report.md")
with open(report_path, "w") as f:
    f.write("# Empirical Conditional Future Dispersion & Cross-Session k-NN Benchmark (Clean 99-D Repertoire)\n\n")
    f.write("**Status:** Clean Empirical Non-Parametric Benchmark Report  \n")
    f.write(f"**Dataset:** 30 Open-Field Sessions ({N_win:,} Clean Windows, W=68.3s -> H=120.0s)  \n")
    f.write("**Evaluation:** Strict Cross-Session k-NN Matching (`session_cand != session_anchor`) with Session-Clustered 95% CIs  \n\n")
    f.write("---\n\n## 1. Summary of Clean Results\n\n")
    f.write(df_results.to_markdown(index=False))

log("CLEAN BENCHMARK COMPLETE!")
