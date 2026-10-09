#!/usr/bin/env python3
"""
===============================================================================
Experiment: Continuous Pose Kinematics vs. Discrete Syllable (P2 -> P2.5 -> P3)
===============================================================================
Tests whether continuous 30-D pose coordinates alone improve future prediction 
over spatial centroids (P2 -> P2.5), and whether discrete syllable classification 
(P3) adds information beyond raw continuous pose.

Evaluates under 5-Fold Session-Grouped Cross-Validation (W=68.3s -> H=120s):
  - P_0: Blind Null
  - P_1: Time Drift
  - P_2: Spatial Context (Centroid x,y + Velocity + Time)
  - P_2.5: Continuous Pose Kinematics (P_2 + 30-D Continuous Pose Vector, NO syllable ID)
  - P_3: Discrete Syllable (P_2 + 100-D One-Hot Syllable ID, NO continuous pose)
  - P_3.5: Hybrid (P_2 + Discrete Syllable + 30-D Continuous Pose Vector)
===============================================================================
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
with open(cache_path, "rb") as f:
    data = pickle.load(f)
session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

FPS = 80.025
W_SEC = 68.3
W_FRAMES = int(W_SEC * FPS)
H_SEC = 120.0
H_FRAMES = int(H_SEC * FPS)
STRIDE_SEC = 5.0
STRIDE_FRAMES = int(STRIDE_SEC * FPS)
NUM_SYLLS = 100

def js_vec(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    P = np.clip(P, eps, 1.0); Q = np.clip(Q, eps, 1.0)
    P = P / np.sum(P, axis=1, keepdims=True); Q = Q / np.sum(Q, axis=1, keepdims=True)
    M = 0.5 * (P + Q)
    return float(np.mean(0.5 * (np.sum(P * np.log2(P / M), axis=1) + np.sum(Q * np.log2(Q / M), axis=1))))

sessions = []
for sess_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    max_frame = bouts[-1]["end_frame"]
    N = max_frame + 1

    frame_sylls = np.zeros(N, dtype=np.int32)
    frame_spatial = np.zeros((N, 4), dtype=np.float32)  # x, y, v, dist_center
    frame_pose = np.zeros((N, 30), dtype=np.float32)    # 30-D continuous pose vector

    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = min(b["syllable"], NUM_SYLLS - 1)
        frame_sylls[sf : ef + 1] = s_id
        vec = b.get("vector", None)
        if vec is not None:
            if len(vec) >= 4:
                frame_spatial[sf : ef + 1] = vec[:4]
            if len(vec) >= 30:
                frame_pose[sf : ef + 1] = vec[:30]
            else:
                frame_pose[sf : ef + 1, :len(vec)] = vec

    oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
    diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
    onsets = np.zeros(N, dtype=bool)
    onsets[0] = True; onsets[diffs] = True
    oh_bout[onsets, frame_sylls[onsets]] = 1.0
    cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])

    next_bout_ptrs = np.zeros(N, dtype=np.int32)
    cur = 0
    for d in diffs:
        next_bout_ptrs[cur:d] = d
        cur = d
    next_bout_ptrs[cur:] = N

    cum_spatial = np.vstack([np.zeros((1, 4), dtype=np.float32), np.cumsum(frame_spatial, axis=0)])
    cum_spatial_sq = np.vstack([np.zeros((1, 4), dtype=np.float32), np.cumsum(frame_spatial**2, axis=0)])

    sessions.append({
        "sess_idx": sess_idx, "N": N, "frame_sylls": frame_sylls,
        "frame_pose": frame_pose, "cum_bout": cum_bout,
        "next_bout_ptrs": next_bout_ptrs, "cum_spatial": cum_spatial,
        "cum_spatial_sq": cum_spatial_sq
    })

all_p1, all_p2, all_p25, all_p3, all_p35, all_tgt, all_sess = [], [], [], [], [], [], []

for s in sessions:
    sess_idx = s["sess_idx"]
    N = s["N"]
    if N < W_FRAMES + H_FRAMES + 200: continue

    t_anchors = np.arange(W_FRAMES, N - H_FRAMES, STRIDE_FRAMES)
    t_tgt_start = s["next_bout_ptrs"][t_anchors]
    valid = (t_tgt_start + H_FRAMES < N)
    t_anchors = t_anchors[valid]
    t_tgt_start = t_tgt_start[valid]
    M = len(t_anchors)
    if M == 0: continue

    # P_1: Time
    t_norm = t_anchors / float(N)
    p1 = np.column_stack([t_norm, t_norm**2]).astype(np.float32)

    # P_2: Spatial Context (x, y, v, dist_center mean & std)
    sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - W_FRAMES]
    sp_mean = sp_sum / float(W_FRAMES)
    sp_sq = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - W_FRAMES]
    sp_var = np.maximum(sp_sq / float(W_FRAMES) - sp_mean**2, 1e-6)
    p2 = np.hstack([p1, sp_mean, np.sqrt(sp_var)]).astype(np.float32)

    # P_2.5: Continuous Pose Vector at anchor t (30-D pose coordinates, NO syllable ID)
    pose_vec = s["frame_pose"][t_anchors]
    p25 = np.hstack([p2, pose_vec]).astype(np.float32)

    # P_3: Discrete Syllable One-Hot at anchor t (100-D, NO continuous pose)
    curr_syll = s["frame_sylls"][t_anchors]
    curr_oh = np.zeros((M, NUM_SYLLS), dtype=np.float32)
    curr_oh[np.arange(M), curr_syll] = 1.0
    p3 = np.hstack([p2, curr_oh]).astype(np.float32)

    # P_3.5: Hybrid (Space + Discrete Syllable + 30-D Continuous Pose)
    p35 = np.hstack([p2, curr_oh, pose_vec]).astype(np.float32)

    # Target
    f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
    target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

    all_p1.append(p1); all_p2.append(p2); all_p25.append(p25)
    all_p3.append(p3); all_p35.append(p35)
    all_tgt.append(target); all_sess.append(np.full(M, sess_idx, dtype=int))

X_p1 = np.vstack(all_p1); X_p2 = np.vstack(all_p2); X_p25 = np.vstack(all_p25)
X_p3 = np.vstack(all_p3); X_p35 = np.vstack(all_p35)
Y_tgt = np.vstack(all_tgt); groups = np.concatenate(all_sess)

gkf = GroupKFold(n_splits=5)
g_p1, g_p2, g_p25, g_p3, g_p35 = [], [], [], [], []

for tr_idx, te_idx in gkf.split(X_p2, groups=groups):
    Y_tr, Y_te = Y_tgt[tr_idx], Y_tgt[te_idx]
    p_glob = np.tile(np.mean(Y_tr, axis=0, keepdims=True), (len(te_idx), 1))
    p_glob /= np.sum(p_glob, axis=1, keepdims=True)
    js_blind = js_vec(Y_te, p_glob)

    # P_1
    p_1 = np.clip(Ridge(5.0).fit(X_p1[tr_idx], Y_tr).predict(X_p1[te_idx]), 1e-4, 1.0)
    g_p1.append(((js_blind - js_vec(Y_te, p_1 / np.sum(p_1, axis=1, keepdims=True))) / js_blind) * 100.0)

    # P_2: Space
    p_2 = np.clip(Ridge(5.0).fit(X_p2[tr_idx], Y_tr).predict(X_p2[te_idx]), 1e-4, 1.0)
    g_p2.append(((js_blind - js_vec(Y_te, p_2 / np.sum(p_2, axis=1, keepdims=True))) / js_blind) * 100.0)

    # P_2.5: Continuous Pose (30-D)
    p_25 = np.clip(Ridge(5.0).fit(X_p25[tr_idx], Y_tr).predict(X_p25[te_idx]), 1e-4, 1.0)
    g_p25.append(((js_blind - js_vec(Y_te, p_25 / np.sum(p_25, axis=1, keepdims=True))) / js_blind) * 100.0)

    # P_3: Discrete Syllable
    p_3 = np.clip(Ridge(5.0).fit(X_p3[tr_idx], Y_tr).predict(X_p3[te_idx]), 1e-4, 1.0)
    g_p3.append(((js_blind - js_vec(Y_te, p_3 / np.sum(p_3, axis=1, keepdims=True))) / js_blind) * 100.0)

    # P_3.5: Discrete Syllable + Continuous Pose
    p_35 = np.clip(Ridge(5.0).fit(X_p35[tr_idx], Y_tr).predict(X_p35[te_idx]), 1e-4, 1.0)
    g_p35.append(((js_blind - js_vec(Y_te, p_35 / np.sum(p_35, axis=1, keepdims=True))) / js_blind) * 100.0)

print("=" * 80)
print(f"RESULTS: Continuous Pose vs. Discrete Syllable (30 Sessions, 30,469 Windows)")
print("=" * 80)
print(f"P_0 (Blind Null):                                +  0.00% ± 0.00%")
print(f"P_1 (Time Drift):                                + {np.mean(g_p1):5.2f}% ± {1.96*np.std(g_p1)/np.sqrt(5):4.2f}%")
print(f"P_2 (Spatial Context: Centroid, Velocity):       + {np.mean(g_p2):5.2f}% ± {1.96*np.std(g_p2)/np.sqrt(5):4.2f}%")
print(f"P_2.5 (Continuous 30-D Pose Kinematics ONLY):    + {np.mean(g_p25):5.2f}% ± {1.96*np.std(g_p25)/np.sqrt(5):4.2f}%")
print(f"P_3 (Discrete Syllable ONLY):                    + {np.mean(g_p3):5.2f}% ± {1.96*np.std(g_p3)/np.sqrt(5):4.2f}%")
print(f"P_3.5 (Discrete Syllable + Continuous Pose):     + {np.mean(g_p35):5.2f}% ± {1.96*np.std(g_p35)/np.sqrt(5):4.2f}%")
print("=" * 80)
