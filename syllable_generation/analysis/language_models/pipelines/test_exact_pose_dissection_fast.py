#!/usr/bin/env python3
"""
Exact Pose Dissection Fast Benchmark
Logs all progress to exact_pose_dissection.log
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

LOG_FILE = os.path.join(LM_DIR, "output", "exact_pose_dissection.log")
with open(LOG_FILE, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INITIALIZED EXACT POSE DISSECTION BENCHMARK\n")

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

cache_path = os.path.join(LM_DIR, "output", "strategies_extended", "continuous_bout_cache.pkl")
log(f"Loading session bout cache: {cache_path}")
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

log("Preprocessing sessions...")
sessions = []
for sess_idx, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    max_frame = bouts[-1]["end_frame"]
    N = max_frame + 1

    frame_sylls = np.zeros(N, dtype=np.int32)
    frame_spatial = np.zeros((N, 6), dtype=np.float32)
    frame_pose = np.zeros((N, 30), dtype=np.float32)

    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        s_id = min(b["syllable"], NUM_SYLLS - 1)
        frame_sylls[sf : ef + 1] = s_id
        vec = b.get("vector", None)
        if vec is not None and len(vec) >= 14:
            frame_spatial[sf : ef + 1, :4] = vec[:4]
            frame_spatial[sf : ef + 1, 4] = vec[10]
            frame_spatial[sf : ef + 1, 5] = vec[12]
            if len(vec) >= 30:
                frame_pose[sf : ef + 1] = vec[:30]
            else:
                frame_pose[sf : ef + 1, :len(vec)] = vec

    diffs = np.where(frame_sylls[1:] != frame_sylls[:-1])[0] + 1
    onsets = np.zeros(N, dtype=bool)
    onsets[0] = True; onsets[diffs] = True

    oh_bout = np.zeros((N, NUM_SYLLS), dtype=np.float32)
    oh_bout[onsets, frame_sylls[onsets]] = 1.0
    cum_bout = np.vstack([np.zeros((1, NUM_SYLLS), dtype=np.float32), np.cumsum(oh_bout, axis=0)])

    next_bout_ptrs = np.zeros(N, dtype=np.int32)
    cur = 0
    for d in diffs:
        next_bout_ptrs[cur:d] = d
        cur = d
    next_bout_ptrs[cur:] = N

    cum_spatial = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial, axis=0)])
    cum_spatial_sq = np.vstack([np.zeros((1, 6), dtype=np.float32), np.cumsum(frame_spatial**2, axis=0)])

    sessions.append({
        "sess_idx": sess_idx, "N": N, "frame_sylls": frame_sylls,
        "frame_pose": frame_pose, "cum_bout": cum_bout,
        "next_bout_ptrs": next_bout_ptrs,
        "cum_spatial": cum_spatial, "cum_spatial_sq": cum_spatial_sq
    })

log(f"Preprocessed {len(sessions)} sessions.")

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

    # P_1: Time drift (2-D)
    t_norm = t_anchors / float(N)
    p1 = np.column_stack([t_norm, t_norm**2]).astype(np.float32)

    # P_2: Canonical Spatial Context (14-D)
    sp_sum = s["cum_spatial"][t_anchors] - s["cum_spatial"][t_anchors - W_FRAMES]
    sp_mean = sp_sum / float(W_FRAMES)
    sp_sq = s["cum_spatial_sq"][t_anchors] - s["cum_spatial_sq"][t_anchors - W_FRAMES]
    sp_var = np.maximum(sp_sq / float(W_FRAMES) - sp_mean**2, 1e-6)
    p2 = np.hstack([p1, sp_mean, np.sqrt(sp_var)]).astype(np.float32)

    # P_2.5: P2 + Instantaneous Continuous 30-D Pose (44-D)
    pose_inst = s["frame_pose"][t_anchors]
    p25 = np.hstack([p2, pose_inst]).astype(np.float32)

    # P_3: P2 + Instantaneous Discrete Syllable One-Hot (114-D)
    curr_syll = s["frame_sylls"][t_anchors]
    curr_oh = np.zeros((M, NUM_SYLLS), dtype=np.float32)
    curr_oh[np.arange(M), curr_syll] = 1.0
    p3 = np.hstack([p2, curr_oh]).astype(np.float32)

    # P_3.5: Hybrid (P2 + Continuous Pose + Discrete Syllable) (144-D)
    p35 = np.hstack([p2, pose_inst, curr_oh]).astype(np.float32)

    # Target
    f_cnt = s["cum_bout"][t_tgt_start + H_FRAMES] - s["cum_bout"][t_tgt_start]
    target = (f_cnt + 1e-4) / (np.sum(f_cnt, axis=1, keepdims=True) + 1e-4 * NUM_SYLLS)

    all_p1.append(p1); all_p2.append(p2); all_p25.append(p25)
    all_p3.append(p3); all_p35.append(p35)
    all_tgt.append(target); all_sess.append(np.full(M, sess_idx, dtype=int))

X_p1 = np.vstack(all_p1); X_p2 = np.vstack(all_p2); X_p25 = np.vstack(all_p25)
X_p3 = np.vstack(all_p3); X_p35 = np.vstack(all_p35)
Y_tgt = np.vstack(all_tgt); groups = np.concatenate(all_sess)

log(f"Extracted {len(Y_tgt):,} evaluation samples across {len(np.unique(groups))} sessions.")
log(f"Feature dimensions: P1={X_p1.shape[1]}D, P2={X_p2.shape[1]}D, P2.5={X_p25.shape[1]}D, P3={X_p3.shape[1]}D, P3.5={X_p35.shape[1]}D")

gkf = GroupKFold(n_splits=5)
g_p1, g_p2, g_p25, g_p3, g_p35 = [], [], [], [], []

for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_p1, groups=groups)):
    fold_num = fold + 1
    Y_tr, Y_te = Y_tgt[tr_idx], Y_tgt[te_idx]

    p_glob = np.tile(np.mean(Y_tr, axis=0, keepdims=True), (len(te_idx), 1))
    p_glob /= np.sum(p_glob, axis=1, keepdims=True)
    js_blind = js_vec(Y_te, p_glob)

    # P1
    p_1 = np.clip(Ridge(5.0).fit(X_p1[tr_idx], Y_tr).predict(X_p1[te_idx]), 1e-4, 1.0)
    g1 = ((js_blind - js_vec(Y_te, p_1 / np.sum(p_1, axis=1, keepdims=True))) / js_blind) * 100.0
    g_p1.append(g1)

    # P2 (Canonical Space + Height + Curvature)
    p_2 = np.clip(Ridge(5.0).fit(X_p2[tr_idx], Y_tr).predict(X_p2[te_idx]), 1e-4, 1.0)
    g2 = ((js_blind - js_vec(Y_te, p_2 / np.sum(p_2, axis=1, keepdims=True))) / js_blind) * 100.0
    g_p2.append(g2)

    # P2.5 (P2 + Continuous 30-D Pose)
    p_25 = np.clip(Ridge(5.0).fit(X_p25[tr_idx], Y_tr).predict(X_p25[te_idx]), 1e-4, 1.0)
    g25 = ((js_blind - js_vec(Y_te, p_25 / np.sum(p_25, axis=1, keepdims=True))) / js_blind) * 100.0
    g_p25.append(g25)

    # P3 (Canonical P2 + Discrete Syllable)
    p_3 = np.clip(Ridge(5.0).fit(X_p3[tr_idx], Y_tr).predict(X_p3[te_idx]), 1e-4, 1.0)
    g3 = ((js_blind - js_vec(Y_te, p_3 / np.sum(p_3, axis=1, keepdims=True))) / js_blind) * 100.0
    g_p3.append(g3)

    # P3.5 (P2 + Continuous Pose + Discrete Syllable)
    p_35 = np.clip(Ridge(5.0).fit(X_p35[tr_idx], Y_tr).predict(X_p35[te_idx]), 1e-4, 1.0)
    g35 = ((js_blind - js_vec(Y_te, p_35 / np.sum(p_35, axis=1, keepdims=True))) / js_blind) * 100.0
    g_p35.append(g35)

    log(f"  [Fold {fold_num}/5] P1: +{g1:5.2f}% | P2(Space): +{g2:5.2f}% | P2.5(Pose): +{g25:5.2f}% | P3(Syll): +{g3:5.2f}% | P3.5(Both): +{g35:5.2f}%")

ci = lambda arr: float(1.96 * np.std(arr) / np.sqrt(len(arr)))

log("\n" + "=" * 85)
log("FINAL VERIFIED RESULTS: CANONICAL P2 vs. CONTINUOUS POSE vs. DISCRETE SYLLABLES")
log("=" * 85)
log(f"P_0 (Blind Null):                                      +  0.00% ± 0.00%")
log(f"P_1 (Time Drift):                        [{X_p1.shape[1]:3d}-D]   + {np.mean(g_p1):5.2f}% ± {ci(g_p1):4.2f}%")
log(f"P_2 (Canonical Space + Height + Curv):   [{X_p2.shape[1]:3d}-D]   + {np.mean(g_p2):5.2f}% ± {ci(g_p2):4.2f}%")
log(f"P_2.5 (P2 + Continuous 30-D Pose):       [{X_p25.shape[1]:3d}-D]   + {np.mean(g_p25):5.2f}% ± {ci(g_p25):4.2f}%")
log(f"P_3 (Canonical P2 + Discrete Syllable):  [{X_p3.shape[1]:3d}-D]   + {np.mean(g_p3):5.2f}% ± {ci(g_p3):4.2f}%")
log(f"P_3.5 (P2 + Continuous Pose + Syllable): [{X_p35.shape[1]:3d}-D]   + {np.mean(g_p35):5.2f}% ± {ci(g_p35):4.2f}%")
log("=" * 85)
log("BENCHMARK COMPLETE!")
