#!/usr/bin/env python3
"""
Lightweight P2 -> P2.5 (Continuous Pose) -> P3 (Discrete Syllable) Dissection
Uses EXACT canonical P2 construction (6-D spatial: x,y,v,dist,height,curvature -> mean+std -> 14-D)
Adds instantaneous 30-D pose vector at anchor (no cumsum over full session -> no OOM).
"""
import pickle
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models")
cache = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/strategies_extended/continuous_bout_cache.pkl"
with open(cache, "rb") as f:
    data = pickle.load(f)
session_bouts = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

FPS, W_SEC, H_SEC, STRIDE_SEC, NUM_SYLLS = 80.025, 68.3, 120.0, 5.0, 100
W_FR, H_FR, STR_FR = int(W_SEC*FPS), int(H_SEC*FPS), int(STRIDE_SEC*FPS)

def js_vec(P, Q, eps=1e-12):
    P = np.clip(P, eps, 1.0); Q = np.clip(Q, eps, 1.0)
    P /= P.sum(1, keepdims=True); Q /= Q.sum(1, keepdims=True)
    M = 0.5*(P+Q)
    return float(np.mean(0.5*(np.sum(P*np.log2(P/M),1) + np.sum(Q*np.log2(Q/M),1))))

print("Preprocessing sessions...", flush=True)
sessions = []
for si, bouts in enumerate(session_bouts):
    if len(bouts) < 100: continue
    N = bouts[-1]["end_frame"] + 1
    fs = np.zeros(N, dtype=np.int32)
    sp6 = np.zeros((N,6), dtype=np.float32)
    pose30 = np.zeros((N,30), dtype=np.float32)
    for b in bouts:
        sf, ef = b["start_frame"], b["end_frame"]
        sid = min(b["syllable"], 99)
        fs[sf:ef+1] = sid
        v = b.get("vector")
        if v is not None and len(v) >= 14:
            sp6[sf:ef+1, :4] = v[:4]
            sp6[sf:ef+1, 4] = v[10]
            sp6[sf:ef+1, 5] = v[12]
            if len(v) >= 30:
                pose30[sf:ef+1] = v[:30]
            else:
                pose30[sf:ef+1, :len(v)] = v
    diffs = np.where(fs[1:] != fs[:-1])[0] + 1
    onsets = np.zeros(N, dtype=bool); onsets[0] = True; onsets[diffs] = True
    oh = np.zeros((N,100), dtype=np.float32); oh[onsets, fs[onsets]] = 1.0
    cb = np.vstack([np.zeros((1,100), dtype=np.float32), np.cumsum(oh, axis=0)])
    nbp = np.zeros(N, dtype=np.int32); cur = 0
    for d in diffs: nbp[cur:d] = d; cur = d
    nbp[cur:] = N
    cs = np.vstack([np.zeros((1,6), dtype=np.float32), np.cumsum(sp6, axis=0)])
    cs2 = np.vstack([np.zeros((1,6), dtype=np.float32), np.cumsum(sp6**2, axis=0)])
    sessions.append({"si": si, "N": N, "fs": fs, "pose30": pose30, "cb": cb, "nbp": nbp, "cs": cs, "cs2": cs2})
print(f"Preprocessed {len(sessions)} sessions.", flush=True)

all_p1, all_p2, all_p25, all_p3, all_p35, all_tgt, all_grp = [], [], [], [], [], [], []
for s in sessions:
    N = s["N"]
    if N < W_FR + H_FR + 200: continue
    ta = np.arange(W_FR, N - H_FR, STR_FR)
    ts = s["nbp"][ta]; valid = (ts + H_FR < N); ta = ta[valid]; ts = ts[valid]
    M = len(ta)
    if M == 0: continue
    # P1: Time
    tn = ta / float(N)
    p1 = np.column_stack([tn, tn**2]).astype(np.float32)
    # P2 Canonical: Time + 6-D Spatial (mean + std over W) -> 14-D
    sp_sum = s["cs"][ta] - s["cs"][ta - W_FR]; sp_mean = sp_sum / float(W_FR)
    sp_sq = s["cs2"][ta] - s["cs2"][ta - W_FR]; sp_std = np.sqrt(np.maximum(sp_sq/float(W_FR) - sp_mean**2, 1e-6))
    p2 = np.hstack([p1, sp_mean, sp_std]).astype(np.float32)
    # P2.5: Canonical P2 + Instantaneous 30-D Pose Vector at Anchor
    p25 = np.hstack([p2, s["pose30"][ta]]).astype(np.float32)
    # P3 Canonical: P2 + One-Hot Discrete Syllable
    curr_oh = np.zeros((M, 100), dtype=np.float32)
    curr_oh[np.arange(M), s["fs"][ta]] = 1.0
    p3 = np.hstack([p2, curr_oh]).astype(np.float32)
    # P3.5: P2 + 30-D Pose + One-Hot Syllable
    p35 = np.hstack([p2, s["pose30"][ta], curr_oh]).astype(np.float32)
    # Target
    fc = s["cb"][ts + H_FR] - s["cb"][ts]
    tgt = (fc + 1e-4) / (fc.sum(1, keepdims=True) + 1e-4 * 100)
    all_p1.append(p1); all_p2.append(p2); all_p25.append(p25)
    all_p3.append(p3); all_p35.append(p35); all_tgt.append(tgt)
    all_grp.append(np.full(M, s["si"], dtype=int))

X1 = np.vstack(all_p1); X2 = np.vstack(all_p2); X25 = np.vstack(all_p25)
X3 = np.vstack(all_p3); X35 = np.vstack(all_p35)
Y = np.vstack(all_tgt); G = np.concatenate(all_grp)
print(f"Extracted {len(Y):,} evaluation windows across {len(np.unique(G))} sessions.", flush=True)
print(f"Feature dims: P1={X1.shape[1]}, P2={X2.shape[1]}, P2.5={X25.shape[1]}, P3={X3.shape[1]}, P3.5={X35.shape[1]}", flush=True)

gkf = GroupKFold(n_splits=5)
g1, g2, g25, g3, g35 = [], [], [], [], []
for f, (tr, te) in enumerate(gkf.split(X1, groups=G)):
    Ytr, Yte = Y[tr], Y[te]
    pg = np.tile(Ytr.mean(0, keepdims=True), (len(te),1)); pg /= pg.sum(1, keepdims=True)
    jb = js_vec(Yte, pg)
    def gain(X_tr, X_te):
        p = np.clip(Ridge(5.0).fit(X_tr, Ytr).predict(X_te), 1e-4, 1.0)
        p /= p.sum(1, keepdims=True)
        return ((jb - js_vec(Yte, p)) / jb) * 100.0
    g1.append(gain(X1[tr], X1[te]))
    g2.append(gain(X2[tr], X2[te]))
    g25.append(gain(X25[tr], X25[te]))
    g3.append(gain(X3[tr], X3[te]))
    g35.append(gain(X35[tr], X35[te]))
    print(f"  Fold {f+1}/5: P2={g2[-1]:.2f}% | P2.5(Pose)={g25[-1]:.2f}% | P3(Syll)={g3[-1]:.2f}% | P3.5(Both)={g35[-1]:.2f}%", flush=True)

ci = lambda g: 1.96*np.std(g)/np.sqrt(5)
print("\n" + "="*85)
print("EXACT DISSECTION: CANONICAL P2 vs CONTINUOUS POSE vs DISCRETE SYLLABLE")
print("="*85)
print("P_0 (Blind Null):                                     +  0.00% ± 0.00%")
print(f"P_1 (Time Drift):                        [{X1.shape[1]:3d}-D]  + {np.mean(g1):5.2f}% ± {ci(g1):4.2f}%")
print(f"P_2 (Space+Height+Curvature over W):     [{X2.shape[1]:3d}-D]  + {np.mean(g2):5.2f}% ± {ci(g2):4.2f}%")
print(f"P_2.5 (P2 + Instantaneous 30-D Pose):    [{X25.shape[1]:3d}-D]  + {np.mean(g25):5.2f}% ± {ci(g25):4.2f}%")
print(f"P_3 (P2 + Discrete Syllable One-Hot):    [{X3.shape[1]:3d}-D]  + {np.mean(g3):5.2f}% ± {ci(g3):4.2f}%")
print(f"P_3.5 (P2 + Pose + Syllable):            [{X35.shape[1]:3d}-D]  + {np.mean(g35):5.2f}% ± {ci(g35):4.2f}%")
print("="*85)
