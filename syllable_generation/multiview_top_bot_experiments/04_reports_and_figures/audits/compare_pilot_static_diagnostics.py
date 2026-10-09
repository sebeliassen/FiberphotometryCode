#!/usr/bin/env python3
"""
compare_pilot_static_diagnostics.py
===================================
Direct diagnostic comparison of dorsal pose dispersion (R^2 and R^2 ratio vs same-K kmeans)
and PCA variance allocation across:
  1. onecam_k240k (1-cam baseline)
  2. tandem_seed404 (Legacy dynamic calibration 23kp)
  3. pilot_static_21kp_seed404 (Static RANSAC 21kp: dorsal + 4 paws)
  4. pilot_static_23kp_seed404 (Static RANSAC 23kp: dorsal + 4 paws + bot anchors)

Evaluated strictly on the 10 pilot benchmark sessions across Arenas 1, 2, and 3.
"""
import json
import re
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import MiniBatchKMeans

REPO = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP = REPO / "syllable_generation/multiview_top_bot_experiments"
PILOT_FUSED_21 = EXP / "data/pilot_fused_static_21kp"
OUT = EXP / "04_reports_and_figures/audits"

# Models to evaluate
ONE_CAM_DIR = REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/karen_project_2/final_karen_model_k240k"
LEGACY_DIR = EXP / "kpms_tandem_cohort_continuous_calibrated"
LEGACY_RES = LEGACY_DIR / "tandem_calibrated_50ar_200slds_seed404_20261002_012237_940073/results.h5"
LEGACY_VFI = LEGACY_DIR / "video_frame_indexes.npz"

P21_DIR = EXP / "pilot_static_21kp"
P21_RES = P21_DIR / "pilot_static_21kp_seed404/results.h5"
P21_VFI = P21_DIR / "video_frame_indexes.npz"

P23_DIR = EXP / "pilot_static_23kp"
P23_RES = P23_DIR / "pilot_static_23kp_seed404/results.h5"
P23_VFI = P23_DIR / "video_frame_indexes.npz"

DORSAL = [
    "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back", "mouse_center", "mid_backend",
    "mid_backend2", "mid_backend3", "tail_base", "left_shoulder", "left_midside", "left_hip",
    "right_shoulder", "right_midside", "right_hip", "head_midpoint"
]
SNIP_LEN, SNIP_STEP, MAX_PER_SYL, CONF = 30, 5, 300, 0.5
rng = np.random.default_rng(42)


def label_arrays(res, vfi, key_to_base, n_by_base, upsample):
    out = {}
    for k in res.keys():
        b = key_to_base(k)
        if b not in n_by_base:
            continue
        lab = out.setdefault(b, np.full(n_by_base[b], -1, np.int16))
        idx = vfi[k][:]
        syl = res[k]["syllable"][:]
        n = min(len(idx), len(syl))
        idx, syl = idx[:n], syl[:n]
        ok = idx < len(lab)
        lab[idx[ok]] = syl[ok]
        if upsample:  # 1-cam was downsampled x2: copy to skipped odd frames
            idx2 = idx[ok] + 1
            ok2 = idx2 < len(lab)
            lab[idx2[ok2]] = syl[ok][ok2]
    return out


def ego_dorsal(df):
    sc = df.columns.levels[0][0]
    P = (
        np.stack([df[(sc, b, "x")].to_numpy() for b in DORSAL], 1),
        np.stack([df[(sc, b, "y")].to_numpy() for b in DORSAL], 1)
    )
    L = np.stack([df[(sc, b, "likelihood")].to_numpy() for b in DORSAL], 1)
    XY = np.stack(P, -1).astype(np.float32)  # n, 17, 2
    ok = np.isfinite(XY).all((1, 2)) & (L >= CONF).all(1)
    c = XY.mean(1)
    v = XY[:, DORSAL.index("nose")] - XY[:, DORSAL.index("tail_base")]
    th = np.arctan2(v[:, 1], v[:, 0])
    cs, sn = np.cos(-th), np.sin(-th)
    R = np.stack([np.stack([cs, -sn], -1), np.stack([sn, cs], -1)], -2)
    E = np.einsum("nij,nkj->nki", R, XY - c[:, None])
    E[~ok] = np.nan
    return E.reshape(len(E), -1)


def runs(lab):
    ch = np.flatnonzero(np.diff(lab) != 0) + 1
    st = np.r_[0, ch]
    en = np.r_[ch, len(lab)]
    return [(s, e, lab[s]) for s, e in zip(st, en) if lab[s] >= 0]


def r2_and_ceiling(S, y):
    tot = ((S - S.mean(0)) ** 2).sum()
    within = 0.0
    for s in np.unique(y):
        g = S[y == s]
        within += ((g - g.mean(0)) ** 2).sum()
    K = len(np.unique(y))
    km = MiniBatchKMeans(K, n_init=3, random_state=0, batch_size=4096).fit(S)
    return 1 - within / tot, 1 - km.inertia_ / tot, K


def center_embedding(n):
    X = np.tril(np.ones((n, n)), k=-1)[1:]
    X = np.eye(n)[1:] - X / X.sum(1)[:, None]
    X = X / np.sqrt((X**2).sum(1))[:, None]
    return X.T


def compute_pca_shares(pca_p, cfg_p):
    pca = joblib.load(pca_p)
    with open(cfg_p) as f:
        cfg = yaml.safe_load(f)
    bps = cfg["use_bodyparts"]
    n_bps = len(bps)
    d = 2
    Gamma = center_embedding(n_bps)
    comps = pca.components_
    var_exp = pca.explained_variance_ratio_

    bp_var = np.zeros(n_bps)
    for c_idx in range(min(10, comps.shape[0])):
        c_emb = comps[c_idx].reshape(n_bps - 1, d)
        c_orig = Gamma @ c_emb
        bp_power = (c_orig**2).sum(axis=1)
        bp_var += bp_power * var_exp[c_idx]

    bp_pct = 100 * bp_var / bp_var.sum()
    dorsal_mask = np.array([bp not in ['bot_nose_tip', 'bot_tail_base', 'left_front_paw', 'right_front_paw', 'left_hind_paw', 'right_hind_paw'] for bp in bps])
    paws_mask = np.array([bp in ['left_front_paw', 'right_front_paw', 'left_hind_paw', 'right_hind_paw'] for bp in bps])
    anchors_mask = np.array([bp in ['bot_nose_tip', 'bot_tail_base'] for bp in bps])

    return {
        "dorsal_share_pct": float(bp_pct[dorsal_mask].sum()),
        "paws_share_pct": float(bp_pct[paws_mask].sum()),
        "anchors_share_pct": float(bp_pct[anchors_mask].sum()) if anchors_mask.any() else 0.0,
        "paw_detail": {bp: float(pct) for bp, pct in zip(np.array(bps)[paws_mask], bp_pct[paws_mask])}
    }


def main():
    print("Loading pilot fused sessions...")
    files = sorted(PILOT_FUSED_21.glob("*_tandem_21kp.h5"))
    base = lambda p: p.name.replace("_tandem_21kp.h5", "")
    n_by_base, ego = {}, {}
    for f in files:
        df = pd.read_hdf(f)
        b = base(f)
        n_by_base[b] = len(df)
        ego[b] = ego_dorsal(df)
    print(f"Loaded {len(files)} pilot sessions.")

    # Load label arrays for each model
    tb = lambda k: re.sub(r"_seg\d+$", "", k)
    kb = lambda k: re.search(r"CenterTopCamVid_(.+?)DLC", k).group(1)

    t_res = h5py.File(LEGACY_RES, 'r'); t_vfi = np.load(LEGACY_VFI)
    k_res = h5py.File(ONE_CAM_DIR / "results.h5", 'r'); k_vfi = np.load(ONE_CAM_DIR / "video_frame_indexes.npz")
    p21_res = h5py.File(P21_RES, 'r'); p21_vfi = np.load(P21_VFI)
    p23_res = h5py.File(P23_RES, 'r'); p23_vfi = np.load(P23_VFI)

    lab_t = label_arrays(t_res, t_vfi, tb, n_by_base, False)
    lab_k = label_arrays(k_res, k_vfi, kb, n_by_base, True)
    lab_p21 = label_arrays(p21_res, p21_vfi, tb, n_by_base, False)
    lab_p23 = label_arrays(p23_res, p23_vfi, tb, n_by_base, False)

    common = sorted(set(lab_t) & set(lab_k) & set(lab_p21) & set(lab_p23))
    print(f"Found {len(common)} sessions common across all 4 models: {common}")

    # Build mutual support mask so all models are evaluated on the exact same valid frames
    eval_models = [
        ("onecam_k240k", lab_k),
        ("tandem_legacy_seed404", lab_t),
        ("pilot_static_21kp", lab_p21),
        ("pilot_static_23kp", lab_p23)
    ]

    r2_results = {}
    for name, labs in eval_models:
        snips, ys, durs_all = [], [], []
        for b in common:
            lab = labs[b].copy()
            # mutual frame support
            valid_all = (lab_t[b] >= 0) & (lab_k[b] >= 0) & (lab_p21[b] >= 0) & (lab_p23[b] >= 0)
            lab[~valid_all] = -1
            E = ego[b]
            for s, e, y in runs(lab):
                durs_all.append(e - s)
                if e - s < SNIP_LEN:
                    continue
                w = E[s:s + SNIP_LEN:SNIP_STEP]
                if np.isnan(w).any():
                    continue
                snips.append(w.ravel())
                ys.append(y)

        S, y = np.array(snips, np.float32), np.array(ys)
        keep = np.concatenate([rng.permutation(np.flatnonzero(y == s))[:MAX_PER_SYL] for s in np.unique(y)])
        S, y = S[keep], y[keep]
        vals, cnt = np.unique(y, return_counts=True)
        good = vals[cnt >= 10]
        m = np.isin(y, good)
        S, y = S[m], y[m]

        r2, r2km, K = r2_and_ceiling(S, y)
        r2_results[name] = {
            "n_snippets": int(len(y)),
            "n_syllables": int(K),
            "R2_labels": float(r2),
            "R2_kmeans_same_K": float(r2km),
            "R2_ratio": float(r2 / r2km),
            "median_bout_raw60fps_frames": float(np.median(durs_all))
        }
        print(f"\n[{name}]")
        print(f"  Snippets: {len(y)}, Syllables K: {K}")
        print(f"  R2 Labels: {r2:.4f}, R2 KMeans Ceiling: {r2km:.4f} -> Ratio: {r2/r2km:.4f}")
        print(f"  Median bout: {np.median(durs_all):.1f} frames")

    # PCA share summaries
    pca_summaries = {
        "tandem_legacy_seed404": compute_pca_shares(LEGACY_DIR / "pca.p", LEGACY_DIR / "config.yml"),
        "pilot_static_21kp": compute_pca_shares(P21_DIR / "pca.p", P21_DIR / "config.yml"),
        "pilot_static_23kp": compute_pca_shares(P23_DIR / "pca.p", P23_DIR / "config.yml")
    }

    final_report = {
        "r2_pose_dispersion": r2_results,
        "pca_variance_allocation": pca_summaries
    }

    out_file = OUT / "pilot_static_vs_legacy_comparison.json"
    out_file.write_text(json.dumps(final_report, indent=2))
    print(f"\nSaved comparison audit to {out_file}")

if __name__ == "__main__":
    main()
