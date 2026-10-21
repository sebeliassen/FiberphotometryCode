#!/usr/bin/env python3
"""
evaluate_pilot_static_comparison.py
===================================
Benchmarking script to evaluate within-syllable dorsal pose homogeneity (Test 1)
and PCA variance distribution across:
  1. onecam_k240k (1-camera baseline)
  2. tandem_old_seed404 (dynamic fusion, 23kp)
  3. pilot_static_21kp (static affine fusion, 17 dorsal + 4 paws)
  4. pilot_static_23kp (static affine fusion, 17 dorsal + 4 paws + 2 bot anchors)

Evaluated strictly on the 10 shared pilot sessions.
"""
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from fpcode.common.paths import get_repo_root, resolve_data_path

REPO = Path(str(get_repo_root()))
EXP = REPO / 'studies/multiview_experiments'
PILOT_21 = resolve_data_path('data/models/pilot_static_21kp')
PILOT_23 = resolve_data_path('data/models/pilot_static_23kp')
OLD_TANDEM = resolve_data_path('data/models/kpms_tandem_cohort_continuous_calibrated')
K240K = REPO / 'vendor/keypoint_moseq_syll/karen_project_2/final_karen_model_k240k'

OUT_DIR = EXP / "04_reports_and_figures/audits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DORSAL = [
    "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back", "mouse_center",
    "mid_backend", "mid_backend2", "mid_backend3", "tail_base", "left_shoulder",
    "left_midside", "left_hip", "right_shoulder", "right_midside", "right_hip",
    "head_midpoint"
]

PILOT_SESSIONS = [
    "2026-03-05T11_01_01_top_left",
    "2026-03-05T12_46_18_top_left",
    "2026-03-06T09_48_02_top_left",
    "2026-03-06T11_14_44_top_left",
    "2026-03-05T11_01_01_bottom_left",
    "2026-03-05T12_46_18_bottom_left",
    "2026-03-06T09_48_02_bottom_left",
    "2026-03-05T11_01_01_bottom_right",
    "2026-03-05T12_46_18_bottom_right",
    "2026-03-06T09_48_02_bottom_right"
]

SNIP_LEN = 30
SNIP_STEP = 5
MAX_PER_SYL = 300
CONF_THRESH = 0.5
RNG = np.random.default_rng(404)


def label_arrays(res_path, vfi_path, key_to_base, n_by_base, upsample=False):
    """Loads syllable labels mapped to raw continuous top camera frame index."""
    out = {}
    with h5py.File(res_path, "r") as res, np.load(vfi_path) as vfi:
        for k in res.keys():
            b = key_to_base(k)
            if b not in n_by_base:
                continue
            lab = out.setdefault(b, np.full(n_by_base[b], -1, dtype=np.int16))
            idx = vfi[k][:]
            syl = res[k]["syllable"][:]
            n = min(len(idx), len(syl))
            idx, syl = idx[:n], syl[:n]
            ok = (idx < len(lab)) & (idx >= 0)
            lab[idx[ok]] = syl[ok]
            if upsample:  # 1-cam is 30fps (downsampled by 2): propagate to odd frame
                idx2 = idx[ok] + 1
                ok2 = idx2 < len(lab)
                lab[idx2[ok2]] = syl[ok][ok2]
    return out


def ego_dorsal_from_h5(df):
    """Computes egocentric dorsal coordinates (17 keypoints x 2) aligned to nose-tail axis."""
    sc = df.columns.levels[0][0]
    px = np.stack([df[(sc, b, "x")].to_numpy() for b in DORSAL], axis=1)
    py = np.stack([df[(sc, b, "y")].to_numpy() for b in DORSAL], axis=1)
    conf = np.stack([df[(sc, b, "likelihood")].to_numpy() for b in DORSAL], axis=1)
    XY = np.stack([px, py], axis=-1).astype(np.float32)  # (N, 17, 2)

    ok = np.isfinite(XY).all(axis=(1, 2)) & (conf >= CONF_THRESH).all(axis=1)
    centroid = XY.mean(axis=1)
    nose_idx = DORSAL.index("nose")
    tail_idx = DORSAL.index("tail_base")
    heading_vec = XY[:, nose_idx] - XY[:, tail_idx]
    theta = np.arctan2(heading_vec[:, 1], heading_vec[:, 0])

    cos_th, sin_th = np.cos(-theta), np.sin(-theta)
    rot_matrix = np.stack([
        np.stack([cos_th, -sin_th], axis=-1),
        np.stack([sin_th, cos_th], axis=-1)
    ], axis=-2)  # (N, 2, 2)

    centered = XY - centroid[:, None, :]
    ego = np.einsum("nij,nkj->nki", rot_matrix, centered)
    ego[~ok] = np.nan
    return ego.reshape(len(ego), -1)  # (N, 34)


def runs(labels):
    """Find contiguous runs of non-negative labels."""
    changes = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(labels)]
    return [(s, e, labels[s]) for s, e in zip(starts, ends) if labels[s] >= 0]


def compute_r2_and_kmeans_ceiling(snippets, labels):
    """Computes within-cluster R2 and MiniBatchKMeans ceiling with matching K."""
    grand_mean = snippets.mean(axis=0)
    total_ss = ((snippets - grand_mean) ** 2).sum()

    within_ss = 0.0
    unique_syls = np.unique(labels)
    K = len(unique_syls)

    for syl in unique_syls:
        cluster = snippets[labels == syl]
        cluster_mean = cluster.mean(axis=0)
        within_ss += ((cluster - cluster_mean) ** 2).sum()

    km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=404, batch_size=4096).fit(snippets)
    kmeans_within_ss = km.inertia_

    r2_labels = 1.0 - (within_ss / total_ss)
    r2_kmeans = 1.0 - (kmeans_within_ss / total_ss)
    ratio = r2_labels / r2_kmeans if r2_kmeans > 0 else 0.0

    return r2_labels, r2_kmeans, ratio, K


def main():
    print("=" * 80)
    print("🎯 MULTI-MODEL COMPARATIVE EVALUATION ON 10 PILOT SESSIONS")
    print("=" * 80)

    # 1. Load dorsal coordinates from fused files
    print("\n1. Loading egocentric dorsal coordinates for 10 pilot sessions...")
    fused_dir = resolve_data_path('data/fused/multiview/pilot_fused_static_21kp')
    n_by_base = {}
    ego_by_base = {}

    for sid in PILOT_SESSIONS:
        h5_path = fused_dir / f"{sid}_tandem_21kp.h5"
        if not h5_path.exists():
            print(f"  Warning: {h5_path} does not exist!")
            continue
        df = pd.read_hdf(h5_path)
        n_by_base[sid] = len(df)
        ego_by_base[sid] = ego_dorsal_from_h5(df)
        print(f"  Loaded {sid}: {len(df):,} frames")

    # 2. Define candidate models
    models = {
        "onecam_k240k": {
            "res": K240K / "results.h5",
            "vfi": K240K / "video_frame_indexes.npz",
            "key_to_base": lambda k: re.search(r"CenterTopCamVid_(.+?)DLC", k).group(1),
            "upsample": True,
            "description": "Single-Cam Baseline (17 Dorsal kp, 30fps x2)"
        },
        "old_tandem_seed404": {
            "res": OLD_TANDEM / "tandem_calibrated_50ar_200slds_seed404_20261002_012237_940073/results.h5",
            "vfi": OLD_TANDEM / "video_frame_indexes.npz",
            "key_to_base": lambda k: re.sub(r"_seg\d+$", "", k),
            "upsample": False,
            "description": "Dynamic Re-anchored Fusion (23 kp)"
        },
        "pilot_static_21kp": {
            "res": PILOT_21 / "pilot_static_21kp_seed404/results.h5",
            "vfi": PILOT_21 / "video_frame_indexes.npz",
            "key_to_base": lambda k: re.sub(r"_seg\d+$", "", k),
            "upsample": False,
            "description": "Static Affine Fusion Arm A (21 kp: 17 dorsal + 4 paws)"
        },
        "pilot_static_23kp": {
            "res": PILOT_23 / "pilot_static_23kp_seed404/results.h5",
            "vfi": PILOT_23 / "video_frame_indexes.npz",
            "key_to_base": lambda k: re.sub(r"_seg\d+$", "", k),
            "upsample": False,
            "description": "Static Affine Fusion Arm B (23 kp: 17 dorsal + 4 paws + 2 anchors)"
        },
        "pilot_static_21kp_k12": {
            "res": PILOT_21 / "pilot_static_21kp_k12_seed404/results.h5",
            "vfi": PILOT_21 / "video_frame_indexes.npz",
            "key_to_base": lambda k: re.sub(r"_seg\d+$", "", k),
            "upsample": False,
            "description": "Static Affine 21kp Calibrated Kappa (1.5e12)"
        },
    }

    # 3. Load syllable labels for available models
    active_labels = {}
    for mname, mcfg in models.items():
        if mcfg["res"].exists() and mcfg["vfi"].exists():
            print(f"  Loading labels for {mname} ({mcfg['description']})...")
            labs = label_arrays(mcfg["res"], mcfg["vfi"], mcfg["key_to_base"], n_by_base, mcfg["upsample"])
            active_labels[mname] = labs
        else:
            print(f"  Skipping {mname} (results.h5 not yet ready at {mcfg['res']})")

    # 4. Find intersection mask of valid frames across all active models for fair comparison
    print(f"\n2. Computing common frame support across {len(active_labels)} active models...")
    common_mask = {}
    for sid in n_by_base.keys():
        mask = np.ones(n_by_base[sid], dtype=bool)
        for mname in active_labels:
            if sid in active_labels[mname]:
                mask &= (active_labels[mname][sid] >= 0)
            else:
                mask[:] = False
        common_mask[sid] = mask
        print(f"  {sid}: {mask.sum():,} common labeled frames ({mask.mean()*100:.1f}%)")

    # 5. Extract snippets and run Test 1 for each active model
    print("\n" + "=" * 80)
    print("📊 TEST 1 RESULTS: WITHIN-SYLLABLE DORSAL POSE COHESION (R²)")
    print("=" * 80)

    results_table = {}
    for mname, labs in active_labels.items():
        snippets, labels_list, bout_durs = [], [], []

        for sid, E in ego_by_base.items():
            if sid not in labs:
                continue
            lab = labs[sid].copy()
            # Restrict strictly to common support frames
            lab[~common_mask[sid]] = -1

            for s, e, y in runs(lab):
                bout_durs.append(e - s)
                if e - s < SNIP_LEN:
                    continue
                # Sample snippet with SNIP_STEP
                snip = E[s:s + SNIP_LEN:SNIP_STEP]
                if np.isnan(snip).any():
                    continue
                snippets.append(snip.ravel())
                labels_list.append(y)

        if len(snippets) == 0:
            print(f"  {mname}: No valid snippets found!")
            continue

        S = np.array(snippets, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int32)

        # Subsample to at most MAX_PER_SYL to prevent frequent syllables from dominating
        keep_indices = []
        for syl in np.unique(y):
            syl_idx = np.flatnonzero(y == syl)
            if len(syl_idx) > MAX_PER_SYL:
                syl_idx = RNG.choice(syl_idx, MAX_PER_SYL, replace=False)
            keep_indices.append(syl_idx)
        keep = np.concatenate(keep_indices)
        S, y = S[keep], y[keep]

        # Filter out rare syllables (< 10 instances)
        vals, counts = np.unique(y, return_counts=True)
        valid_syls = vals[counts >= 10]
        valid_mask = np.isin(y, valid_syls)
        S, y = S[valid_mask], y[valid_mask]

        r2, r2_km, ratio, K = compute_r2_and_kmeans_ceiling(S, y)
        median_dur_frames = float(np.median(bout_durs)) if bout_durs else 0.0
        median_dur_ms = median_dur_frames * (1000.0 / 60.0)

        results_table[mname] = {
            "description": models[mname]["description"],
            "n_snippets": int(len(y)),
            "n_active_syllables": int(K),
            "R2_labels": float(r2),
            "R2_kmeans_same_K": float(r2_km),
            "R2_ratio": float(ratio),
            "median_bout_frames_60fps": median_dur_frames,
            "median_bout_ms": median_dur_ms
        }

        print(f"\nModel: {mname}")
        print(f"  Description:         {models[mname]['description']}")
        print(f"  Snippets Evaluated:  {len(y):,}")
        print(f"  Active Syllables:    {K}")
        print(f"  Dorsal R²:           {r2*100:.2f}%")
        print(f"  K-means Ceiling R²:  {r2_km*100:.2f}%")
        print(f"  Same-K R² Ratio:     {ratio:.4f}")
        print(f"  Median Bout Length:  {median_dur_frames:.1f} frames ({median_dur_ms:.1f} ms)")

    # 6. Save JSON
    out_json = OUT_DIR / "pilot_static_comparison_results.json"
    with open(out_json, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"\nSaved results summary to {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
