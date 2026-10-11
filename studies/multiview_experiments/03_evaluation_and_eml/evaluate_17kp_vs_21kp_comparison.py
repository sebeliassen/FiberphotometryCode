#!/usr/bin/env python3
"""
evaluate_17kp_vs_21kp_comparison.py
=============================================================================
Comprehensive Evaluation Suite comparing 17kp vs 21kp models on the 10 Pilot Sessions:
  - Arm 1: 17kp baseline (17 dorsal TopCam keypoints)
  - Arm 2: 21kp Static (17 dorsal + 4 static affine ventral paws)
  - Arm 3: 21kp Dynamic (17 dorsal + 4 dynamic paws, bottom anchors removed)

Metrics Evaluated:
  1. Convergence & Syllable Landscape Stability:
     - Consecutive-snapshot Adjusted Rand Index (ARI) and AMI across Gibbs iterations
     - Hamming label-flip rate (% of frames changing syllable between snapshots)
     - Late-stage stabilization metric (average ARI and flip rate across late snapshots >= 100)
  2. Posture & Kinematic Expressiveness:
     - Absolute dorsal variance retained in top 10 PCs (px^2)
     - Absolute paw variance retained in top 10 PCs (px^2)
     - Dorsal vs Paw percentage share in top 10 PCs
  3. Syllable Granularity & Homogeneity:
     - Within-syllable dorsal pose snippet variance explained (R^2)
     - Same-K K-means ceiling ratio (Model R^2 / KMeans R^2)
  4. Temporal Bout Dynamics:
     - Active syllable count (>= 0.1% usage)
     - Median bout duration (ms and frames at 60 fps)
     - Usage distribution entropy
=============================================================================
"""
import argparse
import json
import pickle
import re
import sys
from itertools import groupby
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from fpcode.common.paths import get_repo_root, resolve_data_path

REPO_ROOT = Path(str(get_repo_root()))
EXP_DIR = REPO_ROOT / 'studies/multiview_experiments'
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

FPS = 60.0
CONF_THRESH = 0.4
SNIP_LEN = 10
SNIP_STEP = 2
MAX_PER_SYL = 1000

DORSAL_BPS = [
    'nose', 'left_ear_tip', 'right_ear_tip', 'neck',
    'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3',
    'tail_base', 'left_shoulder', 'left_midside', 'left_hip',
    'right_shoulder', 'right_midside', 'right_hip', 'head_midpoint'
]
PAW_BPS = ['left_front_paw', 'right_front_paw', 'left_hind_paw', 'right_hind_paw']

PILOT_SESSIONS = [
    '2026-03-05T11_01_01_top_left',
    '2026-03-05T12_46_18_top_left',
    '2026-03-06T09_48_02_top_left',
    '2026-03-06T11_14_44_top_left',
    '2026-03-05T11_01_01_bottom_left',
    '2026-03-05T12_46_18_bottom_left',
    '2026-03-06T09_48_02_bottom_left',
    '2026-03-05T11_01_01_bottom_right',
    '2026-03-05T12_46_18_bottom_right',
    '2026-03-06T09_48_02_bottom_right'
]


def load_model_syllables_and_latents(results_path):
    """Loads syllables, latents, and centroids from results.h5."""
    syllables = {}
    latents = {}
    centroids = {}
    with h5py.File(results_path, "r") as f:
        for k in f.keys():
            if isinstance(f[k], h5py.Group) and "syllable" in f[k]:
                syllables[k] = f[k]["syllable"][:]
                if "latent_state" in f[k]:
                    latents[k] = f[k]["latent_state"][:]
                if "centroid" in f[k]:
                    centroids[k] = f[k]["centroid"][:]
    return syllables, latents, centroids


def evaluate_checkpoint_convergence(checkpoint_path):
    """Computes pairwise ARI and label flip rate across all snapshots."""
    if not Path(checkpoint_path).exists():
        return {}

    with h5py.File(checkpoint_path, "r") as f:
        if "model_snapshots" not in f:
            return {}
        snapshots = np.sort([int(i) for i in f["model_snapshots"]])
        if len(snapshots) < 2:
            return {}

        z_seqs = {}
        for s in snapshots:
            grp_path = f"model_snapshots/{s}/states/z"
            if grp_path in f:
                z_seqs[s] = f[grp_path][:].ravel()

    avail_snaps = sorted(z_seqs.keys())
    if len(avail_snaps) < 2:
        return {}

    consecutive_metrics = []
    for i in range(len(avail_snaps) - 1):
        s_prev = avail_snaps[i]
        s_curr = avail_snaps[i + 1]
        zp = z_seqs[s_prev]
        zc = z_seqs[s_curr]
        mask = (zp >= 0) & (zc >= 0)
        ari = float(adjusted_rand_score(zp[mask], zc[mask]))
        ami = float(adjusted_mutual_info_score(zp[mask], zc[mask]))
        flip = float(np.mean(zp[mask] != zc[mask])) * 100.0
        consecutive_metrics.append({
            "from_iter": int(s_prev),
            "to_iter": int(s_curr),
            "ari": ari,
            "ami": ami,
            "label_change_pct": flip
        })

    # Summary of late-stage convergence (transitions where from_iter >= 100 or last 2 transitions)
    late_metrics = [m for m in consecutive_metrics if m["from_iter"] >= 100]
    if not late_metrics:
        late_metrics = consecutive_metrics[-2:] if len(consecutive_metrics) >= 2 else consecutive_metrics

    late_ari = float(np.mean([m["ari"] for m in late_metrics])) if late_metrics else float("nan")
    late_flip = float(np.mean([m["label_change_pct"] for m in late_metrics])) if late_metrics else float("nan")

    return {
        "snapshots": [int(s) for s in avail_snaps],
        "consecutive": consecutive_metrics,
        "late_stage_ari": late_ari,
        "late_stage_flip_pct": late_flip
    }


def compute_bout_statistics(syllables_dict, fps=60.0):
    """Computes syllable usage, active count, and bout duration statistics."""
    all_bouts = []
    syll_counts = {}
    total_frames = 0

    for k, seq in syllables_dict.items():
        valid_seq = seq[seq >= 0]
        total_frames += len(valid_seq)
        for syll, group in groupby(valid_seq):
            length = len(list(group))
            all_bouts.append(length)
            syll_counts[syll] = syll_counts.get(syll, 0) + length

    if not all_bouts:
        return {}

    bouts_arr = np.array(all_bouts)
    durations_ms = (bouts_arr / fps) * 1000.0

    active_sylls = [s for s, c in syll_counts.items() if (c / total_frames) >= 0.001]
    probs = np.array(list(syll_counts.values())) / float(total_frames)
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))

    return {
        "total_frames": int(total_frames),
        "total_bouts": int(len(bouts_arr)),
        "active_syllable_count": int(len(active_sylls)),
        "entropy_bits": entropy,
        "median_bout_ms": float(np.median(durations_ms)),
        "mean_bout_ms": float(np.mean(durations_ms)),
        "p10_bout_ms": float(np.percentile(durations_ms, 10)),
        "p90_bout_ms": float(np.percentile(durations_ms, 90)),
        "median_bout_frames": float(np.median(bouts_arr)),
    }


def load_raw_dorsal_trajectories():
    """Loads centered egocentric dorsal coordinates for the 10 pilot sessions."""
    static_data_dir = resolve_data_path('data/fused/multiview/pilot_fused_static_21kp')
    ego_dorsal_dict = {}

    for sid in PILOT_SESSIONS:
        fpath = static_data_dir / f"{sid}_tandem_21kp.h5"
        if not fpath.exists():
            continue
        df = pd.read_hdf(fpath)
        sc = df.columns.levels[0][0]

        P = (
            np.stack([df[(sc, b, "x")].to_numpy() for b in DORSAL_BPS], 1),
            np.stack([df[(sc, b, "y")].to_numpy() for b in DORSAL_BPS], 1)
        )
        L = np.stack([df[(sc, b, "likelihood")].to_numpy() for b in DORSAL_BPS], 1)
        XY = np.stack(P, -1).astype(np.float32)  # (N, 17, 2)
        ok = np.isfinite(XY).all((1, 2)) & (L >= CONF_THRESH).all(1)

        c = XY.mean(1)
        v = XY[:, DORSAL_BPS.index("nose")] - XY[:, DORSAL_BPS.index("tail_base")]
        th = np.arctan2(v[:, 1], v[:, 0])
        cs, sn = np.cos(-th), np.sin(-th)
        R = np.stack([np.stack([cs, -sn], -1), np.stack([sn, cs], -1)], -2)
        E = np.einsum("nij,nkj->nki", R, XY - c[:, None])
        E[~ok] = np.nan
        ego_dorsal_dict[sid] = E.reshape(len(E), -1)

    return ego_dorsal_dict


def reconstruct_full_session_syllables(syllables_dict, vfi_path, ego_dorsal_dict):
    """Reconstructs continuous per-session label arrays from segment syllables."""
    if not Path(vfi_path).exists():
        return {}
    vfi = np.load(vfi_path)
    session_labels = {
        sid: np.full(len(ego_dorsal_dict[sid]), -1, dtype=np.int32)
        for sid in PILOT_SESSIONS if sid in ego_dorsal_dict
    }

    for seg_name, syll_arr in syllables_dict.items():
        if seg_name in vfi:
            frame_indices = vfi[seg_name]
            sid = re.sub(r"_seg\d+$", "", seg_name)
            if sid in session_labels:
                target_len = len(session_labels[sid])
                valid_mask = frame_indices < target_len
                valid_indices = frame_indices[valid_mask]
                valid_len = min(len(syll_arr), len(valid_indices))
                session_labels[sid][valid_indices[:valid_len]] = syll_arr[:valid_len]

    return session_labels


def compute_dorsal_snippet_r2(session_labels_dict, ego_dorsal_dict, rng_seed=404):
    """Computes within-syllable dorsal pose snippet R^2 and KMeans ceiling."""
    rng = np.random.default_rng(rng_seed)
    snips = []
    ys = []

    for sid in PILOT_SESSIONS:
        if sid not in session_labels_dict or sid not in ego_dorsal_dict:
            continue
        lab = session_labels_dict[sid]
        E = ego_dorsal_dict[sid]

        # Segment contiguous runs of identical syllables
        ch = np.flatnonzero(np.diff(lab) != 0) + 1
        st = np.r_[0, ch]
        en = np.r_[ch, len(lab)]

        for s, e in zip(st, en):
            y = lab[s]
            if y < 0 or (e - s) < SNIP_LEN:
                continue
            w = E[s:s + SNIP_LEN:SNIP_STEP]
            if np.isnan(w).any():
                continue
            snips.append(w.ravel())
            ys.append(y)

    if not snips:
        return {"r2_labels": float("nan"), "r2_kmeans": float("nan"), "r2_ratio": float("nan"), "k": 0}

    S = np.array(snips, dtype=np.float32)
    y = np.array(ys, dtype=np.int32)

    # Subsample to avoid over-representation of dominant syllables
    keep = np.concatenate([
        rng.permutation(np.flatnonzero(y == s))[:MAX_PER_SYL]
        for s in np.unique(y)
    ])
    S = S[keep]
    y = y[keep]

    # Filter out rare syllables (< 10 snippets)
    vals, cnt = np.unique(y, return_counts=True)
    good = vals[cnt >= 10]
    m = np.isin(y, good)
    S = S[m]
    y = y[m]

    tot = float(((S - S.mean(0)) ** 2).sum())
    within = 0.0
    for s in np.unique(y):
        g = S[y == s]
        within += float(((g - g.mean(0)) ** 2).sum())

    K = len(np.unique(y))
    km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=0, batch_size=4096).fit(S)
    km_within = float(km.inertia_)

    r2_labels = 1.0 - (within / tot) if tot > 0 else float("nan")
    r2_km = 1.0 - (km_within / tot) if tot > 0 else float("nan")
    r2_ratio = (r2_labels / r2_km) if r2_km > 0 else float("nan")

    return {
        "n_snippets": int(len(S)),
        "k_syllables": int(K),
        "r2_labels": float(r2_labels),
        "r2_kmeans": float(r2_km),
        "r2_ratio": float(r2_ratio)
    }


def compute_pca_variance_metrics(arm_dir):
    """Computes absolute dorsal and paw variance retained in top 10 PCs."""
    pca_file = Path(arm_dir) / "pca.p"
    if not pca_file.exists():
        return {}

    with open(pca_file, "rb") as f:
        pca = pickle.load(f)

    if hasattr(pca, "components_"):
        comps = pca.components_[:10]  # (10, n_features)
        var = pca.explained_variance_[:10]  # (10,)
    elif isinstance(pca, dict) and "components" in pca:
        comps = pca["components"][:10]
        var = pca.get("variance", np.ones(10))[:10]
    else:
        return {}

    n_features = comps.shape[1]
    # Weight components by explained variance
    weighted_sq = (comps ** 2) * var[:, None]  # (10, n_features)

    if n_features == 32:  # 17 keypoints (16 free bodyparts x 2 coords)
        dorsal_var = float(weighted_sq.sum())
        paw_var = 0.0
        dorsal_pct = 100.0
        paw_pct = 0.0
    elif n_features >= 40:  # 21 keypoints (16 dorsal free + 4 paws free)
        dorsal_var = float(weighted_sq[:, :32].sum())
        paw_var = float(weighted_sq[:, 32:40].sum())
        tot = dorsal_var + paw_var
        dorsal_pct = float((dorsal_var / tot) * 100.0) if tot > 0 else 0.0
        paw_pct = float((paw_var / tot) * 100.0) if tot > 0 else 0.0
    else:
        dorsal_var = paw_var = dorsal_pct = paw_pct = float("nan")

    return {
        "abs_dorsal_variance_px2": dorsal_var,
        "abs_paw_variance_px2": paw_var,
        "dorsal_pct_share": dorsal_pct,
        "paw_pct_share": paw_pct,
    }


def evaluate_arm(arm_name, model_dir, ego_dorsal_dict):
    """Evaluates a single model arm comprehensively."""
    m_path = Path(model_dir)
    print("\n================================================================================")
    print(f"📊 EVALUATING ARM: {arm_name.upper()} ({m_path.name})")
    print("================================================================================")

    res_file = m_path / "results.h5"
    chk_file = m_path / "checkpoint.h5"
    arm_parent = m_path.parent
    vfi_path = arm_parent / "video_frame_indexes.npz"

    if not res_file.exists():
        print(f"  ❌ Error: {res_file} not found!")
        return None

    # 1. Syllables & Bouts
    sylls, latents, cents = load_model_syllables_and_latents(res_file)
    bout_stats = compute_bout_statistics(sylls, fps=FPS)
    print(f"  Active Syllables:   {bout_stats.get('active_syllable_count')}")
    print(f"  Median Duration:    {bout_stats.get('median_bout_ms', 0):.1f} ms ({bout_stats.get('median_bout_frames', 0):.1f} frames)")
    print(f"  Entropy:            {bout_stats.get('entropy_bits', 0):.2f} bits")

    # 2. Convergence ARI across Checkpoints
    conv_stats = evaluate_checkpoint_convergence(chk_file)
    print(f"  Snapshots saved:    {conv_stats.get('snapshots', [])}")
    print(f"  Late-Stage ARI:     {conv_stats.get('late_stage_ari', float('nan')):.4f}")
    print(f"  Late-Stage Flip:    {conv_stats.get('late_stage_flip_pct', float('nan')):.1f}%")

    # 3. PCA Variance Decomposition
    pca_stats = compute_pca_variance_metrics(arm_parent)
    print(f"  Abs Dorsal Var:     {pca_stats.get('abs_dorsal_variance_px2', 0):.2f} px^2")
    print(f"  Abs Paw Var:        {pca_stats.get('abs_paw_variance_px2', 0):.2f} px^2")
    print(f"  Dorsal Share:       {pca_stats.get('dorsal_pct_share', 0):.1f}% | Paw Share: {pca_stats.get('paw_pct_share', 0):.1f}%")

    # 4. Dorsal Snippet R^2 and Ceiling Ratio
    session_labels = reconstruct_full_session_syllables(sylls, vfi_path, ego_dorsal_dict)
    r2_stats = compute_dorsal_snippet_r2(session_labels, ego_dorsal_dict)
    print(f"  Dorsal Snippet R^2: {r2_stats.get('r2_labels', float('nan')):.4f}")
    print(f"  KMeans Ceiling R^2: {r2_stats.get('r2_kmeans', float('nan')):.4f}")
    print(f"  Same-K Ratio:       {r2_stats.get('r2_ratio', float('nan')):.4f}")

    return {
        "arm": arm_name,
        "model_dir": str(m_path),
        "bout_statistics": bout_stats,
        "convergence": conv_stats,
        "pca_variance": pca_stats,
        "dorsal_r2": r2_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate 17kp vs 21kp models.")
    parser.add_argument("--suffix", type=str, default="full200", help="Model suffix to locate")
    parser.add_argument("--seed", type=int, default=404, help="Model seed")
    parser.add_argument("--output-json", type=str, default=str(EXP_DIR / "04_reports_and_figures/pilot_17kp_vs_21kp_full_evaluation.json"))
    parser.add_argument("--output-csv", type=str, default=str(EXP_DIR / "04_reports_and_figures/pilot_17kp_vs_21kp_full_scorecard.csv"))
    args = parser.parse_args()

    # Preload raw dorsal trajectories once
    print("Loading raw dorsal trajectories for R^2 evaluation...")
    ego_dorsal_dict = load_raw_dorsal_trajectories()
    print(f"Loaded {len(ego_dorsal_dict)} sessions.")

    # Locate models
    arms = ["17kp", "21kp_static", "21kp_dynamic"]
    evaluated_arms = {}

    for arm in arms:
        arm_dir = EXP_DIR / f"pilot_comparison_{arm}"
        # Look for model with matching suffix
        model_name = f"model_{arm}_{args.suffix}_seed{args.seed}"
        target_dir = arm_dir / model_name
        if not target_dir.exists():
            # Fallback to most recent model in arm_dir
            candidates = [d for d in arm_dir.iterdir() if d.is_dir() and (d / "results.h5").exists()]
            if candidates:
                target_dir = sorted(candidates, key=lambda x: x.stat().st_mtime)[-1]

        if target_dir.exists() and (target_dir / "results.h5").exists():
            res = evaluate_arm(arm, target_dir, ego_dorsal_dict)
            if res:
                evaluated_arms[arm] = res
        else:
            print(f"⚠️ Model directory not found for {arm}: {target_dir}")

    # Build Comparative Scorecard Table
    scorecard_rows = []
    for arm, res in evaluated_arms.items():
        bs = res["bout_statistics"]
        cs = res["convergence"]
        ps = res["pca_variance"]
        r2 = res["dorsal_r2"]

        scorecard_rows.append({
            "Arm": arm,
            "Keypoints": 17 if arm == "17kp" else 21,
            "Active_Syllables": bs.get("active_syllable_count", np.nan),
            "Median_Duration_ms": bs.get("median_bout_ms", np.nan),
            "Median_Duration_frames": bs.get("median_bout_frames", np.nan),
            "Entropy_bits": bs.get("entropy_bits", np.nan),
            "Late_Stage_ARI": cs.get("late_stage_ari", np.nan),
            "Late_Stage_Flip_Pct": cs.get("late_stage_flip_pct", np.nan),
            "Dorsal_R2": r2.get("r2_labels", np.nan),
            "Same_K_Ceiling_Ratio": r2.get("r2_ratio", np.nan),
            "Abs_Dorsal_Variance_px2": ps.get("abs_dorsal_variance_px2", np.nan),
            "Abs_Paw_Variance_px2": ps.get("abs_paw_variance_px2", np.nan),
            "Dorsal_PCA_Share_Pct": ps.get("dorsal_pct_share", np.nan),
            "Paw_PCA_Share_Pct": ps.get("paw_pct_share", np.nan),
        })

    df_scorecard = pd.DataFrame(scorecard_rows)
    print("\n" + "=" * 90)
    print("🏆 PILOT 17KP VS 21KP COMPARATIVE SCORECARD")
    print("=" * 90)
    print(df_scorecard.to_string(index=False))
    print("=" * 90)

    # Save exports
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(evaluated_arms, indent=2))

    out_csv = Path(args.output_csv)
    df_scorecard.to_csv(out_csv, index=False)
    print(f"\nSaved evaluation metrics to: {out_json}")
    print(f"Saved scorecard to:        {out_csv}")


if __name__ == "__main__":
    main()
