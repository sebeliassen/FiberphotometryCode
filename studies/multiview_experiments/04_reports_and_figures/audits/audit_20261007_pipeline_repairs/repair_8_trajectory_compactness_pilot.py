#!/usr/bin/env python3
"""
repair_8_trajectory_compactness_pilot.py
========================================
Rigorous empirical benchmark of observed within-syllable trajectory compactness
versus matched random controls, separately evaluated for dorsal posture and paws.

Directly answers:
Are the density-selected examples significantly more stereotyped and consistent
than random trajectories in the actual observed fused tracking data?

Evaluates:
- Intra-syllable trajectory variance for dorsal keypoints (17 kps)
- Intra-syllable trajectory variance for paw keypoints (4 kps)
- Matched-duration random control variance
- Compactness ratio (Var_syll / Var_random): ratio < 1.0 indicates stereotypy.
"""
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from fpcode.common.paths import get_repo_root, resolve_data_path

REPO = Path(str(get_repo_root()))
EXP = REPO / 'studies/multiview_experiments'
AUDIT_DIR = EXP / "04_reports_and_figures/audits/audit_20261007_pipeline_repairs"
FUSED_DIR = resolve_data_path('data/fused/multiview/cohort_fused_static_21kp')

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from keypoint_moseq.util import filter_angle

BODYPARTS_21KP = [
    "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back",
    "mouse_center", "mid_backend", "mid_backend2", "mid_backend3",
    "tail_base", "left_shoulder", "left_midside", "left_hip",
    "right_shoulder", "right_midside", "right_hip", "head_midpoint",
    "left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"
]
DORSAL_IDXS = list(range(17))
PAW_IDXS = list(range(17, 21))

def interpolate_trajectory(traj: np.ndarray, target_len: int = 20) -> np.ndarray:
    """Interpolates trajectory of shape (T, K, 2) to target length T_target."""
    T, K, D = traj.shape
    if T == target_len:
        return traj
    t_orig = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, target_len)
    f = interp1d(t_orig, traj, axis=0, kind="linear")
    return f(t_new)

def extract_aligned_observed_trajectory(
    df: pd.DataFrame,
    start_frame: int,
    end_frame: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts egocentrically aligned observed trajectory for dorsal and paws.
    Returns (traj_dorsal, traj_paws) of shape (T, K_dorsal, 2) and (T, K_paws, 2).
    """
    sc = df.columns.levels[0][0]
    n_frames = end_frame - start_frame

    # Extract all 21 keypoints
    coords = np.zeros((n_frames, 21, 2), dtype=np.float32)
    for i, bp in enumerate(BODYPARTS_21KP):
        coords[:, i, 0] = df[(sc, bp, "x")].iloc[start_frame:end_frame].to_numpy()
        coords[:, i, 1] = df[(sc, bp, "y")].iloc[start_frame:end_frame].to_numpy()

    # If any NaN, interpolate linearly
    for i in range(21):
        for d in range(2):
            nan_mask = np.isnan(coords[:, i, d])
            if np.any(nan_mask):
                if np.all(nan_mask):
                    coords[:, i, d] = 0.0
                else:
                    valid_idx = np.where(~nan_mask)[0]
                    coords[:, i, d] = np.interp(np.arange(n_frames), valid_idx, coords[valid_idx, i, d])

    # Heading from tail_base (9) to nose (0)
    tail = coords[:, 9, :]
    nose = coords[:, 0, :]
    raw_head = np.arctan2(nose[:, 1] - tail[:, 1], nose[:, 0] - tail[:, 0])
    hd_filt = filter_angle(raw_head, size=9)

    # Egocentric alignment: Center at onset mouse_center (5) and rotate by -heading[0]
    cen0 = coords[0, 5, :].copy()
    rel_coords = coords - cen0[None, None, :]

    th0 = -hd_filt[0]
    R0 = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]], dtype=np.float32)
    ego_coords = rel_coords @ R0.T

    traj_dorsal = ego_coords[:, DORSAL_IDXS, :]
    traj_paws = ego_coords[:, PAW_IDXS, :]
    return traj_dorsal, traj_paws

def run_compactness_audit():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = AUDIT_DIR / "unified_syllable_selection_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print("Evaluating observed trajectory compactness vs matched random controls...")

    fused_cache = {}
    compactness_results = []

    for s_str, bouts in manifest["syllables"].items():
        syll = int(s_str)
        print(f"\n--- Syllable {syll}: {len(bouts)} bouts ---")

        dorsal_trajs = []
        paws_trajs = []
        durations = []

        # Load syllable bouts
        for b in bouts:
            sid = b["segment_name"].rsplit("_seg", 1)[0]
            fused_file = FUSED_DIR / f"{sid}_tandem_21kp.h5"
            if not fused_file.exists(): continue
            if sid not in fused_cache:
                fused_cache[sid] = pd.read_hdf(fused_file)
            df = fused_cache[sid]

            s_on = b["source_top_onset"]
            s_off = b["source_top_offset"]
            dur = s_off - s_on
            if dur < 5 or s_off > len(df): continue

            d_traj, p_traj = extract_aligned_observed_trajectory(df, s_on, s_off)
            dorsal_trajs.append(interpolate_trajectory(d_traj, target_len=20))
            paws_trajs.append(interpolate_trajectory(p_traj, target_len=20))
            durations.append(dur)

        if len(dorsal_trajs) < 4:
            continue

        dorsal_arr = np.array(dorsal_trajs) # (N, 20, 17, 2)
        paws_arr = np.array(paws_trajs)     # (N, 20, 4, 2)

        # Intra-syllable variance (mean squared deviation from syllable mean)
        d_mean = np.mean(dorsal_arr, axis=0, keepdims=True)
        p_mean = np.mean(paws_arr, axis=0, keepdims=True)

        var_dorsal_syll = float(np.mean(np.sum((dorsal_arr - d_mean)**2, axis=(1, 2, 3))))
        var_paws_syll = float(np.mean(np.sum((paws_arr - p_mean)**2, axis=(1, 2, 3))))

        # Matched Random Control
        # Sample random segments of matched durations from available sessions
        rand_dorsal_trajs = []
        rand_paws_trajs = []
        all_sids = list(fused_cache.keys())

        np.random.seed(42 + syll)
        for dur in durations:
            rand_sid = np.random.choice(all_sids)
            df_rand = fused_cache[rand_sid]
            max_start = len(df_rand) - dur - 10
            if max_start > 100:
                rand_start = np.random.randint(100, max_start)
                r_d_traj, r_p_traj = extract_aligned_observed_trajectory(df_rand, rand_start, rand_start + dur)
                rand_dorsal_trajs.append(interpolate_trajectory(r_d_traj, target_len=20))
                rand_paws_trajs.append(interpolate_trajectory(r_p_traj, target_len=20))

        rand_dorsal_arr = np.array(rand_dorsal_trajs)
        rand_paws_arr = np.array(rand_paws_trajs)

        r_d_mean = np.mean(rand_dorsal_arr, axis=0, keepdims=True)
        r_p_mean = np.mean(rand_paws_arr, axis=0, keepdims=True)

        var_dorsal_rand = float(np.mean(np.sum((rand_dorsal_arr - r_d_mean)**2, axis=(1, 2, 3))))
        var_paws_rand = float(np.mean(np.sum((rand_paws_arr - r_p_mean)**2, axis=(1, 2, 3))))

        ratio_dorsal = float(var_dorsal_syll / max(1e-6, var_dorsal_rand))
        ratio_paws = float(var_paws_syll / max(1e-6, var_paws_rand))

        compactness_results.append({
            "syllable": syll,
            "n_bouts": len(dorsal_trajs),
            "median_duration_frames": int(np.median(durations)),
            "dorsal_syllable_variance": var_dorsal_syll,
            "dorsal_random_control_variance": var_dorsal_rand,
            "dorsal_compactness_ratio": ratio_dorsal,
            "dorsal_is_more_compact_than_random": (ratio_dorsal < 1.0),
            "paws_syllable_variance": var_paws_syll,
            "paws_random_control_variance": var_paws_rand,
            "paws_compactness_ratio": ratio_paws,
            "paws_is_more_compact_than_random": (ratio_paws < 1.0),
        })

        print(f"  Dorsal: Syll Var = {var_dorsal_syll:.1f} vs Random = {var_dorsal_rand:.1f} -> Ratio = {ratio_dorsal:.3f} ({'COMPACT ✅' if ratio_dorsal < 1.0 else 'DISPERSED ❌'})")
        print(f"  Paws:   Syll Var = {var_paws_syll:.1f} vs Random = {var_paws_rand:.1f} -> Ratio = {ratio_paws:.3f} ({'COMPACT ✅' if ratio_paws < 1.0 else 'DISPERSED ❌'})")

    res_df = pd.DataFrame(compactness_results)
    res_df.to_csv(AUDIT_DIR / "trajectory_compactness_benchmark.csv", index=False)

    summary = {
        "num_syllables_evaluated": len(compactness_results),
        "mean_dorsal_compactness_ratio": float(res_df["dorsal_compactness_ratio"].mean()),
        "mean_paws_compactness_ratio": float(res_df["paws_compactness_ratio"].mean()),
        "dorsal_stereotypy_pass_rate_pct": float(np.mean(res_df["dorsal_is_more_compact_than_random"]) * 100.0),
        "paws_stereotypy_pass_rate_pct": float(np.mean(res_df["paws_is_more_compact_than_random"]) * 100.0),
        "detailed_syllables": compactness_results
    }

    with open(AUDIT_DIR / "trajectory_compactness_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary: Observed Trajectory Compactness Benchmark ---")
    print(f"Mean Dorsal Compactness Ratio: {summary['mean_dorsal_compactness_ratio']:.3f} ({summary['dorsal_stereotypy_pass_rate_pct']:.1f}% syllables more compact than random)")
    print(f"Mean Paws Compactness Ratio:   {summary['mean_paws_compactness_ratio']:.3f} ({summary['paws_stereotypy_pass_rate_pct']:.1f}% syllables more compact than random)")

if __name__ == "__main__":
    run_compactness_audit()
