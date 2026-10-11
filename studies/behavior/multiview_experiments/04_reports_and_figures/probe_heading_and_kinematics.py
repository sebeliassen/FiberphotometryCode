#!/usr/bin/env python3
"""
probe_heading_and_kinematics.py
=============================================================================
Stage 5 Probe: Heading Alignment and Kinematic Estimators.
1. Torso-based vs. Model Heading:
   - Raw dorsal anterior-posterior vector (head_midpoint/neck -> tail_base).
   - Circular residuals, constant 180-deg vs 0-deg offset, flip rate.
2. Centroid Speed & Heading Angular Velocity:
   - Full-segment median filtering (filter_size=9).
   - Explicitly timed central difference with 6-frame (100 ms @ 60 FPS) denominator.
   - Endpoint handling and angle wrapping.
3. 2D Body Extent (chord) and Paw Support across audited syllables.
=============================================================================
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import h5py
import numpy as np
from fpcode.viz import v2 as vc
import yaml

AUDITED_SYLLABLES = [1, 67, 30, 70, 92, 36, 42]


def run_probe(target_syllables: List[int], out_dir: Path, max_segments: int = 50):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "kinematics_and_heading_probe.json"

    print(f"[Heading & Kinematics Probe] Target syllables: {target_syllables}")
    print(f"[Heading & Kinematics Probe] Output file: {out_file}")

    with open(vc.CONFIG_PATH) as f:
        cohort_config = yaml.safe_load(f)
    fps = float(cohort_config.get("fps", 60.0))
    dt_6frames = 6.0 / fps  # 0.100s at 60 FPS

    vfi_data = np.load(vc.VFI_PATH)

    # Keypoint indices
    tail_idx = vc.BODYPARTS_21KP.index("tail_base")
    neck_idx = vc.BODYPARTS_21KP.index("neck")
    head_idx = vc.BODYPARTS_21KP.index("head_midpoint")

    # Load results.h5
    with h5py.File(vc.RESULTS_PATH, "r") as f:
        seg_keys = sorted([k for k in f.keys() if k not in ("metadata", "model_parameters")])
        if max_segments > 0:
            seg_keys = seg_keys[:max_segments]

        results_data = {}
        for k in seg_keys:
            results_data[k] = {
                "syllables": f[k]["syllable"][:],
                "centroid": f[k]["centroid"][:],
                "heading": f[k]["heading"][:],
            }

    fused_cache = {}

    def get_fused(base_sid: str):
        if base_sid not in fused_cache:
            xy, conf = vc.load_fused_tracks(base_sid, vc.FUSED_DIR)
            fused_cache[base_sid] = (xy, conf)
        return fused_cache[base_sid]

    # Metrics containers
    circular_residuals_all = []
    flips_count = 0
    total_valid_heading_frames = 0
    near_0_count = 0
    near_180_count = 0

    syllable_kinematics = {
        s: {
            "centroid_speeds_px_s": [],
            "angular_velocities_rad_s": [],
            "body_chord_px": [],
        }
        for s in target_syllables
    }

    t0 = time.time()
    for sid in seg_keys:
        syl = results_data[sid]["syllables"]
        c_raw = results_data[sid]["centroid"]
        h_raw = results_data[sid]["heading"]
        vfi = vfi_data[sid]
        n_frames = len(syl)
        base_sid = vc.base_session_id(sid)

        xy_fused, conf_fused = get_fused(base_sid)
        xy_seg = xy_fused[vfi]
        conf_seg = conf_fused[vfi]

        # 1. Full-segment pose filtering (KPMS filter_size=9)
        c_filt, h_filt = vc.filter_segment_pose(c_raw, h_raw, filter_size=9)

        # 2. Torso heading vs Model heading
        tail_xy = xy_seg[:, tail_idx]
        neck_xy = xy_seg[:, neck_idx]
        head_xy = xy_seg[:, head_idx]

        # Raw support check (confidence >= 0.5 and finite)
        supp_tail = (conf_seg[:, tail_idx] >= 0.5) & np.isfinite(tail_xy).all(axis=-1)
        supp_neck = (conf_seg[:, neck_idx] >= 0.5) & np.isfinite(neck_xy).all(axis=-1)
        supp_head = (conf_seg[:, head_idx] >= 0.5) & np.isfinite(head_xy).all(axis=-1)

        valid_torso = supp_tail & (supp_neck | supp_head)
        # Vector from tail to anterior (use head if available, else neck)
        anterior_xy = np.where(supp_head[:, None], head_xy, neck_xy)
        torso_vec = anterior_xy - tail_xy
        torso_heading = np.arctan2(torso_vec[:, 1], torso_vec[:, 0])

        valid_h = valid_torso & np.isfinite(h_filt)
        if valid_h.any():
            # Residual = angle between model heading and torso anterior vector
            diff = (h_filt[valid_h] - torso_heading[valid_h] + np.pi) % (2 * np.pi) - np.pi
            circular_residuals_all.extend(diff.tolist())
            total_valid_heading_frames += int(valid_h.sum())
            near_0_count += int((np.abs(diff) < np.deg2rad(30)).sum())
            near_180_count += int((np.abs(np.abs(diff) - np.pi) < np.deg2rad(30)).sum())

        # 3. Timed central difference derivatives
        # Frames t-3 to t+3 -> central diff over 6 frames = 100 ms
        # Centroid speed (px/s)
        centroid_speed = np.full(n_frames, np.nan, dtype=np.float64)
        angular_vel = np.full(n_frames, np.nan, dtype=np.float64)

        if n_frames >= 7:
            # valid interior indices: 3 .. n_frames - 4
            idx = np.arange(3, n_frames - 3)
            dc = np.linalg.norm(c_filt[idx + 3] - c_filt[idx - 3], axis=-1)
            centroid_speed[idx] = dc / dt_6frames

            # Circular difference for heading
            dh = (h_filt[idx + 3] - h_filt[idx - 3] + np.pi) % (2 * np.pi) - np.pi
            angular_vel[idx] = np.abs(dh) / dt_6frames

        # 2D body extent (chord between neck and tail)
        valid_chord = supp_tail & supp_neck
        body_chord = np.full(n_frames, np.nan, dtype=np.float64)
        body_chord[valid_chord] = np.linalg.norm(neck_xy[valid_chord] - tail_xy[valid_chord], axis=-1)

        # Collect syllable kinematics
        for s_id in target_syllables:
            mask_s = (syl == s_id)
            if not mask_s.any():
                continue
            v_sp = centroid_speed[mask_s & np.isfinite(centroid_speed)]
            v_av = angular_vel[mask_s & np.isfinite(angular_vel)]
            v_bc = body_chord[mask_s & np.isfinite(body_chord)]
            if len(v_sp):
                syllable_kinematics[s_id]["centroid_speeds_px_s"].extend(v_sp.tolist())
            if len(v_av):
                syllable_kinematics[s_id]["angular_velocities_rad_s"].extend(v_av.tolist())
            if len(v_bc):
                syllable_kinematics[s_id]["body_chord_px"].extend(v_bc.tolist())

    print(f"  Processed {len(seg_keys)} segments in {time.time() - t0:.1f}s.")

    # Heading alignment analysis
    residuals_arr = np.array(circular_residuals_all)
    mean_res = float(np.arctan2(np.sin(residuals_arr).mean(), np.cos(residuals_arr).mean()))
    circ_var = 1.0 - float(np.sqrt(np.sin(residuals_arr).mean() ** 2 + np.cos(residuals_arr).mean() ** 2))

    heading_summary = {
        "total_valid_heading_frames": total_valid_heading_frames,
        "mean_circular_residual_rad": round(mean_res, 4),
        "mean_circular_residual_deg": round(float(np.rad2deg(mean_res)), 2),
        "circular_variance": round(circ_var, 4),
        "pct_within_30deg_of_0": round(float(near_0_count / total_valid_heading_frames * 100), 2) if total_valid_heading_frames else 0.0,
        "pct_within_30deg_of_180": round(float(near_180_count / total_valid_heading_frames * 100), 2) if total_valid_heading_frames else 0.0,
        "convention": "0_deg_aligned" if near_0_count > near_180_count else "180_deg_inverted",
    }

    # Syllable kinematics summary
    kinematics_summary = {}
    for s_id in target_syllables:
        sp = np.array(syllable_kinematics[s_id]["centroid_speeds_px_s"])
        av = np.array(syllable_kinematics[s_id]["angular_velocities_rad_s"])
        bc = np.array(syllable_kinematics[s_id]["body_chord_px"])
        kinematics_summary[str(s_id)] = {
            "n_frames_evaluated": int(len(sp)),
            "centroid_speed_px_s": {
                "median": round(float(np.median(sp)), 2) if len(sp) else None,
                "p05": round(float(np.percentile(sp, 5)), 2) if len(sp) else None,
                "p95": round(float(np.percentile(sp, 95)), 2) if len(sp) else None,
            },
            "angular_velocity_rad_s": {
                "median": round(float(np.median(av)), 2) if len(av) else None,
                "p05": round(float(np.percentile(av, 5)), 2) if len(av) else None,
                "p95": round(float(np.percentile(av, 95)), 2) if len(av) else None,
            },
            "body_chord_neck_to_tail_px": {
                "median": round(float(np.median(bc)), 2) if len(bc) else None,
                "p05": round(float(np.percentile(bc, 5)), 2) if len(bc) else None,
                "p95": round(float(np.percentile(bc, 95)), 2) if len(bc) else None,
            },
        }

    report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cohort": "cohort_static_21kp_v2",
            "model": vc.MODEL_NAME,
            "fps": fps,
            "derivative_settings": {
                "method": "explicitly_timed_central_difference",
                "filter": "median_filter_size_9",
                "stencil": "[-3, +3]",
                "denominator_frames": 6,
                "denominator_seconds": dt_6frames,
            },
        },
        "heading_alignment": heading_summary,
        "syllable_kinematics": kinematics_summary,
    }

    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Heading & Kinematics Probe] Results written to {out_file}")
    print(f"  Heading Convention: {heading_summary['convention']} (mean {heading_summary['mean_circular_residual_deg']} deg, "
          f"circ_var {heading_summary['circular_variance']}, "
          f"{heading_summary['pct_within_30deg_of_0']}% near 0 deg vs {heading_summary['pct_within_30deg_of_180']}% near 180 deg)")
    for s_id in target_syllables:
        ks = kinematics_summary[str(s_id)]
        csp = ks["centroid_speed_px_s"]["median"]
        cav = ks["angular_velocity_rad_s"]["median"]
        cbc = ks["body_chord_neck_to_tail_px"]["median"]
        print(f"  Syllable {s_id:2d}: Speed={csp} px/s | AngVel={cav} rad/s | Chord={cbc} px (N={ks['n_frames_evaluated']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--syllables", type=int, nargs="+", default=AUDITED_SYLLABLES)
    parser.add_argument("--out_dir", type=Path, default=vc.DEFAULT_VIZ_DIR)
    parser.add_argument("--max_segments", type=int, default=50)
    args = parser.parse_args()
    run_probe(args.syllables, args.out_dir, args.max_segments)
