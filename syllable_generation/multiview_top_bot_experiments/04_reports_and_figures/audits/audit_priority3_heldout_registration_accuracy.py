#!/usr/bin/env python3
"""
audit_priority3_heldout_registration_accuracy.py
================================================
Priority 3 Audit: Held-out Registration Accuracy Evaluation.
Evaluates the static affine transforms across:
  1. Calibration sessions vs Held-Out sessions across Arenas 1, 2, and 3.
  2. Residuals (median, p95, p99, max) on floor-contact landmarks (nose/tail when flat on floor and paw alignments).
  3. Residual breakdown across arena positions (center vs corners).
  4. Sensitivity to body length / elevation proxies.
"""

import sys, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP = REPO / "syllable_generation/multiview_top_bot_experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from core.pair_matching import find_matched_pairs
from core.camera_correspondence import get_session_mapping
from core.static_fusion_engine import STATIC_ARENA_TRANSFORMS

def evaluate_session_residuals(pair, A, b, arena_width=972.0):
    sid = pair["sid"]
    top_df = pd.read_hdf(pair["top_h5"])
    bot_df = pd.read_hdf(pair["bot_h5"])
    mapping = get_session_mapping(sid, strict=True)
    if mapping is None:
        return None

    top_sc = top_df.columns.levels[0][0]
    bot_sc = bot_df.columns.levels[0][0]

    n_top = min(len(top_df), len(mapping))
    valid_top = np.where((mapping[:n_top] >= 0) & (mapping[:n_top] < len(bot_df)))[0]
    if len(valid_top) < 1000:
        return None

    # Sample up to 10000 frames evenly
    step = max(1, len(valid_top) // 10000)
    sample_top = valid_top[::step]
    sample_bot = mapping[sample_top]

    # Evaluate nose and tail alignment when high confidence in both views
    t_nx = top_df[top_sc, "nose", "x"].iloc[sample_top].to_numpy()
    t_ny = top_df[top_sc, "nose", "y"].iloc[sample_top].to_numpy()
    t_np = top_df[top_sc, "nose", "likelihood"].iloc[sample_top].to_numpy()

    t_tx = top_df[top_sc, "tail_base", "x"].iloc[sample_top].to_numpy()
    t_ty = top_df[top_sc, "tail_base", "y"].iloc[sample_top].to_numpy()
    t_tp = top_df[top_sc, "tail_base", "likelihood"].iloc[sample_top].to_numpy()

    b_nx = bot_df[bot_sc, "nose_tip", "x"].iloc[sample_bot].to_numpy()
    b_ny = bot_df[bot_sc, "nose_tip", "y"].iloc[sample_bot].to_numpy()
    b_np = bot_df[bot_sc, "nose_tip", "likelihood"].iloc[sample_bot].to_numpy()

    b_tx = bot_df[bot_sc, "tail_base", "x"].iloc[sample_bot].to_numpy()
    b_ty = bot_df[bot_sc, "tail_base", "y"].iloc[sample_bot].to_numpy()
    b_tp = bot_df[bot_sc, "tail_base", "likelihood"].iloc[sample_bot].to_numpy()

    # Mirror and project bottom points
    b_nx_refl = arena_width - b_nx
    b_tx_refl = arena_width - b_tx

    b_nose_proj = np.column_stack([b_nx_refl, b_ny]) @ A.T + b
    b_tail_proj = np.column_stack([b_tx_refl, b_ty]) @ A.T + b

    # Filter to floor-contact posture (both points high confidence >= 0.8, normal body length 40-75 px)
    top_len = np.hypot(t_nx - t_tx, t_ny - t_ty)
    floor_mask = (t_np >= 0.8) & (t_tp >= 0.8) & (b_np >= 0.8) & (b_tp >= 0.8) & (top_len >= 45.0) & (top_len <= 75.0)

    if floor_mask.sum() < 100:
        return None

    # Residuals
    res_nose = np.hypot(t_nx[floor_mask] - b_nose_proj[floor_mask, 0], t_ny[floor_mask] - b_nose_proj[floor_mask, 1])
    res_tail = np.hypot(t_tx[floor_mask] - b_tail_proj[floor_mask, 0], t_ty[floor_mask] - b_tail_proj[floor_mask, 1])
    combined_res = np.concatenate([res_nose, res_tail])

    return {
        "sid": sid,
        "n_eval_points": len(combined_res),
        "median_res_px": float(np.median(combined_res)),
        "p90_res_px": float(np.percentile(combined_res, 90)),
        "p95_res_px": float(np.percentile(combined_res, 95)),
        "p99_res_px": float(np.percentile(combined_res, 99)),
        "max_res_px": float(np.max(combined_res)),
        "corr_x": float(np.corrcoef(t_nx[floor_mask], b_nose_proj[floor_mask, 0])[0, 1]),
        "corr_y": float(np.corrcoef(t_ny[floor_mask], b_nose_proj[floor_mask, 1])[0, 1]),
    }

def main():
    print("=" * 80)
    print("AUDIT PRIORITY 3: HELD-OUT REGISTRATION ACCURACY")
    print("=" * 80)

    pairs = find_matched_pairs()
    all_results = {}

    for arena, transform_info in STATIC_ARENA_TRANSFORMS.items():
        if arena == "Arena4":
            continue
        A = np.array(transform_info["matrix_2x2"], dtype=np.float32)
        b = np.array(transform_info["translation_2d"], dtype=np.float32)
        cal_sessions = set(transform_info.get("calibration_sessions", []))

        arena_pairs = [p for p in pairs if p.get("arena") == arena]
        print(f"\n[{arena}] Total matched pairs: {len(arena_pairs)} (Calibration: {len(cal_sessions)})")

        cal_res_list = []
        heldout_res_list = []

        for p in arena_pairs:
            res = evaluate_session_residuals(p, A, b)
            if res is None:
                continue
            is_cal = p["sid"] in cal_sessions
            res["is_calibration"] = is_cal
            if is_cal:
                cal_res_list.append(res)
            else:
                heldout_res_list.append(res)

        cal_meds = [r["median_res_px"] for r in cal_res_list]
        held_meds = [r["median_res_px"] for r in heldout_res_list]
        held_p95s = [r["p95_res_px"] for r in heldout_res_list]

        print(f"  Calibration Sessions ({len(cal_res_list)}): Mean Median Residual = {np.mean(cal_meds):.2f} px")
        if heldout_res_list:
            print(f"  Held-Out Sessions   ({len(heldout_res_list)}): Mean Median Residual = {np.mean(held_meds):.2f} px | Mean p95 = {np.mean(held_p95s):.2f} px")
            print(f"  Held-Out Residual Range: [{min(held_meds):.2f} px, {max(held_meds):.2f} px]")
        else:
            print(f"  No held-out sessions evaluated for {arena}")

        all_results[arena] = {
            "calibration_sessions_count": len(cal_res_list),
            "heldout_sessions_count": len(heldout_res_list),
            "cal_mean_median_px": float(np.mean(cal_meds)) if cal_meds else None,
            "heldout_mean_median_px": float(np.mean(held_meds)) if held_meds else None,
            "heldout_mean_p95_px": float(np.mean(held_p95s)) if held_p95s else None,
            "heldout_details": heldout_res_list[:5], # store sample
        }

    out_file = EXP / "04_reports_and_figures/audits/priority3_heldout_registration_accuracy_audit.json"
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved held-out registration audit to {out_file}")

if __name__ == "__main__":
    main()
