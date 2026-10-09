#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP = REPO / "syllable_generation/multiview_top_bot_experiments"
AUDIT_DIR = EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from core.camera_correspondence import get_session_mapping
from core.pair_matching import find_matched_pairs
from core.static_fusion_engine import STATIC_ARENA_TRANSFORMS

CHAMBER_TO_ARENA = {
    "top_left": "Arena1",
    "bottom_left": "Arena2",
    "bottom_right": "Arena3",
    "top_right": "Arena4"
}

def analyze_all_sessions():
    pairs = find_matched_pairs()
    print(f"Total matched pairs in cohort: {len(pairs)}")

    all_results = []

    for idx, pair in enumerate(pairs):
        sid = pair["sid"]
        quadrant = pair["quadrant"]
        arena = pair["arena"]
        if arena not in STATIC_ARENA_TRANSFORMS:
            continue

        t_info = STATIC_ARENA_TRANSFORMS[arena]
        A = np.array(t_info["matrix_2x2"], dtype=np.float32)
        b = np.array(t_info["translation_2d"], dtype=np.float32)
        cal_sessions = t_info.get("calibration_sessions", [])
        is_cal = sid in cal_sessions

        mapping = get_session_mapping(sid, strict=True)
        if mapping is None:
            continue

        try:
            top_df = pd.read_hdf(pair["top_h5"])
            bot_df = pd.read_hdf(pair["bot_h5"])
        except Exception as e:
            print(f"Error reading {sid}: {e}")
            continue

        top_sc = top_df.columns.levels[0][0]
        bot_sc = bot_df.columns.levels[0][0]

        n_top = min(len(top_df), len(mapping))
        valid_top = np.where((mapping[:n_top] >= 0) & (mapping[:n_top] < len(bot_df)))[0]
        if len(valid_top) < 500:
            continue

        step = max(1, len(valid_top) // 10000)
        sample_top = valid_top[::step]
        sample_bot = mapping[sample_top]

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

        arena_width = 972.0
        b_nx_refl = arena_width - b_nx
        b_tx_refl = arena_width - b_tx

        b_nose_proj = np.column_stack([b_nx_refl, b_ny]) @ A.T + b
        b_tail_proj = np.column_stack([b_tx_refl, b_ty]) @ A.T + b

        top_len = np.hypot(t_nx - t_tx, t_ny - t_ty)
        floor_mask = (t_np >= 0.8) & (t_tp >= 0.8) & (b_np >= 0.8) & (b_tp >= 0.8) & (top_len >= 45.0) & (top_len <= 75.0)

        if floor_mask.sum() < 50:
            continue

        # Signed errors: Top - Proj_Bot
        dx_nose = t_nx[floor_mask] - b_nose_proj[floor_mask, 0]
        dy_nose = t_ny[floor_mask] - b_nose_proj[floor_mask, 1]
        dist_nose = np.hypot(dx_nose, dy_nose)

        dx_tail = t_tx[floor_mask] - b_tail_proj[floor_mask, 0]
        dy_tail = t_ty[floor_mask] - b_tail_proj[floor_mask, 1]
        dist_tail = np.hypot(dx_tail, dy_tail)

        comb_dist = np.concatenate([dist_nose, dist_tail])
        comb_dx = np.concatenate([dx_nose, dx_tail])
        comb_dy = np.concatenate([dy_nose, dy_tail])

        # Date parsing
        date_str = sid.split("T")[0]
        month_str = date_str[:7] # YYYY-MM

        res_row = {
            "sid": sid,
            "date": date_str,
            "month": month_str,
            "arena": arena,
            "quadrant": quadrant,
            "is_calibration": is_cal,
            "n_points": int(len(comb_dist)),
            "comb_median_px": float(np.median(comb_dist)),
            "comb_mean_px": float(np.mean(comb_dist)),
            "comb_p95_px": float(np.percentile(comb_dist, 95)),
            "mean_dx": float(np.mean(comb_dx)),
            "mean_dy": float(np.mean(comb_dy)),
            "median_dx": float(np.median(comb_dx)),
            "median_dy": float(np.median(comb_dy)),
            "nose_median_px": float(np.median(dist_nose)),
            "nose_mean_dx": float(np.mean(dx_nose)),
            "nose_mean_dy": float(np.mean(dy_nose)),
            "tail_median_px": float(np.median(dist_tail)),
            "tail_mean_dx": float(np.mean(dx_tail)),
            "tail_mean_dy": float(np.mean(dy_tail)),
        }
        all_results.append(res_row)
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx+1}/{len(pairs)} sessions...")

    df_res = pd.DataFrame(all_results)
    csv_out = AUDIT_DIR / "all_sessions_registration_residuals.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"Saved {len(df_res)} session rows to {csv_out}")

    # Summary by month and arena
    summary = {}
    for (arena, month), grp in df_res.groupby(["arena", "month"]):
        summary[f"{arena}_{month}"] = {
            "n_sessions": int(len(grp)),
            "is_cal_count": int(grp["is_calibration"].sum()),
            "mean_session_median_px": float(grp["comb_median_px"].mean()),
            "mean_session_p95_px": float(grp["comb_p95_px"].mean()),
            "mean_signed_dx": float(grp["mean_dx"].mean()),
            "mean_signed_dy": float(grp["mean_dy"].mean()),
        }

    summary_file = AUDIT_DIR / "registration_by_arena_month_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_file}")

    print("\n--- Summary by Arena and Month ---")
    for k, v in summary.items():
        print(f"{k}: n={v['n_sessions']}, cal={v['is_cal_count']}, median={v['mean_session_median_px']:.2f}px, dx={v['mean_signed_dx']:.2f}px, dy={v['mean_signed_dy']:.2f}px")

if __name__ == "__main__":
    analyze_all_sessions()
