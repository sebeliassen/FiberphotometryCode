#!/usr/bin/env python3
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.common.paths import get_repo_root

REPO = Path(str(get_repo_root()))
EXP = REPO / 'studies/multiview_experiments'
AUDIT_DIR = EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from fpcode.tracking.camera_correspondence import compute_session_camera_mapping
from fpcode.tracking.clock_sync import find_camera_timestamp_files
from fpcode.tracking.pair_matching import find_matched_pairs


def test_synthetic_delay():
    N = 1000
    dt = 1.0 / 60.0
    t_true = np.arange(N) * dt
    top_basler = t_true + 123.456
    bot_basler = t_true + 789.012
    top_bonsai = t_true + 1000.0
    bot_bonsai = t_true + 1000.0 + dt # 1-frame latency

    with tempfile.TemporaryDirectory() as td:
        top_csv = Path(td) / "top.csv"
        bot_csv = Path(td) / "bot.csv"
        pd.DataFrame({"Cam_Basler_Timestamp": top_basler, "Cam_Bonsai_Timestamp": top_bonsai}).to_csv(top_csv, index=False)
        pd.DataFrame({"Cam_Basler_Timestamp": bot_basler, "Cam_Bonsai_Timestamp": bot_bonsai}).to_csv(bot_csv, index=False)
        mapping, jitter, calib = compute_session_camera_mapping(top_csv, bot_csv, strict=True)
        return {
            "status": calib["status"],
            "is_valid_crystal": calib["is_valid_crystal"],
            "top_500_mapped_to_bot": int(mapping[500]),
            "frame_error": int(mapping[500] - 500)
        }

def test_real_camera_correspondence():
    pairs = find_matched_pairs()
    results = []
    for p in pairs[:15]:
        sid = p["sid"]
        top_csv, bot_csv = find_camera_timestamp_files(p["ts"], p["arena"])
        if top_csv is None or bot_csv is None: continue

        # Basler mapping
        map_bas, jit_bas, cal_bas = compute_session_camera_mapping(top_csv, bot_csv, strict=True)

        # FP3002 mapping
        df_t = pd.read_csv(top_csv, usecols=["FP3002_System_Timestamp"])
        df_b = pd.read_csv(bot_csv, usecols=["FP3002_System_Timestamp"])
        t_fp = df_t["FP3002_System_Timestamp"].to_numpy()
        b_fp = df_b["FP3002_System_Timestamp"].to_numpy()

        n_eval = min(len(t_fp), len(map_bas), 50000)
        fp_map = np.clip(np.searchsorted(b_fp, t_fp[:n_eval]), 0, len(b_fp) - 1)
        # Refine nearest
        prev_idx = np.clip(fp_map - 1, 0, len(b_fp) - 1)
        d_curr = np.abs(b_fp[fp_map] - t_fp[:n_eval])
        d_prev = np.abs(b_fp[prev_idx] - t_fp[:n_eval])
        fp_map = np.where(d_prev < d_curr, prev_idx, fp_map)

        diff = map_bas[:n_eval] - fp_map
        results.append({
            "sid": sid,
            "status": cal_bas["status"],
            "exact_match_pct": float(np.mean(diff == 0) * 100),
            "one_frame_diff_pct": float(np.mean(np.abs(diff) == 1) * 100),
            "max_abs_diff_frames": int(np.max(np.abs(diff)))
        })
    return results

def main():
    syn_res = test_synthetic_delay()
    real_res = test_real_camera_correspondence()
    out = {
        "synthetic_delay_test": syn_res,
        "real_session_correspondence": real_res
    }
    out_file = AUDIT_DIR / "clock_phase_and_real_correspondence_audit.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"Saved clock audit to {out_file}")
    print(f"Synthetic test: frame error = {syn_res['frame_error']}, status = {syn_res['status']}")

if __name__ == "__main__":
    main()
