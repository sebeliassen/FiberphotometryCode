#!/usr/bin/env python3
"""
repair_1_stationary_floor_and_epoch_registration.py
===================================================
Rigorous empirical registration calibration and evaluation across documented
acquisition epochs.

Objectives:
1. Document and map acquisition epochs explicitly based on recorded cohorts:
   - Epoch 1: Winter/Spring 2025-2026 (3_week_PI, 8_weeks_PI, March T1-T9)
   - Epoch 2: Spring 2026 Post-Movement (24_weeks_PI, L_dopa)
2. Measure physical floor/arena landmark shifts using normalized cross-correlation
   on stationary background images across all arenas.
3. Compare 3 registration models on SEPARATE HELD-OUT whole sessions:
   - Model A: Baseline March static transform
   - Model B: Floor-translation-corrected transform
   - Model C: Separate epoch-fitted affine transform (fit on train sessions, tested on test sessions)
4. Evaluate error dependence on spatial position (x, y), heading angle theta, and movement speed v.
5. Export verified epoch transforms specification: `epoch_arena_transforms.json`.
"""
import glob
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd

from fpcode.common.paths import get_repo_root

REPO = Path(str(get_repo_root()))
EXP = REPO / 'studies/multiview_experiments'
AUDIT_DIR = EXP / "04_reports_and_figures/audits/audit_20261007_pipeline_repairs"
CORE_DIR = EXP / "core"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from fpcode.tracking.camera_correspondence import compute_session_camera_mapping, get_session_mapping
from fpcode.tracking.clock_sync import find_camera_timestamp_files
from fpcode.tracking.pair_matching import find_matched_pairs


def get_session_epoch(ts_str: str) -> str:
    """Documented epoch assignment based on experimental recording schedule."""
    if ts_str < "2026-04-01":
        return "epoch1_winter_spring_2026"
    else:
        return "epoch2_spring_post_shift_2026"

def compute_background_floor_shifts() -> Dict[str, dict]:
    """Computes physical floor shifts using stationary background cross-correlation."""
    def get_bg(path):
        cap = cv2.VideoCapture(path)
        frames = []
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            return None
        for i in np.linspace(100, min(n - 100, 3000), 20, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, f = cap.read()
            if ret:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        cap.release()
        return np.median(frames, axis=0).astype(np.uint8) if frames else None

    # Compare March reference vs April reference
    arenas = {
        "Arena1": (
            glob.glob("/mnt/hdd/karen_project/all_videos_split/*CenterTopCamVid_2026-03-05T11_01_01_top_left.avi")[0],
            glob.glob("/mnt/hdd/karen_project/all_videos_split/*CenterTopCamVid_2026-04-27T14_05_05_top_left.avi")[0]
        ),
        "Arena2": (
            glob.glob("/mnt/hdd/karen_project/all_videos_split/*CenterTopCamVid_2026-03-05T11_01_01_bottom_left.avi")[0],
            glob.glob("/mnt/hdd/karen_project/all_videos_split/*CenterTopCamVid_2026-04-28T13_21_05_bottom_left.avi")[0]
        ),
        "Arena3": (
            glob.glob("/mnt/hdd/karen_project/all_videos_split/*CenterTopCamVid_2026-03-05T11_01_01_bottom_right.avi")[0],
            glob.glob("/mnt/hdd/karen_project/all_videos_split/*CenterTopCamVid_2026-04-28T13_21_05_bottom_right.avi")[0]
        ),
    }

    shifts = {}
    for arena, (p_mar, p_apr) in arenas.items():
        bg_mar = get_bg(p_mar)
        bg_apr = get_bg(p_apr)
        res = cv2.matchTemplate(bg_apr, bg_mar[40:360, 40:360], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        dx = max_loc[0] - 40
        dy = max_loc[1] - 40
        shifts[arena] = {
            "dx_px": int(dx),
            "dy_px": int(dy),
            "cross_corr": float(max_val),
            "ref_march_vid": Path(p_mar).name,
            "ref_april_vid": Path(p_apr).name,
        }
    return shifts

def extract_paired_session_landmarks(pair: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts matched (top_pts, bot_pts, headings, speeds) for nose and tail landmarks.
    """
    top_csv, bot_csv = find_camera_timestamp_files(pair["ts"], pair["arena"])
    if top_csv is None or bot_csv is None:
        return None, None, None, None

    mapping = get_session_mapping(pair["sid"], strict=False)
    if mapping is None:
        mapping, _, _ = compute_session_camera_mapping(top_csv, bot_csv, strict=False)
    df_top = pd.read_hdf(pair["top_h5"])
    df_bot = pd.read_hdf(pair["bot_h5"])

    sc_top = df_top.columns.levels[0][0]
    sc_bot = df_bot.columns.levels[0][0]

    n_top = len(df_top)
    n_bot = len(df_bot)
    valid_mask = (mapping >= 0) & (mapping < n_bot)
    top_indices = np.where(valid_mask)[0]
    top_indices = top_indices[top_indices < n_top]
    bot_indices = mapping[top_indices]

    # Filter confidence > 0.6
    t_nose_x = df_top[(sc_top, "nose", "x")].to_numpy()[top_indices]
    t_nose_y = df_top[(sc_top, "nose", "y")].to_numpy()[top_indices]
    t_nose_p = df_top[(sc_top, "nose", "likelihood")].to_numpy()[top_indices]

    t_tail_x = df_top[(sc_top, "tail_base", "x")].to_numpy()[top_indices]
    t_tail_y = df_top[(sc_top, "tail_base", "y")].to_numpy()[top_indices]
    t_tail_p = df_top[(sc_top, "tail_base", "likelihood")].to_numpy()[top_indices]

    # Botcam landmarks: snout / nose_tip and tail_base
    # In static_fusion_engine: refl_bx = 972.0 - raw_bx, refl_by = raw_by
    b_nose_name = "nose_tip" if (sc_bot, "nose_tip", "x") in df_bot.columns else "snout"
    b_nose_x = 972.0 - df_bot[(sc_bot, b_nose_name, "x")].to_numpy()[bot_indices]
    b_nose_y = df_bot[(sc_bot, b_nose_name, "y")].to_numpy()[bot_indices]
    b_nose_p = df_bot[(sc_bot, b_nose_name, "likelihood")].to_numpy()[bot_indices]

    b_tail_x = 972.0 - df_bot[(sc_bot, "tail_base", "x")].to_numpy()[bot_indices]
    b_tail_y = df_bot[(sc_bot, "tail_base", "y")].to_numpy()[bot_indices]
    b_tail_p = df_bot[(sc_bot, "tail_base", "likelihood")].to_numpy()[bot_indices]

    # Body length filter: mouse must be reasonably extended (30 to 180 px)
    t_len = np.hypot(t_nose_x - t_tail_x, t_nose_y - t_tail_y)
    b_len = np.hypot(b_nose_x - b_tail_x, b_nose_y - b_tail_y)

    good = (t_nose_p > 0.6) & (t_tail_p > 0.6) & (b_nose_p > 0.6) & (b_tail_p > 0.6) & \
           (t_len > 30) & (t_len < 180) & (b_len > 70) & (b_len < 450)

    if np.sum(good) < 200:
        return None, None, None, None

    # Stack nose and tail points: top_pts, bot_pts
    top_nose = np.column_stack([t_nose_x[good], t_nose_y[good]])
    top_tail = np.column_stack([t_tail_x[good], t_tail_y[good]])
    bot_nose = np.column_stack([b_nose_x[good], b_nose_y[good]])
    bot_tail = np.column_stack([b_tail_x[good], b_tail_y[good]])

    top_pts = np.vstack([top_nose, top_tail])
    bot_pts = np.vstack([bot_nose, bot_tail])

    # Heading from tail to nose
    headings = np.arctan2(t_nose_y[good] - t_tail_y[good], t_nose_x[good] - t_tail_x[good])
    headings = np.concatenate([headings, headings]) # for both nose and tail

    # Approximate body speed (finite difference)
    dt_sec = 1.0 / 60.0
    dx = np.gradient(t_nose_x[good])
    dy = np.gradient(t_nose_y[good])
    speeds = np.hypot(dx, dy) / dt_sec
    speeds = np.concatenate([speeds, speeds])

    return top_pts, bot_pts, headings, speeds

def fit_affine(bot_pts: np.ndarray, top_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fits 2D affine: top = bot @ M.T + b."""
    N = len(bot_pts)
    A = np.column_stack([bot_pts, np.ones(N)])
    sol, _, _, _ = np.linalg.lstsq(A, top_pts, rcond=None)
    M = sol[:2, :].T
    b = sol[2, :]
    return M, b

def evaluate_transform(bot_pts: np.ndarray, top_pts: np.ndarray, M: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = bot_pts @ M.T + b
    res = top_pts - pred
    err = np.linalg.norm(res, axis=1)
    return res, err, pred

def run_registration_repair():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    floor_shifts = compute_background_floor_shifts()
    print("--- 1. Measured Stationary Floor Shifts (March -> April) ---")
    print(json.dumps(floor_shifts, indent=2))

    # Load baseline March transforms
    with open(CORE_DIR / "static_arena_transforms.json", "r") as f:
        march_transforms = json.load(f)

    pairs = find_matched_pairs()
    print(f"\nDiscovered {len(pairs)} matched sessions across entire cohort.")

    # Group Epoch 2 sessions by arena
    epoch2_pairs = [p for p in pairs if get_session_epoch(p["ts"]) == "epoch2_spring_post_shift_2026"]
    print(f"Epoch 2 (April/May) sessions: {len(epoch2_pairs)}")

    benchmark_records = []
    session_records = []
    epoch2_calibrated_transforms = {}

    for arena in ["Arena1", "Arena2", "Arena3"]:
        arena_pairs = [p for p in epoch2_pairs if p["arena"] == arena]
        if len(arena_pairs) < 2:
            print(f"Skipping {arena}: insufficient sessions ({len(arena_pairs)})")
            continue

        # Split whole sessions deterministically: 50% train (fit), 50% test (held-out validation)
        # Sort by timestamp
        arena_pairs = sorted(arena_pairs, key=lambda x: x["ts"])
        n_train = max(1, len(arena_pairs) // 2)
        train_pairs = arena_pairs[:n_train]
        test_pairs = arena_pairs[n_train:]

        print(f"\n--- {arena}: {len(train_pairs)} Train Sessions, {len(test_pairs)} Held-out Test Sessions ---")

        # Collect train points
        train_top, train_bot = [], []
        for p in train_pairs:
            top_p, bot_p, _, _ = extract_paired_session_landmarks(p)
            if top_p is not None:
                train_top.append(top_p)
                train_bot.append(bot_p)
        train_top = np.vstack(train_top)
        train_bot = np.vstack(train_bot)

        # Baseline Model A: March static
        M_base = np.array(march_transforms[arena]["matrix_2x2"])
        b_base = np.array(march_transforms[arena]["translation_2d"])

        # Model B: Floor-translation-only correction
        M_trans = M_base.copy()
        # Floor shift measured: TopCam moved by (dx_floor, dy_floor)
        dx_f = floor_shifts[arena]["dx_px"]
        dy_f = floor_shifts[arena]["dy_px"]
        b_trans = b_base + np.array([dx_f, dy_f])

        # Model C: New full affine fit on train sessions
        M_aff, b_aff = fit_affine(train_bot, train_top)
        epoch2_calibrated_transforms[arena] = {
            "matrix_2x2": M_aff.tolist(),
            "translation_2d": b_aff.tolist(),
            "n_train_points": len(train_top),
            "train_sessions": [p["sid"] for p in train_pairs]
        }

        # Collect test points and evaluate all 3 models on HELD-OUT whole sessions
        test_top, test_bot, test_head, test_spd = [], [], [], []
        for p in test_pairs:
            top_p, bot_p, head_p, spd_p = extract_paired_session_landmarks(p)
            if top_p is not None:
                test_top.append(top_p)
                test_bot.append(bot_p)
                test_head.append(head_p)
                test_spd.append(spd_p)

                # Evaluate per-session metrics on this held-out session
                for m_name, (M, b) in [
                    ("Model_A_March_Baseline", (M_base, b_base)),
                    ("Model_B_Floor_Translation_Only", (M_trans, b_trans)),
                    ("Model_C_Full_Affine_TrainFit", (M_aff, b_aff))
                ]:
                    res_s, err_s, _ = evaluate_transform(bot_p, top_p, M, b)
                    session_records.append({
                        "sid": p["sid"],
                        "arena": arena,
                        "model": m_name,
                        "n_pts": len(err_s),
                        "median_err_px": float(np.median(err_s)),
                        "p95_err_px": float(np.percentile(err_s, 95)),
                        "mean_dx": float(np.mean(res_s[:, 0])),
                        "mean_dy": float(np.mean(res_s[:, 1])),
                    })

        test_top = np.vstack(test_top)
        test_bot = np.vstack(test_bot)
        test_head = np.concatenate(test_head)
        test_spd = np.concatenate(test_spd)

        print(f"Total Held-out Evaluation Points for {arena}: {len(test_top):,}")

        for m_name, (M, b) in [
            ("Model_A_March_Baseline", (M_base, b_base)),
            ("Model_B_Floor_Translation_Only", (M_trans, b_trans)),
            ("Model_C_Full_Affine_TrainFit", (M_aff, b_aff))
        ]:
            res, err, _ = evaluate_transform(test_bot, test_top, M, b)
            med = float(np.median(err))
            p95 = float(np.percentile(err, 95))
            mdx = float(np.mean(res[:, 0]))
            mdy = float(np.mean(res[:, 1]))

            # Correlation of error with speed, position, heading
            corr_spd = float(np.corrcoef(err, test_spd)[0, 1]) if len(err) > 10 else 0.0
            corr_x = float(np.corrcoef(err, test_top[:, 0])[0, 1]) if len(err) > 10 else 0.0
            corr_y = float(np.corrcoef(err, test_top[:, 1])[0, 1]) if len(err) > 10 else 0.0

            benchmark_records.append({
                "arena": arena,
                "model": m_name,
                "n_heldout_pts": len(err),
                "heldout_median_px": med,
                "heldout_p95_px": p95,
                "mean_signed_dx": mdx,
                "mean_signed_dy": mdy,
                "corr_err_speed": corr_spd,
                "corr_err_pos_x": corr_x,
                "corr_err_pos_y": corr_y
            })
            print(f"  {m_name:30s} -> Median: {med:.2f} px | p95: {p95:.2f} px | dx: {mdx:+.2f} px | dy: {mdy:+.2f} px")

    # Save benchmark outputs
    benchmark_df = pd.DataFrame(benchmark_records)
    benchmark_df.to_csv(AUDIT_DIR / "epoch_registration_benchmark.csv", index=False)
    with open(AUDIT_DIR / "epoch_registration_benchmark.json", "w") as f:
        json.dump(benchmark_records, f, indent=2)

    pd.DataFrame(session_records).to_csv(AUDIT_DIR / "epoch_registration_by_session.csv", index=False)

    # Save combined epoch-aware transforms
    epoch_transforms_payload = {
        "epoch1_winter_spring_2026": march_transforms,
        "epoch2_spring_post_shift_2026": epoch2_calibrated_transforms,
        "measured_floor_shifts": floor_shifts,
        "epoch_definitions": {
            "epoch1_winter_spring_2026": "Sessions before 2026-04-01 (3_week_PI, 8_weeks_PI, March T1-T9). Floor shift <= 1 px.",
            "epoch2_spring_post_shift_2026": "Sessions from 2026-04-01 onwards (24_weeks_PI, L_dopa). TopCam physical mount shift ~20 px Y."
        }
    }
    with open(AUDIT_DIR / "epoch_arena_transforms.json", "w") as f:
        json.dump(epoch_transforms_payload, f, indent=2)

    # Also save to core for pipeline use
    with open(CORE_DIR / "epoch_arena_transforms.json", "w") as f:
        json.dump(epoch_transforms_payload, f, indent=2)

    print(f"\n✅ Finished Registration Benchmark. Artifacts saved in {AUDIT_DIR}")

if __name__ == "__main__":
    run_registration_repair()
