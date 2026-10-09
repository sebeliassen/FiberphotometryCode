#!/usr/bin/env python3
"""
export_tandem_cohort_production_results.py

Standalone post-processing script for the 50 AR + 200 SLDS Tandem Cohort Production run.
Reads the trained `results.h5`, merges TopCam and BotCam kinematics, weaves syllable classifications,
attaches canonical photometry (with NaN fallback for behavioral-only sessions), and serializes
`sessions_karen_project_tandem_cohort_production.pkl`.
"""

import sys
import time
import pickle
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.multiview_top_bot_experiments.audit_cohort_correlations import find_matched_pairs
from syllable_generation.analysis.replications.datta_utils import combined_photometry_and_syllables

DIRTY_PKL = REPO_ROOT / "syllable_generation/data/sessions_karen_project_dirty.pkl"
K240K_PKL_REF = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_k240k.pkl"
RESULTS_H5 = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/kpms_tandem_cohort_production/tandem_cohort_118sess_50ar_200slds/results.h5"
OUTPUT_PROD_PKL = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_production.pkl"

TEMPLATE_PHOT_COLS = [
    "SystemTimestamp",
    "signal_0", "signal_1", "signal_2", "signal_3",
    "signal_0_zscore_ransac", "signal_0_zscore_wallace", "signal_0_dff", "signal_0_zscore", "signal_0_prominence",
    "signal_1_zscore_ransac", "signal_1_zscore_wallace", "signal_1_dff", "signal_1_zscore", "signal_1_prominence",
    "signal_2_zscore_ransac", "signal_2_zscore_wallace", "signal_2_dff", "signal_2_zscore", "signal_2_prominence",
    "signal_3_zscore_ransac", "signal_3_zscore_wallace", "signal_3_dff", "signal_3_zscore", "signal_3_prominence"
]


def send_ntfy(title: str, message: str, priority: str = "default"):
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://ntfy.sh/kpms_tandem_run_alerts_384729104",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[warning] Failed to send ntfy alert: {e}", flush=True)


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("EXPORT TANDEM COHORT PRODUCTION RESULTS", flush=True)
    print("=" * 80, flush=True)

    print("\n1. Discovering matched pairs...", flush=True)
    pairs = find_matched_pairs()
    pair_map = {p["sid"]: p for p in pairs}
    print(f"  Found {len(pairs)} matched session pairs in cohort.", flush=True)

    import re
    bot_avis = list(Path("/mnt/hdd/karen_project").rglob("*BotCamVid*.avi"))
    ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})")
    avi_map = {}
    for avi in bot_avis:
        m = ts_pattern.search(avi.name)
        if not m:
            continue
        ts = m.group(1)
        for a in ["Arena1", "Arena2", "Arena3"]:
            if a in avi.name:
                avi_map[(ts, a)] = avi
                break
    print(f"  Indexed {len(avi_map)} BotCam video files from /mnt/hdd/karen_project.", flush=True)

    print("\n2. Loading dirty sessions...", flush=True)
    with open(DIRTY_PKL, "rb") as f:
        dirty_sessions = pickle.load(f)
    selected_sessions = [s for s in dirty_sessions if s.session_id in pair_map]
    print(f"  Selected {len(selected_sessions)} sessions from dirty pkl.", flush=True)

    print("\n3. Loading K240K reference dataset for top_syllable...", flush=True)
    k240k_map = {}
    if K240K_PKL_REF.exists():
        try:
            with open(K240K_PKL_REF, "rb") as f:
                k240k_all = pickle.load(f)
            for s_ref in k240k_all:
                if s_ref.session_id in pair_map:
                    k240k_map[s_ref.session_id] = s_ref
            print(f"  Mapped {len(k240k_map)} sessions from K240K reference.", flush=True)
        except Exception as e:
            print(f"  [warning] Could not load K240K reference: {e}", flush=True)

    print("\n4. Opening results.h5...", flush=True)
    with h5py.File(RESULTS_H5, "r") as f_res:
        results_keys = set(f_res.keys())
        print(f"  results.h5 contains {len(results_keys)} modeled sessions.", flush=True)

        modeled_sessions = []
        skipped_sessions = []
        phot_present_count = 0
        phot_nan_count = 0

        print("\n5. Weaving syllables, kinematics, and photometry...", flush=True)
        for idx, s in enumerate(selected_sessions):
            sid = s.session_id
            cfg = pair_map[sid]
            rec_key = f"{sid}_tandem_23kp"

            if rec_key not in results_keys:
                skipped_sessions.append(sid)
                print(f"  [{idx+1}/{len(selected_sessions)}] [SKIP] {rec_key} not in results.h5", flush=True)
                continue

            grp = f_res[rec_key]
            tandem_sylls = np.array(grp["syllable"])
            latent_states = np.array(grp["latent_state"])
            moseq_centroid = np.array(grp["centroid"])
            moseq_heading = np.array(grp["heading"])

            n_frames = len(tandem_sylls)
            timeline_index = pd.RangeIndex(0, n_frames)

            # Set s.results_dict_key
            s.results_dict_key = rec_key

            # 5a. Populate s.syllables
            syll_dict = {
                "syllable": pd.Series(tandem_sylls, dtype="Int64"),
            }
            for d in range(latent_states.shape[1]):
                syll_dict[f"latent_state_{d}"] = latent_states[:, d]
            syll_dict["centroid_0"] = moseq_centroid[:, 0]
            syll_dict["centroid_1"] = moseq_centroid[:, 1]
            syll_dict["heading"] = moseq_heading
            s.syllables = pd.DataFrame(syll_dict, index=timeline_index)

            # 5b. Extract TopCam Raw Kinematics
            df_top = pd.read_hdf(cfg["top_h5"])
            top_scorer = df_top.columns.levels[0][0]
            top_nx = df_top[top_scorer, "nose", "x"].iloc[:n_frames].to_numpy(dtype=float)
            top_ny = df_top[top_scorer, "nose", "y"].iloc[:n_frames].to_numpy(dtype=float)
            top_tx = df_top[top_scorer, "tail_base", "x"].iloc[:n_frames].to_numpy(dtype=float)
            top_ty = df_top[top_scorer, "tail_base", "y"].iloc[:n_frames].to_numpy(dtype=float)

            cx_top = 0.5 * (top_nx + top_tx)
            cy_top = 0.5 * (top_ny + top_ty)
            heading_top = np.arctan2(top_ny - top_ty, top_nx - top_tx)

            dx_top = np.diff(cx_top)
            dy_top = np.diff(cy_top)
            vel_top_raw = np.pad(np.sqrt(dx_top**2 + dy_top**2) * 60.0, (0, 1), mode="edge")
            ang_vel_top = np.pad(np.diff(np.unwrap(heading_top)) * 60.0, (0, 1), mode="edge")
            vel_top_smooth = median_filter(np.clip(vel_top_raw, 0, 300.0), size=7, mode="nearest")

            # 5c. Extract BotCam Raw Kinematics (native 972x972 space)
            df_bot = pd.read_hdf(cfg["bot_h5"])
            bot_scorer = df_bot.columns.levels[0][0]
            bot_nx = df_bot[bot_scorer, "nose_tip", "x"].iloc[:n_frames].to_numpy(dtype=float)
            bot_ny = df_bot[bot_scorer, "nose_tip", "y"].iloc[:n_frames].to_numpy(dtype=float)
            bot_tx = df_bot[bot_scorer, "tail_base", "x"].iloc[:n_frames].to_numpy(dtype=float)
            bot_ty = df_bot[bot_scorer, "tail_base", "y"].iloc[:n_frames].to_numpy(dtype=float)

            cx_bot = 0.5 * (bot_nx + bot_tx)
            cy_bot = 0.5 * (bot_ny + bot_ty)
            heading_bot = np.arctan2(bot_ny - bot_ty, bot_nx - bot_tx)

            dx_bot = np.diff(cx_bot)
            dy_bot = np.diff(cy_bot)
            vel_bot_raw = np.pad(np.sqrt(dx_bot**2 + dy_bot**2) * 60.0, (0, 1), mode="edge")
            ang_vel_bot = np.pad(np.diff(np.unwrap(heading_bot)) * 60.0, (0, 1), mode="edge")
            vel_bot_smooth = median_filter(np.clip(vel_bot_raw, 0, 300.0), size=7, mode="nearest")

            onset = np.concatenate([[True], tandem_sylls[1:] != tandem_sylls[:-1]])

            kinematics_dict = {
                "frame_index": timeline_index.to_numpy(),
                "name": [rec_key] * n_frames,
                "centroid_x": cx_top,
                "centroid_y": cy_top,
                "heading": moseq_heading,
                "heading_top": heading_top,
                "heading_moseq": moseq_heading,
                "angular_velocity": ang_vel_top,
                "velocity_px_s": vel_top_smooth,
                "velocity_px_s_raw": vel_top_raw,
                "syllable": pd.Series(tandem_sylls, dtype="Int64"),
                "tandem_syllable": pd.Series(tandem_sylls, dtype="Int64"),
                "group": ["default"] * n_frames,
                "onset": onset,
                "centroid_x_top": cx_top,
                "centroid_y_top": cy_top,
                "angular_velocity_top": ang_vel_top,
                "velocity_px_s_top": vel_top_smooth,
                "velocity_px_s_top_raw": vel_top_raw,
                "centroid_x_bot": cx_bot,
                "centroid_y_bot": cy_bot,
                "heading_bot": heading_bot,
                "angular_velocity_bot": ang_vel_bot,
                "velocity_px_s_bot": vel_bot_smooth,
                "velocity_px_s_bot_raw": vel_bot_raw,
            }

            if sid in k240k_map and "syllable" in k240k_map[sid].all_data_df.columns:
                k_syll = k240k_map[sid].all_data_df["syllable"].iloc[:n_frames]
                kinematics_dict["top_syllable"] = k_syll.to_numpy()

            df_behavior = pd.DataFrame(kinematics_dict, index=timeline_index)

            # 5d. Photometry alignment with NaN fallback
            phot_aligned = None
            has_fp = hasattr(s, "photometry") and isinstance(s.photometry, dict) and "470" in s.photometry
            if has_fp:
                try:
                    phot_df = combined_photometry_and_syllables(s)
                    phot_aligned = phot_df.reindex(timeline_index).reset_index(drop=True)
                    # Ensure all expected columns are present
                    for col in TEMPLATE_PHOT_COLS:
                        if col not in phot_aligned.columns:
                            phot_aligned[col] = np.nan
                    phot_aligned = phot_aligned[TEMPLATE_PHOT_COLS]
                    phot_present_count += 1
                except Exception as e:
                    print(f"  [warning] Photometry alignment failed for {sid}: {e}. Filling with NaNs.", flush=True)
                    phot_aligned = pd.DataFrame(np.nan, index=timeline_index, columns=TEMPLATE_PHOT_COLS)
                    phot_nan_count += 1
            else:
                phot_aligned = pd.DataFrame(np.nan, index=timeline_index, columns=TEMPLATE_PHOT_COLS)
                phot_nan_count += 1

            # 5e. Enrich video_refs_df with Bottom Camera AVI
            key = (cfg["ts"], cfg["arena"])
            bot_avi = avi_map.get(key)
            has_bot_vid = False
            if hasattr(s, "video_refs_df") and s.video_refs_df is not None and not s.video_refs_df.empty:
                if "role" in s.video_refs_df.columns:
                    has_bot_vid = (s.video_refs_df["role"] == "bottom_camera").any()

            if not has_bot_vid and bot_avi is not None:
                bot_row = pd.DataFrame([{
                    "path": str(bot_avi),
                    "path_type": "video",
                    "kind": "avi",
                    "role": "bottom_camera",
                    "ts": cfg["ts"],
                    "chamber": cfg["quadrant"],
                    "camset": f"{cfg['arena']}_BotCamVid",
                    "cam_slot": "bottom_cam",
                    "arena": cfg["arena"],
                    "camera_key": f"bottom_cam:{cfg['quadrant']}",
                    "fps": 60.0,
                    "n_frames": n_frames,
                    "duration_s": n_frames / 60.0,
                    "meta_origin": "opencv"
                }])
                s.video_refs_df = pd.concat([s.video_refs_df, bot_row], ignore_index=True)

            # 5f. Enrich tracking_refs_df with TopCam, BotCam, and Fused 23kp tracking files
            fused_h5 = REPO_ROOT / f"syllable_generation/multiview_top_bot_experiments/fused_h5_full_cohort/{sid}_tandem_23kp.h5"
            tracking_rows = [
                {"path": str(cfg["top_h5"]), "role": "top_dlc_tracking", "kind": "h5", "chamber": cfg["quadrant"]},
                {"path": str(cfg["bot_h5"]), "role": "bot_dlc_tracking", "kind": "h5", "chamber": cfg["quadrant"]},
                {"path": str(fused_h5), "role": "fused_tandem_23kp_tracking", "kind": "h5", "chamber": cfg["quadrant"]}
            ]
            s.tracking_refs_df = pd.DataFrame(tracking_rows)

            s.all_data_df = pd.concat([df_behavior.reset_index(drop=True), phot_aligned], axis=1)
            modeled_sessions.append(s)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(selected_sessions):
                print(f"  Processed {len(modeled_sessions)}/{len(results_keys)} sessions...", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY OF POST-PROCESSING:", flush=True)
    print(f"  Total Modeled Sessions:   {len(modeled_sessions)}", flush=True)
    print(f"  Skipped (Not in H5):       {len(skipped_sessions)} ({skipped_sessions})", flush=True)
    print(f"  Sessions with Photometry: {phot_present_count}", flush=True)
    print(f"  Sessions with NaN Phot:   {phot_nan_count}", flush=True)
    total_frames = sum(len(s.all_data_df) for s in modeled_sessions)
    print(f"  Total Frames Modeled:     {total_frames:,}", flush=True)
    print("=" * 80, flush=True)

    OUTPUT_DATA_PKL = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_tandem_cohort.pkl"

    print(f"\n6. Serializing output pickle to {OUTPUT_PROD_PKL} and {OUTPUT_DATA_PKL}...", flush=True)
    OUTPUT_PROD_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PROD_PKL, "wb") as f:
        pickle.dump(modeled_sessions, f)

    OUTPUT_DATA_PKL.parent.mkdir(parents=True, exist_ok=True)
    # Remove existing link/file if present before writing
    if OUTPUT_DATA_PKL.exists():
        OUTPUT_DATA_PKL.unlink()
    # Create hard link to save 12GB disk space
    import os
    os.link(OUTPUT_PROD_PKL, OUTPUT_DATA_PKL)

    file_size_mb = OUTPUT_PROD_PKL.stat().st_size / (1024 * 1024)
    elapsed_sec = time.time() - t_start

    print("=" * 80, flush=True)
    print(f"🎉 EXPORT COMPLETED SUCCESSFULLY IN {elapsed_sec:.1f}s!", flush=True)
    print(f"  Saved Files: {OUTPUT_PROD_PKL.name} & {OUTPUT_DATA_PKL.name} ({file_size_mb:.2f} MB)")
    print(f"  Locations:   {OUTPUT_PROD_PKL}\n               {OUTPUT_DATA_PKL}")
    print("=" * 80, flush=True)

    send_ntfy(
        "KPMS Cohort Production Export Succeeded! 🎉",
        f"Exported {len(modeled_sessions)} sessions ({total_frames:,} frames) in {elapsed_sec:.1f}s.\n"
        f"File: {OUTPUT_PROD_PKL.name} ({file_size_mb:.1f} MB)\n"
        f"Photometry: {phot_present_count} present, {phot_nan_count} NaN placeholders.",
        priority="high"
    )


if __name__ == "__main__":
    main()
