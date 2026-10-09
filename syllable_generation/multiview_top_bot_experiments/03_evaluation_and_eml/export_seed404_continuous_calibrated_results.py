#!/usr/bin/env python3
"""
export_seed404_continuous_calibrated_results.py
================================================
Post-processing exporter for Champion Seed 404 (Rank 1 EML Winner).
Reads trained results.h5 from kpms_tandem_cohort_continuous_calibrated,
aligns TopCam and BotCam kinematics using crystal-clock continuous correspondence
(core/camera_correspondence.py), attaches canonical photometry columns,
and serializes sessions_karen_project_tandem_cohort_seed404.pkl.
"""

import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from core.camera_correspondence import top_to_bot_frame_continuous
from core.pair_matching import find_matched_pairs
from syllable_generation.analysis.replications.datta_utils import combined_photometry_and_syllables

DIRTY_PKL = REPO_ROOT / "syllable_generation/data/sessions_karen_project_dirty.pkl"
K240K_PKL_REF = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_k240k.pkl"
PROJECT_DIR = EXP_DIR / "kpms_tandem_cohort_continuous_calibrated"
MODEL_NAME = "tandem_calibrated_50ar_200slds_seed404_20261002_012237_940073"
RESULTS_H5 = PROJECT_DIR / MODEL_NAME / "results.h5"
VFI_PATH = PROJECT_DIR / "video_frame_indexes.npz"
OUTPUT_PROD_PKL = EXP_DIR / "data/session_caches/sessions_karen_project_tandem_cohort_seed404.pkl"

TEMPLATE_PHOT_COLS = [
    "SystemTimestamp",
    "signal_0", "signal_1", "signal_2", "signal_3",
    "signal_0_zscore_ransac", "signal_0_zscore_wallace", "signal_0_dff", "signal_0_zscore", "signal_0_prominence",
    "signal_1_zscore_ransac", "signal_1_zscore_wallace", "signal_1_dff", "signal_1_zscore", "signal_1_prominence",
    "signal_2_zscore_ransac", "signal_2_zscore_wallace", "signal_2_dff", "signal_2_zscore", "signal_2_prominence",
    "signal_3_zscore_ransac", "signal_3_zscore_wallace", "signal_3_dff", "signal_3_zscore", "signal_3_prominence"
]

def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("EXPORT CONTINUOUS-CALIBRATED SEED 404 PRODUCTION SESSIONS")
    print(f"Model: {MODEL_NAME}")
    print(f"Results: {RESULTS_H5}")
    print(f"Target Output: {OUTPUT_PROD_PKL}")
    print("=" * 80, flush=True)

    OUTPUT_PROD_PKL.parent.mkdir(parents=True, exist_ok=True)

    print("\n1. Discovering matched pairs and VFI...", flush=True)
    pairs = find_matched_pairs()
    pair_map = {p["sid"]: p for p in pairs}
    print(f"  Found {len(pairs)} matched session pairs in cohort.", flush=True)

    with np.load(VFI_PATH) as z:
        vfi_dict = {k: z[k] for k in z.files}
    print(f"  Loaded VFI dictionary with {len(vfi_dict)} sessions.", flush=True)

    print("\n2. Loading dirty sessions...", flush=True)
    with open(DIRTY_PKL, "rb") as f:
        dirty_sessions = pickle.load(f)
    selected_sessions = [s for s in dirty_sessions if s.session_id in pair_map]
    print(f"  Selected {len(selected_sessions)} sessions from dirty pkl.", flush=True)

    print("\n3. Loading K240K reference dataset...", flush=True)
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
        print(f"  results.h5 contains {len(results_keys)} modeled segments/sessions.", flush=True)

        modeled_sessions = []
        skipped_sessions = []
        phot_present_count = 0
        phot_nan_count = 0

        print("\n5. Weaving syllables, kinematics, and photometry...", flush=True)
        for idx, s in enumerate(selected_sessions):
            sid = s.session_id
            cfg = pair_map[sid]

            # In continuous calibrated model, keys may be segmented (e.g. sid_seg000) or unsegmented
            # Check matching keys
            matching_keys = [k for k in results_keys if k.startswith(sid)]
            if not matching_keys:
                skipped_sessions.append(sid)
                continue

            # If segmented, sort keys in chronological segment order
            matching_keys = sorted(matching_keys)
            rec_key = matching_keys[0]

            # Concatenate segments if multiple
            syll_list = []
            latent_list = []
            centroid_list = []
            heading_list = []
            vfi_list = []

            for mk in matching_keys:
                grp = f_res[mk]
                syll_list.append(np.array(grp["syllable"]))
                latent_list.append(np.array(grp["latent_state"]))
                centroid_list.append(np.array(grp["centroid"]))
                heading_list.append(np.array(grp["heading"]))
                if mk in vfi_dict:
                    vfi_list.append(vfi_dict[mk])
                elif sid in vfi_dict:
                    vfi_list.append(vfi_dict[sid])

            tandem_sylls = np.concatenate(syll_list)
            latent_states = np.concatenate(latent_list, axis=0)
            moseq_centroid = np.concatenate(centroid_list, axis=0)
            moseq_heading = np.concatenate(heading_list)
            vfi_all = np.concatenate(vfi_list) if vfi_list else np.arange(len(tandem_sylls))

            n_frames = len(tandem_sylls)
            timeline_index = pd.RangeIndex(0, n_frames)

            s.results_dict_key = rec_key
            s.model_name = MODEL_NAME
            s.eml_rank = 1
            s.seed = 404

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

            # 5b. Extract TopCam Raw Kinematics via VFI indexing
            df_top = pd.read_hdf(cfg["top_h5"])
            top_scorer = df_top.columns.levels[0][0]

            top_raw_frames = vfi_all[:n_frames]
            top_raw_frames_clipped = np.clip(top_raw_frames, 0, len(df_top) - 1)

            top_nx = df_top[top_scorer, "nose", "x"].iloc[top_raw_frames_clipped].to_numpy(dtype=float)
            top_ny = df_top[top_scorer, "nose", "y"].iloc[top_raw_frames_clipped].to_numpy(dtype=float)
            top_tx = df_top[top_scorer, "tail_base", "x"].iloc[top_raw_frames_clipped].to_numpy(dtype=float)
            top_ty = df_top[top_scorer, "tail_base", "y"].iloc[top_raw_frames_clipped].to_numpy(dtype=float)

            cx_top = 0.5 * (top_nx + top_tx)
            cy_top = 0.5 * (top_ny + top_ty)
            heading_top = np.arctan2(top_ny - top_ty, top_nx - top_tx)

            dx_top = np.diff(cx_top)
            dy_top = np.diff(cy_top)
            vel_top_raw = np.pad(np.sqrt(dx_top**2 + dy_top**2) * 60.0, (0, 1), mode="edge")
            ang_vel_top = np.pad(np.diff(np.unwrap(heading_top)) * 60.0, (0, 1), mode="edge")
            vel_top_smooth = median_filter(np.clip(vel_top_raw, 0, 300.0), size=7, mode="nearest")

            # 5c. Extract BotCam Kinematics WITH STRICT CONTINUOUS CRYSTAL CALIBRATION
            df_bot = pd.read_hdf(cfg["bot_h5"])
            bot_scorer = df_bot.columns.levels[0][0]

            bot_raw_frames = [
                top_to_bot_frame_continuous(sid, int(tf))
                for tf in top_raw_frames
            ]
            bot_valid_mask = np.array([0 <= bf < len(df_bot) for bf in bot_raw_frames])

            # Sample valid rows or interpolate
            bot_indices_safe = np.array([bf if (0 <= bf < len(df_bot)) else 0 for bf in bot_raw_frames])

            bot_nx_raw = df_bot[bot_scorer, "nose_tip", "x"].iloc[bot_indices_safe].to_numpy(dtype=float)
            bot_ny_raw = df_bot[bot_scorer, "nose_tip", "y"].iloc[bot_indices_safe].to_numpy(dtype=float)
            bot_tx_raw = df_bot[bot_scorer, "tail_base", "x"].iloc[bot_indices_safe].to_numpy(dtype=float)
            bot_ty_raw = df_bot[bot_scorer, "tail_base", "y"].iloc[bot_indices_safe].to_numpy(dtype=float)

            # Mask dropped frames
            bot_nx_raw[~bot_valid_mask] = np.nan
            bot_ny_raw[~bot_valid_mask] = np.nan
            bot_tx_raw[~bot_valid_mask] = np.nan
            bot_ty_raw[~bot_valid_mask] = np.nan

            # Forward/backward fill small dropped frame gaps for kinematics continuity
            s_nx = pd.Series(bot_nx_raw).interpolate(limit=5).bfill().ffill().to_numpy()
            s_ny = pd.Series(bot_ny_raw).interpolate(limit=5).bfill().ffill().to_numpy()
            s_tx = pd.Series(bot_tx_raw).interpolate(limit=5).bfill().ffill().to_numpy()
            s_ty = pd.Series(bot_ty_raw).interpolate(limit=5).bfill().ffill().to_numpy()

            cx_bot = 0.5 * (s_nx + s_tx)
            cy_bot = 0.5 * (s_ny + s_ty)
            heading_bot = np.arctan2(s_ny - s_ty, s_nx - s_tx)

            dx_bot = np.diff(cx_bot)
            dy_bot = np.diff(cy_bot)
            vel_bot_raw = np.pad(np.sqrt(dx_bot**2 + dy_bot**2) * 60.0, (0, 1), mode="edge")
            ang_vel_bot = np.pad(np.diff(np.unwrap(heading_bot)) * 60.0, (0, 1), mode="edge")
            vel_bot_smooth = median_filter(np.clip(vel_bot_raw, 0, 300.0), size=7, mode="nearest")

            onset = np.concatenate([[True], tandem_sylls[1:] != tandem_sylls[:-1]])

            kinematics_dict = {
                "frame_index": timeline_index.to_numpy(),
                "topcam_raw_frame": top_raw_frames,
                "botcam_calibrated_frame": np.array(bot_raw_frames),
                "botcam_valid": bot_valid_mask,
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
                    df_comb = combined_photometry_and_syllables(s, kind="dff")
                    if df_comb is not None and len(df_comb) > 0:
                        phot_aligned = df_comb[[c for c in TEMPLATE_PHOT_COLS if c in df_comb.columns]].iloc[:n_frames]
                except Exception:
                    pass

            if phot_aligned is not None and len(phot_aligned) > 0:
                for col in phot_aligned.columns:
                    df_behavior[col] = phot_aligned[col].to_numpy()
                phot_present_count += 1
            else:
                for col in TEMPLATE_PHOT_COLS:
                    df_behavior[col] = np.nan
                phot_nan_count += 1

            s.all_data_df = df_behavior
            modeled_sessions.append(s)

            if (idx + 1) % 20 == 0 or idx == len(selected_sessions) - 1:
                print(f"  Processed {idx+1}/{len(selected_sessions)} sessions... (Photometry present: {phot_present_count}, NaN fallback: {phot_nan_count})", flush=True)

        print(f"\n6. Writing output pickle to {OUTPUT_PROD_PKL}...", flush=True)
        with open(OUTPUT_PROD_PKL, "wb") as f_out:
            pickle.dump(modeled_sessions, f_out, protocol=pickle.HIGHEST_PROTOCOL)

        sz_mb = OUTPUT_PROD_PKL.stat().st_size / (1024 * 1024)
        print("=" * 80, flush=True)
        print(f"🎉 EXPORT COMPLETE in {time.time() - t_start:.1f}s")
        print(f"  Output: {OUTPUT_PROD_PKL} ({sz_mb:.1f} MB)")
        print(f"  Modeled Sessions: {len(modeled_sessions)}")
        print(f"  Skipped Sessions: {len(skipped_sessions)}")
        print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
