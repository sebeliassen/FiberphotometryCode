"""
fpcode.kpms.export
==================
Extraction and weaving engines for Keypoint-MoSeq model results.
Generates results.h5 / results.pkl and weaves multi-view kinematics and photometry
into canonical session data structures.
"""

from __future__ import annotations

import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from fpcode.common.logging import get_logger

logger = get_logger("fpcode.kpms.export")

TEMPLATE_PHOT_COLS = [
    "SystemTimestamp",
    "signal_0", "signal_1", "signal_2", "signal_3",
    "signal_0_zscore_ransac", "signal_0_zscore_wallace", "signal_0_dff", "signal_0_zscore", "signal_0_prominence",
    "signal_1_zscore_ransac", "signal_1_zscore_wallace", "signal_1_dff", "signal_1_zscore", "signal_1_prominence",
    "signal_2_zscore_ransac", "signal_2_zscore_wallace", "signal_2_dff", "signal_2_zscore", "signal_2_prominence",
    "signal_3_zscore_ransac", "signal_3_zscore_wallace", "signal_3_dff", "signal_3_zscore", "signal_3_prominence",
]


from fpcode.kpms.results import extract_and_export_results


def export_tandem_cohort_results(
    results_h5: Union[Path, str],
    dirty_pkl: Union[Path, str],
    output_pkl: Union[Path, str],
    output_data_pkl: Optional[Union[Path, str]] = None,
    k240k_ref_pkl: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """
    Weaves multiview top/bottom tracking kinematics, KPMS syllable classifications,
    and fiberphotometry data into session objects.
    """
    t_start = time.time()
    results_h5 = Path(results_h5).resolve()
    dirty_pkl = Path(dirty_pkl).resolve()
    output_pkl = Path(output_pkl).resolve()

    if not results_h5.exists():
        raise FileNotFoundError(f"results.h5 not found at {results_h5}")
    if not dirty_pkl.exists():
        raise FileNotFoundError(f"dirty sessions pkl not found at {dirty_pkl}")

    from fpcode.photometry.alignment import combined_photometry_and_syllables
    from fpcode.tracking.pair_matching import find_matched_pairs

    pairs = find_matched_pairs()
    pair_map = {p["sid"]: p for p in pairs}

    bot_avis = list(Path("/mnt/hdd/karen_project").rglob("*BotCamVid*.avi")) if Path("/mnt/hdd/karen_project").exists() else []
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

    with open(dirty_pkl, "rb") as f:
        dirty_sessions = pickle.load(f)
    selected_sessions = [s for s in dirty_sessions if s.session_id in pair_map]

    k240k_map = {}
    if k240k_ref_pkl and Path(k240k_ref_pkl).exists():
        try:
            with open(k240k_ref_pkl, "rb") as f:
                k240k_all = pickle.load(f)
            for s_ref in k240k_all:
                if s_ref.session_id in pair_map:
                    k240k_map[s_ref.session_id] = s_ref
        except Exception as e:
            logger.warning(f"Could not load K240K reference: {e}")

    with h5py.File(results_h5, "r") as f_res:
        results_keys = set(f_res.keys())
        modeled_sessions = []
        skipped_sessions = []
        phot_present_count = 0
        phot_nan_count = 0

        for s in selected_sessions:
            sid = s.session_id
            cfg = pair_map[sid]
            rec_key = f"{sid}_tandem_23kp"

            if rec_key not in results_keys:
                skipped_sessions.append(sid)
                continue

            grp = f_res[rec_key]
            tandem_sylls = np.array(grp["syllable"])
            latent_states = np.array(grp["latent_state"])
            moseq_centroid = np.array(grp["centroid"])
            moseq_heading = np.array(grp["heading"])

            n_frames = len(tandem_sylls)
            timeline_index = pd.RangeIndex(0, n_frames)

            s.results_dict_key = rec_key
            syll_dict = {"syllable": pd.Series(tandem_sylls, dtype="Int64")}
            for d in range(latent_states.shape[1]):
                syll_dict[f"latent_state_{d}"] = latent_states[:, d]
            syll_dict["centroid_0"] = moseq_centroid[:, 0]
            syll_dict["centroid_1"] = moseq_centroid[:, 1]
            syll_dict["heading"] = moseq_heading
            s.syllables = pd.DataFrame(syll_dict, index=timeline_index)

            # TopCam Kinematics
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

            # BotCam Kinematics
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

            # Photometry alignment
            has_fp = hasattr(s, "photometry") and isinstance(s.photometry, dict) and "470" in s.photometry
            if has_fp:
                try:
                    phot_df = combined_photometry_and_syllables(s)
                    phot_aligned = phot_df.reindex(timeline_index).reset_index(drop=True)
                    for col in TEMPLATE_PHOT_COLS:
                        if col not in phot_aligned.columns:
                            phot_aligned[col] = np.nan
                    phot_aligned = phot_aligned[TEMPLATE_PHOT_COLS]
                    phot_present_count += 1
                except Exception:
                    phot_aligned = pd.DataFrame(np.nan, index=timeline_index, columns=TEMPLATE_PHOT_COLS)
                    phot_nan_count += 1
            else:
                phot_aligned = pd.DataFrame(np.nan, index=timeline_index, columns=TEMPLATE_PHOT_COLS)
                phot_nan_count += 1

            s.all_data_df = pd.concat([df_behavior.reset_index(drop=True), phot_aligned], axis=1)
            modeled_sessions.append(s)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(modeled_sessions, f)

    if output_data_pkl:
        out_d = Path(output_data_pkl)
        out_d.parent.mkdir(parents=True, exist_ok=True)
        if out_d.exists():
            out_d.unlink()
        try:
            os.link(output_pkl, out_d)
        except OSError:
            with open(out_d, "wb") as f:
                pickle.dump(modeled_sessions, f)

    elapsed = time.time() - t_start
    return {
        "output_pkl": str(output_pkl),
        "total_sessions": len(modeled_sessions),
        "skipped_sessions": skipped_sessions,
        "photometry_present": phot_present_count,
        "photometry_nan": phot_nan_count,
        "elapsed_seconds": elapsed,
    }
