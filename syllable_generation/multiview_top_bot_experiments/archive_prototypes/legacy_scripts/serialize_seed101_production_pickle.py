#!/usr/bin/env python3
"""
serialize_seed101_production_pickle.py
=============================================================================
Dedicated Production Pickle Serializer for Champion Multi-View KPMS Seed 101:
- Extracts exact Seed 101 latent-derived syllables and kinematics from results.h5.
- Preserves distinct latent states and kinematic estimates from Seed 101 without
  altering or overwriting any baseline or other seed pickles.
- Produces: sessions_karen_project_tandem_cohort_seed101.pkl

Environment: kpms_gpu_cuml
=============================================================================
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import h5py
import pandas as pd
from scipy.ndimage import median_filter

# Setup repository paths
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.replications.datta_utils import combined_photometry_and_syllables
from syllable_generation.analysis.utils.sampling import flatten_results_to_timeline
from syllable_generation.multiview_top_bot_experiments.audit_cohort_correlations import find_matched_pairs
from syllable_generation.multiview_top_bot_experiments.core.ttl_sync import get_session_ttl_offset

# Paths
EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
PROD_KPMS_DIR = EXP_DIR / "kpms_tandem_cohort_production"
MODEL_NAME = "tandem_cohort_118sess_50ar_200slds_seed101"
MODEL_DIR = PROD_KPMS_DIR / MODEL_NAME
RESULTS_PATH = MODEL_DIR / "results.h5"
VFI_PATH = PROD_KPMS_DIR / "video_frame_indexes.npz"

DIRTY_PKL_IN = REPO_ROOT / "syllable_generation/data/sessions_karen_project_dirty.pkl"
K240K_PKL_REF = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_k240k.pkl"
OUTPUT_SEED101_PKL = EXP_DIR / "sessions_karen_project_tandem_cohort_seed101.pkl"

def main():
    t0 = time.time()
    print("=" * 80)
    print("🚀 SERIALIZING DEDICATED SEED 101 PRODUCTION PICKLE")
    print(f"Model:          {MODEL_NAME}")
    print(f"Results H5:     {RESULTS_PATH}")
    print(f"Dirty Pickle:   {DIRTY_PKL_IN}")
    print(f"Target Output:  {OUTPUT_SEED101_PKL}")
    print("=" * 80)

    # 1. Load Matched Pairs
    print("[1/5] Discovering matched tandem sessions...", flush=True)
    pairs = find_matched_pairs()
    pair_map = {p["sid"]: p for p in pairs}
    print(f"  Found {len(pairs)} matched session pairs.")

    # 2. Load Results from HDF5
    print("[2/5] Loading Seed 101 results from results.h5...", flush=True)
    results = {}
    with h5py.File(RESULTS_PATH, "r") as f:
        for k in f.keys():
            grp = f[k]
            results[k] = {
                "syllable": grp["syllable"][:],
                "latent_state": grp["latent_state"][:],
                "centroid": grp["centroid"][:],
                "heading": grp["heading"][:],
            }
    print(f"  Loaded results for {len(results)} sessions.")

    # 3. Load VFI and Reference Sessions
    print("[3/5] Loading Video Frame Indexes (VFI) and baseline session objects...", flush=True)
    vfi_dict = {}
    if VFI_PATH.exists():
        with np.load(VFI_PATH) as z:
            vfi_dict = {k: z[k].astype(int) for k in z.files}
        print(f"  Loaded VFI mappings for {len(vfi_dict)} sessions.")

    print(f"  Loading dirty session records from {DIRTY_PKL_IN.name}...", flush=True)
    with open(DIRTY_PKL_IN, "rb") as f:
        all_dirty_sessions = pickle.load(f)
    print(f"  Loaded {len(all_dirty_sessions)} total sessions from dirty pickle.")

    selected_sessions = [s for s in all_dirty_sessions if s.session_id in pair_map]
    print(f"  Filtered to {len(selected_sessions)} cohort sessions matching tandem pairs.")

    k240k_map = {}
    if K240K_PKL_REF.exists():
        print(f"  Loading K240K reference syllables from {K240K_PKL_REF.name}...", flush=True)
        try:
            with open(K240K_PKL_REF, "rb") as f:
                k240k_all = pickle.load(f)
            for s_ref in k240k_all:
                if s_ref.session_id in pair_map:
                    k240k_map[s_ref.session_id] = s_ref
            print(f"  Mapped {len(k240k_map)} reference sessions for top_syllable column.")
        except Exception as e:
            print(f"  [warning] Could not load K240K reference: {e}")

    # 4. Weave Seed 101 Kinematics, Syllables, and Photometry
    print("\n[4/5] Weaving Seed 101 kinematics, syllables, onsets, and photometry...", flush=True)
    modeled_sessions = []
    TEMPLATE_PHOT_COLS = [
        "SystemTimestamp",
        "signal_0", "signal_1", "signal_2", "signal_3",
        "signal_0_zscore_ransac", "signal_0_zscore_wallace", "signal_0_dff", "signal_0_zscore", "signal_0_prominence",
        "signal_1_zscore_ransac", "signal_1_zscore_wallace", "signal_1_dff", "signal_1_zscore", "signal_1_prominence",
        "signal_2_zscore_ransac", "signal_2_zscore_wallace", "signal_2_dff", "signal_2_zscore", "signal_2_prominence",
        "signal_3_zscore_ransac", "signal_3_zscore_wallace", "signal_3_dff", "signal_3_zscore", "signal_3_prominence"
    ]

    for idx, s in enumerate(selected_sessions):
        sid = s.session_id
        cfg = pair_map[sid]
        rec_key = f"{sid}_tandem_23kp"
        s.results_dict_key = rec_key
        s.kpms_seed = 101
        s.kpms_model_name = MODEL_NAME

        if rec_key not in results:
            print(f"  [{idx+1}/{len(selected_sessions)}] [SKIP] {rec_key} not in results.")
            continue

        res = results[rec_key]
        df_top = pd.read_hdf(cfg["top_h5"])
        df_bot = pd.read_hdf(cfg["bot_h5"])
        top_scorer = df_top.columns.levels[0][0]
        bot_scorer = df_bot.columns.levels[0][0]

        N_top = len(df_top)
        N_bot = len(df_bot)
        timeline_index = pd.RangeIndex(0, N_top)

        # 1. Flatten syllables to TopCam Master Timeline via VFI
        if rec_key in vfi_dict:
            flat_res = flatten_results_to_timeline(
                {rec_key: res},
                {rec_key: vfi_dict[rec_key]},
                actual_lengths={rec_key: N_top}
            )[rec_key]
            s.syllables = flat_res
        else:
            tandem_sylls = res["syllable"]
            syll_series = pd.Series(pd.NA, index=timeline_index, dtype="Int64")
            syll_series.iloc[:min(N_top, len(tandem_sylls))] = pd.Series(
                tandem_sylls[:min(N_top, len(tandem_sylls))], dtype="Int64"
            )
            s.syllables = pd.DataFrame({"syllable": syll_series}, index=timeline_index)

        if "syllable" in s.syllables.columns:
            s.syllables["syllable"] = s.syllables["syllable"].astype("Int64")

        # 2. TopCam Raw & Smoothed Kinematics
        top_nx = df_top[top_scorer, "nose", "x"].to_numpy(dtype=float)
        top_ny = df_top[top_scorer, "nose", "y"].to_numpy(dtype=float)
        top_tx = df_top[top_scorer, "tail_base", "x"].to_numpy(dtype=float)
        top_ty = df_top[top_scorer, "tail_base", "y"].to_numpy(dtype=float)

        cx_top = 0.5 * (top_nx + top_tx)
        cy_top = 0.5 * (top_ny + top_ty)
        heading_top = np.arctan2(top_ny - top_ty, top_nx - top_tx)

        dx_top = np.diff(cx_top)
        dy_top = np.diff(cy_top)
        vel_top_raw = np.pad(np.sqrt(dx_top**2 + dy_top**2) * 60.0, (0, 1), mode="edge")
        ang_vel_top = np.pad(np.diff(np.unwrap(heading_top)) * 60.0, (0, 1), mode="edge")
        vel_top_smooth = median_filter(np.clip(vel_top_raw, 0, 300.0), size=7, mode="nearest")

        # 3. BotCam Kinematics Aligned via Hardware TTL Offset
        offset_info = get_session_ttl_offset(sid)
        offset = int(offset_info.get("bot0_in_top", 0) - offset_info.get("top0_in_bot", 0))

        t_start = max(0, offset)
        t_end = min(N_top, N_bot + offset)
        top_overlap = np.arange(t_start, t_end)
        bot_overlap = top_overlap - offset

        bot_nx_raw = df_bot[bot_scorer, "nose_tip", "x"].to_numpy(dtype=float)
        bot_ny_raw = df_bot[bot_scorer, "nose_tip", "y"].to_numpy(dtype=float)
        bot_tx_raw = df_bot[bot_scorer, "tail_base", "x"].to_numpy(dtype=float)
        bot_ty_raw = df_bot[bot_scorer, "tail_base", "y"].to_numpy(dtype=float)

        cx_bot_native = 0.5 * (bot_nx_raw + bot_tx_raw)
        cy_bot_native = 0.5 * (bot_ny_raw + bot_tx_raw)
        heading_bot_native = np.arctan2(bot_ny_raw - bot_ty_raw, bot_nx_raw - bot_tx_raw)

        dx_bot = np.diff(cx_bot_native)
        dy_bot = np.diff(cy_bot_native)
        vel_bot_raw_native = np.pad(np.sqrt(dx_bot**2 + dy_bot**2) * 60.0, (0, 1), mode="edge")
        ang_vel_bot_native = np.pad(np.diff(np.unwrap(heading_bot_native)) * 60.0, (0, 1), mode="edge")
        vel_bot_smooth_native = median_filter(np.clip(vel_bot_raw_native, 0, 300.0), size=7, mode="nearest")

        cx_bot = np.full(N_top, np.nan, dtype=float)
        cy_bot = np.full(N_top, np.nan, dtype=float)
        heading_bot = np.full(N_top, np.nan, dtype=float)
        vel_bot_smooth = np.full(N_top, np.nan, dtype=float)
        vel_bot_raw = np.full(N_top, np.nan, dtype=float)
        ang_vel_bot = np.full(N_top, np.nan, dtype=float)

        if len(top_overlap) > 0:
            cx_bot[top_overlap] = cx_bot_native[bot_overlap]
            cy_bot[top_overlap] = cy_bot_native[bot_overlap]
            heading_bot[top_overlap] = heading_bot_native[bot_overlap]
            vel_bot_smooth[top_overlap] = vel_bot_smooth_native[bot_overlap]
            vel_bot_raw[top_overlap] = vel_bot_raw_native[bot_overlap]
            ang_vel_bot[top_overlap] = ang_vel_bot_native[bot_overlap]

        # 4. Onset Transitions
        syll_series = s.syllables["syllable"]
        syll_vals = syll_series.to_numpy()
        is_valid = ~pd.isna(syll_vals)
        diff_mask = np.zeros(N_top, dtype=bool)
        if np.any(is_valid):
            first_idx = np.where(is_valid)[0][0]
            diff_mask[first_idx] = True
            for t in range(first_idx + 1, N_top):
                if is_valid[t] and (not is_valid[t-1] or syll_vals[t] != syll_vals[t-1]):
                    diff_mask[t] = True

        moseq_heading = s.syllables["heading"].to_numpy() if "heading" in s.syllables.columns else heading_top

        kinematics_dict = {
            "frame_index": timeline_index.to_numpy(),
            "name": [rec_key] * N_top,
            "centroid_x": cx_top,
            "centroid_y": cy_top,
            "heading": moseq_heading,
            "heading_top": heading_top,
            "heading_moseq": moseq_heading,
            "angular_velocity": ang_vel_top,
            "velocity_px_s": vel_top_smooth,
            "velocity_px_s_raw": vel_top_raw,
            "syllable": syll_series,
            "tandem_syllable": syll_series,
            "group": ["default"] * N_top,
            "onset": diff_mask,
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
            k_syll = k240k_map[sid].all_data_df["syllable"]
            k_syll_arr = np.full(N_top, np.nan)
            k_len = min(N_top, len(k_syll))
            k_syll_arr[:k_len] = k_syll.iloc[:k_len].to_numpy()
            kinematics_dict["top_syllable"] = pd.Series(k_syll_arr, dtype="Int64")

        df_behavior = pd.DataFrame(kinematics_dict, index=timeline_index)

        # 5. Photometry Signals
        has_fp = hasattr(s, "photometry") and isinstance(s.photometry, dict) and "470" in s.photometry
        if has_fp:
            try:
                phot_df = combined_photometry_and_syllables(s)
                phot_aligned = phot_df.reindex(timeline_index).reset_index(drop=True)
                for col in TEMPLATE_PHOT_COLS:
                    if col not in phot_aligned.columns:
                        phot_aligned[col] = np.nan
                phot_aligned = phot_aligned[TEMPLATE_PHOT_COLS]
            except Exception as e:
                print(f"  [warning] Photometry alignment failed for {sid}: {e}. Filling with NaNs.")
                phot_aligned = pd.DataFrame(np.nan, index=timeline_index, columns=TEMPLATE_PHOT_COLS)
        else:
            phot_aligned = pd.DataFrame(np.nan, index=timeline_index, columns=TEMPLATE_PHOT_COLS)

        s.all_data_df = pd.concat([df_behavior.reset_index(drop=True), phot_aligned], axis=1)
        modeled_sessions.append(s)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(selected_sessions):
            print(f"  Processed [{idx+1}/{len(selected_sessions)}] sessions...", flush=True)

    # 5. Serialize Dedicated Seed 101 Pickle
    print(f"\n[5/5] Serializing {len(modeled_sessions)} sessions to {OUTPUT_SEED101_PKL.name}...", flush=True)
    OUTPUT_SEED101_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SEED101_PKL, "wb") as f:
        pickle.dump(modeled_sessions, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = OUTPUT_SEED101_PKL.stat().st_size / (1024 * 1024)
    total_frames = sum(len(s.all_data_df) for s in modeled_sessions)
    elapsed_sec = time.time() - t0

    print("=" * 80)
    print("🎉 DEDICATED SEED 101 PRODUCTION PICKLE SERIALIZED SUCCESSFULLY!")
    print(f"  Output File:   {OUTPUT_SEED101_PKL}")
    print(f"  File Size:     {size_mb:.2f} MB")
    print(f"  Sessions:      {len(modeled_sessions)}")
    print(f"  Total Frames:  {total_frames:,}")
    print(f"  Elapsed Time:  {elapsed_sec:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
