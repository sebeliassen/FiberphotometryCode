#!/usr/bin/env python3
"""
audit_cohort_correlations.py
=============================================================================
Audits cross-camera spatial and angular alignment across the full cohort:
1. Centroid X correlation: Corr(X_top, X_bot_corrected)
2. Centroid Y correlation: Corr(Y_top, Y_bot)
3. Circular Heading Correlation: Jammalamadaka-Sarma circular correlation
4. Heading Alignment: Mean Cosine Alignment and Median Absolute Angular Delta (with wrapping)
5. Outlier Detection: Flags and diagnoses any anomalous sessions.

Parallelized across CPU workers for fast execution on all 118 sessions.
=============================================================================
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from fpcode.common.schemas import PathConfig
from fpcode.tracking.pair_matching import (
    circular_correlation,
    find_matched_pairs,
)

REPO_ROOT = PathConfig().repo_root
TOP_H5_DIR = PathConfig().top_h5_dir
BOT_PROD_DIR = PathConfig().bot_prod_dir
OUT_CSV = PathConfig().experiment_dir / "cohort_alignment_audit.csv"

def audit_single_session(pair_info: dict, arena_width: float = 972.0) -> dict:
    sid = pair_info["sid"]
    top_h5_path = pair_info["top_h5"]
    bot_h5_path = pair_info["bot_h5"]
    arena = pair_info["arena"]
    quadrant = pair_info["quadrant"]

    try:
        df_top = pd.read_hdf(top_h5_path)
        df_bot = pd.read_hdf(bot_h5_path)

        st = df_top.columns.levels[0][0]
        sb = df_bot.columns.levels[0][0]

        n_frames = min(len(df_top), len(df_bot))
        active_slice = slice(0, n_frames)

        # TopCam landmarks
        top_nx = df_top[st, "nose", "x"].iloc[active_slice].to_numpy(dtype=np.float32)
        top_ny = df_top[st, "nose", "y"].iloc[active_slice].to_numpy(dtype=np.float32)
        top_tx = df_top[st, "tail_base", "x"].iloc[active_slice].to_numpy(dtype=np.float32)
        top_ty = df_top[st, "tail_base", "y"].iloc[active_slice].to_numpy(dtype=np.float32)
        top_like = np.minimum(
            df_top[st, "nose", "likelihood"].iloc[active_slice].to_numpy(dtype=np.float32),
            df_top[st, "tail_base", "likelihood"].iloc[active_slice].to_numpy(dtype=np.float32)
        )

        # BotCam landmarks (horizontal reflection x -> arena_width - x)
        bot_nx_raw = df_bot[sb, "nose_tip", "x"].iloc[active_slice].to_numpy(dtype=np.float32)
        bot_ny = df_bot[sb, "nose_tip", "y"].iloc[active_slice].to_numpy(dtype=np.float32)
        bot_tx_raw = df_bot[sb, "tail_base", "x"].iloc[active_slice].to_numpy(dtype=np.float32)
        bot_ty = df_bot[sb, "tail_base", "y"].iloc[active_slice].to_numpy(dtype=np.float32)
        bot_like = np.minimum(
            df_bot[sb, "nose_tip", "likelihood"].iloc[active_slice].to_numpy(dtype=np.float32),
            df_bot[sb, "tail_base", "likelihood"].iloc[active_slice].to_numpy(dtype=np.float32)
        )

        bot_nx = arena_width - bot_nx_raw
        bot_tx = arena_width - bot_tx_raw

        # Centroids
        c_top_x = 0.5 * (top_nx + top_tx)
        c_top_y = 0.5 * (top_ny + top_ty)
        c_bot_x = 0.5 * (bot_nx + bot_tx)
        c_bot_y = 0.5 * (bot_ny + bot_ty)

        # Lengths and reliable mask
        len_top = np.sqrt((top_nx - top_tx)**2 + (top_ny - top_ty)**2)
        len_bot = np.sqrt((bot_nx - bot_tx)**2 + (bot_ny - bot_ty)**2)

        reliable = (top_like > 0.5) & (bot_like > 0.5) & (len_top > 20.0) & (len_bot > 40.0)
        n_reliable = int(np.sum(reliable))
        rel_pct = (n_reliable / n_frames) * 100.0 if n_frames > 0 else 0.0

        # Scale
        if n_reliable > 100:
            scale_s = float(np.nanmedian(len_top[reliable]) / np.nanmedian(len_bot[reliable]))
        else:
            scale_s = 0.3716

        # Centroid correlations
        valid_c = reliable & ~np.isnan(c_top_x) & ~np.isnan(c_bot_x) & ~np.isnan(c_top_y) & ~np.isnan(c_bot_y)
        if np.sum(valid_c) > 100:
            corr_x = float(np.corrcoef(c_top_x[valid_c], c_bot_x[valid_c])[0, 1])
            corr_y = float(np.corrcoef(c_top_y[valid_c], c_bot_y[valid_c])[0, 1])
            raw_corr_x = float(np.corrcoef(c_top_x[valid_c], (arena_width - c_bot_x)[valid_c])[0, 1])
        else:
            corr_x, corr_y, raw_corr_x = np.nan, np.nan, np.nan

        # Headings
        theta_top = np.arctan2(top_ny - top_ty, top_nx - top_tx)
        theta_bot = np.arctan2(bot_ny - bot_ty, bot_nx - bot_tx)

        # Angular difference with circular wrapping [-pi, pi]
        d_theta = np.arctan2(np.sin(theta_top - theta_bot), np.cos(theta_top - theta_bot))

        if n_reliable > 100:
            circ_corr = circular_correlation(theta_top[reliable], theta_bot[reliable])
            cos_align = float(np.nanmean(np.cos(d_theta[reliable])))
            med_abs_dtheta_deg = float(np.degrees(np.nanmedian(np.abs(d_theta[reliable]))))
            mean_dtheta_deg = float(np.degrees(np.nanmedian(d_theta[reliable])))
            std_dtheta_deg = float(np.degrees(np.nanstd(d_theta[reliable])))
        else:
            circ_corr, cos_align, med_abs_dtheta_deg, mean_dtheta_deg, std_dtheta_deg = np.nan, np.nan, np.nan, np.nan, np.nan

        # Outlier flag
        is_outlier = (
            (corr_x < 0.95 or np.isnan(corr_x)) or
            (corr_y < 0.95 or np.isnan(corr_y)) or
            (cos_align < 0.90 or np.isnan(cos_align)) or
            (med_abs_dtheta_deg > 15.0 or np.isnan(med_abs_dtheta_deg)) or
            (rel_pct < 40.0)
        )

        return {
            "sid": sid,
            "arena": arena,
            "quadrant": quadrant,
            "n_frames": n_frames,
            "reliable_frames": n_reliable,
            "reliable_pct": rel_pct,
            "corr_x": corr_x,
            "corr_y": corr_y,
            "raw_corr_x": raw_corr_x,
            "circular_corr": circ_corr,
            "mean_cos_heading": cos_align,
            "median_abs_dtheta_deg": med_abs_dtheta_deg,
            "median_dtheta_deg": mean_dtheta_deg,
            "std_dtheta_deg": std_dtheta_deg,
            "scale_s": scale_s,
            "is_outlier": is_outlier,
            "error": None
        }

    except Exception as e:
        return {
            "sid": sid,
            "arena": arena,
            "quadrant": quadrant,
            "n_frames": 0,
            "reliable_frames": 0,
            "reliable_pct": 0.0,
            "corr_x": np.nan,
            "corr_y": np.nan,
            "raw_corr_x": np.nan,
            "circular_corr": np.nan,
            "mean_cos_heading": np.nan,
            "median_abs_dtheta_deg": np.nan,
            "median_dtheta_deg": np.nan,
            "std_dtheta_deg": np.nan,
            "scale_s": np.nan,
            "is_outlier": True,
            "error": str(e)
        }

# find_matched_pairs is imported from core.pair_matching above

def main():
    print("=" * 80)
    print("🔬 FULL COHORT CROSS-CAMERA CORRELATION & HEADING AUDIT")
    print(f"TopCam:   {TOP_H5_DIR}")
    print(f"BotCam:   {BOT_PROD_DIR}")
    print(f"Output:   {OUT_CSV}")
    print("=" * 80)

    pairs = find_matched_pairs()
    print(f"Found {len(pairs)} unique matched session pairs.\n")

    t0 = time.time()
    max_workers = min(16, os.cpu_count() or 4)
    print(f"Executing correlation calculations across {max_workers} CPU workers...\n")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(audit_single_session, p): p for p in pairs}
        for future in as_completed(future_map):
            res = future.result()
            results.append(res)
            flag = "⚠️ OUTLIER" if res["is_outlier"] else "✅ OK"
            print(f"  [{len(results):3d}/{len(pairs)}] {flag} {res['sid']:30s} | "
                  f"CorrX: {res['corr_x']:+.4f} | CorrY: {res['corr_y']:+.4f} | "
                  f"CircCorr: {res['circular_corr']:.3f} | Cos: {res['mean_cos_heading']:.3f} | "
                  f"|dθ|: {res['median_abs_dtheta_deg']:.1f}°", flush=True)

    df = pd.DataFrame(results).sort_values("sid").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"🎉 AUDIT COMPLETE in {elapsed:.1f} seconds!")
    print(f"Saved complete metrics to {OUT_CSV}\n")

    # Cohort Distribution Summary
    print("-" * 80)
    print("📊 COHORT DISTRIBUTION SUMMARY (All Matched Sessions):")
    print("-" * 80)
    valid_df = df[~df["corr_x"].isna()]
    print(f"Total sessions evaluated: {len(df)}")
    print(f"Valid sessions:           {len(valid_df)}")
    print(f"Corr X (top vs bot_refl): Median = {valid_df['corr_x'].median():+.4f} (IQR: {valid_df['corr_x'].quantile(0.25):+.4f} to {valid_df['corr_x'].quantile(0.75):+.4f}, Min: {valid_df['corr_x'].min():+.4f})")
    print(f"Corr Y (top vs bot):      Median = {valid_df['corr_y'].median():+.4f} (IQR: {valid_df['corr_y'].quantile(0.25):+.4f} to {valid_df['corr_y'].quantile(0.75):+.4f}, Min: {valid_df['corr_y'].min():+.4f})")
    print(f"Circular Heading Corr:    Median = {valid_df['circular_corr'].median():.4f} (IQR: {valid_df['circular_corr'].quantile(0.25):.4f} to {valid_df['circular_corr'].quantile(0.75):.4f}, Min: {valid_df['circular_corr'].min():.4f})")
    print(f"Mean Cosine Alignment:    Median = {valid_df['mean_cos_heading'].median():.4f} (IQR: {valid_df['mean_cos_heading'].quantile(0.25):.4f} to {valid_df['mean_cos_heading'].quantile(0.75):.4f}, Min: {valid_df['mean_cos_heading'].min():.4f})")
    print(f"Median |Δθ| (degrees):    Median = {valid_df['median_abs_dtheta_deg'].median():.2f}° (IQR: {valid_df['median_abs_dtheta_deg'].quantile(0.25):.2f}° to {valid_df['median_abs_dtheta_deg'].quantile(0.75):.2f}°, Max: {valid_df['median_abs_dtheta_deg'].max():.2f}°)")
    print(f"Median Scale s:           Median = {valid_df['scale_s'].median():.4f} (IQR: {valid_df['scale_s'].quantile(0.25):.4f} to {valid_df['scale_s'].quantile(0.75):.4f})")

    # Outliers
    outliers = df[df["is_outlier"]]
    print("\n" + "-" * 80)
    print(f"🚨 OUTLIER SESSIONS DETECTED: {len(outliers)} / {len(df)}")
    print("-" * 80)
    if len(outliers) > 0:
        cols_show = ["sid", "arena", "n_frames", "reliable_pct", "corr_x", "corr_y", "circular_corr", "mean_cos_heading", "median_abs_dtheta_deg", "scale_s", "error"]
        print(outliers[cols_show].to_string(index=False))
    else:
        print("  None! Every single session satisfies strict alignment thresholds.")
    print("=" * 80)

if __name__ == "__main__":
    main()
