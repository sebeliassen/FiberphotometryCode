import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from analysis.replications.panel_1z.lib import utils
from scipy import stats

logger = logging.getLogger("Panel1z.VigorValence")

# --- Configuration ---
WINDOW_SEC = 0.3
MIN_TRIALS = 10
KINEMATIC_FEATURES = [
    ('velocity_p95_median', 'Velocity'),
    ('accel_p95_median', 'Acceleration')
]
TARGET_COLOR = "G"

def run(sessions, config):
    """
    Run Vigor vs Valence Correlation Analysis.
    """
    logger.info("Gathering data for Vigor vs Valence...")

    # 1. Collect Neural Peaks
    neural_rows = []

    for s in sessions:
        meta = s.metadata
        geno = meta.get("group", meta.get("genotype", "UNK"))
        if geno in ["UNK", "nan", None]: continue

        df = s.all_data_df
        if df.empty: continue

        fs = meta.get('fs', 30.0)
        win_frames = int(WINDOW_SEC * fs)

        onsets = df["onset"].to_numpy()
        onset_idxs = np.flatnonzero(onsets)
        syllables = df["syllable"].to_numpy()

        sig_meta = meta.get("signal_meta", {})
        for sig_key, info in sig_meta.items():
            if str(info.get("orig_letter", "UNK")).upper() != TARGET_COLOR: continue
            region = info.get("brain_reg", "UNK")
            if region == "UNK": continue

            col = None
            if f"{sig_key}_zscore_datta" in df.columns: col = f"{sig_key}_zscore_datta"
            elif f"{sig_key}_zscore" in df.columns: col = f"{sig_key}_zscore"
            if not col: continue

            sig_data = df[col].to_numpy()
            max_idx = len(sig_data)

            sess_peaks = defaultdict(list)
            for idx in onset_idxs:
                if idx + win_frames >= max_idx: continue
                segment = sig_data[idx : idx + win_frames]
                if np.isnan(segment).all(): continue

                # Robust Peak (max abs)
                peak_val = segment[np.nanargmax(np.abs(segment))]
                sess_peaks[syllables[idx]].append(peak_val)

            for syll, pks in sess_peaks.items():
                if len(pks) >= MIN_TRIALS:
                    neural_rows.append({
                        "session_id": s.session_id,
                        "syllable": syll,
                        "Region": region,
                        "Peak_Z": np.nanmean(pks)
                    })

    if not neural_rows:
        logger.warning("No neural data for Vigor-Valence.")
        return

    df_da = pd.DataFrame(neural_rows)

    # 2. Extract Kinematics from original sessions if not in a combined DF
    # Actually, session objects usually have some pre-aggregated data or we calc it now.
    # In messy_north_star, it assumes a 'long_df' of aggregate stats exists.
    # We'll build it here.
    kin_rows = []
    for s in sessions:
        # Check if session has aggregate stats or we need to compute them
        # Usually sessions passed to Panel 1z should have 'syllable_stats' or similar
        # Based on messy_north_star: long_df = pd.DataFrame([row for s in sessions for row in s.syllable_stats])
        if hasattr(s, 'syllable_stats'):
            for row in s.syllable_stats:
                row['session_id'] = s.session_id
                row['genotype'] = s.metadata.get('group', s.metadata.get('genotype', 'UNK'))
                kin_rows.append(row)
        else:
            # Fallback: compute basic p95 velocity if missing
            logger.warning(f"Session {s.session_id} missing syllable_stats. Kinematics will be empty.")

    if not kin_rows:
        logger.error("No kinematics data (syllable_stats) found in sessions.")
        return

    df_kin = pd.DataFrame(kin_rows)

    # Merge
    merged = pd.merge(df_da, df_kin, on=["session_id", "syllable"], how="inner")
    if merged.empty:
        logger.warning("Merge of Neural and Kinematics resulted in empty DataFrame.")
        return

    # 3. Analyze each feature
    PALETTE = {
        "WT": "grey", "KI": "firebrick",
        "WT Saline": "silver", "WT Amph": "dimgrey",
        "KI Saline": "salmon", "KI Amph": "darkred"
    }

    for kin_col, kin_name in KINEMATIC_FEATURES:
        if kin_col not in merged.columns: continue

        plot_df = merged.groupby(["Region", "genotype", "syllable"]).agg({
            "Peak_Z": "mean",
            kin_col: "mean",
            "count": "sum"
        }).reset_index()

        # Filter for robust points (syllables shared across many instances)
        plot_df = plot_df[plot_df["count"] >= 100]
        if plot_df.empty: continue

        regions = sorted(plot_df["Region"].unique())
        fig, axes = plt.subplots(1, len(regions), figsize=(5*len(regions), 5), dpi=150)
        if len(regions) == 1: axes = [axes]

        stats_summary = []

        for ax, reg in zip(axes, regions):
            subset = plot_df[plot_df["Region"] == reg]
            groups = sorted(subset["genotype"].unique())

            sns.scatterplot(data=subset, x=kin_col, y="Peak_Z", hue="genotype",
                            palette=PALETTE, s=80, alpha=0.7, ax=ax)
            ax.axhline(0, color="black", linestyle="--", alpha=0.3)

            for geno in groups:
                g_data = subset[subset["genotype"] == geno]
                if len(g_data) < 5: continue

                r, p = stats.pearsonr(g_data[kin_col], g_data["Peak_Z"])
                slope, intercept, r_val, p_val, stderr = stats.linregress(g_data[kin_col], g_data["Peak_Z"])

                stats_summary.append({
                    "Region": reg, "Genotype": geno, "r": r, "p": p, "slope": slope, "stderr": stderr, "N": len(g_data)
                })

                color = PALETTE.get(geno, "grey")
                sns.regplot(data=g_data, x=kin_col, y="Peak_Z", scatter=False,
                            color=color, truncate=False, ax=ax)

            ax.set_title(f"{reg} ({kin_name})")
            ax.set_xlabel(kin_name)
            ax.set_ylabel("Peak Valence (Z)")

        utils.save_plot(fig, f"vigor_valence_{kin_name.lower()}", config)
        utils.save_stats(pd.DataFrame(stats_summary), f"vigor_valence_{kin_name.lower()}_stats", config)
