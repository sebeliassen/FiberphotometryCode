import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from analysis.replications.panel_1z.lib import utils
from scipy import stats

logger = logging.getLogger("Panel1z.Specificity")

# --- Configuration ---
WINDOW_SEC = 0.4
MIN_TRIALS = 100
REMOVE_IMMEDIATE_SEC = 0.025 # Skip peaks within first 25ms (artifact check)
TARGET_COLOR = "G"
IGNORE_GENOTYPES = ["UNK", "nan", None]

def run(sessions, config):
    """
    Run Syllable Specificity Analysis (Kruskal-Wallis ANOVA).
    """
    logger.info("Gathering data for Specificity...")

    # Structure: data_buckets[key][syllable] = [peak_values]
    data_buckets = defaultdict(lambda: defaultdict(list))

    for s in sessions:
        meta = s.metadata
        geno = meta.get("group", meta.get("genotype", "UNK"))
        if str(geno) in IGNORE_GENOTYPES: continue

        df = s.all_data_df
        if df.empty: continue

        fs = meta.get('fs', 30.0)
        win_frames = int(WINDOW_SEC * fs)

        # Onsets
        onsets = df["onset"].to_numpy()
        onset_idxs = np.flatnonzero(onsets)
        syll_array = df["syllable"].to_numpy()

        # Signal Loop
        sig_meta = meta.get("signal_meta", {})
        for sig_key, info in sig_meta.items():
            if str(info.get("orig_letter", "UNK")).upper() != TARGET_COLOR: continue

            region = info.get("brain_reg", "UNK")
            if region == "UNK": continue

            # Select Column
            col = None
            if f"{sig_key}_zscore_datta" in df.columns: col = f"{sig_key}_zscore_datta"
            elif f"{sig_key}_zscore" in df.columns: col = f"{sig_key}_zscore"

            if not col: continue
            sig_data = df[col].to_numpy()
            max_idx = len(sig_data)

            key = f"{region} | {meta.get('group', geno)}"

            for idx in onset_idxs:
                if idx + win_frames >= max_idx: continue

                segment = sig_data[idx : idx + win_frames]
                if np.isnan(segment).all(): continue

                # Peak as max abs deviation
                abs_idx = np.nanargmax(np.abs(segment))
                peak_val = segment[abs_idx]

                # Artifact Check
                if REMOVE_IMMEDIATE_SEC and (abs_idx / fs < REMOVE_IMMEDIATE_SEC):
                    continue

                data_buckets[key][syll_array[idx]].append(peak_val)

    logger.info("   Aggregation complete.")

    # 2. Stats & Plots
    stats_rows = []

    # Iterate conditions for separate plots + combined stats
    for key, syll_dict in sorted(data_buckets.items()):
        valid_signed = []
        valid_abs = []
        plot_items = []

        for syll, values in syll_dict.items():
            if len(values) >= MIN_TRIALS:
                arr = np.array(values)
                valid_signed.append(arr)
                valid_abs.append(np.abs(arr))

                plot_items.append({
                    "Syllable": str(syll),
                    "Mean_Abs": np.mean(np.abs(arr)),
                    "Values": arr
                })

        if len(valid_signed) < 2: continue

        # ANOVA
        h_sgn, p_sgn = stats.kruskal(*valid_signed)
        h_abs, p_abs = stats.kruskal(*valid_abs)

        stats_rows.append({
            "Condition": key,
            "H_Signed": h_sgn,
            "p_Signed": p_sgn,
            "H_Abs": h_abs,
            "p_Abs": p_abs,
            "N_Syllables": len(valid_signed)
        })

        # --- Plot for this condition ---
        # Sort top 15
        plot_items.sort(key=lambda x: x["Mean_Abs"], reverse=True)
        top_subset = plot_items[:15]

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, dpi=150)

        # Magnitude
        rows_abs = []
        for item in top_subset:
            for val in np.abs(item["Values"]):
                rows_abs.append({"Syllable": item["Syllable"], "|Peak Z|": val})

        if rows_abs:
            sns.boxplot(data=pd.DataFrame(rows_abs), x="Syllable", y="|Peak Z|", hue="Syllable",
                        legend=False, ax=axes[0], palette="Reds", flierprops={"marker":".","alpha":0.3})
        axes[0].set_title(f"A. Magnitude Specificity: {key}")

        # Signed
        rows_sgn = []
        for item in top_subset:
            for val in item["Values"]:
                rows_sgn.append({"Syllable": item["Syllable"], "Peak Z": val})

        if rows_sgn:
            sns.boxplot(data=pd.DataFrame(rows_sgn), x="Syllable", y="Peak Z", hue="Syllable",
                        legend=False, ax=axes[1], palette="coolwarm", flierprops={"marker":".","alpha":0.3})
        axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
        axes[1].set_title(f"B. Signed Specificity: {key}")

        sns.despine()
        plt.tight_layout()
        safe_key = key.replace(" | ", "_").replace(" ", "_")
        utils.save_plot(fig, f"specificity_top_syllables_{safe_key}", config)

    if not stats_rows:
        logger.warning("No data for Specificity stats.")
        return

    # 3. Save
    df_stats = pd.DataFrame(stats_rows)
    utils.save_stats(df_stats, "specificity_stats", config)
