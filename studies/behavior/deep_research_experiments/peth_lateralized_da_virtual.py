#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# --- SETUP ---
this_file = Path(__file__).resolve()
repo_root = this_file.parents[3]

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = repo_root / "syllable_generation" / "output" / "lateralized_da_virtual"
# We can REUSE the exact same cache from the optimized script since the extracted traces are identical!
PETH_CACHE = repo_root / "syllable_generation" / "output" / "lateralized_da_optimized" / "PETH_lateralized_da_optimized.pkl"

TARGET_SYLLABLES_OPT = {
    # Left physical movements
    2: "Left Turn",
    5: "Turn Left",
    18: "Left Turn",
    28: "Turn Left",
    # Right physical movements
    12: "Right Turn",
    30: "Turn Right",
    46: "Look Right",
    59: "Lean Right"
}

def load_peths():
    if not PETH_CACHE.exists():
        print(f"[ERROR] Cached PETH data not found at {PETH_CACHE}. Run optimized script first.")
        return pd.DataFrame()

    print(f"Loading cached PETH data from {PETH_CACHE.name}...")
    return pd.read_pickle(PETH_CACHE)

def process_and_plot(peth_meta_df, average_by, share_y_axis):
    out_dir_name = f"plots_{average_by}_{'global' if share_y_axis else 'local'}_ylim"
    out_dir = BASE_OUTPUT_DIR / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generating {out_dir_name} ---")

    if average_by == "session":
        group_cols = ['session_id', 'genotype', 'drug', 'brain_reg', 'syllable', 'is_flipped']
        agg_df = peth_meta_df.groupby(group_cols).agg({'trace': lambda x: np.nanmean(np.vstack(x), axis=0), 'time_axis': 'first'}).reset_index()
    else:
        agg_df = peth_meta_df.copy()

    final_long_records = []

    for _, row in agg_df.iterrows():
        n_points = len(row['trace'])
        times = row['time_axis']
        traces = row['trace']

        base_dict = {
            'genotype': row['genotype'],
            'drug': row['drug'],
            'brain_reg': row['brain_reg'],
            'syllable': row['syllable'],
            'is_flipped': row['is_flipped'],
            'session_id': row.get('session_id', 'UNK')
        }

        for i in range(n_points):
            rec = base_dict.copy()
            rec['time'] = times[i]
            rec['zdFF'] = traces[i]
            final_long_records.append(rec)

    plot_df = pd.DataFrame(final_long_records)

    # Set up styling
    sns.set(style="whitegrid", context="talk", font_scale=0.8)

    regions = plot_df['brain_reg'].unique()
    for reg in regions:
        if reg == 'UNK': continue

        reg_dir = out_dir / f"Region_{reg}"
        reg_dir.mkdir(parents=True, exist_ok=True)

        reg_df = plot_df[plot_df['brain_reg'] == reg]
        unique_sylls = reg_df['syllable'].unique()

        for syll in tqdm(unique_sylls, desc=f"Syllables in {reg} ({out_dir_name})"):
            syll_df = reg_df[reg_df['syllable'] == syll]
            syll_name = TARGET_SYLLABLES_OPT[syll]

            row_order = ["WT", "KI"]
            col_order = ["Saline", "Amph-Pre", "Amph-Post"]

            g = sns.relplot(
                data=syll_df,
                x="time", y="zdFF",
                row="genotype", col="drug", row_order=row_order, col_order=col_order,
                kind="line", errorbar="se",
                height=3.5, aspect=1.2, color="purple",
                facet_kws={'sharey': share_y_axis}
            )

            avg_type = "Sessions Averaged" if average_by == "session" else "All Events Averaged"
            y_axis_type = "Shared Y-Axis" if share_y_axis else "Local Y-Axis"
            g.fig.suptitle(f"Virtual Region ({avg_type} | {y_axis_type})\nSyllable {syll}: {syll_name} | {reg}", y=1.08)
            g.set_axis_labels("Time from Onset (s)", "Z-Scored dF/F (Baseline Subtracted)")

            for ax in g.axes.flat:
                title_text = ax.get_title()
                geno_val = "WT" if "genotype = WT" in title_text else ("KI" if "genotype = KI" in title_text else "UNK")
                drug_val = "Saline" if "drug = Saline" in title_text else ("Amph-Pre" if "drug = Amph-Pre" in title_text else ("Amph-Post" if "drug = Amph-Post" in title_text else "UNK"))

                # Filter agg_df for counts
                sub_agg = agg_df[
                    (agg_df['syllable'] == syll) &
                    (agg_df['genotype'] == geno_val) &
                    (agg_df['drug'] == drug_val) &
                    (agg_df['brain_reg'] == reg)
                ]

                n_flip = len(sub_agg[sub_agg['is_flipped'] == 'Flipped Session'])
                n_non = len(sub_agg[sub_agg['is_flipped'] == 'Non-Flipped Session'])
                total_n = n_flip + n_non

                ax.set_title(f"{geno_val} | {drug_val}\n(Total n={total_n} | {n_non} Non, {n_flip} Flip)")

                ax.axvline(0, color='gray', linestyle='--', zorder=0)
                ax.axhline(0, color='gray', linewidth=0.8, zorder=0)

            out_path = reg_dir / f"Syllable_{syll}_virtual_{reg}.png"
            g.fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(g.fig)

def main():
    peth_df = load_peths()
    if peth_df.empty:
        return

    for avg_by in ["event", "session"]:
        for share_y in [True, False]:
            process_and_plot(peth_df, avg_by, share_y)

    print("\nAll virtual averaged plot variations generated successfully.")

if __name__ == "__main__":
    main()
