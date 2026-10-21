import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from analysis.replications.panel_1z.lib import utils
from scipy import stats

logger = logging.getLogger("Panel1z.Hyperactivity")

# --- Configuration ---
METRICS = [
    ('velocity_p95_median', 'Speed (px/s)'),
    ('accel_p95_median', 'Acceleration (px/s²)')
]
MIN_TRIALS = 100
IGNORE_GENOTYPES = ["UNK", "nan", None]

def run(sessions, config):
    """
    Run Global Hyperactivity Phenotype Analysis.
    """
    logger.info("Analyzing Global Hyperactivity...")

    kin_rows = []
    for s in sessions:
        if hasattr(s, 'syllable_stats'):
            for row in s.syllable_stats:
                if row.get('count', 0) >= MIN_TRIALS:
                    kin_rows.append({
                        "animal_id": s.metadata.get("animal_id", "UNK"),
                        "genotype": s.metadata.get("group", s.metadata.get("genotype", "UNK")),
                        "velocity_p95_median": row.get("velocity_p95_median", np.nan),
                        "accel_p95_median": row.get("accel_p95_median", np.nan)
                    })

    if not kin_rows:
        logger.warning("No kinematics data for Hyperactivity.")
        return

    df = pd.DataFrame(kin_rows)
    df = df[~df["genotype"].isin(IGNORE_GENOTYPES)]

    # Aggregate to Animal Level
    animal_stats = df.groupby(["animal_id", "genotype"]).agg({
        'velocity_p95_median': 'mean',
        'accel_p95_median': 'mean'
    }).reset_index()

    if animal_stats.empty:
        logger.warning("No animal stats for Hyperactivity after filtering.")
        return

    # 3. Plotting & Stats
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

    stats_summary = []

    # Dynamic Palette
    PALETTE = {
        "WT": "grey", "KI": "firebrick",
        "WT Saline": "silver", "WT Amph": "dimgrey",
        "KI Saline": "salmon", "KI Amph": "darkred"
    }
    # Fallback to defaults if keys missing

    groups = sorted(animal_stats["genotype"].unique())

    for i, (col, label) in enumerate(METRICS):
        ax = axes[i]

        # Plot
        sns.boxplot(data=animal_stats, x="genotype", y=col, order=groups,
                    palette=PALETTE, showfliers=False, ax=ax)
        sns.stripplot(data=animal_stats, x="genotype", y=col, order=groups,
                      color="black", alpha=0.6, size=6, ax=ax)

        # Stats (Only if exactly 2 groups: WT vs KI)
        # If AMPH (4 groups), this simple T-test logic is insufficient.
        # We'll just skip stats for >2 groups for now or log it.
        if len(groups) == 2 and "WT" in groups and "KI" in groups:
            wt_vals = animal_stats[animal_stats["genotype"]=="WT"][col].dropna()
            ki_vals = animal_stats[animal_stats["genotype"]=="KI"][col].dropna()

            if len(wt_vals) > 1 and len(ki_vals) > 1:
                u, p = stats.mannwhitneyu(wt_vals, ki_vals)
                pct_change = ((ki_vals.mean() - wt_vals.mean()) / wt_vals.mean()) * 100

                sig_str = "*" if p < 0.05 else "ns"
                ax.set_title(f"{label}\n(Diff: {pct_change:+.1f}%, p={p:.1e} {sig_str})")

                stats_summary.append({
                    "Metric": label,
                    "WT_Mean": wt_vals.mean(),
                    "KI_Mean": ki_vals.mean(),
                    "Pct_Change": pct_change,
                    "MannWhitney_p": p
                })
        else:
            ax.set_title(label)

        ax.set_xlabel("")
        ax.set_ylabel(label)

        ax.set_xlabel("")
        ax.set_ylabel(label)

    sns.despine()
    plt.tight_layout()
    utils.save_plot(fig, "hyperactivity_global_phenotype", config)
    utils.save_stats(pd.DataFrame(stats_summary), "hyperactivity_stats", config)
