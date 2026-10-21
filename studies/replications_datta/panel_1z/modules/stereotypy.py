import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from analysis.replications.panel_1z.lib import utils
from scipy import stats

logger = logging.getLogger("Panel1z.Stereotypy")

# --- Configuration ---
MIN_TRIALS_PER_SYLLABLE = 50
IGNORE_GENOTYPES = ["UNK", "nan", None]

def run(sessions, config):
    """
    Run Behavioral Stereotypy Analysis (Duration CV).
    """
    logger.info("Analyzing Behavioral Stereotypy...")

    kin_rows = []
    for s in sessions:
        if hasattr(s, 'syllable_stats'):
            reg = "UNK"
            # Infer region from green signal
            sig_meta = s.metadata.get("signal_meta", {})
            for info in sig_meta.values():
                if str(info.get("orig_letter", "UNK")).upper() == "G":
                    reg = info.get("brain_reg", "UNK")
                    break

            for row in s.syllable_stats:
                if row.get('count', 0) >= MIN_TRIALS_PER_SYLLABLE:
                    kin_rows.append({
                        "animal_id": s.metadata.get("animal_id", "UNK"),
                        "genotype": s.metadata.get("group", s.metadata.get("genotype", "UNK")),
                        "Region": reg,
                        "duration_cv": row.get("duration_cv", np.nan)
                    })

    if not kin_rows:
        logger.warning("No kinematics data for Stereotypy.")
        return

    df = pd.DataFrame(kin_rows)
    df = df[~df["genotype"].isin(IGNORE_GENOTYPES)]
    df = df[df["Region"] != "UNK"]

    # Aggregate to Animal Level (Global)
    animal_global = df.groupby(["animal_id", "genotype"])["duration_cv"].mean().reset_index()

    # Aggregate for Regional Plot (Animal x Region)
    animal_regional = df.groupby(["animal_id", "genotype", "Region"])["duration_cv"].mean().reset_index()

    if animal_global.empty:
        logger.warning("No animal stats for Stereotypy after filtering.")
        return

    # 2. Plotting & Stats
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, width_ratios=[1, 2])

    PALETTE = {
        "WT": "grey", "KI": "firebrick",
        "WT Saline": "silver", "WT Amph": "dimgrey",
        "KI Saline": "salmon", "KI Amph": "darkred"
    }
    groups = sorted(animal_global["genotype"].unique())

    # Global Boxplot
    sns.boxplot(data=animal_global, x="genotype", y="duration_cv",
                palette=PALETTE, order=groups, showfliers=False, ax=axes[0])
    sns.stripplot(data=animal_global, x="genotype", y="duration_cv",
                  color="black", alpha=0.6, size=6, order=groups, ax=axes[0])

    # Stats (Only if exactly 2 groups)
    if len(groups) == 2 and "WT" in groups and "KI" in groups:
        wt_vals = animal_global[animal_global["genotype"]=="WT"]["duration_cv"]
        ki_vals = animal_global[animal_global["genotype"]=="KI"]["duration_cv"]
        u, p = stats.mannwhitneyu(wt_vals, ki_vals)
        axes[0].set_title(f"Global Stereotypy\np={p:.1e}")
    else:
        axes[0].set_title("Global Stereotypy")

    # Region Split
    sns.boxplot(data=animal_regional, x="Region", y="duration_cv", hue="genotype",
                palette=PALETTE, hue_order=groups, showfliers=False, ax=axes[1])
    axes[1].set_title("Stereotypy by Cohort")

    sns.despine()
    plt.tight_layout()
    utils.save_plot(fig, "stereotypy_duration_cv", config)
    utils.save_stats(animal_global, "stereotypy_stats", config)
