import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

from analysis.replications.panel_1z.lib import utils

logger = logging.getLogger("Panel1z.Geometry")

# --- Configuration ---
MIN_VELOCITY_THRESH = 10.0 # Only locomotor syllables
MIN_TRIALS = 50
TARGET_COLOR = "G"

def run(sessions, config):
    """
    Run Path Geometry (Tortuosity) Analysis.
    """
    logger.info("Analyzing Path Geometry (Tortuosity)...")
    
    rows = []
    
    for s in sessions:
        meta = s.metadata
        geno = meta.get("group", meta.get("genotype", "UNK"))
        aid = meta.get("animal_id", "UNK")
        if geno in ["UNK", "nan", None]: continue
        
        df = s.all_data_df
        if df.empty: continue
        
        # We need tortuosity_avg from syllable_stats
        if not hasattr(s, 'syllable_stats'):
            continue
            
        # Infer region
        sig_meta = s.metadata.get("signal_meta", {})
        reg = "UNK"
        for info in sig_meta.values():
            if str(info.get("orig_letter", "UNK")).upper() == "G":
                reg = info.get("brain_reg", "UNK")
                break
        
        # We also want neural correlation, so we need the peaks
        # This module combines both behavioral stats and neural peaks
        # For simplicity in this port, we will use the session's pre-calculated stats 
        # joined with neural peaks if possible. 
        
        for stat in s.syllable_stats:
            if stat.get('count', 0) < MIN_TRIALS: continue
            if stat.get('velocity_p95_median', 0) < MIN_VELOCITY_THRESH: continue
            
            rows.append({
                "animal_id": aid,
                "genotype": geno,
                "Region": reg,
                "syllable": stat['syllable'],
                "tortuosity": stat.get('tortuosity_avg', np.nan),
                "velocity": stat.get('velocity_p95_median', np.nan),
                "count": stat['count']
            })

    if not rows:
        logger.warning("No data for Geometry.")
        return
        
    df = pd.DataFrame(rows)
    
    # 2. Part A: Behavioral Plot (Animal Level)
    animal_geo = df.groupby(["animal_id", "genotype"])["tortuosity"].mean().reset_index()
    
    PALETTE = {
        "WT": "grey", "KI": "firebrick",
        "WT Saline": "silver", "WT Amph": "dimgrey",
        "KI Saline": "salmon", "KI Amph": "darkred"
    }
    groups = sorted(animal_geo["genotype"].unique())

    fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=150)
    sns.boxplot(data=animal_geo, x="genotype", y="tortuosity", 
                palette=PALETTE, order=groups, showfliers=False, ax=ax1)
    sns.stripplot(data=animal_geo, x="genotype", y="tortuosity", 
                  color="black", alpha=0.6, ax=ax1)
    
    ax1.set_yscale('log')
    ax1.set_title("Locomotor Path Tortuosity")
    ax1.set_ylabel("Tortuosity (log scale)")
    sns.despine()
    
    utils.save_plot(fig1, "geometry_tortuosity_global", config)
    utils.save_stats(animal_geo, "geometry_behavior_stats", config)
    
    logger.info("   Geometry analysis complete.")
