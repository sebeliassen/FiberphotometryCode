import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import re
from collections import defaultdict
from sklearn.metrics.pairwise import nan_euclidean_distances

# Try to import skbio, but don't crash if missing (it's a heavy dependency)
try:
    from skbio.stats.distance import permanova, DistanceMatrix
    SKBIO_AVAILABLE = True
except ImportError:
    SKBIO_AVAILABLE = False

from analysis.replications.panel_1z.lib import utils

logger = logging.getLogger("Panel1z.Fingerprint")

# --- Configuration ---
WINDOW_START = 0.0
WINDOW_END   = 0.5
MIN_TRIALS   = 10
TARGET_COLOR = "G"

def run(sessions, config):
    """
    Run Fingerprint / PERMANOVA Analysis.
    """
    if not SKBIO_AVAILABLE:
        logger.warning("skbio not found. Running pairwise distances only. (pip install scikit-bio for PERMANOVA)")

    logger.info("Building Fingerprints...")
    
    rows = []
    
    for s in sessions:
        meta = s.metadata
        aid = meta.get("animal_id", "UNK")
        geno = meta.get("group", meta.get("genotype", "UNK"))
        if geno in ["UNK", "nan", None]: continue
        
        df = s.all_data_df
        if df.empty: continue
        
        fs = meta.get('fs', 30.0)
        w_start = int(WINDOW_START * fs)
        w_end = int(WINDOW_END * fs)
        
        # Onsets
        onsets = df["onset"].to_numpy()
        onset_idxs = np.flatnonzero(onsets)
        syllables = df["syllable"].to_numpy()
        
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
            
            # Aggregate per syllable for this session
            syll_responses = defaultdict(list)
            for idx in onset_idxs:
                if idx + w_end >= max_idx: continue
                
                segment = sig_data[idx + w_start : idx + w_end]
                if np.isnan(segment).all(): continue
                
                syll_responses[syllables[idx]].append(np.nanmean(segment))
                
            # One row per Animal x Region x Syllable
            for syll, values in syll_responses.items():
                if len(values) >= MIN_TRIALS:
                    rows.append({
                        "Animal": aid,
                        "Genotype": geno,
                        "Region": region,
                        "Syllable": syll,
                        "Mean_Z": np.nanmean(values)
                    })

    if not rows:
        logger.warning("No data for Fingerprints.")
        return
        
    df_fp = pd.DataFrame(rows)
    logger.info(f"   Fingerprint DF built: {df_fp.shape}")
    
    # 2. PERMANOVA per Region
    regions = df_fp["Region"].unique()
    logger.info(f"   Regions found: {regions}")
    
    results = []
    dist_pairs = []
    
    for reg in regions:
        sub = df_fp[df_fp["Region"] == reg]
        
        # Pivot: (Animal, Genotype) x Syllable
        pivot = sub.pivot_table(index=["Animal", "Genotype"], columns="Syllable", values="Mean_Z")
        pivot = pivot.dropna(thresh=3, axis=0) # Must share some syllables
        
        logger.info(f"   Region {reg}: Pivot shape {pivot.shape}")
        
        if len(pivot) < 3: 
             logger.warning(f"   Region {reg}: Not enough animals ({len(pivot)} < 3).")
             continue
        
        # Check for multiple genotypes
        genos = pivot.index.get_level_values("Genotype").tolist()
        if len(set(genos)) < 2: continue
        
        if len(set(genos)) < 2: continue
        
        # Distance Matrix (Greedy)
        X = pivot.values
        D = nan_euclidean_distances(X, X)
        
        # Impute missing
        if np.isnan(D).any():
            mu = np.nanmean(D)
            D = np.nan_to_num(D, nan=mu)
            
        # Enforce symmetry
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        
        ids = pivot.index.get_level_values("Animal").tolist()
        
        # 1. Calc Pairs for Plotting (Always do this)
        for i in range(len(D)):
            for j in range(i+1, len(D)):
                dist_pairs.append({
                    "Region": reg,
                    "Distance": D[i,j],
                    "Type": "Within" if genos[i] == genos[j] else "Between"
                })

        # 2. PERMANOVA (If Available)
        if SKBIO_AVAILABLE:
            try:
                dm = DistanceMatrix(D, ids)
                res = permanova(dm, grouping=genos, permutations=999)
                results.append({
                    "Region": reg,
                    "F_Stat": res["test statistic"],
                    "p_value": res["p-value"],
                    "N_Animals": res["sample size"],
                    "N_Syllables": pivot.shape[1]
                })
            except Exception as e:
                logger.error(f"   PERMANOVA error on {reg}: {e}")

    # 3. Save & Plot
    if results:
        df_res = pd.DataFrame(results).sort_values("p_value")
        utils.save_stats(df_res, "permanova_results", config)
        
    if dist_pairs:
        df_pairs = pd.DataFrame(dist_pairs)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        sns.boxplot(data=df_pairs, x="Region", y="Distance", hue="Type", 
                    palette="Set2", showfliers=False, ax=ax)
        sns.stripplot(data=df_pairs, x="Region", y="Distance", hue="Type", 
                      dodge=True, ax=ax, alpha=0.3, size=3, legend=False)
        ax.set_title("Neural Fingerprint Divergence (Pairwise Euclidean)")
        utils.save_plot(fig, "fingerprint_distances", config)
