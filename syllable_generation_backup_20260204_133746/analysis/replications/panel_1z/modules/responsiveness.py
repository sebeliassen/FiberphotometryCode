import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import logging

from analysis.replications.panel_1z.lib import utils

logger = logging.getLogger("Panel1z.Responsiveness")

# --- Configuration ---
BASE_WIN = (-1.0, -0.5)
RESP_WIN = (0.0, 0.5)
MIN_TRIALS = 50
TARGET_COLOR = "G"
IGNORE_GENOTYPES = ["UNK", "nan", None]

def run(sessions, config):
    """
    Run Responsiveness Analysis (Wilcoxon Response vs Baseline).
    """
    logger.info("Gathering data for Responsiveness...")
    
    # Structure: paired_data[key] = {"base": [], "resp": []}
    paired_data = defaultdict(lambda: {"base": [], "resp": []})
    
    for s in sessions:
        meta = s.metadata
        geno = meta.get("group", meta.get("genotype", "UNK"))
        if str(geno) in IGNORE_GENOTYPES: continue
        
        df = s.all_data_df
        if df.empty: continue
        
        fs = meta.get('fs', 30.0)
        
        # Windows to Frames
        b_start = int(BASE_WIN[0] * fs)
        b_end   = int(BASE_WIN[1] * fs)
        r_start = int(RESP_WIN[0] * fs)
        r_end   = int(RESP_WIN[1] * fs)
        
        # Onsets
        onsets = df["onset"].to_numpy()
        onset_idxs = np.flatnonzero(onsets)
        if len(onset_idxs) == 0: continue
        
        # Signal Loop
        sig_meta = meta.get("signal_meta", {})
        for sig_key, info in sig_meta.items():
            # Color
            color = str(info.get("orig_letter", "UNK")).upper()
            if color == "UNK" and "fiber_num" in info:
                color = "G" if int(info["fiber_num"]) % 2 == 0 else "R"
            if color != TARGET_COLOR: continue
            
            # Region (Patched names like L-DLS)
            region = info.get("brain_reg", "UNK")
            if region == "UNK": continue
            
            # Column
            col = None
            if f"{sig_key}_zscore_datta" in df.columns: col = f"{sig_key}_zscore_datta"
            elif f"{sig_key}_zscore" in df.columns: col = f"{sig_key}_zscore"
            
            if not col: continue
            sig_data = df[col].to_numpy()
            max_idx = len(sig_data)
            
            key = f"{region} | {meta.get('group', geno)}"
            
            for idx in onset_idxs:
                # Bound Check
                if (idx + b_start < 0) or (idx + r_end >= max_idx):
                    continue
                    
                b_slice = sig_data[idx + b_start : idx + b_end]
                r_slice = sig_data[idx + r_start : idx + r_end]
                
                # Nan check
                if np.isnan(b_slice).all() or np.isnan(r_slice).all():
                    continue
                    
                v_base = np.nanmean(b_slice)
                v_resp = np.nanmean(r_slice)
                
                if np.isnan(v_base) or np.isnan(v_resp):
                    continue
                
                paired_data[key]["base"].append(v_base)
                paired_data[key]["resp"].append(v_resp)

    logger.info("   Aggregation complete.")
    
    # 2. Stats & Plot
    stats_rows = []
    plot_rows = []
    
    sorted_keys = sorted(paired_data.keys())
    
    for key in sorted_keys:
        base = np.array(paired_data[key]["base"])
        resp = np.array(paired_data[key]["resp"])
        
        if len(base) < MIN_TRIALS: continue
        
        # Test
        stat, p = stats.wilcoxon(resp, base)
        delta = np.median(resp - base)
        
        stats_rows.append({
            "Condition": key,
            "N_Trials": len(base),
            "Median_Delta": delta,
            "Wilcoxon_p": p
        })
        
        # Plot data (downsample if huge to keep PDF/PNG size reasonable)
        diffs = resp - base
        if len(diffs) > 5000:
            diffs = np.random.choice(diffs, 5000, replace=False)
            
        for d in diffs:
            plot_rows.append({"Condition": key, "Signal Change (Z)": d})
            
    if not plot_rows:
        logger.warning("No data for Responsiveness plots.")
        return

    # 3. Save
    df_stats = pd.DataFrame(stats_rows)
    utils.save_stats(df_stats, "responsiveness_stats", config)
    
    df_plot = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    
    sns.violinplot(
        data=df_plot, 
        x="Condition", 
        y="Signal Change (Z)", 
        palette="coolwarm", 
        inner="quartile",
        ax=ax
    )
    ax.set_title(f"Signal Responsiveness\nWindow: {RESP_WIN}s vs {BASE_WIN}s")
    ax.set_ylabel("Change in Signal (Z-Score)")
    plt.xticks(rotation=45, ha='right')
    sns.despine(bottom=True)
    
    utils.save_plot(fig, "responsiveness_violins", config)
