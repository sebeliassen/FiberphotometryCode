import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import logging

# Ensure we can import from panel_1z and unpickle correctly
# We need both the project root (syllable_generation) for relative imports
# AND the parent of project root (FiberphotometryCode) for 'syllable_generation' package resolution
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
parent_dir = os.path.dirname(root_dir)
sys.path.append(root_dir)
sys.path.append(parent_dir)

from analysis.replications.panel_1z.lib import loader, utils

# --- Configuration ---
WINDOW_MS = 200 # Slope Window [0, 200ms]
TRACE_WINDOW_SEC = 0.5 # Extract slightly more context [-0.1, +0.4]
MIN_TRIALS = 30 # Lower threshold for pilot
IGNORE_GENOTYPES = ["UNK", "nan", None]

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PPE_Analysis")

def get_dls_signal(df, meta):
    """
    Identify DLS signal column.
    Strict Logic: Must be 'DLS' in signal_meta or SKIP.
    """
    sig_meta = meta.get("signal_meta", {})
    if not sig_meta:
        return None
        
    for key, info in sig_meta.items():
        region = str(info.get("brain_reg", "UNK"))
        if "DLS" in region:
            # Found DLS (L-DLS or R-DLS)
            if f"{key}_dff" in df.columns: return f"{key}_dff"
            if f"{key}_zscore_datta" in df.columns: return f"{key}_zscore_datta"
            
    return None

def calculate_slope(trace, fs):
    """
    Calculate linear slope of trace.
    Returns: slope
    """
    n = len(trace)
    if n < 2: return np.nan
    
    time = np.arange(n) / fs
    res = stats.linregress(time, trace)
    return res.slope

def run_analysis(sessions, output_dir):
    logger.info("Starting Action Prediction Error (PPE) Analysis with Vigor Regression...")
    
    slope_data = []
    
    for s in sessions:
        meta = s.metadata
        animal = meta.get("animal_id", "UNK")
        geno = meta.get("group", meta.get("genotype", "UNK"))
        if str(geno) in IGNORE_GENOTYPES: continue
        
        fs = meta.get('fs', 30.0)
        df = getattr(s, "all_data_df", None)
        if df is None or df.empty: continue
        
        # 1. Identify DLS Signal
        dls_col = get_dls_signal(df, meta)
        if not dls_col: continue 
            
        signal = df[dls_col].to_numpy()
        
        # Get Velocity
        vel_col = "velocity_px_s" if "velocity_px_s" in df.columns else None
        velocity = df[vel_col].to_numpy() if vel_col else None
        
        # 2. Get Onsets
        if "onset" in df.columns:
            onsets = df["onset"].fillna(0).astype(bool).to_numpy()
        else:
            onsets = (df["syllable"] != df["syllable"].shift(1)).fillna(False).to_numpy()
            
        onset_idxs = np.flatnonzero(onsets)
        syllables = df["syllable"].to_numpy()
        
        # 3. Iterate Syllables
        win_frames = int((WINDOW_MS / 1000) * fs)
        max_idx = len(signal)
        
        for idx in onset_idxs:
            syll = syllables[idx]
            if idx + win_frames >= max_idx: continue
            
            # Extract Trace Segment
            trace = signal[idx : idx + win_frames]
            
            if np.isnan(trace).any(): continue
            
            # Calculate Slope
            slope = calculate_slope(trace, fs)
            
            # Calculate Vigor (Mean Velocity in same window)
            vigor = np.nan
            if velocity is not None:
                v_trace = velocity[idx : idx + win_frames]
                if not np.isnan(v_trace).any():
                    vigor = np.mean(v_trace)
            
            if not np.isnan(slope):
                slope_data.append({
                    "Animal": animal,
                    "Genotype": geno,
                    "Syllable": syll,
                    "Slope": slope,
                    "Vigor": vigor
                })
                
    if not slope_data:
        logger.warning("No valid slope data found!")
        return
        
    df_slopes = pd.DataFrame(slope_data)
    
    # 4. Hierarchical Aggregation (Animal-Level)
    # Goal: Calculate (1) Raw Slope Variance, (2) Vigor-Corrected Residual Variance
    
    top_sylls = df_slopes["Syllable"].value_counts().head(5).index.tolist()
    df_top = df_slopes[df_slopes["Syllable"].isin(top_sylls)]
    
    logger.info(f"Analyzing Top 5 Syllables (Hierarchical + Vigor): {top_sylls}")
    
    stats_rows = []
    
    for syll in top_sylls:
        subset = df_top[df_top["Syllable"] == syll]
        
        animal_metrics = []
        
        for (geno, animal), grp in subset.groupby(["Genotype", "Animal"]):
            if len(grp) < MIN_TRIALS: continue
            
            # A. Raw Metric (Naive)
            raw_noise = np.std(grp["Slope"])
            
            # B. Vigor-Corrected Metric (Regression)
            # Regress Slope ~ Vigor
            # If no vigor data or constant vigor, residual = deviations from mean
            residuals = grp["Slope"] - np.mean(grp["Slope"]) # Default fallback
            
            valid_vigor = grp["Vigor"].dropna()
            if len(valid_vigor) == len(grp) and np.std(valid_vigor) > 1e-6:
                try:
                    slope, intercept, _, _, _ = stats.linregress(grp["Vigor"], grp["Slope"])
                    predicted = slope * grp["Vigor"] + intercept
                    residuals = grp["Slope"] - predicted
                except Exception:
                    pass
            
            corrected_noise = np.std(residuals)
            
            animal_metrics.append({
                "Genotype": geno,
                "Animal": animal,
                "Raw_Noise": raw_noise,
                "Corrected_Noise": corrected_noise,
                "N_Trials": len(grp)
            })
            
        if not animal_metrics: continue
        
        df_met = pd.DataFrame(animal_metrics)
        
        # Run Stats and Plot for BOTH metrics
        for metric, label in [("Raw_Noise", "Raw Slope Noise"), ("Corrected_Noise", "Vigor-Corrected Noise")]:
            genos = sorted(df_met["Genotype"].unique())
            p_vals = {}
            import itertools
            for g1, g2 in itertools.combinations(genos, 2):
                v1 = df_met[df_met["Genotype"] == g1][metric]
                v2 = df_met[df_met["Genotype"] == g2][metric]
                if len(v1) > 1 and len(v2) > 1:
                    try:
                        stat, p = stats.mannwhitneyu(v1, v2)
                        p_vals[f"{g1}_vs_{g2}"] = p
                    except: pass
            
            # Serialize
            p_str = "; ".join([f"{k}: {v:.1e}" for k, v in p_vals.items()])
            stats_rows.append({
                "Syllable": syll,
                "Metric": label,
                "Groups": str(genos),
                "N_Animals": len(df_met),
                "Pairwise_p": p_str,
                "Means": str(df_met.groupby("Genotype")[metric].mean().to_dict())
            })
            
            # Plot
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.stripplot(data=df_met, x="Genotype", y=metric, hue="Genotype", ax=ax, jitter=True, size=8, palette="Set1")
            sns.boxplot(data=df_met, x="Genotype", y=metric, hue="Genotype", ax=ax, showfliers=False, palette="Set1")
            
            title_str = f"Syllable {syll}: {label}"
            if len(p_vals) > 0:
                 min_p = min(p_vals.values())
                 title_str += f"\n(Min Pairwise p={min_p:.1e})"
            
            ax.set_title(title_str)
            ax.set_ylabel("Variance (Std)")
            plt.tight_layout()
            
            # Save
            m_tag = "raw" if "Raw" in label else "corrected"
            s_safe = str(syll).replace("/", "_")
            plot_path = os.path.join(output_dir, "plots", f"ppe_{m_tag}_{s_safe}.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            plt.close()

    # Save Stats
    if stats_rows:
        stats_dir = os.path.join(output_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        pd.DataFrame(stats_rows).to_csv(os.path.join(stats_dir, "ppe_stats.csv"), index=False)
        logger.info("Saved stats and plots.")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True, choices=["ifdn", "amph"])
    args = parser.parse_args()
    
    # Resolving Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    # Load Sessions
    # We use the loader from panel_1z to get the standardized sessions list
    # Note: loader.load_sessions expects a config dict usually, but here we can just pass project
    
    # Mock Config for Loader
    config = {
        "project": args.project, 
        "data_dir": base_dir, # Assumption
        "output_dir": os.path.join(os.path.dirname(__file__), "output")
    }
    
    # We need to manually construct the pickle path as loader might rely on relative paths
    if args.project == "ifdn":
        pkl_path = os.path.join(base_dir, "data", "jonathan_ifdn_sessions_w_sylls.pkl")
    else:
        pkl_path = os.path.join(base_dir, "data", "jonathan_amph_sessions_v2_w_sylls.pkl")
        
    if not os.path.exists(pkl_path):
        logger.error(f"Pickle not found: {pkl_path}")
        return

    logger.info(f"Loading {args.project} sessions from {pkl_path}...")
    sessions = loader.load_sessions(pkl_path=pkl_path)
    
    # Enrich (add group meta)
    from analysis.replications.panel_1z.lib import processing
    sessions = processing.enrich_sessions(sessions)
    
    # Filter for AMPH Project (Post-Injection)
    if args.project == "amph":
        logger.info("Applying AMPH Post-Injection Filter (5 mins)...")
        sessions = processing.filter_post_injection(sessions, 5)

    os.makedirs(config["output_dir"], exist_ok=True)
    run_analysis(sessions, config["output_dir"])

if __name__ == "__main__":
    main()
