import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import logging

from analysis.replications.panel_1z.lib import processing, utils

logger = logging.getLogger("Panel1z.PeakTiming")

# --- Configuration ---
WINDOW_SEC = 1.5
SHUFFLE_JITTER_SEC = 2.0
N_SHUFFLES = 20
MIN_TRIALS = 100
TARGET_COLOR = "G"
IGNORE_GENOTYPES = ["UNK", "nan", None]
IGNORE_REGIONS = ["UNK", "nan", None]

def run(sessions, config):
    """
    Run Peak Timing Analysis (latency and variance).
    """
    logger.info("Gathering data for Peak Timing...")
    
    # Structure: buckets[geno][region] = {syll: [(n_trials, mean_act, mean_shuff), ...]}
    # We store list of tuples because a syllable might appear in multiple sessions
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # 1. Aggregation
    for s in sessions:
        meta = s.metadata
        geno = meta.get("group", meta.get("genotype", "UNK"))
        if str(geno) in IGNORE_GENOTYPES: continue
        
        df = s.all_data_df
        if df.empty: continue
        
        fs = meta.get('fs', 30.0)
        
        # Get Onsets
        if "onset" in df.columns:
            onsets = df["onset"].fillna(0).astype(bool).to_numpy()
        else:
            onsets = (df["syllable"] != df["syllable"].shift(1)).fillna(False).to_numpy()
        
        onset_idxs = np.flatnonzero(onsets)
        if len(onset_idxs) == 0: continue
        
        syllables = df["syllable"].to_numpy()
        
        # Iterate Signals
        sig_meta = meta.get("signal_meta", {})
        for sig_key, info in sig_meta.items():
            # Check Color
            color = str(info.get("orig_letter", "UNK")).upper()
            if color == "UNK" and "fiber_num" in info:
                color = "G" if int(info["fiber_num"]) % 2 == 0 else "R"
            if color != TARGET_COLOR: continue
            
            # Check Region
            region = info.get("brain_reg", "UNK")
            if region in IGNORE_REGIONS: continue
            
            # Get Data
            col = None
            if f"{sig_key}_dff" in df.columns: col = f"{sig_key}_dff"
            elif f"{sig_key}_zscore_datta" in df.columns: col = f"{sig_key}_zscore_datta"
            
            if not col: continue
            
            sig_data = df[col].to_numpy()
            
            # Windows
            win_frames = int(WINDOW_SEC * fs)
            jit_frames = int(SHUFFLE_JITTER_SEC * fs)
            max_idx = len(sig_data)
            
            # Collect Trials per Syllable
            syll_map = defaultdict(list)
            for idx in onset_idxs:
                syll_map[syllables[idx]].append(idx)
                
            for syll, indices in syll_map.items():
                if len(indices) < 5: continue # optimization
                
                valid_starts = []
                valid_ends = []
                
                indices = np.array(indices)
                starts = indices - win_frames
                ends = indices + win_frames
                
                # Check bounds
                valid_mask = (starts >= 0) & (ends <= max_idx)
                if not np.any(valid_mask): continue
                
                v_starts = starts[valid_mask]
                v_ends = ends[valid_mask]
                n_trials = len(v_starts)
                
                # Get Actual Trace Mean
                # list comprehension is safer for jagged arrays if fs varies, but here fs is constant per session
                # stacking is faster
                try:
                    stack = np.array([sig_data[s:e] for s, e in zip(v_starts, v_ends)])
                    if np.isnan(stack).all(): continue
                    mu_actual = np.nanmean(stack, axis=0)
                    
                    # Get Shuffle Trace Mean
                    # We shift the starts by random amount
                    shifts = np.random.randint(-jit_frames, jit_frames, size=(N_SHUFFLES, n_trials))
                    shuff_means = []
                    
                    for i in range(N_SHUFFLES):
                        # Clip to bounds
                        s_starts = np.clip(v_starts + shifts[i], 0, max_idx - 2*win_frames)
                        s_ends = s_starts + 2*win_frames
                        
                        s_stack = np.array([sig_data[s:e] for s, e in zip(s_starts, s_ends)])
                        shuff_means.append(np.nanmean(s_stack, axis=0))
                        
                    mu_shuffle = np.mean(shuff_means, axis=0)
                    
                    buckets[geno][region][syll].append((n_trials, mu_actual, mu_shuffle))
                    
                except Exception as e:
                    continue

    logger.info("   Aggregation complete.")

    # 2. Run Statistics & Plot
    run_peak_latency_test(buckets, config)
    run_variance_test(buckets, config)

def run_peak_latency_test(buckets, config):
    logger.info("   Running Peak Latency (KS-Test)...")
    
    unique_genos = sorted(buckets.keys())
    unique_regions = sorted({r for g in buckets for r in buckets[g]})
    
    if not unique_regions:
        logger.warning("No regions found for Peak Latency.")
        return

    n_rows = len(unique_regions)
    n_cols = len(unique_genos)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), dpi=150, sharey=False)
    if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
    elif n_rows == 1: axes = axes[np.newaxis, :]
    elif n_cols == 1: axes = axes[:, np.newaxis]
    
    stats_rows = []
    
    for r_idx, region in enumerate(unique_regions):
        for c_idx, geno in enumerate(unique_genos):
            ax = axes[r_idx, c_idx]
            data = buckets[geno].get(region, {})
            
            pool_actual = []
            pool_shuffle = []
            
            # Weighted pooling
            for syll, session_list in data.items():
                total = sum(i[0] for i in session_list)
                if total < MIN_TRIALS: continue
                
                w_act = np.zeros_like(session_list[0][1])
                w_shuff = np.zeros_like(session_list[0][2])
                
                for n, act, shuff in session_list:
                    # Fix length mismatch (rare fs drift)
                    if len(act) != len(w_act):
                         act = np.interp(np.linspace(0,1,len(w_act)), np.linspace(0,1,len(act)), act)
                    if len(shuff) != len(w_shuff):
                         shuff = np.interp(np.linspace(0,1,len(w_shuff)), np.linspace(0,1,len(shuff)), shuff)
                         
                    w_act += act * n
                    w_shuff += shuff * n
                    
                pool_actual.append(w_act / total)
                pool_shuffle.append(w_shuff / total)
            
            if not pool_actual:
                ax.text(0.5, 0.5, "No Data", ha='center')
                continue
                
            arr_actual = np.array(pool_actual)
            arr_shuffle = np.array(pool_shuffle)
            
            # Bootstrap Peaks
            n_sylls, n_time = arr_actual.shape
            time_vec = np.linspace(-WINDOW_SEC, WINDOW_SEC, n_time)
            
            peaks_actual = []
            peaks_null = []
            
            for _ in range(1000): # N_BOOTSTRAPS
                idx = np.random.choice(n_sylls, n_sylls, replace=True)
                
                # Actual
                m_act = np.mean(arr_actual[idx], axis=0)
                peaks_actual.append(time_vec[np.argmax(np.abs(m_act))])
                
                # Null
                m_null = np.mean(arr_shuffle[idx], axis=0)
                peaks_null.append(time_vec[np.argmax(np.abs(m_null))])
                
            # Stats
            ks_stat, ks_p = stats.ks_2samp(peaks_actual, peaks_null)
            
            # CI
            ci_low = np.percentile(peaks_actual, 2.5)
            ci_high = np.percentile(peaks_actual, 97.5)
            med = np.median(peaks_actual)
            
            # Direction
            grand_mean = np.mean(arr_actual, axis=0)
            direction = "Peak" if grand_mean[np.argmax(np.abs(grand_mean))] > 0 else "Dip"
            
            stats_rows.append({
                "Region": region,
                "Genotype": geno,
                "Direction": direction,
                "Latency_Median": med,
                "CI_Low": ci_low,
                "CI_High": ci_high,
                "KS_p": ks_p
            })
            
            # Plot
            sns.kdeplot(peaks_null, ax=ax, color="gray", fill=True, alpha=0.3)
            c = "orange" if direction == "Peak" else "skyblue"
            sns.kdeplot(peaks_actual, ax=ax, color=c, fill=True, alpha=0.6)
            
            # Title for EVERY subplot to avoid confusion
            ax.set_title(f"{region} | {geno}\n{direction} @ {med*1000:.0f}ms (p={ks_p:.1e})", fontsize=9)
            ax.axvline(ci_low, color=c, linestyle=":")
            ax.axvline(ci_high, color=c, linestyle=":")
            ax.set_xlim(-1, 1) # Zoom in
            # Only hide Y label, keep X label for clarity or use layout to fix
            if c_idx == 0: ax.set_ylabel("Density")
            else: ax.set_ylabel("")
            
            ax.set_xlabel("Time (s)")

    sns.despine()
    plt.tight_layout()
    utils.save_plot(fig, "peak_latency_distribution", config)
    utils.save_stats(pd.DataFrame(stats_rows), "peak_latency_stats", config)

def run_variance_test(buckets, config):
    logger.info("   Running Variance Analysis (Levene Test)...")
    
    unique_genos = sorted(buckets.keys())
    unique_regions = sorted({r for g in buckets for r in buckets[g]})
    
    fig, axes = plt.subplots(len(unique_regions), len(unique_genos), 
                             figsize=(4*len(unique_genos), 3.5*len(unique_regions)), dpi=150)
    
    # Handle single plot case
    if len(unique_regions) == 1 and len(unique_genos) == 1: axes = np.array([[axes]])
    elif len(unique_regions) == 1: axes = axes[np.newaxis, :]
    elif len(unique_genos) == 1: axes = axes[:, np.newaxis]
    
    stats_rows = []
    
    for r, region in enumerate(unique_regions):
        for c, geno in enumerate(unique_genos):
            ax = axes[r, c]
            data = buckets[geno].get(region, {})
            # ... (data processing omitted for brevity, logic remains same) ...
            
            pks_act = []
            pks_shuff = []
            
            # Get One Time Vec
            time_vec = np.linspace(-WINDOW_SEC, WINDOW_SEC, 200) # dummy
            
            for syll, session_list in data.items():
                total = sum(i[0] for i in session_list)
                if total < MIN_TRIALS: continue
                
                # Reconstruct weighted means
                w_act = np.zeros_like(session_list[0][1])
                w_shuff = np.zeros_like(session_list[0][2])
                
                for n, act, shuff in session_list:
                    # Fix length
                    if len(act) != len(w_act):
                         act = np.interp(np.linspace(0,1,len(w_act)), np.linspace(0,1,len(act)), act)
                         shuff = np.interp(np.linspace(0,1,len(w_shuff)), np.linspace(0,1,len(shuff)), shuff)
                    w_act += act * n
                    w_shuff += shuff * n
                    
                mean_act = w_act / total
                mean_shuff = w_shuff / total
                
                # Current Time Vec (based on actual length)
                curr_t = np.linspace(-WINDOW_SEC, WINDOW_SEC, len(mean_act))
                
                pks_act.append(curr_t[np.argmax(np.abs(mean_act))])
                pks_shuff.append(curr_t[np.argmax(np.abs(mean_shuff))])
                
            if not pks_act:
                ax.text(0.5, 0.5, "No Data", ha='center'); continue
                
            # Stats
            stat, p_levene = stats.levene(pks_act, pks_shuff)
            
            # Window (95% Capture)
            low = np.percentile(pks_act, 2.5)
            high = np.percentile(pks_act, 97.5)
            
            is_tighter = np.std(pks_act) < np.std(pks_shuff)
            
            stats_rows.append({
                "Region": region,
                "Genotype": geno,
                "Levene_p": p_levene,
                "Std_Actual": np.std(pks_act),
                "Std_Shuffle": np.std(pks_shuff),
                "Is_Tighter": is_tighter,
                "Window_Start": low,
                "Window_End": high
            })
            
            # Plot
            sns.kdeplot(pks_shuff, ax=ax, color="gray", fill=True, alpha=0.3)
            sns.kdeplot(pks_act, ax=ax, color="purple", fill=True, alpha=0.5)
            
            ax.axvline(low, color="purple", linestyle=":")
            ax.axvline(high, color="purple", linestyle=":")
            ax.set_title(f"{region} | {geno}\nPrecision: {low*1000:.0f}..{high*1000:.0f}ms\n(p={p_levene:.1e})", fontsize=9)
            ax.set_xlim(-1, 1)
            ax.set_xlabel("Time (s)")
            if c == 0: ax.set_ylabel("Density")
            else: ax.set_ylabel("")
            
    sns.despine()
    plt.tight_layout()
    utils.save_plot(fig, "temporal_precision_window", config)
    utils.save_stats(pd.DataFrame(stats_rows), "temporal_precision_stats", config)
