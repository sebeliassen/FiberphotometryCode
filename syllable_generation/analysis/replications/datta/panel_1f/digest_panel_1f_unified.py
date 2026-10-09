
"""
Unified Digest Panel 1f: Syllable vs Movement Initiation Traces.
Supports both IFDN and AMPH projects.
"""
import pandas as pd
import numpy as np
import argparse
from scipy.signal import bessel, filtfilt
from pathlib import Path
import sys
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from collections import defaultdict

# Pathing
this_file = Path(__file__).resolve()
repo_root = this_file.parents[5] 
datta_dir = this_file.parents[1]

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(datta_dir) not in sys.path:
    sys.path.insert(0, str(datta_dir))

from lib import config

def process_session_digest(parquet_path, n_shuffles, mode):
    np.random.seed(42)
    try:
        df = pd.read_parquet(parquet_path)
        if df.empty: return None
        
        genotype = df["genotype"].iloc[0]
        drug = df["drug"].iloc[0] if "drug" in df.columns else "baseline"
        animal_id = df["animal_id"].iloc[0]
        
        results = []
        
        for sig in df["signal_source"].unique():
            sub = df[df["signal_source"] == sig]
            hemi = sub["hemisphere"].iloc[0]
            region = sub["brain_reg"].iloc[0]
            orig_letter = str(sub["orig_letter"].iloc[0]).upper()
            
            # Filter for Green Only
            if orig_letter != 'G':
                continue
                
            # Derive Method
            method = "Datta Z-Score" if "zscore_datta" in str(sig) else "Standard Z-Score"
            
            for init_type in sub["init_type"].unique():
                t_sub = sub[sub["init_type"] == init_type]
                
                # Pivot
                pivot = t_sub.pivot(index="snippet", columns="x", values="value_renorm")
                vals = pivot.values
                times = pivot.columns.to_numpy()
                n_snip, n_time = vals.shape
                
                # Default: No filter
                vals_proc = vals
                
                if mode == "hifi":
                    # --- 1Hz Bessel High-Pass Filter (Snippet-wise) ---
                    fs_effective = 80.0 
                    b, a = bessel(2, 1.0, fs=fs_effective, btype='high')
                    
                    # Apply filter to each snippet (row-wise)
                    vals_filled = pd.DataFrame(vals).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).fillna(0).values
                    vals_proc = filtfilt(b, a, vals_filled, axis=1)
                
                # Stats (Snippet Means - Absolute Value in 1s window)
                # Note: We use the processed values if hifi, otherwise standard renorm
                stat_win_mask = (times >= -1.0) & (times <= 1.0)
                snippet_means = np.nanmean(np.abs(vals_proc[:, stat_win_mask]), axis=1)
                
                # --- Shuffles ---
                shuffle_sums = np.zeros((n_shuffles, n_time))
                for k in range(n_shuffles):
                    shifts = np.random.randint(0, n_time, n_snip)
                    cols = np.arange(n_time)
                    idx0 = np.arange(n_snip)[:, None]
                    idx1 = (cols[None, :] - shifts[:, None]) % n_time
                    shuffled = vals_proc[idx0, idx1] 
                    shuffle_sums[k, :] = np.nansum(shuffled, axis=0)
                
                # --- Observed ---
                obs_sum = np.nansum(vals_proc, axis=0)
                obs_cnt = np.sum(~np.isnan(vals_proc), axis=0)
                obs_sq = np.nansum(vals_proc**2, axis=0)

                results.append({
                    "key": (genotype, drug, region, hemi, method, init_type),
                    "animal_id": animal_id,
                    "times": times,
                    "obs_sum": obs_sum,
                    "obs_cnt": obs_cnt,
                    "obs_sq": obs_sq,
                    "shuffle_sums": shuffle_sums,
                    "snippet_means": snippet_means,
                    "n_snippets": n_snip
                })
        return results
        
    except Exception as e:
        print(f"Error digesting {parquet_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Unified Digest Panel 1f")
    parser.add_argument("--project", type=str, required=True, choices=["ifdn", "amph"])
    parser.add_argument("--mode", type=str, default="midfi", choices=["std", "midfi", "hifi"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--shuffle-limit", type=int, default=100)
    args = parser.parse_args()
    
    print(f"--- Digesting Panel 1f ({args.project.upper()}) | Mode: {args.mode} | Shuffles: {args.shuffle_limit} ---")
    
    if args.output_dir:
        data_dir = Path(args.output_dir)
    else:
        d_key = "panel_1f_ifdn" if args.project == "ifdn" else "panel_1f_amph"
        data_dir = config.OUTPUT_ROOT / d_key
        
    snippets_dir = data_dir / "datta_1f_SNIPPETS.parquet"
    if not snippets_dir.exists():
        print(f"Missing {snippets_dir}")
        return
        
    files = list(snippets_dir.glob("*.parquet"))
    print(f"Found {len(files)} session files.")
    
    # Process
    results = Parallel(n_jobs=8, verbose=5)(
        delayed(process_session_digest)(f, args.shuffle_limit, args.mode) for f in files
    )
    all_res = [item for sublist in results if sublist for item in sublist]
    
    # Reduce
    aggregated = defaultdict(lambda: {
        "obs_sum": 0, "obs_cnt": 0, "obs_sq": 0,
        "shuffle_sums": 0, "times": None
    })
    animal_data = defaultdict(lambda: defaultdict(list))
    
    # Track which hemispheres contribute to each (geno, drug, reg, method, init_type)
    # Key: (geno, drug, reg, method, init_type) -> Set of hemispheres
    hemi_tracker = defaultdict(set)
    
    for r in all_res:
        k = r["key"] # (geno, drug, reg, hemi, method, init_type)
        
        # Track Hemi for this group
        group_base_key = (k[0], k[1], k[2], k[4], k[5])
        hemi_tracker[group_base_key].add(k[3])
        
        target = aggregated[k]
        if target["times"] is None: target["times"] = r["times"]
        
        target["obs_sum"] += r["obs_sum"]
        target["obs_cnt"] += r["obs_cnt"]
        target["obs_sq"] += r["obs_sq"]
        target["shuffle_sums"] += r["shuffle_sums"]
        animal_data[k][r["animal_id"]].extend(r["snippet_means"])
        
        # Hemispheric Average
        avg_k = (k[0], k[1], k[2], "Avg", k[4], k[5])
        target_avg = aggregated[avg_k]
        if target_avg["times"] is None: target_avg["times"] = r["times"]
        target_avg["obs_sum"] += r["obs_sum"]
        target_avg["obs_cnt"] += r["obs_cnt"]
        target_avg["obs_sq"] += r["obs_sq"]
        target_avg["shuffle_sums"] += r["shuffle_sums"]
        animal_data[avg_k][r["animal_id"]].extend(r["snippet_means"])

    # Compute
    obs_rows, shuff_rows, stats_rows = [], [], []
    for k, v in aggregated.items():
        geno, drug, reg, hemi, method, init_type = k
        
        # SUPPRESSION LOGIC:
        # If this is an "Avg" key, check if we actually have > 1 hemisphere
        if hemi == "Avg":
            group_base_key = (geno, drug, reg, method, init_type)
            if len(hemi_tracker[group_base_key]) < 2:
                # Skip this Avg because it's redundant (only 1 or 0 hemis)
                continue
        
        times, N = v["times"], v["obs_cnt"]
        if np.all(N == 0): continue
        
        mean = np.divide(v["obs_sum"], N, out=np.zeros_like(v["obs_sum"]), where=N>0)
        with np.errstate(divide='ignore', invalid='ignore'):
             var = np.nan_to_num((v["obs_sq"] - (v["obs_sum"]**2)/N) / (N-1), 0)
        sem = np.sqrt(var) / np.sqrt(np.maximum(N, 1))
        
        shuff_means = np.divide(v["shuffle_sums"], N, out=np.zeros_like(v["shuffle_sums"]), where=N>0)
        mu_shuff = np.mean(shuff_means, axis=0)
        std_shuff = np.std(shuff_means, axis=0)
        std_shuff[std_shuff == 0] = 1e-6
        
        z_shuff = (mean - mu_shuff) / std_shuff
        global_shuff_norm = (shuff_means - mu_shuff) / std_shuff
        grand_mean_norm = np.mean(global_shuff_norm, axis=0)
        lo, hi = np.percentile(global_shuff_norm, [2.5, 97.5], axis=0)
        
        for t, m, s, n_val, z, ss in zip(times, mean, sem, N, z_shuff, std_shuff):
            obs_rows.append({
                "genotype": geno, "drug": drug, "brain_reg": reg, "hemisphere": hemi,
                "method": method, "init_type": init_type, "x": t, "mean": m, "sem": s, "n": n_val,
                "z_shuffle": z, "std_shuff": ss
            })
        for t, m, l, h in zip(times, grand_mean_norm, lo, hi):
            shuff_rows.append({
                "genotype": geno, "drug": drug, "brain_reg": reg, "hemisphere": hemi,
                "method": method, "init_type": init_type, "x": t, "mean": m, "lo": l, "hi": h
            })

    # Mouse-level stats
    meta_keys = set((k[0], k[1], k[2], k[3], k[4]) for k in animal_data.keys())
    for mk in meta_keys:
        mov_k, syl_k = mk + ("movement",), mk + ("syllables_offline",)
        if mov_k in animal_data and syl_k in animal_data:
            mov_means = [np.nanmean(snips) for snips in animal_data[mov_k].values() if snips]
            syl_means = [np.nanmean(snips) for snips in animal_data[syl_k].values() if snips]
            if len(mov_means) >= 3 and len(syl_means) >= 3:
                try:
                    res = mannwhitneyu(mov_means, syl_means)
                    stats_rows.append({
                        "genotype": mk[0], "drug": mk[1], "brain_reg": mk[2], "hemisphere": mk[3],
                        "method": mk[4], "stat": res.statistic, "pvalue": res.pvalue, "n_mov": len(mov_means), "n_syl": len(syl_means)
                    })
                except Exception: pass

    pd.DataFrame(obs_rows).to_parquet(data_dir / "datta_1f_OBSERVED.parquet")
    pd.DataFrame(shuff_rows).to_parquet(data_dir / "datta_1f_SHUFFLE_BANDS.parquet")
    pd.DataFrame(stats_rows).to_csv(data_dir / "datta_1f_STATS.csv", index=False)
    print(f"Done. Saved to {data_dir}")

if __name__ == "__main__":
    main()
