import pickle
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from fpcode.common.paths import get_repo_root

# Add project root to path
project_root = get_repo_root()

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from monoamine_interactions.analysis.phase_1_causality import varlingam
from fpcode.photometry.session_signals import prepare_region_traces, legacy_nearest_window

# CONFIGURATION
EVENT_KEY = "juvenile_first_sniff_min"
TARGET_GENOTYPE = "WT"
TARGET_SOURCE = "lNAc_5HT"
TARGET_TARGET = "rNAc_DA"

PRE_SEC = 20
POST_SEC = 40
SLIDING_WINDOW_SEC = 10
STRIDE_SEC = 0.5
N_JOBS = -1
LAG = 5
FPS = 20

frames_calc_window = int(SLIDING_WINDOW_SEC * FPS)
frames_stride = int(STRIDE_SEC * FPS)

pkl_path = project_root / "monoamine_interactions" / "data" / "sessions_jonathan_juvenile_intruder_dirty.pkl"
cache_path = project_root / "monoamine_interactions" / "outputs" / "06_tabular_extraction" / "df_results_cache.pkl"

def extract_data():
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        sessions = pickle.load(f)

    # Deduplicate sessions
    unique_sessions = []
    seen_csvs = set()
    for s in sessions:
        csv_path = s.metadata.get('photometry_csv_path')
        if csv_path and csv_path not in seen_csvs:
            seen_csvs.add(csv_path)
            unique_sessions.append(s)
    sessions = unique_sessions

    def run_windowed_metrics(chunk, regs):
        clean_chunk = chunk.dropna()
        if len(clean_chunk) < frames_calc_window * 0.8:
            return None
        metrics = {'corr': {}, 'causal': {}}
        corr_matrix = clean_chunk.corr().values
        try:
            res = varlingam.run_varlingam(clean_chunk, lags=LAG)
            adj = res["adjacency_matrices"][0]
            success_var = True
        except:
            adj = None
            success_var = False

        for i, r1 in enumerate(regs):
            for j, r2 in enumerate(regs):
                if i == j: continue
                metrics['corr'][(r1, r2)] = corr_matrix[i, j]
                if success_var:
                    metrics['causal'][(r2, r1)] = adj[i, j]
                else:
                    metrics['causal'][(r2, r1)] = np.nan
        return metrics

    all_results = []
    expected_length = None
    time_vec = None

    for s in tqdm(sessions, desc=f"Extracting {EVENT_KEY}"):
        tm = s.metadata.get("timing_meta", {})
        t_event_min = tm.get(EVENT_KEY)
        if pd.isna(t_event_min): continue
        if '470' not in s.photometry: continue

        df_470 = s.photometry['470']
        ts_array = df_470['SystemTimestamp'].values
        t_start_sys = ts_array[0]
        target_ts_event = t_start_sys + (t_event_min * 60)

        start_idx, end_idx = legacy_nearest_window(
            ts_array, target_ts_event - PRE_SEC, target_ts_event + POST_SEC
        )
        if end_idx <= start_idx: continue

        curr_len = end_idx - start_idx
        if expected_length is None: expected_length = curr_len
        if curr_len != expected_length:
            end_idx = start_idx + expected_length
            if end_idx > len(ts_array): continue

        genotype = s.metadata.get('genotype', 'Unknown')
        prepared = prepare_region_traces(s, normalization="none", errors="skip")
        dff_dict = prepared.traces
        if prepared.failures:
            print(f"Signal preparation skipped regions: {prepared.failures}")

        valid_regs = list(dff_dict.keys())
        if len(valid_regs) < 2: continue

        chunks = []
        peth_times = []
        for i in range(start_idx + frames_calc_window, end_idx, frames_stride):
            chunk_center_idx = i - (frames_calc_window // 2)
            rel_time_sec = (ts_array[chunk_center_idx] - target_ts_event) / 60.0 * 60.0
            chunk = pd.DataFrame({r: dff_dict[r][i-frames_calc_window:i] for r in valid_regs})
            chunks.append(chunk)
            peth_times.append(rel_time_sec)

        if time_vec is None:
            time_vec = np.array(peth_times)

        chunk_results = joblib.Parallel(n_jobs=N_JOBS)(
            joblib.delayed(run_windowed_metrics)(c, valid_regs) for c in chunks
        )

        for pair in [("lNAc_5HT", "rNAc_DA")]:
            if pair[0] not in valid_regs or pair[1] not in valid_regs: continue
            corr_series = []
            causal_series = []
            for res in chunk_results:
                if res is not None:
                    corr_val = res['corr'].get((pair[0], pair[1]), np.nan)
                    if pd.isna(corr_val): corr_val = res['corr'].get((pair[1], pair[0]), np.nan)
                    caus_val = res['causal'].get(pair, np.nan)
                    corr_series.append(corr_val)
                    causal_series.append(caus_val)
                else:
                    corr_series.append(np.nan)
                    causal_series.append(np.nan)

            interp_corr = np.interp(time_vec, peth_times, corr_series)
            interp_causal = np.interp(time_vec, peth_times, causal_series)

            all_results.append({
                'event': EVENT_KEY,
                'genotype': genotype,
                'source': pair[0],
                'target': pair[1],
                'corr_array': interp_corr,
                'causal_array': interp_causal
            })
    df_results = pd.DataFrame(all_results)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_pickle(cache_path)
    return df_results

# 1. Load or Calculate Data
if cache_path.exists():
    print(f"Loading cached results from {cache_path}")
    df_results = pd.read_pickle(cache_path)
else:
    df_results = extract_data()

# 2. Extract the specific Target Group arrays
df_target = df_results[(df_results['genotype'] == TARGET_GENOTYPE) &
                       (df_results['source'] == TARGET_SOURCE) &
                       (df_results['target'] == TARGET_TARGET)]

caus_arrays = np.vstack(df_target['causal_array'].values)
mean_caus = np.nanmean(caus_arrays, axis=0)
std_caus = np.nanstd(caus_arrays, axis=0) / np.sqrt(len(caus_arrays))
clean_caus_for_check = np.nan_to_num(mean_caus, nan=0.0)

# 3. Binary Search for Minimum Threshold
print("\n--- SEARCHING FOR MINIMUM THRESHOLD TO KILL PLOT ---")
print("Target Plot: WT | lNAc_5HT -> rNAc_DA")

# Let's test thresholds from 0.01 to 0.5
sigma = 1.5
smoothed_caus = gaussian_filter1d(clean_caus_for_check, sigma=sigma)
max_smoothed = np.nanmax(np.abs(smoothed_caus))
print(f"Max Smoothed Causal Weight (Sigma={sigma}): {max_smoothed:.4f}")

# What is the SNR at the peak?
peak_idx = np.nanargmax(np.abs(mean_caus))
peak_val = np.abs(mean_caus[peak_idx])
peak_err = std_caus[peak_idx]
print(f"Peak Value: {peak_val:.4f}, Peak Error: {peak_err:.4f}, SNR: {(peak_val/peak_err):.2f}")

print("\nTo kill this plot, your parameters must be:")
print(f"1. `threshold` > {max_smoothed:.4f}")
print("OR")
print(f"2. `shadow_rule_multiplier` > {(peak_val/peak_err):.2f}")

# Let's see what threshold is needed for sigma=2.0 and sigma=3.0
for s in [2.0, 3.0, 4.0]:
    sc = gaussian_filter1d(clean_caus_for_check, sigma=s)
    print(f"If sigma={s}, required threshold > {np.nanmax(np.abs(sc)):.4f}")
