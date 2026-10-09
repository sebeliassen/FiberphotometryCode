import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

# Add project root to path
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path("../../").resolve()
    
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from monoamine_interactions.notebooks.utils.photometry import motion_correct_biexponential
from monoamine_interactions.analysis.phase_1_causality import varlingam

# %%
# CONFIGURATION
EVENT_KEY = "all" 
PRE_SEC = 20
POST_SEC = 40

SLIDING_WINDOW_SEC = 10 
STRIDE_SEC = 0.5       
N_JOBS = -1            
LAG = 5                
FPS = 20

frames_calc_window = int(SLIDING_WINDOW_SEC * FPS)
frames_stride = int(STRIDE_SEC * FPS)

SELECTED_PAIRS = [
    ("lNAc_5HT", "rNAc_DA"),
    ("rNAc_DA", "lNAc_5HT"),
    ("lNAc_DA", "rNAc_5HT"),
    ("rNAc_5HT", "lNAc_DA")
]

pkl_path = project_root / "monoamine_interactions" / "data" / "sessions_jonathan_juvenile_intruder_dirty.pkl"
out_dir = project_root / "monoamine_interactions" / "outputs" / "06_tabular_extraction"
out_dir.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load Data
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

all_events = set()
for s in sessions:
    all_events.update(s.metadata.get('timing_meta', {}).keys())
all_events = sorted(list(all_events))
events_to_process = all_events if EVENT_KEY == "all" else [EVENT_KEY]

# %%
# 2. Worker Functions
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

# %%
# 3. Extract Sliding Windows
all_results = []
metadata_per_event = {} 

for event in events_to_process:
    print(f"\nExtracting data for event: {event}")
    expected_length = None
    time_vec = None
    
    for s in tqdm(sessions, desc=f"Extracting {event}"):
        tm = s.metadata.get("timing_meta", {})
        t_event_min = tm.get(event)
        if pd.isna(t_event_min): continue
        if '470' not in s.photometry: continue
            
        df_470 = s.photometry['470']
        df_415 = s.photometry['415']
        ts_array = df_470['SystemTimestamp'].values
        t_start_sys = ts_array[0]
        target_ts_event = t_start_sys + (t_event_min * 60)
        
        start_idx = np.abs(ts_array - (target_ts_event - PRE_SEC)).argmin()
        end_idx = np.abs(ts_array - (target_ts_event + POST_SEC)).argmin()
        if end_idx <= start_idx: continue
            
        curr_len = end_idx - start_idx
        if expected_length is None: expected_length = curr_len
        if curr_len != expected_length:
            end_idx = start_idx + expected_length
            if end_idx > len(ts_array): continue 

        genotype = s.metadata.get('genotype', 'Unknown')
        sm = s.metadata.get("signal_meta", {})
        region_map = {k: v.get("region") for k, v in sm.items() if v.get("region")}
        
        dff_dict = {}
        for r in region_map.values():
            try:
                col = next((k for k, v in region_map.items() if v == r))
                full_dff = motion_correct_biexponential(df_470[col].values, df_415[col].values)
                dff_dict[r] = full_dff
            except: pass
            
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
            metadata_per_event[event] = {'time_vec': time_vec}

        chunk_results = joblib.Parallel(n_jobs=N_JOBS)(
            joblib.delayed(run_windowed_metrics)(c, valid_regs) for c in chunks
        )
        
        for pair in SELECTED_PAIRS:
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
                'event': event,
                'genotype': genotype,
                'source': pair[0],
                'target': pair[1],
                'corr_array': interp_corr,
                'causal_array': interp_causal
            })

df_results = pd.DataFrame(all_results)

# %%
# 4. Feature Extraction Function
def extract_landmarks(time_vec, trace, metric_name, event, name, threshold=0.01):
    # Skip noisy/flat causality traces
    if metric_name == "Causality" and np.nanmax(np.abs(trace)) < threshold:
        return None

    # Masks for different time periods
    pre_mask = (time_vec >= -20) & (time_vec < 0)
    post_mask = (time_vec > 0) & (time_vec <= 40)
    
    # 1. Baseline Stats
    baseline_mean = np.nanmean(trace[pre_mask])
    baseline_var = np.nanvar(trace[pre_mask])
    
    # 2. Kinetics at t=0
    idx_0 = np.abs(time_vec).argmin()
    t0_val = trace[idx_0]
    
    # Initial slope (derivative over the first 2 seconds after t=0)
    idx_2s = np.abs(time_vec - 2.0).argmin()
    t0_slope = (trace[idx_2s] - trace[idx_0]) / (time_vec[idx_2s] - time_vec[idx_0])
    
    # 3. Peaks and Valleys
    post_trace = trace[post_mask]
    post_times = time_vec[post_mask]
    
    if len(post_trace) > 0:
        idx_max = np.nanargmax(post_trace)
        max_y = post_trace[idx_max]
        max_t = post_times[idx_max]
        
        idx_min = np.nanargmin(post_trace)
        min_y = post_trace[idx_min]
        min_t = post_times[idx_min]
        
        first_ext = "Peak" if max_t < min_t else "Valley"
        
        # Area Under Curve
        auc = np.trapz(post_trace, post_times)
    else:
        max_y = max_t = min_y = min_t = auc = np.nan
        first_ext = "Unknown"
        
    # 4. Snapshots
    idx_10 = np.abs(time_vec - 10.0).argmin()
    idx_20 = np.abs(time_vec - 20.0).argmin()
    idx_40 = np.abs(time_vec - 40.0).argmin()
    
    val_10 = trace[idx_10]
    val_20 = trace[idx_20]
    val_40 = trace[idx_40]
    
    return {
        "Event": event,
        "Group": name,
        "Metric": metric_name,
        "Baseline_Mean": baseline_mean,
        "Baseline_Var": baseline_var,
        "t0_Value": t0_val,
        "t0_Slope": t0_slope,
        "Max_t": max_t,
        "Max_y": max_y,
        "Min_t": min_t,
        "Min_y": min_y,
        "First_Extremum": first_ext,
        "AUC": auc,
        "t10_Val": val_10,
        "t20_Val": val_20,
        "t40_Val": val_40
    }

# %%
# 5. Extract Tabular Data
landmark_data = []

for event in events_to_process:
    df_ev = df_results[df_results['event'] == event]
    if len(df_ev) == 0: continue
    time_vec = metadata_per_event[event]['time_vec']
    
    for name, group in df_ev.groupby(['genotype', 'source', 'target']):
        group_name = f"{name[0]} | {name[1]} -> {name[2]}"
        
        corr_arrays = np.vstack(group['corr_array'].values)
        caus_arrays = np.vstack(group['causal_array'].values)
        
        mean_corr = np.nanmean(corr_arrays, axis=0)
        mean_caus = np.nanmean(caus_arrays, axis=0)
        
        # Extract Correlation Landmarks
        corr_landmarks = extract_landmarks(time_vec, mean_corr, "Correlation", event, group_name)
        if corr_landmarks: landmark_data.append(corr_landmarks)
            
        # Extract Causality Landmarks
        caus_landmarks = extract_landmarks(time_vec, mean_caus, "Causality", event, group_name)
        if caus_landmarks: landmark_data.append(caus_landmarks)

df_landmarks = pd.DataFrame(landmark_data)

# Save to CSV
csv_path = out_dir / "trace_landmarks.csv"
df_landmarks.to_csv(csv_path, index=False)
print(f"Saved {len(df_landmarks)} rows to {csv_path}")

# Save to Markdown
md_path = out_dir / "trace_landmarks.md"
with open(md_path, "w") as f:
    f.write("# LLM Trace Landmarks\n\n")
    f.write(df_landmarks.to_markdown(index=False))
print(f"Saved to {md_path}")
