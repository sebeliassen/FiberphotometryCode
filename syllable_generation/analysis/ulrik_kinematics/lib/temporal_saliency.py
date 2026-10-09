"""
Contextual Temporal Saliency & Multimodal Perturbation Pipeline.

Trains single whole-window Random Forest classifiers on complete
photometry traces (-500ms to +500ms) to model behavioral syllable representation in context.
Measures temporal saliency by systematically perturbing (shuffling, zeroing, or adding noise to)
discrete 50ms time bins of the test set and recording the decoding drop.
Fits empirical smoothing splines to the resulting accuracy drops.
"""
import os
import sys
import threading
import time as _time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.interpolate import splrep, splev
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier as skRFC
import warnings
from joblib import Parallel, delayed
from .saliency_modeling import notify_ntfy

# GPU-accelerated cuML classifiers with sklearn fallback
_USE_CUML = False
try:
    from cuml.ensemble import RandomForestClassifier as cuRFC
    _USE_CUML = True
except ImportError:
    pass

warnings.filterwarnings('ignore')

# Constants and Directories
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output/temporal_saliency")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style map for 6 traces (WT/KI x DS/VS/BOTH)
STYLE_MAP = {
    ("WT", "DS"):   {"color": "#2ecc71", "linestyle": "-",  "label": "WT DS"},
    ("WT", "VS"):   {"color": "#3498db", "linestyle": "-",  "label": "WT VS"},
    ("WT", "BOTH"): {"color": "#1abc9c", "linestyle": "--", "label": "WT BOTH (Joint)"},
    ("KI", "DS"):   {"color": "#e74c3c", "linestyle": "-",  "label": "KI DS"},
    ("KI", "VS"):   {"color": "#e67e22", "linestyle": "-",  "label": "KI VS"},
    ("KI", "BOTH"): {"color": "#9b59b6", "linestyle": "--", "label": "KI BOTH (Joint)"},
}

# --- Memory Guard Configuration ---
MAX_RAM_GB = 100.0   # Preemptive kill if RAM usage exceeds this
MAX_VRAM_GB = 23.5   # Preemptive kill if GPU VRAM usage exceeds this
_MEMORY_GUARD_ACTIVE = False
_MEMORY_GUARD_THREAD = None

def _get_ram_usage_gb():
    """Returns current process RSS in GB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    except Exception:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 ** 3)

def _get_vram_usage_gb():
    """Returns current GPU VRAM usage in GB using nvidia-smi."""
    try:
        import subprocess
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        return float(result.strip()) / 1024.0
    except Exception:
        return 0.0

def _memory_watchdog(interval_s=10):
    """Watchdog thread that monitors RAM and VRAM."""
    global _MEMORY_GUARD_ACTIVE
    while _MEMORY_GUARD_ACTIVE:
        ram = _get_ram_usage_gb()
        vram = _get_vram_usage_gb()
        if ram > MAX_RAM_GB:
            print(f"\n[WATCHDOG ERROR] RAM usage reached {ram:.2f} GB (Max {MAX_RAM_GB} GB). Aborting to protect system.")
            os._exit(1)
        if vram > MAX_VRAM_GB:
            print(f"\n[WATCHDOG ERROR] GPU VRAM usage reached {vram:.2f} GB (Max {MAX_VRAM_GB} GB). Aborting to protect GPU.")
            os._exit(1)
        _time.sleep(interval_s)

def start_memory_guard():
    global _MEMORY_GUARD_ACTIVE, _MEMORY_GUARD_THREAD
    if not _MEMORY_GUARD_ACTIVE:
        _MEMORY_GUARD_ACTIVE = True
        _MEMORY_GUARD_THREAD = threading.Thread(target=_memory_watchdog, daemon=True)
        _MEMORY_GUARD_THREAD.start()
        print("Memory guard thread started successfully.")

def stop_memory_guard():
    global _MEMORY_GUARD_ACTIVE
    _MEMORY_GUARD_ACTIVE = False
    print("Memory guard thread stopped.")

# --- 2. Spline Fitting Helper ---
def _fit_smoothing_spline(time_axis, saliency_curve):
    """
    Fits a cubic smoothing spline (k=3) using splrep and splev.
    Returns (fine_time, fine_fitted, coarse_fitted)
    """
    try:
        idx = np.argsort(time_axis)
        xs = time_axis[idx]
        ys = saliency_curve[idx]
        
        # Adaptive smoothing: use a fraction of the data variance scaled by N.
        # This prevents overshoot while still capturing genuine temporal structure.
        n = len(ys)
        data_var = np.var(ys)
        s = max(data_var * n * 0.05, 0.001)  # smoothing proportional to data noise (reduced to show finer temporal structure)
        tck = splrep(xs, ys, k=3, s=s)
        
        fine_time = np.linspace(xs.min(), xs.max(), 200)
        fine_fitted = splev(fine_time, tck)
        coarse_fitted = splev(time_axis, tck)
        
        return fine_time, fine_fitted, coarse_fitted
    except Exception as e:
        print(f"    Warning: Spline fit failed: {e}. Falling back to linear interpolation.")
        fine_time = np.linspace(time_axis.min(), time_axis.max(), 200)
        fine_fitted = np.interp(fine_time, time_axis, saliency_curve)
        return fine_time, fine_fitted, saliency_curve

# --- 3. Data Loading & Extraction Utilities ---
def _get_top_syllables(sessions, k=20):
    from collections import Counter
    c = Counter()
    for s in sessions:
        if not hasattr(s, 'all_data_df') or s.all_data_df.empty: continue
        b = s.all_data_df["syllable"].dropna()
        b = b[b != -1]
        c.update(b.to_dict().values())
    return [x[0] for x in c.most_common(k)]

def _get_functional_channels(s):
    df = s.all_data_df
    sig_meta = s.metadata.get("signal_meta", {})
    functional_channels = []
    
    for key, info in sig_meta.items():
        if info.get("orig_letter") == "G":
            reg = info.get("brain_reg", "UNK").upper()
            if "DMS" in reg or "DLS" in reg or "DS" in reg: broad_reg = "DS"
            elif "VS" in reg or "NAC" in reg: broad_reg = "VS"
            else: broad_reg = "UNK"
            
            zcol = f"{key}_zscore_datta" if f"{key}_zscore_datta" in df.columns else f"{key}_zscore"
            if zcol in df.columns and broad_reg != "UNK":
                functional_channels.append({
                    "col": key,
                    "zcol": zcol,
                    "broad_reg": broad_reg,
                    "hemi": info.get("hemisphere", "UNK").upper(),
                    "reg_full": reg
                })
    return functional_channels

def extract_all_traces(sessions, target_sylls, window_frames=40):
    """
    Extracts photometry traces around syllable transitions.
    Slices raw and continuous kinematic residuals for high performance.
    """
    print("  Pre-extracting full photometry traces around syllable onsets...")
    all_metadata = []
    traces_dict = {}
    
    for s in sessions:
        df = s.all_data_df
        if "onset" not in df.columns or df.empty: continue
        
        channels = _get_functional_channels(s)
        if not channels: continue
        
        numerical_cols = []
        for ch in channels:
            zcol = ch["zcol"]
            res_col = f"{zcol}_residual_kinematic_free"
            pred_col = f"{zcol}_predicted_kinematics"
            for c in [zcol, res_col, pred_col]:
                if c in df.columns and c not in numerical_cols:
                    numerical_cols.append(c)
                    
        kin_cols = ["lin_vel", "lin_acc", "ang_vel", "ang_acc"]
        for c in kin_cols:
            if c in df.columns and c not in numerical_cols:
                numerical_cols.append(c)
                
        df_indexed = df.set_index("frame_index")
        data_np = df_indexed[numerical_cols].to_numpy().astype(np.float64)
        col_to_idx = {col: i for i, col in enumerate(numerical_cols)}
        idx_values = df_indexed.index.values
        idx_to_pos = {idx: pos for pos, idx in enumerate(idx_values)}
        
        # Extract postural latent states (top 10 components)
        syll_df = s.syllables
        if isinstance(syll_df, dict):
            syll_df = list(syll_df.values())[0]
            
        latent_cols = [c for c in syll_df.columns if 'latent_state' in c]
        if not latent_cols:
            latent_cols = [f"latent_state_{i}" for i in range(10)]
        else:
            latent_cols = sorted(latent_cols, key=lambda x: int(x.split('_')[-1]))[:10]
            
        df_lat = df.join(syll_df[latent_cols], how="left")
        # Fill NaNs natively caused by 40Hz downsampling or join mismatches
        df_lat[latent_cols] = df_lat[latent_cols].ffill().bfill().fillna(0.0)
        df_indexed_lat = df_lat.set_index("frame_index")
        latent_np = df_indexed_lat[latent_cols].to_numpy().astype(np.float64)
        
        geno = s.metadata.get("genotype", "UNK").upper()
        drug = s.metadata.get("drug", "N/A")
        if drug in ["N/A", None]: drug = "Saline"
        if drug.lower().startswith("amph"): drug = "Amphetamine"
        elif drug.lower().startswith("sal"): drug = "Saline"
        
        inj_frame = s.metadata.get("injection_frame")
        
        for syll in target_sylls:
            onsets = df[(df["onset"] > 0) & (df["syllable"] == syll)]["frame_index"].values
            
            for onset in onsets:
                t_start = onset - window_frames
                t_end = onset + window_frames
                
                if t_start not in idx_to_pos or t_end not in idx_to_pos: continue
                
                pos_start = idx_to_pos[t_start]
                pos_end = idx_to_pos[t_end] + 1
                
                total_frames = 2 * window_frames + 1
                if (pos_end - pos_start) != total_frames: continue
                
                phase = "post" if (inj_frame is not None and onset >= inj_frame) else "pre"
                
                all_metadata.append({
                    "Session": s.session_id,
                    "frame_index": onset,
                    "Genotype": geno,
                    "Drug": drug,
                    "Phase": phase,
                    "Syllable": syll,
                })
                
                # Store postural latent trace (global/BOTH)
                latent_arr = latent_np[pos_start:pos_end, :]
                if not np.isnan(latent_arr).any():
                    key_latent = f"{s.session_id}_{onset}_BOTH_latent"
                    traces_dict[key_latent] = latent_arr.flatten()
                    # Populate regional keys for single-region compatibility
                    traces_dict[f"{s.session_id}_{onset}_DS_latent"] = latent_arr.flatten()
                    traces_dict[f"{s.session_id}_{onset}_VS_latent"] = latent_arr.flatten()
                
                kin_cols = ["lin_vel", "lin_acc", "ang_vel", "ang_acc"]
                for ch in channels:
                    zcol = ch["zcol"]
                    res_col = f"{zcol}_residual_kinematic_free"
                    
                    if zcol in col_to_idx:
                        key_raw = f"{s.session_id}_{onset}_{ch['broad_reg']}_raw"
                        traces_dict[key_raw] = data_np[pos_start:pos_end, col_to_idx[zcol]]
                        
                    if res_col in col_to_idx:
                        key_res = f"{s.session_id}_{onset}_{ch['broad_reg']}_res"
                        traces_dict[key_res] = data_np[pos_start:pos_end, col_to_idx[res_col]]
                        
                    if all(c in col_to_idx for c in kin_cols):
                        key_pred = f"{s.session_id}_{onset}_{ch['broad_reg']}_pred"
                        traces_dict[key_pred] = data_np[pos_start:pos_end, [col_to_idx[c] for c in kin_cols]].flatten()
                            
    df_meta = pd.DataFrame(all_metadata)
    return df_meta, traces_dict

# --- 4. Classifier Selection & Preparation Helpers ---
def _prepare_cuml_arrays(X, y):
    import cupy as cp
    return cp.asarray(X, dtype=cp.float32), cp.asarray(y, dtype=cp.int32)

def _get_classifier(pred_type, random_state=42):
    if pred_type == "rfc":
        if _USE_CUML:
            return cuRFC(n_estimators=100, max_depth=10, random_state=random_state)
        else:
            return skRFC(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Unknown predictor type: {pred_type}")

# --- 5. Modular Saliency Pipeline Orchestrator ---
def run_saliency_pipeline(sessions, project_name, window_half_s=0.5, model_type="random_forest"):
    """Entry point to run the contextual whole-window temporal saliency decoders."""
    print(f"\n=======================================================")
    print(f" STARTING CONTEXTUAL TEMPORAL SALIENCY PIPELINE: {project_name.upper()} ({model_type.upper()})")
    if window_half_s == 1.0:
        print("  ⚡ EXTENDED ANALYSIS WINDOW MODE: [-1.0s to +1.0s]")
    print(f"=======================================================")
    
    # Ensure kinematic-stripped residuals are computed and cached
    from . import unique_information
    unique_information.strip_kinematics(sessions, model_type=model_type)
    
    top_sylls = _get_top_syllables(sessions, k=20)
    if not top_sylls:
        print("No top syllables found.")
        return
        
    window_frames = int(window_half_s * 80)
    total_frames = 2 * window_frames + 1
    
    if window_half_s == 1.0:
        output_dir = os.path.join(os.path.dirname(__file__), "../output/temporal_saliency_extended")
    else:
        output_dir = os.path.join(os.path.dirname(__file__), "../output/temporal_saliency")
    os.makedirs(output_dir, exist_ok=True)
        
    df_meta, traces_dict = extract_all_traces(sessions, top_sylls, window_frames=window_frames)
    if df_meta.empty or not traces_dict:
        print("No valid onset events extracted.")
        return
        
    # Define contiguous bins of 50ms each (4 frames) spanning the window
    n_bins = int(window_frames / 2)
    bin_edges = np.linspace(-window_frames, window_frames, n_bins + 1, dtype=int)
    bin_centers_ms = []
    bin_slices = []
    for i in range(n_bins):
        c_frame = (bin_edges[i] + bin_edges[i+1]) / 2.0
        bin_centers_ms.append(c_frame * 12.5) # 1 frame = 12.5ms at 80Hz
        
        start_idx = window_frames + bin_edges[i]
        end_idx = window_frames + bin_edges[i+1]
        if i == (n_bins - 1):
            end_idx = total_frames
        bin_slices.append((start_idx, end_idx))
        
    # Predictors and Perturbations
    predictors = ["rfc"]
    perturbations = ["shuffle"]
    
    start_memory_guard()
    
    # Pre-compute group keys and build flat task list for maximum GPU saturation
    df_meta["Group_Key"] = df_meta["Drug"].astype(str) + "_" + df_meta["Genotype"].astype(str) + "_" + df_meta["Phase"].astype(str)
    unique_groups = sorted(df_meta["Group_Key"].unique())
    
    # Pre-build X/y arrays for each (group, region, trace_type) to avoid redundant work
    MAX_SAMPLES = 4000
    prebuilt = {}
    for g_key in unique_groups:
        sub_group = df_meta[df_meta["Group_Key"] == g_key]
        if len(sub_group) < 50: continue
        
        for trace_type in ["raw", "res", "pred", "latent"]:
            # DS and VS single regions
            for region in ["DS", "VS"]:
                X_list = []
                valid_indices = []
                for idx, row in sub_group.iterrows():
                    key = f"{row['Session']}_{row['frame_index']}_{region}_{trace_type}"
                    if key in traces_dict:
                        X_list.append(traces_dict[key])
                        valid_indices.append(idx)
                if len(X_list) < 50: continue
                X = np.stack(X_list)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                y = sub_group.loc[valid_indices, "Syllable"].to_numpy().astype(int)
                if len(X) > MAX_SAMPLES:
                    X, y = resample(X, y, n_samples=MAX_SAMPLES, random_state=42, stratify=y)
                prebuilt[(g_key, region, trace_type)] = (X, y)
                
            # Combined region BOTH (concatenating DS and VS traces, or loading latent directly)
            X_list_both = []
            valid_indices_both = []
            for idx, row in sub_group.iterrows():
                if trace_type == "latent":
                    key_lat = f"{row['Session']}_{row['frame_index']}_BOTH_latent"
                    if key_lat in traces_dict:
                        X_list_both.append(traces_dict[key_lat])
                        valid_indices_both.append(idx)
                else:
                    key_ds = f"{row['Session']}_{row['frame_index']}_DS_{trace_type}"
                    key_vs = f"{row['Session']}_{row['frame_index']}_VS_{trace_type}"
                    if key_ds in traces_dict and key_vs in traces_dict:
                        combined_trace = np.concatenate([traces_dict[key_ds], traces_dict[key_vs]])
                        X_list_both.append(combined_trace)
                        valid_indices_both.append(idx)
            if len(X_list_both) < 50: continue
            X_both = np.stack(X_list_both)
            X_both = np.nan_to_num(X_both, nan=0.0, posinf=0.0, neginf=0.0)
            y_both = sub_group.loc[valid_indices_both, "Syllable"].to_numpy().astype(int)
            if len(X_both) > MAX_SAMPLES:
                X_both, y_both = resample(X_both, y_both, n_samples=MAX_SAMPLES, random_state=42, stratify=y_both)
            prebuilt[(g_key, "BOTH", trace_type)] = (X_both, y_both)
            
    # Task-Level Caching Check
    csv_out = os.path.join(output_dir, f"{project_name.lower()}_temporal_saliency_results_{model_type}.csv")
    cached_df = pd.DataFrame()
    if os.path.exists(csv_out):
        try:
            cached_df = pd.read_csv(csv_out)
            print(f"  ⚡ Found existing cache with {len(cached_df)} rows at {csv_out}")
        except Exception as e:
            print(f"  ⚠️ Error loading cache: {e}. Starting fresh.")
            
    cached_keys = set()
    if not cached_df.empty:
        # Group by the task identifier
        grouped = cached_df.groupby(["Predictor", "Perturbation", "Drug", "Genotype", "Phase", "Region", "Trace_Type"])
        for keys, _ in grouped:
            cached_keys.add(keys)
            
    # Build flat task list
    tasks = []
    cached_tasks = []
    
    for pred in predictors:
        for pert in perturbations:
            for data_key in prebuilt:
                g_key, region, trace_type = data_key
                drug_name, genotype, phase = g_key.split("_")
                task_key = (pred, pert, drug_name, genotype, phase, region, trace_type)
                
                if task_key in cached_keys:
                    sub_cache = cached_df[
                        (cached_df["Predictor"] == pred) &
                        (cached_df["Perturbation"] == pert) &
                        (cached_df["Drug"] == drug_name) &
                        (cached_df["Genotype"] == genotype) &
                        (cached_df["Phase"] == phase) &
                        (cached_df["Region"] == region) &
                        (cached_df["Trace_Type"] == trace_type)
                    ]
                    cached_tasks.append(sub_cache)
                else:
                    tasks.append((pred, pert, data_key))
                    
    # Running tasks
    new_results = []
    if tasks:
        # Auto-scaled concurrency from environment or default safe fallback
        env_parallel = os.environ.get("N_PARALLEL_STRATEGIES")
        if env_parallel is not None:
            try:
                default_parallel = int(env_parallel)
                print(f"  [CONCURRENCY] Loaded N_PARALLEL_STRATEGIES={default_parallel} from environment.")
            except ValueError:
                default_parallel = 3
        else:
            default_parallel = 3
            
        N_PARALLEL = min(default_parallel, len(tasks))
        print(f"Running {len(tasks)} non-cached tasks across {N_PARALLEL} parallel threads...")
        
        # Thread-safe progress milestone notifications
        completed_tasks = 0
        total_tasks = len(tasks)
        milestones_notified = set()
        lock = threading.Lock()
        
        def run_task_wrapper(pred, pert, data_key):
            nonlocal completed_tasks
            res = _run_single_task(
                pred, pert, data_key, prebuilt[data_key], bin_slices, bin_centers_ms, window_frames=window_frames
            )
            with lock:
                completed_tasks += 1
                pct = int((completed_tasks / total_tasks) * 100)
                # Notify on milestones: 25%, 50%, 75%
                for m in [25, 50, 75]:
                    if pct >= m and m not in milestones_notified:
                        milestones_notified.add(m)
                        msg = f"⏳ Extended Saliency [{project_name}]: {m}% of tasks completed ({completed_tasks}/{total_tasks})."
                        print(f"\n[NTFY] {msg}\n", flush=True)
                        _time.sleep(2)  # avoid ntfy.sh rate limiting
                        notify_ntfy(msg)
            return res
            
        results_list = Parallel(n_jobs=N_PARALLEL, backend="threading")(
            delayed(run_task_wrapper)(pred, pert, data_key)
            for pred, pert, data_key in tasks
        )
        new_results = [res for res in results_list if res is not None and not res.empty]
    else:
        print("🎉 All tasks already cached! Skipping GPU computations completely.")
        
    # Combine results
    all_dfs = cached_tasks + new_results
    if not all_dfs:
        print("No results calculated.")
        stop_memory_guard()
        return
        
    df_res_all = pd.concat(all_dfs, ignore_index=True)
    stop_memory_guard()
    
    # Save complete results CSV
    df_res_all.to_csv(csv_out, index=False)
    print(f"Saved complete results CSV to: {csv_out}")
    
    # Fit splines and generate consolidated visualizations
    _fit_and_plot_results(df_res_all, project_name, bin_centers_ms, output_dir, model_type=model_type)
    
    # Send NTFY update
    if window_half_s == 1.0:
        notify_ntfy(f"🎉 Extended Temporal Saliency Pipeline [-1s to +1s] for {project_name.upper()} has completed successfully! Check the plots in `output/temporal_saliency_extended/`.")
    else:
        notify_ntfy(f"🎉 Contextual Temporal Saliency Pipeline for {project_name.upper()} has completed successfully! Check the new consolidated 6-trace plots.")

def _run_single_task(pred, pert, data_key, data, bin_slices, bin_centers_ms, window_frames=40):
    """Runs a single fine-grained task: one (pred, pert, group, region, trace_type)."""
    import traceback
    try:
        g_key, region, trace_type = data_key
        X, y = data
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # We will track the corrected drop per fold for each bin: (base - pert) - (base_null - pert_null)
        n_bins = len(bin_slices)
        corrected_drops = {b: [] for b in range(n_bins)}
        baseline_scores = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf = _get_classifier(pred)
            is_cuml = _USE_CUML and clf.__class__.__module__.startswith("cuml")
            
            # Scramble test labels for the null hypothesis evaluation
            y_test_null = np.random.permutation(y_test)
            
            if is_cuml:
                X_train_c, y_train_c = _prepare_cuml_arrays(X_train, y_train)
                X_test_c, y_test_c = _prepare_cuml_arrays(X_test, y_test)
                _, y_test_null_c = _prepare_cuml_arrays(X_test, y_test_null)
                
                clf.fit(X_train_c, y_train_c)
                
                base_score = clf.score(X_test_c, y_test_c)
                base_score_null = clf.score(X_test_c, y_test_null_c)
            else:
                clf.fit(X_train, y_train)
                
                base_score = clf.score(X_test, y_test)
                base_score_null = clf.score(X_test, y_test_null)
                
            baseline_scores.append(base_score)
            
            # Evaluate perturbation for each bin
            for b_idx, (start_f, end_f) in enumerate(bin_slices):
                X_test_pert = X_test.copy()
                
                # Identify target column slices to perturb
                if trace_type == "latent":
                    slices = [(start_f * 10, end_f * 10)]
                elif trace_type == "pred":
                    slices = [(start_f * 4, end_f * 4)]
                    if region == "BOTH":
                        total_frames = 2 * window_frames + 1
                        slices.append((total_frames * 4 + start_f * 4, total_frames * 4 + end_f * 4))
                else:
                    slices = [(start_f, end_f)]
                    if region == "BOTH":
                        total_frames = 2 * window_frames + 1
                        slices.append((total_frames + start_f, total_frames + end_f))
                    
                for s_start, s_end in slices:
                    if pert == "shuffle":
                        shuf_idx = np.random.permutation(len(X_test_pert))
                        X_test_pert[:, s_start:s_end] = X_test_pert[shuf_idx, s_start:s_end]
                    elif pert == "zero":
                        X_test_pert[:, s_start:s_end] = 0.0
                    elif pert == "noise":
                        b_std = np.std(X_test_pert[:, s_start:s_end], axis=0, keepdims=True)
                        noise = np.random.normal(0, b_std + 1e-6, size=X_test_pert[:, s_start:s_end].shape)
                        X_test_pert[:, s_start:s_end] += noise
                        
                if is_cuml:
                    X_test_pert_c, _ = _prepare_cuml_arrays(X_test_pert, y_test)
                    pert_score = clf.score(X_test_pert_c, y_test_c)
                    pert_score_null = clf.score(X_test_pert_c, y_test_null_c)
                else:
                    pert_score = clf.score(X_test_pert, y_test)
                    pert_score_null = clf.score(X_test_pert, y_test_null)
                    
                real_drop = base_score - pert_score
                null_drop = base_score_null - pert_score_null
                corrected_drop = real_drop - null_drop
                
                corrected_drops[b_idx].append(corrected_drop)
                
        mean_base = np.mean(baseline_scores)
        drug_name, genotype, phase = g_key.split("_")
        
        results = []
        for b_idx in range(n_bins):
            c_ms = bin_centers_ms[b_idx]
            mean_corrected_drop = np.mean(corrected_drops[b_idx])
            sem_corrected_drop = sem(corrected_drops[b_idx])
            
            results.append({
                "Predictor": pred,
                "Perturbation": pert,
                "Drug": drug_name,
                "Genotype": genotype,
                "Phase": phase,
                "Region": region,
                "Trace_Type": trace_type,
                "Bin_Index": b_idx,
                "Time_Center_ms": c_ms,
                "Baseline_Accuracy": mean_base,
                "Saliency_Drop": mean_corrected_drop,
                "Saliency_SEM": sem_corrected_drop
            })
        return pd.DataFrame(results)
    except Exception as e:
        print(f"  [ERROR] Task {pred}/{pert}/{g_key}/{region}/{trace_type} failed: {e}")
        traceback.print_exc()
        return pd.DataFrame()

# --- 6. Fitting & Plotting Sub-Module ---
def _fit_and_plot_results(df_res, project_name, bin_centers_ms, output_dir, model_type="random_forest"):
    """Fits smoothing splines to accuracy drops and plots WT vs KI comparisons organized in subfolders."""
    print("  Fitting empirical smoothing splines and generating subfolder visualizations...")
    
    # Filter out UNK genotype and UNK brain regions
    if "Genotype" in df_res.columns:
        df_res = df_res[~df_res["Genotype"].astype(str).str.upper().str.contains("UNK")]
    if "Region" in df_res.columns:
        df_res = df_res[~df_res["Region"].astype(str).str.upper().str.contains("UNK")]

    # Initialize Fitted_Saliency column
    df_res["Fitted_Saliency"] = np.nan
    
    # 1. Ensure subfolders exist (no longer deleting old plots to preserve other model results)
    print("    Ensuring output subfolders exist...")
    regions = [
        "VS", "DS", "BOTH", "BOTH_TRACES", "UNIQUE_INFO_COMPARISON",
        "FOUR_TRACES", "FOUR_TRACES/DS", "FOUR_TRACES/VS", "FOUR_TRACES/BOTH"
    ]
    for reg in regions:
        reg_dir = os.path.join(output_dir, reg)
        os.makedirs(reg_dir, exist_ok=True)
                    
    # 3. Group by everything EXCEPT Genotype (since WT vs KI are plotted together)
    grouped = df_res.groupby(["Predictor", "Perturbation", "Trace_Type", "Drug", "Phase", "Region"])
    
    # Simple, highly distinct style map for the two comparative genotypes
    style_map = {
        "WT": {"color": "#2ecc71", "linestyle": "-", "linewidth": 2.5, "label": "WT"}, # Green
        "KI": {"color": "#e74c3c", "linestyle": "-", "linewidth": 2.5, "label": "KI"}, # Red
    }
    
    for keys, sub_df in grouped:
        pred, pert, trace_type, drug, phase, region = keys
        
        plt.figure(figsize=(9, 5.5))
        has_data = False
        
        for genotype in ["WT", "KI"]:
            df_curve = sub_df[sub_df["Genotype"] == genotype].sort_values("Time_Center_ms")
            if df_curve.empty:
                continue
            
            time_axis = df_curve["Time_Center_ms"].to_numpy()
            saliency_curve = df_curve["Saliency_Drop"].to_numpy()
            sem_curve = df_curve["Saliency_SEM"].to_numpy()
            
            # Calculate 95% Confidence Interval bounds (Mean +/- 1.96 * SEM)
            # Normalizing by (Baseline_Accuracy - Chance) where Chance = 0.05 (20 syllables)
            chance_level = 0.05
            mean_baseline = df_curve["Baseline_Accuracy"].mean()
            
            # Skip traces where baseline is not meaningfully above chance.
            # When baseline ≈ chance, dividing by (base - chance) ≈ 0 produces
            # unreliable, inflated relative saliency values.
            min_above_chance = 0.02  # require at least 2pp above chance (i.e. baseline >= 7%)
            if mean_baseline - chance_level < min_above_chance:
                print(f"      Skipping {genotype} {region} (baseline {mean_baseline*100:.1f}% too close to chance {chance_level*100:.0f}%)")
                continue
                
            has_data = True
            normalization_factor = mean_baseline - chance_level
            
            relative_saliency = saliency_curve / normalization_factor
            relative_sem = sem_curve / normalization_factor
            
            lower_bounds = relative_saliency - 1.96 * relative_sem
            upper_bounds = relative_saliency + 1.96 * relative_sem
            
            # Fit smoothing spline to relative mean and boundaries
            fine_time, fine_fitted, coarse_fitted = _fit_smoothing_spline(time_axis, relative_saliency)
            _, fine_lower, _ = _fit_smoothing_spline(time_axis, lower_bounds)
            _, fine_upper, _ = _fit_smoothing_spline(time_axis, upper_bounds)
            
            # Save coarse fitted curve in df_res for compatibility/results CSV
            df_res.loc[df_curve.index, "Fitted_Saliency"] = coarse_fitted
            
            style = style_map[genotype]
            
            # Plot smooth spline curve
            plt.plot(
                fine_time, fine_fitted,
                color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"],
                label=f"{style['label']} (Base: {df_curve['Baseline_Accuracy'].mean() * 100.0:.1f}%)"
            )
            
            # Plot shaded 95% Confidence Interval
            plt.fill_between(
                fine_time, fine_lower, fine_upper,
                color=style["color"], alpha=0.15
            )
            
        if not has_data:
            plt.close()
            continue
            
        # Plot Syllable Onset (black dashed line)
        plt.axvline(0.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label='Syllable Onset')
        
        # Plot Null Baseline (gray dotted line representing zero saliency drop / no effect)
        plt.axhline(0.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)
        
        plt.xlabel("Time Relative to Syllable Onset (ms)", fontsize=11, fontweight="bold")
        plt.ylabel("Relative Saliency (Drop / Base - Chance)", fontsize=11, fontweight="bold")
        plt.title(
            f"Temporal Saliency: {region} Striatal Region\n"
            f"Stripping: {model_type.upper()} | Predictor: {pred.upper()} | Type: {'KINEMATICS' if trace_type == 'pred' else trace_type.upper()}\n"
            f"Condition: {drug} {phase.upper()}",
            fontsize=12, fontweight="bold", pad=12
        )
        
        # Place legend nicely on the right
        plt.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        sns.despine()
        plt.tight_layout()
        
        # Save output file in the respective region's subfolder
        fn = f"{project_name.lower()}_saliency_WT_vs_KI_{pert}_{trace_type}_{drug.lower()}_{phase.lower()}_{model_type}.png"
        out_path = os.path.join(output_dir, region, fn)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      Saved plot to subfolder {region}: {fn}")
        
    # 4. Generate BOTH_TRACES comparisons: superimpose DS and VS on a single plot per genotype
    both_traces_grouped = df_res.groupby(["Predictor", "Perturbation", "Trace_Type", "Drug", "Phase", "Genotype"])
    
    style_map_regions = {
        "DS": {"color": "#3498db", "linestyle": "-", "linewidth": 2.5, "label": "DS"}, # Blue
        "VS": {"color": "#e67e22", "linestyle": "-", "linewidth": 2.5, "label": "VS"}, # Orange
    }
    
    for keys, sub_df in both_traces_grouped:
        pred, pert, trace_type, drug, phase, genotype = keys
        
        plt.figure(figsize=(9, 5.5))
        has_data = False
        
        for region in ["DS", "VS"]:
            df_curve = sub_df[sub_df["Region"] == region].sort_values("Time_Center_ms")
            if df_curve.empty:
                continue
                
            time_axis = df_curve["Time_Center_ms"].to_numpy()
            saliency_curve = df_curve["Saliency_Drop"].to_numpy()
            sem_curve = df_curve["Saliency_SEM"].to_numpy()
            
            chance_level = 0.05
            mean_baseline = df_curve["Baseline_Accuracy"].mean()
            
            min_above_chance = 0.02
            if mean_baseline - chance_level < min_above_chance:
                print(f"      Skipping BOTH_TRACES {genotype} {region} (baseline {mean_baseline*100:.1f}% too close to chance {chance_level*100:.0f}%)")
                continue
                
            has_data = True
            normalization_factor = mean_baseline - chance_level
            
            relative_saliency = saliency_curve / normalization_factor
            relative_sem = sem_curve / normalization_factor
            
            lower_bounds = relative_saliency - 1.96 * relative_sem
            upper_bounds = relative_saliency + 1.96 * relative_sem
            
            # Fit smoothing spline to relative mean and boundaries
            fine_time, fine_fitted, coarse_fitted = _fit_smoothing_spline(time_axis, relative_saliency)
            _, fine_lower, _ = _fit_smoothing_spline(time_axis, lower_bounds)
            _, fine_upper, _ = _fit_smoothing_spline(time_axis, upper_bounds)
            
            style = style_map_regions[region]
            
            # Plot smooth spline curve
            plt.plot(
                fine_time, fine_fitted,
                color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"],
                label=f"{style['label']} (Base: {mean_baseline * 100.0:.1f}%)"
            )
            
            # Plot shaded 95% Confidence Interval
            plt.fill_between(
                fine_time, fine_lower, fine_upper,
                color=style["color"], alpha=0.15
            )
            
        if not has_data:
            plt.close()
            continue
            
        # Plot Syllable Onset (black dashed line)
        plt.axvline(0.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label='Syllable Onset')
        
        # Plot Null Baseline (gray dotted line representing zero saliency drop / no effect)
        plt.axhline(0.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)
        
        plt.xlabel("Time Relative to Syllable Onset (ms)", fontsize=11, fontweight="bold")
        plt.ylabel("Relative Saliency (Drop / Base - Chance)", fontsize=11, fontweight="bold")
        plt.title(
            f"Temporal Saliency: DS vs VS Traces comparison\n"
            f"Genotype: {genotype} | Stripping: {model_type.upper()} | Type: {'KINEMATICS' if trace_type == 'pred' else trace_type.upper()}\n"
            f"Condition: {drug} {phase.upper()}",
            fontsize=12, fontweight="bold", pad=12
        )
        
        # Place legend nicely on the right
        plt.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        sns.despine()
        plt.tight_layout()
        
        # Save output file in the BOTH_TRACES subfolder
        fn = f"{project_name.lower()}_saliency_DS_vs_VS_{genotype}_{pert}_{trace_type}_{drug.lower()}_{phase.lower()}_{model_type}.png"
        out_path = os.path.join(output_dir, "BOTH_TRACES", fn)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      Saved plot to subfolder BOTH_TRACES: {fn}")
        
    # 5. Generate UNIQUE_INFO_COMPARISON plots: superimpose Raw, Res, Pred, and Latent (if BOTH)
    comparison_grouped = df_res.groupby(["Predictor", "Perturbation", "Drug", "Phase", "Genotype", "Region"])
    
    style_map_comparison = {
        "raw": {"color": "#ff7675", "linestyle": "-", "linewidth": 2.5, "label": "Raw DA (Total)"}, # Coral/Red
        "res": {"color": "#2ecc71", "linestyle": "-", "linewidth": 2.5, "label": "Residual DA (Unique)"}, # Green
        "pred": {"color": "#3498db", "linestyle": "-", "linewidth": 2.5, "label": "Kinematics (4 Features)"}, # Blue
        "latent": {"color": "#9b59b6", "linestyle": "--", "linewidth": 2.5, "label": "Postural Latent States"}, # Purple
    }
    
    for keys, sub_df in comparison_grouped:
        pred, pert, drug, phase, genotype, region = keys
        
        # Plot all 4 trace types for all regions (DS, VS, BOTH)
        trace_types = ["raw", "res", "pred", "latent"]
            
        plt.figure(figsize=(9, 5.5))
        has_data = False
        
        for t_type in trace_types:
            df_curve = sub_df[sub_df["Trace_Type"] == t_type].sort_values("Time_Center_ms")
            if df_curve.empty:
                continue
                
            time_axis = df_curve["Time_Center_ms"].to_numpy()
            saliency_curve = df_curve["Saliency_Drop"].to_numpy()
            sem_curve = df_curve["Saliency_SEM"].to_numpy()
            
            chance_level = 0.05
            mean_baseline = df_curve["Baseline_Accuracy"].mean()
            
            # Use lower threshold (0.02) to display all valid traces
            min_above_chance = 0.02
            if mean_baseline - chance_level < min_above_chance:
                print(f"      Skipping comparison curve {genotype} {region} {t_type} (baseline {mean_baseline*100:.1f}% too close to chance {chance_level*100:.0f}%)")
                continue
                
            has_data = True
            normalization_factor = mean_baseline - chance_level
            
            relative_saliency = saliency_curve / normalization_factor
            relative_sem = sem_curve / normalization_factor
            
            lower_bounds = relative_saliency - 1.96 * relative_sem
            upper_bounds = relative_saliency + 1.96 * relative_sem
            
            # Fit smoothing spline to relative mean and boundaries
            fine_time, fine_fitted, coarse_fitted = _fit_smoothing_spline(time_axis, relative_saliency)
            _, fine_lower, _ = _fit_smoothing_spline(time_axis, lower_bounds)
            _, fine_upper, _ = _fit_smoothing_spline(time_axis, upper_bounds)
            
            style = style_map_comparison[t_type]
            
            # Plot smooth spline curve
            plt.plot(
                fine_time, fine_fitted,
                color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"],
                label=f"{style['label']} (Base: {mean_baseline * 100.0:.1f}%)"
            )
            
            # Plot shaded 95% Confidence Interval
            plt.fill_between(
                fine_time, fine_lower, fine_upper,
                color=style["color"], alpha=0.15
            )
            
        if not has_data:
            plt.close()
            continue
            
        # Plot Syllable Onset (black dashed line)
        plt.axvline(0.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label='Syllable Onset')
        
        # Plot Null Baseline (gray dotted line representing zero saliency drop / no effect)
        plt.axhline(0.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)
        
        plt.xlabel("Time Relative to Syllable Onset (ms)", fontsize=11, fontweight="bold")
        plt.ylabel("Relative Saliency (Drop / Base - Chance)", fontsize=11, fontweight="bold")
        plt.title(
            f"Unique Info Temporal Saliency: {region} Region\n"
            f"Genotype: {genotype} | Stripping: {model_type.upper()}\n"
            f"Condition: {drug} {phase.upper()}",
            fontsize=12, fontweight="bold", pad=12
        )
        
        # Place legend nicely on the right
        plt.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        sns.despine()
        plt.tight_layout()
        
        # Save output file in the UNIQUE_INFO_COMPARISON and FOUR_TRACES subfolders
        fn = f"{project_name.lower()}_saliency_comparison_{genotype}_{pert}_{region.lower()}_{drug.lower()}_{phase.lower()}_{model_type}.png"
        out_path = os.path.join(output_dir, "UNIQUE_INFO_COMPARISON", fn)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        
        # Save to FOUR_TRACES with region subfolder to prevent clutter
        dir_four_sub = os.path.join(output_dir, "FOUR_TRACES", region.upper())
        os.makedirs(dir_four_sub, exist_ok=True)
        out_path_four = os.path.join(dir_four_sub, fn)
        plt.savefig(out_path_four, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      Saved plot to subfolders UNIQUE_INFO_COMPARISON & FOUR_TRACES/{region.upper()}: {fn}")
        
    _cleanup_old_plots(project_name, output_dir)

def _cleanup_old_plots(project_name, output_dir):
    """Deletes old scattered/individual saliency plots in the output directory."""
    print("  🧹 Cleaning up old scattered plots...")
    import glob
    old_patterns = [
        f"{project_name.lower()}_saliency_fit_*.png",
        f"{project_name.lower()}_predictor_comparison_*.png",
        f"{project_name.lower()}_saliency_consolidated_*.png"
    ]
    deleted_count = 0
    for pat in old_patterns:
        full_pat = os.path.join(output_dir, pat)
        for filepath in glob.glob(full_pat):
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"    Failed to delete {os.path.basename(filepath)}: {e}")
    if deleted_count > 0:
        print(f"  🧹 Successfully deleted {deleted_count} legacy plots.")
