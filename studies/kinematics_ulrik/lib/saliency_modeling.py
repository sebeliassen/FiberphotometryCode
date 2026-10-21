"""
Biophysical Saliency Modeling & Synaptic Integration Pipeline.

Implements modular time-resolved saliency decoders under configurable
perturbation strategies (leaky integration filters, phase jittering, low-pass spectral filtering)
and fits a biophysical mixture model kernel (Planning + Trigger + Decay) using scipy curve_fit.
"""
import os
import threading
import time as _time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
from scipy.stats import sem
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

# GPU-accelerated cuML RandomForest with sklearn fallback
_USE_CUML = False
try:
    from cuml.ensemble import RandomForestClassifier as cuRFC
    _USE_CUML = True
except ImportError:
    pass

from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as skRFC

# Constants and Directories
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output/saliency_modeling")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Memory Guard Configuration ---
MAX_RAM_GB = 100.0   # Preemptive kill if RAM usage exceeds this
MAX_VRAM_GB = 20.0   # Preemptive kill if GPU VRAM usage exceeds this
_MEMORY_GUARD_ACTIVE = False
_MEMORY_GUARD_THREAD = None

def _get_ram_usage_gb():
    """Returns current process RSS in GB."""
    try:
        import resource
        # maxrss is in KB on Linux
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except Exception:
        return 0.0

def _get_vram_usage_gb():
    """Returns total GPU VRAM usage in GB via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            usages = [float(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
            return max(usages) / 1024.0 if usages else 0.0
    except Exception:
        pass
    return 0.0

def _memory_watchdog(interval_s=10):
    """Background thread that monitors RAM and VRAM, killing the process if limits are exceeded."""
    global _MEMORY_GUARD_ACTIVE
    while _MEMORY_GUARD_ACTIVE:
        ram_gb = _get_ram_usage_gb()
        vram_gb = _get_vram_usage_gb()
        if ram_gb > MAX_RAM_GB:
            msg = f"\n🚨 MEMORY GUARD: RAM usage ({ram_gb:.1f} GB) exceeded limit ({MAX_RAM_GB} GB). Killing process to protect system."
            print(msg)
            notify_ntfy(msg)
            os._exit(137)  # SIGKILL-like exit
        if vram_gb > MAX_VRAM_GB:
            msg = f"\n🚨 MEMORY GUARD: VRAM usage ({vram_gb:.1f} GB) exceeded limit ({MAX_VRAM_GB} GB). Killing process to protect system."
            print(msg)
            notify_ntfy(msg)
            os._exit(137)
        _time.sleep(interval_s)

def start_memory_guard():
    """Starts the background memory watchdog thread."""
    global _MEMORY_GUARD_ACTIVE, _MEMORY_GUARD_THREAD
    if _MEMORY_GUARD_ACTIVE:
        return
    _MEMORY_GUARD_ACTIVE = True
    _MEMORY_GUARD_THREAD = threading.Thread(target=_memory_watchdog, daemon=True)
    _MEMORY_GUARD_THREAD.start()
    print(f"  🛡️ Memory guard active: RAM limit={MAX_RAM_GB}GB, VRAM limit={MAX_VRAM_GB}GB")

def stop_memory_guard():
    """Stops the background memory watchdog thread."""
    global _MEMORY_GUARD_ACTIVE
    _MEMORY_GUARD_ACTIVE = False

# 1. Central Configuration Dictionary
# --- Performance Tuning ---
CV_FOLDS = 3          # Number of cross-validation folds (reduced from 5 for speed)
MAX_SAMPLES = 2000    # Max samples per group-bin (reduced from 4000 for speed)
N_PARALLEL_STRATEGIES = int(os.environ.get("N_PARALLEL_STRATEGIES", "6"))  # Number of strategies to run in parallel
N_INNER_THREADS = int(os.environ.get("N_INNER_THREADS", "1"))              # Number of inner threads for bin training

MODEL_CONFIG = {
    "perturbation_strategy": "none",  # 'none', 'residual', 'synaptic_filter', 'jitter', 'spectral', 'naive_permutation'
    "perturbation_params": {
        "tau_synaptic_ms": 150.0,       # Downstream GPCR integration time-constant
        "jitter_std_ms": 50.0,          # Trial-by-trial temporal phase jittering
        "cutoff_hz": 5.0,               # Low-pass filter frequency boundary
    },
    "model_type": "rfc",               # 'rfc'
    "window_s": 0.5,
    "bin_width_ms": 50.0,
}

# 2. Biophysical Perturbation Functions
def apply_synaptic_filter(X, fs, tau_ms):
    """
    Convolves photometry traces with a single-exponential decay kernel h(t) = exp(-t/tau)
    representing downstream receptor leaky integration kinetics.
    """
    dt = 1.0 / fs
    tau_s = tau_ms / 1000.0
    kernel_len = X.shape[1]
    t = np.arange(kernel_len) * dt
    kernel = np.exp(-t / tau_s)
    kernel = kernel / np.sum(kernel) # Preserve amplitude scale

    X_filt = np.zeros_like(X)
    for i in range(len(X)):
        # Causal convolution in the forward direction
        X_filt[i] = np.convolve(X[i], kernel, mode='full')[:kernel_len]
    return X_filt

def apply_spectral_filter(X, fs, cutoff_hz):
    """
    Applies a zero-phase digital low-pass Butterworth filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    if normal_cutoff >= 1.0:
        return X
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    X_filt = np.zeros_like(X)
    for i in range(len(X)):
        X_filt[i] = filtfilt(b, a, X[i])
    return X_filt

def _prepare_cuml_arrays(X, y):
    """Ensure arrays are contiguous float32 for cuML GPU compatibility."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.int32)
    return X, y

_cuml_logged = False
def _get_classifier(model_type):
    """
    Modular classifier selector supporting GPU-accelerated cuML and CPU sklearn.
    """
    global _cuml_logged
    if model_type == "rfc":
        if _USE_CUML:
            if not _cuml_logged:
                print("======================================================================")
                print(" SUCCESS: GPU-Accelerated cuML Random Forests Activated!")
                print("======================================================================")
                _cuml_logged = True
            return cuRFC(n_estimators=50, max_depth=10, random_state=42)
        else:
            return skRFC(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

    else:
        raise ValueError(f"Unknown classifier model type: {model_type}")

def notify_ntfy(message):
    """Sends a notification to the user via ntfy.sh."""
    import urllib.request
    try:
        url = "https://ntfy.sh/woody_videos_seb"
        req = urllib.request.Request(url, data=message.encode('utf-8'), method='POST')
        with urllib.request.urlopen(req, timeout=10) as response:
            pass
    except Exception as e:
        print(f"Warning: Failed to send ntfy notification. {e}")

# 3. Biophysical Mixture Model Kernels
def plan_kernel(t, A_plan, tau_rise):
    """Alpha function modeling the anticipatory choice ramp (t < 0)."""
    val = np.zeros_like(t)
    mask = (t < 0)
    if mask.sum() > 0:
        tau_rise = max(1e-5, tau_rise)
        val[mask] = A_plan * (-t[mask] / tau_rise) * np.exp(1.0 + t[mask] / tau_rise)
    return val

def trigger_kernel(t, A_exec, sigma):
    """Gaussian kernel modeling the sharp transition trigger peak."""
    sigma = max(1e-5, sigma)
    return A_exec * np.exp(-t**2 / (2.0 * sigma**2))

def decay_kernel(t, A_decay, tau_decay):
    """Exponential decay modeling post-onset sensor clearance and feedback (t >= 0)."""
    val = np.zeros_like(t)
    mask = (t >= 0)
    if mask.sum() > 0:
        tau_decay = max(1e-5, tau_decay)
        val[mask] = A_decay * np.exp(-t[mask] / tau_decay)
    return val

def biophysical_mixture_model(t, A_plan, tau_rise, A_exec, sigma, A_decay, tau_decay):
    """Complete 3-component biophysical mixture model."""
    return plan_kernel(t, A_plan, tau_rise) + trigger_kernel(t, A_exec, sigma) + decay_kernel(t, A_decay, tau_decay)

def _fit_biophysical_kernel(time_axis, saliency_curve):
    """
    Fits the biophysical mixture model using non-linear least squares.
    """
    # Shift time axis slightly to prevent fitting instabilities and define baseline shift
    y = np.clip(saliency_curve - 1.0, 0, None) # Subtract chance baseline (1.0)

    # Reasonable bounds and initial guesses
    # [A_plan, tau_rise, A_exec, sigma, A_decay, tau_decay]
    initial_guess = [0.2, 0.15, 0.5, 0.05, 0.3, 0.3]
    bounds = (
        [0.0, 0.01, 0.0, 0.01, 0.0, 0.01],
        [5.0, 1.0, 5.0, 0.5, 5.0, 1.0]
    )

    try:
        popt, pcov = curve_fit(biophysical_mixture_model, time_axis, y, p0=initial_guess, bounds=bounds, maxfev=5000)
        return popt, biophysical_mixture_model(time_axis, *popt) + 1.0 # Return fitted curve offset back by chance
    except Exception as e:
        print(f"    Warning: Biophysical mixture fit failed to converge. {e}")
        return None, None

# 4. Data Extraction Pipeline
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
                    "broad_reg": broad_reg
                })
    return functional_channels

def extract_all_modeling_traces(sessions, target_sylls, config):
    """
    Extracts photometry traces around syllable transitions.
    Applies temporal phase jittering directly to transition frame offsets if configured.
    """
    print("  Extracting traces and applying biophysical perturbations...")
    all_metadata = []

    strategy = config["perturbation_strategy"]
    jitter_std = config["perturbation_params"]["jitter_std_ms"]

    for s in sessions:
        df = s.all_data_df
        if "onset" not in df.columns or df.empty: continue

        channels = _get_functional_channels(s)
        if not channels: continue

        fs = s.metadata.get("fs", 80.0)
        window_frames = int(config["window_s"] * fs)

        if strategy == "residual":
            numerical_cols = []
            for ch in channels:
                col = f"{ch['zcol']}_residual_kinematic_free"
                if col in df.columns:
                    numerical_cols.append(col)
                else:
                    numerical_cols.append(ch["zcol"])
        else:
            numerical_cols = [ch["zcol"] for ch in channels]

        # Direct NumPy indexing for maximum extraction speed
        df_indexed = df.set_index("frame_index")
        data_np = df_indexed[numerical_cols].to_numpy().astype(np.float64)
        col_to_idx = {col: i for i, col in enumerate(numerical_cols)}
        idx_values = df_indexed.index.values
        idx_to_pos = {idx: pos for pos, idx in enumerate(idx_values)}

        geno = s.metadata.get("genotype", "UNK").upper()
        drug = s.metadata.get("drug", "N/A")
        if drug in ["N/A", None]: drug = "Saline"
        if drug.lower().startswith("amph"): drug = "Amphetamine"
        elif drug.lower().startswith("sal"): drug = "Saline"

        inj_frame = s.metadata.get("injection_frame")

        for syll in target_sylls:
            onsets = df[(df["onset"] > 0) & (df["syllable"] == syll)]["frame_index"].values

            for onset in onsets:
                # Apply Trial-by-Trial Temporal Phase Jittering
                if strategy == "jitter" and jitter_std > 0:
                    jitter_s = np.random.normal(0, jitter_std / 1000.0)
                    jitter_frames = int(round(jitter_s * fs))
                    onset_act = onset + jitter_frames
                else:
                    onset_act = onset

                t_start = onset_act - window_frames
                t_end = onset_act + window_frames

                if t_start not in idx_to_pos or t_end not in idx_to_pos: continue

                pos_start = idx_to_pos[t_start]
                pos_end = idx_to_pos[t_end] + 1

                expected_len = 2 * window_frames + 1
                if (pos_end - pos_start) != expected_len: continue

                phase = "post" if (inj_frame is not None and onset >= inj_frame) else "pre"

                vs_arr = None
                ds_arr = None

                for ch in channels:
                    zcol = ch["zcol"]
                    broad_reg = ch["broad_reg"]

                    if strategy == "residual":
                        col_name = f"{zcol}_residual_kinematic_free"
                        if col_name not in col_to_idx:
                            col_name = zcol
                    else:
                        col_name = zcol

                    arr = data_np[pos_start:pos_end, col_to_idx[col_name]]

                    if broad_reg == "VS":
                        vs_arr = arr
                    elif broad_reg == "DS":
                        ds_arr = arr

                all_metadata.append({
                    "Session": s.session_id,
                    "frame_index": onset,
                    "Genotype": geno,
                    "Drug": drug,
                    "Phase": phase,
                    "Syllable": syll,
                    "VS_trace": vs_arr,
                    "DS_trace": ds_arr
                })

    df_meta = pd.DataFrame(all_metadata)
    return df_meta

def _evaluate_single_task(b_idx, c_ms, g_key, region, X, y, CV_FOLDS, model_type, _USE_CUML):
    """Evaluates a single sliding window bin for a group and region."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    clf = _get_classifier(model_type)

    accs = []
    shufs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if _USE_CUML:
            X_train, y_train = _prepare_cuml_arrays(X_train, y_train)
            X_test, y_test = _prepare_cuml_arrays(X_test, y_test)

        clf.fit(X_train, y_train)
        accs.append(clf.score(X_test, y_test))

        y_shuf = np.random.permutation(y_train)
        clf.fit(X_train, y_shuf)
        shufs.append(clf.score(X_test, y_test))

    mean_acc = np.mean(accs)
    mean_shuf = np.mean(shufs)
    fold = mean_acc / mean_shuf if mean_shuf > 0 else 1.0
    sem_fold = sem([a / mean_shuf for a in accs])

    drug_name, genotype, phase = g_key.split("_")

    return {
        "Bin_Index": b_idx,
        "Time_Center_ms": c_ms,
        "Drug": drug_name,
        "Genotype": genotype,
        "Phase": phase,
        "Region": region,
        "Saliency_Fold": fold,
        "Saliency_SEM": sem_fold
    }

def _run_single_strategy(sessions, project_name, strat, model_config):
    """Helper to run a single perturbation strategy (for parallel execution)."""
    config = model_config.copy()
    config["perturbation_strategy"] = strat
    # NOTE: 'residual' strategy requires pre-computed residuals — must be done before parallelization
    result = run_modeling_pipeline(sessions, project_name, custom_config=config)
    notify_ntfy(f"✅ Saliency Modeling: Finished strategy '{strat.upper()}' for dataset {project_name}.")
    return result

# 5. Core Modeling Pipeline
def run_modeling_pipeline(sessions, project_name, custom_config=None):
    """
    Modular orchestrator executing the Saliency Modeling pipeline:
    Applies chosen filters, trains classifiers in sliding-windows, and fits biophysical kernels.
    """
    if custom_config is None:
        strategies = ["none", "residual", "synaptic_filter", "jitter", "spectral", "naive_permutation"]
        print(f"No custom configuration provided. Running comparison across: {strategies}")
        notify_ntfy(f"🚀 Saliency Modeling: Starting complete comparison across {len(strategies)} strategies for dataset {project_name}.")

        # Start memory guard for the full run
        start_memory_guard()

        # Pre-compute residuals once (required by 'residual' strategy)
        if "residual" in strategies:
            from analysis.ulrik_kinematics.lib import unique_information
            print("  Ensuring continuous kinematic-free residuals are calculated...")
            unique_information.strip_kinematics(sessions)

        # Separate strategies that can be parallelized from those that can't
        # 'residual' needs the pre-computed residuals but can run in parallel after that
        # Run strategies in parallel batches on either CPU or GPU
        n_parallel = min(N_PARALLEL_STRATEGIES, len(strategies))
        if n_parallel == 1:
            backend = "sequential"
        else:
            backend = "threading" if _USE_CUML else "loky"
        print(f"  Running {len(strategies)} strategies with parallelism={n_parallel} (GPU={'yes' if _USE_CUML else 'no'} via {backend})...")

        Parallel(n_jobs=n_parallel, backend=backend)(
            delayed(_run_single_strategy)(sessions, project_name, strat, MODEL_CONFIG)
            for strat in strategies
        )

        # After running all strategies, generate the perturbation comparison plot
        stop_memory_guard()
        plot_perturbation_comparison(project_name, strategies=strategies, model_type=MODEL_CONFIG["model_type"])
        notify_ntfy(f"🎉 Saliency Modeling: Completed all comparison plots and CSVs for dataset {project_name}!")
        return
    config = custom_config
    strategy = config["perturbation_strategy"]
    model_type = config["model_type"]

    # --- Cache Check: Skip if CSV already exists ---
    csv_out = os.path.join(OUTPUT_DIR, f"{project_name.lower()}_saliency_modeling_{strategy}_{model_type}.csv")
    if os.path.exists(csv_out):
        print(f"\n  ⚡ CACHE HIT: {os.path.basename(csv_out)} already exists. Skipping computation.")
        return pd.read_csv(csv_out)

    print("\n=======================================================")
    print(f" STARTING SALIENCY MODELING PIPELINE: {project_name.upper()}")
    print(f" Config: Perturbation={strategy} | Model={model_type}")
    print("=======================================================")

    top_sylls = _get_top_syllables(sessions, k=20)
    if not top_sylls:
        print("No top syllables found.")
        return

    df_meta = extract_all_modeling_traces(sessions, top_sylls, config)
    if df_meta.empty:
        print("No valid transitions extracted.")
        return

    fs = sessions[0].metadata.get("fs", 80.0)
    window_frames = int(config["window_s"] * fs)
    total_frames = 2 * window_frames + 1
    time_axis = np.linspace(-config["window_s"], config["window_s"], total_frames)

    # Apply Continuous Biophysical Filters (Synaptic Integration or Low-pass Butterworth)
    for region in ["VS", "DS"]:
        col_name = f"{region}_trace"
        valid_mask = df_meta[col_name].notna()
        if valid_mask.sum() == 0: continue

        traces_np = np.stack(df_meta.loc[valid_mask, col_name].values)
        if strategy == "synaptic_filter":
            traces_np = apply_synaptic_filter(traces_np, fs, config["perturbation_params"]["tau_synaptic_ms"])
        elif strategy == "spectral":
            traces_np = apply_spectral_filter(traces_np, fs, config["perturbation_params"]["cutoff_hz"])

        df_meta.loc[valid_mask, col_name] = pd.Series(list(traces_np), index=df_meta.index[valid_mask])

    # Setup sliding-window boundaries
    bin_width_frames = int(round((config["bin_width_ms"] / 1000.0) * fs))
    if bin_width_frames < 1: bin_width_frames = 1

    bin_centers_frames = []
    bin_centers_ms = []

    # Slide across the window frames
    for i in range(0, total_frames - bin_width_frames + 1, max(1, bin_width_frames // 2)):
        center_frame = i + bin_width_frames / 2.0 - window_frames
        bin_centers_frames.append((i, i + bin_width_frames))
        bin_centers_ms.append((center_frame / fs) * 1000.0)

    n_bins = len(bin_centers_ms)
    print(f"  Evaluating {n_bins} sliding bins (width {config['bin_width_ms']}ms)...")

    results = []

    if strategy == "naive_permutation":
        print("  Running legacy Naive Permutation strategy (training 1 classifier on all bins)...")
        # Build features for each region
        df_meta["Group_Key"] = df_meta["Drug"].astype(str) + "_" + df_meta["Genotype"].astype(str) + "_" + df_meta["Phase"].astype(str)
        unique_groups = sorted(df_meta["Group_Key"].unique())

        for g_key in unique_groups:
            sub = df_meta[df_meta["Group_Key"] == g_key]
            if len(sub) < 50: continue

            for region in ["VS", "DS"]:
                col_name = f"{region}_trace"
                sub_reg = sub[sub[col_name].notna()]
                if len(sub_reg) < 50: continue

                # Construct features of shape (n_trials, n_bins)
                traces_np = np.stack(sub_reg[col_name].values)

                X_list = []
                for b_idx, (start_f, end_f) in enumerate(bin_centers_frames):
                    bin_feat = traces_np[:, start_f:end_f].mean(axis=1)
                    X_list.append(bin_feat)
                X_group = np.column_stack(X_list) # shape (n_trials, n_bins)
                y_group = sub_reg["Syllable"].to_numpy().astype(int)

                # Resample for fair balanced baseline
                if len(X_group) > MAX_SAMPLES:
                    indices_resampled = resample(np.arange(len(X_group)), n_samples=MAX_SAMPLES, random_state=42, stratify=y_group)
                    X_group_res = X_group[indices_resampled]
                    y_group_res = y_group[indices_resampled]
                else:
                    X_group_res = X_group
                    y_group_res = y_group

                # Train and evaluate modular classifier
                skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
                clf = _get_classifier(model_type)

                bin_drops = {b: [] for b in range(n_bins)}

                for train_idx, test_idx in skf.split(X_group_res, y_group_res):
                    X_train, X_test = X_group_res[train_idx], X_group_res[test_idx]
                    y_train, y_test = y_group_res[train_idx], y_group_res[test_idx]

                    if _USE_CUML:
                        X_train, y_train = _prepare_cuml_arrays(X_train, y_train)
                        X_test, y_test = _prepare_cuml_arrays(X_test, y_test)

                    clf.fit(X_train, y_train)
                    baseline_score = clf.score(X_test, y_test)

                    y_shuf = np.random.permutation(y_train)
                    clf_shuf = _get_classifier(model_type)
                    clf_shuf.fit(X_train, y_shuf)
                    shuf_score = clf_shuf.score(X_test, y_test)
                    if shuf_score <= 0: shuf_score = 0.05

                    for b_idx in range(n_bins):
                        X_test_perm = X_test.copy()
                        X_test_perm[:, b_idx] = np.random.permutation(X_test_perm[:, b_idx])

                        permuted_score = clf.score(X_test_perm, y_test)
                        acc_drop = baseline_score - permuted_score
                        scaled_importance = 1.0 + (acc_drop / shuf_score)
                        bin_drops[b_idx].append(scaled_importance)

                drug_name, genotype, phase = g_key.split("_")
                for b_idx in range(n_bins):
                    c_ms = bin_centers_ms[b_idx]
                    mean_val = np.mean(bin_drops[b_idx])
                    sem_val = sem(bin_drops[b_idx])

                    results.append({
                        "Bin_Index": b_idx,
                        "Time_Center_ms": c_ms,
                        "Drug": drug_name,
                        "Genotype": genotype,
                        "Phase": phase,
                        "Region": region,
                        "Saliency_Fold": mean_val,
                        "Saliency_SEM": sem_val
                    })
    else:
        # Pre-stack VS and DS traces once for high-performance vectorized slicing
        vs_traces = None
        ds_traces = None
        vs_mask = df_meta["VS_trace"].notna().to_numpy()
        ds_mask = df_meta["DS_trace"].notna().to_numpy()

        if vs_mask.sum() > 0:
            vs_traces = np.stack(df_meta.loc[vs_mask, "VS_trace"].values)
        if ds_mask.sum() > 0:
            ds_traces = np.stack(df_meta.loc[ds_mask, "DS_trace"].values)

        # Pre-calculate Group Keys to avoid pandas string operations in the hot loop
        df_meta["Group_Key"] = df_meta["Drug"].astype(str) + "_" + df_meta["Genotype"].astype(str) + "_" + df_meta["Phase"].astype(str)
        group_keys = df_meta["Group_Key"].to_numpy()
        unique_groups = sorted(df_meta["Group_Key"].unique())

        # Pre-extract syllables as a numpy array
        syllables = df_meta["Syllable"].to_numpy().astype(int)

        # Pre-slice and calculate means for all bins and regions for extreme speedup
        bin_means = {
            "VS": np.zeros((len(df_meta), n_bins)),
            "DS": np.zeros((len(df_meta), n_bins))
        }

        for b_idx, (start_f, end_f) in enumerate(bin_centers_frames):
            if vs_traces is not None:
                bin_means["VS"][vs_mask, b_idx] = vs_traces[:, start_f:end_f].mean(axis=1)
            if ds_traces is not None:
                bin_means["DS"][ds_mask, b_idx] = ds_traces[:, start_f:end_f].mean(axis=1)

        # Loop over bins (collecting tasks for parallel or sequential execution)
        tasks = []
        for b_idx, (start_f, end_f) in enumerate(bin_centers_frames):
            c_ms = bin_centers_ms[b_idx]

            for g_key in unique_groups:
                g_mask = (group_keys == g_key)
                if g_mask.sum() < 50: continue

                for region in ["VS", "DS"]:
                    reg_mask = vs_mask if region == "VS" else ds_mask
                    combined_mask = g_mask & reg_mask
                    if combined_mask.sum() < 50: continue

                    # Extract the pre-sliced features of shape (n_trials, 1)
                    X = bin_means[region][combined_mask, b_idx:b_idx+1]
                    y = syllables[combined_mask]

                    # Resample for fair balanced baseline
                    if len(X) > MAX_SAMPLES:
                        X, y = resample(X, y, n_samples=MAX_SAMPLES, random_state=42, stratify=y)

                    tasks.append((b_idx, c_ms, g_key, region, X, y))

        if N_INNER_THREADS > 1:
            print(f"  Training {len(tasks)} classifiers in parallel with inner_parallelism={N_INNER_THREADS} (threading backend)...")
            parallel_results = Parallel(n_jobs=N_INNER_THREADS, backend="threading")(
                delayed(_evaluate_single_task)(
                    b_idx, c_ms, g_key, region, X, y, CV_FOLDS, model_type, _USE_CUML
                )
                for b_idx, c_ms, g_key, region, X, y in tasks
            )
            results.extend(parallel_results)
        else:
            for b_idx, c_ms, g_key, region, X, y in tasks:
                res_dict = _evaluate_single_task(
                    b_idx, c_ms, g_key, region, X, y, CV_FOLDS, model_type, _USE_CUML
                )
                results.append(res_dict)

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("  Error: No valid modeling results compiled.")
        return None

    df_res.to_csv(csv_out, index=False)
    print(f"  Saved modeling results to: {csv_out}")

    # 6. Fit Biophysical Kernels & Generate Visualizations
    plot_modeling_results(df_res, project_name, strategy, model_type)

    return df_res

def plot_modeling_results(df_res, project_name, strategy, model_type):
    """
    Fits biophysical kernels and generates premium plots superimposing WT and KI curves
    and displaying estimated mathematical time-constants.
    """
    print("  Fitting biophysical kernels and plotting curves...")
    drugs = sorted(df_res["Drug"].unique())
    phases = sorted(df_res["Phase"].unique())
    regions = sorted(df_res["Region"].unique())

    fit_records = []

    for drug in drugs:
        for phase in phases:
            for region in regions:
                sub = df_res[(df_res["Drug"] == drug) & (df_res["Phase"] == phase) & (df_res["Region"] == region)]
                if sub.empty: continue

                plt.figure(figsize=(10, 6))
                color_map = {"WT": "#2ecc71", "KI": "#e74c3c"}

                for genotype in ["WT", "KI"]:
                    g_sub = sub[sub["Genotype"] == genotype].sort_values("Time_Center_ms")
                    if g_sub.empty: continue

                    c = color_map.get(genotype, "blue")
                    t_ms = g_sub["Time_Center_ms"].to_numpy()
                    t_s = t_ms / 1000.0  # Convert to seconds for kernel scales
                    y_emp = g_sub["Saliency_Fold"].to_numpy()
                    sem_emp = g_sub["Saliency_SEM"].to_numpy()

                    # 1. Plot Empirical Saliency with shaded 95% Confidence Intervals
                    plt.scatter(t_ms, y_emp, color=c, alpha=0.75, s=40)
                    plt.fill_between(t_ms, y_emp - 1.96 * sem_emp, y_emp + 1.96 * sem_emp, color=c, alpha=0.10)

                    # 2. Fit Biophysical Mixture Kernel
                    popt, y_fit = _fit_biophysical_kernel(t_s, y_emp)

                    if popt is not None:
                        # Extract fitting parameters
                        # popt = [A_plan, tau_rise, A_exec, sigma, A_decay, tau_decay]
                        A_plan, tau_rise, A_exec, sigma, A_decay, tau_decay = popt

                        fit_records.append({
                            "Project": project_name,
                            "Perturbation": strategy,
                            "Model": model_type,
                            "Drug": drug,
                            "Phase": phase,
                            "Region": region,
                            "Genotype": genotype,
                            "A_plan": A_plan,
                            "tau_rise_ms": tau_rise * 1000.0,
                            "A_exec": A_exec,
                            "sigma_ms": sigma * 1000.0,
                            "A_decay": A_decay,
                            "tau_decay_ms": tau_decay * 1000.0
                        })

                        lbl = f"{genotype} Fit (tau_rise={tau_rise*1000:.1f}ms, tau_decay={tau_decay*1000:.1f}ms)"
                        plt.plot(t_ms, y_fit, color=c, lw=3, label=lbl)
                    else:
                        plt.plot(t_ms, y_emp, color=c, linestyle=":", lw=2, label=f"{genotype} (Empirical)")

                plt.axhline(1.0, color='black', linestyle=':', label='Chance Baseline')
                plt.axvline(0.0, color='gray', linestyle='--', linewidth=1.5, label='Syllable Onset')

                plt.xlabel("Time Relative to Onset (ms)", fontsize=11, fontweight="bold")
                plt.ylabel("Decoding Accuracy (Fold over shuffle)", fontsize=11, fontweight="bold")

                title = f"Biophysical Saliency Fit: Syllable Decoding Horizon\nGroup: {drug} | Phase: {phase.upper()} | Region: {region} | Strategy: {strategy}"
                plt.title(title, fontsize=12, fontweight="bold", pad=15)

                plt.legend(frameon=True, loc="upper left")
                sns.despine()
                plt.tight_layout()

                out_path = os.path.join(OUTPUT_DIR, f"{project_name.lower()}_saliency_fit_{strategy}_{model_type}_{drug.lower()}_{phase.lower()}_{region.lower()}.png")
                plt.savefig(out_path, dpi=300)
                plt.close()
                print(f"    Saved biophysical fit plot: {out_path}")

    if fit_records:
        df_fits = pd.DataFrame(fit_records)
        fit_csv = os.path.join(OUTPUT_DIR, f"{project_name.lower()}_fitted_biophysical_params_{strategy}_{model_type}.csv")
        df_fits.to_csv(fit_csv, index=False)
        print(f"  Fitted biophysical parameters exported to: {fit_csv}")


def plot_perturbation_comparison(project_name, strategies=["none", "residual", "synaptic_filter", "jitter", "spectral", "naive_permutation"], model_type="rfc"):
    """
    Generates comparative line plots comparing the saliency curves across different
    perturbation strategies side-by-side or overlaid for WT and KI, DS and VS.
    """
    print(f"  Generating perturbation comparison plots for {project_name}...")

    # Load the CSV results for all strategies
    dfs = []
    for strategy in strategies:
        csv_out = os.path.join(OUTPUT_DIR, f"{project_name.lower()}_saliency_modeling_{strategy}_{model_type}.csv")
        if os.path.exists(csv_out):
            df = pd.read_csv(csv_out)
            df["Strategy"] = strategy
            dfs.append(df)

    if not dfs:
        print("    No results found to compare.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    drugs = sorted(df_all["Drug"].unique())
    phases = sorted(df_all["Phase"].unique())
    regions = sorted(df_all["Region"].unique())

    for drug in drugs:
        for phase in phases:
            for region in regions:
                sub = df_all[(df_all["Drug"] == drug) & (df_all["Phase"] == phase) & (df_all["Region"] == region)]
                if sub.empty: continue

                fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

                # Colors for strategies
                strat_colors = {
                    "none": "#34495e",          # Sleek charcoal/black
                    "residual": "#e67e22",      # Warm orange
                    "synaptic_filter": "#2980b9",# Cool blue
                    "jitter": "#9b59b6",        # Elegant purple
                    "spectral": "#1abc9c",      # Turquoise
                    "naive_permutation": "#e74c3c", # Vibrant red
                }

                for idx, genotype in enumerate(["WT", "KI"]):
                    ax = axes[idx]
                    g_sub = sub[sub["Genotype"] == genotype]
                    if g_sub.empty: continue

                    for strat in strategies:
                        s_sub = g_sub[g_sub["Strategy"] == strat].sort_values("Time_Center_ms")
                        if s_sub.empty: continue

                        t_ms = s_sub["Time_Center_ms"].to_numpy()
                        y_emp = s_sub["Saliency_Fold"].to_numpy()
                        sem_emp = s_sub["Saliency_SEM"].to_numpy()

                        c = strat_colors.get(strat, "blue")
                        ax.plot(t_ms, y_emp, color=c, lw=2.5, marker='o', ms=5, label=f"Strategy: {strat.upper()}")
                        ax.fill_between(t_ms, y_emp - 1.96 * sem_emp, y_emp + 1.96 * sem_emp, color=c, alpha=0.1)

                    ax.axhline(1.0, color='black', linestyle=':', label='Chance Baseline')
                    ax.axvline(0.0, color='gray', linestyle='--', linewidth=1.5, label='Syllable Onset')

                    ax.set_xlabel("Time Relative to Onset (ms)", fontsize=11, fontweight="bold")
                    if idx == 0:
                        ax.set_ylabel("Decoding Accuracy (Fold over shuffle)", fontsize=11, fontweight="bold")

                    ax.set_title(f"Genotype: {genotype}", fontsize=12, fontweight="bold")
                    ax.legend(frameon=True, loc="upper left", fontsize=9)
                    sns.despine(ax=ax)

                plt.suptitle(f"Biophysical Perturbation Comparison: {drug} | {phase.upper()} | {region}\nModel: {model_type.upper()}", fontsize=14, fontweight="bold", y=0.98)
                plt.tight_layout()

                out_path = os.path.join(OUTPUT_DIR, f"{project_name.lower()}_perturbation_comparison_{drug.lower()}_{phase.lower()}_{region.lower()}.png")
                plt.savefig(out_path, dpi=300)
                plt.close()
                print(f"    Saved perturbation comparison plot: {out_path}")
