"""
Unique Information Pipeline (Idea 1).

Isolates striatal dopamine dynamics from continuous posture and velocity kinematics
to determine if DA encodes high-level behavioral syllables over low-level motor physics.
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.stats import binned_statistic_2d
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

# Try to import GPU-accelerated cuML models, otherwise fallback to CPU scikit-learn
USING_CUML = False
try:
    from cuml.ensemble import RandomForestClassifier as cumlRFClassifier
    from cuml.ensemble import RandomForestRegressor as cumlRFRegressor
    from cuml.linear_model import LinearRegression as cumlLinearRegression
    USING_CUML = True
    print("\n" + "="*70)
    print(" SUCCESS: GPU-Accelerated cuML Random Forests and OLS Activated!")
    print("="*70 + "\n")
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as sklearnRFClassifier
    from sklearn.ensemble import RandomForestRegressor as sklearnRFRegressor
    from sklearn.linear_model import LinearRegression as sklearnLinearRegression

import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor as sklearnGBRegressor
from sklearn.linear_model import LinearRegression as sklearnLinearRegression
from sklearn.neural_network import MLPRegressor as sklearnMLPRegressor


def _train_pytorch_mlp(X_train, y_train, X_all, epochs=15, batch_size=256, lr=0.01):
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    net = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    net.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = net(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    net.eval()
    X_all_t = torch.tensor(X_all, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = net(X_all_t).cpu().numpy().flatten()

    # Calculate training R^2
    X_train_device = X_train_t.to(device)
    with torch.no_grad():
        train_preds = net(X_train_device).cpu().numpy().flatten()
    ss_res = np.sum((y_train - train_preds) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))

    return preds, r2


def notify_ntfy(message):
    import urllib.request
    try:
        url = "https://ntfy.sh/woody_videos_seb"
        req = urllib.request.Request(url, data=message.encode('utf-8'), method='POST')
        with urllib.request.urlopen(req, timeout=10) as response:
            pass
    except Exception as e:
        print(f"Warning: Failed to send ntfy notification. {e}")

# Suppress warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output/unique_information")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "grids"), exist_ok=True)


def _get_top_syllables(sessions, k=20):
    from collections import Counter
    c = Counter()
    for s in sessions:
        if not hasattr(s, 'all_data_df') or s.all_data_df.empty: continue
        # Contiguous bouts or raw instances
        b = s.all_data_df["syllable"].dropna()
        b = b[b != -1]
        c.update(b.to_dict().values())
    return [x[0] for x in c.most_common(k)]


def _get_functional_channels(s):
    """Identifies photometry channels and returns standard metadata."""
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


def plot_kinematic_grid(sessions, project_name):
    """
    Plots the average Dopamine signal across a 2D Kinematic Grid
    (Velocity vs. Dominant Postural Component: Latent State 0) to
    visualize if DA release maps to physical movement space.
    """
    print(f"\n--- Generating Kinematic Grids ({project_name}) ---")

    sample_s = sessions[0]
    syll_df_sample = sample_s.syllables
    if isinstance(syll_df_sample, dict): syll_df_sample = list(syll_df_sample.values())[0]

    latent_cols = [c for c in syll_df_sample.columns if 'latent_state' in c]
    if not latent_cols:
        latent_cols = [f"latent_state_{i}" for i in range(10)]
    else:
        latent_cols = sorted(latent_cols, key=lambda x: int(x.split('_')[-1]))

    dominant_latent = latent_cols[0] # latent_state_0

    # Check if all grid plots already exist to avoid redundant processing
    all_exist = True
    for drug_lbl in ["Saline", "Amphetamine"]:
        for reg in ["VS", "DS"]:
            out_path = os.path.join(OUTPUT_DIR, "grids", f"{project_name.lower()}_{reg.lower()}_{drug_lbl.lower()}_grid.png")
            if not os.path.exists(out_path):
                all_exist = False
                break
    if all_exist:
        print("  All kinematic grid plots already exist. Skipping grid generation.")
        return

    # Standardize drugs
    drugs = ["Saline", "Amphetamine"]
    regions = ["VS", "DS"]

    for drug_lbl in drugs:
        for reg in regions:
            print(f"  Aggregating grid data for {reg} under {drug_lbl}...")

            all_v = []
            all_l = []
            all_da = []

            for s in sessions:
                drug = s.metadata.get("drug", "N/A")
                if drug in ["N/A", None]: drug = "Saline"
                if drug_lbl == "Amphetamine" and not drug.lower().startswith("amph"): continue
                if drug_lbl == "Saline" and not drug.lower().startswith("sal") and drug != "Saline": continue

                df = s.all_data_df.copy()
                if df.empty: continue
                if "velocity_px_s" not in df.columns: continue

                # Fetch channels matching region
                channels = _get_functional_channels(s)
                channels = [ch for ch in channels if ch["broad_reg"] == reg]
                if not channels: continue

                # Load syllables
                syll_df = s.syllables
                if isinstance(syll_df, dict): syll_df = list(syll_df.values())[0]

                merged = df.join(syll_df[latent_cols])

                for ch in channels:
                    zcol = ch["zcol"]

                    # Drop NaNs
                    valid = merged[[zcol, "velocity_px_s", dominant_latent]].dropna()
                    if valid.empty: continue

                    all_v.extend(valid["velocity_px_s"].values)
                    all_l.extend(valid[dominant_latent].values)
                    all_da.extend(valid[zcol].values)

            if not all_v:
                print(f"    No data found for {reg} under {drug_lbl}.")
                continue

            all_v = np.array(all_v)
            all_l = np.array(all_l)
            all_da = np.array(all_da)

            # Log transform velocity to compress tail and handle skew
            log_v = np.log1p(all_v)

            # 2D Binned Statistics
            try:
                bin_means, x_edges, y_edges, _ = binned_statistic_2d(
                    log_v, all_l, all_da, statistic='mean', bins=35
                )
                bin_counts, _, _, _ = binned_statistic_2d(
                    log_v, all_l, all_da, statistic='count', bins=35
                )

                # Mask bins with fewer than 5 frames to avoid noisy outliers
                bin_means[bin_counts < 5] = np.nan

                # Smooth the grid slightly for visual clarity
                from scipy.ndimage import gaussian_filter
                # Temporarily replace nan with 0 for smoothing
                nan_mask = np.isnan(bin_means)
                smoothed = bin_means.copy()
                smoothed[nan_mask] = 0
                smoothed = gaussian_filter(smoothed, sigma=0.8)
                smoothed[nan_mask] = np.nan

                plt.figure(figsize=(8, 6))
                extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

                im = plt.imshow(
                    smoothed.T, origin='lower', extent=extent,
                    aspect='auto', cmap='magma', interpolation='nearest'
                )
                cbar = plt.colorbar(im)
                cbar.set_label("Average Dopamine Z-Score", rotation=270, labelpad=15)

                plt.title(f"2D Kinematic Grid: {reg} Dopamine ({drug_lbl})\n(Dopamine Release vs. Posture & Velocity)")
                plt.xlabel("Log(Velocity + 1) (px/s)")
                plt.ylabel(f"Dominant Postural Component ({dominant_latent})")

                sns.despine()
                plt.tight_layout()

                out_path = os.path.join(OUTPUT_DIR, "grids", f"{project_name.lower()}_{reg.lower()}_{drug_lbl.lower()}_grid.png")
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()

            except Exception as e:
                print(f"    Error plotting grid: {e}")


def strip_kinematics(sessions, model_type="random_forest"):
    """
    Performs per-session temporal sliding-window kinematic residualization using a specified model.
    Trains the model (Linear, RandomForest, Gradient Boosting, or MLP) to predict continuous DA traces from a temporal window (±0.5s) of 
    exactly 4 physical features: linear velocity, linear acceleration, angular velocity, and angular acceleration.
    Saves the continuous predicted DA trace and the kinematic-free residuals in the session dataframe.
    """
    print(f"\n--- Running Temporal Sliding-Window {model_type.upper()} Regression (Per-Session) ---")

    cache_dir = os.path.join(OUTPUT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    for s_idx, s in enumerate(sessions):
        print(f"  Processing Session {s_idx+1}/{len(sessions)}: {s.session_id}")

        df = s.all_data_df.copy()
        if df.empty: continue
        if "velocity_px_s" not in df.columns: continue

        channels = _get_functional_channels(s)
        if not channels: continue

        # Check angular velocity or heading
        if "angular_velocity" in df.columns:
            ang_col = "angular_velocity"
        elif "heading" in df.columns:
            # Simple delta heading
            dh = np.concatenate(([0], np.diff(df["heading"].values)))
            df["delta_heading"] = dh
            ang_col = "delta_heading"
        else:
            df["zeros_ang"] = 0
            ang_col = "zeros_ang"

        # Compute the 4 physical features
        df["lin_vel"] = df["velocity_px_s"].fillna(0)
        df["lin_acc"] = df["lin_vel"].diff().fillna(0)
        df["ang_vel"] = df[ang_col].fillna(0)
        df["ang_acc"] = df["ang_vel"].diff().fillna(0)

        # Write to s.all_data_df immediately so they exist under cached loads
        for col in ["lin_vel", "lin_acc", "ang_vel", "ang_acc"]:
            s.all_data_df[col] = df[col].values

        cache_path = os.path.join(cache_dir, f"{s.session_id}_{model_type}.parquet")

        # Check cache
        if os.path.exists(cache_path):
            try:
                cache_df = pd.read_parquet(cache_path)
                expected_cols = []
                for ch in channels:
                    zcol = ch["zcol"]
                    expected_cols.extend([f"{zcol}_residual_kinematic_free", f"{zcol}_predicted_kinematics"])

                if all(col in cache_df.columns for col in expected_cols) and "frame_index" in cache_df.columns:
                    print(f"    Loaded residuals from cache: {s.session_id}_{model_type}.parquet")
                    orig_index = s.all_data_df.index
                    s.all_data_df = s.all_data_df.drop(columns=[col for col in expected_cols if col in s.all_data_df.columns], errors="ignore")
                    s.all_data_df = pd.merge(s.all_data_df, cache_df[["frame_index"] + expected_cols], on="frame_index", how="left")
                    s.all_data_df.index = orig_index
                    continue
            except Exception as e:
                print(f"    Cache load failed ({e}), recalculating regressions...")

        # Get sampling rate
        fs = s.metadata.get("fs", 80.0)
        window_s = 0.5
        half_window = int(window_s * fs) # e.g. 40 frames

        base_features = ["lin_vel", "lin_acc", "ang_vel", "ang_acc"]

        # Construct the temporal sliding window features using pre-allocated NumPy array for high performance
        n_features = len(base_features) * (2 * half_window + 1)
        X_np = np.zeros((len(df), n_features), dtype=np.float32)
        shifted_names = []
        col_idx = 0

        for feat in base_features:
            arr = df[feat].values
            for shift in range(-half_window, half_window + 1):
                shifted_names.append(f"{feat}_shift_{shift}")
                if shift > 0:
                    X_np[shift:, col_idx] = arr[:-shift]
                elif shift < 0:
                    X_np[:shift, col_idx] = arr[-shift:]
                else:
                    X_np[:, col_idx] = arr
                col_idx += 1

        # Combine into DataFrame in a single step preserving the original index
        X_df = pd.DataFrame(X_np, columns=shifted_names, index=df.index)

        # Train regressor on each photometry channel
        for ch in channels:
            zcol = ch["zcol"]

            # Ensure target z-score is finite and not NaN
            valid_mask = ~df[zcol].isna() & np.isfinite(df[zcol])

            # Create a robust boundary mask using integer position indexing
            boundary_mask = np.zeros(len(df), dtype=bool)
            boundary_mask[half_window : len(df) - half_window] = True
            valid_mask = valid_mask & boundary_mask

            if valid_mask.sum() < 100: continue

            X = X_df.loc[valid_mask].to_numpy()
            y = df.loc[valid_mask, zcol].to_numpy()

            # Subsample training data to a maximum of 40,000 frames to make training extremely fast
            if len(X) > 40000:
                step = len(X) // 40000
                X_train = X[::step]
                y_train = y[::step]
            else:
                X_train = X
                y_train = y

            # Model selection and regression training
            if model_type == "linear":
                if USING_CUML:
                    X_train_c = np.asarray(X_train, dtype=np.float32)
                    y_train_c = np.asarray(y_train, dtype=np.float32)
                    X_c = np.asarray(X, dtype=np.float32)

                    reg = cumlLinearRegression()
                    reg.fit(X_train_c, y_train_c)
                    y_pred = np.asarray(reg.predict(X_c))
                    r2 = reg.score(X_train_c, y_train_c)
                else:
                    reg = sklearnLinearRegression()
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X)
                    r2 = reg.score(X_train, y_train)

            elif model_type == "random_forest":
                if USING_CUML:
                    # Cast to float32 for high performance cuML GPU execution
                    X_train_c = np.asarray(X_train, dtype=np.float32)
                    y_train_c = np.asarray(y_train, dtype=np.float32)
                    X_c = np.asarray(X, dtype=np.float32)

                    reg = cumlRFRegressor(n_estimators=30, max_depth=8, random_state=42)
                    reg.fit(X_train_c, y_train_c)

                    # Predict on the full valid mask
                    y_pred = np.asarray(reg.predict(X_c))
                    r2 = reg.score(X_train_c, y_train_c)
                else:
                    reg = sklearnRFRegressor(
                        n_estimators=30, max_depth=8, random_state=42, n_jobs=-1
                    )
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X)
                    r2 = reg.score(X_train, y_train)

            elif model_type == "gradient_boosting":
                # Check if we can run GPU-accelerated XGBoost
                try:
                    X_train_c = np.asarray(X_train, dtype=np.float32)
                    y_train_c = np.asarray(y_train, dtype=np.float32)
                    X_c = np.asarray(X, dtype=np.float32)

                    reg = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=30, max_depth=5, random_state=42)
                    reg.fit(X_train_c, y_train_c)
                    y_pred = reg.predict(X_c)
                    r2 = reg.score(X_train_c, y_train_c)
                except Exception as e:
                    print(f"      Warning: GPU XGBoost failed ({e}). Falling back to CPU...")
                    reg = sklearnGBRegressor(n_estimators=30, max_depth=5, random_state=42)
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X)
                    r2 = reg.score(X_train, y_train)

            elif model_type == "mlp":
                has_torch = False
                try:
                    import torch
                    has_torch = torch.cuda.is_available()
                except ImportError:
                    pass

                if has_torch:
                    # Train on GPU using PyTorch for high performance
                    y_pred, r2 = _train_pytorch_mlp(X_train, y_train, X)
                else:
                    reg = sklearnMLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=42)
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X)
                    r2 = reg.score(X_train, y_train)
            else:
                raise ValueError(f"Unknown stripping model: {model_type}")

            # Calculate residuals
            residuals = y - y_pred

            # Save back to s.all_data_df using index mapping
            res_col = f"{zcol}_residual_kinematic_free"
            pred_col = f"{zcol}_predicted_kinematics"

            s.all_data_df[res_col] = np.nan
            s.all_data_df[pred_col] = np.nan

            s.all_data_df.loc[valid_mask, res_col] = residuals
            s.all_data_df.loc[valid_mask, pred_col] = y_pred

            # Save R^2 score for diagnostic logging
            print(f"    Channel {zcol} ({ch['broad_reg']}): Kinematic R^2 = {r2:.4f}")

        # Save generated residuals and predictions to Parquet Cache
        try:
            cache_cols = ["frame_index"]
            for ch in channels:
                zcol = ch["zcol"]
                cache_cols.extend([f"{zcol}_residual_kinematic_free", f"{zcol}_predicted_kinematics"])
            cols_to_save = [c for c in cache_cols if c in s.all_data_df.columns]
            if len(cols_to_save) > 1:
                s.all_data_df[cols_to_save].to_parquet(cache_path)
                print(f"    Saved residuals to cache: {s.session_id}_{model_type}.parquet")
        except Exception as e:
            print(f"    Failed to save cache: {e}")

        notify_ntfy(f"📊 Session Regression Complete ({s_idx+1}/{len(sessions)}): {s.session_id}")


def _evaluate_rfc(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs = []
    shufs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if USING_CUML:
            # Cast variables to float32 and int32 for high performance cuML GPU execution
            X_train_c = np.asarray(X_train, dtype=np.float32)
            y_train_c = np.asarray(y_train, dtype=np.int32)
            X_test_c = np.asarray(X_test, dtype=np.float32)
            y_test_c = np.asarray(y_test, dtype=np.int32)

            clf = cumlRFClassifier(n_estimators=50, max_depth=10, random_state=42)
            clf.fit(X_train_c, y_train_c)
            accs.append(clf.score(X_test_c, y_test_c))

            y_shuf = np.random.permutation(y_train_c)
            clf.fit(X_train_c, y_shuf)
            shufs.append(clf.score(X_test_c, y_test_c))
        else:
            clf = sklearnRFClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            accs.append(clf.score(X_test, y_test))

            y_shuf = np.random.permutation(y_train)
            clf.fit(X_train, y_shuf)
            shufs.append(clf.score(X_test, y_test))

    return np.mean(accs), np.mean(shufs)


def _decode_single_group(group_key, df_raw, df_res, df_pred, df_latent, project_name, dir_2bars, dir_3bars, dir_4bars, dir_5bars, OUTPUT_DIR, groups, model_type="random_forest"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub_raw = df_raw[df_raw["Group_Key"] == group_key]
    sub_res = df_res[df_res["Group_Key"] == group_key]
    sub_pred = df_pred[df_pred["Group_Key"] == group_key]
    sub_latent = df_latent[df_latent["Group_Key"] == group_key]

    if len(sub_raw) < 50:
        print(f"  Skipping {group_key}: not enough samples ({len(sub_raw)}).")
        return None

    X_raw = np.vstack(sub_raw["Trace"].values)
    y_raw = sub_raw["Syllable"].values

    X_res = np.vstack(sub_res["Trace"].values)
    y_res = sub_res["Syllable"].values

    X_pred = np.vstack(sub_pred["Trace"].values)
    y_pred = sub_pred["Syllable"].values

    X_latent = np.vstack(sub_latent["Trace"].values)
    y_latent = sub_latent["Syllable"].values

    # Filter to only syllables present in groups
    valid_indices = [i for i, s in enumerate(y_raw) if s in groups]
    if len(valid_indices) < 50:
        print(f"  Skipping {group_key}: not enough valid group-mapped samples ({len(valid_indices)}).")
        return None

    X_raw = X_raw[valid_indices]
    y_raw = y_raw[valid_indices]
    X_res = X_res[valid_indices]
    y_res = y_res[valid_indices]
    X_pred = X_pred[valid_indices]
    y_pred = y_pred[valid_indices]
    X_latent = X_latent[valid_indices]
    y_latent = y_latent[valid_indices]

    y_group = np.array([groups[s] for s in y_raw])

    if len(np.unique(y_raw)) < 2 or len(np.unique(y_group)) < 2:
        print(f"  Skipping {group_key}: less than 2 classes in syllable identity or kinematic group.")
        return None

    MAX_SAMPLES = 4000
    if len(X_raw) > MAX_SAMPLES:
        indices = np.arange(len(X_raw))
        sampled_indices = resample(indices, n_samples=MAX_SAMPLES, random_state=42, stratify=y_raw)

        X_raw = X_raw[sampled_indices]
        y_raw = y_raw[sampled_indices]
        X_res = X_res[sampled_indices]
        y_res = y_res[sampled_indices]
        X_pred = X_pred[sampled_indices]
        y_pred = y_pred[sampled_indices]
        X_latent = X_latent[sampled_indices]
        y_latent = y_latent[sampled_indices]
        y_group = y_group[sampled_indices]

    print(f"  Training RFC classifiers on {len(X_raw)} instances ({group_key})...")
    acc_raw, shuf_raw = _evaluate_rfc(X_raw, y_raw)
    acc_res, shuf_res = _evaluate_rfc(X_res, y_res)
    acc_pred, shuf_pred = _evaluate_rfc(X_pred, y_pred)
    acc_latent, shuf_latent = _evaluate_rfc(X_latent, y_latent)
    acc_group, shuf_group = _evaluate_rfc(X_raw, y_group)

    fold_raw = acc_raw / shuf_raw if shuf_raw > 0 else 0
    fold_res = acc_res / shuf_res if shuf_res > 0 else 0
    fold_pred = acc_pred / shuf_pred if shuf_pred > 0 else 0
    fold_latent = acc_latent / shuf_latent if shuf_latent > 0 else 0
    fold_group = acc_group / shuf_group if shuf_group > 0 else 0

    print(f"    [{group_key}] Raw RFC: {fold_raw:.2f}x | Residual RFC: {fold_res:.2f}x | Kinematics (4 Features) RFC: {fold_pred:.2f}x | Latent States: {fold_latent:.2f}x | Kinematic Group: {fold_group:.2f}x")

    # Custom 3-Bar Plot: Raw DA (with Kinematic portion greyed out) | Pure Kinematics | Latent States (Truncated Ceiling)
    unique_part = min(fold_raw, fold_res)
    kinematic_explained = max(0.0, fold_raw - fold_res)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Set y-limit based on raw, residual and prediction to avoid squashing them
    y_max_data = max(fold_raw, fold_res, fold_pred)
    y_lim = y_max_data + 1.2
    ax.set_ylim(0, y_lim)

    width = 0.4

    # 1. Bar 1: Raw Dopamine (Stacked: Unique DA in Coral, Kinematic-Explained in Gray)
    ax.bar("Raw Dopamine", unique_part, width=width, color="#ff7675", edgecolor="black", label="Unique DA Info")
    if kinematic_explained > 0:
        ax.bar("Raw Dopamine", kinematic_explained, bottom=unique_part, width=width,
               color="#dcdde1", edgecolor="black", hatch="//", label="Kinematic-Explained")

    # Add text label at the top of Bar 1 showing total Raw DA
    ax.text("Raw Dopamine", fold_raw + 0.05, f"{fold_raw:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Bar 2: Pure Kinematics (4 Features)
    ax.bar("Pure Kinematics\n(4 Features)", fold_pred, width=width, color="#0984e3", edgecolor="black", label="Pure Kinematics")
    ax.text("Pure Kinematics\n(4 Features)", fold_pred + 0.05, f"{fold_pred:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3. Bar 3: Latent States (Truncated Ceiling)
    latent_height = y_lim * 0.82
    ax.bar("Postural\nLatent States", latent_height, width=width, color="#a29bfe", edgecolor="black", alpha=0.9, label="Postural Ceiling")

    # Axis break symbol '//' inside the bar to denote truncation
    ax.text("Postural\nLatent States", latent_height * 0.90, "//", ha="center", va="center", fontsize=16, fontweight="bold", color="black")

    # Float the actual value on top of the truncated bar in bold purple
    ax.text("Postural\nLatent States", latent_height + 0.1, f"{fold_latent:.2f}x",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#6c5ce7")

    ax.axhline(1.0, color='black', linestyle='--', label='Chance (Shuffle)')
    ax.set_ylabel("Decoding Accuracy (Fold over shuffle)", fontsize=11, fontweight="bold")

    parts = group_key.split("_")
    if len(parts) == 4:
        drug_name, genotype, phase, region = parts
    else:
        drug_name, genotype, phase = parts
        region = "All"

    ax.set_title(f"Unique Behavioral Info vs. Kinematic Vigor ({project_name})\nGroup: {drug_name} | {genotype} | Phase: {phase.upper()} | Region: {region}",
                 fontsize=12, fontweight="bold", pad=15)
    sns.despine(ax=ax)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    # Sub-sub directories for ds, vs, both, all
    dir_2bars_sub = os.path.join(dir_2bars, region.upper())
    dir_3bars_sub = os.path.join(dir_3bars, region.upper())
    dir_4bars_sub = os.path.join(dir_4bars, region.upper())
    dir_5bars_sub = os.path.join(dir_5bars, region.upper())

    # Ensure all actual subdirectories exist
    os.makedirs(dir_2bars_sub, exist_ok=True)
    os.makedirs(dir_3bars_sub, exist_ok=True)
    os.makedirs(dir_4bars_sub, exist_ok=True)
    os.makedirs(dir_5bars_sub, exist_ok=True)

    out_path_3bar_sub = os.path.join(dir_3bars_sub, f"{project_name.lower()}_fig_unique_info_{group_key.lower()}_{model_type}.png")
    plt.savefig(out_path_3bar_sub, dpi=300)
    plt.close()

    # Plot 2-bar comparison (Individual Syllable vs Kinematic Group)
    plt.figure(figsize=(6, 5))
    plt.bar(
        ["Individual Syllable", "Kinematic Group"],
        [fold_raw, fold_group],
        color=["#dd8452", "#4c72b0"],
        edgecolor="black",
        width=0.4
    )
    plt.axhline(1.0, color='black', linestyle='--', label='Chance (Shuffle)')
    plt.ylabel("Prediction Accuracy (Fold over shuffle)")
    plt.title(f"Syllable Identity vs. Kinematic Group Decoding ({project_name})\nGroup: {drug_name} | {genotype} | Phase: {phase.upper()} | Region: {region}")
    sns.despine()
    plt.tight_layout()
    out_path_2bar = os.path.join(dir_2bars_sub, f"{project_name.lower()}_fig_unique_info_2bars_{group_key.lower()}_{model_type}.png")
    plt.savefig(out_path_2bar, dpi=300)
    plt.close()

    # Plot 4-bar Calibrated Comparison (with Latent States Upper Bound)
    plt.figure(figsize=(9, 5.5))
    plt.bar(
        ["Raw Dopamine", "Residual\n(Kinematic-Free)", "Kinematics\n(4 Features)", "Latent States\n(Upper Bound)"],
        [fold_raw, fold_res, fold_pred, fold_latent],
        color=["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"],
        edgecolor="black",
        width=0.5
    )
    plt.axhline(1.0, color='black', linestyle='--', label='Chance (Shuffle)')
    plt.ylabel("Decoding Accuracy (Fold over shuffle)")
    plt.title(f"Syllable Decoding & Postural Ceiling ({project_name})\nGroup: {drug_name} | {genotype} | Phase: {phase.upper()} | Region: {region}")
    sns.despine()
    plt.tight_layout()

    out_path_4bar = os.path.join(dir_4bars_sub, f"{project_name.lower()}_fig_unique_info_4bars_{group_key.lower()}_{model_type}.png")
    plt.savefig(out_path_4bar, dpi=300)
    plt.close()

    # Plot 5-bar Calibrated Comparison (with Kinematic Group and Latent States Upper Bound)
    plt.figure(figsize=(10, 5.5))
    plt.bar(
        ["Raw Dopamine", "Residual\n(Kinematic-Free)", "Kinematics\n(4 Features)", "Latent States\n(Upper Bound)", "Kinematic Group"],
        [fold_raw, fold_res, fold_pred, fold_latent, fold_group],
        color=["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#e67e22"],
        edgecolor="black",
        width=0.5
    )
    plt.axhline(1.0, color='black', linestyle='--', label='Chance (Shuffle)')
    plt.ylabel("Decoding Accuracy (Fold over shuffle)")
    plt.title(f"Syllable Decoding & Postural Ceiling ({project_name})\nGroup: {drug_name} | {genotype} | Phase: {phase.upper()} | Region: {region}")
    sns.despine()
    plt.tight_layout()

    out_path_5bar = os.path.join(dir_5bars_sub, f"{project_name.lower()}_fig_unique_info_5bars_{group_key.lower()}_{model_type}.png")
    plt.savefig(out_path_5bar, dpi=300)
    plt.close()

    return {
        "group_key": group_key,
        "fold_raw": fold_raw,
        "fold_res": fold_res,
        "fold_pred": fold_pred,
        "fold_latent": fold_latent,
        "fold_group": fold_group
    }


def decode_residuals(sessions, project_name, model_type="random_forest"):
    """
    Extracts PETHs for raw, residual, and predicted DA, and postural latent states.
    Trains parallel RandomForestClassifiers on identical terms to decode syllables.
    Saves 3-bar and 4-bar figures into separate subdirectories to avoid axis compression
    and preserve legacy paths.
    """
    print(f"\n--- Training Syllable Classifiers on Residuals ({project_name}) ---")

    top_sylls = _get_top_syllables(sessions, k=20)
    if not top_sylls:
        print("No top syllables found.")
        return

    window_s = 0.5

    # Subdirectories to avoid axis compression
    dir_2bars = os.path.join(OUTPUT_DIR, "2_bars")
    dir_3bars = os.path.join(OUTPUT_DIR, "3_bars")
    dir_4bars = os.path.join(OUTPUT_DIR, "4_bars")
    dir_5bars = os.path.join(OUTPUT_DIR, "5_bars")
    os.makedirs(dir_2bars, exist_ok=True)
    os.makedirs(dir_3bars, exist_ok=True)
    os.makedirs(dir_4bars, exist_ok=True)
    os.makedirs(dir_5bars, exist_ok=True)

    # Discover postural latent states (10 dimensions)
    sample_s = sessions[0]
    syll_df_sample = sample_s.syllables
    if isinstance(syll_df_sample, dict): syll_df_sample = list(syll_df_sample.values())[0]

    latent_cols = [c for c in syll_df_sample.columns if 'latent_state' in c]
    if not latent_cols:
        latent_cols = [f"latent_state_{i}" for i in range(10)]
    else:
        latent_cols = sorted(latent_cols, key=lambda x: int(x.split('_')[-1]))

    # Standard PETH extraction helper for raw, residual, predicted, and latent states
    def _extract_traces(sessions, target_sylls):
        raw_traces = []
        res_traces = []
        pred_traces = []
        latent_traces = []

        for s in sessions:
            fs = s.metadata.get("fs", 80.0)
            window_frames = int(window_s * fs)

            df = s.all_data_df
            if "onset" not in df.columns: continue

            channels = _get_functional_channels(s)

            # Load syllables and join with df
            syll_df = s.syllables
            if isinstance(syll_df, dict): syll_df = list(syll_df.values())[0]

            # Forward-fill and back-fill the latent states to handle 30Hz to 80Hz alignment NaNs
            latents_aligned = syll_df[latent_cols].ffill().bfill()

            # Join the 10 latent state columns to the main dataframe
            df_w_latents = df.join(latents_aligned)
            df_indexed = df_w_latents.set_index("frame_index")

            # Select only the relevant continuous numerical columns to construct a pure float64 NumPy array
            kin_cols = ["lin_vel", "lin_acc", "ang_vel", "ang_acc"]
            numerical_cols = list(latent_cols) + kin_cols
            for ch in channels:
                zcol = ch["zcol"]
                res_col = f"{zcol}_residual_kinematic_free"
                pred_col = f"{zcol}_predicted_kinematics"
                for c in [zcol, res_col, pred_col]:
                    if c in df_indexed.columns and c not in numerical_cols:
                        numerical_cols.append(c)

            data_np = df_indexed[numerical_cols].to_numpy().astype(np.float64)
            col_to_idx = {col: i for i, col in enumerate(numerical_cols)}
            idx_values = df_indexed.index.values
            idx_to_pos = {idx: pos for pos, idx in enumerate(idx_values)}

            # Pre-resolve latent state column indices
            latent_idxs = [col_to_idx[c] for c in latent_cols if c in col_to_idx]
            if len(latent_idxs) != len(latent_cols): continue

            geno = s.metadata.get("genotype", "UNK").upper()
            drug = s.metadata.get("drug", "N/A")
            if drug in ["N/A", None]: drug = "Saline"
            # Normalize drug name
            if drug.lower().startswith("amph"): drug = "Amphetamine"
            elif drug.lower().startswith("sal"): drug = "Saline"

            inj_frame = s.metadata.get("injection_frame")

            for syll in target_sylls:
                onsets = df[(df["onset"] > 0) & (df["syllable"] == syll)]["frame_index"].values

                for onset in onsets:
                    start_idx = onset - window_frames
                    end_idx = onset + window_frames

                    if start_idx not in idx_to_pos or end_idx not in idx_to_pos: continue

                    start_pos = idx_to_pos[start_idx]
                    end_pos = idx_to_pos[end_idx] + 1  # +1 to make it inclusive like .loc

                    # Determine phase
                    phase = "post" if (inj_frame is not None and onset >= inj_frame) else "pre"

                    latent_arr = data_np[start_pos:end_pos, latent_idxs]
                    latent_flat = latent_arr.flatten()

                    kin_cols = ["lin_vel", "lin_acc", "ang_vel", "ang_acc"]
                    for ch in channels:
                        zcol = ch["zcol"]
                        res_col = f"{zcol}_residual_kinematic_free"

                        if zcol not in col_to_idx or res_col not in col_to_idx or not all(c in col_to_idx for c in kin_cols): continue

                        raw_arr = data_np[start_pos:end_pos, col_to_idx[zcol]]
                        res_arr = data_np[start_pos:end_pos, col_to_idx[res_col]]
                        pred_arr = data_np[start_pos:end_pos, [col_to_idx[c] for c in kin_cols]].flatten()

                        # Validate length and check for NaNs
                        expected_len = 2 * window_frames + 1
                        if len(raw_arr) != expected_len or len(res_arr) != expected_len or len(pred_arr) != expected_len * len(kin_cols) or len(latent_flat) != expected_len * len(latent_cols): continue
                        if np.isnan(raw_arr).any() or np.isnan(res_arr).any() or np.isnan(pred_arr).any() or np.isnan(latent_flat).any(): continue

                        raw_traces.append({
                            "Drug": drug,
                            "Genotype": geno,
                            "Phase": phase,
                            "Region": ch["broad_reg"],
                            "Syllable": syll,
                            "Trace": raw_arr
                        })
                        res_traces.append({
                            "Drug": drug,
                            "Genotype": geno,
                            "Phase": phase,
                            "Region": ch["broad_reg"],
                            "Syllable": syll,
                            "Trace": res_arr
                        })
                        pred_traces.append({
                            "Drug": drug,
                            "Genotype": geno,
                            "Phase": phase,
                            "Region": ch["broad_reg"],
                            "Syllable": syll,
                            "Trace": pred_arr
                        })
                        latent_traces.append({
                            "Drug": drug,
                            "Genotype": geno,
                            "Phase": phase,
                            "Region": ch["broad_reg"],
                            "Syllable": syll,
                            "Trace": latent_flat
                        })

                    # Process BOTH (concatenated DS and VS) if both exist in the session
                    ds_channels = [ch for ch in channels if ch["broad_reg"] == "DS"]
                    vs_channels = [ch for ch in channels if ch["broad_reg"] == "VS"]
                    if ds_channels and vs_channels:
                        ds_ch = ds_channels[0]
                        vs_ch = vs_channels[0]

                        ds_zcol = ds_ch["zcol"]
                        vs_zcol = vs_ch["zcol"]
                        ds_res_col = f"{ds_zcol}_residual_kinematic_free"
                        vs_res_col = f"{vs_zcol}_residual_kinematic_free"

                        if (ds_zcol in col_to_idx and ds_res_col in col_to_idx and
                            vs_zcol in col_to_idx and vs_res_col in col_to_idx and
                            all(c in col_to_idx for c in kin_cols)):

                            ds_raw = data_np[start_pos:end_pos, col_to_idx[ds_zcol]]
                            vs_raw = data_np[start_pos:end_pos, col_to_idx[vs_zcol]]
                            both_raw = np.concatenate([ds_raw, vs_raw])

                            ds_res = data_np[start_pos:end_pos, col_to_idx[ds_res_col]]
                            vs_res = data_np[start_pos:end_pos, col_to_idx[vs_res_col]]
                            both_res = np.concatenate([ds_res, vs_res])

                            ds_pred = data_np[start_pos:end_pos, [col_to_idx[c] for c in kin_cols]].flatten()
                            vs_pred = data_np[start_pos:end_pos, [col_to_idx[c] for c in kin_cols]].flatten()
                            both_pred = np.concatenate([ds_pred, vs_pred])

                            expected_len = 2 * window_frames + 1
                            if (len(ds_raw) == expected_len and len(vs_raw) == expected_len and
                                len(ds_res) == expected_len and len(vs_res) == expected_len and
                                not np.isnan(both_raw).any() and not np.isnan(both_res).any() and
                                not np.isnan(both_pred).any() and not np.isnan(latent_flat).any()):

                                raw_traces.append({
                                    "Drug": drug,
                                    "Genotype": geno,
                                    "Phase": phase,
                                    "Region": "BOTH",
                                    "Syllable": syll,
                                    "Trace": both_raw
                                })
                                res_traces.append({
                                    "Drug": drug,
                                    "Genotype": geno,
                                    "Phase": phase,
                                    "Region": "BOTH",
                                    "Syllable": syll,
                                    "Trace": both_res
                                })
                                pred_traces.append({
                                    "Drug": drug,
                                    "Genotype": geno,
                                    "Phase": phase,
                                    "Region": "BOTH",
                                    "Syllable": syll,
                                    "Trace": both_pred
                                })
                                latent_traces.append({
                                    "Drug": drug,
                                    "Genotype": geno,
                                    "Phase": phase,
                                    "Region": "BOTH",
                                    "Syllable": syll,
                                    "Trace": latent_flat
                                })
        return pd.DataFrame(raw_traces), pd.DataFrame(res_traces), pd.DataFrame(pred_traces), pd.DataFrame(latent_traces)

    print("  Extracting PETHs...")
    df_raw, df_res, df_pred, df_latent = _extract_traces(sessions, top_sylls)

    if df_raw.empty or df_res.empty or df_pred.empty or df_latent.empty:
        print("  Error: No valid traces extracted for classification.")
        return

    # Filter out UNK genotype and UNK brain regions
    df_raw = df_raw[~df_raw["Genotype"].astype(str).str.upper().str.contains("UNK") & ~df_raw["Region"].astype(str).str.upper().str.contains("UNK")]
    df_res = df_res[~df_res["Genotype"].astype(str).str.upper().str.contains("UNK") & ~df_res["Region"].astype(str).str.upper().str.contains("UNK")]
    df_pred = df_pred[~df_pred["Genotype"].astype(str).str.upper().str.contains("UNK") & ~df_pred["Region"].astype(str).str.upper().str.contains("UNK")]
    df_latent = df_latent[~df_latent["Genotype"].astype(str).str.upper().str.contains("UNK") & ~df_latent["Region"].astype(str).str.upper().str.contains("UNK")]

    # Fast vectorized string concatenation to build Group_Key
    for df_tr in [df_raw, df_res, df_pred, df_latent]:
        df_tr["Group_Key"] = df_tr["Drug"].astype(str) + "_" + df_tr["Genotype"].astype(str) + "_" + df_tr["Phase"].astype(str) + "_" + df_tr["Region"].astype(str)

    unique_groups = sorted(df_raw["Group_Key"].unique())

    # Compute kinematic group mappings
    from analysis.ulrik_kinematics.lib.kinematic_similarity import get_kinematic_similarity
    from scipy.cluster import hierarchy

    print("  Computing kinematic similarity for group mappings...")
    dist_matrix, valid_sylls, Z = get_kinematic_similarity(sessions, project_name, top_sylls)
    if Z is not None and len(valid_sylls) > 0:
        labels = hierarchy.fcluster(Z, 7, criterion='maxclust')
        groups = {valid_sylls[i]: labels[i] for i in range(len(valid_sylls))}
    else:
        groups = {}

    # Run subgroup decodings sequentially, allowing internal RandomForestClassifiers to use all CPU cores
    for group_key in unique_groups:
        _decode_single_group(
            group_key, df_raw, df_res, df_pred, df_latent, project_name,
            dir_2bars, dir_3bars, dir_4bars, dir_5bars, OUTPUT_DIR, groups, model_type=model_type
        )


def run_pipeline(sessions, project_name, model_type="random_forest"):
    """Entry point to run the entire Unique Information Pipeline."""
    print("\n=======================================================")
    print(f" STARTING UNIQUE INFORMATION PIPELINE: {project_name} ({model_type.upper()})")
    print("=======================================================")

    # 1. 2D Kinematic Grids
    plot_kinematic_grid(sessions, project_name)

    # 2. Per-Session Non-Linear Kinematic Residualization
    strip_kinematics(sessions, model_type=model_type)

    # 3. Residual RFC Decoding Comparison
    decode_residuals(sessions, project_name, model_type=model_type)

    print(f"\nCompleted Unique Information Pipeline for {project_name}.")
    print("Output files saved to: output/unique_information/")
