"""
Unique Information Pipeline (Idea 1).

Isolates striatal dopamine dynamics from continuous posture and velocity kinematics
to determine if DA encodes high-level behavioral syllables over low-level motor physics.
"""
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import warnings

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
            if zcol in df.columns:
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


def strip_kinematics(sessions):
    """
    Performs per-session non-linear kinematic residualization.
    Trains a RandomForestRegressor to predict frame-by-frame DA traces
    using MoSeq latent states, velocity, angular velocity, and their first-order derivatives.
    Extracts and stores the kinematic-free residual in the session dataframe.
    """
    print(f"\n--- Running Non-Linear Kinematic Residualization (Per-Session) ---")
    
    for s_idx, s in enumerate(sessions):
        print(f"  Processing Session {s_idx+1}/{len(sessions)}: {s.session_id}")
        
        df = s.all_data_df.copy()
        if df.empty: continue
        if "velocity_px_s" not in df.columns: continue
        
        channels = _get_functional_channels(s)
        if not channels: continue
        
        syll_df = s.syllables
        if isinstance(syll_df, dict): syll_df = list(syll_df.values())[0]
        
        latent_cols = [c for c in syll_df.columns if 'latent_state' in c]
        if not latent_cols:
            latent_cols = [f"latent_state_{i}" for i in range(10)]
        else:
            latent_cols = sorted(latent_cols, key=lambda x: int(x.split('_')[-1]))
            
        merged = df.join(syll_df[latent_cols])
        
        # Check angular velocity or heading
        if "angular_velocity" in merged.columns:
            ang_col = "angular_velocity"
        elif "heading" in merged.columns:
            # Simple delta heading
            dh = np.concatenate(([0], np.diff(merged["heading"].values)))
            merged["delta_heading"] = dh
            ang_col = "delta_heading"
        else:
            merged["zeros_ang"] = 0
            ang_col = "zeros_ang"
            
        # Build features including first-order derivatives to capture dynamics
        kin_features = ["velocity_px_s", ang_col] + latent_cols
        
        # Calculate diffs
        diff_df = merged[kin_features].diff().fillna(0)
        diff_df.columns = [f"{c}_diff" for c in diff_df.columns]
        
        merged = merged.join(diff_df)
        all_features = kin_features + list(diff_df.columns)
        
        # We fill NaNs in features with median or 0
        merged[all_features] = merged[all_features].fillna(0)
        
        # For each channel, train a non-linear model on continuous frames
        for ch in channels:
            zcol = ch["zcol"]
            
            # Ensure target z-score and features are completely finite and not NaN
            valid_mask = ~merged[zcol].isna() & np.isfinite(merged[zcol])
            for feat in all_features:
                valid_mask = valid_mask & ~merged[feat].isna() & np.isfinite(merged[feat])
                
            if valid_mask.sum() < 100: continue
            
            X = merged.loc[valid_mask, all_features].to_numpy()
            y = merged.loc[valid_mask, zcol].to_numpy()
            
            # Subsample slightly to speed up training if sessions are extremely long
            # (e.g. 100,000 frames)
            if len(X) > 40000:
                step = len(X) // 40000
                X_train = X[::step]
                y_train = y[::step]
            else:
                X_train = X
                y_train = y
                
            # Train RandomForestRegressor
            reg = RandomForestRegressor(
                n_estimators=30, max_depth=8, random_state=42, n_jobs=-1
            )
            reg.fit(X_train, y_train)
            
            # Predict on full session to preserve continuous trace
            y_pred = reg.predict(X)
            
            # Calculate residuals
            residuals = y - y_pred
            
            # Save back to s.all_data_df using index mapping
            res_col = f"{zcol}_residual_kinematic_free"
            s.all_data_df[res_col] = np.nan
            s.all_data_df.loc[valid_mask, res_col] = residuals
            
            # Save R^2 score for diagnostic logging
            r2 = reg.score(X_train, y_train)
            print(f"    Channel {zcol} ({ch['broad_reg']}): Kinematic R^2 = {r2:.4f}")


def decode_residuals(sessions, project_name):
    """
    Extracts PETHs for raw and residual DA, trains an RFC to decode syllables,
    and plots a comparison bar chart for all sub-groups (Drug, Genotype, Phase) 
    showing fold improvement over chance. Ensure unique filenames with no overwrites.
    """
    print(f"\n--- Training Syllable Classifiers on Residuals ({project_name}) ---")
    
    top_sylls = _get_top_syllables(sessions, k=20)
    if not top_sylls:
        print("No top syllables found.")
        return
        
    window_s = 0.5
    
    # Standard PETH extraction helper for raw and residual
    def _extract_traces(sessions, target_sylls):
        raw_traces = []
        res_traces = []
        
        for s in sessions:
            fs = s.metadata.get("fs", 80.0)
            window_frames = int(window_s * fs)
            
            df = s.all_data_df
            if "onset" not in df.columns: continue
            
            channels = _get_functional_channels(s)
            df_indexed = df.set_index("frame_index")
            
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
                    
                    if start_idx < df_indexed.index.min() or end_idx > df_indexed.index.max(): continue
                    
                    window_df = df_indexed.loc[start_idx:end_idx]
                    
                    # Determine phase
                    phase = "post" if (inj_frame is not None and onset >= inj_frame) else "pre"
                    
                    for ch in channels:
                        zcol = ch["zcol"]
                        res_col = f"{zcol}_residual_kinematic_free"
                        
                        if zcol not in window_df.columns or res_col not in window_df.columns: continue
                        
                        raw_arr = window_df[zcol].values
                        res_arr = window_df[res_col].values
                        
                        if len(raw_arr) != (2 * window_frames + 1) or len(res_arr) != (2 * window_frames + 1): continue
                        if np.isnan(raw_arr).any() or np.isnan(res_arr).any(): continue
                        
                        raw_traces.append({
                            "Drug": drug,
                            "Genotype": geno,
                            "Phase": phase,
                            "Syllable": syll,
                            "Trace": raw_arr
                        })
                        res_traces.append({
                            "Drug": drug,
                            "Genotype": geno,
                            "Phase": phase,
                            "Syllable": syll,
                            "Trace": res_arr
                        })
        return pd.DataFrame(raw_traces), pd.DataFrame(res_traces)

    print("  Extracting PETHs...")
    df_raw, df_res = _extract_traces(sessions, top_sylls)
    
    if df_raw.empty or df_res.empty:
        print("  Error: No valid traces extracted for classification.")
        return
        
    df_raw["Group_Key"] = df_raw.apply(lambda r: f"{r['Drug']}_{r['Genotype']}_{r['Phase']}", axis=1)
    df_res["Group_Key"] = df_res.apply(lambda r: f"{r['Drug']}_{r['Genotype']}_{r['Phase']}", axis=1)
    
    unique_groups = sorted(df_raw["Group_Key"].unique())
    
    def _evaluate_rfc(X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        
        accs = []
        shufs = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf.fit(X_train, y_train)
            accs.append(clf.score(X_test, y_test))
            
            y_shuf = np.random.permutation(y_train)
            clf.fit(X_train, y_shuf)
            shufs.append(clf.score(X_test, y_test))
            
        return np.mean(accs), np.mean(shufs)

    for group_key in unique_groups:
        sub_raw = df_raw[df_raw["Group_Key"] == group_key]
        sub_res = df_res[df_res["Group_Key"] == group_key]
        
        if len(sub_raw) < 50:
            print(f"  Skipping {group_key}: not enough samples ({len(sub_raw)}).")
            continue
        
        X_raw = np.vstack(sub_raw["Trace"].values)
        y_raw = sub_raw["Syllable"].values
        
        X_res = np.vstack(sub_res["Trace"].values)
        y_res = sub_res["Syllable"].values
        
        MAX_SAMPLES = 4000
        if len(X_raw) > MAX_SAMPLES:
            X_raw, y_raw = resample(X_raw, y_raw, n_samples=MAX_SAMPLES, random_state=42, stratify=y_raw)
            X_res, y_res = resample(X_res, y_res, n_samples=MAX_SAMPLES, random_state=42, stratify=y_res)
            
        print(f"  Training RFC classifiers on {len(X_raw)} instances ({group_key})...")
        acc_raw, shuf_raw = _evaluate_rfc(X_raw, y_raw)
        acc_res, shuf_res = _evaluate_rfc(X_res, y_res)
        
        fold_raw = acc_raw / shuf_raw if shuf_raw > 0 else 0
        fold_res = acc_res / shuf_res if shuf_res > 0 else 0
        
        print(f"    [{group_key}] Raw RFC: {fold_raw:.2f}x over chance | Residual RFC: {fold_res:.2f}x over chance")
        
        # Plot
        plt.figure(figsize=(6, 5))
        plt.bar(
            ["Raw Dopamine", "Residual (Kinematic-Free)"],
            [fold_raw, fold_res],
            color=["#d62728", "#2ca02c"],
            edgecolor="black",
            width=0.5
        )
        
        plt.axhline(1.0, color='black', linestyle='--', label='Chance (Shuffle)')
        plt.ylabel("Decoding Accuracy (Fold over shuffle)")
        
        # Human readable title
        drug_name, genotype, phase = group_key.split("_")
        plt.title(f"Unique Info: Syllable Decoding ({project_name})\nGroup: {drug_name} | {genotype} | Phase: {phase.upper()}")
        
        sns.despine()
        plt.tight_layout()
        
        out_path = os.path.join(OUTPUT_DIR, f"{project_name.lower()}_fig_unique_info_rfc_{group_key.lower()}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()


def run_pipeline(sessions, project_name):
    """Entry point to run the entire Unique Information Pipeline."""
    print(f"\n=======================================================")
    print(f" STARTING UNIQUE INFORMATION PIPELINE: {project_name}")
    print(f"=======================================================")
    
    # 1. 2D Kinematic Grids
    plot_kinematic_grid(sessions, project_name)
    
    # 2. Per-Session Non-Linear Kinematic Residualization
    strip_kinematics(sessions)
    
    # 3. Residual RFC Decoding Comparison
    decode_residuals(sessions, project_name)
    
    print(f"\nCompleted Unique Information Pipeline for {project_name}.")
    print(f"Output files saved to: output/unique_information/")
