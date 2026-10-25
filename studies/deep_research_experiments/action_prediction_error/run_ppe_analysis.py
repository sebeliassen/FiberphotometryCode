import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Activate this checkout only for direct-file execution, through main().
_bootstrap_paths = []
if __name__ == "__main__" and not __package__:
    _repo_root = Path(__file__).resolve().parents[3]
    for _source_dir in (_repo_root, _repo_root / "src"):
        if str(_source_dir) not in sys.path:
            sys.path.insert(0, str(_source_dir))
            _bootstrap_paths.append(str(_source_dir))

try:
    from studies.deep_research_experiments.utils.loaders import load_sessions
except BaseException:
    for _source_dir in _bootstrap_paths:
        if _source_dir in sys.path:
            sys.path.remove(_source_dir)
    raise


def get_dls_signal_col(session):
    """
    Finds the column corresponding to DLS signal (usually Green/470nm).
    Uses 'signal_meta' dictionary in session metadata.
    Returns the base signal key (e.g., 'signal_2').
    """
    meta = session.metadata
    signal_meta = meta.get('signal_meta', {})

    # 1. Try precise match via signal_meta
    for sig_key, info in signal_meta.items():
        # Check brain region (flexible case)
        brain_reg = info.get('brain_reg', '').upper()
        orig_letter = info.get('orig_letter', '').upper() # 'G' for Green/470, 'R' for Red/560/Iso

        # We want DLS and Green signal (Calcium-dependent)
        if 'DLS' in brain_reg and orig_letter == 'G':
            return sig_key

    # 2. Fallback: Search in signal keys directly if meta is incomplete
    # (Skipping aggressive heuristic to avoid False Positives)

    return None

def get_best_signal_column(df, base_key):
    """
    Given a base key (e.g., 'signal_2'), finds the best available version 
    in the dataframe (dff > zscore > raw).
    """
    if f"{base_key}_dff" in df.columns:
        return f"{base_key}_dff"
    elif f"{base_key}_zscore" in df.columns: # Standard zscore
        return f"{base_key}_zscore"
    elif f"{base_key}_zscore_datta" in df.columns: # Datta-specific zscore
         return f"{base_key}_zscore_datta"
    elif base_key in df.columns:
        return base_key
    return None

def run_analysis(test_mode=False, use_history=False, use_kinematics=False, use_timewarp=False, use_force=False, suffix="", metric="variance", *, pkl_path=None, output_dir=None):
    print("--- Action Prediction Error (PPE) Analysis ---")
    print(f"Options: History={use_history}, Kinematics={use_kinematics}, TimeWarp={use_timewarp}, ForceVectors={use_force}, Suffix={suffix}, Metric={metric}")

    if test_mode:
        print("!!! TEST MODE ACTIVE: Processing limited data !!!")

    # 1. Load Data
    # For AMPH only as requested
    if pkl_path is None:
        all_sessions = load_sessions('amph', apply_filter=True)
    else:
        all_sessions = load_sessions('amph', apply_filter=True, pkl_path=Path(pkl_path))

    results = []

    # Group by project for easier debugging
    # But we process all

    counter = 0
    IGNORE_GENOTYPES = {'Unknown', 'UNK', 'nan', None}

    # Import for time warping
    from scipy.signal import resample

    for s in all_sessions:
        if test_mode and counter >= 4: # Process 2 IFDN + 2 AMPH approx
             break

        project = str(s.project or '').lower()

        genotype = s.metadata.get('genotype', 'Unknown')

        # Normalize Drug
        if 'amph' in project:
             drug = s.metadata.get('drug', 'Saline')
             if pd.isna(drug) or str(drug).lower() in ['nan', 'none', '']:
                 drug = 'Saline'
             # Standardize names
             drug_str = str(drug).lower()
             if 'amphetamine' in drug_str: drug = 'Amphetamine'
             elif 'saline' in drug_str: drug = 'Saline'
        else:
             drug = 'Baseline'

        # Filter Genotype
        if pd.isna(genotype) or str(genotype).lower() in ['unknown', 'unk', 'nan', 'none']:
            # print(f"Skipping {s.session_id}: Genotype is {genotype}")
            continue

        # Also filter if drug is somehow still None/Unknown if that matters?
        # But for IFDN we set Baseline.

        if not hasattr(s, 'all_data_df') or s.all_data_df is None or s.all_data_df.empty:
            print(f"Skipping {s.session_id}: all_data_df missing or empty.")
            continue

        df = s.all_data_df.reset_index(drop=True)

        # Verify columns exist
        if 'syllable' not in df.columns:
             print(f"Skipping {s.session_id}: 'syllable' column missing in all_data_df.")
             continue

        # Identify DLS column base key
        dls_base_key = get_dls_signal_col(s)
        if not dls_base_key:
             print(f"Skipping {s.session_id}: No DLS (Green) signal found in metadata.")
             continue

        # Find best column (dff vs raw)
        dls_col = get_best_signal_column(df, dls_base_key)
        if not dls_col:
            print(f"Skipping {s.session_id}: Base key {dls_base_key} found but no data column in all_data_df.")
            continue

        # Detect onsets using 'onset' column if available, else manual
        if 'onset' in df.columns:
            # onset column is often boolean or 0/1
            onsets = df[df['onset'] == 1].index
        else:
            df['syllable_prev'] = df['syllable'].shift(1).fillna(-100)
            onsets = df[df['syllable'] != df['syllable_prev']].index

        if len(onsets) == 0:
            print(f"Skipping {s.session_id}: No onsets found.")
            continue

        # Define window frames here
        fps = s.metadata.get('fps', 80)
        try:
             window_frames = int(0.2 * float(fps))
        except:
             window_frames = 15

        # Instance ID for time warping (need end of syllable)
        if use_timewarp:
             # Create simple instance IDs
             df['instance_id_temp'] = (df['syllable'] != df['syllable'].shift(1)).cumsum()

        syll_counts = df.loc[onsets, 'syllable'].value_counts()
        if test_mode:
            print(f"Session {s.session_id}: DLS={dls_col}, {len(onsets)} onsets")

        top_sylls = syll_counts.head(5).index.tolist()

        if test_mode:
            top_sylls = top_sylls[:2] # Only top 2

        for syll_id in top_sylls:
            if syll_id < 0: continue # Skip usage/garbage

            # Get valid onsets for this syllable
            # Need strict onset? yes.
            # If using 'onset' col, we need to filter onsets where syllable == id
            syll_onsets = df.loc[onsets][df.loc[onsets, 'syllable'] == syll_id].index

            # Extract trial features
            trial_data = []

            for idx in syll_onsets:
                # 1. Trace Extraction logic
                trace = None

                if use_timewarp:
                    # Find end of this instance
                    inst_id = df.at[idx, 'instance_id_temp']
                    # Slice DF for this instance (fast check forward)
                    # Heuristic: limit search to max 2s (approx 160 frames) to avoid huge search
                    # Or use groupby elsewhere. But local search is ok for onsets.
                    # Fast way: check consecutive equal instance_id
                    # Actually, if we have instance_id, we can just grab group.
                    # But groupby is slow per trial.
                    # Let's assume onsets are strict start.
                    # We just seek forward until syllable changes or max frames.
                    # Or, better: pre-calculate lengths.
                    # Simpler: Slice a safe chunk, find change.
                    safe_window = int(2.0 * fps)
                    chunk = df.iloc[idx : idx+safe_window]
                    # mask of same instance
                    mask = (chunk['instance_id_temp'] == inst_id)
                    # Apply mask
                    valid_chunk = chunk[mask]

                    if len(valid_chunk) < 3: # Too short
                        continue

                    raw_trace = valid_chunk[dls_col].values
                    # Resample to fixed window_frames (Time-Warping)
                    trace = resample(raw_trace, window_frames)

                    # Also need warped kinematics? technically yes.
                    # For simple scalars (peak vel), raw max is prob fine?
                    # "Vigor" is usually mean/max of the bout.
                    # We'll stick to max of the full bout for vigor.
                    vel_seq = valid_chunk['velocity_px_s'].values if 'velocity_px_s' in df.columns else np.zeros(len(valid_chunk))
                    acc_seq = valid_chunk['acceleration'].values if 'acceleration' in df.columns else np.zeros(len(valid_chunk))

                else:
                    # Fixed Window
                    if idx + window_frames >= len(df): continue
                    trace = df.iloc[idx : idx+window_frames][dls_col].values
                    vel_seq = df.iloc[idx : idx+window_frames]['velocity_px_s'].values if 'velocity_px_s' in df.columns else np.zeros(window_frames)
                    acc_seq = df.iloc[idx : idx+window_frames]['acceleration'].values if 'acceleration' in df.columns else np.zeros(window_frames)

                # Check for NaNs in trace (photometry)
                if np.isnan(trace).any():
                    continue

                # Check for NaNs in kinematics
                if np.isnan(vel_seq).all(): peak_vel = 0
                else: peak_vel = np.nanmax(vel_seq)

                if np.isnan(acc_seq).all(): peak_acc = 0
                else: peak_acc = np.nanmax(np.abs(acc_seq)) # Magnitude

                # Final guard against NaN features
                if np.isnan(peak_vel) or np.isnan(peak_acc):
                    continue

                # Force Vectors (Directionality)
                # Proxy: Acceleration Sign (Propulsion vs Braking) at Onset?
                # Or Mean Accel?
                # If we use "Force Vectors", we basically split this trial into a subgroup.
                # Let's categorize: "Forward" (Accel > 0), "Backward/Braking" (Accel < 0)
                # We calculate mean acceleration of the window.
                mean_acc_val = np.nanmean(acc_seq) if len(acc_seq) > 0 else 0
                force_dir = 'Pos' if mean_acc_val >= 0 else 'Neg'

                trial_data.append({
                    'trace': trace,
                    'velocity': peak_vel,
                    'acceleration': peak_acc,
                    'force_dir': force_dir,
                    'timestamp': df.iloc[idx]['SystemTimestamp'] if 'SystemTimestamp' in df.columns else idx
                })

            if len(trial_data) < 10: continue

            # --- SPLIT BY FORCE VECTORS? ---
            # If use_force is True, we essentially have TWO rows based on direction.
            # But the loop structure assumes 1 syllable -> 1 result row.
            # We can run the pipeline twice or stick to standard grouping.
            # Simplest: If force vectors on, filter/split?
            # Or add 'direction' as a REGRESSOR?
            # The prompt says: "Split your analysis... get high variance that is actually distinct vector coding."
            # This implies clustering/splitting.
            # Let's iterate over Directions if enabled.

            directions = ['All']
            if use_force:
                directions = ['Pos', 'Neg']

            df_all_trials = pd.DataFrame(trial_data)

            for direction in directions:
                if direction == 'All':
                    curr_trials = df_all_trials
                else:
                    curr_trials = df_all_trials[df_all_trials['force_dir'] == direction]

                if len(curr_trials) < 5: continue

                # Vectorize traces for THIS subset
                traces_subset = np.array(curr_trials['trace'].tolist())
                x_axis = np.arange(window_frames)

                slopes = []
                for t in traces_subset:
                     if len(t) < 2:
                         slopes.append(0)
                         continue
                     # Fit line
                     slope_val = np.polyfit(x_axis, t, 1)[0]
                     slopes.append(slope_val)

                # Add slope to DF for regression
                # We need a new DF for this subset to avoid SettingWithCopy warnings
                df_subset_reg = curr_trials.copy()
                df_subset_reg['slope'] = slopes

                # --- 1. History (Expectation) ---
                if use_history:
                    # Rolling mean of slopes (lag 1)
                    df_subset_reg = df_subset_reg.sort_values('timestamp')
                    df_subset_reg['history'] = df_subset_reg['slope'].shift(1).rolling(window=5, min_periods=1).mean()
                    df_subset_reg['history'] = df_subset_reg['history'].fillna(df_subset_reg['slope'].mean())

                # --- 2. Residuals (Regression) ---
                # Build feature list for regression
                features = []
                if use_history: features.append('history')
                if use_kinematics: features.extend(['velocity', 'acceleration'])

                if features:
                    # Run metrics.get_residual_variance with specific columns?
                    # metrics func currently hardcodes ['history', 'velocity', 'acceleration']
                    # We should probably update metrics func or just select cols here.
                    # For now, let's just prep the DF and call a slightly modified metrics logic or ensure columns exist
                    # If feature missing, fill 0?
                    # Better: quick inline regression if metrics is hardcoded.
                    # Actually I modified metrics to take df_trials.
                    # Let's temporarily ensure cols exist if not using keys.
                    if 'history' not in df_subset_reg.columns: df_subset_reg['history'] = 0
                    # Velocity/Accel exist.

                    # BUT metrics.get_residual_variance hardcodes the X columns.
                    # Simple fix: Assume metrics handles it or we modify metrics.
                    # I will assume I need to modify metrics.py to be flexible or hack it here.
                    # Hack: if not use_history, set history=0. coeff will be 0 or separate.
                    # Actually, better to just let regression handle constant 0.
                    pass

                # IF NO flags, we just want Raw.
                # If flags, we want Residual.

                res_stats = {'residual_variance': np.var(slopes), 'r_squared': 0.0}
                if features:
                     # Calculate Residuals
                     from sklearn.linear_model import LinearRegression
                     X = df_subset_reg[features]
                     y = df_subset_reg['slope']
                     if len(X) > 10:
                         model = LinearRegression()
                         model.fit(X, y)
                         preds = model.predict(X)
                         resid = y - preds
                         res_stats = {
                             'residual_variance': np.var(resid),
                             'residual_mean_abs': np.mean(np.abs(resid)), # Average surprise magnitude
                             'r_squared': model.score(X, y)
                         }

                # --- 3. Raw Stats ---
                raw_var = np.var(slopes)
                raw_mean = np.mean(slopes)
                raw_abs_mean = np.mean(np.abs(slopes))

                result_dict = {
                    'session': s.session_id,
                    'genotype': genotype,
                    'drug': drug,
                    'syllable': syll_id,
                    'direction': direction,
                    'slope_variance_raw': raw_var,
                    'slope_variance_resid': res_stats['residual_variance'],
                    'slope_mean_abs_raw': raw_abs_mean,
                    'slope_mean_abs_resid': res_stats.get('residual_mean_abs', 0),
                    'r_squared': res_stats['r_squared'],
                    'n_trials': len(df_subset_reg)
                }

                # Map target_metric for consolidated plotting
                if metric == 'mean':
                    result_dict['slope_variance'] = raw_abs_mean
                    result_dict['residual_variance_field'] = res_stats.get('residual_mean_abs', 0)
                else:
                    result_dict['slope_variance'] = raw_var
                    result_dict['residual_variance_field'] = res_stats['residual_variance']

                results.append(result_dict)

        counter += 1
        if test_mode and counter >= 4:
            break

    # Create output dataframe
    res_df = pd.DataFrame(results)

    print("\n--- Results Summary ---")
    if not res_df.empty:
        # Save Results
        out_dir = (
            os.fspath(output_dir) if output_dir is not None else
            "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/deep_research_experiments/action_prediction_error/output"
        )
        os.makedirs(out_dir, exist_ok=True)
        csv_name = f"ppe_results{suffix}.csv"
        res_df.to_csv(os.path.join(out_dir, csv_name))
        print(f"Saved to {out_dir}/{csv_name}")

        # Plotting
        try:
            from studies.deep_research_experiments.action_prediction_error.lib import plotting

            # 1. Plot Raw Metric (Variance or Mean Abs Slope)
            ylabel_raw = "Mean Absolute Slope (dF/F/s)" if metric == 'mean' else "Total Slope Variance"
            plotting.plot_slope_variance(res_df, out_dir, filename_suffix=f"{suffix}_raw", ylabel=ylabel_raw)

            # 2. Plot Residual Metric (Residual Mean/MAE or Residual Variance)
            ylabel_resid = "Mean Absolute Residual (MAE)" if metric == 'mean' else "Residual (Error) Variance"
            res_df_resid = res_df.copy()
            res_df_resid['slope_variance'] = res_df_resid['residual_variance_field']
            plotting.plot_slope_variance(res_df_resid, out_dir, filename_suffix=f"{suffix}_residual", ylabel=ylabel_resid)

        except Exception as e:
            print(f"Plotting failed: {e}")

    else:
        print("No results generated.")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Run on small subset")
    parser.add_argument("--history", action="store_true", help="Enable History (RL) correction")
    parser.add_argument("--kinematics", action="store_true", help="Enable Kinematic correction")
    parser.add_argument("--time-warp", action="store_true", help="Enable Time-Warping")
    parser.add_argument("--force-vectors", action="store_true", help="Enable Force Vectors (Directionality)")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output files")
    parser.add_argument("--metric", type=str, default="variance", choices=["variance", "mean"], help="Metric to analyze: 'variance' (PPE) or 'mean' (Signal Strength)")
    parser.add_argument("--sessions-pkl", dest="pkl_path", type=Path, default=None,
                        help="Explicit trusted session pickle (omit to retain AMPH discovery)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for PPE CSV and plots (omit to retain the historical output path)")
    args = parser.parse_args(argv)

    path_options = {}
    if args.pkl_path is not None:
        path_options["pkl_path"] = args.pkl_path
    if args.output_dir is not None:
        path_options["output_dir"] = args.output_dir

    run_analysis(
        test_mode=args.test_mode,
        use_history=args.history,
        use_kinematics=args.kinematics,
        use_timewarp=args.time_warp,
        use_force=args.force_vectors,
        suffix=args.suffix,
        metric=args.metric,
        **path_options
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        for _source_dir in _bootstrap_paths:
            if _source_dir in sys.path:
                sys.path.remove(_source_dir)
