import numpy as np
import pandas as pd
from scipy import signal
import logging

logger = logging.getLogger("Panel1z.Processing")

def get_fs(df):
    """Estimate sampling rate from SystemTimestamp."""
    if "SystemTimestamp" in df.columns and len(df) > 1:
        valid_ts = df["SystemTimestamp"].dropna().to_numpy()
        if len(valid_ts) > 1:
            dt = np.nanmedian(np.diff(valid_ts))
            return 1 / dt if dt > 0 else 30.0
    return 30.0

def apply_bessel_filter(series, fs):
    """
    Apply 1Hz High-Pass Bessel Filter to remove drift.
    Preserves NaNs.
    """
    nyquist = fs / 2.0
    if nyquist <= 1.0: return series
    
    sos = signal.bessel(1, 1.0 / nyquist, btype="high", output="sos")
    
    valid_mask = ~np.isnan(series)
    result = np.full_like(series, np.nan)
    
    # Filter only if enough data
    if np.sum(valid_mask) > 10:
        result[valid_mask] = signal.sosfiltfilt(sos, series[valid_mask])
        
    return result

def enrich_sessions(sessions):
    """
    Pre-calculate kinematics and metadata for all sessions.
    Modifies sessions in-place.
    """
    processed_count = 0
    
    for s in sessions:
        df = getattr(s, "all_data_df", None)
        if df is None or df.empty: continue
        
        # 1. Sampling Rate
        fs = get_fs(df)
        s.metadata['fs'] = fs # Cache for modules
        
        # 2. Kinematics: Acceleration
        # Check if already exists to avoid re-calc cost
        if "acceleration" not in df.columns:
            if "velocity_px_s" in df.columns:
                df["acceleration"] = np.gradient(df["velocity_px_s"], 1/fs)
            else:
                # Fallback if velocity missing (unlikely)
                df["acceleration"] = np.nan
        
        # 3. Geometry: Dist from Center
        if "dist_from_center" not in df.columns:
            if "centroid_x" in df.columns and "centroid_y" in df.columns:
                cx, cy = df["centroid_x"].mean(), df["centroid_y"].mean()
                df["dist_from_center"] = np.sqrt((df["centroid_x"] - cx)**2 + (df["centroid_y"] - cy)**2)
        
        # 3.5 Bake 'Group' (Genotype + Drug)
        geno = s.metadata.get("genotype", "UNK")
        drug = s.metadata.get("drug", "")
        if drug and str(drug).lower() != "nan":
             # Mappings
             if drug == "Amphetamine": drug = "Amph"
             s.metadata["group"] = f"{geno} {drug}".strip()
        else:
             s.metadata["group"] = geno
             
        # 4. Syllable Instances (Runs)
        if "instance_id" not in df.columns:
             # Fast vectorised calc
            df["instance_id"] = (df["syllable"] != df["syllable"].shift(1)).cumsum()
            
        # 5. Onsets Bool (Standardize)
        if "onset" not in df.columns:
            df["onset"] = (df["syllable"] != df["syllable"].shift(1))
            
        # --- NEW: Calculate Syllable Stats ---
        # Group by syllable and calculate metrics
        # We need duration, velocity peaks, and tortuosity
        if "syllable" in df.columns:
            # 1. Duration & Tortuosity
            # Identify instances
            df["instance_id"] = (df["syllable"] != df["syllable"].shift(1)).cumsum()
            
            # --- Vectorized Kinematics ---
            # Path Length
            if "centroid_x" in df.columns and "centroid_y" in df.columns:
                dx = df["centroid_x"].diff().fillna(0)
                dy = df["centroid_y"].diff().fillna(0)
                df["step_dist"] = np.sqrt(dx**2 + dy**2)
                
                # Instance Aggregation
                inst_stats = df.groupby("instance_id").agg({
                    "step_dist": "sum",
                    "centroid_x": ["first", "last"],
                    "centroid_y": ["first", "last"],
                    "syllable": "first",
                    "velocity_px_s": "max" 
                })
                
                # Displacement
                x1 = inst_stats[("centroid_x", "first")]
                x2 = inst_stats[("centroid_x", "last")]
                y1 = inst_stats[("centroid_y", "first")]
                y2 = inst_stats[("centroid_y", "last")]
                disp = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Tortuosity
                path_len = inst_stats[("step_dist", "sum")]
                # Avoid divide by zero
                inst_tort = path_len / disp.replace(0, np.nan)
                inst_tort = inst_tort.fillna(1.0) # If no displacement, assume straight? Or 1.0
                
                # Map back to syllable
                syll_tort_map = inst_stats[("syllable", "first")].to_frame(name="s")
                syll_tort_map["t"] = inst_tort
                
                # Group by Syllable
                tort_by_syll = syll_tort_map.groupby("s")["t"].mean()
            else:
                tort_by_syll = pd.Series()

            # Group instances for other stats
            s_insts_size = df.groupby(["syllable", "instance_id"]).size().groupby(level=0)
            s_vel_max = df.groupby(["syllable", "instance_id"])["velocity_px_s"].max().groupby(level=0)
            
            accel_col = "acceleration" if "acceleration" in df.columns else ("acceleration_px_s2" if "acceleration_px_s2" in df.columns else None)
            if accel_col:
                s_acc_max = df.groupby(["syllable", "instance_id"])[accel_col].max().groupby(level=0)
            
            # Calculate stats per syllable type
            stats_list = []
            unique_sylls = df["syllable"].unique()
            
            for syll in unique_sylls:
                if pd.isna(syll): continue
                
                # Duration
                if syll in s_insts_size.groups:
                    durs = s_insts_size.get_group(syll) / fs
                else:
                    durs = []
                
                if len(durs) == 0: continue

                # Velocity
                vels = s_vel_max.get_group(syll) if syll in s_vel_max.groups else []
                accs = s_acc_max.get_group(syll) if accel_col and syll in s_acc_max.groups else vels
                
                # Tortuosity
                tort = tort_by_syll.get(syll, np.nan)
                
                stats_list.append({
                    "syllable": syll,
                    "count": len(durs),
                    "velocity_p95_median": np.nanpercentile(vels, 95) if len(vels)>0 else 0,
                    "accel_p95_median": np.nanpercentile(accs, 95) if len(accs)>0 else 0,
                    "duration_cv": np.std(durs) / np.mean(durs) if np.mean(durs)>0 else 0,
                    "tortuosity_avg": tort
                })
            s.syllable_stats = stats_list
            
        processed_count += 1
        
    logger.info(f"   Enriched {processed_count} sessions with advanced kinematics.")
    return sessions

def apply_filter_to_sessions(sessions):
    """
    Apply 1Hz Bessel filter to all signal columns (zscore, dff) in-place.
    """
    count = 0 
    for s in sessions:
        df = getattr(s, "all_data_df", None)
        if df is None or df.empty: continue
        
        fs = s.metadata.get('fs', 30.0)
        
        # Identify columns to filter
        targets = [c for c in df.columns if 'zscore' in c or 'dff' in c]
        
        for col in targets:
            filtered = apply_bessel_filter(df[col].to_numpy(), fs)
            df[col] = filtered
            
        count += 1
        
    logger.info(f"   Applied 1Hz Bessel Filter to {count} sessions.")
    return sessions

def filter_post_injection(sessions, delay_mins):
    """
    Truncates session data to keep only [Injection + Delay] onwards.
    """
    count = 0
    dropped = 0
    
    for s in sessions:
        df = getattr(s, "all_data_df", None)
        if df is None or df.empty: continue
        
        md = s.metadata
        fs = md.get('fs', 30.0)
        
        # 1. Determine Start Frame
        inj_frame = md.get("injection_frame")
        if inj_frame is None:
            # Fallback: Check if we can infer it or just skip
            # If default AMPH build, injection_frame should be there.
            # If not, maybe warn?
            # For now, if no injection info, we can't filter.
            continue
            
        delay_frames = int(delay_mins * 60 * fs)
        start_frame = int(inj_frame + delay_frames)
        
        if start_frame >= len(df):
            # Entire session is pre-injection/delay?
            s.all_data_df = pd.DataFrame() # Empty it
            dropped += 1
            continue
            
        # 2. Truncate DF
        # Assuming df is sequential.
        # We slice by integer position (iloc) if index corresponds to frames, 
        # but to be safe we treat 'start_frame' as an absolute index into the recording.
        # IF df index is 0..N, then df.iloc[start_frame:] is correct assuming start_frame is 0-indexed relative to this DF.
        # BUT 'injection_frame' is absolute. 'all_data_df' is usually absolute.
        
        # Determine slicing method
        # If 'frame_idx' column exists, use that.
        # Else assume row index matches absolute time (dangerous if filtered before).
        
        # In current pipeline, all_data_df is usually full session length.
        s.all_data_df = df.iloc[start_frame:].copy()
        
        # 3. Update Metadata if needed
        # (e.g. durations)
        
        count += 1
        
    logger.info(f"   Filtered Post-Injection (Delay={delay_mins}m) for {count} sessions (Dropped {dropped} empty).")
    return [s for s in sessions if not s.all_data_df.empty]
