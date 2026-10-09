import numpy as np
import pandas as pd
from typing import Dict, Any, List
import os
from scipy import stats
from scipy.ndimage import gaussian_filter1d

def calculate_kinematics(session) -> Dict[str, float]:
    """
    Calculate summary kinematics for independent variables (confounds).
    """
    df = session.all_data_df
    res = {}
    
    # Velocity
    if "velocity_px_s" in df.columns:
        vel = df["velocity_px_s"].dropna()
        res["mean_velocity"] = vel.mean()
        res["median_velocity"] = vel.median()
        res["p95_velocity"] = vel.quantile(0.95)
        
        # Acceleration (approximate from velocity gradient)
        # fs is needed for correct units. Assuming 30Hz if not present in metadata.
        fs = session.metadata.get("fs", 30.0)
        # Fix 4: Smooth velocity before differentiation to avoid noise amplification
        vel_smooth = gaussian_filter1d(vel, sigma=2)
        accel = np.gradient(vel_smooth, 1/fs)
        res["mean_acceleration"] = np.mean(np.abs(accel))
        res["p95_acceleration"] = np.percentile(np.abs(accel), 95)
    else:
        res["mean_velocity"] = np.nan
        res["mean_acceleration"] = np.nan
        
    return res

def get_syllable_instances(df):
    """
    Count instances (bouts) of each syllable in a dataframe.
    """
    if df.empty or "syllable" not in df.columns: return pd.Series()
    s = df["syllable"].fillna(-1).to_numpy()
    # Collapse self-transitions to get bout stream
    mask = np.concatenate(([True], s[1:] != s[:-1]))
    bouts = s[mask]
    # Filter out gap (-1)
    bouts = bouts[bouts != -1]
    if len(bouts) == 0: return pd.Series()
    return pd.Series(bouts).value_counts()

def get_syllable_usage(session, top_n: int = 50, min_instances: int = 0) -> pd.Series:
    """
    Get frequency distribution of syllables, optionally filtering by instances.
    """
    df = session.all_data_df
    if "syllable" not in df.columns: return pd.Series()
    
    if min_instances > 0:
        counts = get_syllable_instances(df)
        robust = counts[counts >= min_instances].index
        df = df[df["syllable"].isin(robust)]
        
    if df.empty: return pd.Series()
    usage = df["syllable"].value_counts(normalize=True)
    return usage

def calculate_entropy_metrics(session, top_n_syllables: int = 40) -> Dict[str, float]:
    """
    Calculate Entropy and Stereotypy.
    Implements the "Rigor Manifesto" mandate:
    1. Filter to Top N frequent syllables (to stabilize matrix).
    2. Discrete Entropy of transitions.
    """
    df = session.all_data_df
    
    if "syllable" not in df.columns:
        return {"entropy_transition": np.nan, "stereotypy": np.nan}
        
    # 1. Get Syllable Stream
    # Drop consecutive duplicates to get transition stream? 
    # Or strict framewise?
    # Usually, "Transition Entropy" implies ignoring self-transitions (A->A).
    # "Stationary Entropy" implies framewise.
    # Let's do Bigram Entropy on the reduced sequence (Run-Length Encodedish)
    
    # Updated NaN Strategy:
    # Do NOT drop NaNs. Treating [A, NaN, B] as [A, B] creates false transitions.
    # Instead, fill NaNs with a placeholder (-1) that effectively acts as a boundary.
    raw_seq = df["syllable"].fillna(-1).to_numpy()
    
    if len(raw_seq) < 10:
        return {"entropy_transition": np.nan, "stereotypy": np.nan}
        
    # Collapse self-transitions
    mask = np.concatenate(([True], raw_seq[1:] != raw_seq[:-1]))
    transitions = raw_seq[mask]
    
    # Filter out the placeholder itself from being a valid state?
    # transitions now looks like [A, -1, B]
    # Bigrams: (A, -1), (-1, B).
    # If we filter to Valid Sets (Top N), -1 will definitely NOT be in Top N.
    # So (A, -1) becomes (A, Other). (-1, B) becomes (Other, B).
    # This correctly penalizes the prediction of "A -> ?" if A is followed by a gap.
    
    if len(transitions) < 10:
         return {"entropy_transition": np.nan, "stereotypy": np.nan}

    # Constrain to Top N
    counts = pd.Series(transitions).value_counts()
    
    # Ensure -1 is not in valid syllables even if it's frequent (though improbable to be top unless bad data)
    counts = counts.drop(-1, errors='ignore')
    
    valid_syllables = counts.head(top_n_syllables).index.to_numpy()
    
    # Map sequence
    mapped_seq = np.array([s if s in valid_syllables else -1 for s in transitions])
    
    # Calculate Bigram Probabilities
    bigrams = list(zip(mapped_seq[:-1], mapped_seq[1:]))
    bg_df = pd.DataFrame(bigrams, columns=["src", "dst"])
    
    entropy_sum = 0
    src_counts = bg_df["src"].value_counts(normalize=True) 
    
    for src_syll in src_counts.index:
        if src_syll == -1: continue 
        
        dst_probs = bg_df[bg_df["src"] == src_syll]["dst"].value_counts(normalize=True)
        h_cond = -1 * np.sum(dst_probs * np.log2(dst_probs + 1e-10))
        entropy_sum += src_counts[src_syll] * h_cond
        
    # Fix 3: Use constant H_max based on theoretical capacity (Top N)
    # This ensures stereotypy is not confounded by the animal's specific vocabulary size
    h_max = np.log2(top_n_syllables + 1)
    stereotypy = 1.0 - (entropy_sum / h_max)
    
    return {
        "entropy_transition": entropy_sum,
        "stereotypy": stereotypy,
        "n_unique_transitions": len(bg_df)
    }

def calculate_duration_entropy(session) -> float:
    """
    Entropy of the duration distribution.
    Excludes the duration of NaN (gap) periods.
    """
    df = session.all_data_df
    if "syllable" not in df.columns: return np.nan
    
    # Run Length Encoding
    s = df["syllable"].fillna(-1).to_numpy()
    changes = np.concatenate(([True], s[1:] != s[:-1], [True]))
    change_indices = np.flatnonzero(changes)
    durations = np.diff(change_indices) 
    
    # Get values corresponding to these durations
    # The value starts at change_indices[i]
    values = s[change_indices[:-1]]
    
    # Filter: Keep only durations where value != -1
    valid_mask = (values != -1)
    valid_durations = durations[valid_mask]
    
    if len(valid_durations) < 10: return np.nan
    
    try:
        hist, _ = np.histogram(valid_durations, bins=50, density=True)
        clean_hist = hist[hist > 0]
        h = -1 * np.sum(clean_hist * np.log2(clean_hist))
        return h
    except:
        return np.nan

def analyze_session(session) -> Dict[str, Any]:
    """
    Master function to extract standard row of metrics for a session.
    """
    meta = session.metadata
    
    res = {
        "Animal": meta.get("animal_id"),
        "Genotype": meta.get("genotype"),
        "Drug": meta.get("drug", "N/A"),
        "Group": meta.get("group"),
    }
    
    # 1. Kinematics
    res.update(calculate_kinematics(session))
    
    # 2. Entropy
    res.update(calculate_entropy_metrics(session))
    
    # 3. Duration
    res["entropy_duration"] = calculate_duration_entropy(session)
    
    return res

def get_group_transition_matrix(sessions, top_n: int = 15, ordered_syllables: List[int] = None) -> pd.DataFrame:
    """
    Aggregate transition matrices across sessions.
    If ordered_syllables is provided, uses them as rows/cols (ignore top_n).
    """
    import numpy as np
    import pandas as pd
    
    # 1. Identify syllables to use
    if ordered_syllables is not None:
        top_sylls = ordered_syllables
    else:
        all_sylls = []
        for s in sessions:
            if "syllable" in s.all_data_df.columns:
                all_sylls.append(s.all_data_df["syllable"].dropna())
        
        if not all_sylls: return pd.DataFrame()
        global_counts = pd.concat(all_sylls).value_counts()
        top_sylls = global_counts.head(top_n).index.tolist()
    
    # 2. Build aggregate matrix
    # Fix 1: Calculate Per-Session Matrix first, then Average (Avoid Pseudoreplication)
    
    session_matrices = []
    
    for s in sessions:
        if "syllable" not in s.all_data_df.columns: continue
        seq = s.all_data_df["syllable"].fillna(-1).to_numpy()
        
        # Collapse self-transitions
        mask = np.concatenate(([True], seq[1:] != seq[:-1]))
        trans = seq[mask]
        
        if len(trans) < 2: continue
        
        # Count bigrams for this session
        local_counts = pd.DataFrame(0.0, index=top_sylls, columns=top_sylls)
        
        # Filter for top syllables
        valid_indices = [i for i, x in enumerate(trans) if x in top_sylls]
        
        # Iterating numpy array is slow, but robust for this logic.
        # Faster: Map to integer indices
        syll_to_idx = {syll: i for i, syll in enumerate(top_sylls)}
        mapped = np.array([syll_to_idx.get(x, -1) for x in trans])
        
        # Bigrams (vectorized)
        srcs = mapped[:-1]
        dsts = mapped[1:]
        
        # Filter valid pairs
        valid_mask = (srcs != -1) & (dsts != -1)
        valid_srcs = srcs[valid_mask]
        valid_dsts = dsts[valid_mask]
        
        # Add to local counts
        np.add.at(local_counts.values, (valid_srcs, valid_dsts), 1)
        
        # Normalize to Probability Matrix
        row_sums = local_counts.sum(axis=1)
        local_prob = local_counts.div(row_sums.replace(0, 1), axis=0)
        
        session_matrices.append(local_prob)
                
    if not session_matrices:
        return pd.DataFrame(0.0, index=top_sylls, columns=top_sylls)
        
    # Average across sessions
    # element-wise mean
    concatenated = np.stack([m.values for m in session_matrices])
    mean_matrix_vals = np.nanmean(concatenated, axis=0) # Use nanmean just in case
    
    mean_matrix = pd.DataFrame(mean_matrix_vals, index=top_sylls, columns=top_sylls)
    
    return mean_matrix

def get_group_usage(sessions, top_n: int = 20, min_instances: int = 0) -> pd.DataFrame:
    """
    Calculate average syllable usage across animal groups.
    Returns a long-form DF suitable for seaborn.
    """
    import pandas as pd
    rows = []
    for s in sessions:
        usage = get_syllable_usage(s, min_instances=min_instances)
        animal = s.metadata.get("animal_id")
        geno = s.metadata.get("genotype")
        drug = s.metadata.get("drug", "N/A")
        
        for syll, freq in usage.items():
            rows.append({
                "Animal": animal,
                "Genotype": geno,
                "Drug": drug,
                "Syllable": syll,
                "Frequency": freq
            })
    return pd.DataFrame(rows)

def load_quantitative_map(csv_path: str) -> Dict[int, str]:
    """
    Load mapping from Syllable ID to 'Vigor_Turn' string.
    Returns: {syll_id: "Low_Left", etc.}
    """
    if not os.path.exists(csv_path):
        print(f"Warning: map file not found at {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    # syllable, Vigor, Turn
    mapping = {}
    for _, row in df.iterrows():
        sid = int(row["syllable"])
        # Format: Turn_Vigor (e.g. Left_High) or Vigor_Turn?
        # User asked for Matrix: Rows=Turn, Cols=Vigor.
        # Let's verify the labels later.
        # Key: "Turn_Vigor"
        mapping[sid] = f"{row['Turn']}_{row['Vigor']}"
    return mapping

def get_quantitative_group_usage(sessions, mapping: Dict[int, str], min_instances: int = 0) -> pd.DataFrame:
    """
    Aggregate usage into the 9 Turn x Vigor categories.
    Returns DF with 'Time_Pct' and 'Onset_Count' for the Categories.
    """
    rows = []
    
    # Pre-define the 9 categories to ensure zero-filling
    v_levels = ["Low", "Medium", "High"]
    t_levels = ["Left", "Middle", "Right"]
    categories = [f"{t}_{v}" for t in t_levels for v in v_levels]
    
    # Optional: Filter mapping to only include "robust" syllables across the whole session set?
    # Or per group?
    # User said "one syllable has to occur min 15 times for it to be considered".
    # I'll implement it per session for now, or per group if sessions are grouped.
    
    for s in sessions:
        if "syllable" not in s.all_data_df.columns: continue
        
        df = s.all_data_df
        if min_instances > 0:
            counts = get_syllable_instances(df)
            robust = counts[counts >= min_instances].index
            df = df[df["syllable"].isin(robust)]
            
        if df.empty:
            # Still add 0s for this session/animal to avoid drops
            total_frames = len(s.all_data_df)
            time_map, onset_map = {}, {}
        else:
            syll_series = df["syllable"].fillna(-1)
            ids = syll_series.values
            labels = np.array([mapping.get(int(x), "Other") for x in ids])
            
            # 1. Time
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_frames = len(s.all_data_df) # Total frames of session
            time_map = dict(zip(unique_labels, counts))
            
            # 2. Onsets
            changes = np.concatenate(([True], labels[1:] != labels[:-1]))
            runs = labels[changes]
            # Exclude "Other" from counting as an onset if it's there? 
            # No, keep it standard but user likely cares about the 9.
            unique_runs, run_counts = np.unique(runs, return_counts=True)
            onset_map = dict(zip(unique_runs, run_counts))
            
        animal = s.metadata.get("animal_id")
        geno = s.metadata.get("genotype")
        drug = s.metadata.get("drug", "N/A")
        
        for cat in categories:
            rows.append({
                "Animal": animal,
                "Genotype": geno,
                "Drug": drug,
                "Category": cat,
                "Turn": cat.split("_")[0],
                "Vigor": cat.split("_")[1],
                "Time_Pct": time_map.get(cat, 0) / total_frames,
                "Onset_Count": onset_map.get(cat, 0)
            })
            
    return pd.DataFrame(rows)

def calculate_advanced_stats(session):
    """
    Compute session-level averages for Advanced Kinematics.
    Returns dict: {syllable: {stats...}}
    """
    df = session.all_data_df
    if df.empty or "syllable" not in df.columns: return {}
    
    # Ensure cols exist
    if "acceleration" not in df.columns:
        # Simple gradient
        # calc fs?
        ts = df["SystemTimestamp"].dropna().values
        if len(ts) > 1:
            dt = np.nanmedian(np.diff(ts))
            fs = 1/dt if dt > 0 else 30.0
        else: fs = 30.0
        df["acceleration"] = np.gradient(df["velocity_px_s"], 1/fs)
        
    # Instance grouping
    df["instance_id"] = (df["syllable"] != df["syllable"].shift(1)).cumsum()
    
    # Aggregation
    stats = df.groupby(["syllable", "instance_id"]).agg(
        duration_frames=("syllable", "count"),
        mean_vel=("velocity_px_s", "mean"),
        p95_accel=("acceleration", lambda x: np.percentile(x, 95)),
        path_length=("velocity_px_s", "sum"), # * dt later
        start_x=("centroid_x", "first"),
        end_x=("centroid_x", "last"),
        start_y=("centroid_y", "first"),
        end_y=("centroid_y", "last")
    ).reset_index()
    
    # Convert frames to seconds (approx)
    # We need fs again. passing fs is cleaner but let's assume 30 or calc
    stats["duration_s"] = stats["duration_frames"] / 30.0 
    
    # Tortuosity
    # displacement
    dx = stats["end_x"] - stats["start_x"]
    dy = stats["end_y"] - stats["start_y"]
    disp = np.sqrt(dx**2 + dy**2)
    path = stats["path_length"] / 30.0 # px/s * frames / fs? No. sum(vel) * df? vel=px/s. sum(vel)*1/fs = px.
    stats["path_dist"] = stats["path_length"] / 30.0
    # Fix 17: Use Straightness (Disp/Path) to avoid infinity. 
    # Range [0,1]. 1=Straight, 0=Loop/Stationary.
    stats["straightness"] = disp / (stats["path_dist"] + 1e-6)
    
    # Session Level Agg
    # Group by Syllable
    syll_agg = stats.groupby("syllable").agg(
        mean_duration=("duration_s", "mean"),
        std_duration=("duration_s", "std"),
        mean_velocity=("mean_vel", "mean"),
        mean_explosiveness=("p95_accel", "mean"),
        mean_straightness=("straightness", "mean")
    ).reset_index()
    
    syll_agg["duration_cv"] = syll_agg["std_duration"] / (syll_agg["mean_duration"] + 1e-9)
    
    return syll_agg

def get_fingerprint_data(sessions, top_n=5, min_instances: int = 0):
    """
    Get normalized metrics for top N syllables across all sessions.
    Returns DataFrame ready for Radar Plot.
    """
    all_stats = []
    
    for s in sessions:
        stats = calculate_advanced_stats(s)
        if stats.empty: continue
        
        # Count instances in the original session data to filter the stats
        inst_counts = get_syllable_instances(s.all_data_df)
        
        # Filter stats by robust syllables
        if min_instances > 0:
            robust_sylls = inst_counts[inst_counts >= min_instances].index
            stats = stats[stats["syllable"].isin(robust_sylls)]
            
        if stats.empty: continue
        
        stats["Animal"] = s.metadata.get("animal_id")
        stats["Genotype"] = s.metadata.get("genotype")
        stats["Drug"] = s.metadata.get("drug", "N/A")
        all_stats.append(stats)
        
    if not all_stats: return pd.DataFrame()
    df = pd.concat(all_stats, ignore_index=True)
    
    # Filter Top N Global (from the remaining robust syllables)
    top_sylls = df["syllable"].value_counts().head(top_n).index
    df = df[df["syllable"].isin(top_sylls)]
    
    return df

def load_manual_labels(csv_path: str) -> Dict[str, Dict[int, str]]:
    """
    Load manual labels from CSV.
    Returns: {project_name: {syll_id: label}}
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Manual labels file not found at {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    # Expected cols: project, syllable, label
    mappings = {}
    if "project" in df.columns:
        projects = df["project"].unique()
    else:
        # Fallback if single project file
        projects = ["UNK"]
        
    for proj in projects:
        if proj == "UNK":
            sub = df
        else:
            sub = df[df["project"] == proj]
            
        mappings[proj] = {}
        for _, row in sub.iterrows():
            if pd.notna(row["label"]) and str(row["label"]).strip() != "":
                mappings[proj][int(row["syllable"])] = str(row["label"]).strip()
    return mappings

def get_manual_group_usage(sessions, mapping: Dict[int, str]) -> pd.DataFrame:
    """
    Calculate usage aggregated by manual labels strings.
    Returns DataFrame with 'Time_Pct' and 'Onset_Count'.
    """
    import pandas as pd
    rows = []
    
    for s in sessions:
        if "syllable" not in s.all_data_df.columns: continue
        
        # optimized mapping
        # Create a series for mapping (faster than dict lookup in loop)
        syll_series = s.all_data_df["syllable"].fillna(-1)
        
        # We need a robust map that handles checks
        # convert mapping keys to matching type (float/int) if needed
        # Syllables are usually floats in pandas if NaNs exist
        
        # Map to strings
        # Use simple map with a default? Pandas map is cleanest
        # But we need to handle unmapped
        
        ids = syll_series.values
        labels = np.array([mapping.get(int(x), "Other") for x in ids])
        
        # 1. Time (Frame Counts)
        # ---------------------
        # Value counts on the label array
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_frames = len(labels)
        time_map = dict(zip(unique_labels, counts))
        
        # 2. Onsets (Bout Counts)
        # -----------------------
        # Reduce to runs: e.g. [A, A, B, A] -> [A, B, A]
        if len(labels) > 0:
            changes = np.concatenate(([True], labels[1:] != labels[:-1]))
            runs = labels[changes]
            unique_runs, run_counts = np.unique(runs, return_counts=True)
            onset_map = dict(zip(unique_runs, run_counts))
        else:
            onset_map = {}
            
        # Compile Row
        animal = s.metadata.get("animal_id")
        geno = s.metadata.get("genotype")
        drug = s.metadata.get("drug", "N/A")
        
        # Iterate over all specific labels found + any that might be in mapping 
        # (to ensure 0s are logged? maybe just what's found)
        all_found = set(time_map.keys()) | set(onset_map.keys())
        
        for lbl in all_found:
            rows.append({
                "Animal": animal,
                "Genotype": geno,
                "Drug": drug,
                "Label": lbl,
                "Time_Pct": time_map.get(lbl, 0) / total_frames,
                "Onset_Count": onset_map.get(lbl, 0)
            })
            
    return pd.DataFrame(rows)

def get_manual_group_transition_matrix(sessions, mapping: Dict[int, str], return_distributions: bool = False) -> pd.DataFrame:
    """
    Transition matrix between manual labels.
    Collapses sequences based on labels (e.g. 7->22 both 'dash' becomes 'dash'->'dash' which collapses to single state).
    If return_distributions is True, returns (mean_matrix, session_matrices_list).
    """
    import numpy as np
    
    unique_labels = sorted(list(set(mapping.values())))
    # Add 'Other' if we want complete transitions, or ignore it?
    # User asked for "how plots look with THESE namings". 
    # Usually better to exclude 'Other' from the heatmap to focus on the named dynamics.
    
    matrix_template = pd.DataFrame(0.0, index=unique_labels, columns=unique_labels)
    session_matrices = []
    
    for s in sessions:
        if "syllable" not in s.all_data_df.columns: continue
        seq_raw = s.all_data_df["syllable"].fillna(-1).to_numpy()
        
        # Map to labels
        # Use list comprehension or vector map. Vector map is faster.
        # But dictionary lookup on numpy array? 
        # Fast way:
        mapped_seq = []
        for x in seq_raw:
            mapped_seq.append(mapping.get(int(x), "Other"))
        mapped_seq = np.array(mapped_seq)
        
        # Collapse self-transitions on LABELS
        mask = np.concatenate(([True], mapped_seq[1:] != mapped_seq[:-1]))
        trans = mapped_seq[mask]
        
        if len(trans) < 2: continue
        
        # Count bigrams for this session
        local_counts = matrix_template.copy()
        
        # Map to integer indices for speed
        lbl_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        mapped = np.array([lbl_to_idx.get(x, -1) for x in trans])
        
        srcs = mapped[:-1]
        dsts = mapped[1:]
        
        valid_mask = (srcs != -1) & (dsts != -1)
        valid_srcs = srcs[valid_mask]
        valid_dsts = dsts[valid_mask]
        
        np.add.at(local_counts.values, (valid_srcs, valid_dsts), 1)
        
        # Normalize
        row_sums = local_counts.sum(axis=1)
        local_prob = local_counts.div(row_sums.replace(0, 1), axis=0)
        session_matrices.append(local_prob)
                
    if not session_matrices:
        m = pd.DataFrame(0.0, index=unique_labels, columns=unique_labels)
        return (m, []) if return_distributions else m
        
    # Average across sessions
    concatenated = np.stack([m.values for m in session_matrices])
    mean_matrix_vals = np.nanmean(concatenated, axis=0)
    
    mean_matrix = pd.DataFrame(mean_matrix_vals, index=unique_labels, columns=unique_labels)
    
    if return_distributions:
        return mean_matrix, session_matrices
    return mean_matrix
