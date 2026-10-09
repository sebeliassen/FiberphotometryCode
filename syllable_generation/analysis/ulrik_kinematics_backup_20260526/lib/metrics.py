from typing import Any, Dict

import numpy as np
import pandas as pd
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
        fs = session.metadata.get("fs", 80.0)
        # Fix 4: Smooth velocity before differentiation to avoid noise amplification
        vel_smooth = gaussian_filter1d(vel, sigma=2)
        if len(vel_smooth) > 1:
            accel = np.gradient(vel_smooth, 1/fs)
            res["mean_acceleration"] = np.mean(np.abs(accel))
            res["p95_acceleration"] = np.percentile(np.abs(accel), 95)
            res["explosiveness"] = res["p95_acceleration"]
        else:
            res["mean_acceleration"] = np.nan
            res["p95_acceleration"] = np.nan
            res["explosiveness"] = np.nan
    else:
        res["mean_velocity"] = np.nan
        res["mean_acceleration"] = np.nan
        res["explosiveness"] = np.nan

    # Straightness
    if "centroid_x" in df.columns and "centroid_y" in df.columns and "step_dist" in df.columns:
        valid_df = df.dropna(subset=["centroid_x", "centroid_y", "step_dist"])
        if not valid_df.empty:
            dx = valid_df["centroid_x"].iloc[-1] - valid_df["centroid_x"].iloc[0]
            dy = valid_df["centroid_y"].iloc[-1] - valid_df["centroid_y"].iloc[0]
            disp = np.sqrt(dx**2 + dy**2)
            path_length = valid_df["step_dist"].sum()
            res["straightness"] = disp / (path_length + 1e-6)
        else:
            res["straightness"] = np.nan
    else:
        res["straightness"] = np.nan

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

    if "syllable" in session.all_data_df.columns:
        s = session.all_data_df["syllable"].fillna(-1).to_numpy()
        changes = np.concatenate(([True], s[1:] != s[:-1], [True]))
        change_indices = np.flatnonzero(changes)
        durations = np.diff(change_indices)
        values = s[change_indices[:-1]]
        valid_durations = durations[values != -1]

        # Calculate mean valid bout duration in seconds
        fs = meta.get("fs", 80.0)
        if len(valid_durations) > 0:
            res["duration"] = np.mean(valid_durations) / fs
        else:
            res["duration"] = np.nan
    else:
        res["duration"] = np.nan

    return res

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


# Re-export from split modules for backward compatibility

