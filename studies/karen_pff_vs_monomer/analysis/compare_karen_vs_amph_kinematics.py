#!/usr/bin/env python3
"""
compare_karen_vs_amph_kinematics.py

Compares kinematics and syllable statistics side-by-side between:
1. Karen Final Model (sessions_karen_project_w_sylls.pkl)
2. AMPH v2 Model (jonathan_amph_sessions_v2_w_sylls.pkl)
"""

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pickle
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import pandas as pd

KAREN_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
AMPH_PKL = REPO_ROOT / "syllable_generation" / "data" / "jonathan_amph_sessions_v2_w_sylls.pkl"


def compute_metrics(sessions, name="Dataset", fps=80.0):
    print(f"\nComputing metrics for {name} ({len(sessions)} sessions)...")
    
    # Collect data across sessions
    all_dfs = [s.all_data_df for s in sessions if hasattr(s, "all_data_df") and s.all_data_df is not None]
    
    # 1. Total time and frame statistics
    total_frames = sum(len(df) for df in all_dfs)
    total_hours = total_frames / (fps * 3600.0)
    
    # Sample 100k valid rows across sessions for fast yet accurate distribution stats
    valid_dfs = [df.dropna(subset=['syllable', 'centroid_x', 'velocity_px_s']) for df in all_dfs]
    total_valid_frames = sum(len(df) for df in valid_dfs)
    coverage_pct = (total_valid_frames / total_frames) * 100.0
    
    # Concatenate a large pooled sample (up to 500,000 rows across sessions)
    sample_size_per_sess = min(10000, max(1000, 500000 // len(valid_dfs)))
    pooled_sample = pd.concat([df.sample(min(len(df), sample_size_per_sess), random_state=42) for df in valid_dfs if len(df) > 0])
    
    # Kinematics
    # Position
    x = pooled_sample['centroid_x'].values
    y = pooled_sample['centroid_y'].values
    # Speed
    speed = pooled_sample['velocity_px_s'].values
    # Heading & Angular velocity
    heading = pooled_sample['heading'].values
    ang_vel = pooled_sample['angular_velocity'].values
    
    # Syllable transitions & durations
    # Compute onsets and duration across all sessions
    syll_durations_frames = []
    sylls_per_minute_list = []
    all_syll_counts = pd.Series(dtype=int)
    
    for df in all_dfs:
        s_series = df['syllable'].dropna()
        if len(s_series) == 0:
            continue
        
        # Syllable counts
        all_syll_counts = all_syll_counts.add(s_series.value_counts(), fill_value=0)
        
        # Transitions / Onsets
        changes = (s_series != s_series.shift(1)).values
        changes[0] = True
        num_onsets = changes.sum()
        session_valid_minutes = len(s_series) / (fps * 60.0)
        if session_valid_minutes > 0:
            sylls_per_minute_list.append(num_onsets / session_valid_minutes)
        
        # Run lengths (durations in 80Hz frames)
        rle = np.diff(np.where(np.concatenate(([True], changes[1:], [True])))[0])
        syll_durations_frames.extend(rle)
        
    syll_durations_sec = np.array(syll_durations_frames) / fps
    all_syll_counts = all_syll_counts.sort_values(ascending=False)
    
    # Entropy of syllable distribution
    probs = all_syll_counts / all_syll_counts.sum()
    entropy = -(probs * np.log2(probs + 1e-12)).sum()
    participation_ratio = 1.0 / (probs**2).sum()
    
    return {
        "name": name,
        "n_sessions": len(sessions),
        "total_frames": total_frames,
        "total_hours": total_hours,
        "valid_frames": total_valid_frames,
        "coverage_pct": coverage_pct,
        # Centroid X
        "x_mean": np.mean(x),
        "x_std": np.std(x),
        # Centroid Y
        "y_mean": np.mean(y),
        "y_std": np.std(y),
        # Speed
        "speed_mean": np.mean(speed),
        "speed_std": np.std(speed),
        "speed_median": np.median(speed),
        "speed_p95": np.percentile(speed, 95),
        # Heading
        "heading_mean": np.mean(heading),
        "heading_std": np.std(heading),
        # Angular Velocity
        "ang_vel_mean": np.mean(ang_vel),
        "ang_vel_std": np.std(ang_vel),
        "ang_vel_abs_median": np.median(np.abs(ang_vel)),
        # Syllable dynamics
        "sylls_per_min_mean": np.mean(sylls_per_minute_list),
        "sylls_per_min_std": np.std(sylls_per_minute_list),
        "median_duration_ms": np.median(syll_durations_sec) * 1000.0,
        "mean_duration_ms": np.mean(syll_durations_sec) * 1000.0,
        "p90_duration_ms": np.percentile(syll_durations_sec, 90) * 1000.0,
        "n_unique_syllables": len(all_syll_counts),
        "syllable_entropy_bits": entropy,
        "participation_ratio": participation_ratio,
        "top5_syllables_pct": (all_syll_counts.head(5).sum() / all_syll_counts.sum()) * 100.0,
        "top10_syllables_pct": (all_syll_counts.head(10).sum() / all_syll_counts.sum()) * 100.0,
    }


def main():
    with open(KAREN_PKL, "rb") as f:
        karen_sessions = pickle.load(f)
        
    with open(AMPH_PKL, "rb") as f:
        amph_sessions = pickle.load(f)
        
    karen_m = compute_metrics(karen_sessions, "Karen Final Model", fps=80.0)
    amph_m = compute_metrics(amph_sessions, "AMPH v2 Reference", fps=80.0)
    
    print("\n" + "=" * 85)
    print(f"{'METRIC':<35} | {'KAREN FINAL MODEL':<22} | {'AMPH V2 REFERENCE':<22}")
    print("=" * 85)
    
    rows = [
        ("Sessions count", f"{karen_m['n_sessions']}", f"{amph_m['n_sessions']}"),
        ("Total Recording Time", f"{karen_m['total_hours']:.1f} hours", f"{amph_m['total_hours']:.1f} hours"),
        ("Model Timeline Coverage", f"{karen_m['coverage_pct']:.1f} %", f"{amph_m['coverage_pct']:.1f} %"),
        ("-" * 35, "-" * 22, "-" * 22),
        ("Position X (Mean +/- Std) [px]", f"{karen_m['x_mean']:.1f} +/- {karen_m['x_std']:.1f}", f"{amph_m['x_mean']:.1f} +/- {amph_m['x_std']:.1f}"),
        ("Position Y (Mean +/- Std) [px]", f"{karen_m['y_mean']:.1f} +/- {karen_m['y_std']:.1f}", f"{amph_m['y_mean']:.1f} +/- {amph_m['y_std']:.1f}"),
        ("-" * 35, "-" * 22, "-" * 22),
        ("Speed (Mean +/- Std) [px/s]", f"{karen_m['speed_mean']:.1f} +/- {karen_m['speed_std']:.1f}", f"{amph_m['speed_mean']:.1f} +/- {amph_m['speed_std']:.1f}"),
        ("Speed Median [px/s]", f"{karen_m['speed_median']:.1f}", f"{amph_m['speed_median']:.1f}"),
        ("Speed 95th percentile [px/s]", f"{karen_m['speed_p95']:.1f}", f"{amph_m['speed_p95']:.1f}"),
        ("-" * 35, "-" * 22, "-" * 22),
        ("Heading (Mean +/- Std) [rad]", f"{karen_m['heading_mean']:.2f} +/- {karen_m['heading_std']:.2f}", f"{amph_m['heading_mean']:.2f} +/- {amph_m['heading_std']:.2f}"),
        ("Angular Velocity Std [rad/s]", f"{karen_m['ang_vel_std']:.2f}", f"{amph_m['ang_vel_std']:.2f}"),
        ("Angular Velocity Abs Median [rad/s]", f"{karen_m['ang_vel_abs_median']:.2f}", f"{amph_m['ang_vel_abs_median']:.2f}"),
        ("-" * 35, "-" * 22, "-" * 22),
        ("Syllables per Minute (Mean +/- Std)", f"{karen_m['sylls_per_min_mean']:.1f} +/- {karen_m['sylls_per_min_std']:.1f}", f"{amph_m['sylls_per_min_mean']:.1f} +/- {amph_m['sylls_per_min_std']:.1f}"),
        ("Median Syllable Duration", f"{karen_m['median_duration_ms']:.1f} ms", f"{amph_m['median_duration_ms']:.1f} ms"),
        ("Mean Syllable Duration", f"{karen_m['mean_duration_ms']:.1f} ms", f"{amph_m['mean_duration_ms']:.1f} ms"),
        ("90th Pct Syllable Duration", f"{karen_m['p90_duration_ms']:.1f} ms", f"{amph_m['p90_duration_ms']:.1f} ms"),
        ("-" * 35, "-" * 22, "-" * 22),
        ("Number of Active Syllables", f"{karen_m['n_unique_syllables']}", f"{amph_m['n_unique_syllables']}"),
        ("Syllable Entropy [bits]", f"{karen_m['syllable_entropy_bits']:.2f} bits", f"{amph_m['syllable_entropy_bits']:.2f} bits"),
        ("Participation Ratio (Effective N)", f"{karen_m['participation_ratio']:.1f}", f"{amph_m['participation_ratio']:.1f}"),
        ("Top 5 Syllables Usage", f"{karen_m['top5_syllables_pct']:.1f} %", f"{amph_m['top5_syllables_pct']:.1f} %"),
        ("Top 10 Syllables Usage", f"{karen_m['top10_syllables_pct']:.1f} %", f"{amph_m['top10_syllables_pct']:.1f} %"),
    ]
    
    for label, val_k, val_a in rows:
        if label.startswith("-"):
            print("-" * 85)
        else:
            print(f"{label:<35} | {val_k:<22} | {val_a:<22}")
    print("=" * 85)


if __name__ == "__main__":
    main()
