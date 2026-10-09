"""
Vibe Passports Visualization Module
Located at syllable_generation/analysis/language_models/lib/visualization/vibe_passports.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_vibe_passports_figure(
    session_dfs: dict,
    frame_strat_dict: dict,
    n_topics: int,
    save_path: str
):
    """
    Generates a multi-panel Vibe Passport figure displaying the kinematic profile (Speed, Distance to Center, Turning Rate, Cadence)
    for each strategy cluster.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Collect frame-level data per strategy
    strat_records = []

    for s_id, df in session_dfs.items():
        if s_id not in frame_strat_dict:
            continue
        strats = frame_strat_dict[s_id]
        n = min(len(df), len(strats))
        sub_df = df.iloc[:n]
        sub_strats = strats[:n]

        vel = sub_df["velocity_px_s"].values if "velocity_px_s" in sub_df else np.zeros(n)
        cx = sub_df["centroid_x"].values if "centroid_x" in sub_df else np.zeros(n)
        cy = sub_df["centroid_y"].values if "centroid_y" in sub_df else np.zeros(n)
        c_mean_x, c_mean_y = np.nanmean(cx), np.nanmean(cy)
        dist_c = np.sqrt((cx - c_mean_x) ** 2 + (cy - c_mean_y) ** 2)
        ang = np.abs(sub_df["angular_velocity"].values) if "angular_velocity" in sub_df else np.zeros(n)

        sylls = sub_df["syllable"].values if "syllable" in sub_df else np.zeros(n)
        cadence = np.zeros(n, dtype=float)
        if len(sylls) > 1:
            cadence[1:] = (sylls[1:] != sylls[:-1]).astype(float)

        for i in range(n):
            st = sub_strats[i]
            if st >= 0:  # Exclude noise
                strat_records.append({
                    "strategy_id": int(st),
                    "velocity": vel[i],
                    "dist_center": dist_c[i],
                    "angular_vel": ang[i],
                    "cadence": cadence[i]
                })

    df_all = pd.DataFrame(strat_records)
    if df_all.empty:
        print("Warning: No frame strategy records found for Vibe Passports.")
        return

    # Group by strategy_id
    grouped = df_all.groupby("strategy_id").agg({
        "velocity": ["mean", "sem"],
        "dist_center": ["mean", "sem"],
        "angular_vel": ["mean", "sem"],
        "cadence": ["mean", "sem"]
    })

    unique_strats = sorted(df_all["strategy_id"].unique())
    n_strats = len(unique_strats)

    # Setup Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("9-Strategy Kinematic Passports ('Behavioral Mood Profiles')", fontsize=16, fontweight="bold")

    colors = sns.color_palette("tab10", n_colors=n_strats)

    # 1. Locomotion Speed
    ax1 = axes[0, 0]
    means_vel = [grouped.loc[s, ("velocity", "mean")] for s in unique_strats]
    sems_vel = [grouped.loc[s, ("velocity", "sem")] for s in unique_strats]
    ax1.bar([f"Vibe {s}" for s in unique_strats], means_vel, yerr=sems_vel, color=colors, capsize=4)
    ax1.set_title("Average Locomotion Speed (px/s)", fontweight="bold")
    ax1.set_ylabel("Velocity (px/s)")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Distance to Center
    ax2 = axes[0, 1]
    means_dist = [grouped.loc[s, ("dist_center", "mean")] for s in unique_strats]
    sems_dist = [grouped.loc[s, ("dist_center", "sem")] for s in unique_strats]
    ax2.bar([f"Vibe {s}" for s in unique_strats], means_dist, yerr=sems_dist, color=colors, capsize=4)
    ax2.set_title("Distance to Arena Center (Thigmotaxis)", fontweight="bold")
    ax2.set_ylabel("Distance (px)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Turning Rate
    ax3 = axes[1, 0]
    means_ang = [grouped.loc[s, ("angular_vel", "mean")] for s in unique_strats]
    sems_ang = [grouped.loc[s, ("angular_vel", "sem")] for s in unique_strats]
    ax3.bar([f"Vibe {s}" for s in unique_strats], means_ang, yerr=sems_ang, color=colors, capsize=4)
    ax3.set_title("Average Turning Rate (|Angular Velocity|)", fontweight="bold")
    ax3.set_ylabel("Angular Speed (rad/s)")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # 4. Syllable Cadence
    ax4 = axes[1, 1]
    means_cad = [grouped.loc[s, ("cadence", "mean")] for s in unique_strats]
    sems_cad = [grouped.loc[s, ("cadence", "sem")] for s in unique_strats]
    ax4.bar([f"Vibe {s}" for s in unique_strats], means_cad, yerr=sems_cad, color=colors, capsize=4)
    ax4.set_title("Syllable Transition Cadence (Switches/s)", fontweight="bold")
    ax4.set_ylabel("Transition Rate")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Vibe Passports figure to: {save_path}")
