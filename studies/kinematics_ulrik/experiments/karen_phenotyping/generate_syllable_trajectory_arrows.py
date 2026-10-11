#!/usr/bin/env python3
"""
generate_syllable_trajectory_arrows.py

Generates publication-quality 2D trajectory arrow plots for Keypoint-MoSeq syllables.
For each syllable:
  - Canonical egocentric alignment: Translates initial position to (0,0) and rotates
    initial heading to point along +X (forward).
  - Averages trajectory coordinates and heading angles across hundreds of instances.
  - Draws sequential arrows over time:
      * Base of arrow = mouse centroid (x, y)
      * Pointy end = oriented by instantaneous heading angle
      * Arrow color = continuous colormap across time (0% Start -> 100% End)
  - Outputs 1 clean PNG per syllable with Monomer vs. PFF side-by-side comparison.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

from fpcode.common.paths import get_repo_root, resolve_data_path

warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

REPO_ROOT = get_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_PKL = resolve_data_path('data/sessions') / "sessions_karen_project_w_sylls.pkl"
OUT_DIR = resolve_data_path('data/analysis_cache/ulrik_kinematics_output') / "karen_genotype_comparison" / "syllable_trajectory_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-white" if "seaborn-v0_8-white" in plt.style.available else "default")

# Syllables to generate: 6 nominal occupancy states + top speed/behavioral archetypes
TARGET_SYLLABLES = [74, 68, 83, 31, 40, 14, 43, 22, 9, 5, 70]
SYLLABLE_LABELS = {
    74: "Syllable 74: Low-Mobility Postural Pause (Nominal p=0.011, +56.8% PFF)",
    68: "Syllable 68: Low-Mobility Head-Cast / Turn (Nominal p=0.015, +55.2% PFF)",
    83: "Syllable 83: Immobility / Prolonged Grooming (Nominal p=0.021, +142.0% PFF)",
    31: "Syllable 31: Exploration Stepping / Forward Transition (Nominal p=0.029, +40.0% PFF)",
    40: "Syllable 40: High-Speed Locomotor Sprint Run (Nominal p=0.033, -32.4% PFF)",
    14: "Syllable 14: Exploration Stepping / Cornering (Nominal p=0.035, +29.8% PFF)",
    43: "Syllable 43: Fast Forward Locomotion / Sprint (Speed p=0.017, -17.4 px/s PFF)",
    22: "Syllable 22: Turning / Steering Reorientation (Speed p=0.039, -8.1 px/s PFF)",
    9:  "Syllable 9: Exploration Run / Forward Stepping (Speed p=0.079, -3.6 px/s PFF)",
    5:  "Syllable 5: Micro Locomotion / Forward Burst (Speed p=0.007, -6.6 px/s PFF)",
    70: "Syllable 70: Extended Stationary State (Low Velocity Grooming/Pause)"
}

CONDITION_ORDER = ["3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa"]


def extract_aligned_syllable_trajectories(sessions, target_sylls, n_phase_bins=25, max_instances_per_syll=500):
    print(f"[TrajectoryExtractor] Extracting and aligning canonical instances for {len(target_sylls)} syllables...", flush=True)

    # Store aligned trajectories: (x, y, heading) arrays resampled to n_phase_bins
    data_store = {s_id: {"Monomer": [], "PFF": []} for s_id in target_sylls}

    for s in sessions:
        df = s.all_data_df.dropna(subset=["centroid_x", "centroid_y", "velocity_px_s", "syllable"])
        if len(df) < 1000:
            continue
        g = s.metadata["genotype"]
        cond = s.metadata.get("condition", "")
        if cond not in CONDITION_ORDER:
            continue

        x = df["centroid_x"].values
        y = df["centroid_y"].values
        speed = df["velocity_px_s"].values
        sylls = df["syllable"].values.astype(int)

        # Calculate instantaneous heading
        if "heading" in df.columns and not df["heading"].isna().all():
            heading = df["heading"].values
        else:
            dx = np.gradient(x)
            dy = np.gradient(y)
            heading = np.arctan2(dy, dx)

        for s_id in target_sylls:
            if len(data_store[s_id][g]) >= max_instances_per_syll:
                continue

            is_s = (sylls == s_id).astype(int)
            diff_s = np.diff(np.concatenate(([0], is_s, [0])))
            starts = np.where(diff_s == 1)[0]
            ends = np.where(diff_s == -1)[0]

            for st, en in zip(starts, ends):
                length = en - st
                if 5 <= length <= 100:  # Valid syllable duration ~60ms - 1250ms
                    inst_x = x[st:en]
                    inst_y = y[st:en]
                    inst_h = heading[st:en]
                    inst_v = speed[st:en]

                    # 1. Translate to origin (0, 0)
                    x_rel = inst_x - inst_x[0]
                    y_rel = inst_y - inst_y[0]

                    # 2. Rotate so initial heading points along +X (angle = 0)
                    theta_0 = np.median(inst_h[:min(3, length)])
                    cos_t = np.cos(-theta_0)
                    sin_t = np.sin(-theta_0)

                    x_rot = x_rel * cos_t - y_rel * sin_t
                    y_rot = x_rel * sin_t + y_rel * cos_t

                    # Relative heading over time
                    # Unwrap angles to avoid 2pi jumps during rotation
                    h_unwrapped = np.unwrap(inst_h)
                    h_rel = h_unwrapped - theta_0

                    # 3. Resample to standardized phase steps
                    t_orig = np.linspace(0, 1, length)
                    t_target = np.linspace(0, 1, n_phase_bins)

                    res_x = np.interp(t_target, t_orig, x_rot)
                    res_y = np.interp(t_target, t_orig, y_rot)
                    res_h = np.interp(t_target, t_orig, h_rel)
                    res_v = np.interp(t_target, t_orig, inst_v)

                    data_store[s_id][g].append({
                        "x": res_x,
                        "y": res_y,
                        "heading": res_h,
                        "speed": res_v,
                        "duration_ms": (length / 80.0) * 1000.0,
                        "displacement_px": np.sqrt(x_rot[-1]**2 + y_rot[-1]**2)
                    })

    print("[TrajectoryExtractor] Extraction complete across all target syllables.", flush=True)
    return data_store


def plot_syllable_trajectory(s_id, instances_dict, out_dir):
    title = SYLLABLE_LABELS.get(s_id, f"Syllable {s_id}")
    mon_insts = instances_dict["Monomer"]
    pff_insts = instances_dict["PFF"]

    if len(mon_insts) == 0 or len(pff_insts) == 0:
        print(f"  [Warning] Insufficient instances for Syllable {s_id}")
        return

    # Compute mean trajectories
    mon_x = np.mean([inst["x"] for inst in mon_insts], axis=0)
    mon_y = np.mean([inst["y"] for inst in mon_insts], axis=0)
    mon_h = np.mean([inst["heading"] for inst in mon_insts], axis=0)
    mon_v = np.mean([np.mean(inst["speed"]) for inst in mon_insts])
    mon_dur = np.median([inst["duration_ms"] for inst in mon_insts])
    mon_disp = np.mean([inst["displacement_px"] for inst in mon_insts])

    pff_x = np.mean([inst["x"] for inst in pff_insts], axis=0)
    pff_y = np.mean([inst["y"] for inst in pff_insts], axis=0)
    pff_h = np.mean([inst["heading"] for inst in pff_insts], axis=0)
    pff_v = np.mean([np.mean(inst["speed"]) for inst in pff_insts])
    pff_dur = np.median([inst["duration_ms"] for inst in pff_insts])
    pff_disp = np.mean([inst["displacement_px"] for inst in pff_insts])

    # Precomputed Syllable Occupancy & Regulation stats
    OCC_STATS = {
        74: {"mon_pct": 0.486, "pff_pct": 0.762, "shift_pct": +56.8, "p_val": 0.0109, "status": "UPREGULATED IN PFF (+56.8%)", "color": "#d95f02"},
        68: {"mon_pct": 0.416, "pff_pct": 0.645, "shift_pct": +55.2, "p_val": 0.0153, "status": "UPREGULATED IN PFF (+55.2%)", "color": "#d95f02"},
        83: {"mon_pct": 0.108, "pff_pct": 0.262, "shift_pct": +142.0, "p_val": 0.0210, "status": "UPREGULATED IN PFF (+142.0%)", "color": "#d95f02"},
        31: {"mon_pct": 1.859, "pff_pct": 2.603, "shift_pct": +40.0, "p_val": 0.0294, "status": "UPREGULATED IN PFF (+40.0%)", "color": "#d95f02"},
        40: {"mon_pct": 0.982, "pff_pct": 0.663, "shift_pct": -32.4, "p_val": 0.0327, "status": "DOWNREGULATED IN PFF (-32.4%)", "color": "#2b5c8f"},
        14: {"mon_pct": 1.434, "pff_pct": 1.862, "shift_pct": +29.8, "p_val": 0.0351, "status": "UPREGULATED IN PFF (+29.8%)", "color": "#d95f02"},
        43: {"mon_pct": 0.857, "pff_pct": 0.627, "shift_pct": -26.9, "p_val": 0.1171, "status": "TREND DOWN IN PFF (-26.9% / Speed -17.4 px/s)", "color": "#2b5c8f"},
        22: {"mon_pct": 1.405, "pff_pct": 1.299, "shift_pct": -7.5, "p_val": 0.2662, "status": "CONSERVED USAGE (-7.5% / Speed -8.1 px/s)", "color": "#555555"},
        9:  {"mon_pct": 1.057, "pff_pct": 0.872, "shift_pct": -17.5, "p_val": 0.1083, "status": "MODERATE USAGE SHIFT (-17.5% / Speed -3.6 px/s)", "color": "#555555"},
        5:  {"mon_pct": 1.846, "pff_pct": 1.528, "shift_pct": -17.3, "p_val": 0.3872, "status": "MODERATE USAGE SHIFT (-17.3% / Speed -6.6 px/s)", "color": "#555555"},
        70: {"mon_pct": 0.752, "pff_pct": 0.771, "shift_pct": +2.5, "p_val": 0.6841, "status": "CONSERVED USAGE (+2.5%)", "color": "#555555"},
    }

    stat_info = OCC_STATS.get(s_id, {"mon_pct": 0.0, "pff_pct": 0.0, "shift_pct": 0.0, "p_val": 1.0, "status": "EXPLORATORY STATE", "color": "#555555"})

    # -------------------------------------------------------------
    # Figure Setup: 2-Panel Side-by-Side (Monomer vs. PFF)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), facecolor="white")

    # Colormap over time (0% -> 100%)
    cmap = cm.viridis
    n_pts = len(mon_x)
    time_fractions = np.linspace(0, 1, n_pts)
    colors = [cmap(t) for t in time_fractions]

    # Determine common bounding box
    all_x = np.concatenate([mon_x, pff_x])
    all_y = np.concatenate([mon_y, pff_y])
    max_range = max(np.ptp(all_x), np.ptp(all_y), 15.0) * 0.75 + 10.0
    x_center = np.mean(all_x)
    y_center = np.mean(all_y)

    arrow_len = max(max_range * 0.12, 3.5)

    configs = [
        ("Monomer (Control)", axes[0], mon_x, mon_y, mon_h, mon_insts, mon_v, mon_dur, mon_disp, "#2b5c8f", f"Monomer Baseline Usage: {stat_info['mon_pct']:.3f}%"),
        ("PFF (Parkinson's Model)", axes[1], pff_x, pff_y, pff_h, pff_insts, pff_v, pff_dur, pff_disp, "#d95f02", f"PFF Phenotype Usage: {stat_info['pff_pct']:.3f}% ({stat_info['shift_pct']:+.1f}%)")
    ]

    for label, ax, traj_x, traj_y, traj_h, insts, mean_v, med_dur, mean_d, theme_color, occ_label in configs:
        ax.set_facecolor("#fafbfc")
        ax.grid(True, color="#e5e9f0", linestyle="--", linewidth=0.8, alpha=0.8)

        # 1. Plot individual sample trajectories as faint spaghetti lines
        for inst in insts[:40]:
            ax.plot(inst["x"], inst["y"], color=theme_color, alpha=0.06, linewidth=0.8, zorder=1)

        # 2. Plot the mean continuous trajectory line
        ax.plot(traj_x, traj_y, color="#333333", linewidth=1.5, linestyle=":", alpha=0.6, zorder=2)

        # 3. Plot sequential oriented arrows over time
        arrow_indices = np.linspace(0, n_pts - 1, 14, dtype=int)

        for idx_step in arrow_indices:
            cx = traj_x[idx_step]
            cy = traj_y[idx_step]
            theta = traj_h[idx_step]
            c_val = colors[idx_step]

            # Pointy end (Head of arrow)
            hx = cx + arrow_len * np.cos(theta)
            hy = cy + arrow_len * np.sin(theta)

            # Centroid marker (Back base of arrow)
            ax.plot(cx, cy, marker="o", markersize=4.5, color=c_val, markeredgecolor="#222222", markeredgewidth=0.6, zorder=4)

            # Draw FancyArrowPatch from Centroid (cx, cy) to Head (hx, hy)
            arrow = FancyArrowPatch(
                (cx, cy), (hx, hy),
                arrowstyle="-|>,head_length=5.0,head_width=3.5",
                color=c_val,
                linewidth=2.0,
                alpha=0.9,
                zorder=5
            )
            ax.add_patch(arrow)

        # Highlight Start (t=0) and End (t=100%)
        ax.plot(traj_x[0], traj_y[0], marker="s", markersize=8, color="#2ca02c", markeredgecolor="black", markeredgewidth=1.2, label="Start (0%)", zorder=6)
        ax.plot(traj_x[-1], traj_y[-1], marker="X", markersize=9, color="#d62728", markeredgecolor="black", markeredgewidth=1.2, label="End (100%)", zorder=6)

        # Subpanel Titles & Occupancy Badge
        ax.set_title(f"{label}\n{occ_label}", fontsize=11.0, fontweight="bold", pad=8, color=theme_color)

        info_text = (
            f"Mean Speed: {mean_v:.1f} px/s\n"
            f"Median Duration: {med_dur:.0f} ms\n"
            f"Net Displacement: {mean_d:.1f} px\n"
            f"Net Turning Angle: {np.degrees(traj_h[-1] - traj_h[0]):.1f}°"
        )
        ax.text(
            0.03, 0.96, info_text,
            transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.92, zorder=10)
        )

        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(y_center - max_range, y_center + max_range)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Aligned Centroid X (px) [→ Initial Heading]", fontsize=9.5)
        ax.set_ylabel("Aligned Centroid Y (px)", fontsize=9.5)
        if label.startswith("Monomer"):
            ax.legend(loc="lower left", fontsize=8.0, framealpha=0.9)

    # Prominent Upregulation / Downregulation Banner at Top Center
    banner_text = f"GENOTYPE USAGE STATUS:  {stat_info['status']}  (Nominal p = {stat_info['p_val']:.4f}, FDR q = 0.585)"
    fig.text(
        0.5, 0.935, banner_text,
        ha="center", va="center",
        fontsize=10.0, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=stat_info["color"], edgecolor="none", alpha=0.95, zorder=20)
    )

    # Add horizontal Colorbar for Time (0% -> 100%)
    cbar_ax = fig.add_axes([0.30, 0.04, 0.40, 0.025])
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation="horizontal")
    cb.set_label("Syllable Execution Time / Phase: 0% (Start)  ──►  100% (End)", fontsize=9.0, fontweight="bold")
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    plt.suptitle(title, fontsize=12.5, fontweight="bold", y=0.985)
    plt.subplots_adjust(bottom=0.14, top=0.87, wspace=0.20)

    out_file = out_dir / f"syllable_{s_id:02d}_trajectory_arrow.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved trajectory arrow plot: {out_file.name}", flush=True)


def main():
    print("="*70)
    print("   GENERATING KEYPOINT-MOSEQ SYLLABLE TRAJECTORY ARROW PLOTS")
    print("="*70, flush=True)

    with open(DATA_PKL, "rb") as f:
        all_sessions = pickle.load(f)

    sessions = [s for s in all_sessions if s.metadata.get("genotype") in ["PFF", "Monomer"] and s.metadata.get("condition") in CONDITION_ORDER]
    print(f"Loaded {len(sessions)} valid sessions.", flush=True)

    data_store = extract_aligned_syllable_trajectories(sessions, TARGET_SYLLABLES, n_phase_bins=30)

    print("\nRendering individual trajectory arrow PNGs per syllable...")
    for s_id in TARGET_SYLLABLES:
        plot_syllable_trajectory(s_id, data_store[s_id], OUT_DIR)

    print("\n>>> All Syllable Trajectory Arrow Plots Generated Successfully!")
    print(f">>> Output Directory: {OUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
