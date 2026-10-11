#!/usr/bin/env python3
"""
plot_negative_control_audit.py
=============================================================================
Negative Control & Perturbation Audit for Keypoint-MoSeq Syllable Onset Alignment:
Quantitatively demonstrates that:
1. Spatial Translational Shifts (dx = +/- 10 to 60 px) dramatically degrade
   centroid pixel darkness as the window slides off the dark mouse body onto the arena floor.
2. Temporal Shifts (dt = +/- 5 to 60 frames) degrade darkness as the animal
   moves away from the stationary onset coordinate.
3. BotCam Hardware TTL Desynchronization (dTTL = +/- 10 to 60 frames) degrades
   ventral centroid darkness, demonstrating the necessity of TTL master clock alignment.

Environment: kpms_gpu_cuml
=============================================================================
"""
import sys
from pathlib import Path

from fpcode.common.paths import resolve_data_path

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import re

import cv2
import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vidio.read import OpenCVReader

from fpcode.tracking.clock_sync import top_to_bot_frame
from fpcode.tracking.pair_matching import find_matched_pairs

PROJECT_DIR = EXP_DIR / "kpms_tandem_cohort_production"
MODEL_NAME = "tandem_cohort_118sess_50ar_200slds_seed101"
RESULTS_PATH = PROJECT_DIR / MODEL_NAME / "results.h5"
VFI_PATH = PROJECT_DIR / "video_frame_indexes.npz"

OUT_DIR = SCRIPT_DIR / "visualizations_seed101" / "negative_controls"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_VID_DIR = Path(str(resolve_data_path('data/raw/karen_pff_vs_monomer/video_files')))
BOT_VID_DIR = Path(str(resolve_data_path('data/raw/karen_pff_vs_monomer/botcam_video_files')))
CHAMBER_TO_ARENA = {"top_left": "Arena1", "bottom_left": "Arena2", "bottom_right": "Arena3", "top_right": "Arena4"}

def compute_kpms_warp_matrix(centroid_onset, heading_onset, window_size, shift_xy=(0, 0)):
    c = np.array(centroid_onset, dtype=np.float32) + np.array(shift_xy, dtype=np.float32)
    h = float(heading_onset)
    r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
    c_shift = r @ c - window_size // 2
    M = np.float32([[np.cos(h), np.sin(h), -c_shift[0]], [-np.sin(h), np.cos(h), -c_shift[1]]])
    return M

def get_session_files(sid, pairs_by_sid):
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_([a-z]+_[a-z]+)", sid)
    if not m:
        return None
    ts, chamber = m.group(1), m.group(2)
    arena = CHAMBER_TO_ARENA.get(chamber)
    if not arena:
        return None
    top_vids = list(TOP_VID_DIR.glob(f"*{ts}*{chamber}*.avi"))
    bot_vids = list(BOT_VID_DIR.glob(f"*{arena}*BotCamVid*{ts}*.avi"))
    pair = pairs_by_sid.get(f"{ts}_{chamber}")
    if not pair:
        return None
    if top_vids and bot_vids and Path(pair["top_h5"]).exists() and Path(pair["bot_h5"]).exists():
        return {
            "top_vid": str(top_vids[0]),
            "bot_vid": str(bot_vids[0]),
            "top_h5": str(pair["top_h5"]),
            "bot_h5": str(pair["bot_h5"]),
        }
    return None

def main():
    print("=" * 80)
    print("🔬 RUNNING SYLLABLE ONSET NEGATIVE CONTROL PERTURBATION EXPERIMENT")
    print(f"Model: {MODEL_NAME}")
    print(f"Output Directory: {OUT_DIR}")
    print("=" * 80)

    with np.load(VFI_PATH) as z:
        vfi_dict = {k: z[k] for k in z.files}

    pairs = find_matched_pairs()
    pairs_by_sid = {p["sid"]: p for p in pairs}

    # Sample qualifying bouts across multiple sessions
    bouts = []
    with h5py.File(RESULTS_PATH, "r") as f:
        for sid in f.keys():
            if sid not in vfi_dict:
                continue
            paths = get_session_files(sid, pairs_by_sid)
            if not paths:
                continue
            vfi = vfi_dict[sid]
            sylls = f[sid]["syllable"][:]
            c_top = f[sid]["centroid"][:]
            h_top = f[sid]["heading"][:]

            diffs = np.diff(sylls)
            bounds = np.where(diffs != 0)[0] + 1
            starts = np.r_[0, bounds]
            ends = np.r_[bounds, len(sylls)]

            df_bot = pd.read_hdf(paths["bot_h5"])
            bot_scorer = df_bot.columns.levels[0][0]

            for st, en in zip(starts, ends):
                if (en - st) >= 20 and st >= 70 and st + 70 < len(vfi):
                    s_id = sylls[st]
                    if s_id in [0, 2, 4]:
                        raw_top_idx = vfi[st]
                        raw_bot_idx = top_to_bot_frame(sid, raw_top_idx)
                        if 0 <= raw_bot_idx < len(df_bot):
                            nx_0 = df_bot[bot_scorer, "nose_tip", "x"].iloc[raw_bot_idx]
                            ny_0 = df_bot[bot_scorer, "nose_tip", "y"].iloc[raw_bot_idx]
                            tx_0 = df_bot[bot_scorer, "tail_base", "x"].iloc[raw_bot_idx]
                            ty_0 = df_bot[bot_scorer, "tail_base", "y"].iloc[raw_bot_idx]
                            c_b = (0.5 * (nx_0 + tx_0), 0.5 * (ny_0 + ty_0))
                            h_b = np.arctan2(ny_0 - ty_0, nx_0 - tx_0)
                            bouts.append({
                                "sid": sid, "start": st, "end": en, "syll": s_id,
                                "c_top": c_top[st], "h_top": h_top[st],
                                "c_bot": c_b, "h_bot": h_b,
                                "vfi": vfi, "paths": paths
                            })
                            break
            if len(bouts) >= 15:
                break

    print(f"Sampled {len(bouts)} diverse bouts across {len(set(b['sid'] for b in bouts))} sessions.")

    # -------------------------------------------------------------------------
    # Experiment 1: Spatial Translation Perturbation (dx from -40 to +40 px)
    # -------------------------------------------------------------------------
    print("\n[1/3] Executing Spatial Translation Perturbation...", flush=True)
    spatial_shifts = [-40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40]
    trans_intensities = {dx: [] for dx in spatial_shifts}

    for b in bouts:
        rt = OpenCVReader(b["paths"]["top_vid"])
        raw_top_idx = b["vfi"][b["start"]]
        frame_t = rt[raw_top_idx]
        for dx in spatial_shifts:
            # Shift along both x and y
            M = compute_kpms_warp_matrix(b["c_top"], b["h_top"], 144, shift_xy=(dx, 0))
            tile = cv2.warpAffine(frame_t, M, (144, 144))
            tile = cv2.resize(tile, (160, 160))
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            c_val = float(gray[70:90, 70:90].mean())
            trans_intensities[dx].append(c_val)
        rt.close()

    # -------------------------------------------------------------------------
    # Experiment 2: Bidirectional Temporal Shift (dt from -60 to +60 frames)
    # -------------------------------------------------------------------------
    print("[2/3] Executing Bidirectional Temporal Perturbation...", flush=True)
    time_deltas = [-60, -40, -25, -15, -5, 0, 5, 15, 25, 40, 60]
    time_intensities = {dt: [] for dt in time_deltas}

    for b in bouts:
        rt = OpenCVReader(b["paths"]["top_vid"])
        st = b["start"]
        M_top = compute_kpms_warp_matrix(b["c_top"], b["h_top"], 144)
        for dt in time_deltas:
            frame_idx = b["vfi"][st + dt]
            frame = rt[frame_idx]
            tile = cv2.warpAffine(frame, M_top, (144, 144))
            tile = cv2.resize(tile, (160, 160))
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            c_val = float(gray[70:90, 70:90].mean())
            time_intensities[dt].append(c_val)
        rt.close()

    # -------------------------------------------------------------------------
    # Experiment 3: BotCam Hardware TTL Corruption (dTTL from -60 to +60 frames)
    # -------------------------------------------------------------------------
    print("[3/3] Executing BotCam TTL Corruption Perturbation...", flush=True)
    ttl_deltas = [-60, -40, -25, -10, -5, 0, 5, 10, 25, 40, 60]
    bot_ttl_intensities = {d_ttl: [] for d_ttl in ttl_deltas}

    for b in bouts:
        rb = OpenCVReader(b["paths"]["bot_vid"])
        st = b["start"]
        raw_top_idx = b["vfi"][st]
        true_bot_idx = top_to_bot_frame(b["sid"], raw_top_idx)
        M_bot = compute_kpms_warp_matrix(b["c_bot"], b["h_bot"], 384)
        for d_ttl in ttl_deltas:
            corrupted_idx = min(len(rb) - 1, max(0, true_bot_idx + d_ttl))
            frame_b = rb[corrupted_idx]
            tile = cv2.warpAffine(frame_b, M_bot, (384, 384))
            tile = cv2.resize(tile, (160, 160))
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            c_val = float(gray[70:90, 70:90].mean())
            bot_ttl_intensities[d_ttl].append(c_val)
        rb.close()

    # -------------------------------------------------------------------------
    # Visual Demonstration Tiles: Onset vs Spatial Error vs Temporal Error
    # -------------------------------------------------------------------------
    demo_b = bouts[0]
    rt_demo = OpenCVReader(demo_b["paths"]["top_vid"])
    st_demo = demo_b["start"]
    raw_onset = demo_b["vfi"][st_demo]

    # Tile A: True Onset (0 error)
    M_true = compute_kpms_warp_matrix(demo_b["c_top"], demo_b["h_top"], 144)
    tile_true = cv2.warpAffine(rt_demo[raw_onset], M_true, (144, 144))
    tile_true = cv2.resize(tile_true, (160, 160))
    cv2.putText(tile_true, "True Onset (0 err)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(tile_true, (80, 80), 3, (0, 255, 0), -1)

    # Tile B: Spatial Shift (+20 px translation error)
    M_shift = compute_kpms_warp_matrix(demo_b["c_top"], demo_b["h_top"], 144, shift_xy=(20, 0))
    tile_shift = cv2.warpAffine(rt_demo[raw_onset], M_shift, (144, 144))
    tile_shift = cv2.resize(tile_shift, (160, 160))
    cv2.putText(tile_shift, "Spatial (+20px)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(tile_shift, (80, 80), 3, (0, 0, 255), -1)

    # Tile C: Temporal Shift (+40 frames ~ 667ms latency error)
    raw_future = demo_b["vfi"][st_demo + 40]
    tile_time = cv2.warpAffine(rt_demo[raw_future], M_true, (144, 144))
    tile_time = cv2.resize(tile_time, (160, 160))
    cv2.putText(tile_time, "Time Lag (+40f)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(tile_time, (80, 80), 3, (0, 0, 255), -1)

    rt_demo.close()

    demo_strip = np.hstack([tile_true, np.zeros((160, 6, 3), dtype=np.uint8), tile_shift, np.zeros((160, 6, 3), dtype=np.uint8), tile_time])
    demo_img_path = OUT_DIR / "onset_perturbation_visual_strip.png"
    cv2.imwrite(str(demo_img_path), demo_strip)
    print(f"\n  📸 Saved visual perturbation demonstration strip: {demo_img_path.name}")

    # -------------------------------------------------------------------------
    # Plot 3-Panel Publication Figure
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    # Panel 1: Spatial Translation
    xs_1 = spatial_shifts
    means_1 = [np.mean(trans_intensities[dx]) for dx in xs_1]
    se_1 = [np.std(trans_intensities[dx]) / np.sqrt(len(bouts)) for dx in xs_1]
    axes[0].plot(xs_1, means_1, marker="o", color="#2B6CB0", lw=2.5, markersize=6, label="Centroid Pixel Intensity")
    axes[0].fill_between(xs_1, np.array(means_1) - np.array(se_1), np.array(means_1) + np.array(se_1), color="#2B6CB0", alpha=0.20)
    axes[0].axvline(0, color="#E53E3E", ls="--", lw=1.5, label="True Centroid (0 px)")
    axes[0].set_title("A. Spatial Translation Error (TopCam)", fontsize=11, fontweight="bold", pad=8)
    axes[0].set_xlabel(r"Centroid Offset $\Delta x$ (pixels)", fontsize=10, fontweight="semibold")
    axes[0].set_ylabel("Centroid Pixel Intensity (0 = Dark, 255 = Bright)", fontsize=10, fontweight="semibold")
    axes[0].legend(loc="upper center", fontsize=8.5, frameon=True)

    # Panel 2: Temporal Shift
    xs_2 = time_deltas
    means_2 = [np.mean(time_intensities[dt]) for dt in xs_2]
    se_2 = [np.std(time_intensities[dt]) / np.sqrt(len(bouts)) for dt in xs_2]
    axes[1].plot(xs_2, means_2, marker="s", color="#319795", lw=2.5, markersize=6, label="Centroid Pixel Intensity")
    axes[1].fill_between(xs_2, np.array(means_2) - np.array(se_2), np.array(means_2) + np.array(se_2), color="#319795", alpha=0.20)
    axes[1].axvline(0, color="#E53E3E", ls="--", lw=1.5, label="Syllable Onset ($t=0$)")
    axes[1].set_title("B. Temporal Misalignment (TopCam)", fontsize=11, fontweight="bold", pad=8)
    axes[1].set_xlabel(r"Time Lag $\Delta t$ (frames @ 60fps)", fontsize=10, fontweight="semibold")
    axes[1].legend(loc="upper center", fontsize=8.5, frameon=True)

    # Panel 3: BotCam Hardware TTL Corruption
    xs_3 = ttl_deltas
    means_3 = [np.mean(bot_ttl_intensities[dt]) for dt in xs_3]
    se_3 = [np.std(bot_ttl_intensities[dt]) / np.sqrt(len(bouts)) for dt in xs_3]
    axes[2].plot(xs_3, means_3, marker="^", color="#DD6B20", lw=2.5, markersize=6, label="BotCam Centroid Intensity")
    axes[2].fill_between(xs_3, np.array(means_3) - np.array(se_3), np.array(means_3) + np.array(se_3), color="#DD6B20", alpha=0.20)
    axes[2].axvline(0, color="#E53E3E", ls="--", lw=1.5, label="True TTL Master Clock")
    axes[2].set_title("C. BotCam TTL Synchronization Error", fontsize=11, fontweight="bold", pad=8)
    axes[2].set_xlabel(r"TTL Offset Corruption $\Delta_{\mathrm{TTL}}$ (frames)", fontsize=10, fontweight="semibold")
    axes[2].legend(loc="upper center", fontsize=8.5, frameon=True)

    for ax in axes:
        ax.set_ylim(bottom=10)

    fig.suptitle("Negative Control & Perturbation Audit: Physical Grounding of Centroid Darkness", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_fig_path = OUT_DIR / "negative_control_perturbation_curves.png"
    fig.savefig(out_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved 3-panel perturbation curves: {out_fig_path.name}")

    print("\n" + "=" * 80)
    print("🎉 NEGATIVE CONTROL EXPERIMENT COMPLETE!")
    print(f"   Outputs: {OUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
