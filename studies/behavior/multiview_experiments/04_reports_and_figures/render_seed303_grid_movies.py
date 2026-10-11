#!/usr/bin/env python3
"""
render_seed303_grid_movies.py
=============================================================================
Publication-grade Tandem Dual-View Syllable Grid Movies for Champion Seed 303:
- Adheres 100% to standard Keypoint-MoSeq grid movie architecture:
  1. Crop position and heading are FIXED at syllable onset (no frame-by-frame tracking/rotation).
  2. White centroid dot appears at syllable onset and vanishes at offset.
  3. Frame timing strictly indexed via `video_frame_indexes` (VFI) to eliminate timing mismatch.
  4. Full keypoint & skeleton overlay enabled on both TopCam (17 dorsal KPs) and BotCam (6 ventral KPs).
  5. Scale-aware windows (144px TopCam vs 288px BotCam) resized to 160x160 px tiles.
  6. Strictly neutral labeling ("Syllable {s:02d}").

Environment: kpms_gpu_cuml
=============================================================================
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import re

import cv2
import h5py
import imageio
import numpy as np
import pandas as pd
from keypoint_moseq.viz import overlay_keypoints_on_image
from vidio.read import OpenCVReader

from fpcode.tracking.pair_matching import find_matched_pairs
from fpcode.tracking.clock_sync import top_to_bot_frame

# --- Paths & Directories ---
PROJECT_DIR = EXP_DIR / "kpms_tandem_cohort_production"
MODEL_NAME = "tandem_cohort_118sess_50ar_200slds_seed303"
RESULTS_PATH = PROJECT_DIR / MODEL_NAME / "results.h5"
VFI_PATH = PROJECT_DIR / "video_frame_indexes.npz"

OUT_DIR = SCRIPT_DIR / "visualizations_seed303" / "grid_movies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_VID_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/studies/karen_pff_vs_monomer/video_files")
BOT_VID_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/studies/karen_pff_vs_monomer/botcam_video_files")

# Target syllables to render
TARGET_SYLLABLES = [6, 2, 1, 5, 10, 13]

CHAMBER_TO_ARENA = {
    "top_left": "Arena1",
    "bottom_left": "Arena2",
    "bottom_right": "Arena3",
    "top_right": "Arena4"
}

# Anatomical bodypart orders
TOP_BPS = [
    "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back", "mouse_center",
    "mid_backend", "mid_backend2", "mid_backend3", "tail_base", "left_shoulder",
    "left_midside", "left_hip", "right_shoulder", "right_midside", "right_hip", "head_midpoint"
]
BOT_BPS = ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw", "nose_tip", "tail_base"]

# Skeletons
EDGES_DORSAL = [
    (0, 16), (16, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (3, 10), (10, 11), (11, 12), (12, 9),
    (3, 13), (13, 14), (14, 15), (15, 9),
    (16, 1), (16, 2)
]
EDGES_VENTRAL = [
    (0, 1), (2, 3), (0, 2), (1, 3), (4, 0), (4, 1), (5, 2), (5, 3), (4, 5)
]

# Color palettes (floats in range [0, 1] for matplotlib/opencv overlay)
COLORS_DORSAL = np.linspace(0.4, 0.95, 17)[:, None] * np.array([0.2, 0.6, 0.9], dtype=float)
COLORS_VENTRAL = np.array([
    [0.9, 0.2, 0.2], [0.9, 0.2, 0.2],   # Front paws: Crimson/red
    [0.95, 0.5, 0.1], [0.95, 0.5, 0.1], # Hind paws: Amber/orange
    [0.85, 0.7, 0.1], [0.85, 0.7, 0.1]  # Nose/tail anchors: Gold
], dtype=float)

def compute_kpms_warp_matrix(centroid_onset, heading_onset, window_size):
    """
    Computes standard Keypoint-MoSeq fixed crop affine transform.
    Maps centroid_onset to (window_size/2, window_size/2) rotated by heading_onset.
    """
    c = np.array(centroid_onset, dtype=np.float32)
    h = float(heading_onset)
    r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
    c_shift = r @ c - window_size // 2
    M = np.float32([[np.cos(h), np.sin(h), -c_shift[0]], [-np.sin(h), np.cos(h), -c_shift[1]]])
    return M

def get_session_files(sid, pairs_by_sid):
    """Retrieves paths for TopCam video, BotCam video, TopCam H5, and BotCam H5."""
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

    top_h5 = pair["top_h5"]
    bot_h5 = pair["bot_h5"]

    if top_vids and bot_vids and Path(top_h5).exists() and Path(bot_h5).exists():
        return {
            "top_vid": str(top_vids[0]),
            "bot_vid": str(bot_vids[0]),
            "top_h5": str(top_h5),
            "bot_h5": str(bot_h5),
        }
    return None

def main():
    print("=" * 80)
    print("🎬 RENDERING TANDEM DUAL-VIEW SYLLABLE GRID MOVIES (SEED 303)")
    print(f"Model: {MODEL_NAME}")
    print(f"Target Syllables ({len(TARGET_SYLLABLES)}): {TARGET_SYLLABLES}")
    print(f"Output Directory: {OUT_DIR}")
    print("=" * 80)

    # 1. Load VFI and pair mapping
    print("[1/4] Loading Video Frame Indexes (VFI) and session records...", flush=True)
    with np.load(VFI_PATH) as z:
        vfi_dict = {k: z[k] for k in z.files}

    pairs = find_matched_pairs()
    pairs_by_sid = {p["sid"]: p for p in pairs}

    # 2. Index qualifying syllable bouts
    print("[2/4] Indexing qualifying bouts across cohort results...", flush=True)
    bout_catalog = {s: [] for s in TARGET_SYLLABLES}

    with h5py.File(RESULTS_PATH, "r") as f:
        sess_keys = list(f.keys())
        for k in sess_keys:
            if k not in vfi_dict:
                continue
            if len(vfi_dict[k]) < 20000: # Exclude aborted short test runs (< 5 min)
                continue
            paths = get_session_files(k, pairs_by_sid)
            if not paths:
                continue

            sylls = f[k]["syllable"][:]
            diffs = np.diff(sylls)
            bounds = np.where(diffs != 0)[0] + 1
            starts = np.r_[0, bounds]
            ends = np.r_[bounds, len(sylls)]

            for start, end in zip(starts, ends):
                s_id = sylls[start]
                # Keep bouts with duration >= 15 frames and sufficient pre/post buffer
                if s_id in bout_catalog and (end - start) >= 15 and start >= 10:
                    bout_catalog[s_id].append({
                        "sid": k,
                        "start": start,
                        "end": end,
                        "len": end - start,
                        **paths
                    })

    for s in TARGET_SYLLABLES:
        print(f"  Syllable {s:2d}: Found {len(bout_catalog[s])} qualifying bouts across cohort.")

    # 3. Render grid movie per target syllable
    print("\n[3/4] Rendering composite 3x4 grid movies...", flush=True)
    n_instances = 12
    grid_rows, grid_cols = 3, 4
    pre, post = 10, 40 # 50 frames total = ~833 ms @ 60fps
    clip_frames = pre + post
    n_loops = 3        # 150 frames = 2.5 seconds @ 60fps

    # Sizing & Scaling:
    # TopCam median spine = 56.1 px, BotCam median spine = 148.8 px (ratio = 0.3767).
    # Scaling window_size_bot = 144 / 0.3767 = 384 px ensures the mouse appears at
    # identical physical magnification in both views.
    window_size_top = 144
    window_size_bot = 384 # Scaled so avg nose-to-tail dist matches TopCam
    scaled_window_size = 160
    dot_radius = 4

    gap_x = 16 # Black vertical gap between pairs (columns)
    gap_y = 16 # Black horizontal gap between rows

    for s in TARGET_SYLLABLES:
        bouts = bout_catalog[s]
        if len(bouts) < n_instances:
            print(f"  [WARN] Not enough bouts for Syllable {s}, skipping.")
            continue

        step = max(1, len(bouts) // n_instances)
        chosen_bouts = [bouts[i * step] for i in range(n_instances)]

        desc = f"Syllable {s:02d}"
        print(f"\n▶ Rendering {desc} (12 instances, {clip_frames} frames/instance)...", flush=True)

        instance_clips = [] # list of 12 arrays, each (clip_frames, 160, 320, 3)

        with h5py.File(RESULTS_PATH, "r") as res_file:
            for b_idx, bout in enumerate(chosen_bouts):
                sid = bout["sid"]
                start = bout["start"]
                end = bout["end"]
                vfi = vfi_dict[sid]

                # Ensure within bounds
                if start - pre < 0 or start + post >= len(vfi):
                    continue

                frame_ixs = vfi[start - pre : start + post]

                # Dorsal tracking from results.h5
                grp = res_file[sid]
                c_top_all = grp["centroid"][:]
                h_top_all = grp["heading"][:]

                # Load raw DLC tracking tables for keypoints overlay
                df_top = pd.read_hdf(bout["top_h5"])
                df_bot = pd.read_hdf(bout["bot_h5"])
                top_scorer = df_top.columns.levels[0][0]
                bot_scorer = df_bot.columns.levels[0][0]

                # TopCam onset centroid & heading (FIXED for the clip)
                c_top_onset = c_top_all[start]
                h_top_onset = h_top_all[start]
                M_top = compute_kpms_warp_matrix(c_top_onset, h_top_onset, window_size_top)
                scale_factor_top = scaled_window_size / float(window_size_top)

                # BotCam onset centroid & heading from raw DLC (aligned via TTL)
                onset_vid_idx = frame_ixs[pre]
                onset_bot_idx = min(len(df_bot) - 1, max(0, top_to_bot_frame(sid, onset_vid_idx)))
                nx_0 = df_bot[bot_scorer, "nose_tip", "x"].iloc[onset_bot_idx]
                ny_0 = df_bot[bot_scorer, "nose_tip", "y"].iloc[onset_bot_idx]
                tx_0 = df_bot[bot_scorer, "tail_base", "x"].iloc[onset_bot_idx]
                ty_0 = df_bot[bot_scorer, "tail_base", "y"].iloc[onset_bot_idx]
                c_bot_onset = (0.5 * (nx_0 + tx_0), 0.5 * (ny_0 + ty_0))
                h_bot_onset = np.arctan2(ny_0 - ty_0, nx_0 - tx_0)
                M_bot = compute_kpms_warp_matrix(c_bot_onset, h_bot_onset, window_size_bot)
                scale_factor_bot = scaled_window_size / float(window_size_bot)

                # Open video readers
                reader_t = OpenCVReader(bout["top_vid"])
                reader_b = OpenCVReader(bout["bot_vid"])

                cells = []
                for ii, (offset, raw_vid_idx) in enumerate(zip(range(-pre, post), frame_ixs)):
                    curr_model_frame = start + offset
                    raw_bot_idx = top_to_bot_frame(sid, raw_vid_idx)
                    raw_bot_clamped = min(len(df_bot) - 1, max(0, raw_bot_idx))
                    vid_bot_clamped = min(len(reader_b) - 1, max(0, raw_bot_idx))

                    frame_t = reader_t[raw_vid_idx]
                    frame_b = reader_b[vid_bot_clamped]

                    # 1. Overlay Keypoints on TopCam frame
                    kps_top = np.array([
                        [df_top[top_scorer, bp, "x"].iloc[raw_vid_idx], df_top[top_scorer, bp, "y"].iloc[raw_vid_idx]]
                        for bp in TOP_BPS
                    ])
                    frame_t = overlay_keypoints_on_image(
                        frame_t, kps_top, edges=EDGES_DORSAL,
                        keypoint_colors=COLORS_DORSAL, node_size=3, line_width=2
                    )

                    # 2. Overlay Keypoints on BotCam frame (synchronized via TTL)
                    kps_bot = np.array([
                        [df_bot[bot_scorer, bp, "x"].iloc[raw_bot_clamped], df_bot[bot_scorer, bp, "y"].iloc[raw_bot_clamped]]
                        for bp in BOT_BPS
                    ])
                    frame_b = overlay_keypoints_on_image(
                        frame_b, kps_bot, edges=EDGES_VENTRAL,
                        keypoint_colors=COLORS_VENTRAL, node_size=4, line_width=2
                    )

                    # 3. Apply standard fixed KPMS affine warp
                    tile_t = cv2.warpAffine(frame_t, M_top, (window_size_top, window_size_top), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
                    tile_t = cv2.resize(tile_t, (scaled_window_size, scaled_window_size))

                    tile_b = cv2.warpAffine(frame_b, M_bot, (window_size_bot, window_size_bot), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
                    tile_b = cv2.resize(tile_b, (scaled_window_size, scaled_window_size))
                    # Horizontal flip on BotCam to match dorsal body axis chirality
                    tile_b = cv2.flip(tile_b, 1)

                    # 4. White centroid dot during active syllable (standard KPMS feature)
                    if 0 <= offset <= (end - start):
                        # TopCam moving centroid dot
                        c_t_curr = c_top_all[curr_model_frame] if curr_model_frame < len(c_top_all) else c_top_onset
                        pos_t = tuple([int(x) for x in (M_top @ np.append(c_t_curr, 1)) * scale_factor_top])
                        if 0 <= pos_t[0] < scaled_window_size and 0 <= pos_t[1] < scaled_window_size:
                            cv2.circle(tile_t, pos_t, dot_radius, (255, 255, 255), -1, cv2.LINE_AA)

                        # BotCam moving centroid dot (accounting for flip)
                        nx_c = df_bot[bot_scorer, "nose_tip", "x"].iloc[raw_bot_clamped]
                        ny_c = df_bot[bot_scorer, "nose_tip", "y"].iloc[raw_bot_clamped]
                        tx_c = df_bot[bot_scorer, "tail_base", "x"].iloc[raw_bot_clamped]
                        ty_c = df_bot[bot_scorer, "tail_base", "y"].iloc[raw_bot_clamped]
                        c_b_curr = (0.5 * (nx_c + tx_c), 0.5 * (ny_c + ty_c))
                        pos_b = [int(x) for x in (M_bot @ np.append(c_b_curr, 1)) * scale_factor_bot]
                        pos_b[0] = scaled_window_size - 1 - pos_b[0] # mirror flip coordinate
                        if 0 <= pos_b[0] < scaled_window_size and 0 <= pos_b[1] < scaled_window_size:
                            cv2.circle(tile_b, tuple(pos_b), dot_radius, (255, 255, 255), -1, cv2.LINE_AA)

                    tandem_cell = np.hstack([tile_t, tile_b])
                    cells.append(tandem_cell)

                reader_t.close()
                reader_b.close()
                instance_clips.append(np.array(cells))

        # 4. Assemble into 3x4 grid movie frames with clean black gaps (no title bar)
        grid_frames = []
        canvas_h = grid_rows * scaled_window_size + (grid_rows - 1) * gap_y
        canvas_w = grid_cols * (scaled_window_size * 2) + (grid_cols - 1) * gap_x

        for f_idx in range(clip_frames):
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            for inst_idx, clip in enumerate(instance_clips):
                r = inst_idx // grid_cols
                c = inst_idx % grid_cols

                y0 = r * (scaled_window_size + gap_y)
                y1 = y0 + scaled_window_size

                x0 = c * (scaled_window_size * 2 + gap_x)
                x1 = x0 + (scaled_window_size * 2)

                canvas[y0:y1, x0:x1] = clip[f_idx]

                # Subtle hairline divider inside each pair between TopCam and BotCam
                div_x = x0 + scaled_window_size
                canvas[y0:y1, div_x:div_x+1] = (50, 50, 50)

            grid_frames.append(canvas)

        # Smooth loop
        looped_frames = grid_frames * n_loops

        # Write MP4
        out_mp4 = OUT_DIR / f"syllable_{s:02d}_tandem_grid.mp4"
        writer = imageio.get_writer(str(out_mp4), fps=60, codec="libx264", quality=8)
        for frame in looped_frames:
            writer.append_data(frame)
        writer.close()

        file_size_mb = out_mp4.stat().st_size / (1024 * 1024)
        print(f"  ✅ Saved {out_mp4.name} ({len(looped_frames)} frames @ 60fps, {file_size_mb:.2f} MB, canvas: {canvas_w}x{canvas_h})", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("🎉 ALL TANDEM DUAL-VIEW GRID MOVIES RENDERED SUCCESSFULLY!", flush=True)
    print(f"   Outputs Directory: {OUT_DIR}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
