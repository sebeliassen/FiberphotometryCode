#!/usr/bin/env python3
"""
render_all_seed101_grid_movies.py
=============================================================================
Publication-grade Tandem Dual-View Syllable Grid Movies for Champion Seed 101:
- TWO MODES:
  1. --mode density : Built-in contrastive density sampling (prototypical exemplars)
     -> Output: visualizations_seed101/grid_movies_density/
  2. --mode random  : Uniform spread sampling across cohort sessions
     -> Output: visualizations_seed101/grid_movies_random/
- FULL COHORT COVERAGE:
  Handles all syllables in Champion Seed 101. Syllables with fewer than 12
  instances are gracefully tiled across the 12 panels so every syllable is watchable.
- Visual Architecture:
  1. Pure clean video frames WITHOUT pose overlays.
  2. Stationary onset crop (no tracking/rotation chatter).
  3. White centroid indicator dot appears at onset and vanishes at offset.
  4. Frame-accurate FP3002 TTL hardware clock synchronization.
  5. Scale-matched 160x160 px tiles (144px TopCam vs 384px BotCam normalized).

Environment: kpms_gpu_cuml
=============================================================================
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np
import pandas as pd
from vidio.read import OpenCVReader

# 2. Fast-seek monkeypatch to OpenCVReader supporting scalar ints, slices, and array-likes
_original_read = OpenCVReader.read
def _fast_read(self, framenum):
    if isinstance(framenum, slice):
        return [self.read(i) for i in self.slice_to_list(framenum)]
    if isinstance(framenum, (list, tuple, np.ndarray)):
        return [self.read(int(i)) for i in framenum]
    framenum = int(framenum)
    gap = framenum - self.fnum
    if 0 < gap <= 30:
        for _ in range(gap):
            ret, _ = self.file_object.read()
            if not ret:
                break
        self.fnum = framenum
    return _original_read(self, framenum)
OpenCVReader.read = _fast_read

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from core.pair_matching import find_matched_pairs
from core.ttl_sync import top_to_bot_frame

# Paths
PROJECT_DIR = EXP_DIR / "kpms_tandem_cohort_production"
MODEL_NAME = "tandem_cohort_118sess_50ar_200slds_seed101"
RESULTS_PATH = PROJECT_DIR / MODEL_NAME / "results.h5"
VFI_PATH = PROJECT_DIR / "video_frame_indexes.npz"

TOP_VID_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/karen_project/video_files")
BOT_VID_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/karen_project/botcam_video_files")

CHAMBER_TO_ARENA = {
    "top_left": "Arena1",
    "bottom_left": "Arena2",
    "bottom_right": "Arena3",
    "top_right": "Arena4"
}

def compute_kpms_warp_matrix(centroid_onset, heading_onset, window_size):
    """Maps centroid_onset to (window_size/2, window_size/2) rotated by heading_onset."""
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
    parser = argparse.ArgumentParser(description="Render Tandem Dual-View Syllable Grid Movies for Seed 101")
    parser.add_argument("--mode", type=str, choices=["density", "random"], default="density",
                        help="Instance selection mode: 'density' (exemplar nearest neighbors) or 'random' (uniform spread)")
    parser.add_argument("--syllables", nargs="+", type=int, default=None,
                        help="Specific syllable IDs to render (default: render all available syllables)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing MP4 files")
    args = parser.parse_args()

    mode = args.mode
    out_dir = SCRIPT_DIR / "visualizations_seed101" / f"grid_movies_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🎬 RENDERING SEED 101 DUAL-VIEW GRID MOVIES (MODE: {mode.upper()})")
    print(f"Model: {MODEL_NAME}")
    print(f"Sampling Mode: {mode.upper()}")
    print(f"Target Output Directory: {out_dir}")
    print("=" * 80)

    # 1. Load VFI and pair mapping
    print("[1/4] Loading Video Frame Indexes (VFI) and session records...", flush=True)
    with np.load(VFI_PATH) as z:
        vfi_dict = {k: z[k] for k in z.files}

    pairs = find_matched_pairs()
    pairs_by_sid = {p["sid"]: p for p in pairs}

    # 2. Index all qualifying syllable bouts across cohort
    print("[2/4] Indexing all syllable bouts across cohort results...", flush=True)
    bout_catalog = {}

    with h5py.File(RESULTS_PATH, "r") as f:
        sess_keys = list(f.keys())
        for k in sess_keys:
            if k not in vfi_dict or len(vfi_dict[k]) < 20000:
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
                s_id = int(sylls[start])
                dur = end - start
                if dur >= 3 and start >= 15:
                    if s_id not in bout_catalog:
                        bout_catalog[s_id] = []
                    bout_catalog[s_id].append({
                        "sid": k,
                        "start": start,
                        "end": end,
                        "len": dur,
                        **paths
                    })

    available_sylls = sorted(bout_catalog.keys())
    target_sylls = sorted(args.syllables) if args.syllables is not None else available_sylls
    target_sylls = [s for s in target_sylls if s in bout_catalog and len(bout_catalog[s]) > 0]

    print(f"  Indexed bouts for {len(available_sylls)} distinct syllables.")
    print(f"  Target syllables to render ({len(target_sylls)}): {target_sylls}")

    # 3. Density / Random Sampling per Syllable
    n_instances = 12
    grid_rows, grid_cols = 3, 4
    pre, post = 10, 40 # 50 frames total = ~833 ms @ 60fps
    clip_frames = pre + post
    n_loops = 3        # 150 frames = 2.5 seconds @ 60fps

    window_size_top = 144
    window_size_bot = 384
    scaled_window_size = 160
    dot_radius = 4
    gap_x, gap_y = 16, 16

    print(f"\n[3/4] Rendering 3x4 grid movies for {len(target_sylls)} syllables...", flush=True)

    for idx, s in enumerate(target_sylls):
        bouts = bout_catalog[s]
        desc = f"Syllable {s:02d}"
        mp4_path = out_dir / f"syllable_{s:02d}_tandem_grid_{mode}.mp4"

        # Check if already rendered (resumable)
        if not args.overwrite and mp4_path.exists() and mp4_path.stat().st_size > 10000:
            print(f"  [{idx+1}/{len(target_sylls)}] {desc} already rendered ({mode}, {mp4_path.stat().st_size // 1024} KB), skipping.", flush=True)
            continue

        n_avail = len(bouts)
        # Select 12 bouts according to mode
        if mode == "density" and n_avail >= 12:
            # Contrastive duration/spatial exemplar: find median duration bout and its nearest neighbors
            durs = np.array([b["len"] for b in bouts])
            med_dur = np.median(durs)
            dur_diffs = np.abs(durs - med_dur)
            best_order = np.argsort(dur_diffs)
            chosen_bouts = [bouts[i] for i in best_order[:n_instances]]
            sampling_label = f"Density Exemplars ({n_avail} bouts in cohort)"
        elif mode == "random" and n_avail >= 12:
            # Uniform spread across sessions
            step = max(1, n_avail // n_instances)
            chosen_bouts = [bouts[i * step] for i in range(n_instances)]
            sampling_label = f"Uniform Spread ({n_avail} bouts in cohort)"
        else:
            # Low instance fallback: tile available instances across the 12 panels
            chosen_bouts = [bouts[i % n_avail] for i in range(n_instances)]
            sampling_label = f"Rare Syllable ({n_avail} unique bouts tiled)"

        print(f"\n▶ [{idx+1}/{len(target_sylls)}] {desc}: {sampling_label} -> {mp4_path.name}", flush=True)

        instance_clips = []
        with h5py.File(RESULTS_PATH, "r") as res_file:
            for b_idx, bout in enumerate(chosen_bouts):
                sid = bout["sid"]
                start = bout["start"]
                end = bout["end"]
                vfi = vfi_dict[sid]

                if start - pre < 0 or start + post >= len(vfi):
                    # Edge fallback: shift window
                    pad_pre = max(0, start)
                    pad_post = min(len(vfi) - 1, start + post)
                    frame_ixs = vfi[pad_pre : pad_post]
                    if len(frame_ixs) < clip_frames:
                        frame_ixs = np.pad(frame_ixs, (0, clip_frames - len(frame_ixs)), mode="edge")
                else:
                    frame_ixs = vfi[start - pre : start + post]

                # Dorsal tracking from results.h5
                grp = res_file[sid]
                c_top_all = grp["centroid"][:]
                h_top_all = grp["heading"][:]

                # Load BotCam DLC tracking table once for onset heading/centroid
                df_bot = pd.read_hdf(bout["bot_h5"])
                bot_scorer = df_bot.columns.levels[0][0]

                # TopCam onset centroid & heading (FIXED for entire clip)
                c_top_onset = c_top_all[start]
                h_top_onset = h_top_all[start]
                M_top = compute_kpms_warp_matrix(c_top_onset, h_top_onset, window_size_top)
                scale_factor_top = scaled_window_size / float(window_size_top)

                # BotCam onset centroid & heading (hardware aligned via TTL)
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

                # Read Video Streams
                top_reader = OpenCVReader(bout["top_vid"])
                bot_reader = OpenCVReader(bout["bot_vid"])

                top_frames_raw = [top_reader[int(fix)] for fix in frame_ixs]
                bot_frame_ixs = [
                    min(len(df_bot) - 1, max(0, top_to_bot_frame(sid, fix)))
                    for fix in frame_ixs
                ]
                bot_frames_raw = [bot_reader[int(fix)] for fix in bot_frame_ixs]
                top_reader.close()
                bot_reader.close()

                # Compose side-by-side tile
                tile_frames = []
                for t in range(clip_frames):
                    tf = top_frames_raw[t]
                    bf = bot_frames_raw[t]

                    # Warp crops
                    crop_top = cv2.warpAffine(tf, M_top, (window_size_top, window_size_top), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
                    crop_bot = cv2.warpAffine(bf, M_bot, (window_size_bot, window_size_bot), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))

                    # Normalize tile scales to 160x160 px
                    tile_top = cv2.resize(crop_top, (scaled_window_size, scaled_window_size), interpolation=cv2.INTER_LINEAR)
                    tile_bot = cv2.resize(crop_bot, (scaled_window_size, scaled_window_size), interpolation=cv2.INTER_LINEAR)
                    # Horizontal flip on BotCam to match dorsal body axis chirality
                    tile_bot = cv2.flip(tile_bot, 1)

                    # Syllable Active Window Indicator: Official KPMS moving centroid tracking
                    offset = t - pre
                    if 0 <= offset < bout["len"]:
                        curr_model_frame = start + offset
                        c_t_curr = c_top_all[curr_model_frame] if curr_model_frame < len(c_top_all) else c_top_onset
                        if np.isnan(c_t_curr[0]) or np.isnan(c_t_curr[1]):
                            c_t_curr = c_top_onset
                        pos_t = tuple([int(round(x)) for x in (M_top @ np.append(c_t_curr, 1)) * scale_factor_top])
                        if 0 <= pos_t[0] < scaled_window_size and 0 <= pos_t[1] < scaled_window_size:
                            cv2.circle(tile_top, pos_t, dot_radius, (255, 255, 255), -1, cv2.LINE_AA)

                        # Moving centroid dot on BotCam
                        raw_bot_idx = bot_frame_ixs[t]
                        nx_c = df_bot[bot_scorer, "nose_tip", "x"].iloc[raw_bot_idx]
                        ny_c = df_bot[bot_scorer, "nose_tip", "y"].iloc[raw_bot_idx]
                        tx_c = df_bot[bot_scorer, "tail_base", "x"].iloc[raw_bot_idx]
                        ty_c = df_bot[bot_scorer, "tail_base", "y"].iloc[raw_bot_idx]
                        c_bot_curr = (0.5 * (nx_c + tx_c), 0.5 * (ny_c + ty_c))
                        if np.isnan(c_bot_curr[0]) or np.isnan(c_bot_curr[1]):
                            c_bot_curr = c_bot_onset
                        pos_b = tuple([int(round(x)) for x in (M_bot @ np.append(c_bot_curr, 1)) * scale_factor_bot])
                        pos_b_flipped = (scaled_window_size - 1 - pos_b[0], pos_b[1])
                        if 0 <= pos_b_flipped[0] < scaled_window_size and 0 <= pos_b_flipped[1] < scaled_window_size:
                            cv2.circle(tile_bot, pos_b_flipped, dot_radius, (255, 255, 255), -1, cv2.LINE_AA)

                    # View headers
                    cv2.putText(tile_top, "TOP", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
                    cv2.putText(tile_bot, "BOT", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

                    # Stack TopCam on Left, BotCam on Right
                    tandem_tile = np.concatenate([tile_top, tile_bot], axis=1)

                    # Tile border
                    cv2.rectangle(tandem_tile, (0, 0), (tandem_tile.shape[1] - 1, tandem_tile.shape[0] - 1), (50, 50, 50), 1)

                    # Label rare instances
                    if n_avail < 12:
                        inst_tag = f"Inst {b_idx % n_avail + 1}/{n_avail}"
                        cv2.putText(tandem_tile, inst_tag, (tandem_tile.shape[1] - 75, 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1, cv2.LINE_AA)

                    tile_frames.append(tandem_tile)

                instance_clips.append(tile_frames)

        # Assemble 3x4 Grid Composite
        tile_w = scaled_window_size * 2
        tile_h = scaled_window_size
        banner_h = 48
        grid_w = grid_cols * tile_w + (grid_cols + 1) * gap_x
        grid_h = grid_rows * tile_h + (grid_rows + 1) * gap_y + banner_h

        assembled_frames = []
        for t in range(clip_frames):
            canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            # Header Banner
            title_text = f"TANDEM MULTI-VIEW KEYPOINT-MOSEQ | {desc.upper()} ({mode.upper()})"
            sub_text = f"TopCam (17 KPs Dorsal) Left  |  BotCam (6 KPs Ventral) Right  •  {sampling_label}  •  FPS: 60"
            cv2.putText(canvas, title_text, (gap_x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, sub_text, (gap_x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

            # Place 12 tiles in row-major order
            for i, clip in enumerate(instance_clips):
                r = i // grid_cols
                c = i % grid_cols
                x = gap_x + c * (tile_w + gap_x)
                y = banner_h + gap_y + r * (tile_h + gap_y)
                canvas[y : y + tile_h, x : x + tile_w] = clip[t]

            assembled_frames.append(canvas)

        # Loop 3x for smooth review (150 frames = 2.5s @ 60 FPS)
        full_movie = assembled_frames * n_loops
        with imageio.get_writer(str(mp4_path), fps=60, codec="libx264", quality=8) as writer:
            for f in full_movie:
                writer.append_data(f)

    print("\n" + "=" * 80)
    print(f"🎉 COMPLETED ALL GRID MOVIES IN MODE: {mode.upper()}!")
    print(f"   Output Directory: {out_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
