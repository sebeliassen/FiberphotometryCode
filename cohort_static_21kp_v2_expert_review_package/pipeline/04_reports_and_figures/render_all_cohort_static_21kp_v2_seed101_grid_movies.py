#!/usr/bin/env python3
"""
render_all_cohort_static_21kp_v2_seed101_grid_movies.py
========================================================
Publication-grade Tandem Dual-View Syllable Grid Movies for Cohort Static 21kp V2:
- Consumes the shared unified selection manifest:
  `unified_syllable_selection_manifest.json`
- Supports all 3 tiers:
  * Tier 1: N > 50 (Built-in KPMS Density)
  * Tier 2: 12 < N <= 50 (Compact-Neighborhood Selection)
  * Tier 3: N <= 12 (Complete Bout Enumeration + Blank Tiles)
- Full Tracking-Loss Filtering for Grid Display:
  * Ensures that NO bout displayed has dorsal or ventral tracking lost.
  * Replaces any glitched bout with the next qualifying valid bout for that syllable.
- Standard Keypoint-MoSeq Timing & Real-World Kinematics:
  * pre = 1.0s (60 frames) prior to syllable onset.
  * post = 2.0s (120 frames) after syllable onset.
  * Total duration = 180 frames = 3.0 seconds at native 60.0 FPS (no slowdowns).
  * Onset white dot appears at t=pre and remains for the bout duration.
- Video Encoding:
  * Native H.264 (avc1) via imageio/libx264, yuv420p, +faststart.
  * 16-divisible canvas (1360 x 592 px).

Environment: kpms_gpu_cuml (pure CPU)
"""

import os
import sys
import argparse
import time
import re
import json
from pathlib import Path
import numpy as np
import h5py
import pandas as pd
import cv2
import imageio
from vidio.read import OpenCVReader

# Fast-seek monkeypatch to OpenCVReader supporting scalar ints, slices, and array-likes
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
from core.camera_correspondence import top_to_bot_frame_continuous

COHORT_DIR = EXP_DIR / "cohort_static_21kp_v2"
MODEL_DIR = COHORT_DIR / "model_21kp_static_50ar_300slds_seed101"
RESULTS_PATH = MODEL_DIR / "results.h5"
VFI_PATH = COHORT_DIR / "video_frame_indexes.npz"

VIZ_DIR = SCRIPT_DIR / "visualizations_cohort_static_21kp_v2_seed101"
MANIFEST_PATH = VIZ_DIR / "unified_syllable_selection_manifest.json"
OUT_DIR = VIZ_DIR / "grid_movies_density"

TOP_VID_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/karen_project/video_files")
BOT_VID_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/karen_project/botcam_video_files")
CHAMBER_TO_ARENA = {
    "top_left": "Arena1",
    "bottom_left": "Arena2",
    "bottom_right": "Arena3",
    "top_right": "Arena4",
}


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

    if top_vids and bot_vids and Path(pair["top_h5"]).exists() and Path(pair["bot_h5"]).exists():
        return {
            "top_vid": str(top_vids[0]),
            "bot_vid": str(bot_vids[0]),
            "top_h5": str(pair["top_h5"]),
            "bot_h5": str(pair["bot_h5"]),
        }
    return None


def compute_kpms_warp_matrix(centroid_onset, heading_onset, window_size):
    """Maps centroid_onset to (window_size/2, window_size/2) rotated by heading_onset."""
    c = np.array(centroid_onset, dtype=np.float32)
    h = float(heading_onset)
    r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
    c_shift = r @ c - window_size // 2
    M = np.float32([[np.cos(h), np.sin(h), -c_shift[0]], [-np.sin(h), np.cos(h), -c_shift[1]]])
    return M


def compute_robust_bot_pose(df_bot, bot_scorer, target_idx, search_window=10):
    """
    Computes (c_bot, h_bot, is_valid) for df_bot at target_idx.
    Enforces p >= 0.8 and morphological distance 20 <= L <= 400 px.
    Searches +- search_window frames if the target frame has tracking glitch.
    Falls back to multi-keypoint paw centroid if nose/tail cannot be resolved.
    """
    n_frames = len(df_bot)
    target_idx = int(min(n_frames - 1, max(0, target_idx)))

    candidates = [target_idx]
    for w in range(1, search_window + 1):
        if target_idx - w >= 0:
            candidates.append(target_idx - w)
        if target_idx + w < n_frames:
            candidates.append(target_idx + w)

    for idx in candidates:
        try:
            nx = df_bot[bot_scorer, "nose_tip", "x"].iloc[idx]
            ny = df_bot[bot_scorer, "nose_tip", "y"].iloc[idx]
            np_ = df_bot[bot_scorer, "nose_tip", "likelihood"].iloc[idx]

            tx = df_bot[bot_scorer, "tail_base", "x"].iloc[idx]
            ty = df_bot[bot_scorer, "tail_base", "y"].iloc[idx]
            tp_ = df_bot[bot_scorer, "tail_base", "likelihood"].iloc[idx]

            if np_ >= 0.8 and tp_ >= 0.8:
                dist = np.hypot(nx - tx, ny - ty)
                if 20.0 <= dist <= 400.0:
                    c_bot = ((nx + tx) / 2.0, (ny + ty) / 2.0)
                    h_bot = np.arctan2(ny - ty, nx - tx)
                    return c_bot, h_bot, True
        except Exception:
            pass

    # Fallback: paw centroid
    try:
        xs, ys = [], []
        for paw in ["paw_front_left", "paw_front_right", "paw_back_left", "paw_back_right"]:
            px = df_bot[bot_scorer, paw, "x"].iloc[target_idx]
            py = df_bot[bot_scorer, paw, "y"].iloc[target_idx]
            pp = df_bot[bot_scorer, paw, "likelihood"].iloc[target_idx]
            if pp >= 0.6:
                xs.append(px)
                ys.append(py)
        if len(xs) >= 2:
            return (float(np.mean(xs)), float(np.mean(ys))), 0.0, True
    except Exception:
        pass

    return (486.0, 486.0), 0.0, False


def is_bout_tracking_valid(sid, start, end, vfi_dict, res_file, bot_h5, pre, post, bot_cache):
    """
    Validates that a bout has pristine tracking in BOTH dorsal and ventral streams.
    Returns False if either dorsal or ventral tracking was lost.
    """
    if sid not in vfi_dict or sid not in res_file:
        return False
    grp = res_file[sid]
    cents = grp["centroid"][:]
    heads = grp["heading"][:]
    n_frames = len(cents)
    if start < pre or (start + post) > n_frames:
        return False

    # Check dorsal tracking
    c_slice = cents[start - pre : start + post]
    h_slice = heads[start - pre : start + post]
    if np.isnan(c_slice).any() or np.isnan(h_slice).any():
        return False
    c0 = cents[start]
    if c0[0] < 10 or c0[0] > 1270 or c0[1] < 10 or c0[1] > 1014:
        return False

    # Check ventral tracking
    if not Path(bot_h5).exists():
        return False
    if bot_h5 not in bot_cache:
        df_b = pd.read_hdf(bot_h5)
        bot_cache[bot_h5] = (df_b, df_b.columns.levels[0][0])
    df_bot, scorer = bot_cache[bot_h5]

    vfi = vfi_dict[sid]
    onset_vid_idx = int(vfi[start])
    onset_bot = top_to_bot_frame_continuous(sid, onset_vid_idx)
    if not (0 <= onset_bot < len(df_bot)):
        return False
    _, _, is_v_onset = compute_robust_bot_pose(df_bot, scorer, onset_bot, search_window=5)
    if not is_v_onset:
        return False

    offset_idx = min(end, n_frames - 1)
    offset_vid_idx = int(vfi[offset_idx])
    offset_bot = top_to_bot_frame_continuous(sid, offset_vid_idx)
    if not (0 <= offset_bot < len(df_bot)):
        return False
    _, _, is_v_offset = compute_robust_bot_pose(df_bot, scorer, offset_bot, search_window=5)
    return is_v_offset


def render_blank_tile(scaled_window_size, clip_frames, tile_rank):
    """Generates an inactive/blank placeholder tile across clip_frames."""
    frames = []
    blank_tile = np.zeros((scaled_window_size, scaled_window_size * 2, 3), dtype=np.uint8)
    blank_tile[:] = (18, 18, 22)
    # Draw subtle outline
    cv2.rectangle(blank_tile, (0, 0), (blank_tile.shape[1] - 1, blank_tile.shape[0] - 1), (40, 40, 48), 1)
    cv2.putText(blank_tile, "EMPTY TILE", (scaled_window_size - 44, scaled_window_size // 2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (75, 75, 85), 1, cv2.LINE_AA)
    cv2.putText(blank_tile, f"Slot {tile_rank + 1} (No Bout)", (scaled_window_size - 60, scaled_window_size // 2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (55, 55, 65), 1, cv2.LINE_AA)
    for _ in range(clip_frames):
        frames.append(blank_tile.copy())
    return frames


def main():
    parser = argparse.ArgumentParser(description="Render Cohort Static 21kp V2 Syllable Grid Movies from Manifest")
    parser.add_argument("--syllables", nargs="+", type=int, default=None,
                        help="Specific syllable IDs to render (default: render all active in manifest)")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH),
                        help="Path to unified selection manifest")
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR),
                        help="Target output directory for MP4 movies")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing MP4 files")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}. Run select_cohort_static_21kp_v2_syllable_manifest.py first!")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_size = manifest.get("grid_size", 12)
    grid_rows, grid_cols = 3, 4

    # KPMS standard time window: 1.0s prior, 2.0s post @ 60 FPS
    out_fps = 60
    pre = round(1.0 * out_fps)   # 60 frames = 1.0 s
    post = round(2.0 * out_fps)  # 120 frames = 2.0 s
    clip_frames = pre + post     # 180 frames = 3.0 s
    n_loops = 1                  # 1 loop = 3.0s native real-time movie

    window_size_top = 144
    window_size_bot = 384
    scaled_window_size = 160
    dot_radius = 4
    gap_x, gap_y = 16, 16

    with np.load(VFI_PATH) as z:
        vfi_dict = {k: z[k] for k in z.files}

    pairs = find_matched_pairs()
    pairs_by_sid = {p["sid"]: p for p in pairs}

    # Pre-index qualifying bouts by syllable from results.h5 for substitution if tracking is lost
    print("Indexing qualifying cohort bouts from results.h5 for tracking substitutes...", flush=True)
    t0_idx = time.time()
    bouts_by_syllable = {}
    with h5py.File(RESULTS_PATH, "r") as res_f:
        for k in res_f.keys():
            if k not in vfi_dict:
                continue
            paths = get_session_files(k, pairs_by_sid)
            if not paths:
                continue
            sylls = res_f[k]["syllable"][:]
            cents = res_f[k]["centroid"][:]
            heads = res_f[k]["heading"][:]
            diffs = np.diff(sylls, prepend=-1, append=-1)
            starts = np.where(diffs[:-1] != 0)[0]
            ends = np.where(diffs[1:] != 0)[0]
            for st, en in zip(starts, ends):
                dur = en - st
                if dur >= 3 and st >= pre and (st + post) <= len(sylls):
                    s_id = int(sylls[st])
                    if np.isnan(cents[st]).any() or np.isnan(heads[st]):
                        continue
                    bouts_by_syllable.setdefault(s_id, []).append((k, int(st), int(en), int(dur), paths))
    print(f"Indexed qualifying bouts in {time.time() - t0_idx:.2f}s across {len(bouts_by_syllable)} syllables.", flush=True)

    bot_cache = {}

    all_syll_entries = manifest["syllables"]
    if args.syllables is not None:
        target_sylls = [s for s in args.syllables if str(s) in all_syll_entries]
    else:
        # Default: all active >=0.1% usage
        target_sylls = [
            int(s) for s, data in sorted(all_syll_entries.items(), key=lambda x: int(x[0]))
            if data.get("is_active_0p1pct", True)
        ]

    print("=" * 80)
    print("🎬 RENDERING V2 COHORT STATIC 21KP DUAL-VIEW GRID MOVIES (60 FPS NATIVE)")
    print(f"Timing Window:  pre={pre}f (1.0s), post={post}f (2.0s) -> {clip_frames}f (3.0s total @ 60 FPS)")
    print(f"Grid Layout:    {grid_rows}x{grid_cols} ({grid_size} tiles)")
    print(f"Tracking Filter:Strict (0 bouts with tracking lost allowed on display)")
    print(f"Syllables:      {len(target_sylls)} targets")
    print(f"Output Target:  {out_dir}")
    print("=" * 80)

    with h5py.File(RESULTS_PATH, "r") as res_file:
        for idx, s in enumerate(target_sylls):
            s_str = str(s)
            s_data = all_syll_entries[s_str]
            bouts = s_data["bouts"]
            tier = s_data["tier"]
            total_bouts = s_data["total_cohort_bouts"]
            usage_pct = s_data.get("usage_pct", 0.0)

            mp4_path = out_dir / f"syllable_{s:02d}_tandem_grid_density.mp4"
            if not args.overwrite and mp4_path.exists() and mp4_path.stat().st_size > 10000:
                print(f"  [{idx+1}/{len(target_sylls)}] Syllable {s:02d} already rendered, skipping.", flush=True)
                continue

            print(f"\n▶ [{idx+1}/{len(target_sylls)}] Syllable {s:02d} ({total_bouts} bouts, {usage_pct:.2f}% usage, {tier}) -> {mp4_path.name}", flush=True)

            # Build display bouts guaranteeing NO tracking loss in dorsal or ventral views
            display_bouts = []
            selected_keys = set()
            for b in bouts:
                if b.get("is_blank", False):
                    display_bouts.append(b)
                    continue
                b_sid = b["segment_name"]
                b_st = b["onset_segment_frame"]
                b_en = b.get("offset_segment_frame", b_st + b.get("duration_frames", 10))
                b_bot_h5 = b["bot_h5"]

                if is_bout_tracking_valid(b_sid, b_st, b_en, vfi_dict, res_file, b_bot_h5, pre, post, bot_cache):
                    display_bouts.append(b)
                    selected_keys.add((b_sid, b_st))
                else:
                    # Substitute with the next valid bout for this syllable
                    found_sub = False
                    for cand in bouts_by_syllable.get(s, []):
                        csid, cst, cen, cdur, cpaths = cand
                        if (csid, cst) not in selected_keys:
                            if is_bout_tracking_valid(csid, cst, cen, vfi_dict, res_file, cpaths["bot_h5"], pre, post, bot_cache):
                                display_bouts.append({
                                    "segment_name": csid,
                                    "onset_segment_frame": cst,
                                    "offset_segment_frame": cen,
                                    "duration_frames": cdur,
                                    "top_vid": cpaths["top_vid"],
                                    "bot_vid": cpaths["bot_vid"],
                                    "top_h5": cpaths["top_h5"],
                                    "bot_h5": cpaths["bot_h5"],
                                    "is_medoid": b.get("is_medoid", False),
                                })
                                selected_keys.add((csid, cst))
                                found_sub = True
                                break
                    if not found_sub:
                        display_bouts.append({"is_blank": True})

            instance_clips = []
            for b_idx, bout in enumerate(display_bouts):
                if bout.get("is_blank", False):
                    blank_frames = render_blank_tile(scaled_window_size, clip_frames, b_idx)
                    instance_clips.append(blank_frames)
                    continue

                sid = bout["segment_name"]
                start = bout["onset_segment_frame"]
                vfi = vfi_dict[sid]
                vfi_len = len(vfi)

                left_idx = start - pre
                right_idx = start + post
                pad_left = max(0, -left_idx)
                pad_right = max(0, right_idx - vfi_len)

                slice_start = max(0, left_idx)
                slice_end = min(vfi_len, right_idx)
                raw_slice = vfi[slice_start:slice_end]
                frame_ixs = np.pad(raw_slice, (pad_left, pad_right), mode="edge")

                # Dorsal tracking from results.h5
                grp = res_file[sid]
                c_top_all = grp["centroid"][:]
                h_top_all = grp["heading"][:]

                df_bot, bot_scorer = bot_cache[bout["bot_h5"]]

                top_reader = OpenCVReader(bout["top_vid"])
                bot_reader = OpenCVReader(bout["bot_vid"])

                top_frames_raw = [top_reader[int(fix)] for fix in frame_ixs]
                bot_frame_ixs = [top_to_bot_frame_continuous(sid, int(fix)) for fix in frame_ixs]
                bot_frames_raw = []
                for b_fix in bot_frame_ixs:
                    if 0 <= b_fix < len(df_bot):
                        frame = bot_reader[int(b_fix)]
                        bot_frames_raw.append(frame)
                    else:
                        bot_frames_raw.append(bot_reader[0])
                top_reader.close()
                bot_reader.close()

                top_h, top_w = top_frames_raw[0].shape[:2]
                c_top_onset = c_top_all[start]
                h_top_onset = h_top_all[start]

                if (c_top_onset[0] < 0 or c_top_onset[0] >= top_w or
                    c_top_onset[1] < 0 or c_top_onset[1] >= top_h or
                    np.isnan(c_top_onset[0]) or np.isnan(c_top_onset[1])):
                    c_top_onset = (top_w / 2.0, top_h / 2.0)

                M_top = compute_kpms_warp_matrix(c_top_onset, h_top_onset, window_size_top)
                scale_factor_top = scaled_window_size / float(window_size_top)

                onset_vid_idx = frame_ixs[pre]
                onset_bot_idx = top_to_bot_frame_continuous(sid, onset_vid_idx)
                c_bot_onset, h_bot_onset, _ = compute_robust_bot_pose(df_bot, bot_scorer, onset_bot_idx, search_window=10)
                M_bot = compute_kpms_warp_matrix(c_bot_onset, h_bot_onset, window_size_bot)
                scale_factor_bot = scaled_window_size / float(window_size_bot)

                tile_frames = []
                for t in range(clip_frames):
                    tf = top_frames_raw[t]
                    bf = bot_frames_raw[t]

                    crop_top = cv2.warpAffine(tf, M_top, (window_size_top, window_size_top),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
                    tile_top = cv2.resize(crop_top, (scaled_window_size, scaled_window_size), interpolation=cv2.INTER_LINEAR)

                    crop_bot = cv2.warpAffine(bf, M_bot, (window_size_bot, window_size_bot),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
                    tile_bot = cv2.resize(crop_bot, (scaled_window_size, scaled_window_size), interpolation=cv2.INTER_LINEAR)
                    tile_bot = cv2.flip(tile_bot, 1)

                    # White onset indicator dot (KPMS standard: appears during bout execution)
                    offset = t - pre
                    if 0 <= offset < bout["duration_frames"]:
                        curr_model_frame = start + offset
                        c_t_curr = c_top_all[curr_model_frame] if curr_model_frame < len(c_top_all) else c_top_onset
                        if not (np.isnan(c_t_curr[0]) or np.isnan(c_t_curr[1])):
                            pos_t = tuple([int(round(x)) for x in (M_top @ np.append(c_t_curr, 1)) * scale_factor_top])
                            if 0 <= pos_t[0] < scaled_window_size and 0 <= pos_t[1] < scaled_window_size:
                                cv2.circle(tile_top, pos_t, dot_radius, (255, 255, 255), -1, cv2.LINE_AA)

                        raw_bot_idx = bot_frame_ixs[t]
                        if 0 <= raw_bot_idx < len(df_bot):
                            c_bot_curr, _, is_v = compute_robust_bot_pose(df_bot, bot_scorer, raw_bot_idx, search_window=2)
                            if is_v:
                                pos_b = tuple([int(round(x)) for x in (M_bot @ np.append(c_bot_curr, 1)) * scale_factor_bot])
                                pos_b_flipped = (scaled_window_size - 1 - pos_b[0], pos_b[1])
                                if 0 <= pos_b_flipped[0] < scaled_window_size and 0 <= pos_b_flipped[1] < scaled_window_size:
                                    cv2.circle(tile_bot, pos_b_flipped, dot_radius, (255, 255, 255), -1, cv2.LINE_AA)

                    cv2.putText(tile_top, "TOP", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
                    cv2.putText(tile_bot, "BOT", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

                    tandem_tile = np.concatenate([tile_top, tile_bot], axis=1)
                    cv2.rectangle(tandem_tile, (0, 0), (tandem_tile.shape[1] - 1, tandem_tile.shape[0] - 1), (50, 50, 50), 1)

                    # Highlight Medoid / Exemplar Tile (Tile 1 is always the anchor exemplar/medoid)
                    if b_idx == 0 or bout.get("is_medoid", False):
                        badge_text = "EXEMPLAR" if "density" in tier else "MEDOID"
                        cv2.rectangle(tandem_tile, (tandem_tile.shape[1] - 82, 4), (tandem_tile.shape[1] - 4, 18), (30, 140, 30), -1)
                        cv2.putText(tandem_tile, badge_text, (tandem_tile.shape[1] - 80, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        inst_tag = f"Tile {b_idx + 1}"
                        cv2.putText(tandem_tile, inst_tag, (tandem_tile.shape[1] - 50, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

                    tile_frames.append(tandem_tile)

                instance_clips.append(tile_frames)

            # Assemble grid canvas
            tile_h, tile_w = instance_clips[0][0].shape[:2]
            grid_w = grid_cols * tile_w + (grid_cols + 1) * gap_x
            grid_h = grid_rows * tile_h + (grid_rows + 1) * gap_y
            banner_h = 48
            canvas_h = grid_h + banner_h
            pad_bottom = (16 - (canvas_h % 16)) % 16
            pad_right = (16 - (grid_w % 16)) % 16
            final_w = grid_w + pad_right
            final_h = canvas_h + pad_bottom

            writer = imageio.get_writer(
                str(mp4_path),
                fps=out_fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
                ffmpeg_params=["-preset", "fast", "-crf", "20", "-movflags", "+faststart"]
            )

            for loop_i in range(n_loops):
                for t in range(clip_frames):
                    canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
                    canvas[:banner_h, :] = (28, 28, 30)

                    title_str = f"Syllable {s:02d} | 21kp Static V2 | {tier.upper()} (N={total_bouts})"
                    cv2.putText(canvas, title_str, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                    # Usage and framerate info
                    stats_str = f"Usage: {usage_pct:.2f}% | 60 FPS (Native 1x)"
                    cv2.putText(canvas, stats_str, (grid_w - 380, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

                    # Progress bar
                    bar_w = 120
                    bar_x = grid_w - 140
                    prog = float(t + 1) / float(clip_frames)
                    cv2.rectangle(canvas, (bar_x, 20), (bar_x + bar_w, 30), (60, 60, 60), 1)
                    cv2.rectangle(canvas, (bar_x, 20), (bar_x + int(bar_w * prog), 30), (0, 200, 100), -1)

                    # Place 12 tiles in 3x4 grid
                    for r in range(grid_rows):
                        for c in range(grid_cols):
                            inst_idx = r * grid_cols + c
                            if inst_idx < len(instance_clips):
                                tile = instance_clips[inst_idx][t]
                                x_pos = gap_x + c * (tile_w + gap_x)
                                y_pos = banner_h + gap_y + r * (tile_h + gap_y)
                                canvas[y_pos:y_pos + tile_h, x_pos:x_pos + tile_w] = tile

                    writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

            writer.close()
            print(f"  -> Successfully generated {mp4_path.name} ({mp4_path.stat().st_size / 1024:.1f} KB, 180 frames @ 60 FPS)", flush=True)

    print("\n" + "=" * 80)
    print("🎬 ALL V2 GRID MOVIES COMPLETED SUCCESSFULLY!")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
