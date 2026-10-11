#!/usr/bin/env python3
"""
render_all_seed404_grid_movies.py
=============================================================================
Publication-grade Tandem Dual-View Syllable Grid Movies for Champion Seed 404:
- TWO MODES:
  1. --mode density : Built-in contrastive density sampling (prototypical exemplars)
     -> Output: visualizations_seed404/grid_movies_density/
  2. --mode random  : Uniform spread sampling across cohort sessions
     -> Output: visualizations_seed404/grid_movies_random/
- FULL COHORT COVERAGE:
  Handles all syllables in Champion Seed 404. Syllables with fewer than 12
  instances are gracefully tiled across the 12 panels so every syllable is watchable.
- Visual Architecture:
  1. Pure clean video frames WITHOUT pose overlays.
  2. Stationary onset crop (no tracking/rotation chatter).
  3. White centroid indicator dot appears at onset and vanishes at offset.
  4. Frame-accurate FP3002 TTL / Basler crystal hardware clock synchronization.
  5. Scale-matched 160x160 px tiles (144px TopCam vs 384px BotCam normalized).
  6. Exact 16-divisible canvas (1360 x 592 px) eliminating macroblock padding.

Environment: kpms_gpu_cuml
=============================================================================
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np
import pandas as pd
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

from fpcode.tracking.camera_correspondence import top_to_bot_frame_continuous
from fpcode.tracking.pair_matching import find_matched_pairs

# Paths
PROJECT_DIR = EXP_DIR / "kpms_tandem_cohort_continuous_calibrated"
MODEL_NAME = "tandem_calibrated_50ar_200slds_seed404_20261002_012237_940073"
RESULTS_PATH = PROJECT_DIR / MODEL_NAME / "results.h5"
VFI_PATH = PROJECT_DIR / "video_frame_indexes.npz"

TOP_VID_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/studies/karen_pff_vs_monomer/video_files")
BOT_VID_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/studies/karen_pff_vs_monomer/botcam_video_files")

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
                    c = (float(0.5 * (nx + tx)), float(0.5 * (ny + ty)))
                    h = float(np.arctan2(ny - ty, nx - tx))
                    return c, h, True
        except Exception:
            continue

    # Multi-keypoint fallback across all bodyparts with likelihood >= 0.8
    try:
        bodyparts = df_bot.columns.levels[1]
        xs, ys = [], []
        for bp in bodyparts:
            x = df_bot[bot_scorer, bp, "x"].iloc[target_idx]
            y = df_bot[bot_scorer, bp, "y"].iloc[target_idx]
            p = df_bot[bot_scorer, bp, "likelihood"].iloc[target_idx]
            if p >= 0.8 and not np.isnan(x) and not np.isnan(y):
                xs.append(x)
                ys.append(y)
        if len(xs) >= 2:
            return (float(np.mean(xs)), float(np.mean(ys))), 0.0, True
    except Exception:
        pass

    return (486.0, 486.0), 0.0, False

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
    parser = argparse.ArgumentParser(description="Render Tandem Dual-View Syllable Grid Movies for Seed 404")
    parser.add_argument("--mode", type=str, choices=["density", "random"], default="density",
                        help="Instance selection mode: 'density' (exemplar nearest neighbors) or 'random' (uniform spread)")
    parser.add_argument("--syllables", nargs="+", type=int, default=None,
                        help="Specific syllable IDs to render (default: render all available syllables)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing MP4 files")
    args = parser.parse_args()

    mode = args.mode
    out_dir = SCRIPT_DIR / "visualizations_seed404" / f"grid_movies_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🎬 RENDERING SEED 404 DUAL-VIEW GRID MOVIES (MODE: {mode.upper()})")
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
        sess_keys = sorted(f.keys())
        for k in sess_keys:
            if k not in vfi_dict or len(vfi_dict[k]) < 500:
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

        manifest_dir = EXP_DIR / "frozen_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"frozen_bouts_seed404_syll{s:02d}_{mode}.json"

        n_avail = len(bouts)
        chosen_bouts = None

        if manifest_path.exists() and not args.overwrite:
            try:
                with open(manifest_path, "r") as mf:
                    loaded_data = json.load(mf)
                raw_bouts = loaded_data["bouts"] if isinstance(loaded_data, dict) and "bouts" in loaded_data else loaded_data
                # Validate that all bouts belong to syllable s in current results.h5
                manifest_valid = True
                with h5py.File(RESULTS_PATH, "r") as res_chk:
                    for b in raw_bouts:
                        bsid = b["sid"]
                        bstart = int(b["start"])
                        bend = int(b["end"])
                        if bsid not in res_chk:
                            manifest_valid = False
                            break
                        syll_slice = res_chk[bsid]["syllable"][bstart:bend]
                        if not np.all(syll_slice == s):
                            manifest_valid = False
                            break
                if manifest_valid and len(raw_bouts) == n_instances:
                    chosen_bouts = raw_bouts
                    sampling_label = f"Verified Frozen Manifest ({len(chosen_bouts)} bouts from {manifest_path.name})"
                else:
                    print(f"  ⚠️ Existing manifest {manifest_path.name} failed syllable {s} validation. Resampling fresh bouts...", flush=True)
            except Exception as e:
                print(f"  ⚠️ Error reading manifest {manifest_path.name}: {e}. Resampling fresh bouts...", flush=True)

        if chosen_bouts is None:
            if mode == "density" and n_avail >= 12:
                # Typical duration exemplar: median duration bout and nearest neighbors
                durs = np.array([b["len"] for b in bouts])
                med_dur = np.median(durs)
                dur_diffs = np.abs(durs - med_dur)
                best_order = np.argsort(dur_diffs)
                chosen_bouts = [bouts[i] for i in best_order[:n_instances]]
                sampling_label = f"Median Duration Exemplars ({n_avail} bouts in cohort)"
            elif mode == "random" and n_avail >= 12:
                # Uniform spread across sessions
                step = max(1, n_avail // n_instances)
                chosen_bouts = [bouts[i * step] for i in range(n_instances)]
                sampling_label = f"Uniform Spread ({n_avail} bouts in cohort)"
            else:
                # Low instance fallback: tile available instances across the 12 panels
                chosen_bouts = [bouts[i % n_avail] for i in range(n_instances)]
                sampling_label = f"Rare Syllable ({n_avail} unique bouts tiled)"

            serializable_bouts = []
            for b in chosen_bouts:
                sb = {k: int(v) if isinstance(v, (np.integer, int)) else str(v) for k, v in b.items()}
                serializable_bouts.append(sb)

            manifest_payload = {
                "model_name": MODEL_NAME,
                "syllable_id": s,
                "mode": mode,
                "n_bouts": len(serializable_bouts),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "bouts": serializable_bouts,
            }
            with open(manifest_path, "w") as mf:
                json.dump(manifest_payload, mf, indent=2)
            chosen_bouts = serializable_bouts

        print(f"\n▶ [{idx+1}/{len(target_sylls)}] {desc}: {sampling_label} -> {mp4_path.name}", flush=True)

        instance_clips = []
        with h5py.File(RESULTS_PATH, "r") as res_file:
            for b_idx, bout in enumerate(chosen_bouts):
                sid = bout["sid"]
                start = bout["start"]
                end = bout["end"]
                vfi = vfi_dict[sid]

                # Symmetric boundary padding ensuring frame_ixs[pre] is STRICTLY vfi[start]
                vfi_len = len(vfi)
                left_idx = start - pre
                right_idx = start + post

                pad_left = max(0, -left_idx)
                pad_right = max(0, right_idx - vfi_len)

                slice_start = max(0, left_idx)
                slice_end = min(vfi_len, right_idx)

                raw_slice = vfi[slice_start:slice_end]
                frame_ixs = np.pad(raw_slice, (pad_left, pad_right), mode="edge")
                assert len(frame_ixs) == clip_frames, f"Frame count mismatch: {len(frame_ixs)} != {clip_frames}"
                assert int(frame_ixs[pre]) == int(vfi[start]), f"Onset alignment invariant failed! frame_ixs[{pre}]={frame_ixs[pre]} != vfi[{start}]={vfi[start]}"

                # Dorsal tracking from results.h5
                grp = res_file[sid]
                c_top_all = grp["centroid"][:]
                h_top_all = grp["heading"][:]

                # Load BotCam DLC tracking table once for onset heading/centroid
                df_bot = pd.read_hdf(bout["bot_h5"])
                bot_scorer = df_bot.columns.levels[0][0]

                # Read Video Streams
                top_reader = OpenCVReader(bout["top_vid"])
                bot_reader = OpenCVReader(bout["bot_vid"])

                top_frames_raw = [top_reader[int(fix)] for fix in frame_ixs]
                bot_frame_ixs = [
                    top_to_bot_frame_continuous(sid, int(fix))
                    for fix in frame_ixs
                ]
                bot_frames_raw = []
                bot_frames_valid = []
                last_valid_bf = None
                for b_fix in bot_frame_ixs:
                    if 0 <= b_fix < len(df_bot):
                        frame = bot_reader[int(b_fix)]
                        last_valid_bf = frame
                        bot_frames_raw.append(frame)
                        bot_frames_valid.append(True)
                    else:
                        bot_frames_valid.append(False)
                        if last_valid_bf is not None:
                            # Dim repeated frame slightly to avoid confusing with live movement
                            dimmed = (last_valid_bf.astype(np.float32) * 0.45).astype(np.uint8)
                            bot_frames_raw.append(dimmed)
                        else:
                            bot_frames_raw.append(bot_reader[0])
                top_reader.close()
                bot_reader.close()

                top_h, top_w = top_frames_raw[0].shape[:2]
                bot_h, bot_w = bot_frames_raw[0].shape[:2]

                # TopCam onset centroid & heading (FIXED for entire clip)
                c_top_onset = c_top_all[start]
                h_top_onset = h_top_all[start]

                df_top_override = None
                top_scorer_override = None
                if (c_top_onset[0] < 0 or c_top_onset[0] >= top_w or
                    c_top_onset[1] < 0 or c_top_onset[1] >= top_h or
                    np.isnan(c_top_onset[0]) or np.isnan(c_top_onset[1])):
                    try:
                        df_top_override = pd.read_hdf(bout["top_h5"])
                        top_scorer_override = df_top_override.columns.levels[0][0]
                        raw_top_idx = int(frame_ixs[pre])
                        if raw_top_idx < len(df_top_override):
                            if "mouse_center" in df_top_override.columns.levels[1]:
                                cx = df_top_override[top_scorer_override, "mouse_center", "x"].iloc[raw_top_idx]
                                cy = df_top_override[top_scorer_override, "mouse_center", "y"].iloc[raw_top_idx]
                                cp = df_top_override[top_scorer_override, "mouse_center", "likelihood"].iloc[raw_top_idx]
                                if cp > 0.4 and 0 <= cx < top_w and 0 <= cy < top_h:
                                    c_top_onset = (cx, cy)
                            if "nose" in df_top_override.columns.levels[1] and "tail_base" in df_top_override.columns.levels[1]:
                                nx = df_top_override[top_scorer_override, "nose", "x"].iloc[raw_top_idx]
                                ny = df_top_override[top_scorer_override, "nose", "y"].iloc[raw_top_idx]
                                tx = df_top_override[top_scorer_override, "tail_base", "x"].iloc[raw_top_idx]
                                ty = df_top_override[top_scorer_override, "tail_base", "y"].iloc[raw_top_idx]
                                h_top_onset = np.arctan2(ny - ty, nx - tx)
                    except Exception:
                        pass

                M_top = compute_kpms_warp_matrix(c_top_onset, h_top_onset, window_size_top)
                scale_factor_top = scaled_window_size / float(window_size_top)

                onset_vid_idx = frame_ixs[pre]
                onset_bot_idx = top_to_bot_frame_continuous(sid, onset_vid_idx)
                bot_stream_active = (0 <= onset_bot_idx < len(df_bot))
                if onset_bot_idx < 0:
                    valid_candidates = [bf for bf in bot_frame_ixs if 0 <= bf < len(df_bot)]
                    onset_bot_idx = valid_candidates[0] if valid_candidates else 0
                    if valid_candidates:
                        bot_stream_active = True
                c_bot_onset, h_bot_onset, is_v_onset = compute_robust_bot_pose(df_bot, bot_scorer, onset_bot_idx, search_window=10)
                M_bot = compute_kpms_warp_matrix(c_bot_onset, h_bot_onset, window_size_bot)
                scale_factor_bot = scaled_window_size / float(window_size_bot)

                # Compose side-by-side tile
                tile_frames = []
                for t in range(clip_frames):
                    tf = top_frames_raw[t]
                    bf = bot_frames_raw[t]

                    crop_top = cv2.warpAffine(tf, M_top, (window_size_top, window_size_top), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
                    tile_top = cv2.resize(crop_top, (scaled_window_size, scaled_window_size), interpolation=cv2.INTER_LINEAR)

                    if bot_stream_active and is_v_onset:
                        crop_bot = cv2.warpAffine(bf, M_bot, (window_size_bot, window_size_bot), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
                        tile_bot = cv2.resize(crop_bot, (scaled_window_size, scaled_window_size), interpolation=cv2.INTER_LINEAR)
                        tile_bot = cv2.flip(tile_bot, 1)
                        if not bot_frames_valid[t]:
                            cv2.putText(tile_bot, "DROPPED", (scaled_window_size - 68, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 140, 255), 1, cv2.LINE_AA)
                    else:
                        tile_bot = np.zeros((scaled_window_size, scaled_window_size, 3), dtype=np.uint8)
                        status_msg = "ACQ ENDED" if not bot_stream_active else "TRACKING LOST"
                        x_offset = 40 if status_msg == "ACQ ENDED" else 26
                        cv2.putText(tile_bot, "VENTRAL", (48, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)
                        cv2.putText(tile_bot, status_msg, (x_offset, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)

                    offset = t - pre
                    if 0 <= offset < bout["len"]:
                        curr_model_frame = start + offset
                        c_t_curr = c_top_all[curr_model_frame] if curr_model_frame < len(c_top_all) else c_top_onset
                        if df_top_override is not None:
                            raw_t_fix = int(frame_ixs[t])
                            if raw_t_fix < len(df_top_override):
                                cx = df_top_override[top_scorer_override, "mouse_center", "x"].iloc[raw_t_fix]
                                cy = df_top_override[top_scorer_override, "mouse_center", "y"].iloc[raw_t_fix]
                                cp = df_top_override[top_scorer_override, "mouse_center", "likelihood"].iloc[raw_t_fix]
                                if cp > 0.4 and 0 <= cx < top_w and 0 <= cy < top_h:
                                    c_t_curr = (cx, cy)
                        elif (np.isnan(c_t_curr[0]) or np.isnan(c_t_curr[1]) or
                            c_t_curr[0] < 0 or c_t_curr[0] >= top_w or
                            c_t_curr[1] < 0 or c_t_curr[1] >= top_h):
                            c_t_curr = c_top_onset
                        pos_t = tuple([int(round(x)) for x in (M_top @ np.append(c_t_curr, 1)) * scale_factor_top])
                        if 0 <= pos_t[0] < scaled_window_size and 0 <= pos_t[1] < scaled_window_size:
                            cv2.circle(tile_top, pos_t, dot_radius, (255, 255, 255), -1, cv2.LINE_AA)

                        if bot_stream_active and is_v_onset and bot_frames_valid[t]:
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

                    if n_avail < 12:
                        inst_tag = f"Inst {b_idx % n_avail + 1}/{n_avail}"
                        cv2.putText(tandem_tile, inst_tag, (tandem_tile.shape[1] - 75, 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1, cv2.LINE_AA)

                    tile_frames.append(tandem_tile)

                instance_clips.append(tile_frames)

        # Assemble 3x4 Grid Composite (1360 x 592 px cleanly divisible by 16)
        tile_w = scaled_window_size * 2
        tile_h = scaled_window_size
        banner_h = 48
        grid_w = grid_cols * tile_w + (grid_cols + 1) * gap_x
        grid_h = grid_rows * tile_h + (grid_rows + 1) * gap_y + banner_h

        assembled_frames = []
        for t in range(clip_frames):
            canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            title_text = f"TANDEM MULTI-VIEW KEYPOINT-MOSEQ | {desc.upper()} ({mode.upper()})"
            sub_text = f"TopCam (17 KPs Dorsal) Left  |  BotCam (6 KPs Ventral) Right  |  {sampling_label}  |  FPS: 60"
            cv2.putText(canvas, title_text, (gap_x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, sub_text, (gap_x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

            for i, clip in enumerate(instance_clips):
                r = i // grid_cols
                c = i % grid_cols
                x = gap_x + c * (tile_w + gap_x)
                y = banner_h + gap_y + r * (tile_h + gap_y)
                canvas[y : y + tile_h, x : x + tile_w] = clip[t]

            assembled_frames.append(canvas)

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
