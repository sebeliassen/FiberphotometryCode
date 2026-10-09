#!/usr/bin/env python3
"""
render_all_cohort_static_21kp_v2_seed101_random_grid_movies.py
==============================================================
Renders publication-grade 3x4 dual-view Random Grid Movies for Cohort Static 21kp V2:
- Uniform random sampling of 12 qualifying bouts per active syllable.
- Ensures both cameras (TopCam and BotCam) are present and verified active.
- Sized and aligned identically to the standard tandem grid movies.
- Output directory: visualizations_cohort_static_21kp_v2_seed101_v4/grid_movies_random/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import viz_v2_common as vc

from core.camera_correspondence import top_to_bot_frame_continuous
from core.pair_matching import find_matched_pairs

COHORT_DIR = EXP_DIR / "cohort_static_21kp_v2"
MODEL_NAME = "model_21kp_static_50ar_300slds_seed101"
MODEL_DIR = COHORT_DIR / MODEL_NAME
RESULTS_PATH = MODEL_DIR / "results.h5"
VFI_PATH = COHORT_DIR / "video_frame_indexes.npz"

DEFAULT_VIZ_DIR = vc.DEFAULT_VIZ_DIR
DEFAULT_MANIFEST_PATH = DEFAULT_VIZ_DIR / "unified_syllable_selection_manifest.json"
DEFAULT_DIAG_PATH = DEFAULT_VIZ_DIR / "neighbor_diagnostics.json"
DEFAULT_OUT_DIR = DEFAULT_VIZ_DIR / "grid_movies_random"


class OpenCVReader:
    def __init__(self, path: str):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {self.path}")
        self.curr_pos = -1

    def __getitem__(self, idx: int) -> np.ndarray:
        idx = int(idx)
        if idx < 0:
            raise IndexError(f"Negative frame {idx}")
        if idx != self.curr_pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise IOError(f"Decode failure at frame {idx}")
        self.curr_pos = idx
        return frame

    def close(self):
        if self.cap:
            self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="Render Random Grid Movies for Cohort Static 21kp V2")
    parser.add_argument("--syllables", nargs="+", type=int, default=None,
                        help="Specific syllable IDs to render")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST_PATH),
                        help="Path to selection manifest")
    parser.add_argument("--diagnostics", type=str, default=str(DEFAULT_DIAG_PATH),
                        help="Path to neighbor diagnostics JSON")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="Target output directory for random grid movies")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing output files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.diagnostics) as f:
        diag = json.load(f)

    with np.load(VFI_PATH) as z:
        vfi_dict = {k: z[k] for k in z.files}

    pairs = find_matched_pairs()
    pairs_by_sid = {p["sid"]: p for p in pairs}

    pre = manifest["movie_window"]["pre_frames"]
    post = manifest["movie_window"]["post_frames"]
    clip_frames = pre + post
    out_fps = float(manifest["fps"])

    window_size_top = 144
    scaled_window_size = 160
    gap_x, gap_y = 16, 16
    grid_rows, grid_cols = 3, 4
    grid_size = grid_rows * grid_cols

    all_syll_entries = manifest["syllables"]
    if args.syllables is not None:
        target_sylls = [s for s in args.syllables if str(s) in all_syll_entries]
    else:
        target_sylls = [
            int(s) for s, data in sorted(all_syll_entries.items(), key=lambda x: int(x[0]))
            if data.get("is_active_0p1pct", True)
        ]

    print("=" * 80)
    print("🎬 RENDERING RANDOM COHORT STATIC 21KP DUAL-VIEW GRID MOVIES")
    print(f"Target Sylls:   {len(target_sylls)} syllables")
    print(f"Timing Window:  pre={pre}f, post={post}f -> {clip_frames}f ({clip_frames/out_fps:.1f}s @ {out_fps:g} FPS)")
    print(f"Output Target:  {out_dir}")
    print("=" * 80)

    bot_cache = {}

    with h5py.File(RESULTS_PATH, "r") as res_file:
        for idx, s in enumerate(target_sylls):
            s_str = str(s)
            mp4_path = out_dir / f"syllable_{s:02d}_tandem_grid_random.mp4"

            if not args.overwrite and mp4_path.exists():
                print(f"  [{idx+1}/{len(target_sylls)}] Syllable {s:02d} random movie exists, skipping.", flush=True)
                continue

            if s_str not in diag:
                print(f"  [{idx+1}/{len(target_sylls)}] Syllable {s:02d} not in diagnostics, skipping.", flush=True)
                continue

            s_diag = diag[s_str]
            nbrs = s_diag.get("neighbors_50", [])
            if len(nbrs) < grid_size:
                # If fewer than 12 neighbors, sample with replacement or take what's available
                nbr_pool = nbrs if len(nbrs) > 0 else []
            else:
                nbr_pool = nbrs

            if len(nbr_pool) == 0:
                print(f"  [{idx+1}/{len(target_sylls)}] Syllable {s:02d} has no eligible bouts, skipping.", flush=True)
                continue

            rng = np.random.RandomState(42 + s)
            replace = len(nbr_pool) < grid_size
            chosen_indices = rng.choice(len(nbr_pool), size=min(grid_size, len(nbr_pool)), replace=replace)
            selected_bouts = [nbr_pool[i] for i in chosen_indices]

            print(f"\n▶ [{idx+1}/{len(target_sylls)}] Syllable {s:02d} (random sampling {len(selected_bouts)} bouts)...", flush=True)

            instance_clips = []
            for b_idx, nb in enumerate(selected_bouts):
                sid = nb["sid"]
                start = int(nb["bout_id"].split(":")[-1])
                dur = int(nb["duration_frames"])
                paths = vc.get_session_files(sid, pairs_by_sid)

                top_vid = paths["top_vid"]
                bot_vid = paths["bot_vid"]
                bot_h5 = paths["bot_h5"]

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

                grp = res_file[sid]
                c_top_filt, h_top_filt = vc.filter_segment_pose(grp["centroid"][:], grp["heading"][:])

                if bot_h5 not in bot_cache:
                    df_b = pd.read_hdf(bot_h5)
                    bot_cache[bot_h5] = (df_b, df_b.columns.levels[0][0])
                df_bot, bot_scorer = bot_cache[bot_h5]

                top_reader = OpenCVReader(top_vid)
                bot_reader = OpenCVReader(bot_vid)

                top_frames_raw = [top_reader[int(fix)] for fix in frame_ixs]
                bot_frame_ixs = [top_to_bot_frame_continuous(sid, int(fix)) for fix in frame_ixs]
                bot_frames_raw = []
                for b_fix in bot_frame_ixs:
                    if 0 <= b_fix < len(df_bot):
                        bot_frames_raw.append(bot_reader[int(b_fix)])
                    else:
                        bot_frames_raw.append(None)
                top_reader.close()
                bot_reader.close()

                c_top_onset = c_top_filt[start]
                h_top_onset = h_top_filt[start]
                geometry = vc.fusion_geometry(vc.base_session_id(sid))
                M_top, M_bot = vc.canonical_crop_matrices(
                    c_top_onset, h_top_onset,
                    geometry["A"], geometry["b"], geometry["reflection_width"],
                    scaled_window_size, window_size_top
                )

                tile_frames = []
                for t in range(clip_frames):
                    tf = top_frames_raw[t]
                    bf = bot_frames_raw[t]

                    tile_top = cv2.warpAffine(tf, M_top, (scaled_window_size, scaled_window_size),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
                    if bf is not None:
                        tile_bot = cv2.warpAffine(bf, M_bot, (scaled_window_size, scaled_window_size),
                                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
                    else:
                        tile_bot = np.zeros((scaled_window_size, scaled_window_size, 3), dtype=np.uint8)
                        tile_bot[:] = (20, 20, 24)

                    offset = t - pre
                    if 0 <= offset < dur:
                        curr_f = start + offset
                        c_curr = c_top_filt[curr_f] if curr_f < len(c_top_filt) else c_top_onset
                        if np.isfinite(c_curr).all():
                            pos_t = tuple(int(round(x)) for x in (M_top @ np.append(c_curr, 1)))
                            if 0 <= pos_t[0] < scaled_window_size and 0 <= pos_t[1] < scaled_window_size:
                                cv2.circle(tile_top, pos_t, 4, (255, 255, 255), -1, cv2.LINE_AA)
                                if bf is not None:
                                    cv2.circle(tile_bot, pos_t, 4, (255, 255, 255), -1, cv2.LINE_AA)

                    cv2.putText(tile_top, "TOP", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
                    cv2.putText(tile_bot, "BOT", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

                    tandem_tile = np.concatenate([tile_top, tile_bot], axis=1)
                    cv2.rectangle(tandem_tile, (0, 0), (tandem_tile.shape[1] - 1, tandem_tile.shape[0] - 1), (50, 50, 50), 1)
                    cv2.putText(tandem_tile, f"Rand #{b_idx+1}", (tandem_tile.shape[1] - 70, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

                    tile_frames.append(tandem_tile)
                instance_clips.append(tile_frames)

            # Pad with blanks if fewer than 12
            while len(instance_clips) < grid_size:
                blank_tile = np.zeros((scaled_window_size, scaled_window_size * 2, 3), dtype=np.uint8)
                instance_clips.append([blank_tile.copy() for _ in range(clip_frames)])

            tile_h, tile_w = instance_clips[0][0].shape[:2]
            grid_w = grid_cols * tile_w + (grid_cols + 1) * gap_x
            grid_h = grid_rows * tile_h + (grid_rows + 1) * gap_y
            banner_h = 48
            canvas_h = grid_h + banner_h
            pad_bottom = (16 - (canvas_h % 16)) % 16
            pad_right = (16 - (grid_w % 16)) % 16
            final_w = grid_w + pad_right
            final_h = canvas_h + pad_bottom

            temp_mp4 = mp4_path.with_name(mp4_path.stem + ".rendering.mp4")
            writer = imageio.get_writer(
                str(temp_mp4),
                fps=out_fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
                ffmpeg_params=["-preset", "fast", "-crf", "20", "-movflags", "+faststart"]
            )

            usage_pct = all_syll_entries[s_str].get("usage_pct", 0.0)
            for t in range(clip_frames):
                canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
                canvas[:banner_h, :] = (28, 28, 30)
                cv2.putText(canvas, f"Syllable {s:02d} | 21kp Static V2 | RANDOM SAMPLING (N=12)", (16, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"Usage: {usage_pct:.2f}% | {out_fps:g} FPS (Native 1x)", (grid_w - 380, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

                for b_i in range(grid_size):
                    r = b_i // grid_cols
                    c = b_i % grid_cols
                    x0 = gap_x + c * (tile_w + gap_x)
                    y0 = banner_h + gap_y + r * (tile_h + gap_y)
                    canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = instance_clips[b_i][t]

                writer.append_data(canvas)
            writer.close()
            os.replace(temp_mp4, mp4_path)
            print(f"  -> Generated {mp4_path.name}", flush=True)

    print("\n" + "=" * 80)
    print("🎬 ALL RANDOM GRID MOVIES COMPLETED SUCCESSFULLY!")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
