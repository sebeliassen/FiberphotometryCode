#!/usr/bin/env python3
"""
render_all_cohort_static_21kp_v2_seed101_grid_movies.py
========================================================
Publication-Grade Dual-View 3x4 Grid Movie Renderer for Cohort Static 21kp V2:
- Renders the EXACT bouts selected in `unified_syllable_selection_manifest.json`!
- Zero silent session-ordered substitutions: any target execution failure becomes
  an explicitly flagged blank tile preserving 1-to-1 manifest alignment.
- Handles dropped BotCam frames (-1) by rendering a dimmed tile with [BOT DROPPED]
  rather than loading frame 0.
- Handles unoriented BotCam pose by centering without arbitrary rotation and adding
  [BOT UNROTATED] badge.
- Generates JSON provenance sidecars (.provenance.json) tracking exact rendered IDs,
  hashes of manifest, model results, code, and calibration.

Environment: kpms_gpu_cuml (pure CPU)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import h5py
import imageio
import numpy as np
import pandas as pd

from fpcode.common.paths import get_repo_root
from fpcode.tracking.camera_correspondence import top_to_bot_frame_continuous
from fpcode.tracking.pair_matching import find_matched_pairs
from fpcode.viz import v2 as vc

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = get_repo_root()
EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"

COHORT_DIR = EXP_DIR / "cohort_static_21kp_v2"
MODEL_NAME = "model_21kp_static_50ar_300slds_seed101"
MODEL_DIR = COHORT_DIR / MODEL_NAME
RESULTS_PATH = MODEL_DIR / "results.h5"
VFI_PATH = COHORT_DIR / "video_frame_indexes.npz"

DEFAULT_VIZ_DIR = vc.DEFAULT_VIZ_DIR
DEFAULT_MANIFEST_PATH = DEFAULT_VIZ_DIR / "unified_syllable_selection_manifest.json"
DEFAULT_OUT_DIR = DEFAULT_VIZ_DIR / "grid_movies_density"


class OpenCVReader:
    """Thread-safe random-access OpenCV video reader with sequential seek caching."""
    def __init__(self, path: str):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {self.path}")
        self.curr_pos = -1

    def __getitem__(self, idx: int) -> np.ndarray:
        idx = int(idx)
        if idx < 0:
            raise IndexError(f"Negative video frame {idx} in {self.path}")
        if idx != self.curr_pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise IOError(f"Decode failure at frame {idx} in {self.path}; refusing to certify a black frame as video")
        self.curr_pos = idx
        return frame

    def close(self):
        if self.cap:
            self.cap.release()


def compute_kpms_warp_matrix(centroid_onset: Tuple[float, float], heading_onset: float, window_size: int) -> np.ndarray:
    """Maps centroid_onset to (window_size/2, window_size/2) rotated by heading_onset."""
    c = np.array(centroid_onset, dtype=np.float32)
    h = float(heading_onset) if not np.isnan(heading_onset) else 0.0
    r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
    c_shift = r @ c - window_size // 2
    M = np.float32([[np.cos(h), np.sin(h), -c_shift[0]], [-np.sin(h), np.cos(h), -c_shift[1]]])
    return M


def validate_bout_for_render(bout: Dict[str, Any], vfi_dict: Dict[str, np.ndarray],
                             res_file: h5py.File, pre: int, post: int) -> Tuple[bool, str]:
    """Target validator: verify video files and tracking exist for this bout."""
    if not Path(bout.get("top_vid", "")).exists():
        return False, "top_video_missing"
    if not Path(bout.get("bot_vid", "")).exists():
        return False, "bot_video_missing"
    if not Path(bout.get("bot_h5", "")).exists():
        return False, "bot_h5_missing"

    sid = bout.get("segment_name", "")
    if sid not in vfi_dict or sid not in res_file:
        return False, "session_not_in_results"

    grp = res_file[sid]
    cents = grp["centroid"][:]
    heads = grp["heading"][:]
    start = int(bout.get("onset_segment_frame", 0))
    if start < pre or (start + post) > len(cents):
        return False, "window_out_of_bounds"

    c0 = cents[start]
    h0 = heads[start]
    if np.isnan(c0).any() or np.isnan(h0):
        return False, "nan_pose_at_onset"

    return True, "ok"


def render_blank_tile(scaled_window_size: int, clip_frames: int, tile_rank: int,
                      is_flagged: bool = False, reason: str = "") -> List[np.ndarray]:
    """Generates an inactive/blank placeholder tile across clip_frames."""
    frames = []
    blank_tile = np.zeros((scaled_window_size, scaled_window_size * 2, 3), dtype=np.uint8)
    blank_tile[:] = (18, 18, 22)
    cv2.rectangle(blank_tile, (0, 0), (blank_tile.shape[1] - 1, blank_tile.shape[0] - 1), (40, 40, 48), 1)

    if is_flagged:
        cv2.putText(blank_tile, "FLAGGED BLANK", (scaled_window_size - 60, scaled_window_size // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 220), 1, cv2.LINE_AA)
        reason_txt = reason[:24] if reason else "Target QC Failed"
        cv2.putText(blank_tile, reason_txt, (scaled_window_size - 68, scaled_window_size // 2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 160), 1, cv2.LINE_AA)
    else:
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
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST_PATH),
                        help="Path to unified selection manifest")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="Target output directory for MP4 movies")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing MP4 files")
    parser.add_argument("--overlay_keypoints", action="store_true", help="Overlay original supported dorsal/paw points for verification")
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
    grid_rows, grid_cols = vc.grid_layout(grid_size)

    # 1.0s prior, 2.0s post @ 60 FPS
    out_fps = float(manifest["fps"])
    movie_win = manifest["movie_window"]
    if float(movie_win["fps"]) != out_fps:
        raise ValueError("Movie-window FPS differs from manifest FPS")
    pre = int(movie_win["pre_frames"])
    post = int(movie_win["post_frames"])
    clip_frames = pre + post     # 180 frames = 3.0 s
    n_loops = 1

    window_size_top = 144
    window_size_bot = 384
    scaled_window_size = 160
    dot_radius = 4
    gap_x, gap_y = 16, 16

    with np.load(VFI_PATH) as z:
        vfi_dict = {k: z[k] for k in z.files}

    pairs = find_matched_pairs()
    pairs_by_sid = {p["sid"]: p for p in pairs}

    bot_cache = {}

    all_syll_entries = manifest["syllables"]
    if args.syllables is not None:
        target_sylls = [s for s in args.syllables if str(s) in all_syll_entries]
    else:
        target_sylls = [
            int(s) for s, data in sorted(all_syll_entries.items(), key=lambda x: int(x[0]))
            if data.get("is_active_0p1pct", True)
        ]

    # Provenance hashing cache
    hash_cache = vc.HashCache(out_dir / ".hash_cache.json")
    vc.verify_manifest_inputs(manifest, hash_cache)
    code_files = [Path(__file__).resolve(), Path(vc.__file__)]

    print("=" * 80)
    print("🎬 RENDERING REPAIRED V2 COHORT STATIC 21KP DUAL-VIEW GRID MOVIES")
    print(f"Timing Window:  pre={pre}f (1.0s), post={post}f (2.0s) -> {clip_frames}f (3.0s total @ {out_fps} FPS)")
    print(f"Grid Layout:    {grid_rows}x{grid_cols} ({grid_size} tiles)")
    print(f"Manifest Path:  {manifest_path}")
    print(f"Output Target:  {out_dir}")
    print("=" * 80)

    with h5py.File(RESULTS_PATH, "r") as res_file:
        for idx, s in enumerate(target_sylls):
            s_str = str(s)
            s_data = all_syll_entries[s_str]
            manifest_bouts = s_data["bouts"]
            tier = s_data["tier"]
            total_bouts = s_data["total_cohort_bouts"]
            usage_pct = s_data.get("usage_pct", 0.0)

            mp4_path = out_dir / f"syllable_{s:02d}_tandem_grid_density.mp4"

            # Cache key for provenance
            cache_key = vc.make_cache_key(
                kind="grid_movie",
                syllable=s,
                manifest=manifest,
                manifest_path=manifest_path,
                hash_cache=hash_cache,
                code_files=code_files,
                render_settings={
                    "out_fps": out_fps, "pre": pre, "post": post, "clip_frames": clip_frames,
                    "grid_rows": grid_rows, "grid_cols": grid_cols,
                    "window_size_top": window_size_top, "window_size_bot": window_size_bot,
                    "scaled_window_size": scaled_window_size
                    ,"canonical_bottom_warp":"top_crop_times_recorded_fusion_affine",
                    "overlay_keypoints":args.overlay_keypoints
                }
            )

            reuse, reason = vc.cache_decision([mp4_path], cache_key)
            if not args.overwrite and reuse:
                print(f"  [{idx+1}/{len(target_sylls)}] Syllable {s:02d} cached ({reason}), skipping.", flush=True)
                continue

            print(f"\n▶ [{idx+1}/{len(target_sylls)}] Syllable {s:02d} ({total_bouts} bouts, {usage_pct:.2f}% usage, {tier}) -> {mp4_path.name}", flush=True)

            # Resolve display bouts WITHOUT silent substitution
            display_bouts = vc.resolve_display_bouts(
                manifest_bouts,
                lambda b: validate_bout_for_render(b, vfi_dict, res_file, pre, post),
                log=print
            )

            # Assert rendered sequence matches manifest sequence exactly
            vc.assert_rendered_equals_manifest(display_bouts, manifest_bouts)
            invalid = [b for b in display_bouts if b.get("flagged_invalid")]
            if invalid:
                raise RuntimeError(f"Final selection failed display validation: {[(b['bout_id'],b['invalid_reason']) for b in invalid]}. Rebuild selection; do not substitute or emit a divergent plot/movie set.")

            instance_clips = []
            rendered_metadata = []

            for b_idx, bout in enumerate(display_bouts):
                if bout.get("is_blank", False):
                    is_flagged = bool(bout.get("flagged_invalid", False))
                    reason_txt = bout.get("invalid_reason", "")
                    blank_frames = render_blank_tile(scaled_window_size, clip_frames, b_idx,
                                                     is_flagged=is_flagged, reason=reason_txt)
                    instance_clips.append(blank_frames)
                    rendered_metadata.append({
                        "tile_index": b_idx, "is_blank": True, "flagged_invalid": is_flagged,
                        "invalid_reason": reason_txt, "bout_id": bout.get("bout_id")
                    })
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

                # Filtered continuous pose
                grp = res_file[sid]
                c_top_filt, h_top_filt = vc.filter_segment_pose(grp["centroid"][:], grp["heading"][:])

                bot_h5_path = bout["bot_h5"]
                if bot_h5_path not in bot_cache:
                    df_b = pd.read_hdf(bot_h5_path)
                    bot_cache[bot_h5_path] = (df_b, df_b.columns.levels[0][0])
                df_bot, bot_scorer = bot_cache[bot_h5_path]

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
                        # Dropped BotCam frame: represent as None
                        bot_frames_raw.append(None)
                top_reader.close()
                bot_reader.close()

                top_h, top_w = top_frames_raw[0].shape[:2]
                c_top_onset = c_top_filt[start]
                h_top_onset = h_top_filt[start]

                if (c_top_onset[0] < 0 or c_top_onset[0] >= top_w or
                    c_top_onset[1] < 0 or c_top_onset[1] >= top_h or
                    np.isnan(c_top_onset[0]) or np.isnan(c_top_onset[1])):
                    raise ValueError(f"Onset centroid outside TopCam image for {bout['bout_id']}")

                geometry = vc.fusion_geometry(vc.base_session_id(sid))
                M_top, M_bot = vc.canonical_crop_matrices(c_top_onset,h_top_onset,
                    geometry["A"],geometry["b"],geometry["reflection_width"],scaled_window_size,window_size_top)
                overlay_xy = overlay_conf = None
                if args.overlay_keypoints:
                    overlay_xy, overlay_conf = vc.load_fused_tracks(vc.base_session_id(sid))

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
                        cv2.putText(tile_bot, "BOT DROPPED", (scaled_window_size // 2 - 48, scaled_window_size // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 200), 1, cv2.LINE_AA)

                    # White onset indicator dot (KPMS standard: appears during canonical bout execution)
                    offset = t - pre
                    if 0 <= offset < bout["duration_frames"]:
                        curr_model_frame = start + offset
                        c_t_curr = c_top_filt[curr_model_frame] if curr_model_frame < len(c_top_filt) else c_top_onset
                        pos_t = None
                        if np.isfinite(c_t_curr).all():
                            pos_t = tuple(int(round(x)) for x in (M_top @ np.append(c_t_curr, 1)))
                            if 0 <= pos_t[0] < scaled_window_size and 0 <= pos_t[1] < scaled_window_size:
                                cv2.circle(tile_top, pos_t, dot_radius, (255, 255, 255), -1, cv2.LINE_AA)

                        if bf is not None and pos_t is not None and 0 <= pos_t[0] < scaled_window_size and 0 <= pos_t[1] < scaled_window_size:
                            cv2.circle(tile_bot, pos_t, dot_radius, (255, 255, 255), -1, cv2.LINE_AA)

                    if overlay_xy is not None:
                        src = int(frame_ixs[t])
                        if not 0 <= src < len(overlay_xy):
                            raise ValueError(f"Overlay VFI out of range for {bout['bout_id']}")
                        supported = vc.valid_track_mask(overlay_xy[src],overlay_conf[src],manifest["qc_config"]["conf_threshold"])
                        for kp in np.flatnonzero(supported):
                            point = tuple(int(round(x)) for x in M_top @ np.r_[overlay_xy[src,kp],1.])
                            if 0 <= point[0] < scaled_window_size and 0 <= point[1] < scaled_window_size:
                                if kp < 17:
                                    cv2.circle(tile_top,point,2,(255,200,60),-1,cv2.LINE_AA)
                                elif bf is not None:
                                    cv2.circle(tile_bot,point,2,(80,120,255),-1,cv2.LINE_AA)

                    cv2.putText(tile_top, "TOP", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
                    cv2.putText(tile_bot, "BOT", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

                    tandem_tile = np.concatenate([tile_top, tile_bot], axis=1)
                    cv2.rectangle(tandem_tile, (0, 0), (tandem_tile.shape[1] - 1, tandem_tile.shape[0] - 1), (50, 50, 50), 1)

                    # Badge: Tile 0 is always the anchor exemplar/medoid
                    if b_idx == 0 or bout.get("is_anchor", False) or bout.get("is_medoid", False):
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
                rendered_metadata.append({
                    "tile_index": b_idx, "is_blank": False, "bout_id": bout["bout_id"],
                    "sid": sid, "onset_segment_frame": start, "duration_frames": bout["duration_frames"],
                    "top_frame_indices":[int(f) for f in frame_ixs],
                    "bot_frame_indices":[int(f) for f in bot_frame_ixs],
                    "top_crop_matrix":M_top.tolist(),"bot_crop_matrix":M_bot.tolist(),
                    "fusion_geometry":{k:v for k,v in geometry.items() if k not in ("A","b")},
                    "onset_centroid":c_top_onset.tolist(),"onset_heading":float(h_top_onset),
                    "top_onset_frame_sha256":vc.hashlib.sha256(top_frames_raw[pre].tobytes()).hexdigest(),
                    "bot_onset_frame_sha256":vc.hashlib.sha256(bot_frames_raw[pre].tobytes()).hexdigest() if bot_frames_raw[pre] is not None else None,
                    "dropped_bot_frames":sum(f is None for f in bot_frames_raw),
                })

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

            temp_mp4 = mp4_path.with_name(mp4_path.stem+".rendering.mp4")
            writer = imageio.get_writer(
                str(temp_mp4),
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

                    stats_str = f"Usage: {usage_pct:.2f}% | {out_fps} FPS (Native 1x)"
                    cv2.putText(canvas, stats_str, (grid_w - 380, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

                    # Progress bar
                    bar_w = 120
                    bar_x = grid_w - 140
                    prog = float(t + 1) / float(clip_frames)
                    cv2.rectangle(canvas, (bar_x, 20), (bar_x + bar_w, 30), (60, 60, 60), 1)
                    cv2.rectangle(canvas, (bar_x, 20), (bar_x + int(bar_w * prog), 30), (0, 200, 100), -1)

                    # Place tiles in grid
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
            os.replace(temp_mp4,mp4_path)

            # Write provenance sidecar atomically
            sidecar_file = vc.write_sidecar(
                output_paths=[mp4_path],
                key=cache_key,
                manifest_path=manifest_path,
                hash_cache=hash_cache,
                payload={
                    "syllable": s,
                    "rendered_tiles": rendered_metadata,
                    "kpms_provenance": vc.kpms_provenance()
                }
            )

            print(f"  -> Generated {mp4_path.name} ({mp4_path.stat().st_size / 1024:.1f} KB, 180 frames @ {out_fps} FPS) [Provenance: {sidecar_file.name}]", flush=True)

    print("\n" + "=" * 80)
    print("🎬 REPAIRED V2 GRID MOVIES COMPLETED SUCCESSFULLY!")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
