#!/usr/bin/env python3
"""
render_all_cohort_static_21kp_v2_seed101_trajectories.py
=========================================================
Publication-grade Trajectory Visualizations for Cohort Static 21kp V2:
- Standard Keypoint-MoSeq Median Trajectories:
  * Computed as the elementwise median (np.nanmedian) across a large representative
    sample of qualifying bouts (up to 150 bouts per syllable) across cohort sessions.
  * Temporal window: pre = 10 frames (-167 ms) and post = 30 frames (+500 ms) @ 60 FPS
    (total 40 frames, compensating for 60 FPS vs legacy 30 FPS).
  * Egocentrically aligned to onset centroid and heading (c0, h0).
- Color Palettes:
  * Dorsal Posture (17 kps): Cool Blues (plt.cm.Blues).
  * Ventral Paws (4 kps): Red front paws, Orange/Yellow hind paws.
- Layouts:
  1. Tandem Side-by-Side Plots (Dorsal Spine on Left | Ventral Paws on Right):
     - Decoupled view-specific axis limits (tandem_viz_guide.md standard).
     - Strobe/fade progression across time.
     - Vector PDF + High-res PNG + Native 60 FPS MP4 + GIF preview.
  2. Superimposed Trajectories:
     - Dorsal spine in cool blues + ventral paws in red/orange-yellow in shared frame.
     - Vector PDF + High-res PNG + Native 60 FPS MP4 + GIF preview.
- Provenance sidecars (.provenance.json) tracking sample counts and window definitions.

Environment: kpms_gpu_cuml (pure CPU)
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import h5py
import pickle
import yaml
import imageio.v2 as imageio
from PIL import Image

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXP_DIR.parent.parent

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import viz_v2_common as vc
from jax_moseq.utils import unbatch

COHORT_DIR = EXP_DIR / "cohort_static_21kp_v2"
MODEL_NAME = "model_21kp_static_50ar_300slds_seed101"
MODEL_DIR = COHORT_DIR / MODEL_NAME
RESULTS_PATH = MODEL_DIR / "results.h5"
CONFIG_PATH = COHORT_DIR / "config.yml"
PREPROCESSED_PATH = COHORT_DIR / "preprocessed_data.pkl"

DEFAULT_VIZ_DIR = vc.DEFAULT_VIZ_DIR
DEFAULT_MANIFEST_PATH = DEFAULT_VIZ_DIR / "unified_syllable_selection_manifest.json"


def get_view_limits(pts: np.ndarray, pctl: float = 0.0, pad: float = 0.15) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Computes axis limits padded appropriately for visualization."""
    X = pts.reshape(-1, 2)
    xmin, ymin = np.nanpercentile(X, pctl, axis=0)
    xmax, ymax = np.nanpercentile(X, 100.0 - pctl, axis=0)
    w, h = max(float(xmax - xmin), 1.0), max(float(ymax - ymin), 1.0)
    return (float(xmin - w * pad), float(xmax + w * pad)), (float(ymin - h * pad), float(ymax + h * pad))


def interp_traj(traj: np.ndarray, num_steps: int = 12) -> np.ndarray:
    """Resamples a (T, K, 2) trajectory to num_steps points along time."""
    T, K, D = traj.shape
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, num_steps)
    res = np.zeros((num_steps, K, D), dtype=np.float32)
    for k in range(K):
        for d in range(D):
            res[:, k, d] = np.interp(x_new, x_old, traj[:, k, d])
    return res


def rasterize_figure(fig) -> np.ndarray:
    """Converts a matplotlib figure canvas to a detached uint8 RGB array."""
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    elif hasattr(fig.canvas, "tostring_rgb"):
        w, h = fig.canvas.get_width_height()
        return np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8").reshape((h, w, 3)).copy()
    return np.asarray(fig.canvas.print_to_buffer()[0])[:, :, :3].copy()


def save_gif(image_list: List[np.ndarray], gif_filename: Path, duration: float = 0.10) -> None:
    """Native Keypoint-MoSeq GIF serializer using PIL Image objects."""
    pil_images = [Image.fromarray(np.uint8(img)) for img in image_list]
    centiseconds = np.rint(np.arange(len(pil_images) + 1) * duration * 100).astype(int)
    delays_ms = (np.maximum(1, np.diff(centiseconds)) * 10).tolist()
    pil_images[0].save(
        str(gif_filename),
        save_all=True,
        append_images=pil_images[1:],
        duration=delays_ms,
        loop=0,
    )


def save_native_movie(image_list: List[np.ndarray], path: Path, fps: float) -> None:
    """Saves native 60 FPS MP4 video clip."""
    temp = path.with_name(path.stem + ".rendering.mp4")
    with imageio.get_writer(
        str(temp),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-preset", "fast", "-crf", "20"]
    ) as writer:
        for frame in image_list:
            writer.append_data(frame)
    os.replace(temp, path)


def render_side_by_side_bundle(
    median_traj: np.ndarray,
    edges_dorsal: List[Tuple[int, int]],
    edges_ventral: List[Tuple[int, int]],
    colors_dorsal: np.ndarray,
    colors_ventral: np.ndarray,
    s: int,
    n_bouts: int,
    fps: float = 60.0,
    pre: int = 10,
    num_steps: int = 12
) -> Tuple[plt.Figure, List[np.ndarray]]:
    """Renders side-by-side static figure and animation frames."""
    total_frames = len(median_traj)
    X_res = interp_traj(median_traj, num_steps=num_steps)
    xlim_d, ylim_d = get_view_limits(median_traj[:, :17], pad=0.15)
    xlim_v, ylim_v = get_view_limits(median_traj[:, 17:], pad=0.20)

    # 1. Static Strobe Figure
    fig_static, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.8), dpi=150)
    fig_static.patch.set_facecolor("white")
    fig_static.suptitle(
        f"Syllable {s:02d} Trajectory ([-10, +30]f @ {fps:g} FPS)",
        fontsize=12, fontweight="bold", y=0.96
    )

    ax1.set_title("TopCam Dorsal Posture (17 kps)", fontsize=11, pad=8)
    ax1.set_xlim(xlim_d); ax1.set_ylim(ylim_d[1], ylim_d[0])
    ax1.set_aspect("equal"); ax1.axis("off")

    ax2.set_title("BotCam Ventral Paws (4 kps)", fontsize=11, pad=8)
    ax2.set_xlim(xlim_v); ax2.set_ylim(ylim_v[1], ylim_v[0])
    ax2.set_aspect("equal"); ax2.axis("off")

    for i in range(num_steps):
        is_last = (i == num_steps - 1)
        prog = i / (num_steps - 1)
        alpha = 0.15 + 0.85 * (prog ** 1.5)
        lw = 2.0 if is_last else 1.0
        sz = 22 if is_last else 10

        pt_d = X_res[i, :17]
        for ii, jj in edges_dorsal:
            ax1.plot([pt_d[ii, 0], pt_d[jj, 0]], [pt_d[ii, 1], pt_d[jj, 1]],
                     color="black", linewidth=lw * 1.15, alpha=alpha * 0.8, zorder=i * 4)
            ax1.plot([pt_d[ii, 0], pt_d[jj, 0]], [pt_d[ii, 1], pt_d[jj, 1]],
                     color=colors_dorsal[ii], linewidth=lw, alpha=alpha, zorder=i * 4 + 1)
        ax1.scatter(pt_d[:, 0], pt_d[:, 1], s=sz, color=colors_dorsal,
                    edgecolors="black", linewidths=0.5, alpha=alpha, zorder=i * 4 + 2)

        pt_v = X_res[i, 17:]
        for ii, jj in edges_ventral:
            ax2.plot([pt_v[ii, 0], pt_v[jj, 0]], [pt_v[ii, 1], pt_v[jj, 1]],
                     color="gray", linewidth=lw * 0.9, alpha=alpha * 0.6, zorder=i * 4)
        for k in range(4):
            ax2.scatter(pt_v[k, 0], pt_v[k, 1], s=sz * 1.2, color=colors_ventral[k],
                        edgecolors="black", linewidths=0.6, alpha=alpha, zorder=i * 4 + 2)

    # 2. Animation Frames across the 40 frames
    fig_anim, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.8), dpi=120)
    fig_anim.patch.set_facecolor("white")
    fig_anim.suptitle(
        f"Syllable {s:02d} Trajectory ([-10, +30]f @ {fps:g} FPS)",
        fontsize=12, fontweight="bold", y=0.96
    )
    a1.set_title("TopCam Dorsal Posture (17 kps)", fontsize=11, pad=8)
    a1.set_xlim(xlim_d); a1.set_ylim(ylim_d[1], ylim_d[0])
    a1.set_aspect("equal"); a1.axis("off")

    a2.set_title("BotCam Ventral Paws (4 kps)", fontsize=11, pad=8)
    a2.set_xlim(xlim_v); a2.set_ylim(ylim_v[1], ylim_v[0])
    a2.set_aspect("equal"); a2.axis("off")

    time_txt = a1.text(0.02, 0.02, "", transform=a1.transAxes, fontsize=9, family="monospace")
    history_steps = set(np.rint(np.linspace(0, total_frames - 1, 12)).astype(int))

    active_lines = []
    active_scatters = []
    rasters = []

    for t_idx in range(total_frames):
        if t_idx in history_steps:
            pt_dh = median_traj[t_idx, :17]
            for ii, jj in edges_dorsal:
                a1.plot([pt_dh[ii, 0], pt_dh[jj, 0]], [pt_dh[ii, 1], pt_dh[jj, 1]],
                        color="black", linewidth=1.0, alpha=0.15, zorder=1)
                a1.plot([pt_dh[ii, 0], pt_dh[jj, 0]], [pt_dh[ii, 1], pt_dh[jj, 1]],
                        color=colors_dorsal[ii], linewidth=0.9, alpha=0.18, zorder=2)
            a1.scatter(pt_dh[:, 0], pt_dh[:, 1], s=8, color=colors_dorsal,
                       edgecolors="black", linewidths=0.3, alpha=0.20, zorder=3)

            pt_vh = median_traj[t_idx, 17:]
            for ii, jj in edges_ventral:
                a2.plot([pt_vh[ii, 0], pt_vh[jj, 0]], [pt_vh[ii, 1], pt_vh[jj, 1]],
                        color="gray", linewidth=0.8, alpha=0.15, zorder=1)
            for k in range(4):
                a2.scatter(pt_vh[k, 0], pt_vh[k, 1], s=10, color=colors_ventral[k],
                           edgecolors="black", linewidths=0.4, alpha=0.20, zorder=3)

        for art in active_lines + active_scatters:
            art.remove()
        active_lines.clear()
        active_scatters.clear()

        pt_d = median_traj[t_idx, :17]
        for ii, jj in edges_dorsal:
            l1, = a1.plot([pt_d[ii, 0], pt_d[jj, 0]], [pt_d[ii, 1], pt_d[jj, 1]],
                          color="black", linewidth=2.0, alpha=0.9, zorder=20)
            l2, = a1.plot([pt_d[ii, 0], pt_d[jj, 0]], [pt_d[ii, 1], pt_d[jj, 1]],
                          color=colors_dorsal[ii], linewidth=1.8, alpha=1.0, zorder=21)
            active_lines.extend([l1, l2])
        sc1 = a1.scatter(pt_d[:, 0], pt_d[:, 1], s=25, color=colors_dorsal,
                         edgecolors="black", linewidths=0.7, alpha=1.0, zorder=22)
        active_scatters.append(sc1)

        pt_v = median_traj[t_idx, 17:]
        for ii, jj in edges_ventral:
            l3, = a2.plot([pt_v[ii, 0], pt_v[jj, 0]], [pt_v[ii, 1], pt_v[jj, 1]],
                          color="gray", linewidth=1.5, alpha=0.8, zorder=20)
            active_lines.append(l3)
        for k in range(4):
            sc2 = a2.scatter(pt_v[k, 0], pt_v[k, 1], s=28, color=colors_ventral[k],
                             edgecolors="black", linewidths=0.7, alpha=1.0, zorder=22)
            active_scatters.append(sc2)

        t_sec = (t_idx - pre) / fps
        time_txt.set_text(f"t = {t_sec:+.3f}s | {fps:g} fps")
        rasters.append(rasterize_figure(fig_anim))

    plt.close(fig_anim)
    return fig_static, rasters


def render_superimposed_bundle(
    median_traj: np.ndarray,
    edges_super: List[Tuple[int, int]],
    colors_super: np.ndarray,
    s: int,
    n_bouts: int,
    fps: float = 60.0,
    pre: int = 10,
    num_steps: int = 12
) -> Tuple[plt.Figure, List[np.ndarray]]:
    """Renders superimposed static figure and animation frames."""
    total_frames = len(median_traj)
    X_res = interp_traj(median_traj, num_steps=num_steps)
    xlim_s, ylim_s = get_view_limits(median_traj, pad=0.15)

    # 1. Static Strobe Figure
    fig_static, ax_s = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    fig_static.patch.set_facecolor("white")
    ax_s.set_title(
        f"Syllable {s:02d} Superimposed Trajectory ([-10, +30]f @ {fps:g} FPS)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax_s.set_xlim(xlim_s); ax_s.set_ylim(ylim_s[1], ylim_s[0])
    ax_s.set_aspect("equal"); ax_s.axis("off")

    for i in range(num_steps):
        is_last = (i == num_steps - 1)
        prog = i / (num_steps - 1)
        alpha = 0.15 + 0.85 * (prog ** 1.5)
        lw = 2.0 if is_last else 1.0
        sz = 20 if is_last else 9

        pt = X_res[i]
        for ii, jj in edges_super:
            ax_s.plot([pt[ii, 0], pt[jj, 0]], [pt[ii, 1], pt[jj, 1]],
                      color="black", linewidth=lw * 1.15, alpha=alpha * 0.8, zorder=i * 4)
            ax_s.plot([pt[ii, 0], pt[jj, 0]], [pt[ii, 1], pt[jj, 1]],
                      color=colors_super[ii], linewidth=lw, alpha=alpha, zorder=i * 4 + 1)
        ax_s.scatter(pt[:, 0], pt[:, 1], s=sz, color=colors_super,
                     edgecolors="black", linewidths=0.5, alpha=alpha, zorder=i * 4 + 2)

    # 2. Animation Frames across the 40 frames
    fig_anim, ax_a = plt.subplots(figsize=(6.4, 4.8), dpi=120)
    fig_anim.patch.set_facecolor("white")
    ax_a.set_title(
        f"Syllable {s:02d} Superimposed Trajectory ([-10, +30]f @ {fps:g} FPS)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax_a.set_xlim(xlim_s); ax_a.set_ylim(ylim_s[1], ylim_s[0])
    ax_a.set_aspect("equal"); ax_a.axis("off")

    time_txt = ax_a.text(0.02, 0.02, "", transform=ax_a.transAxes, fontsize=9, family="monospace")
    history_steps = set(np.rint(np.linspace(0, total_frames - 1, 12)).astype(int))

    active_lines = []
    active_scatters = []
    rasters = []

    for t_idx in range(total_frames):
        if t_idx in history_steps:
            pt_h = median_traj[t_idx]
            for ii, jj in edges_super:
                ax_a.plot([pt_h[ii, 0], pt_h[jj, 0]], [pt_h[ii, 1], pt_h[jj, 1]],
                          color="black", linewidth=1.0, alpha=0.15, zorder=1)
                ax_a.plot([pt_h[ii, 0], pt_h[jj, 0]], [pt_h[ii, 1], pt_h[jj, 1]],
                          color=colors_super[ii], linewidth=0.9, alpha=0.18, zorder=2)
            ax_a.scatter(pt_h[:, 0], pt_h[:, 1], s=8, color=colors_super,
                         edgecolors="black", linewidths=0.3, alpha=0.20, zorder=3)

        for art in active_lines + active_scatters:
            art.remove()
        active_lines.clear()
        active_scatters.clear()

        pt = median_traj[t_idx]
        for ii, jj in edges_super:
            l1, = ax_a.plot([pt[ii, 0], pt[jj, 0]], [pt[ii, 1], pt[jj, 1]],
                            color="black", linewidth=2.0, alpha=0.9, zorder=20)
            l2, = ax_a.plot([pt[ii, 0], pt[jj, 0]], [pt[ii, 1], pt[jj, 1]],
                            color=colors_super[ii], linewidth=1.8, alpha=1.0, zorder=21)
            active_lines.extend([l1, l2])
        sc = ax_a.scatter(pt[:, 0], pt[:, 1], s=25, color=colors_super,
                          edgecolors="black", linewidths=0.7, alpha=1.0, zorder=22)
        active_scatters.append(sc)

        t_sec = (t_idx - pre) / fps
        time_txt.set_text(f"t = {t_sec:+.3f}s | {fps:g} fps")
        rasters.append(rasterize_figure(fig_anim))

    plt.close(fig_anim)
    return fig_static, rasters


def main():
    parser = argparse.ArgumentParser(description="Render Standard KPMS Median Trajectories for Cohort Static 21kp V2")
    parser.add_argument("--syllables", nargs="+", type=int, default=None,
                        help="Specific syllable IDs to render")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST_PATH),
                        help="Path to selection manifest")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Target output directory for trajectories")
    parser.add_argument("--max_bouts", type=int, default=None,
                        help="Max qualifying bouts per syllable to sample for median calculation (default: None for all bouts)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing output files")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}!")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    viz_dir = manifest_path.parent
    out_dir = Path(args.out_dir) if args.out_dir else (viz_dir / "trajectories")
    super_dir = out_dir / "superimposed"
    side_dir = out_dir / "side_by_side"
    super_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)

    # Skeleton configuration
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    use_bps = config["use_bodyparts"]
    skeleton = config.get("skeleton", [])
    edges_all = [
        (use_bps.index(e[0]), use_bps.index(e[1]))
        for e in skeleton if e[0] in use_bps and e[1] in use_bps
    ]

    extra_paw_edges = [
        (17, 18),  # LF - RF
        (19, 20),  # LH - RH
        (17, 19),  # LF - LH
        (18, 20),  # RF - RH
    ]
    edges_super = list(set(edges_all + extra_paw_edges))
    edges_dorsal = [(i, j) for (i, j) in edges_all if i < 17 and j < 17]
    edges_ventral = [(i - 17, j - 17) for (i, j) in extra_paw_edges]

    # Exactly requested palettes: cool Blues for dorsal, Red and Orange/Yellow for ventral
    colors_dorsal = plt.cm.Blues(np.linspace(0.45, 0.95, 17))[:, :3]
    colors_ventral = np.array([
        [0.90, 0.15, 0.15],  # Left Front (Red)
        [0.90, 0.15, 0.15],  # Right Front (Red)
        [0.95, 0.55, 0.10],  # Left Hind (Orange/Yellow)
        [0.95, 0.55, 0.10],  # Right Hind (Orange/Yellow)
    ])
    colors_super = np.vstack([colors_dorsal, colors_ventral])

    # Load coordinates
    print("Loading unbatched coordinates from preprocessed_data.pkl...", flush=True)
    with open(PREPROCESSED_PATH, "rb") as pf:
        data, metadata = pickle.load(pf)
    coords_dict = unbatch(data["Y"], *metadata)
    del data

    hash_cache = vc.HashCache(out_dir / ".hash_cache.json")
    code_files = [Path(__file__).resolve(), SCRIPT_DIR / "viz_v2_common.py"]

    fps = 60.0
    pre = 10   # 167 ms @ 60 FPS
    post = 30  # 500 ms @ 60 FPS
    max_bouts = args.max_bouts

    with h5py.File(RESULTS_PATH, "r") as res_file:
        all_syll_entries = manifest["syllables"]
        if args.syllables is not None:
            target_sylls = [s for s in args.syllables if str(s) in all_syll_entries]
        else:
            target_sylls = [
                int(s) for s, data in sorted(all_syll_entries.items(), key=lambda x: int(x[0]))
                if data.get("is_active_0p1pct", True)
            ]

        print("=" * 80)
        print("🎬 RENDERING STANDARD KPMS MEDIAN TRAJECTORIES (CANONICAL N-BOUT ELEMENTWISE MEDIAN)")
        print(f"Manifest:       {manifest_path.name}")
        print(f"Target Sylls:   {len(target_sylls)} syllables")
        print(f"Window:         [-{pre}, +{post}] frames ({-pre/fps*1000:.0f}ms to +{post/fps*1000:.0f}ms @ {fps:g} FPS)")
        print(f"Max Bouts/Syll: {max_bouts}")
        print(f"Superimposed:   {super_dir}")
        print(f"Side-by-Side:   {side_dir}")
        print("=" * 80)

        # Index qualifying bouts across sessions
        print("Indexing qualifying bouts across all sessions...", flush=True)
        session_keys = list(res_file.keys())
        bouts_by_syll = {s: [] for s in target_sylls}
        target_set = set(target_sylls)

        for sid in session_keys:
            syll = res_file[sid]["syllable"][:]
            starts, ends = vc.half_open_bouts(syll)
            for st, en in zip(starts, ends):
                s_curr = int(syll[st])
                if s_curr in target_set:
                    if st >= pre and st + post <= len(syll):
                        if max_bouts is None or max_bouts <= 0 or len(bouts_by_syll[s_curr]) < max_bouts:
                            bouts_by_syll[s_curr].append((sid, st))

        filt_pose_cache = {}
        def get_filtered_pose(sid):
            if sid not in filt_pose_cache:
                filt_pose_cache[sid] = vc.filter_segment_pose(res_file[sid]["centroid"][:], res_file[sid]["heading"][:])
            return filt_pose_cache[sid]

        for idx, s in enumerate(target_sylls):
            pdf_super = super_dir / f"syllable_{s:02d}_trajectory_superimposed.pdf"
            png_super = super_dir / f"syllable_{s:02d}_trajectory_superimposed.png"
            gif_super = super_dir / f"syllable_{s:02d}_trajectory_superimposed.gif"
            pdf_side = side_dir / f"syllable_{s:02d}_trajectory_side_by_side.pdf"
            png_side = side_dir / f"syllable_{s:02d}_trajectory_side_by_side.png"
            gif_side = side_dir / f"syllable_{s:02d}_trajectory_side_by_side.gif"
            mp4_super = super_dir / f"syllable_{s:02d}_trajectory_superimposed.mp4"
            mp4_side = side_dir / f"syllable_{s:02d}_trajectory_side_by_side.mp4"

            out_files = [pdf_super, png_super, gif_super, pdf_side, png_side, gif_side, mp4_super, mp4_side]

            cache_key = vc.make_cache_key(
                kind="standard_median_trajectories",
                syllable=s,
                manifest=manifest,
                manifest_path=manifest_path,
                hash_cache=hash_cache,
                code_files=code_files,
                render_settings={"pre": pre, "post": post, "fps": fps, "max_bouts": max_bouts, "method": "nanmedian"}
            )

            reuse, reason = vc.cache_decision(out_files, cache_key)
            if not args.overwrite and reuse:
                print(f"  [{idx+1}/{len(target_sylls)}] Syllable {s:02d} cached ({reason}), skipping.", flush=True)
                continue

            qual_bouts = bouts_by_syll[s]
            if len(qual_bouts) == 0:
                print(f"  Warning: No qualifying bouts for Syllable {s:02d}, skipping.", flush=True)
                continue

            print(f"\n▶ [{idx+1}/{len(target_sylls)}] Syllable {s:02d} (extracting {len(qual_bouts)} bouts for median)...", flush=True)

            ego_bouts = []
            for sid, st in qual_bouts:
                c_filt, h_filt = get_filtered_pose(sid)
                c0 = c_filt[st]
                h0 = h_filt[st]
                if not np.isfinite(c0).all() or not np.isfinite(h0):
                    continue
                Y_all = coords_dict[sid]
                Y_win = Y_all[st - pre : st + post]
                ego = vc.egocentric_window(Y_win, c0, h0)
                ego_bouts.append(ego)

            if len(ego_bouts) == 0:
                print(f"  Warning: No finite egocentric poses for Syllable {s:02d}, skipping.", flush=True)
                continue

            median_traj = np.nanmedian(np.array(ego_bouts), axis=0) # (40, 21, 2)
            n_actual = len(ego_bouts)

            # 1. Side-by-Side
            fig_side, rasters_side = render_side_by_side_bundle(
                median_traj, edges_dorsal, edges_ventral, colors_dorsal, colors_ventral,
                s=s, n_bouts=n_actual, fps=fps, pre=pre, num_steps=12
            )
            fig_side.savefig(pdf_side, bbox_inches="tight")
            fig_side.savefig(png_side, bbox_inches="tight", dpi=300)
            plt.close(fig_side)
            save_gif(rasters_side, gif_side, duration=1.0 / fps)
            save_native_movie(rasters_side, mp4_side, fps)
            del rasters_side

            # 2. Superimposed
            fig_super, rasters_super = render_superimposed_bundle(
                median_traj, edges_super, colors_super,
                s=s, n_bouts=n_actual, fps=fps, pre=pre, num_steps=12
            )
            fig_super.savefig(pdf_super, bbox_inches="tight")
            fig_super.savefig(png_super, bbox_inches="tight", dpi=300)
            plt.close(fig_super)
            save_gif(rasters_super, gif_super, duration=1.0 / fps)
            save_native_movie(rasters_super, mp4_super, fps)
            del rasters_super

            # Write provenance sidecar
            vc.write_sidecar(
                output_paths=out_files,
                key=cache_key,
                manifest_path=manifest_path,
                hash_cache=hash_cache,
                payload={
                    "syllable": s,
                    "method": "standard_kpms_median_trajectory",
                    "n_bouts_median": n_actual,
                    "pre_frames": pre,
                    "post_frames": post,
                    "fps": fps,
                    "timing_reference": "native_fps_mp4; GIF is an approximate preview"
                }
            )

            print(f"  -> Generated Syllable {s:02d} side_by_side & superimposed (PDF, PNG, MP4, GIF)", flush=True)

    print("\n" + "=" * 80)
    print("🎬 ALL STANDARD KPMS MEDIAN TRAJECTORIES COMPLETED SUCCESSFULLY!")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
