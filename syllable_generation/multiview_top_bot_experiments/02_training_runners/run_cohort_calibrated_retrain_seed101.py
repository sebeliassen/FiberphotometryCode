#!/usr/bin/env python3
"""
run_cohort_calibrated_retrain_seed101.py
=============================================================================
Full Cohort Production Tandem (Multi-View) Keypoint-MoSeq Corrected Retraining:
50 AR Iterations + 200 SLDS Iterations (250 Total Gibbs Iterations)
across ALL 118 Tandem Sessions with Continuous Rate-Aware Basler Synchronization,
Dual-Anchor Confidence & Plausibility Gating, and Prolonged Missing-Ventral Splitting.

Key Safeguards:
1. Strict Continuous Sync: 100% of included sessions have validated continuous
   Basler crystal clock mapping. No silent static offset fallbacks.
2. Dual-Anchor Plausibility & Gating: Both TopCam and BotCam nose/tail anchors
   are validated for confidence (p >= 0.8), finite coordinates, and body length
   (Top: 15-180 px, Bot: 25-350 px). Paws are strictly confidence-capped.
3. Prolonged Missing-Ventral Exclusion: Runs of missing ventral data (>= 180 frames / 3.0s)
   are excluded and sequences split into independent contiguous segments, preventing
   temporal joins across gaps and eliminating phantom dropout states (e.g. Syllable 45).
4. Ground Truth VFI: video_frame_indexes.npz maps every frame in every segment
   back to its exact raw video timeline index.
5. Host RAM Tensor Allocation: Formatting runs strictly in Host DRAM (128 GB pool)
   via jax.default_device(cpu), eliminating GPU VRAM OOM during device_put.
6. Preserves Champion Seed 101: Operates in a separate project directory:
   kpms_tandem_cohort_continuous_calibrated/
   Preserving all original baseline artifacts.
7. High Performance: Uses Amdahl 27x Banded Cholesky JAX kernels on NVIDIA RTX 4090.
=============================================================================
"""

import os
import sys

# Live unbuffered stdout/stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Hardware environment flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["USE_MEMORY_PATCH"] = "1"

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import copy
import time
import signal
import shutil
import pickle
import traceback
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from itertools import groupby
import pandas as pd
from scipy.ndimage import median_filter

import jax
import h5py
import gc
import psutil
import matplotlib
matplotlib.use("Agg")
from keypoint_moseq.fitting import plot_progress
import keypoint_moseq as kpms
from jax_moseq.utils.debugging import convert_data_precision

# Setup repository paths
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
KPMS_UTILS_DIR = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll"
if str(KPMS_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(KPMS_UTILS_DIR))

import utils.resample_batching as rb
from utils.batch_init import batched_init_model_by_video
from syllable_generation.syllabizers.keypoint_moseq_syll.utils.patch_checkpoint import apply_kpms_checkpoint_patch
from syllable_generation.syllabizers.keypoint_moseq_syll.utils.compact_checkpoint import compact_checkpoint

from syllable_generation.analysis.replications.datta_utils import combined_photometry_and_syllables
from syllable_generation.datamodel import SessionRecord
from syllable_generation.multiview_top_bot_experiments.audit_cohort_correlations import find_matched_pairs
from syllable_generation.multiview_top_bot_experiments.core.camera_correspondence import get_session_mapping

# Configuration constants
EXPERIMENT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
FUSED_DIR = EXPERIMENT_DIR / "fused_h5_continuous_calibrated"
DIRTY_PKL_IN = REPO_ROOT / "syllable_generation/data/sessions_karen_project_dirty.pkl"
K240K_PKL_REF = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_k240k.pkl"

PROD_KPMS_DIR = EXPERIMENT_DIR / "kpms_tandem_cohort_continuous_calibrated"
PROD_DATA_DIR = PROD_KPMS_DIR / "data"
MODEL_NAME = "tandem_cohort_calibrated_50ar_200slds_seed101"
OUTPUT_PROD_PKL = EXPERIMENT_DIR / "sessions_karen_project_tandem_cohort_continuous_calibrated_seed101.pkl"

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

NUM_AR_ITERS = 50
NUM_SLDS_ITERS = 200
TOTAL_ITERS = NUM_AR_ITERS + NUM_SLDS_ITERS
SAVE_EVERY_N_ITERS = 25

INIT_KAPPA = 7.5e13
DECREASE_KAPPA_FACTOR = 10.0
SLDS_KAPPA = INIT_KAPPA / DECREASE_KAPPA_FACTOR
TARGET_FPS = 60

# Prolonged missing ventral threshold (180 frames = 3.0s at 60 FPS)
MIN_GAP_FRAMES = 180
MIN_SEGMENT_FRAMES = 180

VENTRAL_BPS = [
    'left_front_paw', 'right_front_paw',
    'left_hind_paw', 'right_hind_paw',
    'bot_nose_tip', 'bot_tail_base'
]

def send_ntfy(title: str, msg: str, priority: str = "default", tags: str = "robot"):
    try:
        req = urllib.request.Request(
            NTFY_URL,
            data=msg.encode("utf-8"),
            headers={
                "Title": title.encode("ascii", "replace").decode("ascii"),
                "Priority": priority,
                "Tags": tags
            }
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            pass
        print(f"[ntfy] Alert sent: '{title}'", flush=True)
    except Exception as e:
        print(f"[ntfy warning] Could not send alert ({e})", flush=True)

def handle_shutdown_signal(signum, frame):
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n[shutdown] Caught signal {sig_name} (PID {os.getpid()}). Flushing and alerting...", flush=True)
    send_ntfy(
        f"KPMS Tandem Run Interrupted ({sig_name})",
        f"Process caught {sig_name}. Resuming this script will continue from the latest checkpoint.",
        priority="high",
        tags="warning,pause"
    )
    sys.exit(128 + signum)

signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)

def split_session_into_dual_view_segments(
    df: pd.DataFrame,
    min_gap_frames: int = MIN_GAP_FRAMES,
    min_segment_frames: int = MIN_SEGMENT_FRAMES,
    ventral_conf_thresh: float = 0.4
):
    """
    Splits recording into independent continuous dual-view sequence segments.
    Excludes runs of missing ventral data >= min_gap_frames (3.0s).
    """
    N = len(df)
    is_valid = np.zeros(N, dtype=bool)
    for bp in VENTRAL_BPS:
        col_x = ('fused_multiview', bp, 'x')
        col_p = ('fused_multiview', bp, 'likelihood')
        if col_x in df.columns and col_p in df.columns:
            finite = np.isfinite(df[col_x].to_numpy())
            conf = df[col_p].to_numpy() >= ventral_conf_thresh
            is_valid |= (finite & conf)

    runs = []
    idx = 0
    for val, g in groupby(is_valid):
        length = sum(1 for _ in g)
        runs.append((val, idx, idx + length, length))
        idx += length

    valid_mask = np.ones(N, dtype=bool)
    for val, s, e, l in runs:
        if not val and l >= min_gap_frames:
            valid_mask[s:e] = False

    segments = []
    seg_idx = 0
    for val, g in groupby(valid_mask):
        length = sum(1 for _ in g)
        if val and length >= min_segment_frames:
            segments.append((seg_idx, seg_idx + length))
        seg_idx += length

    return segments, is_valid

def _savetree_bounded(tree, group, name, path_prefix=""):
    """
    Recursively saves a pytree to HDF5 group leaf-by-leaf without gathering
    the entire tree onto host RAM at once.
    Transfers JAX arrays individually from GPU, writes directly, and frees host memory.
    Writes existing NumPy arrays directly from CPU memory with zero copies.
    Logs large arrays (>10 MB) with shape, dtype, size, and current host RSS.
    """
    current_path = f"{path_prefix}/{name}" if path_prefix else name
    if hasattr(tree, "device") or isinstance(tree, (np.ndarray, float, int, str, np.number)):
        rss_gb = psutil.Process().memory_info().rss / 1e9
        is_jax = hasattr(tree, "device")
        if is_jax:
            arr = np.asarray(jax.device_get(tree))
        else:
            arr = tree
        if hasattr(arr, "shape"):
            nbytes = arr.nbytes
            if nbytes > 1e7:  # > 10MB
                print(f"    [save_leaf] {current_path}: shape={arr.shape}, dtype={arr.dtype}, size={nbytes/1e6:.1f}MB, RSS={rss_gb:.2f}GB (jax={is_jax})", flush=True)
            if hasattr(arr, "dtype") and arr.dtype.kind == "U":
                dt = h5py.special_dtype(vlen=str)
                group.create_dataset(name, data=arr.astype(object), dtype=dt)
            else:
                group.create_dataset(name, data=arr)
        else:
            group.create_dataset(name, data=arr)
        if is_jax:
            del arr
            gc.collect()
    else:
        subgroup = group.create_group(name)
        subgroup.attrs["type"] = type(tree).__name__
        if isinstance(tree, (tuple, list)):
            for k, subtree in enumerate(tree):
                _savetree_bounded(subtree, subgroup, f"arr{k}", current_path)
        elif isinstance(tree, dict):
            for k, subtree in tree.items():
                _savetree_bounded(subtree, subgroup, k, current_path)
        else:
            raise ValueError(f"Unrecognized type {type(tree)}")


def _copy_group_bounded(src_grp, dst_grp, name, chunk_mb=64):
    """
    Stream an HDF5 group or dataset in bounded chunks (default 64 MiB)
    to prevent dirty writeback accumulation and unbounded kernel page cache spikes.
    """
    target = dst_grp.create_group(name)
    for k, v in src_grp.attrs.items():
        target.attrs[k] = v
    for key, item in src_grp.items():
        if isinstance(item, h5py.Group):
            _copy_group_bounded(item, target, key, chunk_mb=chunk_mb)
        elif isinstance(item, h5py.Dataset):
            item_bytes = item.dtype.itemsize * item.size
            if item_bytes <= chunk_mb * 1024 * 1024 or item.ndim == 0:
                target.create_dataset(key, data=item[...], dtype=item.dtype)
            else:
                ds = target.create_dataset(key, shape=item.shape, dtype=item.dtype)
                slice_len = max(1, int((chunk_mb * 1024 * 1024) / (item.dtype.itemsize * max(1, int(np.prod(item.shape[1:]))))))
                for s_i in range(0, item.shape[0], slice_len):
                    e_i = min(item.shape[0], s_i + slice_len)
                    ds[s_i:e_i] = item[s_i:e_i]


def save_rolling_checkpoint(
    project_dir: Path,
    model_name: str,
    model: dict,
    data: dict,
    metadata: tuple,
    current_iter: int,
    anchor_iter: int = 50,
    generate_progress_plots: bool = True
):
    """
    Atomically saves the current model snapshot to checkpoint.h5 and retains only:
      - Iteration anchor_iter (Phase 1 AR anchor)
      - The latest iteration (current_iter)
    Commits via atomic rename (tmp_path -> checkpoint.h5) so incomplete writes
    never corrupt previous checkpoints.
    Leaves immutable observation data in 'preprocessed_data.pkl', eliminating
    redundant 36+ GB host memory duplication.
    Flushes and drops dirty writeback via POSIX_FADV_DONTNEED to eliminate host OOM.
    """
    model_dir = Path(project_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "checkpoint.h5"
    tmp_path = model_dir / f"checkpoint.h5.tmp_{os.getpid()}_{current_iter}"

    t0 = time.time()
    rss_start = psutil.Process().memory_info().rss / 1e9
    print(f"  [checkpoint] Starting atomic rolling checkpoint for iter {current_iter} (RSS: {rss_start:.2f} GB)...", flush=True)

    with h5py.File(tmp_path, "w") as tmp:
        snaps_grp = tmp.create_group("model_snapshots")
        # Copy anchor snapshot from previous checkpoint if present and not current
        if checkpoint_path.exists() and current_iter != anchor_iter:
            try:
                with h5py.File(checkpoint_path, "r") as src:
                    if "model_snapshots" in src and str(anchor_iter) in src["model_snapshots"]:
                        _copy_group_bounded(src[f"model_snapshots/{anchor_iter}"], snaps_grp, str(anchor_iter))
            except Exception as e:
                print(f"  [checkpoint warning] Could not copy anchor {anchor_iter} from previous checkpoint: {e}", flush=True)

        # Write current snapshot leaf-by-leaf
        _savetree_bounded(model, snaps_grp, str(current_iter))
        # Write metadata
        _savetree_bounded(metadata, tmp, "metadata")
        # Provenance and tracking metadata
        retained = [anchor_iter, current_iter] if current_iter != anchor_iter else [anchor_iter]
        tmp.attrs["retained_iterations"] = retained
        tmp.attrs["source_data_cache"] = str(Path(project_dir) / "preprocessed_data.pkl")
        tmp.flush()

        # Flush dirty writeback to disk and drop clean page cache
        try:
            fd = tmp.id.get_vfd_handle()
            os.fdatasync(fd)
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass

    # Atomic commit
    os.replace(tmp_path, checkpoint_path)
    try:
        fd_ckpt = os.open(str(checkpoint_path), os.O_RDONLY)
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            os.posix_fadvise(fd_ckpt, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd_ckpt)
    except Exception:
        pass

    ckpt_size_gb = checkpoint_path.stat().st_size / 1e9
    t_save = time.time() - t0
    rss_end = psutil.Process().memory_info().rss / 1e9
    print(f"  [checkpoint] Checkpoint committed at iter {current_iter} | Retained snapshots: {retained} | Disk size: {ckpt_size_gb:.2f} GB | RSS: {rss_end:.2f} GB | Time: {t_save:.2f}s.", flush=True)

    if generate_progress_plots:
        try:
            plot_progress(
                model,
                data,
                str(checkpoint_path),
                current_iter,
                project_dir=str(project_dir),
                model_name=model_name,
                savefig=True,
            )
            print(f"  [progress plot] Updated fitting_progress.pdf at iter {current_iter}.", flush=True)
        except Exception as e:
            print(f"  [progress plot warning] Could not update progress plot ({e})", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full Cohort Tandem KPMS Calibrated Production Training")
    parser.add_argument("--init-kappa", type=float, default=INIT_KAPPA, help="Initial Kappa for AR phase")
    parser.add_argument("--decrease-kappa-factor", type=float, default=DECREASE_KAPPA_FACTOR, help="Reduction factor for SLDS phase")
    parser.add_argument("--num-ar-iters", type=int, default=NUM_AR_ITERS, help="Number of AR iters")
    parser.add_argument("--num-slds-iters", type=int, default=NUM_SLDS_ITERS, help="Number of SLDS iters")
    parser.add_argument("--save-every-n-iters", type=int, default=SAVE_EVERY_N_ITERS, help="Checkpoint save interval")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--seed", type=int, default=101, help="Random seed for model initialization")
    parser.add_argument("--output-pkl", type=str, default=str(OUTPUT_PROD_PKL), help="Target output pickle path")
    parser.add_argument("--fit-only", action="store_true", help="Save fitted checkpoint and defer export")
    args = parser.parse_args()

    init_kappa = args.init_kappa
    decrease_kappa_factor = args.decrease_kappa_factor
    slds_kappa = init_kappa / decrease_kappa_factor
    num_ar_iters = args.num_ar_iters
    num_slds_iters = args.num_slds_iters
    total_iters = num_ar_iters + num_slds_iters
    save_every_n_iters = args.save_every_n_iters
    model_name = args.model_name
    seed = args.seed
    output_prod_pkl = Path(args.output_pkl)

    t_program_start = time.time()
    pid = os.getpid()

    # Discover matched pairs
    pairs = find_matched_pairs()
    pair_map = {p["sid"]: p for p in pairs}
    print("=" * 80, flush=True)
    print(f"🚀 CORRECTED TANDEM KPMS COHORT RUN (SEED {seed}): {len(pairs)} SESSIONS", flush=True)
    print(f"  Plan:                {num_ar_iters} AR + {num_slds_iters} SLDS = {total_iters} TOTAL ITERS", flush=True)
    print(f"  Date & Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Process PID:         {pid}", flush=True)
    print(f"  Model Name:          {model_name}", flush=True)
    print(f"  Random Seed:         {seed}", flush=True)
    print(f"  Hardware:            RTX 4090 (24.5 GB VRAM) with Amdahl 27x Optimizations", flush=True)
    print(f"  Init Kappa:          {init_kappa:.2e} -> SLDS Kappa: {slds_kappa:.2e}", flush=True)
    print(f"  Save Checkpoints:    Every {save_every_n_iters} iterations", flush=True)
    print(f"  Gap Threshold:       {MIN_GAP_FRAMES} frames ({MIN_GAP_FRAMES/TARGET_FPS:.1f}s)", flush=True)
    print(f"  Project Directory:   {PROD_KPMS_DIR}", flush=True)
    print(f"  Target Pickle:       {output_prod_pkl}", flush=True)
    print("=" * 80, flush=True)

        # 0. Apply checkpoint compaction monkeypatch
    print("\n[Step 0/6] Activating transparent KPMS checkpoint compaction monkeypatch...", flush=True)
    apply_kpms_checkpoint_patch()

    # 1. Apply high-performance kernels
    print("\n[Step 1/6] Applying Amdahl 27x Optimizations (Banded Cholesky + GPU Kernels)...", flush=True)
    rb.apply_optimizations()

    # Send launch notification
    send_ntfy(
        f"KPMS Calibrated Cohort Run Started ({model_name}) 🚀",
        f"PID: {pid}\n"
        f"Plan: {num_ar_iters} AR + {num_slds_iters} SLDS = {total_iters} iters\n"
        f"Sessions: {len(pairs)} tandem sessions with dual-anchor gating & continuous sync\n"
        f"Kappa: {init_kappa:.2e} -> {slds_kappa:.2e}\n"
        f"Checkpoints every {save_every_n_iters} iters",
        priority="default",
        tags="rocket,brain"
    )

    # 2. Ingest Base Sessions & Enrich
    print("\n[Step 2/6] Loading base SessionRecord objects from dirty pickle...", flush=True)
    with open(DIRTY_PKL_IN, "rb") as f:
        dirty_sessions = pickle.load(f)

    selected_sessions = [s for s in dirty_sessions if s.session_id in pair_map]
    del dirty_sessions
    gc.collect()
    print(f"  Loaded {len(selected_sessions)} matching SessionRecord objects.")

    for s in selected_sessions:
        sid = s.session_id
        cfg = pair_map[sid]
        fused_path = FUSED_DIR / f"{sid}_tandem_23kp.h5"

        t_df = s.tracking_refs_df.copy()
        if not (t_df["role"] == "ventral").any():
            ventral_track_row = {
                "path": str(cfg["bot_h5"]),
                "path_type": "tracking",
                "kind": "h5",
                "role": "ventral",
                "camset": "BotCamVid",
                "cam_slot": "bottom_cam",
                "arena": cfg["arena"],
                "chamber": s.metadata.get("chamber"),
                "ts": s.metadata.get("timestamp"),
                "camera_key": f"bottom_cam:{s.metadata.get('chamber')}",
                "chamber_index": s.metadata.get("chamber_index", 0)
            }
            t_df = pd.concat([t_df, pd.DataFrame([ventral_track_row])], ignore_index=True)

        if not (t_df["role"] == "fused_23kp").any():
            fused_track_row = {
                "path": str(fused_path),
                "path_type": "tracking",
                "kind": "h5",
                "role": "fused_23kp",
                "camset": "TandemMultiview",
                "cam_slot": "multiview_fused",
                "arena": cfg["arena"],
                "chamber": s.metadata.get("chamber"),
                "ts": s.metadata.get("timestamp"),
                "camera_key": f"multiview_fused:{s.metadata.get('chamber')}",
                "chamber_index": s.metadata.get("chamber_index", 0)
            }
            t_df = pd.concat([t_df, pd.DataFrame([fused_track_row])], ignore_index=True)
        s.tracking_refs_df = t_df

    # 3. Setup KPMS Project & Data Directory
    print("\n[Step 3/6] Setting up KPMS calibrated project environment...", flush=True)
    PROD_KPMS_DIR.mkdir(parents=True, exist_ok=True)
    PROD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    config_source = EXPERIMENT_DIR / "kpms_tandem_cohort_production/config.yml"
    config_dest = PROD_KPMS_DIR / "config.yml"
    if not config_dest.exists():
        shutil.copy(config_source, config_dest)

    config = kpms.load_config(str(PROD_KPMS_DIR))
    config["fps"] = TARGET_FPS
    config["seg_length"] = 10030
    config["trans_hypparams"]["kappa"] = init_kappa
    kpms.update_config(str(PROD_KPMS_DIR), **config)

    # Multi-tier disk caching: Load or compute preprocessed tracking data
    preprocessed_cache = PROD_KPMS_DIR / "preprocessed_data.pkl"
    vfi_path = PROD_KPMS_DIR / "video_frame_indexes.npz"

    if preprocessed_cache.exists() and vfi_path.exists():
        print(f"  [CACHE HIT] Loading preprocessed cohort tracking data from {preprocessed_cache}...", flush=True)
        with open(preprocessed_cache, "rb") as f:
            data, metadata = pickle.load(f)
        with np.load(vfi_path) as z:
            video_frame_indexes = {k: z[k] for k in z.files}
        print(f"  Loaded VFI mappings for {len(video_frame_indexes)} segments.")
        data["conf"] = np.asarray(data["conf"], dtype=np.float64)
        data["mask"] = np.asarray(data["mask"], dtype=np.float64)
    else:
        print("  [CACHE MISS] Ingesting fused H5 files with prolonged missing-ventral splitting...", flush=True)
        fused_files = sorted(FUSED_DIR.glob("*_tandem_23kp.h5"))
        print(f"  Found {len(fused_files)} fused H5 files in {FUSED_DIR}.")

        split_coords = {}
        split_confs = {}
        video_frame_indexes = {}
        total_raw_frames = 0
        total_retained_frames = 0

        for f_idx, f_path in enumerate(fused_files):
            sid = f_path.name.replace("_tandem_23kp.h5", "")
            mapping = get_session_mapping(sid)
            if mapping is None or len(mapping) == 0:
                raise ValueError(f"CRITICAL: Session {sid} has no continuous mapping!")

            df = pd.read_hdf(f_path)
            N = len(df)
            total_raw_frames += N
            scorer = df.columns.get_level_values(0)[0]
            bps = config["bodyparts"]
            missing_bps = [bp for bp in bps if (scorer, bp, "x") not in df.columns]
            if missing_bps:
                raise ValueError(f"CRITICAL: Session {sid} missing bodyparts from config: {missing_bps}")

            segments, is_valid = split_session_into_dual_view_segments(df)
            retained_here = sum(e - s for s, e in segments)
            total_retained_frames += retained_here

            # Extract coordinates and confidences
            coords_arr = np.concatenate([
                np.column_stack([df[(scorer, bp, "x")].to_numpy() for bp in bps])[:, :, None],
                np.column_stack([df[(scorer, bp, "y")].to_numpy() for bp in bps])[:, :, None]
            ], axis=-1).astype(np.float32)

            confs_arr = np.column_stack([df[(scorer, bp, "likelihood")].to_numpy() for bp in bps]).astype(np.float32)

            for seg_i, (s_idx, e_idx) in enumerate(segments):
                seg_name = f"{sid}_seg{seg_i:03d}"
                split_coords[seg_name] = coords_arr[s_idx:e_idx].copy()
                split_confs[seg_name] = confs_arr[s_idx:e_idx].copy()
                video_frame_indexes[seg_name] = np.arange(s_idx, e_idx, dtype=np.int64)
            del coords_arr, confs_arr, df

            if (f_idx + 1) % 25 == 0 or (f_idx + 1) == len(fused_files):
                print(f"  Processed {f_idx + 1:3d}/{len(fused_files)} sessions | Segments: {len(split_coords):5d} | Retained: {total_retained_frames:,}/{total_raw_frames:,} ({total_retained_frames/total_raw_frames*100:.1f}%)", flush=True)

        # Save VFI
        print(f"  Saving video_frame_indexes.npz ({len(video_frame_indexes)} segments)...", flush=True)
        np.savez_compressed(vfi_path, **video_frame_indexes)

        auto_mask_sym = PROD_KPMS_DIR / "auto_mask.npz"
        if auto_mask_sym.exists() or auto_mask_sym.is_symlink():
            auto_mask_sym.unlink()
        auto_mask_sym.symlink_to(vfi_path)

        # Format Data strictly in Host DRAM (bypasses GPU VRAM device_put OOM)
        print(f"  Formatting data with Keypoint-MoSeq across {len(split_coords)} segments in Host DRAM...", flush=True)
        cpu_device = jax.devices("cpu")[0]
        with jax.default_device(cpu_device):
            data, metadata = kpms.format_data(split_coords, split_confs, **config)

        # Free ingestion buffers immediately to reduce peak RAM before serialization
        del split_coords, split_confs
        gc.collect()
        rss_post_format = psutil.Process().memory_info().rss / 1e9
        print(f"  [memory] Post-format_data RSS: {rss_post_format:.2f} GB (freed ingestion buffers)", flush=True)

        # Convert JAX CPU arrays directly to numpy without redundant allocations
        data = {
            "Y": np.asarray(data["Y"], dtype=np.float64),
            "conf": np.asarray(data["conf"], dtype=np.float32),
            "mask": np.asarray(data["mask"], dtype=np.float32),
        }
        gc.collect()

        # Serialize to disk with streaming pickle to avoid doubling RAM
        print(f"  Caching preprocessed data to {preprocessed_cache}...", flush=True)
        rss_pre_pkl = psutil.Process().memory_info().rss / 1e9
        print(f"  [memory] Pre-pickle RSS: {rss_pre_pkl:.2f} GB", flush=True)
        with open(preprocessed_cache, "wb") as f:
            pickler = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
            pickler.dump((data, metadata))
        rss_post_pkl = psutil.Process().memory_info().rss / 1e9
        print(f"  [memory] Post-pickle RSS: {rss_post_pkl:.2f} GB", flush=True)

    total_frames = int(data["mask"].sum())
    n_segs, seg_len, n_kps, d = data["Y"].shape
    print(f"  Active animal frames: {total_frames:,}")
    print(f"  Batched Y tensor:     {n_segs} sequences x {seg_len} frames x {n_kps} keypoints x {d}D ({data['Y'].nbytes / 1e9:.2f} GB in Host RAM)")

    # Load or fit PCA
    pca_p = PROD_KPMS_DIR / "pca.p"
    if pca_p.exists():
        pca = kpms.load_pca(str(PROD_KPMS_DIR))
        print("  Loaded existing PCA from project.", flush=True)
    else:
        print("  Fitting PCA on representative cohort subset (D=10)...", flush=True)
        max_pca_seqs = min(50, len(data["Y"]))
        sample_indices = np.linspace(0, len(data["Y"]) - 1, max_pca_seqs, dtype=int)
        data_pca_subset = {
            k: v[sample_indices] if (hasattr(v, "__len__") and len(v) == len(data["Y"])) else v
            for k, v in data.items()
        }
        pca = kpms.fit_pca(**data_pca_subset, **config)
        kpms.save_pca(pca, str(PROD_KPMS_DIR))
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"  Top 10 PCs Cumulative Variance Explained: {cum_var[-1]*100:.2f}%")

    # Preemptive Analytical Batch Sizing
    print("  Executing Preemptive Analytical Tensor Sizing...", flush=True)
    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(data, verbose=True)

    # 4. Checkpoint Resumption or Clean Initialization
    checkpoint_file = PROD_KPMS_DIR / model_name / "checkpoint.h5"
    current_iter = 0
    model = None

    if checkpoint_file.exists():
        try:
            print(f"\n  [RESUME] Found existing checkpoint at {checkpoint_file}. Resuming...", flush=True)
            with h5py.File(checkpoint_file, "r") as saved:
                current_iter = max(int(k) for k in saved["model_snapshots"])
            saved_metadata = kpms.io.load_hdf5(str(checkpoint_file), "metadata")
            if len(metadata) != len(saved_metadata) or not all(np.array_equal(a, b) for a, b in zip(metadata, saved_metadata)):
                raise ValueError("Checkpoint metadata differs from preprocessing cache")
            model = kpms.io.load_hdf5(str(checkpoint_file), f"model_snapshots/{current_iter}")
            if data is not None and "Y" in data and isinstance(metadata, (tuple, list)):
                assert len(metadata[0]) == data["Y"].shape[0], (
                    f"Metadata keys ({len(metadata[0])}) mismatch data sequences ({data['Y'].shape[0]})"
                )
            rb.KPMSStateCache.clear()
            rb.precompute_optimal_batch_config(data, model=model, verbose=True)
            print(f"  [RESUME] Resumed successfully from iteration {current_iter}.", flush=True)
            send_ntfy(
                "KPMS Resumed from Checkpoint",
                f"Resumed training '{model_name}' at iteration {current_iter}/{total_iters}.",
                priority="default",
                tags="play,brain"
            )
        except Exception as e:
            raise RuntimeError(f"Checkpoint resume failed: {checkpoint_file}") from e

    if model is None:
        init_model_cache = PROD_KPMS_DIR / f"init_model_seed_{seed}.pkl"
        gpu_device = jax.devices()[0]
        import jax.numpy as jnp

        if init_model_cache.exists():
            print(f"\n  [CACHE HIT] Loading pre-initialized model state from {init_model_cache} (Seed {seed})...", flush=True)
            t0_cache = time.time()
            with open(init_model_cache, "rb") as f:
                model = pickle.load(f)
            for k in list(model["states"].keys()):
                v = model["states"][k]
                if k == "s":
                    model["states"][k] = np.asarray(v)
                else:
                    model["states"][k] = jax.device_put(jnp.asarray(v), gpu_device)
            model = kpms.update_hypparams(model, kappa=float(init_kappa))
            gc.collect()
            t_cache = time.time() - t0_cache
            print(f"  [CACHE HIT] Model loaded and states placed on GPU in {t_cache:.2f}s (No duplicate deepcopy).", flush=True)
        else:
            print(f"\n  [INIT] Initializing fresh AR-HMM model with batched init (Seed {seed})...", flush=True)
            t0_init = time.time()
            init_bs = min(rb.CONFIG.discrete_bs, 20)
            rng_key = jax.random.PRNGKey(seed)
            model = batched_init_model_by_video(
                data, pca, config,
                params_subset_size=5, batch_size=init_bs, seed=rng_key
            )
            model = kpms.update_hypparams(model, kappa=float(init_kappa))
            t_init = time.time() - t0_init
            print(f"  [INIT] Initialization complete in {t_init:.2f}s ({t_init/60:.2f} min).", flush=True)

            print(f"  [CACHE SAVE] Caching initialized model to {init_model_cache}...", flush=True)
            cpu_states = {k: np.asarray(v) for k, v in model["states"].items()}
            cache_dict = {
                "params": {k: np.asarray(v) if hasattr(v, "shape") else v for k, v in model["params"].items()},
                "hypparams": model["hypparams"],
                "states": cpu_states,
                "noise_prior": np.asarray(model["noise_prior"]) if hasattr(model.get("noise_prior"), "shape") else model.get("noise_prior", 1.0),
                "seed": model["seed"]
            }
            with open(init_model_cache, "wb") as f:
                pickle.dump(cache_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [CACHE SAVED] Initial model cached ({init_model_cache.stat().st_size / 1e6:.1f} MB).", flush=True)
            del cpu_states, cache_dict
            gc.collect()

        current_iter = 0

    # 5. Model Fitting Loop
    print(f"\n[Step 4/6] Executing KPMS Training Loop ({num_ar_iters} AR + {num_slds_iters} SLDS)...", flush=True)

    # PHASE 1: AR-ONLY FITTING (0 -> 50)
    if current_iter < num_ar_iters:
        print("\n" + "=" * 70, flush=True)
        print(f"  PHASE 1: AR-Only Fitting (iter {current_iter} -> {num_ar_iters})", flush=True)
        print(f"  Kappa: {init_kappa:.2e}", flush=True)
        print("=" * 70, flush=True)

        ar_start_wall = time.time()
        chunk_size = save_every_n_iters

        while current_iter < num_ar_iters:
            next_target = min(current_iter + chunk_size, num_ar_iters)
            print(f"\n>>> Running AR Chunk: iter {current_iter} -> {next_target}...", flush=True)

            model, _ = kpms.fit_model(
                model, data, metadata,
                project_dir=str(PROD_KPMS_DIR),
                model_name=model_name,
                ar_only=True,
                start_iter=current_iter,
                num_iters=next_target,
                parallel_message_passing=False,
                save_every_n_iters=None,
                generate_progress_plots=False,
                verbose=True
            )
            current_iter = next_target

            # Save rolling checkpoint (retains anchor iter 50 + latest)
            save_rolling_checkpoint(
                PROD_KPMS_DIR,
                model_name,
                model,
                data,
                metadata,
                current_iter,
                anchor_iter=num_ar_iters,
                generate_progress_plots=True
            )

            z_curr = np.array(model["states"]["z"])
            mask_curr = np.array(data["mask"])[:, 3:]
            dur_list = kpms.viz.get_durations(z_curr, mask_curr)
            med_frames = float(np.median(dur_list)) if len(dur_list) > 0 else 0.0
            med_ms = (med_frames / TARGET_FPS) * 1000.0
            mean_frames = float(np.mean(dur_list)) if len(dur_list) > 0 else 0.0
            n_active_sylls = len(np.unique(z_curr[mask_curr > 0]))

            sec_per_it = (time.time() - ar_start_wall) / current_iter
            ar_rem_sec = (num_ar_iters - current_iter) * sec_per_it
            eta_str = str(timedelta(seconds=int(ar_rem_sec)))

            print(f"--- Chunk Complete: AR {current_iter}/{num_ar_iters} ({sec_per_it:.2f}s/it, AR ETA: {eta_str}) | Median: {med_frames:.1f}f ({med_ms:.1f}ms) | Mean: {mean_frames:.1f}f | Sylls: {n_active_sylls} ---", flush=True)

            send_ntfy(
                f"KPMS AR Progress: {current_iter}/{num_ar_iters}",
                f"AR {current_iter}/{num_ar_iters} complete ({(current_iter/num_ar_iters)*100:.0f}%)\n"
                f"Median Syllable: {med_frames:.1f} frames ({med_ms:.1f} ms) | Active: {n_active_sylls}\n"
                f"Speed: {sec_per_it:.2f}s/it | AR ETA: {eta_str}",
                priority="default",
                tags="chart_with_upwards_trend"
            )

        ar_total_min = (time.time() - ar_start_wall) / 60.0
        print(f"\n[PHASE 1 COMPLETE] {num_ar_iters} AR iterations finished in {ar_total_min:.1f} min.", flush=True)
        send_ntfy(
            f"KPMS Phase 1 ({num_ar_iters} AR) Complete 🎉",
            f"{num_ar_iters} AR iterations finished in {ar_total_min:.1f} min.\n"
            f"Transitioning to Phase 2 SLDS (10x kappa reduction: {slds_kappa:.1e}).",
            priority="default",
            tags="tada,repeat"
        )
    else:
        print(f"  [Phase 1] Skipping AR (Already reached iter {current_iter} >= {num_ar_iters}).", flush=True)

    # TRANSITION: 10x Kappa reduction for SLDS
    print("\n" + "=" * 70, flush=True)
    print(f"  TRANSITION: Applying 10x Kappa Reduction for Phase 2 SLDS", flush=True)
    model = kpms.update_hypparams(model, kappa=float(slds_kappa))
    print(f"  New Kappa: {slds_kappa:.2e}", flush=True)
    print("=" * 70, flush=True)

    # PHASE 2: SLDS FITTING
    if current_iter < total_iters:
        print("\n" + "=" * 70, flush=True)
        print(f"  PHASE 2: Full SLDS Fitting (iter {current_iter} -> {total_iters})", flush=True)
        print(f"  SLDS Kappa: {slds_kappa:.2e}", flush=True)
        print("=" * 70, flush=True)

        slds_start_wall = time.time()
        slds_start_iter = current_iter
        chunk_size = save_every_n_iters

        while current_iter < total_iters:
            next_target = min(current_iter + chunk_size, total_iters)
            print(f"\n>>> Running SLDS Chunk: iter {current_iter} -> {next_target}...", flush=True)

            model, _ = kpms.fit_model(
                model, data, metadata,
                project_dir=str(PROD_KPMS_DIR),
                model_name=model_name,
                ar_only=False,
                start_iter=current_iter,
                num_iters=next_target,
                parallel_message_passing=False,
                save_every_n_iters=None,
                generate_progress_plots=False,
                verbose=True
            )
            current_iter = next_target

            # Save rolling checkpoint (retains anchor iter 50 + latest)
            save_rolling_checkpoint(
                PROD_KPMS_DIR,
                model_name,
                model,
                data,
                metadata,
                current_iter,
                anchor_iter=num_ar_iters,
                generate_progress_plots=True
            )

            z_curr = np.array(model["states"]["z"])
            mask_curr = np.array(data["mask"])[:, 3:]
            dur_list = kpms.viz.get_durations(z_curr, mask_curr)
            med_frames = float(np.median(dur_list)) if len(dur_list) > 0 else 0.0
            med_ms = (med_frames / TARGET_FPS) * 1000.0
            mean_frames = float(np.mean(dur_list)) if len(dur_list) > 0 else 0.0
            n_active_sylls = len(np.unique(z_curr[mask_curr > 0]))

            slds_done = current_iter - num_ar_iters
            sec_per_it = (time.time() - slds_start_wall) / (current_iter - slds_start_iter)
            slds_rem_sec = (total_iters - current_iter) * sec_per_it
            eta_str = str(timedelta(seconds=int(slds_rem_sec)))
            est_time = (datetime.now() + timedelta(seconds=slds_rem_sec)).strftime('%H:%M:%S')

            print(f"--- Chunk Complete: SLDS {slds_done}/{num_slds_iters} (Total {current_iter}/{total_iters}) | {sec_per_it:.2f}s/it | Median: {med_frames:.1f}f ({med_ms:.1f}ms) | Mean: {mean_frames:.1f}f | Sylls: {n_active_sylls} | ETA: {eta_str} (Est: {est_time}) ---", flush=True)

            send_ntfy(
                f"KPMS SLDS Progress: {slds_done}/{num_slds_iters}",
                f"SLDS {slds_done}/{num_slds_iters} complete (Total {current_iter}/{total_iters}, {(current_iter/total_iters)*100:.0f}%)\n"
                f"Median Syllable: {med_frames:.1f} frames ({med_ms:.1f} ms) | Active: {n_active_sylls}\n"
                f"Speed: {sec_per_it:.2f}s/it | ETA: {eta_str} (Est: {est_time})",
                priority="default",
                tags="hourglass_flowing_sand,brain"
            )

        slds_total_min = (time.time() - slds_start_wall) / 60.0
        print(f"\n[PHASE 2 COMPLETE] {num_slds_iters} SLDS iterations finished in {slds_total_min:.1f} min.", flush=True)
    else:
        print(f"  [Phase 2] Training already complete ({current_iter} >= {total_iters}).", flush=True)

    if args.fit_only:
        print(f"[FIT COMPLETE] Iteration {current_iter}; checkpoint: {checkpoint_file}", flush=True)
        return

    # 6. Extract Results and Weave Kinematics & Photometry
    print("\n[Step 5/6] Extracting final model results and weaving tandem kinematics...", flush=True)
    # Directly reuse in-memory model and metadata; do not reload checkpoint or dataset
    results = kpms.extract_results(model, metadata, str(PROD_KPMS_DIR), model_name, save_results=True)

    # Post-training Checkpoint Compaction
    checkpoint_file = PROD_KPMS_DIR / model_name / "checkpoint.h5"
    print(f"\n[Compaction] Executing post-training checkpoint compaction on {checkpoint_file}...", flush=True)
    try:
        compact_script = Path(__file__).resolve().parents[3] / "diagnostics/oom_harness/compact_standalone.py"
        subprocess.run([
            sys.executable, str(compact_script),
            str(checkpoint_file),
            "--keep-iters", str(num_ar_iters), str(total_iters),
            "--chunk-mb", "64"
        ], check=True)
    except Exception as e:
        print(f"  [Compaction Warning] Post-hoc compaction encountered error ({e}), uncompacted rolling checkpoint retained.", flush=True)

    # Load video_frame_indexes
    vfi_path = PROD_KPMS_DIR / "video_frame_indexes.npz"
    with np.load(vfi_path) as z:
        vfi_dict = {k: z[k].astype(int) for k in z.files}
    print(f"  Loaded VFI mappings for {len(vfi_dict)} segments.")

    k240k_map = {}
    if K240K_PKL_REF.exists():
        try:
            with open(K240K_PKL_REF, "rb") as f:
                k240k_all = pickle.load(f)
            for s_ref in k240k_all:
                if s_ref.session_id in pair_map:
                    k240k_map[s_ref.session_id] = s_ref
        except Exception as e:
            print(f"  [warning] Could not load K240K reference: {e}", flush=True)

    modeled_sessions = []
    for s in selected_sessions:
        sid = s.session_id
        cfg = pair_map[sid]
        
        seg_keys = sorted([k for k in results.keys() if k.startswith(sid)])
        if not seg_keys:
            print(f"  [skip] No segments for {sid} in results.", flush=True)
            continue

        df_top = pd.read_hdf(cfg["top_h5"])
        df_bot = pd.read_hdf(cfg["bot_h5"])
        top_scorer = df_top.columns.levels[0][0]
        bot_scorer = df_bot.columns.levels[0][0]

        N_top = len(df_top)
        timeline_index = pd.RangeIndex(0, N_top)

        syll_series = pd.Series(pd.NA, index=timeline_index, dtype="Int64")
        for seg_k in seg_keys:
            vfi = vfi_dict[seg_k]
            sylls = results[seg_k]["syllable"]
            syll_series.iloc[vfi] = pd.Series(sylls, index=syll_series.iloc[vfi].index, dtype="Int64")

        s.syllables = pd.DataFrame({"syllable": syll_series}, index=timeline_index)
        s.syllables["syllable"] = s.syllables["syllable"].astype("Int64")

        top_nx = df_top[top_scorer, "nose", "x"].to_numpy(dtype=float)
        top_ny = df_top[top_scorer, "nose", "y"].to_numpy(dtype=float)
        top_tx = df_top[top_scorer, "tail_base", "x"].to_numpy(dtype=float)
        top_ty = df_top[top_scorer, "tail_base", "y"].to_numpy(dtype=float)

        cx_top = 0.5 * (top_nx + top_tx)
        cy_top = 0.5 * (top_ny + top_ty)
        heading_top = np.arctan2(top_ny - top_ty, top_nx - top_tx)

        dx_top = np.diff(cx_top)
        dy_top = np.diff(cy_top)
        vel_top_raw = np.pad(np.sqrt(dx_top**2 + dy_top**2) * 60.0, (0, 1), mode="edge")
        ang_vel_top = np.pad(np.diff(np.unwrap(heading_top)) * 60.0, (0, 1), mode="edge")

        s.kinematics_df = pd.DataFrame({
            "centroid_x": cx_top,
            "centroid_y": cy_top,
            "heading": heading_top,
            "velocity": vel_top_raw,
            "angular_velocity": ang_vel_top
        }, index=timeline_index)

        if sid in k240k_map and hasattr(k240k_map[sid], "photometry_df") and k240k_map[sid].photometry_df is not None:
            s.photometry_df = copy.deepcopy(k240k_map[sid].photometry_df)
        elif hasattr(s, "photometry_df") and s.photometry_df is not None:
            pass

        try:
            combined_df = combined_photometry_and_syllables(s)
            s.combined_df = combined_df
        except Exception:
            pass

        modeled_sessions.append(s)

    # 7. Serialize final SessionRecords
    print(f"\n[Step 6/6] Serializing final calibrated SessionRecord objects to {output_prod_pkl}...", flush=True)
    with open(output_prod_pkl, "wb") as f:
        pickle.dump(modeled_sessions, f, protocol=pickle.HIGHEST_PROTOCOL)

    t_program_end = time.time()
    total_elapsed_min = (t_program_end - t_program_start) / 60.0
    print("\n" + "=" * 80)
    print(f"🎉 CORRECTED TANDEM COHORT RETRAINING COMPLETE (SEED {seed}) in {total_elapsed_min:.1f} min!")
    print(f"   Modeled Sessions: {len(modeled_sessions)} / {len(pairs)}")
    print(f"   Checkpoint:       {checkpoint_file}")
    print(f"   Output Pickle:    {output_prod_pkl}")
    print("=" * 80)

    send_ntfy(
        f"KPMS Calibrated Cohort Run COMPLETE ({model_name}) 🏆",
        f"Finished 250 iterations in {total_elapsed_min:.1f} min.\n"
        f"Saved to {output_prod_pkl.name}",
        priority="high",
        tags="tada,trophy"
    )

if __name__ == "__main__":
    main()
