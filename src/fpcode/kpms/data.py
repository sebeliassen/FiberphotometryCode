"""
fpcode.kpms.data
================
Data preprocessing, presence mask generation, parallelized outlier filtering,
sequence subsetting, and memory-safe PCA fitting for Keypoint-MoSeq models.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import keypoint_moseq as kpms
import numpy as np
from tqdm import tqdm

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


def _clean_short_runs(mask: np.ndarray, min_true: int = 1, min_false: int = 1) -> np.ndarray:
    """Remove ON runs shorter than min_true, and OFF runs shorter than min_false."""
    m = mask.copy()
    if m.size == 0:
        return m

    x = m.astype(np.int8)
    boundaries = np.flatnonzero(np.r_[True, x[1:] != x[:-1], True])
    starts = boundaries[:-1]
    ends = boundaries[1:]
    vals = m[starts]

    if min_true > 1:
        for s, e, v in zip(starts, ends, vals):
            if v and (e - s) < min_true:
                m[s:e] = False

    x = m.astype(np.int8)
    boundaries = np.flatnonzero(np.r_[True, x[1:] != x[:-1], True])
    starts = boundaries[:-1]
    ends = boundaries[1:]
    vals = m[starts]

    if min_false > 1:
        for s, e, v in zip(starts, ends, vals):
            if (not v) and (e - s) < min_false:
                m[s:e] = True

    return m


def presence_mask_from_conf(
    conf: np.ndarray,
    p_thresh: float = 0.9,
    frac_min_parts: float = 0.4,
    smooth_win: int = 7,
    enter_thresh: float = 0.65,
    exit_thresh: float = 0.45,
    off_grace_frames: int = 6,
    min_on_frames: int = 10,
    min_off_frames: int = 6,
) -> np.ndarray:
    """
    Generate boolean presence mask from bodypart confidence array (T, N).
    Uses hysteresis and smoothing to avoid rapid flapping on brief occlusions.
    """
    T, N = conf.shape

    valid = conf >= p_thresh
    count = valid.sum(axis=1)
    frac = count / float(N)

    if smooth_win > 1:
        k = np.ones(smooth_win, float) / smooth_win
        sm = np.convolve(frac, k, mode="same")
    else:
        sm = frac

    k_min = max(1, int(np.ceil(frac_min_parts * N)))
    seed = count >= k_min

    on = np.zeros(T, dtype=bool)
    inside = False
    below = 0
    for t in range(T):
        if not inside:
            if (sm[t] >= enter_thresh) or seed[t]:
                inside = True
                below = 0
        else:
            if (sm[t] <= exit_thresh) and (not seed[t]):
                below += 1
                if below >= off_grace_frames:
                    inside = False
                    below = 0
            else:
                below = 0
        on[t] = inside

    if min_on_frames > 1 or min_off_frames > 1:
        on = _clean_short_runs(on, min_true=min_on_frames, min_false=min_off_frames)

    return on


def _init_worker() -> None:
    import logging
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    logging.getLogger("jax").setLevel(logging.CRITICAL)


def _process_outliers_single(args: tuple) -> tuple:
    recording_name, raw_coords, raw_confs, project_dir, outlier_scale_factor, bodyparts, overwrite, kwargs = args
    import matplotlib
    matplotlib.use("Agg")
    from keypoint_moseq.util import (
        find_medoid_distance_outliers,
        interpolate_keypoints,
        plot_medoid_distance_outliers,
    )

    raw_coords = np.asarray(raw_coords).copy()
    raw_confs = np.asarray(raw_confs)
    outliers = find_medoid_distance_outliers(raw_coords, outlier_scale_factor=outlier_scale_factor)
    clean_coords = interpolate_keypoints(raw_coords, outliers["mask"])
    clean_confs = np.where(outliers["mask"], 0, raw_confs)

    plot_kwargs = kwargs.copy()
    for key in [
        "project_dir", "recording_name", "original_coordinates", "interpolated_coordinates",
        "outlier_mask", "outlier_thresholds", "bodyparts", "overwrite",
    ]:
        plot_kwargs.pop(key, None)

    plot_medoid_distance_outliers(
        project_dir,
        recording_name,
        raw_coords,
        clean_coords,
        outliers["mask"],
        outliers["thresholds"],
        bodyparts=bodyparts,
        overwrite=overwrite,
        **plot_kwargs,
    )
    return recording_name, clean_coords, clean_confs


def filter_with_masks(
    coordinates: Dict[str, np.ndarray],
    confidences: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    project_dir: str | Path,
    return_video_frame_indexes: bool = False,
    downsample_rate: int = 1,
    **kwargs,
) -> Tuple[Any, ...]:
    """
    Compress by masks (keep only mask==1 frames), run parallelized outlier removal on compressed dicts.
    """
    names = sorted(set(coordinates) & set(confidences) & set(masks))
    if not names:
        raise ValueError("No overlapping keys among coordinates/confidences/masks.")

    if downsample_rate > 1:
        coordinates = {k: v[::downsample_rate] for k, v in coordinates.items()}
        confidences = {k: v[::downsample_rate] for k, v in confidences.items()}
        masks = {k: v[::downsample_rate] for k, v in masks.items()}

    for name in names:
        T = coordinates[name].shape[0]
        if confidences[name].shape[0] != T:
            raise ValueError(f"{name}: T mismatch conf vs coord")
        m = np.asarray(masks[name], bool)
        if m.shape != (T,):
            raise ValueError(f"{name}: mask must be shape ({T},), got {m.shape}")

    keep = {k: np.flatnonzero(np.asarray(masks[k], bool)) for k in tqdm(names, desc="Compressing masks")}
    coords_cmp = {k: np.asarray(coordinates[k][keep[k]]) if keep[k].size else coordinates[k][:0] for k in names}
    confs_cmp = {k: np.asarray(confidences[k][keep[k]]) if keep[k].size else confidences[k][:0] for k in names}

    nonempty = [k for k in names if keep[k].size]
    coords_out, confs_out = {}, {}
    if nonempty:
        outlier_scale_factor = kwargs.get("outlier_scale_factor", 6.0)
        bodyparts = kwargs.get("bodyparts", None)
        overwrite = kwargs.get("overwrite", False)

        tasks = [
            (
                k,
                np.asarray(coords_cmp[k]),
                np.asarray(confs_cmp[k]),
                str(project_dir),
                outlier_scale_factor,
                bodyparts,
                overwrite,
                kwargs,
            )
            for k in nonempty
        ]

        if len(nonempty) <= 4:
            results = [_process_outliers_single(t) for t in tqdm(tasks, desc="Outlier removal")]
        else:
            mp_context = multiprocessing.get_context("spawn")
            max_workers = min(16, os.cpu_count() or 4)
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context, initializer=_init_worker) as executor:
                results = list(tqdm(
                    executor.map(_process_outliers_single, tasks),
                    total=len(tasks),
                    desc="Outlier removal",
                ))

        for k, clean_coords, clean_confs in results:
            coords_out[k] = clean_coords
            confs_out[k] = clean_confs

    filtered_coordinates = {
        k: (coords_out[k] if k in coords_out else coordinates[k][:0].astype(float, copy=False))
        for k in names
    }
    filtered_confidences = {
        k: (confs_out[k] if k in confs_out else confidences[k][:0].astype(float, copy=False))
        for k in names
    }

    if return_video_frame_indexes:
        video_frame_indexes = {
            k: (keep[k] * downsample_rate).astype(int)
            for k in names
        }
        return filtered_coordinates, filtered_confidences, video_frame_indexes

    return filtered_coordinates, filtered_confidences


def clamp_conf_nonneg(conf_dict: Dict[str, np.ndarray]) -> None:
    """In-place clamp of confidences to [0, inf)."""
    for _, v in tqdm(conf_dict.items(), desc="Clamping confidences"):
        np.maximum(v, 0.0, out=v)


def subset_spread_blocks(
    length: int,
    frac: float = 0.1,
    n_blocks: int = 5,
    block_len: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Build a 1D index into a kept/filtered sequence selecting ~frac*T frames
    as several contiguous blocks spread over time.
    """
    T = int(length)
    if T <= 0 or frac <= 0:
        return np.zeros((0,), dtype=int)

    target = max(1, int(round(frac * T)))
    if n_blocks <= 0:
        n_blocks = 1

    if block_len is None:
        block_len = max(1, target // n_blocks)
    block_len = min(block_len, T)

    starts = np.linspace(0, max(0, T - block_len), num=n_blocks, dtype=int)
    if seed is not None and T > block_len:
        rng = np.random.default_rng(seed)
        max_jit = max(1, (T - block_len) // (n_blocks + 1))
        jitter = rng.integers(0, max_jit, size=n_blocks)
        starts = np.minimum(starts + jitter, T - block_len)

    idxs = []
    for s in starts:
        if len(idxs) >= target:
            break
        e = int(min(T, s + block_len))
        idxs.extend(range(s, e))

    idx = np.array(sorted(set(idxs)), dtype=int)
    if idx.size > target:
        idx = idx[:target]
    return idx


def subset_after_filter(
    filtered_coordinates: Dict[str, np.ndarray],
    filtered_confidences: Dict[str, np.ndarray],
    video_frame_indexes: Dict[str, np.ndarray],
    frac: float = 0.1,
    n_blocks: int = 5,
    block_len: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Apply subset_spread_blocks per recording after trimming/outlier removal."""
    sub_coords, sub_confs, sub_vfi = {}, {}, {}
    for k in filtered_coordinates.keys():
        T = len(filtered_coordinates[k])
        if T == 0:
            sub_coords[k] = filtered_coordinates[k][:0]
            sub_confs[k] = filtered_confidences[k][:0]
            sub_vfi[k] = video_frame_indexes[k][:0]
            continue

        idx = subset_spread_blocks(T, frac=frac, n_blocks=n_blocks, block_len=block_len, seed=seed)
        sub_coords[k] = filtered_coordinates[k][idx]
        sub_confs[k] = filtered_confidences[k][idx]
        sub_vfi[k] = video_frame_indexes[k][idx]

    return sub_coords, sub_confs, sub_vfi


def subset_by_activity(
    filtered_coordinates: Dict[str, np.ndarray],
    filtered_confidences: Dict[str, np.ndarray],
    video_frame_indexes: Dict[str, np.ndarray],
    frac: float = 0.5,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Selects the contiguous window of length (T * frac) with highest total centroid velocity."""
    sub_coords, sub_confs, sub_vfi = {}, {}, {}
    for k in filtered_coordinates.keys():
        coords = filtered_coordinates[k]
        T = len(coords)

        if T == 0:
            sub_coords[k] = filtered_coordinates[k][:0]
            sub_confs[k] = filtered_confidences[k][:0]
            sub_vfi[k] = video_frame_indexes[k][:0]
            continue

        target_len = int(T * frac)
        if target_len >= T:
            sub_coords[k] = filtered_coordinates[k]
            sub_confs[k] = filtered_confidences[k]
            sub_vfi[k] = video_frame_indexes[k]
            continue

        if target_len < 1:
            sub_coords[k] = filtered_coordinates[k][:0]
            sub_confs[k] = filtered_confidences[k][:0]
            sub_vfi[k] = video_frame_indexes[k][:0]
            continue

        centroids = np.nanmean(coords, axis=1)
        diffs = np.diff(centroids, axis=0)
        vel = np.linalg.norm(diffs, axis=1)
        vel = np.pad(vel, (0, 1), mode="constant", constant_values=0)

        cumsum = np.cumsum(vel)
        cs_pad = np.pad(cumsum, (1, 0), mode="constant", constant_values=0)
        idxs = np.arange(T - target_len + 1)
        window_sums = cs_pad[idxs + target_len] - cs_pad[idxs]

        best_start = int(np.argmax(window_sums))
        best_end = best_start + target_len

        sub_coords[k] = filtered_coordinates[k][best_start:best_end]
        sub_confs[k] = filtered_confidences[k][best_start:best_end]
        sub_vfi[k] = video_frame_indexes[k][best_start:best_end]

    return sub_coords, sub_confs, sub_vfi


def safe_fit_pca(data_dict: dict, project_dir: str | Path, max_seqs: int = 30) -> dict:
    """
    Subsample sequences if dataset is large to avoid GPU VRAM exhaustion during PCA fitting.
    """
    cfg = kpms.load_config(str(project_dir))
    if "Y" in data_dict and len(data_dict["Y"]) > max_seqs:
        sample_idx = np.linspace(0, len(data_dict["Y"]) - 1, max_seqs, dtype=int)
        sub_d = {
            k: v[sample_idx] if (hasattr(v, "__len__") and len(v) == len(data_dict["Y"])) else v
            for k, v in data_dict.items()
        }
        return kpms.fit_pca(**sub_d, **cfg)
    return kpms.fit_pca(**data_dict, **cfg)
