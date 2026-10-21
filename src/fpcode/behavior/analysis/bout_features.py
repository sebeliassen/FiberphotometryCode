"""Vectorized execution summaries on unchanged, explicitly supported source bouts."""

from numbers import Integral, Real

import numpy as np

from fpcode.behavior.primitives.bouts import run_boundaries


def summarize_source_bouts(frames, labels, centroid, *, fps, heading=None, latent=None,
                           minimum_support=.8, memory_budget_bytes=512*1024**2):
    """Preserve source intervals/gaps and missing features, without fitting geometry.

    Centroids are declared image pixels, heading radians, latent channels carry
    model-specific latent units. Derivatives never cross gaps or bout edges.
    Summaries use finite observations only and require the declared support
    fraction. Position is an image-coordinate proxy, not distance to real walls.
    """
    frames, labels, centroid = np.asarray(frames), np.asarray(labels), np.asarray(centroid, dtype=float)
    n = len(frames)
    if (isinstance(memory_budget_bytes, bool) or not isinstance(memory_budget_bytes, Integral)
            or memory_budget_bytes < 1):
        raise ValueError("memory_budget_bytes must be a positive integer")
    if (frames.ndim != 1 or frames.dtype.kind not in "iu" or labels.shape != frames.shape
            or labels.dtype.kind not in "iu" or centroid.shape != (n, 2) or np.any(frames < 0)
            or np.any(frames[1:] <= frames[:-1]) or isinstance(fps, bool) or not isinstance(fps, Real)
            or not np.isfinite(fps) or fps <= 0 or isinstance(minimum_support, bool)
            or not isinstance(minimum_support, Real) or not np.isfinite(minimum_support)
            or not 0 < minimum_support <= 1):
        raise ValueError("source labels, native coordinates, centroid, rate and support policy must be explicit")
    if not n:
        raise ValueError("nonempty source required")
    if int(frames[-1]) >= np.iinfo(np.int64).max or int(labels.max()) > np.iinfo(np.int64).max:
        raise ValueError("source native frames/labels exceed representable int64 output coordinates")
    if heading is not None:
        heading = np.asarray(heading, dtype=float)
        if heading.shape != frames.shape:
            raise ValueError("heading must align to native samples")
    if latent is not None:
        latent = np.asarray(latent, dtype=float)
        if latent.ndim != 2 or len(latent) != n or not 1 <= latent.shape[1] <= 64:
            raise ValueError("latent channels must align to native samples with 1 to 64 dimensions")
    latent_dimensions = 0 if latent is None else latent.shape[1]
    # Bound the worst case of one bout per frame: returned feature matrix,
    # per-channel means/variances and stacking/filter copies scale with schema.
    # Input arrays already owned by the caller are excluded from this cap.
    output_columns = 6 + 2*int(heading is not None) + latent_dimensions + int(latent is not None)
    estimated_memory = n*(256 + 32*output_columns + 16*latent_dimensions)
    if estimated_memory > memory_budget_bytes:
        raise MemoryError("bout projection working-array budget exceeded")
    starts, stops = run_boundaries(labels, frame_indices=frames)
    good = labels[starts] >= 0
    lengths = stops-starts
    pose_valid = np.isfinite(centroid).all(axis=1) & (labels >= 0)
    steps = np.full(n, np.nan)
    valid_pair = pose_valid[1:] & pose_valid[:-1] & (np.diff(frames) == 1)
    steps[1:][valid_pair] = np.linalg.norm(np.diff(centroid, axis=0)[valid_pair], axis=1)
    steps[starts] = np.nan

    def sums(values):
        return np.add.reduceat(np.where(np.isfinite(values), values, 0.), starts)

    def means(values):
        count = np.add.reduceat(np.isfinite(values).astype(int), starts)
        result = np.divide(sums(values), count, out=np.full(len(starts), np.nan), where=count > 0)
        result[count < minimum_support*lengths] = np.nan
        return result

    duration = (frames[stops-1]+1-frames[starts])/float(fps)
    path = sums(steps)
    valid_count = np.add.reduceat(np.isfinite(steps).astype(int), starts)
    path[valid_count < minimum_support*np.maximum(lengths-1, 1)] = np.nan
    speed = np.divide(path*fps, lengths-1, out=np.full(len(starts), np.nan), where=lengths > 1)
    endpoint_valid = pose_valid[starts] & pose_valid[stops-1] & (lengths > 1)
    displacement = np.linalg.norm(centroid[stops-1]-centroid[starts], axis=1)
    net_speed = np.where(endpoint_valid, displacement*fps/np.maximum(lengths-1, 1), np.nan)
    columns = [duration, speed, net_speed, path, means(centroid[:, 0]), means(centroid[:, 1])]
    names = ["duration_s", "mean_speed_px_s", "net_speed_px_s", "path_length_px", "centroid_x_px", "centroid_y_px"]
    units = ["s", "px/s", "px/s", "px", "px", "px"]
    if heading is not None:
        changes = np.full(n, np.nan)
        valid = np.isfinite(heading[1:]) & np.isfinite(heading[:-1]) & (np.diff(frames) == 1)
        changes[1:][valid] = (np.diff(heading)[valid]+np.pi) % (2*np.pi)-np.pi
        changes[starts] = np.nan
        count = np.add.reduceat(np.isfinite(changes).astype(int), starts)
        turn, absolute = sums(changes), sums(np.abs(changes))
        insufficient = count < minimum_support*np.maximum(lengths-1, 1)
        turn[insufficient], absolute[insufficient] = np.nan, np.nan
        columns.extend([turn, absolute])
        names.extend(["heading_change_rad", "absolute_heading_change_rad"])
        units.extend(["rad", "rad"])
    if latent is not None:
        variances = []
        for i in range(latent.shape[1]):
            mean = means(latent[:, i])
            variances.append(np.maximum(means(latent[:, i]**2)-mean**2, 0))
            columns.append(mean)
            names.append(f"latent_mean_{i}")
            units.append("source-model latent unit")
        columns.append(np.mean(np.asarray(variances), axis=0))
        names.append("latent_dispersion")
        units.append("source-model latent unit squared")
    return {"start_frames": frames[starts[good]].astype(np.int64),
            "stop_frames": (frames[stops[good]-1]+1).astype(np.int64),
            "parent_labels": labels[starts[good]].astype(np.int64),
            "values": np.column_stack(columns)[good], "feature_names": tuple(names), "feature_units": tuple(units),
            "minimum_support": float(minimum_support), "fps": float(fps)}
