"""Project historical session tables to explicit, gap-preserving frame arrays."""

import numpy as np
import pandas as pd

from fpcode.contracts.pose import PoseSeries
from fpcode.contracts.timelines import TimeAxis


def pose_from_dlc_table(table, *, frames, bodyparts, clock, unit, coordinate_frame, scorer=None,
                        invalid_confidence_policy='reject'):
    """Decode an already-loaded DLC table by native frame identity, never row order.

    The caller verifies the file's recording identity. Missing native frame keys
    stay missing. No FPS, geometry, camera transform or confidence is invented.
    The explicit ``mask`` policy converts finite out-of-range confidence sentinels
    to NaN/unsupported samples. It does not clip or modify the source table.
    """
    if not isinstance(table, pd.DataFrame) or not isinstance(table.columns, pd.MultiIndex) or table.columns.nlevels != 3:
        raise ValueError("pose table requires three-level scorer/bodypart/coordinate columns")
    index = table.index.to_numpy()
    if index.dtype.kind not in 'iu' or np.any(index < 0) or np.any(index[1:] <= index[:-1]):
        raise ValueError("pose index must contain unique nonnegative increasing native frames")
    frames = np.asarray(frames)
    if frames.ndim != 1 or frames.dtype.kind not in 'iu' or np.any(frames < 0) or np.any(frames[1:] <= frames[:-1]):
        raise ValueError("requested native frames must be unique nonnegative increasing integers")
    scorers = table.columns.get_level_values(0).unique()
    if scorer is None:
        if len(scorers) != 1:
            raise ValueError("choose an explicit scorer when multiple trackers are present")
        scorer = scorers[0]
    parts = tuple(bodyparts)
    if not parts or len(set(parts)) != len(parts):
        raise ValueError("bodyparts must be nonempty and unique")
    if invalid_confidence_policy not in ('reject', 'mask'):
        raise ValueError("choose reject or mask for invalid confidence")
    required = [(scorer, part, axis) for part in parts for axis in ('x', 'y', 'likelihood')]
    if not table.columns.is_unique or any(column not in table for column in required):
        raise ValueError("pose table lacks unique requested landmark/coordinate columns")
    aligned = table.loc[:, required].reindex(frames).to_numpy(dtype=float, copy=True).reshape(len(frames), len(parts), 3)
    if invalid_confidence_policy == 'mask':
        confidence = aligned[:, :, 2]
        confidence[(confidence < 0) | (confidence > 1)] = np.nan
    return PoseSeries(TimeAxis(frames, 'frame', clock), aligned[:, :, :2], aligned[:, :, 2],
                      parts, unit, coordinate_frame)


def record_frame_arrays(record, *, export_columns=("velocity_px_s",)):
    """Read an already-loaded record; never deserialize files or infer frame rate.

    Returns original video frame indices, nullable labels encoded as -1, model
    centroid/heading/latents and optional exported velocity. A camera's explicit
    primary frame rate is required. The owning study selects the cohort.
    """
    table = record.syllables
    if not hasattr(table, "columns") or "syllable" not in table:
        raise ValueError("record needs a per-frame syllable table")
    frames = np.asarray(table.index)
    if frames.dtype.kind not in "iu" or np.any(np.diff(frames) <= 0):
        raise ValueError("syllable table index must be increasing original video frames")
    refs = record.video_refs_df
    primary = refs.loc[refs["role"] == "primary", "fps"].dropna().unique()
    if len(primary) != 1 or not np.isfinite(primary[0]) or primary[0] <= 0:
        raise ValueError("one explicit primary-video frame rate is required")
    labels = table["syllable"].to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(labels)
    if np.any(labels[valid] != np.floor(labels[valid])):
        raise ValueError("syllable assignments must be integers or missing")
    labels = np.where(valid, labels, -1).astype(np.int32)
    latent_cols = [c for c in table if str(c).startswith("latent_state_")]
    latent_cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
    arrays = {
        "frames": frames.astype(np.int64), "labels": labels,
        "centroid": table[["centroid_0", "centroid_1"]].to_numpy(dtype=np.float32),
        "heading": table["heading"].to_numpy(dtype=np.float32),
        "latent": table[latent_cols].to_numpy(dtype=np.float32),
        "fps": np.array(float(primary[0])),
    }
    exported = getattr(record, "all_data_df", None)
    if hasattr(exported, "columns") and "frame_index" in exported:
        aligned = exported.set_index("frame_index").reindex(frames)
        for name in export_columns:
            if name in aligned:
                arrays["exported_" + name] = aligned[name].to_numpy(dtype=float, na_value=np.nan)
    return arrays


def verify_result_projection(arrays, results, video_frames):
    """Check exact label placement against a named result and its native VFI.

    Returns counts. Overlap is forbidden, including overlapping equal labels.
    Missing labels outside model segments remain missing in the projection.
    """
    frames, labels = arrays["frames"], arrays["labels"]
    expected = np.full(len(frames), -1, dtype=np.int32)
    for key, result in results.items():
        native = np.asarray(video_frames[key])
        z = np.asarray(result["syllable"])
        if native.dtype.kind not in "iu" or len(native) != len(z) or np.any(np.diff(native) <= 0):
            raise ValueError(f"invalid original video-frame mapping: {key}")
        rows = np.searchsorted(frames, native)
        if np.any(rows >= len(frames)) or not np.array_equal(frames[rows], native):
            raise ValueError(f"model frames absent from session: {key}")
        if np.any(expected[rows] >= 0):
            raise ValueError(f"overlapping model segments: {key}")
        expected[rows] = z
    mismatch = int(np.count_nonzero(labels != expected))
    if mismatch:
        raise ValueError(f"session/model label projection differs at {mismatch} original frames")
    return {"model_segments": len(results), "verified_labels": int(np.sum(expected >= 0)),
            "unmodelled_frames": int(np.sum(expected < 0)), "label_mismatches": mismatch}
