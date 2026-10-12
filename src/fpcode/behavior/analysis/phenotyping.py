"""Gap-aware frame, syllable, bout and within-session behavioural measurements.

All clocks, vocabulary sizes and thresholds are explicit. Physical quantities
retain their pixel/second units; model-latent quantities are named separately.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from fpcode.behavior.observations.kinematics import smooth_centroid_median, smooth_heading_circular
from fpcode.behavior.primitives.bouts import run_boundaries


def diversity(counts):
    """Entropy and effective vocabulary, without assigning meaning to richness."""
    counts = np.asarray(counts, dtype=float)
    p = counts[counts > 0] / max(counts.sum(), 1)
    entropy = -np.sum(p * np.log2(p))
    return {"entropy_bits": float(entropy), "effective_shannon": float(2 ** entropy),
            "effective_simpson": float(1 / np.sum(p ** 2)) if len(p) else 0.0,
            "observed_states": int(len(p)), "top5_fraction": float(np.sort(p)[-5:].sum())}


def frame_kinematics(centroid, heading, latent, frames, labels, fps, *, filter_frames, position_filter_frames=9):
    """Differentiate only consecutive supported frames; smooth within segments."""
    if fps <= 0 or filter_frames < 1 or filter_frames % 2 != 1 or position_filter_frames < 1 or position_filter_frames % 2 != 1:
        raise ValueError("positive fps and an odd positive smoothing window are required")
    c, h, x = (np.asarray(a, dtype=float) for a in (centroid, heading, latent))
    n = len(labels)
    if c.shape != (n, 2) or h.shape != (n,) or x.ndim != 2 or len(x) != n:
        raise ValueError("frame fields have inconsistent dimensions")
    valid = (labels >= 0) & np.isfinite(c).all(axis=1) & np.isfinite(h)
    filtered_c, filtered_h = c.copy(), h.copy()
    starts, ends = run_boundaries(valid.astype(int), frame_indices=frames)
    for start, end in zip(starts, ends):
        if valid[start]:
            filtered_c[start:end] = smooth_centroid_median(c[start:end], position_filter_frames)
            filtered_h[start:end] = smooth_heading_circular(h[start:end], position_filter_frames)
    adjacency = valid[1:] & valid[:-1] & (np.diff(frames) == 1)
    speed = np.full(n, np.nan)
    yaw = np.full(n, np.nan)
    latent_speed = np.full(n, np.nan)
    speed[1:] = np.where(adjacency, np.linalg.norm(np.diff(c, axis=0), axis=1) * fps, np.nan)
    yaw[1:] = np.where(adjacency, np.angle(np.exp(1j * np.diff(filtered_h))) * fps, np.nan)
    if x.shape[1]:
        latent_speed[1:] = np.where(adjacency, np.linalg.norm(np.diff(x, axis=0), axis=1) * fps, np.nan)
    smooth = np.r_[np.nan, np.where(adjacency, np.linalg.norm(np.diff(filtered_c, axis=0), axis=1) * fps, np.nan)]
    starts, ends = run_boundaries(np.isfinite(smooth).astype(int), frame_indices=frames)
    for start, end in zip(starts, ends):
        if np.isfinite(smooth[start]):
            smooth[start:end] = median_filter(smooth[start:end], size=filter_frames, mode="nearest")
    acceleration = np.r_[np.nan, np.diff(smooth) * fps]
    jerk = np.r_[np.nan, np.diff(acceleration) * fps]
    return {"speed_px_s": speed, "smooth_speed_px_s": smooth, "yaw_rad_s": yaw,
            "acceleration_px_s2": acceleration, "jerk_px_s3": jerk,
            "latent_speed_units_s": latent_speed, "filtered_centroid": filtered_c, "filtered_heading": filtered_h}


def _finite_stat(values, operation):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(operation(values)) if len(values) else np.nan


def analyze_frames(arrays, *, num_states, pause_speed_px_s, filter_frames=5, phase_bins=15,
                   time_bins=8, max_profile_bouts=150, seed=101):
    """Return reusable tables; invalid labels and dropped frames are boundaries.

    Phase-normalized profiles describe shape only. Bout duration, native peak
    latency and onset-aligned profiles preserve actual execution timing.
    """
    z, frames = arrays["labels"], arrays["frames"]
    fps = float(arrays["fps"])
    if np.any(z >= num_states):
        raise ValueError("vocabulary excludes an observed syllable")
    kin = frame_kinematics(arrays["centroid"], arrays["heading"], arrays["latent"], frames, z, fps,
                           filter_frames=filter_frames)
    valid = z >= 0
    starts, ends = run_boundaries(z, frame_indices=frames)
    keep = z[starts] >= 0
    starts, ends = starts[keep], ends[keep]
    segments = np.cumsum(np.r_[True, (~valid[:-1]) | (~valid[1:]) | (np.diff(frames) != 1)])
    records = []
    for idx, (a, b) in enumerate(zip(starts, ends)):
        v = kin["smooth_speed_px_s"][a:b]
        raw_v = kin["speed_px_s"][a:b]
        good = np.flatnonzero(np.isfinite(v))
        left_censored = a == 0 or not valid[a - 1] or frames[a] != frames[a - 1] + 1
        right_censored = b == len(z) or not valid[b] or frames[b] != frames[b - 1] + 1
        peak = int(good[np.argmax(v[good])]) if len(good) else 0
        position = kin["filtered_centroid"][a:b]
        displacement = np.linalg.norm(position[-1] - position[0])
        path = np.linalg.norm(np.diff(position, axis=0), axis=1).sum()
        lat = arrays["latent"][a:b]
        records.append({
            "bout_index": idx, "syllable_id": int(z[a]), "segment": int(segments[a]),
            "start_row": int(a), "end_row": int(b), "start_frame": int(frames[a]),
            "end_frame_exclusive": int(frames[b - 1] + 1), "duration_s": (b - a) / fps,
            "left_censored": left_censored, "right_censored": right_censored,
            "mean_speed_px_s": _finite_stat(v, np.mean), "peak_speed_px_s": _finite_stat(v, np.max),
            "raw_mean_speed_px_s": _finite_stat(raw_v, np.mean), "speed_sd_px_s": _finite_stat(v, np.std),
            "pause_fraction": float(np.mean(v[good] < pause_speed_px_s)) if len(good) else np.nan,
            "peak_latency_s": peak / fps if len(good) else np.nan,
            "peak_phase": peak / max(b - a - 1, 1) if len(good) else np.nan,
            "entry_speed_px_s": v[good[0]] if len(good) else np.nan,
            "exit_speed_px_s": v[good[-1]] if len(good) else np.nan,
            "speed_change_px_s": v[good[-1]] - v[good[0]] if len(good) else np.nan,
            "mean_abs_yaw_rad_s": _finite_stat(abs(kin["yaw_rad_s"][a:b]), np.mean),
            "mean_signed_yaw_rad_s": _finite_stat(kin["yaw_rad_s"][a:b], np.mean),
            "net_turn_rad": float(np.angle(np.exp(1j * (kin["filtered_heading"][b - 1] - kin["filtered_heading"][a])))),
            "mean_abs_acceleration_px_s2": _finite_stat(abs(kin["acceleration_px_s2"][a:b]), np.mean),
            "mean_abs_jerk_px_s3": _finite_stat(abs(kin["jerk_px_s3"][a:b]), np.mean),
            "displacement_px": float(displacement), "path_length_px": float(path),
            "straightness": float(displacement / path) if path > 0 else np.nan,
            "latent_path_units": float(np.linalg.norm(np.diff(lat, axis=0), axis=1).sum()),
            "latent_dispersion_units2": float(np.nanvar(lat, axis=0).sum()),
        })
    bouts = pd.DataFrame(records)
    if bouts.empty:
        raise ValueError("no supported syllable bouts")
    minutes = valid.sum() / fps / 60
    frame_counts = np.bincount(z[valid], minlength=num_states)
    rows = []
    feature_columns = [c for c in bouts.select_dtypes(include="number")
                       if c not in {"bout_index", "syllable_id", "segment", "start_row", "end_row", "start_frame",
                                    "end_frame_exclusive"}]
    for syllable in range(num_states):
        sub = bouts[bouts.syllable_id == syllable]
        row = {"syllable_id": syllable, "time_fraction": frame_counts[syllable] / valid.sum(),
               "bout_count": len(sub), "bouts_per_min": len(sub) / minutes,
               "uncensored_bout_count": int((~sub.left_censored & ~sub.right_censored).sum()),
               "duration_median_s": sub.duration_s.median(), "duration_p90_s": sub.duration_s.quantile(.9),
               "duration_cv": sub.duration_s.std() / sub.duration_s.mean() if len(sub) > 1 else np.nan,
               "first_onset_s": (sub.start_frame.min() - frames[0]) / fps if len(sub) else np.nan}
        for col in feature_columns:
            row[col] = sub[col].mean()
        rows.append(row)
    syllables = pd.DataFrame(rows)
    transitions = []
    pairs = []
    for i in range(len(bouts) - 1):
        if bouts.segment.iat[i] == bouts.segment.iat[i + 1]:
            pairs.append((int(bouts.syllable_id.iat[i]), int(bouts.syllable_id.iat[i + 1])))
    count_matrix = np.zeros((num_states, num_states), dtype=int)
    if pairs:
        a, b = np.asarray(pairs).T
        np.add.at(count_matrix, (a, b), 1)
    totals = count_matrix.sum(axis=1)
    for a, b in zip(*np.nonzero(count_matrix)):
        transitions.append({"from_syllable": a, "to_syllable": b, "count": count_matrix[a, b],
                            "source_opportunities": totals[a], "probability": count_matrix[a, b] / totals[a],
                            "rate_per_min": count_matrix[a, b] / minutes})
    entropy_rate = 0.0
    for row in count_matrix:
        p = row[row > 0] / max(row.sum(), 1)
        entropy_rate += row.sum() / max(count_matrix.sum(), 1) * -np.sum(p * np.log2(p))
    summary = {"valid_duration_s": valid.sum() / fps, "modelled_fraction": float(valid.mean()),
               "bout_count": len(bouts), "bouts_per_min": len(bouts) / minutes,
               "switches_per_min": len(pairs) / minutes, "mean_bout_duration_s": bouts.duration_s.mean(),
               "mean_speed_px_s": _finite_stat(kin["smooth_speed_px_s"], np.mean),
               "speed_p95_px_s": _finite_stat(kin["smooth_speed_px_s"], lambda a: np.quantile(a, .95)),
               "pause_fraction": _finite_stat(kin["smooth_speed_px_s"], lambda a: np.mean(a < pause_speed_px_s)),
               "transition_entropy_bits": entropy_rate, **diversity(frame_counts)}
    profiles, onset_profiles = [], []
    rng = np.random.default_rng(seed)
    q = np.linspace(0, 1, phase_bins)
    for syllable, sub in bouts.groupby("syllable_id"):
        eligible = sub[(sub.end_row - sub.start_row >= 3) & ~sub.left_censored & ~sub.right_censored]
        if len(eligible) > max_profile_bouts:
            eligible = eligible.iloc[np.sort(rng.choice(len(eligible), max_profile_bouts, replace=False))]
        aligned = {k: [] for k in ("smooth_speed_px_s", "yaw_rad_s", "latent_speed_units_s")}
        native = {k: [] for k in aligned}
        for bout in eligible.itertuples():
            a, b = bout.start_row, bout.end_row
            for key in aligned:
                value = kin[key][a:b]
                if np.isfinite(value).all():
                    aligned[key].append(np.interp(q, np.linspace(0, 1, len(value)), value))
                    padded = np.full(19, np.nan)
                    take = min(len(value), 19)
                    padded[:take] = value[:take]
                    native[key].append(padded)
        for key in aligned:
            if not aligned[key]:
                continue
            matrix = np.asarray(aligned[key])
            for phase in range(phase_bins):
                profiles.append({"syllable_id": syllable, "feature": key, "phase": q[phase],
                                 "mean": matrix[:, phase].mean(), "sd": matrix[:, phase].std(),
                                 "n_bouts": len(matrix)})
            matrix = np.asarray(native[key])
            for frame in range(19):
                values = matrix[:, frame]
                values = values[np.isfinite(values)]
                if len(values):
                    onset_profiles.append({"syllable_id": syllable, "feature": key, "time_s": frame / fps,
                                           "mean": values.mean(), "sd": values.std(), "n_bouts": len(values)})
    bins = []
    elapsed = (frames - frames[0]) / fps
    total_duration = (frames[-1] - frames[0] + 1) / fps
    for i in range(time_bins):
        sel = valid & (elapsed >= i * total_duration / time_bins) & (elapsed < (i + 1) * total_duration / time_bins)
        if not sel.any():
            continue
        bin_bouts = bouts[(bouts.start_frame >= frames[0] + i * total_duration * fps / time_bins)
                          & (bouts.start_frame < frames[0] + (i + 1) * total_duration * fps / time_bins)]
        bins.append({"time_bin": i, "elapsed_fraction": (i + .5) / time_bins,
                     "valid_duration_s": sel.sum() / fps,
                     "mean_speed_px_s": _finite_stat(kin["smooth_speed_px_s"][sel], np.mean),
                     "pause_fraction": _finite_stat(kin["smooth_speed_px_s"][sel], lambda a: np.mean(a < pause_speed_px_s)),
                     "bouts_per_min": len(bin_bouts) / (sel.sum() / fps / 60),
                     **diversity(np.bincount(z[sel], minlength=num_states))})
    return {"bouts": bouts, "syllables": syllables, "transitions": pd.DataFrame(transitions),
            "summary": summary, "phase_profiles": pd.DataFrame(profiles),
            "onset_profiles": pd.DataFrame(onset_profiles), "time_bins": pd.DataFrame(bins)}
