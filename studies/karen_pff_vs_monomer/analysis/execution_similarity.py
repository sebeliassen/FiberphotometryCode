"""Explicit, pooled execution distances for the bounded Karen coherence repair.

Posture, movement, tempo and known annotation agreement stay separately inspectable.
No genotype/drug inputs, tracking interpolation, automatic mirror or video warping.
"""

import numpy as np

PROFILES = {
    "phase_physical": {"posture": .45, "movement": .45, "tempo": .10},
    "phase_semantic": {"posture": .45, "movement": .35, "tempo": .10, "semantic": .10},
}


def regularized_whitener(covariance, shrinkage):
    covariance = np.asarray(covariance, float)
    regularized = ((1 - shrinkage) * covariance
                   + shrinkage * np.diag(np.maximum(np.diag(covariance), 1e-6)))
    eig, vec = np.linalg.eigh(regularized)
    return (vec / np.sqrt(np.maximum(eig, 1e-6))) @ vec.T


def kinematics(frames, centroid, heading, fps=60.):
    """px/s and signed rad/s; smooth within contiguous valid native blocks only."""
    from scipy.ndimage import median_filter

    frames, centroid, heading = np.asarray(frames), np.asarray(centroid), np.asarray(heading)
    if np.any(np.diff(frames) <= 0) or fps <= 0:
        raise ValueError("increasing native frames and positive FPS required")
    result = np.full((len(frames), 2), np.nan)
    bounds = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1, len(frames)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        xy = median_filter(centroid[a:b], size=(5, 1), mode="nearest")
        h = median_filter(np.unwrap(heading[a:b]), size=9, mode="nearest")
        speed = np.linalg.norm(np.gradient(xy, 1 / fps, axis=0), axis=1)
        omega = np.gradient(h, 1 / fps)
        result[a:b, 0] = np.log1p(speed)  # speed unit is explicitly 1 native px/s
        result[a:b, 1] = np.arcsinh(omega)  # signed; angular unit is 1 rad/s
    return result


def phase_blocks(track, starts, stops, semantic_codes, motion):
    """Three landmarks per each of three actual grouped bouts; tempo is separate."""
    starts, stops = np.asarray(starts, int), np.asarray(stops, int)
    if starts.ndim != 2 or starts.shape[1] != 3 or stops.shape != starts.shape:
        raise ValueError("expected three native bout intervals per call")
    if np.any(stops <= starts) or np.any(starts[:, 1:] != stops[:, :-1]):
        raise ValueError("bout intervals must be positive and contiguous")
    phase = np.array([0., .5, 1.])
    native = starts[:, :, None] + np.floor((stops - starts - 1)[:, :, None] * phase).astype(int)
    positions = np.searchsorted(track["frames"], native)
    if np.any(positions >= len(track["frames"])) or not np.array_equal(track["frames"][positions], native):
        raise ValueError("phase samples cross missing/excluded native support")
    pose = track["latent"][positions].reshape(-1, 9, 10).astype(float)
    move = motion[positions].reshape(-1, 9, 2)
    tempo = np.log((stops - starts) / 60.)
    semantics = semantic_codes[track["labels"][positions].astype(int)].reshape(-1, 9)
    if not all(np.isfinite(x).all() for x in (pose, move, tempo)):
        raise ValueError("execution lacks finite phase posture/movement/tempo")
    return {"posture": pose, "movement": move, "tempo": tempo, "semantic": semantics}


def fit_scaling(blocks, selected, shrinkage=.1, subjects=None):
    """Shared robust units and regularized within-posture whitening, no regression."""
    if not 0 < shrinkage <= 1:
        raise ValueError("positive covariance shrinkage required")
    pose = blocks["posture"][selected].reshape(-1, 10)
    centre = np.median(pose, axis=0)
    scale = np.maximum(np.diff(np.quantile(pose, [.25, .75], axis=0), axis=0)[0], 1e-6)
    robust = (pose - centre) / scale
    # Estimate correlation after bounded calibration only; retain unbounded values
    # at application so the calibration cap cannot hide an execution outlier.
    cov = np.cov(np.clip(robust, -5, 5), rowvar=False)
    whitening = regularized_whitener(cov, shrinkage)
    move = blocks["movement"][selected].reshape(-1, 2)
    mc = np.median(move, axis=0)
    ms = np.maximum(np.diff(np.quantile(move, [.25, .75], axis=0), axis=0)[0], .1)
    tempo = blocks["tempo"][selected].ravel()
    tc = float(np.median(tempo))
    ts = max(float(np.diff(np.quantile(tempo, [.25, .75]))[0]), .1)
    recipe = dict(posture_centre=centre.tolist(), posture_iqr=scale.tolist(),
                posture_whitening=whitening.tolist(), covariance_shrinkage=shrinkage,
                movement_centre=mc.tolist(), movement_iqr=ms.tolist(),
                tempo_centre=tc, tempo_iqr=ts, landmark_count=9,
                movement_units="log1p(speed/(1px/s)), asinh(signed angular speed/(1rad/s))",
                temporal_policy="three landmarks per grouped bout; no elastic alignment",
                location_policy="not included: calibrated arena boundaries unavailable",
                regression_policy="no automatic locomotion regression or chirality folding")
    # Feature units and coordinate count alone do not equate typical block
    # distances. Calibrate that final quantity on one shared, fixed pooled sample.
    normalized = scaled_blocks(blocks, recipe)
    rng = np.random.default_rng(2026)
    first = rng.choice(selected, 8192)
    second = rng.choice(selected, 8192)
    if subjects is not None:
        subjects = np.asarray(subjects)
        if len(np.unique(subjects[selected])) < 2:
            raise ValueError("cross-mouse scaling needs at least two mice")
        same = subjects[first] == subjects[second]
        while same.any():
            second[same] = rng.choice(selected, int(same.sum()))
            same = subjects[first] == subjects[second]
    recipe["block_pair_d2"] = {
        key: max(float(np.median(np.mean((normalized[key][first] - normalized[key][second]) ** 2, axis=1))), 1e-6)
        for key in ("posture", "movement", "tempo")
    }
    recipe["block_pair_calibration"] = dict(pairs=8192, seed=2026, cross_mouse=subjects is not None,
                                             statistic="median mean-squared distance; frozen pooled calibration")
    return recipe


def scaled_blocks(blocks, scale):
    pose = (blocks["posture"] - scale["posture_centre"]) / scale["posture_iqr"]
    pose = pose @ np.asarray(scale["posture_whitening"])
    move = (blocks["movement"] - scale["movement_centre"]) / scale["movement_iqr"]
    tempo = (blocks["tempo"] - scale["tempo_centre"]) / scale["tempo_iqr"]
    result = {"posture": pose.reshape(len(pose), -1), "movement": move.reshape(len(move), -1),
              "tempo": tempo, "semantic": blocks["semantic"]}
    for key, distance in scale.get("block_pair_d2", {}).items():
        result[key] /= np.sqrt(distance)
    return result


def weighted_vectors(blocks, profile):
    """Mean-square contribution per block; dimensionality cannot multiply weight."""
    profile = {k: v for k, v in profile.items() if k != "semantic"}
    if not profile or any(v < 0 for v in profile.values()) or sum(profile.values()) <= 0:
        raise ValueError("positive nonnegative physical block weights required")
    total = sum(profile.values())
    return np.concatenate([blocks[k] * np.sqrt(v / total / blocks[k].shape[1])
                           for k, v in profile.items()], axis=1)


def exemplar_distances(blocks, subjects, profile):
    """Physical medoid; optional soft semantic term on mutually known landmarks.

    The semantic profile deliberately keeps a physical medoid, then supplements
    neighbour distances. Unknown labels give no evidence of semantic agreement;
    unavailable semantic terms redistribute their weight to physical blocks.
    """
    from studies.sequence_viz.recreate_generator_videos import rank_exemplar_neighbors

    physical = weighted_vectors(blocks, profile)
    _, _, exemplar = rank_exemplar_neighbors(physical, subjects)
    parts = {k: np.mean((blocks[k] - blocks[k][exemplar]) ** 2, axis=1)
             for k in profile if k != "semantic"}
    labels = blocks["semantic"]
    common = (labels > 0) & (labels[exemplar] > 0)
    count = common.sum(axis=1)
    semantic = np.divide(((labels != labels[exemplar]) & common).sum(axis=1), count,
                         out=np.full(len(labels), np.nan), where=count > 0)
    distance = sum(profile[k] * parts[k] for k in parts)
    denominator = np.full(len(labels), sum(profile[k] for k in parts))
    if "semantic" in profile:
        evidence_weight = profile["semantic"] * count / labels.shape[1]
        distance += evidence_weight * np.nan_to_num(semantic)
        denominator += evidence_weight
    distance /= denominator
    parts["semantic"] = semantic
    parts["semantic_shared_fraction"] = count / labels.shape[1]
    return distance, exemplar, parts


def percentiles(distance):
    from scipy.stats import rankdata

    if len(distance) == 1:
        return np.array([100.])
    return 100 * (len(distance) - rankdata(distance, method="average")) / (len(distance) - 1)


def distinct_execution_neighbours(distance, recordings, starts, stops, count=5):
    """Do not count overlapping calls in one other recording as independent votes."""
    recordings, starts, stops = np.asarray(recordings), np.asarray(starts), np.asarray(stops)
    indices = np.zeros((len(distance), count), int)
    available = np.zeros((len(distance), count), bool)
    for row, ordered in enumerate(np.argsort(distance, axis=1, kind="stable")):
        selected = []
        for candidate in ordered:
            if not np.isfinite(distance[row, candidate]):
                break
            if any(recordings[candidate] == recordings[old] and starts[candidate] < stops[old]
                   and stops[candidate] > starts[old] for old in selected):
                continue
            selected.append(candidate)
            if len(selected) == count:
                break
        indices[row, :len(selected)] = selected
        available[row, :len(selected)] = True
    return indices, available
