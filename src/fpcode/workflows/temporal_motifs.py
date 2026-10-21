"""Native-time syllable templates and frame-resolution micro-movement motifs."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.behavior.analysis.phenotyping import frame_kinematics
from fpcode.behavior.episodes.seqnmf import SeqNMF, activation_events, reconstruct
from fpcode.behavior.primitives.bouts import run_boundaries


def _pack(blocks, length):
    matrices, masks = [], []
    for block in blocks:
        if block.shape[1] < length:
            continue
        matrices.extend((block, np.zeros((block.shape[0], length), dtype=np.float32)))
        masks.extend((np.r_[np.ones(block.shape[1] - length + 1, bool), np.zeros(length - 1, bool)],
                      np.zeros(length, bool)))
    if not matrices:
        return None, None
    return np.concatenate(matrices, axis=1), np.concatenate(masks)


def run_temporal_motifs(sessions, cache_dir, measurement_dir, chains_dir, *, vocabulary, annotations,
                        bin_seconds=.1, template_seconds=1.2, seed=101, reuse_native=False,
                        reuse_native_templates=False, workers=1):
    """Freeze training templates; separate real-time persistence from action order.

    Native-time seqNMF retains dwell times and may identify sustained actions.
    Micro-motifs use nonnegative speed, absolute yaw and latent-motion magnitude
    inside individual syllable bouts. No discrete micromovement identity is
    assumed; these remain exploratory numerical templates.
    """
    cache_dir, measurement_dir, chains_dir = map(Path, (cache_dir, measurement_dir, chains_dir))
    if workers < 1:
        raise ValueError("workers must be positive")
    splits = json.loads((chains_dir / "animal_splits.json").read_text())
    train = set(splits["train"])
    rng = np.random.default_rng(seed)
    length = round(template_seconds / bin_seconds)
    native_blocks, full = [], {}
    micro = {}
    usage = pd.read_csv(measurement_dir / "syllables.csv")
    named = annotations[~annotations.category.str.startswith(("Unassigned", "Dump"))].syllable_id
    targeted = usage[usage.syllable_id.isin(named)].groupby("syllable_id").time_fraction.mean().nlargest(5).index.tolist()
    for meta in sessions.itertuples():
        with np.load(cache_dir / (meta.session_id + ".npz"), allow_pickle=False) as source:
            arrays = {key: source[key] for key in source}
        z, frames, fps = arrays["labels"], arrays["frames"], float(arrays["fps"])
        bin_frames = round(bin_seconds * fps)
        if abs(bin_frames / fps - bin_seconds) > 1e-6:
            raise ValueError("native bin duration must be an exact recording-frame multiple")
        starts, ends = run_boundaries((z >= 0).astype(int), frame_indices=frames)
        blocks = []
        for a, b in zip(starts, ends):
            if z[a] < 0 or b - a < bin_frames * length:
                continue
            bins = (b - a) // bin_frames
            labels = z[a:a + bins * bin_frames].reshape(bins, bin_frames)
            x = np.zeros((vocabulary, bins), dtype=np.float32)
            for offset in range(bin_frames):
                np.add.at(x, (labels[:, offset], np.arange(bins)), 1 / bin_frames)
            blocks.append(x)
            if meta.animal_id in train:
                size = min(250, bins)
                start = int(rng.integers(max(1, bins - size + 1)))
                native_blocks.append(x[:, start:start + size])
        full[meta.session_id] = blocks
        kin = frame_kinematics(arrays["centroid"], arrays["heading"], arrays["latent"], frames, z, fps, filter_frames=5)
        features = np.stack((kin["smooth_speed_px_s"], abs(kin["yaw_rad_s"]), kin["latent_speed_units_s"]), axis=0)
        starts, ends = run_boundaries(z, frame_indices=frames)
        local = {}
        for syllable in targeted:
            candidates = [(a, b) for a, b in zip(starts, ends) if z[a] == syllable and b - a >= 10
                          and np.isfinite(features[:, a:b]).all()]
            chosen = rng.choice(len(candidates), min(len(candidates), 80), replace=False) if candidates else []
            local[syllable] = [features[:, candidates[i][0]:candidates[i][1]].astype(np.float32) for i in chosen]
        micro[meta.session_id] = local
    if not reuse_native:
        _native_motifs(sessions, native_blocks, full, chains_dir, measurement_dir, splits, length, bin_seconds, seed,
                       reuse_templates=reuse_native_templates, workers=workers)
    elif not (measurement_dir / "native_time_motif_usage.csv").exists():
        raise FileNotFoundError("reuse_native requires an existing native_time_motif_usage.csv")
    micro_rows, catalog = [], []
    for syllable in targeted:
        training = [block for meta in sessions.itertuples() if meta.animal_id in train
                    for block in micro[meta.session_id][syllable]]
        if not training:
            continue
        combined = np.concatenate(training, axis=1)
        scales = np.maximum(np.quantile(combined, .75, axis=1), 1e-6)
        training = [np.clip(block / scales[:, None], 0, 10) for block in training]
        xt, vt = _pack(training, 8)
        fitted = SeqNMF(3, 8, penalty=.01, iterations=70, seed=seed).fit(xt, valid_starts=vt)
        thresholds = np.quantile(fitted.h[:, vt], .90, axis=1)
        np.savez_compressed(chains_dir / f"micro_templates_s{syllable:02d}.npz", w=fitted.w, scales=scales, thresholds=thresholds)
        catalog.append({"syllable_id": syllable, "features": "speed_px_s abs_yaw_rad_s latent_speed_units_s",
                        "factors": 3, "template_frames": 8, "sampled_training_bouts": len(training),
                        "training_error": fitted.loss_history[-1]})
        for meta in sessions.itertuples():
            blocks = micro[meta.session_id][syllable]
            if not blocks:
                continue
            xs, vs = _pack([np.clip(b / scales[:, None], 0, 10) for b in blocks], 8)
            hs = fitted.infer(xs, valid_starts=vs, iterations=25)
            calls = activation_events(hs, fitted.w, xs, thresholds=thresholds, valid_starts=vs)
            exposure = sum(b.shape[1] for b in blocks) / meta.fps
            for factor in range(3):
                count = sum(call["factor"] == factor for call in calls)
                micro_rows.append({"session_id": meta.session_id, "animal_id": meta.animal_id, "group": meta.group,
                                   "condition": meta.condition, "micro_factor": f"{syllable}/f{factor}",
                                   "rate_per_syllable_second": count / exposure, "sampled_bouts": len(blocks),
                                   "sampled_syllable_seconds": exposure})
        print(f"Frame micro-motifs S{syllable:02d} complete", flush=True)
    pd.DataFrame(micro_rows).to_csv(measurement_dir / "micro_motif_usage.csv", index=False)
    pd.DataFrame(catalog).to_csv(chains_dir / "micro_motif_catalog.csv", index=False)


def _native_motifs(sessions, native_blocks, full, chains_dir, measurement_dir, splits, length, bin_seconds, seed,
                   *, reuse_templates=False, workers=1):
    if reuse_templates:
        with np.load(chains_dir / "native_time_templates.npz") as saved:
            if int(saved["length"]) != length or not np.isclose(float(saved["bin_seconds"]), bin_seconds):
                raise ValueError("cached native template clock differs from requested settings")
            model = SeqNMF(saved["w"].shape[1], length, penalty=.01, seed=seed)
            model.w, thresholds = saved["w"], saved["thresholds"]
    else:
        x, valid = _pack(native_blocks, length)
        model = SeqNMF(8, length, penalty=.01, iterations=60, seed=seed).fit(x, valid_starts=valid)
        thresholds = np.quantile(model.h[:, valid], .90, axis=1)
        np.savez_compressed(chains_dir / "native_time_templates.npz", w=model.w, length=length,
                            bin_seconds=bin_seconds, thresholds=thresholds)

    def measure(meta):
        xs, vs = _pack(full[meta.session_id], length)
        hs = model.infer(xs, valid_starts=vs, iterations=25)
        events = activation_events(hs, model.w, xs, thresholds=thresholds, valid_starts=vs)
        minutes = sum(block.shape[1] for block in full[meta.session_id]) * bin_seconds / 60
        rows, tests = [], []
        for factor in range(model.factors):
            calls = [event for event in events if event["factor"] == factor]
            rows.append({"session_id": meta.session_id, "animal_id": meta.animal_id, "group": meta.group,
                         "condition": meta.condition, "native_factor": factor, "rate_per_min": len(calls) / minutes,
                         "mean_template_error": np.mean([e["template_error"] for e in calls]) if calls else np.nan})
        if meta.animal_id in splits["test"]:
            real_error = float(np.linalg.norm(xs - reconstruct(model.w, hs)) / np.linalg.norm(xs))
            tests.append({"session_id": meta.session_id, "animal_id": meta.animal_id, "real_time_error": real_error})
        print("Native-time seqNMF", meta.session_id, flush=True)
        return rows, tests

    rows, tests = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for local_rows, local_tests in executor.map(measure, sessions.itertuples()):
            rows.extend(local_rows)
            tests.extend(local_tests)
    pd.DataFrame(rows).to_csv(measurement_dir / "native_time_motif_usage.csv", index=False)
    pd.DataFrame(tests).to_csv(chains_dir / "native_time_heldout_reconstruction.csv", index=False)
