"""Compose explicitly mapped fused observations into posture/paw tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.tracking.pose_measurements import body_relative_features


def run_pose_atlas(sessions, cache_dir, pose_paths, output_dir, *, landmarks, confidence_threshold,
                   vocabulary, minimum_valid_fraction=.8):
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    records, audit = [], []
    for number, meta in enumerate(sessions.itertuples()):
        path = Path(pose_paths[meta.session_id])
        raw = pd.read_hdf(path)
        scorer = raw.columns.levels[0][0]
        names = list(landmarks.values())
        index_map = {key: names.index(value) for key, value in landmarks.items()}
        with np.load(cache_dir / (meta.session_id + ".npz"), allow_pickle=False) as source:
            frames, labels, fps = source["frames"], source["labels"], float(source["fps"])
        if raw.index.dtype.kind not in "iu" or not raw.index.is_unique:
            raise ValueError("fused pose index must name unique original video frames")
        # Exact index reindexing leaves absent observations missing.
        aligned = raw.reindex(frames)
        pose = np.stack([aligned.loc[:, [(scorer, name, "x"), (scorer, name, "y")]].to_numpy(float)
                         for name in names], axis=1)
        confidence = np.column_stack([aligned[(scorer, name, "likelihood")].to_numpy(float) for name in names])
        features = body_relative_features(pose, confidence, landmarks=index_map,
                                         confidence_threshold=confidence_threshold, frames=frames, fps=fps)
        observed = int(np.sum(np.isfinite(pose).all(axis=(1, 2))))
        audit.append({"session_id": meta.session_id, "pose_path": str(path.resolve()),
                      "file_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                      "exactly_aligned_pose_frames": observed, "modelled_frames": int((labels >= 0).sum())})
        for syllable in range(vocabulary):
            selected = labels == syllable
            if selected.sum() < 10:
                continue
            for name, feature in features.items():
                values = feature[selected]
                valid = np.isfinite(values)
                fraction = float(valid.mean())
                mean = values[valid].mean() if fraction >= minimum_valid_fraction else np.nan
                sd = values[valid].std() if fraction >= minimum_valid_fraction else np.nan
                records.append({"session_id": meta.session_id, "animal_id": meta.animal_id, "group": meta.group,
                                "condition": meta.condition, "pose_feature": f"{syllable}/{name}",
                                "mean": mean, "sd": sd, "valid_fraction": fraction,
                                "n_valid_frames": int(valid.sum())})
        print(f"Posture/paw measurements {number + 1}/{len(sessions)}", flush=True)
    pd.DataFrame(records).to_csv(output_dir / "pose_syllable_measurements.csv", index=False)
    pd.DataFrame(audit).to_csv(output_dir / "pose_input_audit.csv", index=False)
