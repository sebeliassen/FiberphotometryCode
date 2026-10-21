"""Fixed primitive/movement controls from saved native raw bouts; no fitting."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MOVEMENT = (("mean_speed_px_s", "within_raw_bout_mean_speed_px_s", "px/s"),
            ("mean_abs_angular_velocity_rad_s", "within_raw_bout_mean_abs_angular_velocity_rad_s", "rad/s"))


def _consume(chunk, states, mapping):
    """Vectorized chunk sums; retain one last bout/session to preserve merges."""
    for sid, part in chunk.groupby("session_id", sort=False):
        s = states[str(sid)]
        raw = part.token.to_numpy(int)
        if np.any(raw < 0) or np.any(raw >= len(mapping)):
            raise ValueError("raw token outside frozen mapping")
        tokens = mapping[raw]
        start = part.start_frame.to_numpy(np.int64)
        end = part.end_frame_exclusive.to_numpy(np.int64)
        block = part.block.to_numpy(np.int64)
        widths = end - start
        if np.any(widths <= 0) or np.any(start[1:] < end[:-1]):
            raise ValueError("raw bouts must be positive, sorted and nonoverlapping")
        if s["last"] is not None and start[0] < s["last"][1]:
            raise ValueError("raw bouts reordered across CSV chunks")
        previous_token = np.r_[s["last"][0] if s["last"] else -1, tokens[:-1]]
        previous_end = np.r_[s["last"][1] if s["last"] else -1, end[:-1]]
        previous_block = np.r_[s["last"][2] if s["last"] else -1, block[:-1]]
        new = (tokens != previous_token) | (start != previous_end) | (block != previous_block)
        known = tokens < 20
        s["counts"] += np.bincount(tokens[new & known], minlength=20)
        s["frames"] += np.bincount(tokens[known], weights=widths[known], minlength=20)
        s["unresolved_frames"] += int(widths[~known].sum())
        s["total_frames"] += int(widths.sum())
        s["raw_bouts"] += len(part)
        s["last"] = (int(tokens[-1]), int(end[-1]), int(block[-1]))
        pairs = widths - 1
        for name, _, _ in MOVEMENT:
            values = part[name].to_numpy(float)
            good = np.isfinite(values) & (pairs > 0)
            s[name][0] += float(np.dot(values[good], pairs[good]))
            s[name][1] += int(pairs[good].sum())
            s[name][2] += int(pairs[(~np.isfinite(values)) & (pairs > 0)].sum())


def _initial():
    return {"counts": np.zeros(20, dtype=np.int64), "frames": np.zeros(20),
            "unresolved_frames": 0, "total_frames": 0, "raw_bouts": 0, "last": None,
            **{name: [0., 0, 0] for name, _, _ in MOVEMENT}}


def _rows(states, sessions, exposures):
    rows = []
    for session in sessions.to_dict("records"):
        sid = str(session["session_id"])
        s = states[sid]
        fps, seconds = float(session["fps"]), float(exposures[sid])
        if not np.isfinite(fps) or fps <= 0 or not np.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid native clock/exposure")
        if not np.isclose(s["total_frames"], seconds * fps, rtol=0, atol=1e-6):
            raise ValueError(f"raw-bout time does not conserve audited exposure: {sid}")

        def put(family, theme, feature, numerator, denominator, support, units):
            rows.append({"session_id": sid, "animal_id": session["animal_id"],
                         "group": session["group"], "wave": session["wave"],
                         "family": family, "theme": theme, "feature": feature,
                         "value": numerator / denominator if denominator > 0 else np.nan,
                         "numerator": numerator, "denominator": denominator,
                         "aggregation": "ratio", "value_scale": 1., "support": int(support),
                         "opportunities": denominator, "units": units})

        for token in range(20):
            theme = f"__primitive_token{token}__"
            put("primitive_controls", theme, "bouts_per_min", int(s["counts"][token]),
                seconds / 60, s["counts"][token], "bouts/min")
            put("primitive_controls", theme, "time_fraction", float(s["frames"][token]) / fps,
                seconds, s["counts"][token], "fraction")
        put("primitive_controls", "__unresolved_annotation__", "time_fraction",
            s["unresolved_frames"] / fps, seconds, s["raw_bouts"], "fraction")
        for name, feature, unit in MOVEMENT:
            total, pairs, _ = s[name]
            put("movement_controls", "__catalogue__", feature, total, pairs, pairs, unit)
    return pd.DataFrame(rows)


def build(output, *, chunk_rows=50000):
    """Write/add fixed controls, preserving the frozen atlas and original raw IDs.

    CSV chunks bound memory; movement means weight each finite raw-bout mean by
    its within-bout adjacent-pair count (native duration in frames minus one).
    Transitions between raw bouts are absent, so these are not all-frame means.
    Unresolved annotation time is a quality share, never a learned pooled state.
    """
    if chunk_rows < 1:
        raise ValueError("positive chunk_rows required")
    output = Path(output)
    library = json.loads((output / "library.json").read_text())
    mapping = np.asarray(library["mapping"]["raw_to_grouped"], dtype=np.int64)
    sessions = pd.read_csv(output / "sessions.csv", dtype={"session_id": str})
    if sessions.session_id.duplicated().any():
        raise ValueError("sessions must have unique recording IDs")
    states = {sid: _initial() for sid in sessions.session_id}
    exposures = {sid: json.loads((output / "session_tables" / sid / "audit.json").read_text())["valid_seconds"]
                 for sid in states}
    columns = ["session_id", "token", "start_frame", "end_frame_exclusive", "block", *[m[0] for m in MOVEMENT]]
    for chunk in pd.read_csv(output / "source_bouts.csv.gz", usecols=columns,
                             dtype={"session_id": str}, chunksize=chunk_rows):
        if not set(chunk.session_id).issubset(states):
            raise ValueError("unknown source-bout session")
        _consume(chunk, states, mapping)
    readouts = _rows(states, sessions, exposures)
    readouts.to_csv(output / "baseline_readouts.csv.gz", index=False)

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    metadata = {"status": "PASS", "sessions": len(sessions), "rows": len(readouts),
                "choice": "fixed group-blind primitive frequency/time and within-raw-bout movement controls; no endpoint selection",
                "primitive_names": {f"__primitive_token{t}__": library["mapping"]["tokens"][str(t)]["name"] for t in range(20)},
                "unknown_policy": "unresolved annotation quality share only; original raw IDs retained in unchanged source_bouts; unknowns, gaps and blocks prevent named merges",
                "movement_policy": "finite within-raw-bout means weighted by native interval frame count minus one; includes unresolved raw states; omits raw-bout transition pairs; not an all-frame mean",
                "units": {feature: units for _, feature, units in MOVEMENT},
                "chunk_rows": chunk_rows, "total_raw_bouts": sum(s["raw_bouts"] for s in states.values()),
                "audited_valid_seconds": sum(exposures.values()),
                "missing_movement_pair_counts": {name: sum(s[name][2] for s in states.values()) for name, _, _ in MOVEMENT},
                "sha256": {name: digest(output / name) for name in ("library.json", "sessions.csv", "source_bouts.csv.gz")}}
    (output / "baseline_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return readouts
