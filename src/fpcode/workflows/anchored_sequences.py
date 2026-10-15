"""Measure frozen raw-motif hypotheses after explicit state aggregation."""

from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.behavior.episodes.chains import exact_occurrences, prefix_progression
from fpcode.behavior.primitives.resolutions import remap_states

IDENTITY = ["session_id", "animal_id", "group", "condition"]


def measure_anchored_sequences(sessions, measurement_dir, output_dir, source_catalog, mapping, *, min_length=3):
    """Map, RLE and deduplicate training-derived motifs; record collapsed patterns.

    Source motif identities are retained in the audit, not assumed equal to new
    discovery factor numbers. Exact detections are nonoverlapping per template.
    Prefix progression is conditional continuation, not task intent.
    """
    output_dir, measurement_dir = Path(output_dir), Path(measurement_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unique, audit = {}, []
    for source in source_catalog.itertuples():
        raw = np.array([int(t) for t in source.tokens.split()])
        mapped = remap_states(raw, mapping)
        mapped = tuple(mapped[np.r_[True, mapped[1:] != mapped[:-1]]].tolist())
        retained = len(mapped) >= min_length
        if retained:
            unique.setdefault(mapped, "anchor_" + source.chain_id)
        audit.append({"raw_chain_id": source.chain_id, "raw_tokens": source.tokens,
                      "mapped_tokens": " ".join(map(str, mapped)), "mapped_length": len(mapped),
                      "retained": retained, "anchor_id": unique.get(mapped),
                      "interpretation": "mapped >=3-bout pattern" if retained else "collapsed; no longer the original multi-step routine"})
    audit = pd.DataFrame(audit)
    audit.to_csv(output_dir / "anchor_mapping_audit.csv", index=False)
    rows, events = [], []
    for meta in sessions.itertuples():
        bouts = pd.read_csv(measurement_dir / "bouts_by_session" / (meta.session_id + ".csv"))
        duration = bouts.duration_s.sum()
        identity = {key: getattr(meta, key) for key in IDENTITY}
        for tokens, anchor_id in unique.items():
            calls = exact_occurrences(bouts, tokens)
            durations, speeds = [], []
            for a, b in calls:
                sub = bouts.iloc[a:b]
                durations.append(sub.duration_s.sum())
                speeds.append(np.average(sub.mean_speed_px_s, weights=sub.duration_s))
                events.append({**identity, "anchor_id": anchor_id, "start_bout": a, "end_bout": b,
                               "start_frame": sub.start_frame.iloc[0], "end_frame_exclusive": sub.end_frame_exclusive.iloc[-1]})
            progression = prefix_progression(bouts, tokens)
            progression = progression[~progression.prefix_censored] if len(progression) else progression
            row = {**identity, "anchor_id": anchor_id, "occurrences": len(calls),
                   "occurrences_per_min": len(calls) / (duration / 60), "coverage_fraction": sum(durations) / duration,
                   "duration_s": np.mean(durations) if len(calls) >= 10 else np.nan,
                   "mean_speed_px_s": np.mean(speeds) if len(calls) >= 10 else np.nan,
                   "prefix_opportunities": len(progression),
                   "completion_fraction": progression.exact_completion.mean() if len(progression) >= 10 else np.nan}
            rows.append(row)
        print("Anchored chains", meta.session_id, flush=True)
    measured = pd.DataFrame(rows, columns=IDENTITY + ["anchor_id", "occurrences", "occurrences_per_min", "coverage_fraction",
                                                    "duration_s", "mean_speed_px_s", "prefix_opportunities", "completion_fraction"])
    measured.to_csv(output_dir / "anchor_session_metrics.csv", index=False)
    pd.DataFrame(events, columns=IDENTITY + ["anchor_id", "start_bout", "end_bout", "start_frame", "end_frame_exclusive"]).to_csv(
        output_dir / "anchor_occurrences.csv", index=False)
    pieces = []
    for family, measures in (("anchored_chain_usage", ["occurrences_per_min", "coverage_fraction"]),
                             ("anchored_chain_execution", ["duration_s", "mean_speed_px_s"]),
                             ("anchored_chain_progression", ["completion_fraction"])):
        long = measured.melt(id_vars=IDENTITY + ["anchor_id"], value_vars=measures, var_name="measure", value_name="value")
        long["family"], long["feature"] = family, long.anchor_id + "/" + long.measure
        pieces.append(long[IDENTITY + ["family", "feature", "value"]])
    result = pd.concat(pieces, ignore_index=True)
    result.to_csv(output_dir / "anchor_features_long.csv", index=False)
    return result
