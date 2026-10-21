"""Native-clock measurements of frozen, explicitly refined behavioural themes.

No model fitting or group inference. Complete consecutive named cores are counted
throughout valid recordings; factor detections supply provenance, not membership.
"""

import json

import numpy as np
import pandas as pd

from .build_core_review import merge_intervals


def matches(table, stage_sets):
    """All consecutive matches; unknowns and native validity blocks cannot skip."""
    if not stage_sets or any(not stage for stage in stage_sets):
        raise ValueError("nonempty ordered stage sets required")
    n = len(table) - len(stage_sets) + 1
    if n <= 0:
        return np.empty(0, dtype=int)
    tokens, blocks = table.token.to_numpy(int), table.block.to_numpy(int)
    good = np.ones(n, bool)
    for lag, stage in enumerate(stage_sets):
        good &= np.isin(tokens[lag:lag + n], stage)
        good &= blocks[lag:lag + n] == blocks[:n]
        if lag:
            good &= (table.start_frame.to_numpy()[lag:lag + n]
                     == table.end_frame_exclusive.to_numpy()[lag - 1:lag - 1 + n])
    return np.flatnonzero(good)


def intervals_from_mask(frames, mask):
    """Half-open native support, preserving missing-frame and exclusion barriers."""
    frames, mask = np.asarray(frames), np.asarray(mask, bool)
    if frames.ndim != 1 or len(frames) != len(mask) or np.any(np.diff(frames) <= 0):
        raise ValueError("native frames must be strictly increasing")
    idx = np.flatnonzero(mask)
    if not len(idx):
        return np.empty((0, 2), dtype=np.int64)
    cuts = np.r_[0, np.flatnonzero(np.diff(frames[idx]) != 1) + 1, len(idx)]
    return np.asarray([(frames[idx[a]], frames[idx[b - 1]] + 1)
                       for a, b in zip(cuts[:-1], cuts[1:])], dtype=np.int64)


def union_frames(starts, stops):
    return sum(b - a for a, b in merge_intervals(zip(starts, stops)))


class Kinematics:
    """Float64 prefix calculations; no interpolation or change to native masks."""

    def __init__(self, cache, valid):
        self.frames = np.asarray(cache["frames"], np.int64)
        self.valid = np.asarray(valid, bool)
        self.fps = float(cache["fps"])
        self.xy = np.asarray(cache["centroid"], np.float64)
        self.latent = np.asarray(cache["latent"], np.float64)
        if self.fps <= 0 or np.any(np.diff(self.frames) <= 0):
            raise ValueError("invalid explicit clock")
        adj = (np.diff(self.frames) == 1) & self.valid[1:] & self.valid[:-1]
        speed = np.r_[np.nan, np.linalg.norm(np.diff(self.xy, axis=0), axis=1) * self.fps]
        angle = np.asarray(cache["heading"], np.float64)
        turn = np.r_[np.nan, np.arctan2(np.sin(np.diff(angle)), np.cos(np.diff(angle)))]
        speed[1:][~adj] = np.nan
        turn[1:][~adj] = np.nan
        self.prefix = {}
        for name, values in (("speed", speed), ("turn", turn), ("abs_turn", np.abs(turn))):
            finite = np.isfinite(values)
            self.prefix[name] = (np.r_[0., np.cumsum(np.where(finite, values, 0.))],
                                 np.r_[0, np.cumsum(finite)])
        finite = np.isfinite(self.latent).all(axis=1)
        self.latent_count = np.r_[0, np.cumsum(finite)]
        safe = np.where(np.isfinite(self.latent), self.latent, 0.)
        self.latent_sum = np.vstack([np.zeros(safe.shape[1]), np.cumsum(safe, axis=0)])
        self.latent_square = np.vstack([np.zeros(safe.shape[1]), np.cumsum(safe ** 2, axis=0)])
        self.invalid = np.r_[0, np.cumsum(~self.valid)]
        self.broken = np.r_[0, np.cumsum(np.r_[False, np.diff(self.frames) != 1])]

    def measure(self, starts, stops):
        starts, stops = np.asarray(starts, np.int64), np.asarray(stops, np.int64)
        a, b = np.searchsorted(self.frames, starts), np.searchsorted(self.frames, stops)
        n = b - a
        if (np.any(n <= 0) or np.any(a >= len(self.frames))
                or np.any(self.frames[a] != starts) or np.any(self.frames[b - 1] + 1 != stops)
                or np.any(n != stops - starts) or np.any(self.invalid[b] != self.invalid[a])
                or np.any(self.broken[b] != self.broken[a + 1])):
            raise ValueError("measurement window crosses unsupported native time")
        result = {"duration_s": (stops - starts) / self.fps}
        expected = n - 1
        for source, target, multiplier in (("speed", "mean_speed_px_s", 1.),
                                           ("abs_turn", "mean_abs_angular_velocity_rad_s", self.fps)):
            sums, counts = self.prefix[source]
            count = counts[b] - counts[a + 1]
            total = sums[b] - sums[a + 1]
            result[target] = np.divide(total * multiplier, count, out=np.full(len(n), np.nan),
                                       where=(count == expected) & (count > 0))
        displacement = np.linalg.norm(self.xy[b - 1] - self.xy[a], axis=1)
        result["net_displacement_px"] = displacement
        speed_sum, speed_n = self.prefix["speed"]
        result["path_length_px"] = np.where(speed_n[b] - speed_n[a + 1] == expected,
                                             (speed_sum[b] - speed_sum[a + 1]) / self.fps, np.nan)
        turn_sum, turn_n = self.prefix["turn"]
        result["signed_heading_change_rad"] = np.where(turn_n[b] - turn_n[a + 1] == expected,
                                                       turn_sum[b] - turn_sum[a + 1], np.nan)
        mean = (self.latent_sum[b] - self.latent_sum[a]) / n[:, None]
        variance = (self.latent_square[b] - self.latent_square[a]) / n[:, None] - mean ** 2
        result["latent_dispersion"] = np.where(self.latent_count[b] - self.latent_count[a] == n,
                                               np.sqrt(np.maximum(variance.mean(axis=1), 0.)), np.nan)
        result["motion_support_fraction"] = np.divide(
            speed_n[b] - speed_n[a + 1], expected, out=np.full(len(n), np.nan), where=expected > 0)
        return pd.DataFrame(result)


def session_measurements(meta, cache, mapping, themes):
    """Return inclusive events, stage and prefix branches, and mouse-ready rows."""
    from .run_reference_seqnmf import frame_bouts, merge_bouts
    from .theme_organisation import context_readouts, first_order_readouts, session_readouts

    frames, labels = cache["frames"], cache["labels"]
    excluded = cache.get("analysis_excluded", np.zeros(len(frames), bool))
    fps = float(cache["fps"])
    valid = (labels >= 0) & (labels < 100) & ~excluded
    raw = frame_bouts(frames, labels, excluded)
    grouped = merge_bouts(raw, mapping)
    known = valid.copy()
    known[valid] = mapping[labels[valid]] < 20
    valid_intervals = intervals_from_mask(frames, valid)
    named_intervals = intervals_from_mask(frames, known)
    exposure = float(np.sum(valid) / fps)
    geometry = Kinematics(cache, valid)
    rows, events, stage_rows, branch_rows = [], [], [], []

    def put(family, theme, feature, numerator, denominator, *, value=None, support=0, units="", aggregation=None):
        finite = np.isfinite(numerator) and np.isfinite(denominator)
        if aggregation is None:
            aggregation = ("mean" if feature.split("_g")[0] in
                           ("longest_run_calls", "concentration", "effective_routines", "transition_entropy_nats")
                           else "ratio")
        if value is None:
            value = float(numerator / denominator) if denominator > 0 and finite else np.nan
        rows.append(dict(family=family, theme=theme, feature=feature, value=value,
                         numerator=numerator, denominator=denominator, aggregation=aggregation,
                         value_scale=1., support=int(support), opportunities=denominator, units=units))

    def average(family, theme, feature, values, units=""):
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        put(family, theme, feature, values.sum(), len(values), support=len(values), units=units)

    for theme in themes:
        tid, sets = theme["id"], theme["stage_sets"]
        positions = matches(grouped, sets)
        first, last = grouped.iloc[positions], grouped.iloc[positions + len(sets) - 1]
        own = pd.DataFrame(dict(theme=tid, start_frame=first.start_frame.to_numpy(int),
                                end_frame_exclusive=last.end_frame_exclusive.to_numpy(int),
                                grouped_start=positions, grouped_stop=positions + len(sets), score=1.))
        own["occurrence_id"] = [f"{meta['session_id']}:{tid}:{p}" for p in positions]
        own["observed_tokens"] = [json.dumps(grouped.token.iloc[p:p + len(sets)].astype(int).tolist()) for p in positions]
        if len(own):
            measured = geometry.measure(own.start_frame, own.end_frame_exclusive)
            own = pd.concat([own.reset_index(drop=True), measured], axis=1)
            for lag in range(len(sets)):
                s = grouped.iloc[positions + lag]
                part = geometry.measure(s.start_frame, s.end_frame_exclusive)
                part["occurrence_id"], part["theme"], part["stage"] = own.occurrence_id, tid, lag + 1
                part["token"] = s.token.to_numpy(int)
                part["start_frame"], part["end_frame_exclusive"] = s.start_frame.to_numpy(int), s.end_frame_exclusive.to_numpy(int)
                part["duration_fraction"] = part.duration_s / own.duration_s
                stage_rows.append(part)
                for feature in ("duration_s", "duration_fraction", "mean_speed_px_s", "mean_abs_angular_velocity_rad_s"):
                    average("intra_stage", tid, f"stage{lag + 1}_{feature}", part[feature])
                for token in sets[lag]:
                    put("intra_variants", tid, f"stage{lag + 1}_token{token}_fraction",
                        int((part.token == token).sum()), len(part), support=len(part), units="fraction")
        else:
            for feature in ("duration_s", "mean_speed_px_s", "net_displacement_px", "path_length_px",
                            "mean_abs_angular_velocity_rad_s", "signed_heading_change_rad", "latent_dispersion"):
                own[feature] = pd.Series(dtype=float)
        events.append(own)
        put("recruitment", tid, "occurrences_per_min", len(own) * 60, exposure,
            support=len(own), units="events/min valid modelled time")
        covered = union_frames(own.start_frame, own.end_frame_exclusive) / fps if len(own) else 0.
        put("recruitment", tid, "union_time_fraction", covered, exposure, support=len(own), units="fraction")
        for feature in ("duration_s", "mean_speed_px_s", "mean_abs_angular_velocity_rad_s", "net_displacement_px",
                        "path_length_px", "signed_heading_change_rad", "latent_dispersion"):
            average("intra_execution", tid, feature, own[feature])
        roots = theme["root_stage_sets"]
        root_positions = matches(grouped, roots)
        observable, next_tokens, completed = 0, [], 0
        for p in root_positions:
            after = p + len(roots)
            seen = after < len(grouped) and grouped.block.iloc[after] == grouped.block.iloc[p]
            token = int(grouped.token.iloc[after]) if seen else -1
            observable += int(seen)
            next_tokens.append(token)
            matched_end = bool(seen and token in sets[len(roots)]) if len(roots) < len(sets) else None
            completed += int(matched_end is True)
            branch_rows.append(dict(theme=tid, root_start_frame=int(grouped.start_frame.iloc[p]),
                root_end_frame_exclusive=int(grouped.end_frame_exclusive.iloc[after - 1]),
                root_tokens=json.dumps(grouped.token.iloc[p:after].astype(int).tolist()),
                next_token=token, right_censored=not seen, branch_matches_theme_end=matched_end))
        put("branching", tid, "root_occurrences_per_min", len(root_positions) * 60, exposure,
            support=len(root_positions), units="roots/min valid modelled time")
        put("branching", tid, "root_censored_fraction", len(root_positions) - observable, len(root_positions), support=len(root_positions))
        for token in range(20):
            put("branching", tid, f"continuation_token{token}_fraction", next_tokens.count(token), observable, support=observable)
        put("branching", tid, "continuation_unresolved_fraction", sum(t >= 20 for t in next_tokens), observable, support=observable)
        if len(roots) < len(sets):
            put("branching", tid, "root_to_theme_end_fraction", completed, observable, support=observable)

    events = pd.concat(events, ignore_index=True)
    stages = pd.concat(stage_rows, ignore_index=True) if stage_rows else pd.DataFrame()
    branches = pd.DataFrame(branch_rows)
    put("recruitment", "__catalogue__", "occurrences_per_min", len(events) * 60, exposure, support=len(events))
    put("recruitment", "__catalogue__", "union_time_fraction",
        union_frames(events.start_frame, events.end_frame_exclusive) / fps if len(events) else 0., exposure, support=len(events))
    put("support", "__catalogue__", "named_time_fraction", float(known.sum()), float(valid.sum()), support=int(valid.sum()))

    # A separate exclusive timeline is used for organisation. Unknown frames are
    # barriers here, while recruitment rates retain all valid modelled exposure.
    organisation = session_readouts(events, named_intervals, fps, theme_ids=[t["id"] for t in themes],
                                    recording_end_frame=int(frames[-1] + 1))
    for r in organisation["readouts"].to_dict("records"):
        gap = r.get("gap_seconds", np.nan)
        feature = r["feature"] + (f"_g{gap:g}s" if np.isfinite(gap) else "")
        family = "inter_" + r["family"]
        put(family, r["theme"], feature, r["numerator"], r["denominator"], value=r["value"])
    selected = organisation["selected"]
    for_order = first_order_readouts(organisation, replicates=199,
        seed=20261021 + sum(map(ord, meta["session_id"])), memory_mb=64)
    for r in for_order["readouts"].to_dict("records"):
        put("inter_first_order", r["theme"], r["feature"], np.nan, np.nan, value=r["value"], aggregation="mean")
    execution = events.set_index("occurrence_id")
    joined = selected.copy()
    for feature in ("duration_s", "mean_speed_px_s", "mean_abs_angular_velocity_rad_s"):
        joined[feature] = joined.occurrence_id.map(execution[feature])
    context = context_readouts(joined, named_intervals, fps, metrics=("duration_s", "mean_speed_px_s"),
                              recording_end_frame=int(frames[-1] + 1))
    for r in context.to_dict("records"):
        put("history", r["theme"], r["feature"], np.nan, np.nan, value=r["value"],
            support=min(r.get("n_same", 0), r.get("n_other", 0)), aggregation="mean")

    # Matched same-raw-syllable contrasts, within support block/5min/occupancy.
    raw_features = geometry.measure(raw.start_frame, raw.end_frame_exclusive)
    middle = (raw.start_frame.to_numpy(int) + raw.end_frame_exclusive.to_numpy(int) - 1) // 2
    midpos = np.searchsorted(frames, middle)
    xy = np.asarray(cache["centroid"], float)
    centre = np.nanmedian(xy[valid], axis=0)
    location = xy[midpos]
    finite_location = np.isfinite(location).all(axis=1)
    quadrant = (location[:, 0] >= centre[0]).astype(int) * 2 + (location[:, 1] >= centre[1]).astype(int)
    in_any = np.zeros(len(raw), bool)
    membership = {}
    for theme in themes:
        inside = np.zeros(len(raw), bool)
        for e in events[events.theme == theme["id"]].itertuples():
            a = np.searchsorted(raw.start_frame, e.start_frame)
            b = np.searchsorted(raw.end_frame_exclusive, e.end_frame_exclusive, side="right")
            inside[a:b] = True
        membership[theme["id"]] = inside
        in_any |= inside
    for theme in themes:
        inside = membership[theme["id"]]
        for feature in ("duration_s", "mean_speed_px_s", "mean_abs_angular_velocity_rad_s", "latent_dispersion"):
            use = (inside | ~in_any) & finite_location & np.isfinite(raw_features[feature].to_numpy())
            table = pd.DataFrame(dict(token=raw.token.to_numpy()[use], block=raw.block.to_numpy()[use],
                time=(raw.start_frame.to_numpy()[use] // int(300 * fps)), space=quadrant[use],
                inside=inside[use], value=raw_features[feature].to_numpy()[use]))
            summary = table.groupby(["token", "block", "time", "space", "inside"]).value.agg(["mean", "count"]).unstack("inside")
            if ("count", True) not in summary or ("count", False) not in summary:
                put("intra_matched", theme["id"], feature + "_inside_minus_outside", np.nan, np.nan)
                continue
            both = summary[(summary[("count", True)] >= 3) & (summary[("count", False)] >= 3)]
            d = both[("mean", True)] - both[("mean", False)]
            weights = both[("count", True)]
            put("intra_matched", theme["id"], feature + "_inside_minus_outside",
                float((d * weights).sum()), float(weights.sum()), support=int(weights.sum()))

    identity = {name: meta[name] for name in ("session_id", "animal_id", "group", "wave")}
    tables = dict(readouts=pd.DataFrame(rows), occurrences=events, stages=stages, branches=branches,
                  source_bouts=pd.concat([raw.reset_index(drop=True), raw_features], axis=1),
                  first_order=for_order["null_summary"], context=context)
    for name, table in tables.items():
        for key, value in identity.items():
            table[key] = value
    audit = dict(session_id=meta["session_id"], valid_seconds=exposure, named_seconds=float(known.sum() / fps),
                 valid_blocks=len(valid_intervals), named_blocks=len(named_intervals), full_calls=len(events),
                 theme_counts=events.theme.value_counts().to_dict(), organisation=organisation["summary"])
    return tables, organisation, audit
