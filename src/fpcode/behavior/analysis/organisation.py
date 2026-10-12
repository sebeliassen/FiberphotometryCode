"""Auditable organisation of frozen chain calls on an explicit native clock.

Intervals are half open. Overlap selection precedes any membership exclusion;
excluded calls and missing data remain barriers in every downstream measure.
"""

from bisect import bisect_left

import numpy as np
import pandas as pd


def valid_frame_intervals(frames, labels):
    """Contiguous labelled native-frame blocks; missing/negative labels break them."""
    frames, labels = np.asarray(frames), np.asarray(labels)
    if frames.ndim != 1 or labels.shape != frames.shape or np.any(np.diff(frames) <= 0):
        raise ValueError("frames must be a strictly increasing vector aligned with labels")
    valid = frames[labels >= 0].astype(np.int64)
    if not len(valid):
        return np.empty((0, 2), dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(valid) != 1) + 1, len(valid)]
    return np.column_stack((valid[boundaries[:-1]], valid[boundaries[1:] - 1] + 1))


def exclusive_timeline(events, valid_intervals, *, fps, recording_end_frame):
    """Return selected calls, all call decisions and a complete interval timeline.

    Priority: descending selection_score, lexical chain_id, ascending start/end,
    then original row number. Compare native intervals, independently of bouts.
    Calls outside a single valid block are rejected explicitly.
    """
    blocks = np.asarray(valid_intervals, dtype=np.int64).reshape(-1, 2)
    if fps <= 0 or recording_end_frame < 0:
        raise ValueError("positive fps and nonnegative recording end required")
    if len(blocks) and (np.any(blocks[:, 1] <= blocks[:, 0]) or blocks[0, 0] < 0
                        or blocks[-1, 1] > recording_end_frame
                        or np.any(blocks[1:, 0] < blocks[:-1, 1])):
        raise ValueError("valid intervals must be ordered, disjoint and within the recording")
    calls = events.copy().reset_index(drop=True)
    for name in ("chain_id", "start_frame", "end_frame_exclusive", "selection_score"):
        if name not in calls:
            raise ValueError("missing call column: " + name)
    calls["call_id"] = np.arange(len(calls))
    if len(calls) and (not np.isfinite(calls.selection_score.to_numpy(dtype=float)).all()
                       or np.any(calls.end_frame_exclusive <= calls.start_frame)):
        raise ValueError("calls need finite scores and positive intervals")
    starts = calls.start_frame.to_numpy(dtype=np.int64)
    ends = calls.end_frame_exclusive.to_numpy(dtype=np.int64)
    block_ids = np.searchsorted(blocks[:, 0], starts, side="right") - 1 if len(blocks) else np.full(len(calls), -1)
    inside = block_ids >= 0
    if len(blocks):
        inside &= ends <= blocks[np.maximum(block_ids, 0), 1]
    calls["valid_block"] = np.where(inside, block_ids, -1)
    calls["selected"] = False
    calls["rejection_reason"] = "outside_valid_block"
    calls["overlap_with_call_id"] = -1
    occupied = {}
    priority = calls[inside].sort_values(
        ["selection_score", "chain_id", "start_frame", "end_frame_exclusive", "call_id"],
        ascending=[False, True, True, True, True], kind="stable")
    for row in priority.itertuples():
        chosen = occupied.setdefault(row.valid_block, [])
        position = bisect_left(chosen, (row.start_frame,))
        collision = None
        if position and chosen[position - 1][1] > row.start_frame:
            collision = chosen[position - 1][2]
        elif position < len(chosen) and chosen[position][0] < row.end_frame_exclusive:
            collision = chosen[position][2]
        if collision is None:
            chosen.insert(position, (row.start_frame, row.end_frame_exclusive, row.call_id))
            calls.loc[row.call_id, ["selected", "rejection_reason"]] = [True, "selected"]
        else:
            calls.loc[row.call_id, ["rejection_reason", "overlap_with_call_id"]] = ["overlap", collision]
    selected = calls[calls.selected].sort_values(["start_frame", "end_frame_exclusive", "chain_id"], kind="stable").reset_index(drop=True)
    spans = []
    cursor = 0
    for block, (start, end) in enumerate(blocks):
        if start > cursor:
            spans.append((cursor, start, "missing", -1, -1))
        cursor = start
        for row in selected[selected.valid_block == block].itertuples():
            if row.start_frame > cursor:
                spans.append((cursor, row.start_frame, "unmatched", block, -1))
            spans.append((row.start_frame, row.end_frame_exclusive, "call", block, row.call_id))
            cursor = row.end_frame_exclusive
        if cursor < end:
            spans.append((cursor, end, "unmatched", block, -1))
        cursor = end
    if cursor < recording_end_frame:
        spans.append((cursor, recording_end_frame, "missing", -1, -1))
    spans = pd.DataFrame(spans, columns=["start_frame", "end_frame_exclusive", "kind", "valid_block", "call_id"])
    spans["duration_s"] = (spans.end_frame_exclusive - spans.start_frame) / fps
    return selected, calls, spans


def organisation_arrays(selected, *, retained_ids, fps):
    """Freeze categorical arrays and reference blocks, keeping exclusions in place.

    Frequency reference blocks are contiguous valid segments separated by excluded
    calls, never by observed repetitions or merely by long unmatched intervals.
    """
    selected = selected.copy()
    kept = selected.chain_id.isin(retained_ids).to_numpy()
    ids, vocabulary = pd.factorize(selected.chain_id, sort=True)
    block = selected.valid_block.to_numpy(dtype=int)
    reference = np.full(len(selected), -1, dtype=int)
    current = -1
    for i in range(len(selected)):
        if kept[i]:
            if not i or not kept[i - 1] or block[i] != block[i - 1]:
                current += 1
            reference[i] = current
    selected["included"] = kept
    selected["reference_block"] = reference
    return {"events": selected, "ids": ids, "vocabulary": vocabulary.to_numpy(), "kept": kept,
            "valid_block": block, "reference": reference,
            "starts": selected.start_frame.to_numpy(dtype=np.int64),
            "ends": selected.end_frame_exclusive.to_numpy(dtype=np.int64), "fps": float(fps)}


def eligible_pairs(arrays, *, gap_seconds):
    """Mask every consecutive selected-call pair, before any event filtering."""
    if gap_seconds < 0:
        raise ValueError("gap threshold must be nonnegative")
    gap_frames = arrays["starts"][1:] - arrays["ends"][:-1]
    return (arrays["kept"][:-1] & arrays["kept"][1:]
            & (arrays["valid_block"][:-1] == arrays["valid_block"][1:])
            & (gap_frames >= 0) & (gap_frames <= gap_seconds * arrays["fps"] + 1e-9))


def pair_audit(arrays, *, gaps):
    events = arrays["events"]
    out = pd.DataFrame({"from_call_id": events.call_id.to_numpy()[:-1],
                        "to_call_id": events.call_id.to_numpy()[1:],
                        "from_chain_id": events.chain_id.to_numpy()[:-1],
                        "to_chain_id": events.chain_id.to_numpy()[1:],
                        "from_end_frame": arrays["ends"][:-1], "to_start_frame": arrays["starts"][1:],
                        "gap_s": (arrays["starts"][1:] - arrays["ends"][:-1]) / arrays["fps"],
                        "same_chain": arrays["ids"][:-1] == arrays["ids"][1:]})
    same_block = arrays["valid_block"][:-1] == arrays["valid_block"][1:]
    both = arrays["kept"][:-1] & arrays["kept"][1:]
    out["chain_only_eligible"] = same_block & both
    for gap in gaps:
        eligible = eligible_pairs(arrays, gap_seconds=gap)
        tag = f"g{gap:g}"
        out[tag + "_eligible"] = eligible
        out[tag + "_reason"] = np.select([~same_block, ~both, ~eligible],
                                          ["missing_boundary", "excluded_call", "excessive_gap"], default="eligible")
    return out


def return_masks(ids, pairs, lag, *, simple=False):
    """Fixed return opportunities and successes; no intervening return to start."""
    ids = np.asarray(ids)
    n = max(0, len(ids) - lag)
    opportunities = np.ones(n, dtype=bool)
    successes = np.ones(n, dtype=bool)
    for step in range(lag):
        opportunities &= pairs[step:step + n]
    if n:
        successes &= ids[:n] == ids[lag:lag + n]
        for step in range(1, lag):
            successes &= ids[step:step + n] != ids[:n]
        if simple:
            for i in range(1, lag):
                for j in range(i + 1, lag):
                    successes &= ids[i:i + n] != ids[j:j + n]
    return opportunities, successes & opportunities


def repetition_runs(arrays, *, gap_seconds):
    """Singleton-inclusive runs; unknown boundaries are censored, never switches."""
    ids, kept = arrays["ids"], arrays["kept"]
    pairs = eligible_pairs(arrays, gap_seconds=gap_seconds)
    self_pair = pairs & (ids[:-1] == ids[1:])
    indices = np.flatnonzero(kept)
    columns = ["run_id", "chain_id", "first_call_id", "last_call_id", "length_calls", "call_time_s",
               "left_censored", "right_censored", "observed_start", "observed_exit", "complete"]
    rows = []
    if not len(indices):
        return pd.DataFrame(columns=columns)
    starts = indices[~np.r_[False, self_pair][indices]]
    ends = indices[~np.r_[self_pair, False][indices]]
    events = arrays["events"]
    for run, (first, last) in enumerate(zip(starts, ends)):
        left = not (first > 0 and pairs[first - 1])
        right = not (last < len(pairs) and pairs[last])
        rows.append([run, events.chain_id.iloc[first], int(events.call_id.iloc[first]), int(events.call_id.iloc[last]),
                     last - first + 1, float(np.sum(arrays["ends"][first:last + 1] - arrays["starts"][first:last + 1]) / arrays["fps"]),
                     left, right, not left, not right, not left and not right])
    return pd.DataFrame(rows, columns=columns)


def organisation_metrics(arrays, *, gap_seconds, modelled_seconds):
    """Counts/sums/denominators suitable for pooling recordings within animal/wave."""
    ids, kept = arrays["ids"], arrays["kept"]
    pairs = eligible_pairs(arrays, gap_seconds=gap_seconds)
    runs = repetition_runs(arrays, gap_seconds=gap_seconds)
    n = int(kept.sum())
    count = np.bincount(ids[kept], minlength=len(arrays["vocabulary"])) if n else np.zeros(len(arrays["vocabulary"]))
    p = count[count > 0] / n if n else np.array([])
    chain_only = (kept[:-1] & kept[1:] & (arrays["valid_block"][:-1] == arrays["valid_block"][1:]))
    duration = (arrays["ends"] - arrays["starts"]) / arrays["fps"]
    result = {"modelled_seconds": float(modelled_seconds), "calls": n, "pair_opportunities": int(pairs.sum()),
              "self_pairs": int((pairs & (ids[:-1] == ids[1:])).sum()), "runs": len(runs),
              "chain_only_pair_opportunities": int(chain_only.sum()),
              "chain_only_self_pairs": int((chain_only & (ids[:-1] == ids[1:])).sum()),
              "run_call_sum": n, "covered_seconds": float(duration[kept].sum()),
              "repeated_seconds": float(runs.loc[runs.length_calls >= 2, "call_time_s"].sum()),
              "complete_runs": int(runs.complete.sum()), "complete_run_call_sum": int(runs.loc[runs.complete, "length_calls"].sum()),
              "left_censored_runs": int(runs.left_censored.sum()), "right_censored_runs": int(runs.right_censored.sum()),
              "observed_initiations": int(runs.observed_start.sum()), "observed_exits": int(runs.observed_exit.sum()),
              "longest_run": int(runs.length_calls.max()) if len(runs) else np.nan,
              "concentration": float(np.sum(p ** 2)) if n else np.nan,
              "effective_routines": float(np.exp(-np.sum(p * np.log(p)))) if n else np.nan}
    for lag in (2, 3, 4):
        opp, hit = return_masks(ids, pairs, lag)
        _, simple = return_masks(ids, pairs, lag, simple=True)
        result.update({f"return{lag}_opportunities": int(opp.sum()), f"return{lag}_hits": int(hit.sum()),
                       f"simple{lag}_hits": int(simple.sum())})
    from_ids, to_ids = ids[:-1][pairs], ids[1:][pairs]
    transitions = np.zeros((len(count), len(count)))
    np.add.at(transitions, (from_ids, to_ids), 1)
    total = transitions.sum()
    entropy = 0.
    for row in transitions:
        if row.sum():
            q = row[row > 0] / row.sum()
            entropy -= row.sum() / total * np.sum(q * np.log(q))
    result["transition_entropy_nats"] = entropy if total else np.nan
    return result, runs


def metric_values(counts):
    """Derived values; zero opportunities stay missing, positive-exposure rates may be zero."""
    def ratio(a, b):
        return counts.get(a, np.nan) / counts[b] if counts.get(b, 0) > 0 else np.nan

    out = {"self_probability": ratio("self_pairs", "pair_opportunities"),
           "chain_only_self_probability": ratio("chain_only_self_pairs", "chain_only_pair_opportunities"),
           "mean_run_calls": ratio("run_call_sum", "runs"), "complete_mean_run_calls": ratio("complete_run_call_sum", "complete_runs"),
           "repeated_call_fraction": ratio("self_pairs", "calls"), "initiations_per_call": ratio("runs", "calls"),
           "initiations_per_min": 60 * ratio("runs", "modelled_seconds"),
           "observed_initiations_per_min": 60 * ratio("observed_initiations", "modelled_seconds"),
           "calls_per_min": 60 * ratio("calls", "modelled_seconds"), "coverage_fraction": ratio("covered_seconds", "modelled_seconds"),
           "repeated_time_fraction": ratio("repeated_seconds", "modelled_seconds"),
           "repeated_covered_fraction": ratio("repeated_seconds", "covered_seconds"),
           "right_censored_fraction": ratio("right_censored_runs", "runs")}
    for lag in (2, 3, 4):
        out[f"return{lag}_probability"] = ratio(f"return{lag}_hits", f"return{lag}_opportunities")
        out[f"simple{lag}_probability"] = ratio(f"simple{lag}_hits", f"return{lag}_opportunities")
    out["self_excess"] = out["self_probability"] - ratio("frequency_expected_self_count", "pair_opportunities")
    out["return2_excess"] = out["return2_probability"] - ratio("frequency_expected_return2_count", "return2_opportunities")
    for name in ("longest_run", "concentration", "effective_routines", "transition_entropy_nats"):
        out[name] = counts.get(name, np.nan)
    return out
