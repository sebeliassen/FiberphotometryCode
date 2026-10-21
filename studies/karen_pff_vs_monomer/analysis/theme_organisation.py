"""Auditable within-session organisation of a small exploratory theme atlas.

Native intervals are half open. All calls enter deterministic overlap selection
before optional atlas membership filtering. Rejected calls remain in the decision
log; excluded selected calls and invalid frame blocks remain succession barriers.
A nearby succession means adjacent selected calls separated by at most the stated
number of seconds. Unmatched valid time is retained in the exposure/timeline.
This descriptive wrapper performs neither model fitting nor group inference.
"""

import numpy as np
import pandas as pd

from fpcode.behavior.sequences.organisation import (
    eligible_pairs,
    exclusive_timeline,
    organisation_arrays,
    organisation_metrics,
    pair_audit,
    return_masks,
)
from fpcode.statistics.chain_nulls import frequency_expectations

CATALOGUE = "__catalogue__"
_REQUIRED = ("occurrence_id", "theme", "start_frame", "end_frame_exclusive", "score")


def _frames(values, label):
    """Refuse silent native-clock truncation before entering shared functions."""
    a = np.asarray(values)
    if a.size and (not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all()
                   or np.any(a < 0) or np.any(a != np.floor(a))
                   or np.any(a >= 2 ** 63)):
        raise ValueError(f"{label} must contain finite nonnegative integer frames")
    return a.astype(np.int64)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan


def _recurrences(arrays, theme):
    """End-to-next-start latencies, including intervening themes/unmatched time.

    Reference blocks are defined by validity and excluded selected calls, never by
    theme identity or the observed recurrence. These are elapsed recurrence times,
    not claims that separated events form an uninterrupted chain of themes.
    """
    events = arrays["events"]
    where = np.flatnonzero(arrays["kept"] & (events.chain_id.to_numpy() == theme))
    if len(where) < 2:
        return np.empty(0)
    same = arrays["reference"][where[:-1]] == arrays["reference"][where[1:]]
    return ((arrays["starts"][where[1:]] - arrays["ends"][where[:-1]])
            / arrays["fps"])[same]


def session_readouts(calls, valid_intervals, fps=60, *, gap_seconds=(1., 0., 2.),
                     recording_end_frame=None, theme_ids=None):
    """Return long readouts plus every overlap, boundary, pair and run decision.

    ``calls`` has occurrence_id, theme, start_frame, end_frame_exclusive and score.
    Unique occurrence IDs make overlap ties reproducible under input row reorder.
    Higher score wins; subsequent priority is the shared lexical theme/start/end
    policy. ``theme_ids`` fixes the atlas across sessions (including absent themes)
    and, if provided, acts as a retained-ID list *after* overlap selection.

    Result keys: readouts, decisions, selected, timeline, pairs, runs, summary.
    Readouts have feature/family/theme/value/numerator/denominator/gap_seconds.
    The 1-second setting is primary by default; 0/2 seconds are sensitivity checks.
    Zero opportunities are NaN; counts/rates can be zero with positive exposure.
    Numerators and denominators permit appropriate pooling of ratio features.
    Maxima and nonlinear catalogue descriptors have NaN sufficient statistics and
    require recomputation rather than ratio pooling. No p-values are generated.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    calls = pd.DataFrame(calls).copy()
    for key in _REQUIRED:
        if key not in calls:
            if len(calls):
                raise ValueError("missing call column: " + key)
            calls[key] = pd.Series(dtype=float if key == "score" else object)
    for key in ("occurrence_id", "theme"):
        if calls[key].isna().any() or any(not str(x) for x in calls[key]):
            raise ValueError(f"{key} needs nonempty identities")
        calls[key] = calls[key].astype(str)
    if calls.occurrence_id.duplicated().any():
        raise ValueError("occurrence_id must be unique within session")
    if (calls.theme == CATALOGUE).any():
        raise ValueError("reserved catalogue theme identity")
    calls["score"] = calls.score.astype(float)
    calls["start_frame"] = _frames(calls.start_frame.to_numpy(), "call starts")
    calls["end_frame_exclusive"] = _frames(calls.end_frame_exclusive.to_numpy(), "call ends")
    blocks = np.asarray(valid_intervals)
    if not blocks.size:
        blocks = np.empty((0, 2), dtype=np.int64)
    if blocks.ndim != 2 or blocks.shape[1] != 2:
        raise ValueError("valid_intervals must have shape (n, 2)")
    blocks = _frames(blocks, "valid intervals")
    if recording_end_frame is None:
        recording_end_frame = int(blocks[-1, 1]) if len(blocks) else 0
    recording_end_frame = int(_frames(np.asarray([recording_end_frame]), "recording end")[0])
    gaps = tuple(dict.fromkeys(float(g) for g in np.atleast_1d(gap_seconds)))
    if not gaps or any(not np.isfinite(g) or g < 0 for g in gaps):
        raise ValueError("gap thresholds must be finite, nonnegative and nonempty")
    atlas = sorted(calls.theme.unique()) if theme_ids is None else [str(x) for x in theme_ids]
    if len(set(atlas)) != len(atlas) or any(not x or x == CATALOGUE for x in atlas):
        raise ValueError("theme_ids must have unique nonempty, nonreserved IDs")
    # Last tie-break of the shared selector is input row, so freeze this by ID.
    events = calls.sort_values("occurrence_id", kind="stable").reset_index(drop=True)
    events["chain_id"] = events.theme
    events["selection_score"] = events.score
    selected, decisions, timeline = exclusive_timeline(
        events, blocks, fps=fps, recording_end_frame=recording_end_frame)
    arrays = organisation_arrays(selected, retained_ids=atlas, fps=fps)
    selected = arrays["events"]
    decisions["included"] = decisions.selected & decisions.chain_id.isin(atlas)
    decisions["analysis_reason"] = np.where(
        decisions.selected & ~decisions.chain_id.isin(atlas), "excluded_theme", decisions.rejection_reason)
    decision_index = decisions.set_index("call_id")
    timeline["theme"] = timeline.call_id.map(decision_index.chain_id)
    timeline["included"] = timeline.call_id.map(decision_index.included).eq(True)
    pairs_table = pair_audit(arrays, gaps=gaps)
    by_id = decisions.set_index("call_id").occurrence_id
    for side in ("from", "to"):
        pairs_table[side + "_occurrence_id"] = pairs_table[side + "_call_id"].map(by_id)
    seconds = float(np.sum(blocks[:, 1] - blocks[:, 0]) / fps)
    ids, kept = arrays["ids"], arrays["kept"]
    names = selected.chain_id.to_numpy()
    durations = (arrays["ends"] - arrays["starts"]) / fps
    rows, all_runs = [], []

    def add(feature, family, theme, numerator, denominator, gap=np.nan, *, value=None):
        rows.append({"feature": feature, "family": family, "theme": theme,
                     "value": _ratio(numerator, denominator) if value is None else float(value),
                     "numerator": float(numerator), "denominator": float(denominator),
                     "gap_seconds": float(gap)})

    for theme in [CATALOGUE, *atlas]:
        mask = kept if theme == CATALOGUE else kept & (names == theme)
        n = int(mask.sum())
        add("exclusive_call_count", "recruitment", theme, n, 1)
        add("exclusive_calls_per_min", "recruitment", theme, n, seconds / 60)
        add("exclusive_coverage_fraction", "recruitment", theme, durations[mask].sum(), seconds)
        add("exclusive_mean_duration_s", "recruitment", theme, durations[mask].sum(), n)
        if theme != CATALOGUE:
            recurrences = _recurrences(arrays, theme)
            add("within_valid_block_recurrence_gap_s", "organisation", theme,
                recurrences.sum(), len(recurrences))

    for gap in gaps:
        pairs = eligible_pairs(arrays, gap_seconds=gap)
        counts, runs = organisation_metrics(arrays, gap_seconds=gap, modelled_seconds=seconds)
        frequency = frequency_expectations(ids, arrays["reference"], pairs)
        runs = runs.copy()
        runs["gap_seconds"] = gap
        for side in ("first", "last"):
            runs[side + "_occurrence_id"] = runs[side + "_call_id"].map(by_id)
        all_runs.append(runs)
        for theme in [CATALOGUE, *atlas]:
            origin = np.ones(max(0, len(names) - 1), dtype=bool) if theme == CATALOGUE else names[:-1] == theme
            opportunities = pairs & origin
            equal = ids[:-1] == ids[1:]
            denominator = int(opportunities.sum())
            hits = int((opportunities & equal).sum())
            add("self_repeat_probability", "organisation", theme, hits, denominator, gap)
            add("switch_probability", "organisation", theme, denominator - hits, denominator, gap)
            relevant = runs if theme == CATALOGUE else runs[runs.chain_id == theme]
            add("mean_run_calls", "organisation", theme, relevant.length_calls.sum(), len(relevant), gap)
            complete = relevant[relevant.complete.astype(bool)]
            add("complete_mean_run_calls", "organisation", theme, complete.length_calls.sum(), len(complete), gap)
            add("run_segments_per_min", "organisation", theme, len(relevant), seconds / 60, gap)
            add("observed_entries_per_min", "organisation", theme, relevant.observed_start.sum(), seconds / 60, gap)
            add("right_censored_run_fraction", "organisation", theme, relevant.right_censored.sum(), len(relevant), gap)
            longest = relevant.length_calls.max() if len(relevant) else np.nan
            add("longest_run_calls", "organisation", theme, np.nan, np.nan, gap, value=longest)
            for lag in (2, 3):
                opp, success = return_masks(ids, pairs, lag)
                if theme != CATALOGUE:
                    origin = names[:len(opp)] == theme
                    opp, success = opp & origin, success & origin
                add(f"return{lag}_probability", "organisation", theme, success.sum(), opp.sum(), gap)
            if theme != CATALOGUE:
                # Denominator includes self transitions, not only observed switches.
                for target in atlas:
                    to_target = names[1:] == target
                    add("transition_to:" + target, "organisation", theme,
                        (opportunities & to_target).sum(), denominator, gap)
        for metric, numerator, denominator, observed in (
            ("self", frequency["frequency_expected_self_count"], counts["pair_opportunities"], counts["self_pairs"]),
            ("return2", frequency["frequency_expected_return2_count"], counts["return2_opportunities"], counts["return2_hits"]),
        ):
            add("frequency_expected_" + metric + "_probability", "frequency_control", CATALOGUE,
                numerator, denominator, gap)
            add("frequency_" + metric + "_excess", "frequency_control", CATALOGUE,
                observed - numerator, denominator, gap)
        for metric in ("concentration", "effective_routines", "transition_entropy_nats"):
            add(metric, "organisation", CATALOGUE, np.nan, np.nan, gap, value=counts[metric])

    summary = {
        "fps": float(fps), "valid_seconds": seconds, "valid_blocks": len(blocks),
        "input_calls": len(calls), "selected_calls": len(selected),
        "included_calls": int(kept.sum()), "atlas": atlas, "gaps_seconds": gaps,
        "primary_gap_seconds": gaps[0],
        "overlap_policy": "higher score, lexical theme, start/end, lexical occurrence_id",
        "frequency_control": "exact finite-population label permutation within validity/exclusion reference blocks; fixed gap slots",
        "run_policy": "singletons included; unknown boundaries censored; observed entries require an eligible different predecessor",
        "recurrence_policy": "same-theme end-to-next-start within a validity/exclusion reference block; may span unmatched valid time and intervening themes",
        "scope": "descriptive exclusive event projection; original overlapping calls retained in decisions",
    }
    return {"readouts": pd.DataFrame(rows), "decisions": decisions, "selected": selected,
            "timeline": timeline, "pairs": pairs_table, "runs": pd.concat(all_runs, ignore_index=True),
            "summary": summary}


def first_order_readouts(result, *, replicates=199, seed=43, memory_mb=64, gap_seconds=1.):
    """Optional bounded first-order diagnostic on this session's frozen slots.

    Not run by ``session_readouts``. The plug-in transition MLE uses this same
    session's eligible pairs; disconnected starts stay fixed and unseen rows use
    empirical theme frequency. This controls for first-order succession but does
    not integrate transition-fit uncertainty. A null p-value is a descriptive
    within-session diagnostic, not independent biological replication.
    """
    from fpcode.behavior.sequences.organisation import metric_values
    from fpcode.statistics.chain_nulls import simulate_chain_null

    if not np.isfinite(gap_seconds) or gap_seconds < 0:
        raise ValueError("gap_seconds must be finite and nonnegative")
    arrays = organisation_arrays(result["selected"], retained_ids=result["summary"]["atlas"],
                                 fps=result["summary"]["fps"])
    pairs = eligible_pairs(arrays, gap_seconds=gap_seconds)
    counts, _ = organisation_metrics(arrays, gap_seconds=gap_seconds,
                                    modelled_seconds=result["summary"]["valid_seconds"])
    observed = metric_values(counts)
    null = simulate_chain_null(arrays, pairs=pairs, kind="first_order", observed=observed,
                               seed=seed, replicates=replicates, memory_mb=memory_mb)
    null = null[null.metric.isin(["return2_probability", "return3_probability"])].copy()
    null["gap_seconds"] = gap_seconds
    rows = []
    for entry in null.to_dict("records"):
        rows.append({"feature": "first_order_" + entry["metric"].replace("_probability", "_excess"),
                     "family": "first_order_control", "theme": CATALOGUE,
                     "value": entry.get("excess", np.nan), "numerator": np.nan,
                     "denominator": np.nan, "gap_seconds": float(gap_seconds),
                     "observed": entry["observed"], "null_mean": entry.get("null_mean", np.nan),
                     "null_low": entry.get("null_low", np.nan), "null_high": entry.get("null_high", np.nan),
                     "replicates": entry["replicates"], "mc_mean_se": entry.get("mc_mean_se", np.nan),
                     "p_surrogate_two_sided": entry.get("p_surrogate_two_sided", np.nan)})
    return {"readouts": pd.DataFrame(rows), "null_summary": null,
            "summary": {"seed": int(seed), "requested_replicates": int(replicates),
                        "memory_mb": float(memory_mb), "gap_seconds": float(gap_seconds),
                        "fit": "within-session eligible transition MLE; disconnected initial IDs fixed",
                        "limitation": "plug-in descriptive null; no transition-fit uncertainty or biological replication"}}


def context_readouts(selected_withmetrics, valid_intervals, fps=60, *, gap_seconds=1.,
                     min_per_context=3, metrics=("duration_s", "speed"), recording_end_frame=None):
    """Current-theme execution after a nearby same versus different predecessor.

    Input must already be exclusive selected calls; overlaps/invalid-block calls
    raise rather than silently choosing a new event catalogue. Optional ``included``
    flags preserve excluded selected themes as adjacency barriers. Each metric
    needs at least ``min_per_context`` finite executions in *both* contexts. Rows
    with missing metrics/support are explicit and have a NaN contrast. Native
    duration_s is derived from endpoints; other columns keep caller-supplied units.
    This association is descriptive, with no causal or independent-trial claim.
    """
    if not isinstance(min_per_context, (int, np.integer)) or min_per_context < 1:
        raise ValueError("min_per_context must be a positive integer")
    events = pd.DataFrame(selected_withmetrics).copy()
    for key in ("theme", "start_frame", "end_frame_exclusive"):
        if key not in events:
            if len(events):
                raise ValueError("missing selected event column: " + key)
            events[key] = pd.Series(dtype=object)
    if "occurrence_id" not in events:
        events["occurrence_id"] = [f"selected-{i}" for i in range(len(events))]
    if "score" not in events:
        events["score"] = events.get("selection_score", 1.)
    atlas = sorted(events.theme.astype(str).unique())
    if "included" in events:
        if events.included.isna().any() or not events.included.map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError("included must contain explicit booleans")
        if (events.groupby("theme").included.nunique() > 1).any():
            raise ValueError("included membership must be consistent within each theme")
        atlas = sorted(events.loc[events.included, "theme"].astype(str).unique())
    projection = session_readouts(events, valid_intervals, fps=fps, gap_seconds=(gap_seconds,),
                                 recording_end_frame=recording_end_frame, theme_ids=atlas)
    if not projection["decisions"].selected.all():
        raise ValueError("context input must already be exclusive and within valid blocks")
    arrays = organisation_arrays(projection["selected"], retained_ids=atlas, fps=fps)
    ordered = arrays["events"]
    ordered["duration_s"] = (arrays["ends"] - arrays["starts"]) / fps
    pairs = eligible_pairs(arrays, gap_seconds=gap_seconds)
    names = ordered.chain_id.to_numpy()
    same = names[:-1] == names[1:]
    rows = []
    for theme in atlas:
        current_theme = names[1:] == theme
        for metric in dict.fromkeys(metrics):
            missing = metric not in ordered
            values = np.full(len(ordered), np.nan) if missing else ordered[metric].to_numpy(dtype=float)
            finite = np.isfinite(values[1:])
            same_values = values[1:][pairs & current_theme & same & finite]
            other_values = values[1:][pairs & current_theme & ~same & finite]
            supported = min(len(same_values), len(other_values)) >= min_per_context
            mean_same = float(same_values.mean()) if len(same_values) else np.nan
            mean_other = float(other_values.mean()) if len(other_values) else np.nan
            rows.append({"feature": "preceding_same_minus_other:" + str(metric), "family": "context",
                         "theme": theme, "value": mean_same - mean_other if supported else np.nan,
                         "numerator": np.nan, "denominator": np.nan, "gap_seconds": float(gap_seconds),
                         "n_same": len(same_values), "n_other": len(other_values),
                         "sum_same": float(same_values.sum()), "sum_other": float(other_values.sum()),
                         "mean_same": mean_same, "mean_other": mean_other,
                         "minimum_per_context": min_per_context,
                         "status": "missing_metric" if missing else "supported" if supported else "insufficient_support"})
    columns = ["feature", "family", "theme", "value", "numerator", "denominator", "gap_seconds",
               "n_same", "n_other", "sum_same", "sum_other", "mean_same", "mean_other",
               "minimum_per_context", "status"]
    return pd.DataFrame(rows, columns=columns)
