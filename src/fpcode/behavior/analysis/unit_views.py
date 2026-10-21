"""Adapt common unit streams to the existing organisation measurements.

This is a representation bridge, not a new detector or statistical estimator.
The same consumer accepts primitive, episode and assembly catalogues; counts may
differ. Exclusive selection and membership exclusions remain explicit policies.
"""

from bisect import bisect_left, bisect_right
from collections import Counter
from numbers import Integral

import numpy as np
import pandas as pd

from fpcode.behavior.analysis.organisation import exclusive_timeline, organisation_arrays, organisation_metrics
from fpcode.behavior.compatibility import StreamRequirements, require_compatible
from fpcode.behavior.unit_adapters import occurrence_table
from fpcode.contracts.behavior_units import UnitStream


def organisation_view(stream: UnitStream, *, overlap_policy="reject", retained_type_ids=None):
    """Return (existing arrays, overlap audit), resolving before exclusions.

    ``reject`` accepts exclusive streams without needing comparable scores.
    ``score_ranked`` explicitly opts into the existing deterministic resolver and
    requires complete scores of one declared kind. Score comparability within a
    detector remains the caller's modelling assumption, not established here.
    """
    if overlap_policy not in {"reject", "score_ranked"}:
        raise ValueError("choose reject or score_ranked overlap policy explicitly")
    require_compatible(stream, StreamRequirements(exclusive=overlap_policy == "reject",
                                                 scored=overlap_policy == "score_ranked",
                                                 uniform_score_kind=overlap_policy == "score_ranked"))
    if any(span.stop > np.iinfo(np.int64).max for span in stream.valid_intervals):
        raise ValueError("existing organisation consumer requires native bounds within int64")
    table = occurrence_table(stream).rename(columns={"type_id": "chain_id"})
    table["selection_score"] = table.score if overlap_policy == "score_ranked" else 1.
    blocks = np.array([(span.start, span.stop) for span in stream.valid_intervals], dtype=np.int64).reshape(-1, 2)
    end = stream.valid_intervals[-1].stop if stream.valid_intervals else 0
    selected, decisions, _ = exclusive_timeline(table, blocks, fps=stream.fps, recording_end_frame=end)
    retained = {unit.type_id for unit in stream.catalog} if retained_type_ids is None else set(retained_type_ids)
    if retained - {unit.type_id for unit in stream.catalog}:
        raise ValueError("retained type identities must belong to the stream catalogue")
    return organisation_arrays(selected, retained_ids=retained, fps=stream.fps), decisions


def measure_unit_organisation(stream: UnitStream, *, gap_seconds, overlap_policy="reject", retained_type_ids=None):
    """Use existing sufficient counts on the stream's explicit modelled exposure."""
    arrays, decisions = organisation_view(stream, overlap_policy=overlap_policy, retained_type_ids=retained_type_ids)
    exposure = sum(span.stop - span.start for span in stream.valid_intervals) / stream.fps
    counts, runs = organisation_metrics(arrays, gap_seconds=gap_seconds, modelled_seconds=exposure)
    return {"counts": counts, "runs": runs, "selection_audit": decisions}


def _positive_integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(name + " must be a positive integer")


def measure_unit_composition(stream: UnitStream, source_stream: UnitStream, *, max_sequence_events=512):
    """Describe calls using an explicitly associated, exclusive lower vocabulary.

    These are categorical composition diagnostics, not semantic behaviour labels
    or a quality score. Repeated single-action routines can be useful. IDs retain
    their source model scope; changing lower resolution changes these metrics.
    Partial edge bouts, unassigned spans and declared validity boundaries remain
    visible. Sequences longer than the stated cap retain counts/support, with
    sequence-dependent measurements missing rather than silently truncated.
    """
    _positive_integer(max_sequence_events, "max_sequence_events")
    require_compatible(stream)
    require_compatible(source_stream, StreamRequirements(session=stream.session, clock=stream.clock,
                                                       fps=stream.fps, exclusive=True))
    lower = sorted(source_stream.occurrences, key=lambda event: (event.interval.start, event.interval.stop))
    starts = [event.interval.start for event in lower]
    stops = [event.interval.stop for event in lower]
    rows = []
    for call in stream.occurrences:
        start, stop = call.interval.start, call.interval.stop
        lo, hi = bisect_right(stops, start), bisect_left(starts, stop)
        children = lower[lo:hi]
        supported = sum(min(stop, c.interval.stop) - max(start, c.interval.start) for c in children)
        block = any(span.start <= start and stop <= span.stop for span in source_stream.valid_intervals)
        complete = bool(block and supported == stop - start)
        clipped = bool(children and (children[0].interval.start < start or children[-1].interval.stop > stop))
        sequence = tuple(c.type_id for c in children) if len(children) <= max_sequence_events else None
        distinct, entropy, period = np.nan, np.nan, np.nan
        if sequence:
            counts = np.array(list(Counter(sequence).values()), dtype=float)
            probabilities = counts / counts.sum()
            distinct, entropy = len(counts), float(-np.sum(probabilities * np.log2(probabilities)))
            period = next((p for p in range(1, len(sequence) + 1)
                           if len(sequence) % p == 0 and sequence == sequence[:p] * (len(sequence) // p)), len(sequence))
        rows.append(dict(study=stream.session.study, recording=stream.session.recording, subject=stream.session.subject,
                         model_id=stream.model_id, source_model_id=source_stream.model_id,
                         occurrence_id=call.occurrence_id, type_id=call.type_id,
                         start_frame=start, stop_frame=stop, duration_s=(stop-start)/stream.fps,
                         lower_events=len(children), lower_support_fraction=supported/(stop-start),
                         lower_support_complete=complete, within_one_lower_valid_block=block,
                         edge_bout_clipped=clipped, sequence_over_budget=sequence is None,
                         source_sequence=sequence, distinct_lower_types=distinct, event_entropy_bits=entropy,
                         shortest_repeat_period_events=period,
                         exact_repeat_copies=len(sequence)/period if sequence else np.nan))
    columns = ["study", "recording", "subject", "model_id", "source_model_id", "occurrence_id", "type_id",
               "start_frame", "stop_frame", "duration_s", "lower_events", "lower_support_fraction",
               "lower_support_complete", "within_one_lower_valid_block", "edge_bout_clipped",
               "sequence_over_budget", "source_sequence", "distinct_lower_types", "event_entropy_bits",
               "shortest_repeat_period_events", "exact_repeat_copies"]
    table = pd.DataFrame(rows, columns=columns)
    table.attrs["composition_scope"] = dict(model_id=stream.model_id, source_model_id=source_stream.model_id,
                                             catalog=stream.catalog, source_catalog=source_stream.catalog,
                                             clock=stream.clock, fps=stream.fps, max_sequence_events=int(max_sequence_events))
    return table


def _sample_pairs(first, second, *, same, limit, rng):
    """Bound pair storage without constructing an occurrence-square matrix."""
    n, m = len(first), len(second)
    total = n * (n-1) // 2 if same else n*m
    if total <= limit:
        return [(first[i], second[j]) for i in range(n) for j in range(i+1 if same else 0, m)]
    selected = set()
    while len(selected) < limit:
        i, j = int(rng.integers(n)), int(rng.integers(m))
        if same:
            if i == j:
                continue
            i, j = min(i, j), max(i, j)
        selected.add((i, j))
    return [(first[i], second[j]) for i, j in sorted(selected)]


def _sequence_agreement(first, second):
    """One minus Levenshtein distance divided by the longer event sequence."""
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    return 1. - previous[-1] / max(len(first), len(second))


def aggregate_unit_composition(tables, *, max_pairs_per_type=128, seed=0):
    """Return separate support, repetition and sequence-grouping diagnostic panels.

    Consume in-memory tables from :func:`measure_unit_composition`, retaining the
    catalogue metadata in their attrs. Pool only within identical detector and
    source model identities/catalogues. Pair comparisons use fully supported,
    unclipped, within-budget bouts and never compare IDs from different source
    vocabularies. Per-type within/other-type pairs are uniformly sampled up to the
    explicit cap; seed and realised pair counts accompany every estimate. Similar
    sequence composition does not establish similar physical execution or video
    coherence. Sparse types and one-animal support describe fragmentation, without
    declaring a universal preferred number of routines. Estimates weight calls,
    not animals; pair counts are not independent biological replicates. Overlapping
    candidates remain present unless the caller supplies a selected stream.
    """
    _positive_integer(max_pairs_per_type, "max_pairs_per_type")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    scopes, groups = {}, {}
    for table in tables:
        scope = table.attrs.get("composition_scope")
        if scope is None:
            raise ValueError("composition tables must retain catalogue scope metadata")
        key = (scope["model_id"], scope["source_model_id"])
        if key in scopes and scopes[key] != scope:
            raise ValueError("same model identities require identical catalogues, clock/fps and sequence budget")
        scopes[key] = scope
        groups.setdefault(key, []).append(table)
    rng = np.random.default_rng(int(seed))
    type_rows, catalogue_rows = [], []
    for key in sorted(groups):
        nonempty = [table for table in groups[key] if len(table)]
        table = pd.concat(nonempty, ignore_index=True) if nonempty else groups[key][0].copy()
        identities = ["study", "recording", "occurrence_id"]
        if table.duplicated(identities).any():
            raise ValueError("duplicate scoped occurrence observations cannot be pooled")
        if len(table) and (set(table.model_id) != {key[0]} or set(table.source_model_id) != {key[1]}):
            raise ValueError("row model identities disagree with catalogue scope")
        good = table[table.lower_support_complete.astype(bool) & ~table.edge_bout_clipped.astype(bool)
                     & ~table.sequence_over_budget.astype(bool)]
        observed = table.type_id.value_counts()
        catalogue_rows.append(dict(model_id=key[0], source_model_id=key[1], occurrences=len(table),
                                   declared_types=len(scopes[key]["catalog"]), observed_types=len(observed),
                                   singleton_types=int((observed == 1).sum()),
                                   concentration=float(np.sum((observed/len(table))**2)) if len(table) else np.nan,
                                   comparable_occurrences=len(good), max_pairs_per_type=int(max_pairs_per_type),
                                   max_sequence_events=scopes[key]["max_sequence_events"], seed=int(seed)))
        for unit in sorted(scopes[key]["catalog"], key=lambda unit: unit.type_id):
            part, comparable = table[table.type_id == unit.type_id], good[good.type_id == unit.type_id]
            own = comparable.source_sequence.tolist()
            other = good[good.type_id != unit.type_id].source_sequence.tolist()
            within = _sample_pairs(own, own, same=True, limit=max_pairs_per_type, rng=rng)
            between = _sample_pairs(own, other, same=False, limit=max_pairs_per_type, rng=rng)
            known = part[part.subject.notna()][["study", "subject"]].drop_duplicates()
            row = dict(model_id=key[0], source_model_id=key[1], type_id=unit.type_id,
                       occurrences=len(part), recordings=len(part[["study", "recording"]].drop_duplicates()),
                       known_animals=len(known), unknown_subject_occurrences=int(part.subject.isna().sum()),
                       comparable_occurrences=len(own), unsupported_occurrences=int((~part.lower_support_complete.astype(bool)).sum()),
                       clipped_edge_occurrences=int(part.edge_bout_clipped.sum()),
                       over_budget_occurrences=int(part.sequence_over_budget.sum()),
                       exact_sequence_variants=len(set(own)), within_pairs=len(within), other_type_pairs=len(between),
                       mean_within_sequence_agreement=float(np.mean([_sequence_agreement(a, b) for a, b in within])) if within else np.nan,
                       mean_other_type_sequence_agreement=float(np.mean([_sequence_agreement(a, b) for a, b in between])) if between else np.nan,
                       mean_distinct_lower_types=float(comparable.distinct_lower_types.mean()),
                       mean_exact_repeat_copies=float(comparable.exact_repeat_copies.mean()),
                       duration_cv=float(part.duration_s.std(ddof=1)/part.duration_s.mean()) if len(part) > 1 else np.nan,
                       max_pairs_per_type=int(max_pairs_per_type), max_sequence_events=scopes[key]["max_sequence_events"], seed=int(seed))
            type_rows.append(row)
    return {"types": pd.DataFrame(type_rows), "catalogues": pd.DataFrame(catalogue_rows)}
