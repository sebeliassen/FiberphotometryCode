"""Adapt common unit streams to the existing organisation measurements.

This is a representation bridge, not a new detector or statistical estimator.
The same consumer accepts primitive, episode and assembly catalogues; counts may
differ. Exclusive selection and membership exclusions remain explicit policies.
"""

import numpy as np

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
