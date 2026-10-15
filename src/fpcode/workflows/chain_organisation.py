"""Reusable measurement workflow for explicit frozen chain-library policies."""

from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.behavior.analysis.organisation import (
    eligible_pairs,
    exclusive_timeline,
    metric_values,
    organisation_arrays,
    organisation_metrics,
    pair_audit,
    return_masks,
)
from fpcode.statistics.chain_nulls import frequency_expectations, simulate_chain_null


def _write(table, path):
    if len(table.columns):
        table.to_csv(path, index=False)


def measure_library_session(events, catalog, valid_intervals, *, fps, recording_end_frame,
                            output, gaps, primary_gap, seed, replicates, memory_mb,
                            detailed_sensitivity=False):
    """Measure one library/session; policy and inputs are supplied by its caller.

    Analytic frequency expectations cover all gaps. Batched frequency intervals
    cover the primary gap in each scope; first-order, time-quarter and leave-chain
    diagnostics are requested explicitly for the primary library by the caller.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seconds = np.sum(valid_intervals[:, 1] - valid_intervals[:, 0]) / fps
    selected, decisions, spans = exclusive_timeline(events, valid_intervals, fps=fps, recording_end_frame=recording_end_frame)
    _write(decisions, output / "call_decisions.csv.gz")
    _write(spans, output / "interval_timeline.csv.gz")
    rows, nulls, contributions, exclusions, quarters, exits = [], [], [], [], [], []
    pairs_tables, runs_tables, selected_tables, windows = [], [], [], []
    for scope in ("inclusive", "interpretable"):
        keep_ids = catalog.chain_id if scope == "inclusive" else catalog.loc[catalog.interpretable, "chain_id"]
        arrays = organisation_arrays(selected, retained_ids=set(keep_ids), fps=fps)
        pair_table = pair_audit(arrays, gaps=gaps)
        pair_table["scope"] = scope
        pairs_tables.append(pair_table)
        arrays["events"]["scope"] = scope
        selected_tables.append(arrays["events"])
        for gap in gaps:
            counts, runs = organisation_metrics(arrays, gap_seconds=gap, modelled_seconds=seconds)
            pairs = eligible_pairs(arrays, gap_seconds=gap)
            counts.update(frequency_expectations(arrays["ids"], arrays["reference"], pairs))
            row = {"scope": scope, "gap_seconds": gap, **counts, **metric_values(counts)}
            rows.append(row)
            runs["scope"], runs["gap_seconds"] = scope, gap
            runs_tables.append(runs)
            if gap != primary_gap:
                continue
            simulated = simulate_chain_null(arrays, pairs=pairs, kind="frequency", observed=row,
                                            seed=seed, replicates=replicates, memory_mb=memory_mb)
            simulated["scope"], simulated["gap_seconds"] = scope, gap
            nulls.append(simulated)
            events_ids = arrays["events"].chain_id.to_numpy()
            for chain in keep_ids:
                from_chain = pairs & (events_ids[:-1] == chain)
                contributions.append({"scope": scope, "chain_id": chain, "calls": int(np.sum(arrays["kept"] & (events_ids == chain))),
                                      "pair_opportunities": int(from_chain.sum()),
                                      "self_pairs": int((from_chain & (events_ids[:-1] == events_ids[1:])).sum())})
            for chain in keep_ids:
                dropped = organisation_arrays(selected, retained_ids=set(keep_ids) - {chain}, fps=fps)
                drop_counts, _ = organisation_metrics(dropped, gap_seconds=gap, modelled_seconds=seconds)
                drop_counts.update(frequency_expectations(dropped["ids"], dropped["reference"], eligible_pairs(dropped, gap_seconds=gap)))
                exclusions.append({"scope": scope, "removed_chain": chain, **drop_counts, **metric_values(drop_counts)})
            if not detailed_sensitivity:
                continue
            markov = simulate_chain_null(arrays, pairs=pairs, kind="first_order", observed=row,
                                         seed=seed + 1, replicates=replicates, memory_mb=memory_mb)
            markov["scope"], markov["gap_seconds"] = scope, gap
            nulls.append(markov)
            quarter = np.minimum(3, ((arrays["starts"] + arrays["ends"]) / 2 / max(1, recording_end_frame) * 4).astype(int))
            quarter_arrays = arrays.copy()
            # Cross-quarter pairs are omitted; counts remain fixed within original
            # reference blocks AND time quarters. No excluded-call barrier is lost.
            quarter_arrays["valid_block"] = arrays["valid_block"] * 4 + quarter
            quarter_arrays["reference"] = arrays["reference"] * 4 + quarter
            quarter_arrays["reference"][~arrays["kept"]] = -1
            qp = eligible_pairs(quarter_arrays, gap_seconds=gap)
            qc, _ = organisation_metrics(quarter_arrays, gap_seconds=gap, modelled_seconds=seconds)
            qc.update(frequency_expectations(arrays["ids"], quarter_arrays["reference"], qp))
            quarters.append({"scope": scope, "quarter": "stratified_null", **qc, **metric_values(qc)})
            for q in range(4):
                keep = arrays["events"].copy()
                # Assign calls by midpoint, keeping outside-quarter calls as barriers.
                keep.loc[quarter != q, "chain_id"] = "__outside_time_quarter__"
                qa = organisation_arrays(keep, retained_ids=set(keep_ids), fps=fps)
                a, b = q * recording_end_frame / 4, (q + 1) * recording_end_frame / 4
                exposure = np.maximum(0, np.minimum(valid_intervals[:, 1], b) - np.maximum(valid_intervals[:, 0], a)).sum() / fps
                qc, _ = organisation_metrics(qa, gap_seconds=gap, modelled_seconds=exposure)
                qc.update(frequency_expectations(qa["ids"], qa["reference"], eligible_pairs(qa, gap_seconds=gap)))
                quarters.append({"scope": scope, "quarter": str(q + 1), **qc, **metric_values(qc)})
            for lag in (2, 3, 4):
                opp, hit = return_masks(arrays["ids"], pairs, lag)
                _, simple = return_masks(arrays["ids"], pairs, lag, simple=True)
                for i in np.flatnonzero(opp):
                    windows.append({"scope": scope, "lag": lag, "first_call_id": int(arrays["events"].call_id.iloc[i]),
                                    "last_call_id": int(arrays["events"].call_id.iloc[i + lag]),
                                    "returned": bool(hit[i]), "simple_cycle": bool(simple[i]),
                                    "chain_ids": " ".join(events_ids[i:i + lag + 1])})
            positions = np.zeros(len(events_ids), dtype=int)
            for i in np.flatnonzero(arrays["kept"]):
                positions[i] = positions[i - 1] + 1 if i and pairs[i - 1] and arrays["ids"][i] == arrays["ids"][i - 1] else 1
            for position in np.unique(positions[positions > 0]):
                observed = pairs & (positions[:-1] == position)
                exits.append({"scope": scope, "repeat_position": int(position), "observed_successors": int(observed.sum()),
                              "switches": int((observed & (arrays["ids"][:-1] != arrays["ids"][1:])).sum()),
                              "censored_calls": int(np.sum((positions == position) & ~np.r_[pairs, False]))})
    _write(pd.concat(pairs_tables, ignore_index=True), output / "pairs.csv.gz")
    nonempty_runs = [table for table in runs_tables if len(table)]
    _write(pd.concat(nonempty_runs, ignore_index=True) if nonempty_runs else runs_tables[0], output / "runs.csv.gz")
    _write(pd.concat(selected_tables, ignore_index=True), output / "selected_calls.csv.gz")
    _write(pd.DataFrame(windows), output / "return_windows.csv.gz")
    return {"session_metrics": pd.DataFrame(rows), "null_summaries": pd.concat(nulls, ignore_index=True),
            "chain_contributions": pd.DataFrame(contributions), "leave_chain_out": pd.DataFrame(exclusions),
            "time_quarters": pd.DataFrame(quarters), "exit_curves": pd.DataFrame(exits),
            "selection_audit": {"input_calls": len(events), "selected_calls": len(selected),
                                "overlap_rejections": int((decisions.rejection_reason == "overlap").sum()),
                                "invalid_rejections": int((decisions.rejection_reason == "outside_valid_block").sum())}}


def pool_organisation_measurements(table, *, identity):
    """Pool sufficient counts within the caller's independent observation identity.

    Probabilities use summed numerators/denominators, rates exposure, run means
    run counts. Nonadditive diagnostics use equal-recording means; longest run
    uses maximum and is explicitly descriptive. Source table retains every count.
    """
    values = set(metric_values({}).keys())
    nonadditive = {"longest_run", "concentration", "effective_routines", "transition_entropy_nats"}
    counts = [c for c in table.select_dtypes(include="number").columns if c not in set(identity) | values | nonadditive]
    pooled = table.groupby(identity, dropna=False, as_index=False)[counts].sum(min_count=1)
    for measure in nonadditive:
        if measure in table:
            aggregate = table.groupby(identity, dropna=False)[measure].agg("max" if measure == "longest_run" else "mean").reset_index()
            pooled = pooled.merge(aggregate, on=identity, how="left")
    derived = pd.DataFrame([metric_values(row) for row in pooled.to_dict("records")])
    for name in derived:
        pooled[name] = derived[name].to_numpy()
    attendance = table.groupby(identity, dropna=False).size().rename("n_recordings").reset_index()
    return pooled.merge(attendance, on=identity)
