"""Study readouts on frozen, native whole-bout calls; no fitting or file I/O.

Local type IDs remain local to the supplied model. Inter controls condition on
fixed selected-call slots, not on simulated detector input. Partial human labels
describe primitives, not independently annotated chain ground truth.
"""
from __future__ import annotations

import math
from collections import Counter

EXECUTION_FEATURES = ("duration_s", "mean_speed_px_s", "net_speed_px_s", "path_length_px",
                      "heading_change_rad", "absolute_heading_change_rad", "latent_dispersion")


def epochs(record, metadata, *, baseline_seconds=900, post_guard_seconds=120):
    """Explicit behavioural-clock epochs; calls crossing edges are not clipped."""
    end = int(record["stop_frames"][-1])
    result = [("whole_supported", 0, end)]
    if record["cohort"] == "jonathan":
        entry, injection = int(metadata["entry_frame"]), int(metadata["injection_frame"])
        width, guard = round(baseline_seconds*record["fps"]), round(post_guard_seconds*record["fps"])
        if not 0 <= entry < injection < end:
            raise ValueError("verified Jonathan entry/injection outside recording")
        result += [("pre_injection_15min", max(entry, injection-width), injection),
                   ("post_injection_2to17min", injection+guard, min(end, injection+guard+width))]
    return result


def _exposure(intervals, start, stop, fps):
    return sum(max(0, min(b, stop)-max(a, start)) for a, b in intervals)/fps


def _finite(value):
    return float(value) if math.isfinite(value) else None


def recording_readouts(record, prediction, metadata, catalog, annotation_mapping, context, *,
                       null_repeats=499, seed=101, null_memory_mb=64, context_seconds=300):
    """Return long observations, denominator-rich inter rows and call descriptors.

    Each execution contrast matches original source ID, five-minute time stratum
    and training-median occupancy quadrant, with >=3 bouts on both sides. Raw
    pixel/angle units are retained. No arena-wall or body-part names are inferred.
    The outside reference consists of supported bouts outside every selected call.
    """
    import numpy as np
    import pandas as pd

    from fpcode.behavior.analysis.organisation import (
        eligible_pairs,
        metric_values,
        organisation_arrays,
        organisation_metrics,
    )
    from fpcode.statistics.chain_nulls import frequency_expectations, simulate_chain_null

    starts, stops, fps = record["start_frames"], record["stop_frames"], record["fps"]
    values, names, parents = record["values"], list(record["feature_names"]), record["parent_ids"]
    blocks = []
    for a, b in zip(starts, stops):
        if blocks and blocks[-1][1] == a:
            blocks[-1][1] = int(b)
        else:
            blocks.append([int(a), int(b)])
    valid = np.asarray(prediction["valid_intervals"], dtype=np.int64).reshape(-1, 2)
    available = np.zeros(len(starts), bool)
    if len(valid):
        position = np.searchsorted(valid[:, 0], starts, side="right")-1
        available = (position >= 0) & (stops <= valid[np.maximum(position, 0), 1])
    # Occupancy quadrants describe source coordinates; they are not arena zones.
    x, y = names.index("centroid_x_px"), names.index("centroid_y_px")
    space = (values[:, x] >= context["location"][x]).astype(int)*2 + (values[:, y] >= context["location"][y]).astype(int)
    finite_space = np.isfinite(values[:, [x, y]]).all(axis=1)
    observations, inter, descriptors = [], [], []
    consumed = set(prediction.get("construction_features", prediction.get("consumed_features", [])))

    def put(epoch, family, feature, value, *, type_id="__all__", parent_id="", numerator=None,
            denominator=None, n=0, circular=False):
        observations.append(dict(epoch=epoch, family=family, feature=feature, type_id=type_id, parent_id=parent_id,
            value=_finite(value), numerator=None if numerator is None else _finite(numerator),
            denominator=None if denominator is None else _finite(denominator), n_support=int(n),
            detector_consumed=bool(circular)))

    for epoch_index, (epoch, lo, hi) in enumerate(epochs(record, metadata)):
        if hi <= lo:
            continue
        exposure = _exposure(valid, lo, hi, fps)
        source_exposure = _exposure(blocks, lo, hi, fps)
        calls = [c for c in prediction["selected_calls"] if lo <= c["start_frame"] < c["stop_frame"] <= hi]
        eligible = available & (starts >= lo) & (stops <= hi)
        membership = np.full(len(starts), "", dtype=object)
        call_ranges = []
        for c in calls:
            a = int(np.searchsorted(starts, c["start_frame"]))
            b = int(np.searchsorted(stops, c["stop_frame"], side="right"))
            if a >= b or starts[a] != c["start_frame"] or stops[b-1] != c["stop_frame"]:
                raise ValueError("selected call is not aligned to original whole bouts")
            if np.any(membership[a:b] != "") or not eligible[a:b].all():
                raise ValueError("selected call overlaps or leaves original supported epoch")
            membership[a:b] = c["type_id"]
            call_ranges.append((c, a, b))
        durations = (stops-starts)/fps
        inside = membership != ""
        put(epoch, "support", "source_exposure_s", source_exposure, n=int(eligible.sum()))
        put(epoch, "support", "model_exposure_s", exposure, n=int(eligible.sum()))
        put(epoch, "support", "feature_support_fraction", exposure/source_exposure if source_exposure else np.nan,
            numerator=exposure, denominator=source_exposure)
        put(epoch, "support", "edge_crossing_calls", sum(c["start_frame"] < hi and c["stop_frame"] > lo for c in
            prediction["selected_calls"])-len(calls))
        for token in ["__all__"]+list(catalog):
            own = calls if token == "__all__" else [c for c in calls if c["type_id"] == token]
            d = np.array([(c["stop_frame"]-c["start_frame"])/fps for c in own])
            put(epoch, "recruitment", "calls_per_min", len(own)*60/exposure if exposure else np.nan,
                type_id=token, numerator=len(own)*60, denominator=exposure, n=len(own))
            put(epoch, "recruitment", "coverage_fraction", d.sum()/exposure if exposure else np.nan,
                type_id=token, numerator=d.sum(), denominator=exposure, n=len(own))
            put(epoch, "execution", "mean_call_duration_s", d.mean() if len(d) else np.nan,
                type_id=token, numerator=d.sum(), denominator=len(d), n=len(d))
        # Fixed constituent baseline uses exactly the supported epoch, with original IDs.
        for parent in record["source_vocabulary"]:
            mask = eligible & (parents == parent)
            put(epoch, "primitive", "calls_per_min", mask.sum()*60/exposure if exposure else np.nan,
                parent_id=parent, numerator=mask.sum()*60, denominator=exposure, n=mask.sum())
            put(epoch, "primitive", "coverage_fraction", durations[mask].sum()/exposure if exposure else np.nan,
                parent_id=parent, numerator=durations[mask].sum(), denominator=exposure, n=mask.sum())
        for feature in EXECUTION_FEATURES:
            if feature not in names:
                continue
            j = names.index(feature)
            mask = eligible & np.isfinite(values[:, j])
            put(epoch, "primitive_execution", feature, values[mask, j].mean() if mask.any() else np.nan,
                numerator=values[mask, j].sum(), denominator=mask.sum(), n=mask.sum(), circular=feature in consumed)
        if calls:
            table = pd.DataFrame([dict(chain_id=c["type_id"], start_frame=c["start_frame"],
                end_frame_exclusive=c["stop_frame"], call_id=i) for i, c in enumerate(calls)])
            table["valid_block"] = np.searchsorted(valid[:, 0], table.start_frame, side="right")-1
            # Slow recruitment regimes are additional barriers for organisation/nulls.
            strata = (table.start_frame.to_numpy()/fps/context_seconds).astype(int)
            table["valid_block"] = pd.factorize(pd.MultiIndex.from_arrays([table.valid_block, strata]))[0]
        else:
            table = pd.DataFrame(columns=["chain_id", "start_frame", "end_frame_exclusive", "call_id", "valid_block"])
        arr = organisation_arrays(table, retained_ids=catalog, fps=fps)
        for gap in (10., 30.):
            counts, runs = organisation_metrics(arr, gap_seconds=gap, modelled_seconds=exposure)
            pairs = eligible_pairs(arr, gap_seconds=gap)
            counts.update(frequency_expectations(arr["ids"], arr["reference"], pairs))
            metrics = metric_values(counts)
            for feature in ("self_probability", "mean_run_calls", "complete_mean_run_calls", "return2_probability",
                            "self_excess", "return2_excess", "effective_routines",
                            "transition_entropy_nats"):
                value = metrics.get(feature, counts.get(feature, np.nan))
                if "return2" in feature:
                    denominator = counts["return2_opportunities"]
                elif "self" in feature:
                    denominator = counts["pair_opportunities"]
                elif "complete_mean" in feature:
                    denominator = counts["complete_runs"]
                elif "mean_run" in feature:
                    denominator = counts["runs"]
                else:
                    denominator = counts["calls"]
                put(epoch, "inter", f"g{gap:g}_"+feature, value, denominator=denominator, n=counts["calls"])
            inter.append(dict(epoch=epoch, gap_seconds=gap, counts=counts))
            for token in catalog:
                ids = arr["ids"]
                vocab = list(arr["vocabulary"])
                k = vocab.index(token) if token in vocab else -1
                outgoing = pairs & (ids[:-1] == k)
                for other in catalog:
                    j = vocab.index(other) if other in vocab else -2
                    n = int((outgoing & (ids[1:] == j)).sum())
                    put(epoch, "inter_specific", f"g{gap:g}_to:"+other, n/outgoing.sum() if outgoing.any() else np.nan,
                        type_id=token, numerator=n, denominator=outgoing.sum(), n=n)
                triples = pairs[:-1] & pairs[1:]
                source = triples & (ids[:-2] == k)
                hits = source & (ids[:-2] == ids[2:]) & (ids[:-2] != ids[1:-1])
                put(epoch, "inter_specific", f"g{gap:g}_ABA", hits.sum()/source.sum() if source.any() else np.nan,
                    type_id=token, numerator=hits.sum(), denominator=source.sum(), n=hits.sum())
                ownruns = runs[runs.chain_id == token]
                put(epoch, "inter_specific", f"g{gap:g}_mean_run_calls", ownruns.length_calls.mean() if len(ownruns) else np.nan,
                    type_id=token, numerator=ownruns.length_calls.sum(), denominator=len(ownruns), n=len(ownruns))
            if gap == 10. and null_repeats >= 2 and len(calls) >= 3:
                for kind in ("frequency", "first_order"):
                    nulls = simulate_chain_null(arr, pairs=pairs, kind=kind, observed=metrics,
                        seed=seed+epoch_index*1009+(kind == "first_order"), replicates=null_repeats, memory_mb=null_memory_mb)
                    for row in nulls.to_dict("records"):
                        inter.append(dict(epoch=epoch, gap_seconds=gap, null=row))
                        for suffix in ("excess", "mc_mean_se"):
                            put(epoch, "inter_null", "g10_"+kind+"_"+row["metric"]+"_"+suffix,
                                row.get(suffix, np.nan), denominator=counts["calls"], n=row["replicates"])
        # Match same constituent/time/occupancy, retaining parent-specific summaries.
        stratum = pd.MultiIndex.from_arrays([parents, (starts/fps/context_seconds).astype(int), space])
        for token in ["__all__"]+list(catalog):
            own = inside if token == "__all__" else membership == token
            for feature in EXECUTION_FEATURES:
                if feature not in names:
                    continue
                j = names.index(feature)
                mask = eligible & finite_space & np.isfinite(values[:, j]) & (own | ~inside)
                frame = pd.DataFrame(dict(value=values[mask, j], inside=own[mask]), index=stratum[mask])
                if not len(frame):
                    continue
                summary = frame.set_index("inside", append=True).groupby(level=[0, 1, 2, 3]).value.agg(["mean", "count"]).unstack(3)
                stats = pd.DataFrame({"n_in": summary.get(("count", True), 0),
                    "n_out": summary.get(("count", False), 0), "mean_in": summary.get(("mean", True), np.nan),
                    "mean_out": summary.get(("mean", False), np.nan)}, index=summary.index)
                stats = stats[(stats.n_in >= 3) & (stats.n_out >= 3)]
                if not len(stats):
                    put(epoch, "intra", feature+"_inside_minus_outside", np.nan, type_id=token)
                    continue
                diff = stats.mean_in-stats.mean_out
                put(epoch, "intra", feature+"_inside_minus_outside", np.average(diff, weights=stats.n_in),
                    type_id=token, numerator=float((diff*stats.n_in).sum()), denominator=float(stats.n_in.sum()),
                    n=int(stats.n_in.sum()), circular=feature in consumed)
                for parent, group in stats.groupby(level=0):
                    difference = group.mean_in-group.mean_out
                    put(epoch, "intra_specific", feature+"_inside_minus_outside", np.average(difference, weights=group.n_in),
                        type_id=token, parent_id=parent, numerator=float((difference*group.n_in).sum()),
                        denominator=float(group.n_in.sum()), n=int(group.n_in.sum()), circular=feature in consumed)
        # Partial label coverage and enrichment; unresolved labels never counted as known.
        label = np.array([annotation_mapping.get(str(p), "unmapped:"+str(p)) for p in parents])
        known = np.array([t.startswith("annotation:") for t in label]) & eligible
        known_time = durations[known].sum()
        for annotation in sorted(set(label[known])):
            ann = known & (label == annotation)
            ann_time = durations[ann].sum()
            put(epoch, "annotation_coverage", annotation, durations[ann & inside].sum()/ann_time if ann_time else np.nan,
                numerator=durations[ann & inside].sum(), denominator=ann_time, n=ann.sum(),
                circular=prediction.get("annotation_input", False))
            for token in catalog:
                own = membership == token
                own_known = durations[own & known].sum()
                share = durations[own & ann].sum()/own_known if own_known else np.nan
                base = ann_time/known_time if known_time else np.nan
                put(epoch, "annotation_enrichment", annotation, share/base if base else np.nan,
                    type_id=token, denominator=own_known, n=int((own & known).sum()),
                    circular=prediction.get("annotation_input", False))
        selected_keys = {(c["type_id"], c["start_frame"], c["stop_frame"]) for c in calls}
        stages = [s for s in prediction["diagnostics"].get("stages", [])
                  if (s["type_id"], s["start_frame"], s["stop_frame"]) in selected_keys]
        for token in ["__all__"]+list(catalog):
            own = stages if token == "__all__" else [s for s in stages if s["type_id"] == token]
            evaluable = [s for s in own if s["completion"] and s["completion"]["status"] in {"completed", "deviated"}]
            if stages:
                completed = sum(s["completion"]["status"] == "completed" for s in evaluable)
                put(epoch, "hsmm_progression", "observed_completion_fraction", completed/len(evaluable) if evaluable else np.nan,
                    type_id=token, numerator=completed, denominator=len(evaluable), n=len(evaluable), circular=True)
                for order in sorted({p["order"] for s in own for p in s["phases"]}):
                    fractions = [sum(p["stop_frame"]-p["start_frame"] for p in s["phases"] if p["order"] == order)
                                 /(s["stop_frame"]-s["start_frame"]) for s in own]
                    put(epoch, "hsmm_progression", f"phase{order}_time_fraction", np.mean(fractions) if fractions else np.nan,
                        type_id=token, numerator=sum(fractions), denominator=len(fractions), n=len(fractions), circular=True)
        for c, a, b in call_ranges:
            descriptors.append(dict(epoch=epoch, type_id=c["type_id"], start_frame=c["start_frame"], stop_frame=c["stop_frame"],
                parent_counts=dict(Counter(parents[a:b])), distinct_constituents=len(set(parents[a:b])),
                bouts=b-a, duration_s=(c["stop_frame"]-c["start_frame"])/fps))
    return dict(observations=observations, inter=inter, descriptors=descriptors)
