"""Group-blind, supported local exemplars of an already frozen theme population.

Distances consume the corrected descriptors unchanged. The three anchors describe
variations for display; neither nearest-anchor assignment nor separated local
balls constitutes a fitted clustering model or evidence for biological modes.
"""

import math
from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist


def _identity(calls):
    required = ("recording", "subject", "start_frame", "stop_frame", "occurrence_id")
    identities = []
    for call in calls:
        if any(key not in call for key in required):
            raise ValueError("each call requires recording/subject/start_frame/stop_frame/occurrence_id")
        start, stop = int(call["start_frame"]), int(call["stop_frame"])
        if start != call["start_frame"] or stop != call["stop_frame"] or not 0 <= start < stop:
            raise ValueError("call frames must define a nonnegative half-open integer core")
        if any(not str(call[key]).strip() for key in ("recording", "subject", "occurrence_id")):
            raise ValueError("call recording, subject and occurrence_id must be nonempty")
        identities.append(dict(recording=str(call["recording"]), subject=str(call["subject"]),
                               start_frame=start, stop_frame=stop, occurrence_id=str(call["occurrence_id"])))
    if len({row["occurrence_id"] for row in identities}) != len(identities):
        raise ValueError("occurrence_id must be unique")
    subjects = {}
    for row in identities:
        prior = subjects.setdefault(row["recording"], row["subject"])
        if prior != row["subject"]:
            raise ValueError("one recording cannot identify multiple subjects")
    return identities


def _overlaps(index, selected, calls):
    candidate = calls[index]
    return any(candidate["recording"] == calls[other]["recording"]
               and candidate["start_frame"] < calls[other]["stop_frame"]
               and calls[other]["start_frame"] < candidate["stop_frame"] for other in selected)


def _panels(anchor, order, selected, calls, count):
    """Anchor first, then nearest compatible distinct mice, then other mice."""
    if _overlaps(anchor, selected, calls):
        return None
    chosen, subjects = [int(anchor)], {calls[anchor]["subject"]}
    for require_new_mouse in (True, False):
        for index in order:
            index = int(index)
            if index in chosen or index in selected or _overlaps(index, selected + chosen, calls):
                continue
            if require_new_mouse and calls[index]["subject"] in subjects:
                continue
            chosen.append(index)
            subjects.add(calls[index]["subject"])
            if len(chosen) == count:
                return chosen
    return None


def _distances(values, subjects, max_memory_gb):
    """Bound the principal arrays before allocating a float64 pairwise matrix."""
    n, width = values.shape
    counts = Counter(subjects)
    # Matrix, descriptor conversion, worst subject-column gather, two small
    # subject-distance arrays, several per-execution vectors and allocator slack.
    estimated_bytes = math.ceil(1.25 * 8 * (n * n + n * width + n * max(counts.values())
                                          + 2 * n * len(counts) + 16 * n)) + 16 * 1024 ** 2
    cap_bytes = int(max_memory_gb * 1024 ** 3)
    if estimated_bytes > cap_bytes:
        raise ValueError(f"pairwise memory estimate {estimated_bytes} exceeds cap {cap_bytes}")
    distance = np.empty((n, n), dtype=np.float64)
    # cdist writes directly into a bounded output view, without an N x N x D
    # broadcasting temporary or cancellation from the norm/dot-product identity.
    for begin in range(0, n, 128):
        cdist(values[begin:begin + 128], values, metric="sqeuclidean", out=distance[begin:begin + 128])
    if not np.isfinite(distance).all():
        raise ValueError("squared Euclidean distances overflowed")
    return distance, estimated_bytes, cap_bytes


def select_neighbourhoods(values, calls, *, regions=3, panels=3, max_memory_gb=1.0):
    """Select supported exemplars and compatible actual executions per theme.

    ``values`` contains the already weighted/scaled descriptors, one row per
    ``calls`` entry. Only the five identity fields are inspected; group, wave,
    condition and phenotype are deliberately unused. Three distinct mice are
    required in the population. Local density is the mean squared distance to
    the nearest execution from each of up to five *other* mice. This avoids a
    prolific mouse alone creating a dense exemplar neighbourhood.

    The first anchor is densest. Further anchors maximise distance from previous
    anchors divided by their own local-density distance, within the densest 75%
    of candidates. This supplies supported spread without fitting clusters. If
    that pool cannot provide enough nonoverlapping panels, remaining candidates
    are considered and explicitly marked as density-pool fallbacks. Every anchor
    is shown with its two nearest compatible neighbours, preferring distinct mice.

    Overlap suppression concerns half-open *core* intervals across all selected
    panels within this theme; playback context may overlap. Population summaries
    retain every eligible execution, including overlapping original detections.
    """
    if not isinstance(regions, int) or not isinstance(panels, int) or regions < 1 or panels < 2:
        raise ValueError("regions must be positive and panels must be at least two")
    if not np.isfinite(max_memory_gb) or max_memory_gb <= 0:
        raise ValueError("max_memory_gb must be finite and positive")
    values = np.asarray(values, dtype=np.float64)
    identities = _identity(list(calls))
    if values.ndim != 2 or values.shape[0] != len(identities) or values.shape[1] == 0:
        raise ValueError("descriptor rows must match call identities and have nonzero width")
    if not np.isfinite(values).all():
        raise ValueError("descriptors must be finite")
    if len(identities) < regions * panels:
        raise ValueError("not enough unique executions for requested panels")
    subjects = np.asarray([row["subject"] for row in identities])
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 3:
        raise ValueError("supported neighbourhoods require at least three distinct mice")
    distance, estimated_bytes, cap_bytes = _distances(values, subjects, max_memory_gb)
    nearest_mouse = np.empty((len(values), len(unique_subjects)), dtype=float)
    for column, subject in enumerate(unique_subjects):
        nearest_mouse[:, column] = np.min(distance[:, subjects == subject], axis=1)
        nearest_mouse[subjects == subject, column] = np.inf
    k = min(5, len(unique_subjects) - 1)
    nearest_mouse.partition(k - 1, axis=1)
    local_density = nearest_mouse[:, :k].mean(axis=1)
    radius_squared = nearest_mouse[:, k - 1].copy()
    del nearest_mouse
    canonical = sorted(range(len(identities)), key=lambda i: (
        identities[i]["occurrence_id"], identities[i]["recording"],
        identities[i]["start_frame"], identities[i]["stop_frame"]))
    tie_order = np.empty(len(values), dtype=int)
    tie_order[canonical] = np.arange(len(values))
    density_cutoff = float(np.quantile(local_density, .75))
    dense_pool = local_density <= density_cutoff
    epsilon = np.finfo(float).eps * max(1., float(np.median(local_density)))
    chosen, anchors, summaries = [], [], []
    warnings = []
    for region_number in range(1, regions + 1):
        if not anchors:
            priority = local_density
        else:
            separation = np.min(distance[:, anchors], axis=1)
            priority = -separation / np.maximum(local_density, epsilon)
        # Density-pool candidates precede explicit fallbacks; identity breaks ties.
        candidates = np.lexsort((tie_order, priority, ~dense_pool))
        selected = None
        for anchor in candidates:
            if int(anchor) in chosen:
                continue
            order = np.lexsort((tie_order, distance[anchor]))
            selected = _panels(int(anchor), order, chosen, identities, panels)
            if selected is not None:
                break
        if selected is None:
            warnings.append(f"Only {len(summaries)} neighbourhoods could supply distinct nonoverlapping panels")
            break
        anchor = int(anchor)
        support = distance[anchor] <= radius_squared[anchor]
        support_mice = int(len(np.unique(subjects[support])))
        if support_mice < 3:
            raise RuntimeError("cross-mouse local support unexpectedly fell below three mice")
        if anchors:
            nearest_previous = min(anchors, key=lambda other: distance[anchor, other])
            separation_squared = float(distance[anchor, nearest_previous])
            radii_sum = math.sqrt(radius_squared[anchor]) + math.sqrt(radius_squared[nearest_previous])
            separation = math.sqrt(separation_squared)
            roundoff = 8 * np.finfo(float).eps * max(1., radii_sum, separation)
            separated = bool(separation > radii_sum + roundoff)
            status = "separated_local_neighbourhood" if separated else "overlapping_variation_anchor"
        else:
            separation_squared, separated, status = None, None, "densest_supported_anchor"
        support_by_mouse = [float(np.mean(support[subjects == mouse])) for mouse in unique_subjects]
        summaries.append(dict(id=f"region_{region_number}", anchor_index=anchor,
                              anchor_occurrence_id=identities[anchor]["occurrence_id"],
                              selected_indices=selected,
                              selected_occurrence_ids=[identities[index]["occurrence_id"] for index in selected],
                              selected_distances_squared=[float(distance[anchor, index]) for index in selected],
                              selected_mice=int(len({identities[index]["subject"] for index in selected})),
                              support_radius_squared=float(radius_squared[anchor]),
                              local_density_distance_squared=float(local_density[anchor]),
                              support_count=int(support.sum()), support_share=float(support.mean()),
                              support_mice=support_mice,
                              support_animal_balanced_share=float(np.mean(support_by_mouse)),
                              nearest_previous_separation_squared=separation_squared,
                              local_balls_separated=separated, status=status,
                              density_pool_fallback=bool(not dense_pool[anchor])))
        chosen.extend(selected)
        anchors.append(anchor)
    if len(anchors) == 0:
        raise ValueError("no compatible neighbourhood could supply the requested panels")
    anchor_distances = distance[:, anchors]
    assignment = np.argmin(anchor_distances, axis=1)
    within = anchor_distances <= radius_squared[np.asarray(anchors)][None, :]
    # This is geometric overlap between local support balls, not posterior class
    # uncertainty or a statistical cluster-membership probability.
    assignments = [dict(index=index, occurrence_id=identities[index]["occurrence_id"],
                        nearest_region=summaries[int(label)]["id"],
                        distance_squared=float(anchor_distances[index, label]),
                        within_nearest_local_radius=bool(within[index, label]),
                        within_any_local_radius=bool(within[index].any()),
                        overlapping_local_support=bool(within[index].sum() > 1))
                   for index, label in enumerate(assignment)]
    for number, summary in enumerate(summaries):
        mask = assignment == number
        summary.update(nearest_partition_count=int(mask.sum()), nearest_partition_share=float(mask.mean()),
                       nearest_partition_mice=int(len(np.unique(subjects[mask]))))
    return dict(regions=summaries, selected_indices=chosen, assignments=assignments,
                n_executions=len(values), n_mice=len(unique_subjects), descriptor_width=values.shape[1],
                local_support_other_mice=k, density_pool_quantile=.75, density_pool_cutoff=density_cutoff,
                distance_tolerance_squared=0., priority_denominator_floor_squared=float(epsilon),
                population_local_radius_coverage=float(within.any(axis=1).mean()),
                population_local_radius_overlap=float((within.sum(axis=1) > 1).mean()),
                policy=dict(distance="unchanged input descriptors; float64 squared Euclidean",
                            local_density="mean nearest-execution squared distance across up to five other mice",
                            anchor_selection="densest first; density-adjusted spread in densest75percent, then marked fallbacks",
                            panel_selection="anchor plus nearest core-nonoverlapping executions, preferring distinct mice",
                            core_overlap="suppressed across every selected panel within this theme; half-open native frame ranges",
                            assignments="descriptive nearest-anchor partition; not fitted clusters or biological modes",
                            support="all executions inside anchor radius reaching up to five other mice; balls can overlap",
                            phenotype_selection=False),
                resource_estimate=dict(estimated_peak_array_bytes=estimated_bytes, cap_bytes=cap_bytes,
                                       pairwise_dtype="float64", pairwise_rows_per_chunk=128),
                identities=identities, warnings=warnings)
