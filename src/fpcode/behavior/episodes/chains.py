"""Bounded motif discovery, edit alignment and occurrence grouping on bouts."""

from collections import Counter

import numpy as np
import pandas as pd


def segmented_tokens(bouts):
    """Independent valid segments, keeping original bout row positions."""
    for _, sub in bouts.groupby("segment", sort=False):
        yield sub.index.to_numpy(), sub.syllable_id.to_numpy(dtype=int)


def ngram_counts(bouts, lengths=(2, 3, 4, 5, 6)):
    counts = Counter()
    for _, tokens in segmented_tokens(bouts):
        for length in lengths:
            counts.update(tuple(tokens[i:i + length]) for i in range(len(tokens) - length + 1))
    return counts


def edit_alignment(template, observed):
    """Levenshtein alignment with explicit insertion/deletion/substitution counts."""
    a, b = tuple(template), tuple(observed)
    cost = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    cost[:, 0], cost[0, :] = np.arange(len(a) + 1), np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost[i, j] = min(cost[i - 1, j] + 1, cost[i, j - 1] + 1,
                             cost[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    i, j = len(a), len(b)
    inserted = omitted = substituted = 0
    while i or j:
        if i and j and cost[i, j] == cost[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            substituted += int(a[i - 1] != b[j - 1])
            i, j = i - 1, j - 1
        elif i and cost[i, j] == cost[i - 1, j] + 1:
            omitted += 1
            i -= 1
        else:
            inserted += 1
            j -= 1
    return {"edit_distance": int(cost[-1, -1]), "insertions": inserted,
            "omissions": omitted, "substitutions": substituted}


def exact_occurrences(bouts, template):
    """Greedy chronological non-overlapping matches, separately in each segment."""
    occurrences = []
    length = len(template)
    if length < 2:
        raise ValueError("a chain requires at least two bouts")
    for indices, tokens in segmented_tokens(bouts):
        stop = 0
        for i in range(len(tokens) - length + 1):
            if i >= stop and np.array_equal(tokens[i:i + length], template):
                occurrences.append((int(indices[i]), int(indices[i + length - 1]) + 1))
                stop = i + length
    return occurrences


def prefix_progression(bouts, template, *, prefix_length=2, extra_bouts=2):
    """Observed continuation after a specific prefix; never infer motor intention.

    Prefixes at segment ends are censored. A best short edit alignment records
    variants independently of the exact completion outcome. Shared prefixes may
    be opportunities for multiple templates and are not exclusive attempts.
    """
    if not 1 <= prefix_length < len(template):
        raise ValueError("prefix must be shorter than the full template")
    records = []
    for indices, tokens in segmented_tokens(bouts):
        for i in range(len(tokens) - prefix_length + 1):
            if not np.array_equal(tokens[i:i + prefix_length], template[:prefix_length]):
                continue
            available = len(tokens) - i
            censored = available < len(template)
            realized = tokens[i:i + len(template)]
            reached = 0
            for expected, actual in zip(template, realized):
                if expected != actual:
                    break
                reached += 1
            best = None
            for length in range(max(prefix_length, len(template) - extra_bouts),
                                min(available, len(template) + extra_bouts) + 1):
                observed = tokens[i:i + length]
                alignment = edit_alignment(template, observed)
                rank = (alignment["edit_distance"], abs(length - len(template)), length)
                if best is None or rank < best[0]:
                    best = (rank, alignment, observed)
            records.append({"start_bout": int(indices[i]), "prefix_censored": censored,
                            "exact_completion": reached == len(template), "phases_reached": reached,
                            "observed_tokens": " ".join(map(str, best[2])), **best[1]})
    return pd.DataFrame(records)


def group_occurrences(occurrences, *, gap_seconds):
    """Group same-chain events using an explicit gap rule, without naming intent.

    Negative gaps (overlap), segment changes and intervening different chains
    prevent grouping. Returns run IDs, repeat position and gap measurements.
    """
    if gap_seconds < 0:
        raise ValueError("gap threshold must be nonnegative")
    out = occurrences.sort_values(["start_s", "end_s", "chain_id"]).reset_index(drop=True).copy()
    run_ids, positions, gaps = [], [], []
    run, position = -1, 0
    previous = None
    for event in out.itertuples():
        gap = event.start_s - previous.end_s if previous is not None else np.nan
        continuing = (previous is not None and event.chain_id == previous.chain_id
                      and event.segment == previous.segment and 0 <= gap <= gap_seconds)
        if continuing:
            position += 1
        else:
            run += 1
            position = 1
        run_ids.append(run)
        positions.append(position)
        gaps.append(gap)
        previous = event
    out["run_id"], out["repeat_position"], out["preceding_gap_s"] = run_ids, positions, gaps
    if len(out):
        out["run_length"] = out.groupby("run_id")["chain_id"].transform("size")
    return out
