"""Frequency and first-order controls on fixed, explicitly eligible event slots."""

import numpy as np
import pandas as pd


def sample_first_order_bouts(segments, *, length, initial, vocabulary, replicates, seed,
                            forbid_self=True, memory_mb=128):
    """Plug-in categorical first-order surrogates fitted without segment joins.

    Fit on explicit reference segments, fix the requested initial token and
    sample replicates in parallel. Unsupported rows use empirical frequencies
    on allowed support. With RLE input, self transitions stay structurally zero;
    singleton-only support cannot generate a multi-bout RLE sequence. This null
    does not fix exact state counts or integrate transition-fit uncertainty.
    """
    if min(length, vocabulary, replicates, memory_mb) <= 0 or not 0 <= initial < vocabulary:
        raise ValueError("invalid first-order surrogate dimensions/support")
    if (vocabulary ** 2 * 24 + replicates * (length * 8 + vocabulary * 8)) > memory_mb * 1024 ** 2:
        raise MemoryError("first-order surrogate exceeds requested memory budget")
    counts = np.zeros((vocabulary, vocabulary), dtype=float)
    frequency = np.zeros(vocabulary, dtype=float)
    for tokens in segments:
        z = np.asarray(tokens)
        if z.ndim != 1 or not np.issubdtype(z.dtype, np.integer) or np.any(z < 0) or np.any(z >= vocabulary):
            raise ValueError("reference tokens differ from declared vocabulary")
        if forbid_self and np.any(z[1:] == z[:-1]):
            raise ValueError("RLE reference contains adjacent identical bouts")
        frequency += np.bincount(z, minlength=vocabulary)
        np.add.at(counts, (z[:-1], z[1:]), 1)
    if not frequency.sum():
        raise ValueError("empty first-order reference")
    probabilities = counts.copy()
    for state in range(vocabulary):
        if not probabilities[state].sum():
            probabilities[state] = frequency
        if forbid_self:
            probabilities[state, state] = 0
        if probabilities[state].sum():
            probabilities[state] /= probabilities[state].sum()
    cdf = probabilities.cumsum(axis=1)
    supported = probabilities.sum(axis=1) > 0
    cdf[supported, -1] = 1.
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, length), dtype=np.int64)
    result[:, 0] = initial
    for index in range(1, length):
        previous = result[:, index - 1]
        if not supported[previous].all():
            raise ValueError("reference has no allowed outgoing RLE transition")
        result[:, index] = np.sum(rng.random(replicates)[:, None] >= cdf[previous], axis=1)
    return result


def frequency_expectations(ids, reference_blocks, pairs):
    """Exact finite-population self/ABA expected counts, independent of gap slots.

    All eligible pairs must be inside a frequency reference block. ABA excludes
    AAA. Reference blocks are fixed independently of observed label repetitions.
    """
    ids, reference = np.asarray(ids), np.asarray(reference_blocks)
    pairs = np.asarray(pairs, dtype=bool)
    if np.any(pairs & ((reference[:-1] < 0) | (reference[:-1] != reference[1:]))):
        raise ValueError("eligible pairs must remain within a reference block")
    triple_slots = pairs[:-1] & pairs[1:]
    included = reference >= 0
    if not included.any():
        return {"frequency_expected_self_count": 0., "frequency_expected_return2_count": 0.}
    size = np.bincount(reference[included]).astype(float)
    vocabulary = max(1, int(ids.max()) + 1)
    cells, counts = np.unique(reference[included] * vocabulary + ids[included], return_counts=True)
    blocks = cells // vocabulary
    weighted = counts.astype(float) * (counts - 1)
    self_sums = np.bincount(blocks, weights=weighted, minlength=len(size))
    return_sums = np.bincount(blocks, weights=weighted * (size[blocks] - counts), minlength=len(size))
    pair_count = np.bincount(reference[:-1][pairs], minlength=len(size))
    triple_count = np.bincount(reference[:-2][triple_slots], minlength=len(size))
    self_probability = np.divide(self_sums, size * (size - 1), out=np.zeros_like(size), where=size >= 2)
    return_probability = np.divide(return_sums, size * (size - 1) * (size - 2), out=np.zeros_like(size), where=size >= 3)
    expected_self = pair_count @ self_probability
    expected_return = triple_count @ return_probability
    return {"frequency_expected_self_count": float(expected_self),
            "frequency_expected_return2_count": float(expected_return)}


def _batch_scores(labels, arrays, pairs):
    """Compute multiple surrogate measures while reusing fixed eligibility masks."""
    self_pairs = (labels[:, :-1] == labels[:, 1:]) & pairs
    counts = self_pairs.sum(axis=1)
    n_calls = arrays["kept"].sum()
    values = {"self_probability": counts / pairs.sum() if pairs.sum() else np.full(len(labels), np.nan),
              "mean_run_calls": n_calls / (n_calls - counts) if n_calls else np.full(len(labels), np.nan)}
    for lag in (2, 3, 4):
        opp = (np.all(np.lib.stride_tricks.sliding_window_view(pairs, lag), axis=1)
               if len(pairs) >= lag else np.zeros(0, dtype=bool))
        n = len(opp)
        success = labels[:, :n] == labels[:, lag:lag + n]
        for step in range(1, lag):
            success &= labels[:, step:step + n] != labels[:, :n]
        values[f"return{lag}_probability"] = (success & opp).sum(axis=1) / opp.sum() if opp.sum() else np.full(len(labels), np.nan)
    current = np.zeros(len(labels), dtype=int)
    longest = current.copy()
    for i, kept in enumerate(arrays["kept"]):
        if kept:
            current = np.where(self_pairs[:, i - 1], current + 1, 1) if i else np.ones(len(labels), dtype=int)
            longest = np.maximum(longest, current)
        else:
            current[:] = 0
    values["longest_run"] = longest.astype(float) if n_calls else np.full(len(labels), np.nan)
    return values


def simulate_chain_null(arrays, *, pairs, kind, observed, seed, replicates=1999, memory_mb=128):
    """Bounded CPU simulation, returning summary intervals and Monte Carlo SE.

    Frequency null exactly permutes IDs inside declared reference blocks.
    First-order null fits within-session eligible transitions (MLE); unobserved
    rows fall back to session empirical frequency. Fixed disconnected starts
    retain their observed IDs. Finite-sample transition-fit uncertainty is not
    integrated. Self probability is deliberately not tested against Markov.
    """
    if kind not in ("frequency", "first_order") or replicates < 2 or memory_mb <= 0:
        raise ValueError("invalid surrogate kind, budget or memory cap")
    ids, kept, reference = arrays["ids"], arrays["kept"], arrays["reference"]
    n = len(ids)
    pairs = np.asarray(pairs, dtype=bool)
    if len(pairs) != max(0, n - 1):
        raise ValueError("one explicit eligibility flag per adjacent slot pair required")
    # Conservative allowance for labels, random ranks, indices and comparisons.
    bytes_per_replicate = max(1, n) * 48 + 128
    budget = int(memory_mb * 1024 ** 2)
    fixed_bytes = len(arrays["vocabulary"]) ** 2 * 8 * 4 if kind == "first_order" else 0
    if fixed_bytes + bytes_per_replicate > budget:
        raise MemoryError("memory cap cannot hold one surrogate; raise the cap or reduce event scope")
    batch = max(1, min(256, replicates, (budget - fixed_bytes) // bytes_per_replicate))
    rng = np.random.default_rng(seed)
    included = np.flatnonzero(reference >= 0)
    cuts = np.flatnonzero(np.diff(reference[included]) != 0) + 1
    locations = np.split(included, cuts) if len(included) else []
    cdf = None
    fallback_rows = 0
    if kind == "first_order" and kept.sum():
        size = len(arrays["vocabulary"])
        transitions = np.zeros((size, size))
        np.add.at(transitions, (ids[:-1][pairs], ids[1:][pairs]), 1)
        frequency = np.bincount(ids[kept], minlength=size).astype(float)
        frequency /= frequency.sum()
        sums = transitions.sum(axis=1)
        fallback_rows = int(np.sum((sums == 0) & (frequency > 0)))
        probabilities = np.divide(transitions, sums[:, None], out=np.zeros_like(transitions), where=sums[:, None] > 0)
        probabilities[sums == 0] = frequency
        cdf = probabilities.cumsum(axis=1)
        cdf[:, -1] = 1.
    collected = {}
    for start in range(0, replicates, batch):
        width = min(batch, replicates - start)
        labels = np.broadcast_to(ids, (width, n)).copy()
        if kind == "frequency":
            for where in locations:
                if len(where) > 1:
                    order = np.argsort(rng.random((width, len(where))), axis=1)
                    labels[:, where] = ids[where][order]
        elif cdf is not None:
            for i in range(1, n):
                if pairs[i - 1]:
                    uniforms = rng.random(width)
                    labels[:, i] = np.sum(uniforms[:, None] >= cdf[labels[:, i - 1]], axis=1)
        scores = _batch_scores(labels, arrays, pairs)
        for metric, values in scores.items():
            if kind == "first_order" and metric == "self_probability":
                continue
            collected.setdefault(metric, []).append(values)
    rows = []
    for metric, pieces in collected.items():
        values = np.concatenate(pieces)
        values = values[np.isfinite(values)]
        obs = observed.get(metric, np.nan)
        if not len(values):
            rows.append({"null_kind": kind, "metric": metric, "replicates": 0, "observed": obs})
            continue
        mean = values.mean()
        rows.append({"null_kind": kind, "metric": metric, "replicates": len(values), "observed": obs,
                     "null_mean": mean, "null_low": np.quantile(values, .025), "null_high": np.quantile(values, .975),
                     "excess": obs - mean, "mc_mean_se": values.std(ddof=1) / np.sqrt(len(values)),
                     "p_surrogate_two_sided": ((1 + np.sum(abs(values - mean) >= abs(obs - mean) - 1e-12)) / (len(values) + 1)) if np.isfinite(obs) else np.nan,
                     "batch_replicates": batch, "conservative_batch_bytes": fixed_bytes + batch * bytes_per_replicate,
                     "fallback_rows": fallback_rows})
    return pd.DataFrame(rows)
