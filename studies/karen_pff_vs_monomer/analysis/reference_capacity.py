"""Quantitative capacity diagnostics and explicit bout-order nulls; no GPU import."""

import numpy as np


def template_cosines(first, second, max_shift=1):
    """Full-norm cosine: shifted partial matches pay for the unmatched stages."""
    a = first.transpose(1, 0, 2)
    b = second.transpose(1, 0, 2)
    norm = np.linalg.norm(a.reshape(len(a), -1), axis=1)[:, None]
    norm = norm * np.linalg.norm(b.reshape(len(b), -1), axis=1)[None, :]
    best = np.full((len(a), len(b)), -np.inf)
    phase_locked = None
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            aa, bb = a[:, :, -shift:], b[:, :, :shift]
        elif shift > 0:
            aa, bb = a[:, :, :-shift], b[:, :, shift:]
        else:
            aa, bb = a, b
        dot = aa.reshape(len(a), -1) @ bb.reshape(len(b), -1).T
        similarity = dot / np.maximum(norm, 1e-8)
        best = np.maximum(best, similarity)
        if shift == 0:
            phase_locked = similarity
    return np.clip(phase_locked, 0, 1), np.clip(best, 0, 1)


def execution_masks(calls, factors, columns, length=3):
    masks = np.zeros((factors, columns), bool)
    for row in calls.itertuples():
        if row.end_exclusive - row.start != length:
            raise ValueError("call length differs from frozen model length")
        masks[int(row.factor), int(row.start) : int(row.end_exclusive)] = True
    return masks


def execution_overlap(first, second, durations):
    """Native-time union Jaccard and directional containment, not summed calls."""
    weights = np.sqrt(np.asarray(durations, dtype=np.float64))
    a, b = first * weights, second * weights
    intersection = a @ b.T
    sa, sb = (a * a).sum(axis=1), (b * b).sum(axis=1)
    union = sa[:, None] + sb[None, :] - intersection
    jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    in_first = np.divide(intersection, sa[:, None], out=np.zeros_like(intersection), where=sa[:, None] > 0)
    in_second = np.divide(intersection, sb[None, :], out=np.zeros_like(intersection), where=sb[None, :] > 0)
    return np.clip(jaccard, 0, 1), np.clip(in_first, 0, 1), np.clip(in_second, 0, 1)


def supported_blocks(tokens):
    valid = np.asarray(tokens) >= 0
    bounds = np.flatnonzero(np.r_[True, valid[1:] != valid[:-1], True])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if valid[a]]


def transition_counts(tokens, vocabulary):
    counts = np.zeros((vocabulary, vocabulary), dtype=np.int64)
    z = np.asarray(tokens, int)
    np.add.at(counts, (z[:-1], z[1:]), 1)
    return counts


def eulerian_shuffle(tokens, rng):
    """Random Euler trail preserving exact directed transitions and endpoints.

    This is not claimed to be a uniform sample over all Euler trails. The
    transition-conditioned randomization has limited power in rigid blocks.
    """
    z = np.asarray(tokens, int)
    if len(z) < 3:
        return z.copy()
    outgoing = {}
    for before, after in zip(z[:-1].tolist(), z[1:].tolist()):
        outgoing.setdefault(before, []).append(after)
    for values in outgoing.values():
        rng.shuffle(values)
    stack, reverse = [int(z[0])], []
    while stack:
        values = outgoing.get(stack[-1])
        if values:
            stack.append(values.pop())
        else:
            reverse.append(stack.pop())
    result = np.asarray(reverse[::-1], dtype=int)
    if len(result) != len(z) or result[0] != z[0] or result[-1] != z[-1]:
        raise AssertionError("invalid randomized Euler trail")
    return result


def order_null(tokens, kind, seed, vocabulary):
    """Keep padding/support fixed; never RLE again after shuffling bout tokens."""
    rng = np.random.RandomState(seed)
    result = np.asarray(tokens).copy()
    changed, count = 0, 0
    for start, end in supported_blocks(tokens):
        original = np.asarray(tokens[start:end])
        if kind == "frequency":
            shuffled = rng.permutation(original)
        elif kind == "first_order":
            shuffled = eulerian_shuffle(original, rng)
            if not np.array_equal(transition_counts(original, vocabulary), transition_counts(shuffled, vocabulary)):
                raise AssertionError("first-order null changed directed edge counts")
        else:
            raise ValueError("unknown order-null type")
        if not np.array_equal(np.bincount(original, minlength=vocabulary), np.bincount(shuffled, minlength=vocabulary)):
            raise AssertionError("null changed bout-frequency counts")
        result[start:end] = shuffled
        changed += int((original != shuffled).sum())
        count += len(original)
    return result, {
        "kind": kind,
        "seed": seed,
        "changed_token_fraction": changed / max(1, count),
        "padding_preserved": bool(np.array_equal(result[tokens < 0], tokens[tokens < 0])),
    }
