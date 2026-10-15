"""Boundary-aware adaptation of the recovered September multiplicative updates.

The historical activation-competition term and per-lag W normalisation are
preserved. Unsupported onsets are exactly zero. This is distinct from the
newer shared-library data-dependent seqNMF penalty. No GPU imports at import.
"""

import numpy as np


def shift_right(value, lag, xp=np):
    return xp.pad(value, ((0, 0), (lag, 0)))[:, :value.shape[1]]


def shift_left(value, lag, xp=np):
    return xp.pad(value, ((0, 0), (0, lag)))[:, lag:]


def reconstruction(w, h, xp=np):
    result = xp.zeros((w.shape[0], h.shape[1]), dtype=w.dtype)
    for lag in range(w.shape[2]):
        result = result + w[:, :, lag] @ shift_right(h, lag, xp)
    return result


def update(w, h, x, valid, penalty, *, templates=True, xp=np):
    """Historical update, with explicit onset masks applied after every H update."""
    ratio = x / xp.maximum(reconstruction(w, h, xp), 1e-7)
    numerator, denominator = xp.zeros_like(h), xp.zeros_like(h)
    ones = xp.ones_like(x)
    for lag in range(w.shape[2]):
        numerator = numerator + shift_left(w[:, :, lag].T @ ratio, lag, xp)
        denominator = denominator + shift_left(w[:, :, lag].T @ ones, lag, xp)
    denominator = denominator + penalty * (h.sum(axis=0, keepdims=True) - h)
    h = xp.where(valid[None, :], xp.maximum(h * numerator / xp.maximum(denominator, 1e-7), 1e-7), 0.)
    if not templates:
        return w, h
    ratio = x / xp.maximum(reconstruction(w, h, xp), 1e-7)
    layers = []
    for lag in range(w.shape[2]):
        shifted = shift_right(h, lag, xp)
        layer = w[:, :, lag] * ((ratio @ shifted.T) / xp.maximum(ones @ shifted.T, 1e-7))
        layer = layer / (xp.linalg.norm(layer, axis=0, keepdims=True) + 1e-7)
        layers.append(layer)
    return xp.stack(layers, axis=2), h


def backend(device):
    """Initialize only on explicit empirical execution; never silently fall back."""
    if device == "cpu":
        return np, lambda *a, **kw: update(*a, **kw), {"device": "cpu", "dtype": "float32"}
    import jax
    import jax.numpy as jnp

    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("requested GPU is unavailable")
    jax.config.update("jax_default_matmul_precision", "highest")
    step = jax.jit(lambda w, h, x, valid, penalty: update(w, h, x, valid, penalty, xp=jnp))
    return jnp, step, {"device": str(devices[0]), "kind": devices[0].device_kind,
                       "dtype": "float32", "matmul_precision": "highest", "jax": jax.__version__}


def initialise(vocabulary, factors, length, columns, valid, seed):
    rng = np.random.RandomState(seed)
    w = rng.uniform(.1, 1., (vocabulary, factors, length)).astype(np.float32)
    h = rng.uniform(.1, 1., (factors, columns)).astype(np.float32)
    for lag in range(length):
        w[:, :, lag] /= np.linalg.norm(w[:, :, lag], axis=0, keepdims=True) + 1e-7
    h[:, ~valid] = 0
    return w, h


def fit(x, valid, *, factors, length, penalty, iterations, seed, runtime, progress=None):
    xp, step, _ = runtime
    w, h = initialise(x.shape[0], factors, length, x.shape[1], valid, seed)
    w, h, xx, vv = xp.asarray(w), xp.asarray(h), xp.asarray(x), xp.asarray(valid)
    history = []
    for iteration in range(iterations):
        w, h = step(w, h, xx, vv, penalty)
        if iteration % 10 == 0 or iteration == iterations - 1:
            error = float(xp.linalg.norm(xx - reconstruction(w, h, xp)) / xp.linalg.norm(xx))
            history.append({"iteration": iteration + 1, "relative_error": error})
            if progress:
                progress(iteration + 1, error)
    return np.asarray(w), np.asarray(h), history


def infer(w, x, valid, *, penalty, iterations, seed, runtime):
    xp, _, _ = runtime
    ww, xx, vv = xp.asarray(w), xp.asarray(x), xp.asarray(valid)
    rng = np.random.RandomState(seed)
    h = rng.uniform(.1, 1., (w.shape[1], x.shape[1])).astype(np.float32)
    h[:, ~valid] = 0
    h = xp.asarray(h)
    if xp is np:
        def step(a, b, c, d, e):
            return update(a, b, c, d, e, templates=False)
    else:
        import jax

        step = jax.jit(lambda a, b, c, d, e: update(a, b, c, d, e, templates=False, xp=xp))
    for _ in range(iterations):
        _, h = step(ww, h, xx, vv, penalty)
    return np.asarray(h)


def retrieve(tokens, valid, w, h, thresholds, *, max_error=.68, stage_weight=.15, min_support=.5):
    """Historical NMS/error/support definitions, using frozen training thresholds."""
    length, total = w.shape[2], len(tokens)
    calls = []
    for factor in range(w.shape[1]):
        hk, template = h[factor], w[:, factor, :]
        candidates = np.flatnonzero(valid & (hk > thresholds[factor]))
        candidates = candidates[(candidates > 0) & (candidates < total - length - 1)]
        candidates = candidates[(hk[candidates] >= hk[candidates - 1]) & (hk[candidates] >= hk[candidates + 1])]
        suppressed = np.zeros(total, bool)
        selected = []
        for start in candidates[np.argsort(-hk[candidates], kind="stable")]:
            if not suppressed[start]:
                selected.append(int(start))
                suppressed[max(0, start - 2):min(total, start + 3)] = True
        norm = float(np.sum(template**2)) + 1e-8
        for start in sorted(selected):
            observed = tokens[start:start + length]
            weights = template[observed, np.arange(length)]
            dot = float(weights.sum())
            amplitude = max(0., dot / norm)
            error = max(0., float((length - 2 * amplitude * dot + amplitude**2 * (norm - 1e-8)) / (length + 1e-8)))
            support = float(np.mean(weights > stage_weight))
            if error <= max_error and support >= min_support:
                calls.append({"factor": factor, "start": start, "end_exclusive": start + length,
                              "activation": float(hk[start]), "template_error": error,
                              "stage_support": support, "observed_tokens": observed.tolist()})
    return calls
