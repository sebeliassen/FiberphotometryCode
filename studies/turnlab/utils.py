import numpy as np
from scipy.signal import savgol_filter


def _ensure_dir(path):
    import os
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def _interp_then_smooth(x, win, poly):
    """
    Linear-interpolate NaNs then Savitzky–Golay smooth.
    - Keeps array length.
    - If window is too small/long, returns the (interpolated) input.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x

    # Interpolate NaNs (edge-hold at ends)
    if np.isnan(x).any():
        idx  = np.arange(n)
        good = ~np.isnan(x)
        if good.any():
            x = x.copy()
            x[~good] = np.interp(idx[~good], idx[good], x[good])
            if np.isnan(x[0]):  x[0]  = x[good][0]
            if np.isnan(x[-1]): x[-1] = x[good][-1]
        else:
            # all NaNs
            return np.zeros_like(x)

    # S-G smoothing
    w = int(win) if win is not None else 0
    if w < 5 or w > n:
        return x
    if w % 2 == 0:
        w -= 1
    if w <= 0:
        return x
    poly = int(poly) if poly is not None else 3
    if poly >= w:
        poly = max(1, w - 2)
    return savgol_filter(x, w, poly)

def _prefix_for(col):
    """'signal_0_470_dff_z' -> 'signal_0'"""
    return "_".join(col.split("_")[:2])

def _region_hemi_for(session_idx, prefix):
    """Look up (region, hemisphere) from sessions[session_idx].signal_meta[prefix]."""
    from .context import get_sessions
    s = get_sessions()[session_idx]
    meta = getattr(s, 'signal_meta', {})
    if prefix not in meta:
        return None, None
    region, hemi = meta[prefix][0], meta[prefix][1]
    return region, hemi

def _collect_regions_hemis(signal_cols):
    """Unique regions/hemis across sessions for the given signal prefixes."""
    from .context import get_sessions
    prefixes = {_prefix_for(c) for c in signal_cols}
    regs, hemis = set(), set()
    for s in (get_sessions() or []):
        meta = getattr(s, 'signal_meta', {})
        for p in prefixes:
            if p in meta:
                r, h = meta[p][0], meta[p][1]
                if r is not None: regs.add(r)
                if h is not None: hemis.add(h)
    regs = sorted(regs)
    hemis_order = ['left','right'] if {'left','right'}.issubset(hemis) else sorted(hemis)
    return regs, hemis_order
