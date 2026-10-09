# utils/frame_map_placeholders.py
# Helpers to fabricate *placeholder* frame-index mappings for testing/analysis.
# We intentionally avoid the name "video_frame_indexes" to prevent confusion
# with the real artifact written by the pipeline.

def build_placeholder_frame_index_map(results, mode="identity", full_lengths=None):
    """
    Create a fake mapping from kept-frame rows (length = len(results[k]["syllable"]))
    back to original frame indices.

    Parameters
    ----------
    results : dict
        KPMS results dict: key -> {"syllable": (N,), "latent_state": (N,d), ...}
    mode : {"identity","spread"}
        - "identity":   [0, 1, 2, ..., N-1]  (safest, simplest)
        - "spread":     distribute the N kept frames across [0, full_len-1]
                         (requires full_lengths; just for nicer visualization)
    full_lengths : dict or None
        Optional key -> int (true video length). Needed for "spread".

    Returns
    -------
    frame_index_map : dict[key] -> np.ndarray[int]  (placeholder indices)
    """
    import numpy as np

    fmap = {}
    for key, v in results.items():
        N = int(len(v["syllable"]))  # KPMS always provides this
        if mode == "identity" or full_lengths is None:
            fmap[key] = np.arange(N, dtype=int)
            continue

        L = int(full_lengths.get(key, N))
        if L <= N:  # degenerate; can’t strictly spread
            fmap[key] = np.arange(N, dtype=int)
            continue

        idx = np.linspace(0, L - 1, N)
        idx = np.rint(idx).astype(int)
        # ensure monotone nondecreasing & within bounds
        idx = np.maximum.accumulate(idx)
        idx[idx >= L] = L - 1

        # nudge duplicates forward where possible
        diffs = np.diff(idx, prepend=-1)
        ties = (diffs == 0).nonzero()[0]
        for t in ties:
            if idx[t] < L - 1:
                idx[t] = min(idx[t - 1] + 1 if t > 0 else idx[t], L - 1)

        fmap[key] = idx

    return fmap


def save_placeholder_frame_index_map(frame_index_map, out_dir, filename="frame_index_map_placeholder.npz"):
    """
    Persist the placeholder map as a clearly-named NPZ so downstream code can load it
    *without* mistaking it for the real video_frame_indexes.
    """
    import numpy as np
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / filename, **frame_index_map)
    return str(out_dir / filename)


def is_placeholder_frame_index_map(path):
    """
    Quick helper to detect our placeholder file by name.
    """
    from pathlib import Path
    return Path(path).name == "frame_index_map_placeholder.npz"
