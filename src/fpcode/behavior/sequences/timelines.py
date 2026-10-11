"""Syllable timeline reconstruction with explicit historical gap policies."""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

LEGACY_DENSE_BOTH_ENDS = "legacy_dense_both_ends"
LEGACY_SPARSE_NO_FILL = "legacy_sparse_no_fill"

def flatten_results_to_timeline(
    results: Dict[str, dict],
    vfi: Dict[str, np.ndarray],
    required_keys: Sequence[str] = ("syllable", "latent_state", "centroid", "heading"),
    actual_lengths: Optional[Dict[str, int]] = None,
    *,
    policy: str = LEGACY_DENSE_BOTH_ENDS,
) -> Dict[str, pd.DataFrame]:
    """
    Maps Keypoint-MoSeq results onto absolute video timeline using Video Frame Indexes (VFI).
    """
    if policy == LEGACY_SPARSE_NO_FILL:
        return flatten_results_sparse_legacy(results, vfi, required_keys, actual_lengths)
    if policy != LEGACY_DENSE_BOTH_ENDS:
        raise ValueError("unknown syllable timeline policy")
    flattened = {}
    for k, vdict in results.items():
        if k not in vfi:
            continue
        idx = np.asarray(vfi[k], dtype=int)
        full_len = actual_lengths.get(k) if actual_lengths else (int(np.max(idx)) + 1 if len(idx) > 0 else 0)

        df = pd.DataFrame(index=np.arange(full_len))
        for key in required_keys:
            if key not in vdict:
                continue
            val = np.asarray(vdict[key])
            if val.ndim == 1:
                col_series = pd.Series(val, index=idx[: len(val)])
                df[key] = col_series
            elif val.ndim == 2:
                for c in range(val.shape[1]):
                    col_series = pd.Series(val[:, c], index=idx[: len(val)])
                    df[f"{key}_{c}"] = col_series
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        flattened[k] = df
    return flattened


def flatten_results_sparse_legacy(
    results: Dict[str, dict],
    vfi: Dict[str, np.ndarray],
    required_keys=("syllable", "latent_state", "centroid", "heading"),
    actual_lengths: Optional[Dict[str, int]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Maps results to the timeline using VFI.
    Assumes VFI contains Absolute Ground Truth Indices.
    """
    def _flatten_one(vdict: dict, idx: np.ndarray, req_keys, full_len: Optional[int] = None) -> pd.DataFrame:
        col_blocks, col_names, result_rows = [], [], None

        for k in req_keys:
            if k not in vdict:
                continue
            a = np.asarray(vdict[k])
            result_rows = a.shape[0] if result_rows is None else result_rows

            if a.ndim == 1:
                col_blocks.append(a.reshape(-1, 1))
                col_names.append(k)
            elif a.ndim == 2:
                d = a.shape[1]
                col_blocks.append(a)
                col_names.extend([f"{k}_{i}" for i in range(d)])

        idx = np.asarray(idx, dtype=int).ravel()
        idx_size = idx.size

        # Filter Results to match VFI length
        if result_rows != idx_size:
            # Simple truncation/padding to match VFI count
            col_blocks = [b[:idx_size] if b.shape[0] > idx_size else b for b in col_blocks]
            data_len = col_blocks[0].shape[0] if col_blocks else 0
            if data_len < idx_size:
                diff = idx_size - data_len
                new_blocks = []
                for b in col_blocks:
                    pad = np.tile(b[-1], (diff, 1)) if b.ndim == 2 else np.full(diff, b[-1])
                    new_blocks.append(np.vstack([b, pad]) if b.ndim == 2 else np.concatenate([b, pad]))
                col_blocks = new_blocks

        data = np.concatenate(col_blocks, axis=1) if col_blocks else np.empty((idx_size, 0))
        df_trim = pd.DataFrame(data, index=idx, columns=col_names)

        # Reindex to full session length (Absolute Timeline)
        if full_len is not None:
            # We want to place our sparse rows (at 0, 2, 4...) onto the full dense timeline (0, 1, 2, 3, 4...)
            # We do this by reindexing, which introduces NaNs at gaps (1, 3...)
            full_index = np.arange(int(full_len), dtype=int)
            df_full = df_trim.reindex(full_index)
        else:
            df_full = df_trim

        if "syllable" in df_full.columns:
            df_full["syllable"] = df_full["syllable"].astype("Int64")
        return df_full

    out = {}
    for key, vdict in results.items():
        if key not in vfi:
            continue
        flen = actual_lengths.get(key) if actual_lengths else None
        out[key] = _flatten_one(vdict, vfi[key], required_keys, full_len=flen)
    return out

__all__ = ["flatten_results_to_timeline", "flatten_results_sparse_legacy", "LEGACY_DENSE_BOTH_ENDS", "LEGACY_SPARSE_NO_FILL"]
