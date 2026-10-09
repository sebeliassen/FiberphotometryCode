"""
syllable_generation/analysis/utils/sampling.py

Unified source for:
1. Rate Detection (VFI Scale)
2. Results Upsampling (Model Step Tiling)
3. Results Flattening (Timeline Alignment)
4. Kinematics Densification (FFill)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Optional, Any

def tile_results(results: Dict[str, dict], factor: int, verbose: bool = False):
    """
    Tiles ALL arrays in the results dictionary by the given factor.
    Useful if you need to artificially increase resolution of model outputs
    before mapping (though usually VFI mapping + FFill is preferred).
    """
    if factor <= 1: return
    if verbose:
        print(f"[sampling] Tiling Results by factor {factor}...")
    
    for k in results:
        for key, val in results[k].items():
            if isinstance(val, (np.ndarray, list)):
                val = np.asarray(val)
                if val.ndim == 1:
                    results[k][key] = np.repeat(val, factor)
                else:
                    results[k][key] = np.repeat(val, factor, axis=0)


def flatten_results_to_timeline(
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
            if k not in vdict: continue
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
        if key not in vfi: continue
        flen = actual_lengths.get(key) if actual_lengths else None
        out[key] = _flatten_one(vdict, vfi[key], required_keys, full_len=flen)
    return out


def build_aligned_dfs(
    sessions: Iterable,
    moseq_df: pd.DataFrame,
    vfi_dict: Dict[str, np.ndarray],
    verbose: bool = False
) -> List[pd.DataFrame]:
    """
    Builds the list of 'fixed' (time-aligned) moseq DataFrames for the sessions.
    Automatically detects gaps in VFI mapping and applies FFill (densification) if needed.
    Can use session photometry length to ensure full coverage (extrapolation) if VFI is short.
    """
    ordered_dfs = []
    moseq_grouped = dict(list(moseq_df.groupby("name")))

    for i, s in enumerate(sessions):
        rk = getattr(s, 'results_dict_key', None)
        if not rk or rk not in moseq_grouped or rk not in vfi_dict:
            raise ValueError(f"Missing data for session {i} ({rk})")
            
        grp = moseq_grouped[rk].copy()
        
        # Reset frame_index to be 0..M so it maps 1-to-1 with VFI list
        grp = grp.reset_index(drop=True)
        grp["frame_index"] = np.arange(len(grp))

        raw_indices = grp["frame_index"].values.astype(int)
        used_vfi = vfi_dict[rk]
        
        # Map to absolute timeline
        valid_mask = raw_indices < len(used_vfi)
        abs_indices = used_vfi[raw_indices[valid_mask]]
        grp = grp[valid_mask]
        
        # Assign absolute frame indices
        grp["frame_index"] = abs_indices
        
        # Determine Target Length (Use Photometry if available, else VFI max)
        target_len = abs_indices.max() + 1
        
        # Priority 1: Explicit n_frames from video metadata (The "Mother" Anchor)
        if hasattr(s, 'video_refs_df') and hasattr(s.video_refs_df, 'empty') and not s.video_refs_df.empty:
             # Look for 'primary' role first, or just take max n_frames? Primary is safest.
             if 'role' in s.video_refs_df.columns and 'n_frames' in s.video_refs_df.columns:
                 primary = s.video_refs_df[s.video_refs_df['role'] == 'primary']
                 if not primary.empty:
                     try:
                         v_frames = int(primary.iloc[0]['n_frames'])
                         target_len = max(target_len, v_frames)
                     except (ValueError, TypeError):
                         pass

        if hasattr(s, 'photometry') and 'combined_filtered' in s.photometry:
             # Use a slight buffer logic or exact match? Ideally exact match.
             # Legacy logic effectively extrapolated to match photometry if rate was aligned.
             target_len = max(target_len, len(s.photometry['combined_filtered']))

        # GAP DETECTION & FILL / EXTRAPOLATION
        # If VFI indices are [0, 2, 4], we have gaps. We need to fill them.
        # If VFI ends at 98 but Photometry is 100, we need to fill 99.
        if len(abs_indices) > 0:
             # We densify if there are gaps OR if we need to extend to target_len
             min_idx = abs_indices.min()
             max_span = target_len - min_idx
             
             # Check if we need to act (Coverage < Span OR Gaps exist)
             # Simplest robust approach: Reindex to full range of interest
             if target_len > len(abs_indices):
                 if verbose:
                     print(f"[sampling] Densifying/Extrapolating {rk} (VFI Count={len(abs_indices)} -> Target={target_len}).")
                 
                 full_idx = np.arange(target_len)
                 
                 # Separate reindex and ffill to control filling behavior
                 grp_reindexed = grp.set_index("frame_index").reindex(full_idx)
                 grp = grp_reindexed.ffill().reset_index()
                 
                 # MASK EXTENSION: Enforce NaNs after the last actual VFI index
                 # This prevents the last syllable from bleeding into the 'n_frames' extension
                 last_vfi = abs_indices.max()
                 if target_len > last_vfi + 1:
                     # Identify rows after the last VFI point
                     # Since we reset_index, the index 0..N corresponds to frame_index 0..N
                     # so we can just use loc with slices
                     mask_start = last_vfi + 1
                     cols_to_mask = [c for c in grp.columns if c != "frame_index"]
                     grp.loc[mask_start:, cols_to_mask] = np.nan
                 # Fix dtypes after reindex/ffill
                 grp["frame_index"] = grp["frame_index"].astype(int)
                 if "syllable" in grp.columns:
                     # Use Int64 (nullable) or match classic int64. Let's use Int64 to be safe with NaNs if ffill fails at start.
                     # But verification used int64. Let's try to match int64 if no NaNs.
                     # Syllable usually shouldn't be NaN after ffill if check start.
                     grp["syllable"] = grp["syllable"].astype("Int64") # Nullable Int is safest
                 
                 grp["frame_index"] = grp.index # Should match if index was reset? No, reindex result has index=full_idx.
                 # Wait. set_index().reindex().reset_index() makes the index a column.
                 # The column 'frame_index' is now the index.
                 # So grp['frame_index'] is correct.
        
        grp = grp.reset_index(drop=True)
        ordered_dfs.append(grp)
        
    return ordered_dfs

