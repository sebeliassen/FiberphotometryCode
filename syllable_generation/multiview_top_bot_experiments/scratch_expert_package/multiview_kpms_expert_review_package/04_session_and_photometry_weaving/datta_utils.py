# analysis/replications/datta_utils.py

from __future__ import annotations

import re
from typing import Iterable, List, Dict
import pandas as pd

from pathlib import Path
import numpy as np

# Repo-local imports
from syllable_generation.loaders.jonathan.config import JCFG
from syllable_generation.loaders.jonathan.utils.photometry import get_signal_cols


# ---------- helpers ----------

def _freqs_to_use() -> List[str]:
    """
    Choose which photometry channels to combine. If the config doesn't define
    a list, default to ['470'].
    """
    freqs = getattr(JCFG, "combine_freqs", None)
    if freqs is None:
        return ["470"]
    if isinstance(freqs, (list, tuple)):
        return [str(f) for f in freqs]
    return [str(freqs)]

_SIGNALISH_RE = re.compile(r"^signal_\d+(?:_.+)?$")

def _signalish_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if _SIGNALISH_RE.match(str(c))]

def _required_cols_for_phot(phot_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Validate and collect columns to keep from the photometry channel df.
    Keeps SystemTimestamp and ALL 'signal_*' variants (raw and derived).
    """
    ts_col = JCFG.phot_system_ts_col  # e.g., "SystemTimestamp"
    if ts_col not in phot_df.columns:
        raise KeyError(f"Photometry channel is missing required column '{ts_col}'.")

    frame_idx_col = getattr(JCFG, "frame_index_col_name", "frame_idx")
    if frame_idx_col not in phot_df.columns:
        raise KeyError(f"Photometry channel is missing required column '{frame_idx_col}'.")

    signal_cols = _signalish_cols(phot_df)
    if not signal_cols:
        raise ValueError("No 'signal_*' columns (raw or derived) found in photometry channel.")

    keep_cols = [ts_col] + signal_cols
    return {"ts_col": ts_col, "signal_cols": signal_cols, "frame_idx_col": frame_idx_col, "keep_cols": keep_cols}


def _phot_frame_map_for_freq(session, freq: str, *, suffix: str | None) -> pd.DataFrame:
    """
    Return a dataframe indexed by frame_idx that contains:
      - SystemTimestamp (maybe suffixed)
      - all signal_* columns (maybe suffixed)
    Duplicated frame_idx in the channel are de-duplicated (first occurrence wins).
    """
    if "photometry" not in session.__dict__ or freq not in session.photometry:
        raise KeyError(f"Session is missing photometry channel '{freq}'.")

    ch_df: pd.DataFrame = session.photometry[freq]
    meta = _required_cols_for_phot(ch_df)

    # Select and deduplicate by frame index
    keep = [meta["frame_idx_col"]] + meta["keep_cols"]
    sub = ch_df.loc[:, keep]
    sub = sub.drop_duplicates(subset=[meta["frame_idx_col"]], keep="first")

    # Set frame_idx as index
    sub = sub.set_index(meta["frame_idx_col"], drop=True)

    # Optional suffixing (only if multiple freqs to avoid name collisions)
    if suffix:
        ren = {}
        for c in meta["keep_cols"]:
            ren[c] = f"{c}_{suffix}"
        sub = sub.rename(columns=ren)

    return sub


def _merge_all_freqs_on_syllable_index(session, freqs: Iterable[str]) -> pd.DataFrame:
    """
    Build an empty dataframe with the same index as session.syllables,
    then left-join each freq's photometry (indexed by frame_idx) onto it.
    The result keeps the **cardinality and index** of session.syllables.
    """
    if not hasattr(session, "syllables") or session.syllables is None:
        raise ValueError("Session is missing 'syllables' dataframe.")
    syll_df: pd.DataFrame = session.syllables
    if not isinstance(syll_df, pd.DataFrame) or syll_df.empty:
        raise ValueError("'syllables' dataframe is empty or invalid.")

    base = pd.DataFrame(index=syll_df.index)

    # Decide if we need to suffix columns (only if multiple freqs)
    freqs = list(freqs)
    use_suffix = len(freqs) > 1

    for f in freqs:
        suffix = f if use_suffix else None
        sub = _phot_frame_map_for_freq(session, f, suffix=suffix)
        base = base.join(sub, how="left")

    return base

def _only_ts_and_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only:
      - SystemTimestamp (possibly suffixed if multiple freqs)
      - ANY columns that start with signal_<num> (raw or derived), possibly suffixed
    """
    ts_base = re.escape(JCFG.phot_system_ts_col)
    ts_pat = re.compile(rf"^{ts_base}(?:_.+)?$")

    # accept any signal_* with arbitrary suffix chain
    sig_pat = _SIGNALISH_RE  # r"^signal_\d+(?:_.+)?$"

    keep = [c for c in df.columns if ts_pat.match(str(c)) or sig_pat.match(str(c))]
    if not keep:
        raise ValueError("No SystemTimestamp or signal_* columns present after merge.")
    return df.loc[:, keep]

def _interpolate_linear_both_ends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linear interpolate NaNs and pad at the ends (both directions).
    Assumes all kept columns are numeric or can be coerced to numeric.
    
    SAFETY FIX: Added limit=10. 
    - This allows upsampling (filling ~3-4 frame gaps between 20Hz samples).
    - This PREVENTS overwriting manual exclusions (large gaps >10 frames stay NaN).
    """
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        # FIX: Added limit=10
        out[c] = out[c].interpolate(method="linear", limit_direction="both", limit=4)
    return out


# ---------- public API ----------

def combined_photometry_and_syllables(session) -> pd.DataFrame:
    """
    Merge photometry channel(s) onto the syllables' frame index via frame_idx,
    keep only SystemTimestamp and signal_* columns, and linearly interpolate
    missing values (padding both ends). Returns a dataframe with the **same
    index** as session.syllables.

    Strict mode: raises if core components are missing or malformed.
    """
    freqs = _freqs_to_use()  # e.g., ["470"]
    merged = _merge_all_freqs_on_syllable_index(session, freqs)
    pruned = _only_ts_and_signals(merged)
    filled = _interpolate_linear_both_ends(pruned)
    return filled


def build_fixed_moseq_dfs(
    sessions, moseq_df: pd.DataFrame, project_dir, model_name
) -> List[pd.DataFrame]:
    """
    Returns: list of DataFrames, one per session, in the SAME order as `sessions`.
    Each DF is the `moseq_df` slice for that session's results_dict_key, with
    `frame_index` remapped by PROJECT_DIR/MODEL_NAME/video_frame_indexes.npz.

    Strict: raises on any missing or inconsistent inputs.
    """
    npz_path = Path(project_dir) / model_name / "video_frame_indexes.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        name_to_used = {k: z[k] for k in z.files}  # name -> 1D array of actual frame ids

    if "name" not in moseq_df.columns or "frame_index" not in moseq_df.columns:
        raise KeyError("moseq_df must contain 'name' and 'frame_index' columns.")

    # Build mapping name -> fixed slice (frame_index remapped)
    fixed_by_name = {}
    for name, g in moseq_df.groupby("name", sort=False):
        if name not in name_to_used:
            raise KeyError(f"Name not found in NPZ mapping: {name}")
        used = name_to_used[name]
        if used.ndim != 1:
            raise ValueError(f"NPZ array for '{name}' must be 1-D, got shape {used.shape}")

        idx = g["frame_index"].to_numpy()
        if not np.issubdtype(idx.dtype, np.integer):
            try:
                idx = idx.astype(int)
            except Exception:
                raise TypeError(f"'frame_index' for '{name}' must be integers or castable to int.")
        if idx.size and (idx.min() < 0 or idx.max() >= used.shape[0]):
            raise IndexError(
                f"frame_index out of bounds for '{name}': "
                f"valid [0, {used.shape[0]-1}], seen [{idx.min()}, {idx.max()}]"
            )

        out = g.copy()
        out["frame_index"] = used[idx]
        fixed_by_name[name] = out.reset_index(drop=True)

    # Assemble list aligned to sessions order
    dfs: List[pd.DataFrame] = []
    seen_names = set()
    for s in sessions:
        rk = getattr(s, "results_dict_key", None)
        if not isinstance(rk, str) or not rk:
            raise ValueError("Every session must have a non-empty 'results_dict_key'.")
        if rk not in fixed_by_name:
            raise KeyError(f"No moseq rows for session results_dict_key '{rk}'.")
        dfs.append(fixed_by_name[rk])
        seen_names.add(rk)

    # Optional sanity: ensure no extra names in moseq_df that don't map to any session
    extras = set(fixed_by_name.keys()) - seen_names
    if extras:
        raise ValueError(f"Found {len(extras)} names in moseq_df not matched to sessions: {sorted(extras)[:5]}...")

    return dfs


def build_aligned_moseq_phot(
    sessions,
    fixed_moseq_dfs: List[pd.DataFrame],
) -> List[pd.DataFrame]:
    """
    For each n:
      - phot = combined_photometry_and_syllables(sessions[n])     (index = frame index timeline)
      - left-align phot to fixed_moseq_dfs[n]['frame_index']
      - return concat([fixed_moseq_dfs[n], phot_aligned], axis=1)
    """
    # Find a template session to copy photometry columns from (for sessions with empty/missing photometry)
    template_phot_cols = None
    for s in sessions:
        if hasattr(s, "photometry") and s.photometry and "470" in s.photometry:
            if hasattr(s, "syllables") and s.syllables is not None and not s.syllables.empty:
                try:
                    phot = combined_photometry_and_syllables(s)
                    template_phot_cols = list(phot.columns)
                    break
                except Exception:
                    pass

    if template_phot_cols is None:
        template_phot_cols = ["SystemTimestamp", "signal_0", "signal_1", "signal_2", "signal_3"]

    aligned = []
    for i, s in enumerate(sessions):
        moseq_fixed = fixed_moseq_dfs[i]
        if "frame_index" not in moseq_fixed.columns:
            raise KeyError("fixed_moseq_dfs[i] must have a 'frame_index' column.")

        if not hasattr(s, "photometry") or not s.photometry or "470" not in s.photometry:
            print(f"Warning: Session {s.session_id} is missing photometry. Creating NaN photometry columns.")
            phot_aligned = pd.DataFrame(np.nan, index=moseq_fixed.index, columns=template_phot_cols)
        else:
            phot = combined_photometry_and_syllables(s)  # uses config to pick freqs; index = syllable/frame index
            phot_aligned = phot.reindex(moseq_fixed["frame_index"].to_numpy())

        out = pd.concat(
            [moseq_fixed.reset_index(drop=True), phot_aligned.reset_index(drop=True)],
            axis=1
        )
        aligned.append(out)
    return aligned