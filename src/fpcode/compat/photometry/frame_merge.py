"""
Historical photometry frame-table merging
===========================
Temporal alignment and multi-channel merging of photometry signals with MoSeq
discrete syllable sequences and video frame timestamps.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

_SIGNALISH_RE = re.compile(r"^signal_\d+(?:_.+)?$")
_TS_RE = re.compile(r"^SystemTimestamp(?:_.+)?$")

# Historical pandas interpolation is by row position, including endpoint fill.
# New coordinate-aware consumers use fpcode.alignment.align_signal explicitly.
LEGACY_ROW_LINEAR_BOTH_ENDS = "legacy_row_linear_both_ends"


def _signalish_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if _SIGNALISH_RE.match(str(c))]


def _required_cols_for_phot(
    phot_df: pd.DataFrame,
    phot_ts_col: str = "SystemTimestamp",
    frame_idx_col: str = "frame_idx",
) -> Dict[str, List[str]]:
    """Validate and collect columns to keep from the photometry channel dataframe."""
    if phot_ts_col not in phot_df.columns:
        raise KeyError(f"Photometry channel is missing required column '{phot_ts_col}'.")
    if frame_idx_col not in phot_df.columns:
        raise KeyError(f"Photometry channel is missing required column '{frame_idx_col}'.")

    signal_cols = _signalish_cols(phot_df)
    if not signal_cols:
        raise ValueError("No 'signal_*' columns found in photometry channel.")

    keep_cols = [phot_ts_col] + signal_cols
    return {
        "ts_col": phot_ts_col,
        "signal_cols": signal_cols,
        "frame_idx_col": frame_idx_col,
        "keep_cols": keep_cols,
    }


def _phot_frame_map_for_freq(
    session,
    freq: str,
    *,
    suffix: Optional[str] = None,
    phot_ts_col: str = "SystemTimestamp",
    frame_idx_col: str = "frame_idx",
) -> pd.DataFrame:
    """Return a dataframe indexed by frame_idx that contains timestamp and signal columns."""
    if not hasattr(session, "photometry"):
        raise AttributeError("Session object lacks 'photometry' attribute.")
    if freq not in session.photometry:
        raise KeyError(f"Requested frequency '{freq}' not present in session.photometry.")

    phot_df = session.photometry[freq]
    cols = _required_cols_for_phot(phot_df, phot_ts_col=phot_ts_col, frame_idx_col=frame_idx_col)

    fidx = phot_df[cols["frame_idx_col"]]
    subset = phot_df.loc[:, cols["keep_cols"]].copy()

    if suffix:
        subset = subset.add_suffix(suffix)

    subset.index = fidx
    subset = subset[~subset.index.duplicated(keep="first")]
    return subset


def _merge_all_freqs_on_syllable_index(
    session,
    freqs: Iterable[str],
    phot_ts_col: str = "SystemTimestamp",
    frame_idx_col: str = "frame_idx",
) -> pd.DataFrame:
    """Left-join each frequency's frame map onto session.syllables index."""
    if not hasattr(session, "syllables"):
        raise AttributeError("Session object lacks 'syllables' attribute.")

    base_idx = session.syllables.index
    df_merged = pd.DataFrame(index=base_idx)
    freqs_list = list(freqs)

    for freq in freqs_list:
        suffix = f"_{freq}" if len(freqs_list) > 1 else None
        frame_map = _phot_frame_map_for_freq(
            session, freq, suffix=suffix, phot_ts_col=phot_ts_col, frame_idx_col=frame_idx_col
        )
        df_merged = df_merged.join(frame_map, how="left")

    return df_merged


def _only_ts_and_signals(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if _TS_RE.match(str(c)) or _SIGNALISH_RE.match(str(c))]
    if not keep:
        raise ValueError("No SystemTimestamp or signal_* columns present after merge.")
    return df.loc[:, keep]


def _interpolate_linear_both_ends(df: pd.DataFrame, limit: int = 4) -> pd.DataFrame:
    """Linear interpolate NaNs and pad ends with configurable gap limit."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].interpolate(method="linear", limit_direction="both", limit=limit)
    return out


def combined_photometry_and_syllables(
    session,
    freqs: Optional[List[str]] = None,
    phot_ts_col: str = "SystemTimestamp",
    frame_idx_col: str = "frame_idx",
    interpolation_limit: int = 4,
    *,
    policy: str = LEGACY_ROW_LINEAR_BOTH_ENDS,
) -> pd.DataFrame:
    """
    Merge photometry channel(s) onto the syllables frame index via frame_idx,
    keep SystemTimestamp and signal_* columns, and linearly interpolate missing values.
    Returns a dataframe with the same index as session.syllables.
    """
    if policy != LEGACY_ROW_LINEAR_BOTH_ENDS:
        raise ValueError("unsupported legacy alignment policy; use fpcode.alignment for strict alignment")
    if freqs is None:
        freqs = ["470"]

    merged = _merge_all_freqs_on_syllable_index(
        session, freqs, phot_ts_col=phot_ts_col, frame_idx_col=frame_idx_col
    )
    pruned = _only_ts_and_signals(merged)
    filled = _interpolate_linear_both_ends(pruned, limit=interpolation_limit)
    return filled


def build_fixed_moseq_dfs(
    sessions: List[Any],
    moseq_df: pd.DataFrame,
    project_dir: Union[Path, str],
    model_name: str,
) -> List[pd.DataFrame]:
    """
    Remaps each session's moseq_df slice frame_index by PROJECT_DIR/MODEL_NAME/video_frame_indexes.npz.
    Returns a list of DataFrames in the same order as sessions.
    """
    npz_path = Path(project_dir) / model_name / "video_frame_indexes.npz"
    with np.load(npz_path, allow_pickle=False) as z:
        name_to_used = {k: z[k] for k in z.files}

    if "name" not in moseq_df.columns or "frame_index" not in moseq_df.columns:
        raise KeyError("moseq_df must contain 'name' and 'frame_index' columns.")

    out = []
    for s in sessions:
        key = getattr(s, "results_dict_key", None)
        if key is None or key not in name_to_used:
            raise KeyError(f"Session results_dict_key '{key}' not found in video_frame_indexes.npz")

        slice_df = moseq_df[moseq_df["name"] == key].copy()
        used_frames = name_to_used[key]

        if len(slice_df) != len(used_frames):
            raise ValueError(
                f"Length mismatch for key '{key}': {len(slice_df)} rows vs {len(used_frames)} used frames."
            )

        slice_df["frame_index"] = used_frames
        out.append(slice_df)

    return out


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
            phot = combined_photometry_and_syllables(s)  # preserved default470 policy; index = frame index
            phot_aligned = phot.reindex(moseq_fixed["frame_index"].to_numpy())

        out = pd.concat(
            [moseq_fixed.reset_index(drop=True), phot_aligned.reset_index(drop=True)],
            axis=1
        )
        aligned.append(out)
    return aligned
