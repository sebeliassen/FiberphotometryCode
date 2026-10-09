"""Compatibility imports for photometry grouping, statistics and Bonsai alignment."""

import pandas as pd

from fpcode.statistics import compute_mean_ci
from .aggregation import group_signals

def merge_phot_bonsai_cam(
    phot_df: pd.DataFrame,
    bonsai_df: pd.DataFrame,
    freq_label: str,
    cam_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aligns raw photometry fluorometer samples with Bonsai tracking timestamps
    and Basler camera exposure frame indices using nearest-neighbor timestamp matching.
    """
    phot = phot_df.copy()
    bonsai_col = f"TimestampBonsai_{freq_label}"
    phot[bonsai_col] = bonsai_df[bonsai_col]

    merged = pd.merge_asof(
        left=phot,
        right=cam_df[["cam_frame_num", "BonsaiTrackingTimestamp"]],
        left_on=bonsai_col,
        right_on="BonsaiTrackingTimestamp",
        direction="nearest",
    )
    return merged.drop(columns=[bonsai_col])

__all__ = ["group_signals", "compute_mean_ci", "merge_phot_bonsai_cam"]
