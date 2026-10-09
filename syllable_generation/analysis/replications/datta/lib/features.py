"""
Feature Engineering: Kinematic derivatives and signal metadata extraction.
"""
import re

import numpy as np
import pandas as pd

from . import config


def _grad(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calculates gradient (derivative) of y with respect to t."""
    y = y.astype(float)
    t = t.astype(float)
    if len(y) < 2:
        return np.full_like(y, np.nan, dtype=float)
    return np.gradient(y, t)

def peak_rate_cross(x: pd.Series, threshold: float = 1.96) -> float:
    """Calculates the rate of threshold crossings (normalized by valid transitions)."""
    v = x.to_numpy()
    if v.size <= 1: return np.nan

    # Shifted arrays for transition detection
    v_curr = v[:-1]
    v_next = v[1:]

    # Logic: (Below -> Above) AND (Not NaN -> Not NaN)
    crossing = (v_curr < threshold) & (v_next > threshold) & np.isfinite(v_curr) & np.isfinite(v_next)

    # Normalize by number of VALID transitions
    valid_transitions = np.isfinite(v_curr) & np.isfinite(v_next)
    n_valid = valid_transitions.sum()

    if n_valid == 0:
        return np.nan

    return crossing.sum() / n_valid

def calculate_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Adds acceleration, jerk, and transition columns to the dataframe."""

    # Ensure necessary columns exist
    if "velocity_px_s" not in df.columns or "SystemTimestamp" not in df.columns:
        return df # Or raise error?

    # Labels and Transitions
    if "syllable" in df.columns:
        labels = df["syllable"].replace({-5: pd.NA, -1: pd.NA}).astype("Int64")
        df["labels"] = labels
        # Note: We don't dropna here, we let the caller handle row filtering if needed,
        # but original script dropped NaNs immediately. Let's stick to original logic:
        df = df.dropna(subset=["labels"]).copy()

        change = df["labels"].diff().ne(0)
        df["transition"] = change.fillna(False).astype(float)
        df["syllable_number"] = df["transition"].cumsum().astype("uint32")

        # Duration (via GroupBy)
        syl_size = df.groupby(["syllable_number"], observed=True).size().rename("duration")
        df = df.set_index(["syllable_number"]).join(syl_size).reset_index()

    # Kinematics Derivatives
    df["timestamp"] = df["SystemTimestamp"].astype(float)
    vel = df["velocity_px_s"].to_numpy()
    t   = df["timestamp"].to_numpy()

    acc = _grad(vel, t)
    jerk = _grad(acc, t)

    df["acceleration"] = acc
    df["jerk"] = jerk

    return df

def extract_signal_metadata(df: pd.DataFrame, session_metadata: dict) -> list[dict]:
    """Parses signal columns and metadata to return a list of defined signals."""
    sig_annot_rows = []

    # Identify Signal Columns
    all_sig_cols = sorted([c for c in df.columns if re.match(config.SIGNAL_COL_REGEX, c)])

    smeta = session_metadata.get("signal_meta", {}) or {}

    for c in all_sig_cols:
        k = c.split("_")[1] # signal_X_zscore -> X
        sig_key = f"signal_{k}"
        m = smeta.get(sig_key, {})

        brain_reg = m.get("brain_reg", "UNK")

        if brain_reg == "UNK":
            continue

        sig_annot_rows.append({
            "signal": c,
            "base_signal": sig_key,
            "brain_reg": brain_reg,
            "hemisphere": m.get("hemisphere", "UNK"),
            "fiber_num": m.get("fiber_num", "UNK"),
            "orig_letter": m.get("orig_letter", "UNK")
        })

    return sig_annot_rows
