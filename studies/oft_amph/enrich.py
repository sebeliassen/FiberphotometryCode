"""Recovered AMPH metadata helpers; historical empirical execution is gated.

The exact original source is preserved as non-importable text under historical/.
Its enrichment is incomplete and its frame-clock inference remains unresolved.
"""

from pathlib import Path

import numpy as np
import pandas as pd

STATUS = "metadata_only; execution gated"


class AMPHPolicyError(RuntimeError):
    """AMPH execution requires a separately reviewed complete clock policy."""


_EXECUTION_GATE = (
    "Historical AMPH enrichment is incomplete: the preserved source has no "
    "return and does not apply its mask helpers. Execution requires a separately "
    "reviewed explicit frame-clock policy; clock conversion cannot be inferred "
    "from index magnitude."
)


def _val(row, name):
    """
    Aux helper to robustly find a column in a CSV row (fuzzy match).
    """
    target = name.lower().replace(" ", "").replace("-", "")
    for c in row.index:
        if c.lower().replace(" ", "").replace("-", "") == target:
            return str(row[c]).strip()
    return ""


def load_animal_map(ctx):
    enrich_dir = Path(__file__).resolve().parent
    overrides_csv = enrich_dir / "overrides" / "animals.csv"

    ctx_csv = Path(ctx.csv_root) / "animals.csv" if ctx.csv_root else None

    candidates = [overrides_csv, ctx_csv]
    target = next((p for p in candidates if p is not None and p.exists()), None)

    if not target:
        print(f"[Enrich] ERROR: animals.csv NOT found. Checked: {[str(p) for p in candidates]}")
        return {}

    print(f"[Enrich] Loaded map from: {target}")

    df = pd.read_csv(target, dtype=str).fillna("")
    mapping = {}

    for _, row in df.iterrows():
        aid = _val(row, "ID")
        if aid:
            mapping[aid] = {
                "L": _val(row, "Left Hemisphere"),
                "R": _val(row, "Right Hemisphere"),
                "Geno": _val(row, "Geno"),
            }
    return mapping


def load_sessions_map(overrides_dir):
    """
    Loads session-level overrides from sessions.csv.
    Expected columns: SessionID, AnimalID, Drug, Dose_mg_kg, InjectionFrame, Exclude
    """
    p = overrides_dir / "sessions.csv"
    if not p.exists():
        return {}

    try:
        df = pd.read_csv(p, dtype=str).fillna("")
        if df.empty:
            return {}
    except:  # noqa: E722 - Preserve the pinned AMPH parser broad-catch behavior.
        return {}

    print(f"[Enrich] Loading session overrides from: {p}")
    mapping = {}
    for _, row in df.iterrows():
        sid = str(row.get("SessionID", "")).strip()
        if sid:
            mapping[sid] = {
                "animal_id": str(row.get("AnimalID", "")).strip(),
                "drug": str(row.get("Drug", "")).strip(),
                "dose": str(row.get("Dose_mg_kg", "")).strip(),
                "inj_delay": str(row.get("InjectionDelay_sec", "1200")).strip(),
                "exclude": str(row.get("Exclude", "False")).lower() == "true",
            }
    return mapping


def load_exclusions_map(overrides_dir):
    """
    Loads manual exclusions.csv with support for SessionID.
    Expected CSV columns: SessionID, ID, Region, Intervals
    """
    p = overrides_dir / "exclusions.csv"
    if not p.exists():
        return []

    try:
        df = pd.read_csv(p, dtype=str).fillna("")
        if df.empty:
            return []
    except pd.errors.EmptyDataError:
        return []

    print(f"[Enrich] Loading manual exclusions from: {p}")

    exclusions = []
    for _, row in df.iterrows():
        # Load SessionID (support both "SessionID" and "Session" as headers)
        sid = str(row.get("SessionID", "")).strip()
        if not sid:
            sid = str(row.get("Session", "")).strip()

        aid = str(row.get("ID", "")).strip()
        reg = str(row.get("Region", "")).strip()
        raw_intervals = str(row.get("Intervals", ""))

        # We require at least Region and Intervals.
        # We need EITHER SessionID OR AnimalID to match (preferably SessionID).
        if (not sid and not aid) or not reg or not raw_intervals:
            continue

        parsed_intervals = []
        for part in raw_intervals.split(";"):
            part = part.strip()
            mode = "idx"
            if "ts:" in part:
                mode = "ts"
                part = part.replace("ts:", "")
            elif "idx:" in part:
                mode = "idx"
                part = part.replace("idx:", "")

            try:
                start, end = map(float, part.split("-"))
                parsed_intervals.append((mode, start, end))
            except ValueError:
                pass

        if parsed_intervals:
            exclusions.append(
                {
                    "sid": sid,  # Store Session ID
                    "aid": aid,  # Store Animal ID
                    "region": reg,
                    "intervals": parsed_intervals,
                }
            )

    return exclusions


def load_auto_mask_data(overrides_dir):
    """
    Loads auto_mask.npz if it exists.
    """
    p = overrides_dir / "auto_mask.npz"
    if not p.exists():
        return None

    try:
        print(f"[Enrich] Loading auto-masks from: {p}")
        return np.load(p)
    except Exception as e:
        print(f"[Enrich] Failed to load auto_mask.npz: {e}")
        return None


def _get_signal_label(session, sig_key):
    """
    Aux helper to extract the region label (e.g. 'DLS-L') for a signal.
    """
    sm = session.metadata.get("signal_meta", {})
    entry = sm.get(sig_key)

    if entry:
        br = entry.get("brain_reg")
        hemi = entry.get("hemisphere")
        if br and hemi:
            return f"{hemi}-{br}"

    # Fallback
    raw = session.metadata.get(sig_key)
    if raw and "-" in raw:
        return raw
    return None


def get_region_to_cols_map(session):
    """
    Maps Region Label -> List of Columns.
    """
    df = session.photometry.get("combined_filtered")
    if df is None:
        return {}

    region_to_cols = {}

    # Iterate through logical signals 0..3
    for i in range(4):
        sig_key = f"signal_{i}"

        label = _get_signal_label(session, sig_key)
        if not label:
            continue

        # Identify Columns by LOGICAL KEY
        relevant_cols = [c for c in df.columns if c == sig_key or c.startswith(f"{sig_key}_")]

        if label not in region_to_cols:
            region_to_cols[label] = []
        region_to_cols[label].extend(relevant_cols)

    return region_to_cols


def apply_auto_masks(ctx, session, npz_data):
    raise AMPHPolicyError(_EXECUTION_GATE)


def apply_manual_exclusions(ctx, session, exclusions, auto_mask_data=None):
    raise AMPHPolicyError(_EXECUTION_GATE)


def enrich(ctx, sessions, overrides_dir=None, skip_exclusions=False):
    raise AMPHPolicyError(_EXECUTION_GATE)
