# FILE: projects/jonathan_openfield_example/enrich.py
import pandas as pd
from pathlib import Path
import numpy as np
import re

def _val(row, name):
    """
    Aux helper to robustly find a column in a CSV row (fuzzy match).
    Matches 'Left Hemisphere', 'left-hemisphere', etc.
    """
    target = name.lower().replace(" ", "").replace("-", "")
    for c in row.index:
        if c.lower().replace(" ", "").replace("-", "") == target:
            return str(row[c]).strip()
    return ""

def load_animal_map(ctx):
    """
    Robustly finds animals.csv and returns {aid: {'L': ..., 'R': ..., 'Geno': ...}}.
    """
    enrich_dir = Path(__file__).resolve().parent
    overrides_csv = enrich_dir / "overrides" / "animals.csv"
    
    ctx_csv = Path(ctx.csv_root) / "animals.csv" if ctx.csv_root else None
    legacy_csv = Path(ctx.csv_root) / "Jonathan_IFxDN_csvs" / "animals.csv" if ctx.csv_root else None

    candidates = [overrides_csv, ctx_csv, legacy_csv]
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
                'L': _val(row, "Left Hemisphere"),
                'R': _val(row, "Right Hemisphere"),
                'Geno': _val(row, "Geno")
            }
    return mapping


def load_exclusions_map(overrides_dir):
    """
    Loads manual exclusions.csv.
    """
    p = overrides_dir / "exclusions.csv"
    if not p.exists():
        return []
    
    try:
        df = pd.read_csv(p, dtype=str).fillna("")
        if df.empty: return []
    except pd.errors.EmptyDataError:
        return []
    
    print(f"[Enrich] Loading manual exclusions from: {p}")
    
    exclusions = []
    for _, row in df.iterrows():
        aid = str(row.get("ID", "")).strip()
        reg = str(row.get("Region", "")).strip()
        raw_intervals = str(row.get("Intervals", ""))
        
        if not aid or not reg or not raw_intervals:
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
                start, end = map(float, part.split('-'))
                parsed_intervals.append((mode, start, end))
            except ValueError:
                pass
        
        if parsed_intervals:
            exclusions.append({"aid": aid, "region": reg, "intervals": parsed_intervals})
            
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
    Uses 'signal_meta' (standardized) if available, matching exclusions.csv format.
    """
    sm = session.metadata.get("signal_meta", {})
    entry = sm.get(sig_key)
    
    if entry:
        br = entry.get("brain_reg")
        hemi = entry.get("hemisphere")
        if br and hemi:
            return f"{br}-{hemi}"
    
    # Fallback to raw string if 'signal_meta' isn't fully populated
    raw = session.metadata.get(sig_key)
    if raw and "-" in raw:
        return raw
    return None


def get_region_to_cols_map(session):
    """
    Maps Region Label -> List of Columns.
    FIX: Maps via logical signal keys (signal_0) rather than physical fiber IDs.
    """
    df = session.photometry.get("combined_filtered")
    if df is None: return {}
    
    region_to_cols = {}
    
    # Iterate through logical signals 0..3
    for i in range(4): 
        sig_key = f"signal_{i}"
        
        label = _get_signal_label(session, sig_key)
        if not label: continue
        
        # Identify Columns by LOGICAL KEY
        # This correctly finds "signal_0" even if it came from Fiber 4 (G4)
        relevant_cols = [
            c for c in df.columns 
            if c == sig_key or c.startswith(f"{sig_key}_")
        ]
        
        if label not in region_to_cols: 
            region_to_cols[label] = []
        region_to_cols[label].extend(relevant_cols)
                    
    return region_to_cols


def apply_auto_masks(ctx, session, npz_data):
    """
    Applies AUTOMATED exclusions from NPZ.
    """
    if npz_data is None: return

    df = session.photometry.get("combined_filtered")
    if df is None or getattr(df, "empty", True): return
    
    sid = session.session_id
    npz_keys = list(npz_data.files)
    match_key = next((k for k in npz_keys if sid in k), None)
    
    if not match_key:
        return

    valid_indices = npz_data[match_key]
    
    # Identify all signal columns for this animal
    region_map = get_region_to_cols_map(session)
    all_signal_cols = []
    for cols in region_map.values():
        all_signal_cols.extend(cols)
    
    if not all_signal_cols: return
    
    valid_cols = [c for c in set(all_signal_cols) if c in df.columns]
    if not valid_cols: return
    
    idx_col_name = getattr(ctx.cfg, "frame_index_col_name", "frame_idx")
    if idx_col_name not in df.columns: return
    
    current_frames = df[idx_col_name].values
    
    is_valid = np.isin(current_frames, valid_indices)
    to_nuke = ~is_valid
    
    if to_nuke.any():
        df.loc[to_nuke, valid_cols] = np.nan
        session.photometry["combined_filtered"] = df


def apply_manual_exclusions(ctx, session, exclusions):
    """
    Applies MANUAL exclusions from CSV.
    """
    if not exclusions: return

    df = session.photometry.get("combined_filtered")
    if df is None or getattr(df, "empty", True): return

    region_to_cols = get_region_to_cols_map(session)
    my_aid = str(session.metadata.get("animal_id", "")).strip()
    
    ts_col_name = getattr(ctx.cfg, "phot_system_ts_col", "SystemTimestamp")
    idx_col_name = getattr(ctx.cfg, "frame_index_col_name", "frame_idx")

    ts_vec = df[ts_col_name].values if ts_col_name in df.columns else None
    idx_vec = df[idx_col_name].values if idx_col_name in df.columns else df.index.values
    
    for exc in exclusions:
        if exc["aid"] != my_aid:
            continue
        
        target_cols = region_to_cols.get(exc["region"])
        if not target_cols:
            continue

        for (mode, start, end) in exc["intervals"]:
            mask = None
            if mode == "ts" and ts_vec is not None:
                mask = (ts_vec >= start) & (ts_vec <= end)
            elif mode == "idx":
                mask = (idx_vec >= start) & (idx_vec <= end)
            
            if mask is not None and mask.any():
                valid_cols = [c for c in target_cols if c in df.columns]
                if valid_cols:
                    df.loc[mask, valid_cols] = np.nan

    session.photometry["combined_filtered"] = df


def enrich(ctx, sessions, overrides_dir=None, skip_exclusions=False):
    animal_map = load_animal_map(ctx)
    
    if not overrides_dir:
        overrides_dir = Path(__file__).resolve().parent / "overrides"
    
    if skip_exclusions:
        print("[Enrich] FORCE_DIRTY enabled: Skipping exclusions.")
        manual_exclusions = []
        auto_mask_data = None
    else:
        manual_exclusions = load_exclusions_map(overrides_dir)
        auto_mask_data = load_auto_mask_data(overrides_dir)

    count = 0
    for s in sessions:
        md = s.metadata
        aid = str(md.get("animal_id") or "").strip()
        
        # 1. Apply Animal Metadata
        if aid in animal_map:
            info = animal_map[aid]
            if info['Geno']: md["genotype"] = info['Geno'].upper()
            if info['L']: 
                md["signal_0"] = info['L']
                md["signal_1"] = info['L']
            if info['R']: 
                md["signal_2"] = info['R']
                md["signal_3"] = info['R']
        
        # 2. Apply Exclusions (The Union)
        if "combined_filtered" in s.photometry:
            # A. Auto (Machine)
            apply_auto_masks(ctx, s, auto_mask_data)
            
            # B. Manual (Human)
            apply_manual_exclusions(ctx, s, manual_exclusions)
        
        count += 1

    print(f"[Enrich] Enriched {count} sessions." + 
          (f" Applied manual exclusions." if manual_exclusions else "") +
          (f" Applied auto masks." if auto_mask_data is not None else ""))
    
    return sessions