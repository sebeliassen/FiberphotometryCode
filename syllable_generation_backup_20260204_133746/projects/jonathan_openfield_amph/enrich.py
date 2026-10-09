# FILE: projects/jonathan_openfield_amph/enrich.py
import pandas as pd
from pathlib import Path
import numpy as np
import regex as re

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
                'L': _val(row, "Left Hemisphere"),
                'R': _val(row, "Right Hemisphere"),
                'Geno': _val(row, "Geno")
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
        if df.empty: return {}
    except:
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
                "exclude": str(row.get("Exclude", "False")).lower() == "true"
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
        if df.empty: return []
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
                mode = "ts"; part = part.replace("ts:", "")
            elif "idx:" in part:
                mode = "idx"; part = part.replace("idx:", "")
            
            try:
                start, end = map(float, part.split('-'))
                parsed_intervals.append((mode, start, end))
            except ValueError:
                pass
        
        if parsed_intervals:
            exclusions.append({
                "sid": sid,     # Store Session ID
                "aid": aid,     # Store Animal ID
                "region": reg, 
                "intervals": parsed_intervals
            })
            
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
    if df is None: return {}
    
    region_to_cols = {}
    
    # Iterate through logical signals 0..3
    for i in range(4): 
        sig_key = f"signal_{i}"
        
        label = _get_signal_label(session, sig_key)
        if not label: continue
        
        # Identify Columns by LOGICAL KEY
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
    
    # Keep valid, NaN out the rest
    is_valid = np.isin(current_frames, valid_indices)
    to_nuke = ~is_valid
    
    if to_nuke.any():
        df.loc[to_nuke, valid_cols] = np.nan
        session.photometry["combined_filtered"] = df

def apply_manual_exclusions(ctx, session, exclusions, auto_mask_data=None):
    """
    Applies MANUAL exclusions from CSV.
    If auto_mask_data is provided, indices are treated as Video Frames (index into VFI) 
    and translated to Photometry Frames.
    """
    if not exclusions: return

    df = session.photometry.get("combined_filtered")
    if df is None or getattr(df, "empty", True): return

    region_to_cols = get_region_to_cols_map(session)
    my_sid = str(session.session_id).strip()
    my_aid = str(session.metadata.get("animal_id", "")).strip()
    
    # Pre-load VFI if available for translation
    vfi = None
    if auto_mask_data is not None:
        match_key = next((k for k in auto_mask_data.keys() if my_sid in k), None)
        if match_key:
            vfi = auto_mask_data[match_key]
            # --- HEURISTIC SCALING (Must match apply_auto_masks / Entry calculation) ---
            if len(vfi) > 0 and vfi.max() < 300000:
                vfi = np.round(vfi * 2.0).astype(int)

    # Get column vectors for indexing
    ts_col_name = getattr(ctx.cfg, "phot_system_ts_col", "SystemTimestamp")
    idx_col_name = getattr(ctx.cfg, "frame_index_col_name", "frame_idx")

    ts_vec = df[ts_col_name].values if ts_col_name in df.columns else None
    idx_vec = df[idx_col_name].values if idx_col_name in df.columns else df.index.values
    
    for exc in exclusions:
        if exc["sid"] and exc["sid"] != my_sid: continue
        if exc["aid"] and exc["aid"] != my_aid: continue
        
        target_cols = region_to_cols.get(exc["region"])
        if not target_cols: continue

        for (mode, start, end) in exc["intervals"]:
            mask = None
            if mode == "ts" and ts_vec is not None:
                mask = (ts_vec >= start) & (ts_vec <= end)
            elif mode == "idx":
                if vfi is not None:
                    # Translate Video Frame Index -> Photometry Frame Index
                    s_idx = max(0, int(start))
                    e_idx = min(len(vfi) - 1, int(end))
                    target_phot_indices = vfi[s_idx:e_idx+1]
                    mask = np.isin(idx_vec, target_phot_indices)
                else:
                    # Fallback to absolute indexing if no VFI
                    mask = (idx_vec >= start) & (idx_vec <= end)
            
            if mask is not None and mask.any():
                valid_cols = [c for c in target_cols if c in df.columns]
                if valid_cols:
                    df.loc[mask, valid_cols] = np.nan

    session.photometry["combined_filtered"] = df

def enrich(ctx, sessions, overrides_dir=None, skip_exclusions=False):
    if not overrides_dir:
        overrides_dir = Path(__file__).resolve().parent / "overrides"
        
    animal_map = load_animal_map(ctx)
    session_map = load_sessions_map(overrides_dir)
    
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
        
        # 1.5 Apply Session Overrides (Drug, Dose, etc.)
        sid = s.session_id
        if sid in session_map:
            over = session_map[sid]
            if over["animal_id"]: md["animal_id"] = over["animal_id"]
            if over["drug"]: md["drug"] = over["drug"]
            if over["dose"]: md["dose_mg_kg"] = over["dose"]
            
            # Exclusion logic
            if over["exclude"]:
                print(f"[Enrich] EXCLUDING session: {sid}")
                md["exclude"] = True

        # 2. Dynamic Injection Frame Calculation
        # Load Auto-Mask to find Entry Frame
        if auto_mask_data:
            match_key = next((k for k in auto_mask_data.keys() if sid in k), None)
            if match_key:
                indices = auto_mask_data[match_key]
            if len(indices) > 0:
                # --- HEURISTIC CHECK FOR 40Hz vs 80Hz ---
                # Compare max index to session length (approx 450k frames)
                # If max index < 300k, it's likely 40Hz -> Scale x2
                max_idx = indices.max()
                if max_idx < 300000: # Heuristic threshold
                    scale = 2.0
                    indices = np.round(indices * scale).astype(int)
                
                # First valid frame = Entry Time
                start_frame = indices.min()
                
                # Get Delay from CSV (default 1200s = 20 mins)
                inj_delay_sec = 1200
                if sid in session_map:
                    raw_delay = session_map[sid].get("inj_delay", "")
                    if raw_delay: inj_delay_sec = float(raw_delay)
                
                # Calculate Absolute Injection Frame
                # Frame = Entry + (Seconds * 80Hz)
                calc_inj = start_frame + int(inj_delay_sec * 80)
                
                md["injection_frame"] = int(calc_inj)
                md["entry_frame"] = int(start_frame) # Store for reference