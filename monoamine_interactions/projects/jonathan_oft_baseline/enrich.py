# FILE: monoamine_interactions/jonathan_oft_baseline/enrich.py
import pandas as pd
from pathlib import Path
import numpy as np
import os
import re

def _val(row, name):
    target = name.lower().replace(" ", "").replace("-", "")
    for c in row.index:
        if c.lower().replace(" ", "").replace("-", "") == target:
            return str(row[c]).strip()
    return ""

def _is_unknown(val):
    """Check if value is empty or some variant of 'unknown' or 'unk'."""
    if not val: return True
    v = str(val).strip().lower()
    return v in ["unk", "unknown", "nan", "none", "n/a", "na"]

def load_animal_map(ctx):
    """
    Loads animals.csv: {mouse_id: {data...}}
    """
    enrich_dir = Path(__file__).resolve().parent
    target = enrich_dir / "overrides" / "animals.csv"
    
    if not target.exists():
        print(f"[Enrich] ERROR: animals.csv NOT found at {target}")
        return {}
    
    print(f"[Enrich] Loaded map from: {target}")
    df = pd.read_csv(target, dtype=str).fillna("")
    mapping = {}
    
    for _, row in df.iterrows():
        aid = _val(row, "mouse_id")
        if aid:
            mapping[aid] = {
                'prefix': _val(row, "folder_prefix"),
                'geno': _val(row, "genotype"),
                'f1': _val(row, "fiber1_id"),
                'r1': _val(row, "region1_sens"),
                'f2': _val(row, "fiber2_id"),
                'r2': _val(row, "region2_sens")
            }
    return mapping

def parse_ids_from_filename(filename):
    """
    Parses 'T1_4.8.e.e_...' or 'T2_2.6.10.e_...'
    Returns list of 4 IDs (or None) corresponding to positions:
    0: TL, 1: TR, 2: BL, 3: BR (Reading Order assumption)
    """
    # Regex for T#_ID.ID.ID.ID
    # IDs can be numbers or 'e'
    m = re.search(r"(T\d+)_([0-9e]+)\.([0-9e]+)\.([0-9e]+)\.([0-9e]+)", filename)
    if m:
        return [m.group(2), m.group(3), m.group(4), m.group(5)]
    return None

def get_chamber_position(chamber_name):
    """
    Maps chamber name to Reading Order Index (0=TL, 1=TR, 2=BL, 3=BR)
    """
    c = chamber_name.lower()
    if "top_left" in c: return 0
    if "bottom_left" in c: return 1
    if "bottom_right" in c: return 2
    if "top_right" in c: return 3
    return -1

def load_exclusions(ctx):
    """
    Loads exclusions.csv keyed by SessionID (preferred) or ID (legacy animal_id).
    Format: SessionID,ID,Region,Intervals  OR  ID,Region,Intervals
    Returns: { session_id_or_animal_id: { region: [interval_str, ...] } }
    """
    enrich_dir = Path(__file__).resolve().parent
    target = enrich_dir / "overrides" / "exclusions.csv"
    
    if not target.exists():
        return {}
    
    df = pd.read_csv(target, dtype=str).fillna("")
    exclusions = {}
    
    for _, row in df.iterrows():
        # Prefer SessionID column (written by GUI), fall back to ID
        sid = _val(row, "SessionID")
        aid = _val(row, "ID")
        key = sid if sid else aid  # session_id takes priority
        reg = _val(row, "Region")
        intervals_raw = _val(row, "Intervals")
        
        if key and reg and intervals_raw:
            if key not in exclusions: exclusions[key] = {}
            if reg not in exclusions[key]: exclusions[key][reg] = []
            exclusions[key][reg].append(intervals_raw)
            
    return exclusions

def get_region_to_cols_map(session):
    """
    Maps Region Label (e.g. VTA_NE-Left) -> List of Columns.
    """
    df = session.photometry.get("combined_filtered")
    if df is None: return {}
    
    region_to_cols = {}
    sm = session.metadata.get("signal_meta", {})
    
    # Iterate through logical signals 0..3
    for i in range(4): 
        sig_key = f"signal_{i}"
        if sig_key not in sm: continue
        
        meta = sm[sig_key]
        reg = meta.get("region", session.metadata.get(sig_key))
        if not reg: continue
        
        hemi = meta.get("hemisphere", "")
        label = f"{reg}-{hemi}" if hemi else reg
        
        # Identify Columns by LOGICAL KEY
        # Logic: direct match or prefix match (e.g. signal_0_dff)
        relevant_cols = [
            c for c in df.columns 
            if c == sig_key or c.startswith(f"{sig_key}_")
        ]
        
        if label not in region_to_cols: 
            region_to_cols[label] = []
        region_to_cols[label].extend(relevant_cols)
                    
    return region_to_cols

def apply_manual_exclusions_to_session(session, exclusions):
    """
    Applies MANUAL exclusions from CSV by NaN-ing out data.
    exclusions: list of dicts {region: [...], ...} or similar structure.
    Here we expect the specific list of intervals for this session/animal.
    """
    if not exclusions: return

    df = session.photometry.get("combined_filtered")
    if df is None or getattr(df, "empty", True): return

    region_to_cols = get_region_to_cols_map(session)
    
    # Get column vectors for indexing
    # We assume standard column names for now
    ts_col_name = "SystemTimestamp"
    idx_col_name = "frame_idx"

    ts_vec = df[ts_col_name].values if ts_col_name in df.columns else None
    idx_vec = df[idx_col_name].values if idx_col_name in df.columns else df.index.values
    
    for reg_name, intervals_list in exclusions.items():
        target_cols = region_to_cols.get(reg_name)
        if not target_cols: continue
        
        for raw_item in intervals_list:
            for raw in raw_item.split(";"):
                raw = raw.strip()
                if not raw: continue
                # Parse interval string locally if not parsed yet
                # Format: "ts:start-end" or "idx:start-end" (or plain "start-end" -> idx)
                mode = "idx"
                part = raw
                if "ts:" in part:
                    mode = "ts"; part = part.replace("ts:", "")
                elif "idx:" in part:
                    mode = "idx"; part = part.replace("idx:", "")
                
                try:
                    start, end = map(float, part.split('-'))
                except ValueError:
                    continue
    
                mask = None
                if mode == "ts" and ts_vec is not None:
                    mask = (ts_vec >= start) & (ts_vec <= end)
                elif mode == "idx":
                    mask = (idx_vec >= start) & (idx_vec <= end)
                
                if mask is not None and mask.any():
                    valid_cols = [c for c in target_cols if c in df.columns]
                    if valid_cols:
                        df.loc[mask, valid_cols] = np.nan
                        # print(f"[Enrich] NaN'd {len(valid_cols)} cols for {reg_name} in range {start}-{end} ({mode})")

    session.photometry["combined_filtered"] = df

def enrich(ctx, sessions, overrides_dir=None, skip_exclusions=False):
    """
    Enrichment logic. Called twice:
    Pass 1 (Pre-Realize): Set fiber_indices_effective.
    Pass 2 (Post-Realize): Apply region labels and exclusions based on actual signal keys.
    """
    print(f"[DEBUG] enrich() called with {len(sessions)} sessions.")
    animal_map = load_animal_map(ctx)
    exclusion_map = load_exclusions(ctx) if not skip_exclusions else {}
    
    count = 0
    for s in sessions:
        md = s.metadata
        
        # --- PASS 0: Animal ID Resolving (Always) ---
        aid = md.get("animal_id")
        if not aid:
            orig_path = md.get("original_video_path")
            if orig_path:
                fname = Path(orig_path).name
                ids = parse_ids_from_filename(fname)
                chamber = md.get("chamber", "")
                pos_idx = get_chamber_position(chamber)
                if ids and pos_idx >= 0:
                    possible_id = ids[pos_idx]
                    if possible_id != 'e':
                        aid = possible_id
                        md["animal_id"] = aid

        # --- PASS 1: Pre-Realize (No signal_meta yet) ---
        if "signal_meta" not in md:
            if aid and aid in animal_map:
                info = animal_map[aid]
                if not _is_unknown(info['geno']):
                    md["genotype"] = info['geno']
                
                used_indices = []
                for f_id in [info['f1'], info['f2']]:
                    if f_id and not _is_unknown(f_id):
                        m = re.search(r"\d+", f_id)
                        if m: used_indices.append(int(m.group()))
                
                if used_indices:
                    md["fiber_indices_effective"] = sorted(list(set(used_indices)))
            count += 1
            continue

        # --- PASS 2: Post-Realize (Loader has created signal_meta) ---
        if aid and aid in animal_map:
            info = animal_map[aid]
            # Map of FiberNum (int) -> RegionLabel
            fiber_to_region = {}
            for f_str, r_lbl in [(info['f1'], info['r1']), (info['f2'], info['r2'])]:
                if f_str and not _is_unknown(r_lbl):
                    m = re.search(r"(\d+)", f_str)
                    if m: fiber_to_region[int(m.group(1))] = r_lbl

            # Clear any stale logical signal keys in metadata
            for i in range(4):
                md.pop(f"signal_{i}", None)

            # Iterate through actual signal keys created by loader
            sm = md.get("signal_meta", {})
            for sig_key, entry in sm.items():
                # fiber_num from loader can be string or int
                raw_fn = entry.get("fiber_num")
                if raw_fn is None: continue
                fiber_num = int(raw_fn)

                if fiber_num in fiber_to_region:
                    region = fiber_to_region[fiber_num]
                    md[sig_key] = region
                    entry["region"] = region
                    entry["brain_reg"] = region
                    
                    # Add Hemisphere based on Fiber 1 (Left) or Fiber 2 (Right)
                    # We match fiber_num to info['f1'] or info['f2']
                    f1_num = int(re.search(r"(\d+)", info['f1']).group(1)) if info['f1'] else -1
                    f2_num = int(re.search(r"(\d+)", info['f2']).group(1)) if info['f2'] else -1
                    
                    if fiber_num == f1_num:
                        entry["hemisphere"] = "Left"
                    elif fiber_num == f2_num:
                        entry["hemisphere"] = "Right"

            # 3. Apply Exclusions — match by session_id first, then animal_id
            sid = s.session_id
            excl_key = sid if sid in exclusion_map else (aid if aid in exclusion_map else None)
            if excl_key:
                sess_excl = exclusion_map[excl_key]
                if "exclusions" not in md: md["exclusions"] = {}
                
                # Store metadata
                for sig_key, meta in sm.items():
                    reg = meta.get("region", md.get(sig_key))
                    hemi = meta.get("hemisphere", "")
                    full_label = f"{reg}-{hemi}" if hemi else reg
                    
                    if full_label and full_label in sess_excl:
                        md["exclusions"][sig_key] = sess_excl[full_label]

                # APPLY Exclusions (NaN out data in combined_filtered)
                print(f"[Enrich] Applying exclusions for {excl_key} (key type: {'session' if excl_key == sid else 'animal'})")
                apply_manual_exclusions_to_session(s, sess_excl)

        elif aid:
             print(f"[DEBUG] WARNING: Mouse {aid} not found in animals.csv")
        
        count += 1

    print(f"[Enrich] Enriched {count} sessions.")
    return sessions
