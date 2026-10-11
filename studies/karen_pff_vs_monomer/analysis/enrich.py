# FILE: monoamine_interactions/projects/karen_project/enrich.py
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
    if not val: return True
    v = str(val).strip().lower()
    return v in ["unk", "unknown", "nan", "none", "n/a", "na"]

def load_animal_map(overrides_dir):
    target = overrides_dir / "animals.csv"
    if not target.exists():
        print(f"[Enrich] ERROR: animals.csv NOT found at {target}")
        return {}
    
    print(f"[Enrich] Loaded animal map from: {target}")
    df = pd.read_csv(target, dtype=str).fillna("")
    mapping = {}
    
    for _, row in df.iterrows():
        aid = _val(row, "mouse_id")
        if aid:
            mapping[aid] = {
                'geno': _val(row, "genotype"),
                'L': _val(row, "left_hemisphere"),
                'R': _val(row, "right_hemisphere")
            }
    return mapping

def load_sessions_map(overrides_dir):
    target = overrides_dir / "sessions.csv"
    if not target.exists():
        return {}
    
    print(f"[Enrich] Loaded sessions map from: {target}")
    df = pd.read_csv(target, dtype=str).fillna("")
    mapping = {}
    for _, row in df.iterrows():
        sid = str(row.get("SessionID", "")).strip()
        if sid:
            mapping[sid] = {
                "animal_id": str(row.get("AnimalID", "")).strip(),
                "exclude": str(row.get("Exclude", "False")).lower() == "true"
            }
    return mapping

def load_exclusions(overrides_dir):
    target = overrides_dir / "exclusions.csv"
    if not target.exists():
        return {}
    
    try:
        df = pd.read_csv(target, dtype=str).fillna("")
        if df.empty: return {}
    except pd.errors.EmptyDataError:
        return {}
        
    exclusions = {}
    for _, row in df.iterrows():
        sid = _val(row, "SessionID")
        aid = _val(row, "ID")
        key = sid if sid else aid
        reg = _val(row, "Region")
        intervals_raw = _val(row, "Intervals")
        
        if key and reg and intervals_raw:
            if key not in exclusions: exclusions[key] = {}
            if reg not in exclusions[key]: exclusions[key][reg] = []
            exclusions[key][reg].append(intervals_raw)
            
    return exclusions

def load_auto_mask_data(overrides_dir):
    p = overrides_dir / "auto_mask.npz"
    if not p.exists():
        return None
    try:
        print(f"[Enrich] Loading auto-masks from: {p}")
        return np.load(p)
    except Exception as e:
        print(f"[Enrich] Failed to load auto_mask.npz: {e}")
        return None

def get_region_to_cols_map(session):
    df = session.photometry.get("combined_filtered")
    if df is None: return {}
    
    region_to_cols = {}
    sm = session.metadata.get("signal_meta", {})
    
    for i in range(4): 
        sig_key = f"signal_{i}"
        if sig_key not in sm: continue
        
        meta = sm[sig_key]
        reg = meta.get("region", session.metadata.get(sig_key))
        if not reg: continue
        
        hemi = meta.get("hemisphere", "")
        label = f"{reg}-{hemi}" if hemi else reg
        
        relevant_cols = [
            c for c in df.columns 
            if c == sig_key or c.startswith(f"{sig_key}_")
        ]
        
        if label not in region_to_cols: 
            region_to_cols[label] = []
        region_to_cols[label].extend(relevant_cols)
                    
    return region_to_cols

def apply_auto_masks(ctx, session, npz_data):
    if npz_data is None: return

    df = session.photometry.get("combined_filtered")
    if df is None or getattr(df, "empty", True): return
    
    sid = session.session_id
    npz_keys = list(npz_data.files)
    match_key = next((k for k in npz_keys if sid in k), None)
    
    if not match_key:
        return

    valid_indices = npz_data[match_key]
    
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

def apply_manual_exclusions_to_session(session, exclusions):
    if not exclusions: return

    df = session.photometry.get("combined_filtered")
    if df is None or getattr(df, "empty", True): return

    region_to_cols = get_region_to_cols_map(session)
    
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

    session.photometry["combined_filtered"] = df

def _parse_condition_and_drug(md):
    """
    Identifies experimental condition, drug, and weeks post-injection (PI).
    """
    p_path = str(md.get("photometry_csv_path", "") or "")
    t_path = str(md.get("alltimestamps_csv_path", "") or "")
    v_path = str(md.get("original_video_path", "") or "")
    ts = str(md.get("timestamp", "") or "")
    date_str = ts.split("T")[0] if "T" in ts else ts
    
    combined_str = f"{p_path} {t_path} {v_path} {ts}".lower()
    
    if "3_week_pi" in combined_str or date_str in ["2025-12-03", "2025-12-04"]:
        return {
            "condition": "3w_PI",
            "condition_label": "3 Weeks PI (Baseline)",
            "drug": "None",
            "dose_mg_kg": 0.0,
            "weeks_post_injection": 3,
            "is_drug_session": False,
        }
    elif "8_weeks_pi" in combined_str or date_str in ["2026-01-08", "2026-01-09"]:
        return {
            "condition": "8w_PI",
            "condition_label": "8 Weeks PI",
            "drug": "None",
            "dose_mg_kg": 0.0,
            "weeks_post_injection": 8,
            "is_drug_session": False,
        }
    elif "24_weeks_pi" in combined_str or date_str in ["2026-04-27", "2026-04-28"]:
        return {
            "condition": "24w_PI",
            "condition_label": "24 Weeks PI (Late Stage)",
            "drug": "None",
            "dose_mg_kg": 0.0,
            "weeks_post_injection": 24,
            "is_drug_session": False,
        }
    elif "l_dopa" in combined_str or date_str in ["2026-05-16", "2026-05-17"]:
        return {
            "condition": "L_dopa",
            "condition_label": "L-Dopa + Benserazide",
            "drug": "L-Dopa + Benserazide",
            "dose_mg_kg": None,
            "weeks_post_injection": 26,
            "is_drug_session": True,
        }
    elif date_str.startswith("2026-03") or any(f"t{i}_" in combined_str for i in range(1, 10)):
        return {
            "condition": "16w_PI",
            "condition_label": "16 Weeks PI (Established)",
            "drug": "None",
            "dose_mg_kg": 0.0,
            "weeks_post_injection": 16,
            "is_drug_session": False,
        }
    else:
        return {
            "condition": "Unknown",
            "condition_label": "Unknown",
            "drug": "None",
            "dose_mg_kg": 0.0,
            "weeks_post_injection": None,
            "is_drug_session": False,
        }

def enrich(ctx, sessions, overrides_dir=None, skip_exclusions=False):
    if not overrides_dir:
        overrides_dir = Path(__file__).resolve().parent / "overrides"
    
    print(f"[Enrich] karen_project: Enriching {len(sessions)} sessions (skip_exclusions={skip_exclusions}).")
    animal_map = load_animal_map(overrides_dir)
    session_map = load_sessions_map(overrides_dir)
    exclusion_map = load_exclusions(overrides_dir) if not skip_exclusions else {}
    auto_mask_data = load_auto_mask_data(overrides_dir) if not skip_exclusions else None

    # Build normalized animal map to handle numeric & pilot alias variations
    normalized_animal_map = {}
    for k, v in animal_map.items():
        normalized_animal_map[k] = v
        clean_k = re.sub(r'[^0-9]', '', k)
        if clean_k and clean_k not in normalized_animal_map:
            normalized_animal_map[clean_k] = v

    # Step 1: Resolve animal_id using sessions.csv, falling back to parsed trial name / original_video_path
    for s in sessions:
        md = s.metadata
        sid = s.session_id
        if sid in session_map:
            over = session_map[sid]
            md["animal_id"] = over["animal_id"]
            if over["exclude"]:
                md["exclude"] = True
                print(f"[Enrich] Excluding session: {sid}")
        else:
            # Fallback
            trial_id = md.get("trial_id")
            vpath = md.get("original_video_path")
            if not trial_id and vpath:
                v_name = os.path.basename(vpath)
                parts = v_name.split("_")
                if len(parts) >= 2:
                    trial_id = f"{parts[0]}_{parts[1]}"
                    md["trial_id"] = trial_id

            ci = md.get("chamber_index")
            if trial_id and ci is not None:
                parts = trial_id.split("_")
                if len(parts) >= 2:
                    ids = parts[1].split(".")
                    if ci < len(ids):
                        raw_aid = ids[ci].strip()
                        clean_aid = re.sub(r'\(.*?\)', '', raw_aid).strip()
                        if clean_aid.lower() != 'e':
                            md["animal_id"] = clean_aid
                        else:
                            md["animal_id"] = None
                    else:
                        md["animal_id"] = None
                else:
                    md["animal_id"] = None
            else:
                md["animal_id"] = None

        # Parse condition, drug, weeks_post_injection
        cond_info = _parse_condition_and_drug(md)
        for ck, cv in cond_info.items():
            md[ck] = cv

    # Step 2: Pass 1 / Pass 2
    count = 0
    for s in sessions:
        md = s.metadata
        aid = md.get("animal_id")
        
        # Ensure condition info is set
        cond_info = _parse_condition_and_drug(md)
        for ck, cv in cond_info.items():
            md[ck] = cv
        
        # --- PASS 1: Pre-Realize (No signal_meta yet) ---
        if "signal_meta" not in md:
            if aid and (aid in animal_map or aid in normalized_animal_map):
                info = animal_map.get(aid) or normalized_animal_map.get(aid)
                if not _is_unknown(info['geno']):
                    md["genotype"] = info['geno']
                # Do not set fiber_indices_effective here!
                # Since fiber_indices_effective is not set, the loader defaults to standard chamber indexes,
                # which perfectly maps the correct columns for that chamber index in any session.
            count += 1
            continue

        # --- PASS 2: Post-Realize (Loader has created signal_meta) ---
        if aid and (aid in animal_map or aid in normalized_animal_map):
            info = animal_map.get(aid) or normalized_animal_map.get(aid)
            
            # Clear any stale logical keys in metadata
            for i in range(4):
                md.pop(f"signal_{i}", None)

            k = md.get("chamber_index")
            sm = md.get("signal_meta", {})
            for sig_key, entry in sm.items():
                raw_fn = entry.get("fiber_num")
                if raw_fn is None: continue
                fiber_num = int(raw_fn)

                # Dynamically match left/right fibers based on chamber index k:
                # Left Green: 4*k, Right Green: 4*k + 2
                if k is not None:
                    if fiber_num == 4 * k:
                        region = info['L']
                        hemi = "Left"
                    elif fiber_num == 4 * k + 2:
                        region = info['R']
                        hemi = "Right"
                    else:
                        region = "Unknown"
                        hemi = "Unknown"
                else:
                    # Fallback if chamber_index is missing
                    if fiber_num % 4 == 0:
                        region = info['L']
                        hemi = "Left"
                    elif fiber_num % 4 == 2:
                        region = info['R']
                        hemi = "Right"
                    else:
                        region = "Unknown"
                        hemi = "Unknown"

                md[sig_key] = region
                entry["region"] = region
                entry["brain_reg"] = region
                entry["hemisphere"] = hemi

            # 3. Apply Exclusions — match by session_id first, then animal_id
            sid = s.session_id
            excl_key = sid if sid in exclusion_map else (aid if aid in exclusion_map else None)
            
            if "combined_filtered" in s.photometry:
                # A. Apply Auto Mask
                apply_auto_masks(ctx, s, auto_mask_data)

                # B. Apply Manual Exclusions
                if excl_key:
                    sess_excl = exclusion_map[excl_key]
                    if "exclusions" not in md: md["exclusions"] = {}
                    
                    for sig_key, meta in sm.items():
                        reg = meta.get("region", md.get(sig_key))
                        hemi = meta.get("hemisphere", "")
                        full_label = f"{reg}-{hemi}" if hemi else reg
                        
                        if full_label and full_label in sess_excl:
                            md["exclusions"][sig_key] = sess_excl[full_label]

                    apply_manual_exclusions_to_session(s, sess_excl)
        
        count += 1

    print(f"[Enrich] Enriched {count} sessions.")
    return sessions
