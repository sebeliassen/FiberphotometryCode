# syllable_generation/dialects/jonathan/fiberphotometry.py
# steps/fiberphotometry.py
from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime

from ..context import BuildContext
from ..config import JCFG
from ..utils.time import parse_session_ts
from ..utils.photometry import *

def attach_photometry_refs(ctx, sessions, tolerance_seconds=None):
    tol = JCFG.tolerance_seconds if tolerance_seconds is None else float(tolerance_seconds)
    root = ctx.csv_root
    if not root:
        return sessions

    entries = list_combined_csv_entries(root, JCFG)
    if not entries:
        return sessions

    sess_info = []
    for s in sessions:
        ts = s.metadata.get("timestamp")
        ci = s.metadata.get("chamber_index")
        if isinstance(ts, str) and ci is not None:
            try:
                sess_info.append((s, parse_session_ts(ts), int(ci)))
            except Exception:
                pass
    if not sess_info:
        return sessions

    for s, dt_sess, ci in sess_info:
        match = best_csv_match(dt_sess, entries, tol)
        if not match:
            continue
        p, dt_csv, trial_id = match

        aid = animal_id_for(trial_id, ci)  # keep your helper name if you still expose it here
        toks = parse_trial_tokens(trial_id)

        s.metadata.update({
            "trial_id": trial_id,
            "animal_id": aid,
            "n_chambers": len(toks),
            "photometry_ts_csv": dt_csv.isoformat(),
            "photometry_csv_path": str(p),
            "photometry_source": "csv_combined",
        })
    return sessions

def realize_photometry_columns(ctx, sessions):
    for s in sessions:
        csv_path = s.metadata.get("photometry_csv_path")
        if not csv_path:
            continue

        p = Path(csv_path)
        if not p.exists():
            continue

        header_df = pd.read_csv(p, nrows=0)
        cols = list(header_df.columns)

        plan = plan_from_header(cols, s, JCFG)
        non_fiber_cols = plan["non_fiber_cols"]
        signals        = plan["signals"]
        led_col        = plan["led_col"]
        s.metadata["effective_indices"] = plan.get("effective_indices", [])

        sig_cols = [col for _, col in signals]
        usecols = dedup_preserve_order(non_fiber_cols + sig_cols)

        fiber_to_signal = update_signal_meta_from_signals(s, signals)

        df = pd.read_csv(p, usecols=usecols)
        if fiber_to_signal:
            df = df.rename(columns=fiber_to_signal)

        # --- attach frame_idx from sibling AllTimestampsCombined CSV (strict) ---
        entries = list_alltimestamps_entries(p.parent, JCFG)
        if not entries:
            raise FileNotFoundError(
                f"No '*AllTimestampsCombined*.csv' files found next to photometry CSV: {p.parent}"
            )

        ts_str = s.metadata.get("timestamp")
        if not isinstance(ts_str, str):
            raise ValueError(f"Session {s.session_id}: missing 'timestamp' for AllTimestampsCombined selection.")

        dt_sess = parse_session_ts(ts_str)
        match = best_csv_match(dt_sess, entries, JCFG.tolerance_seconds)
        if not match:
            raise FileNotFoundError(
                f"No AllTimestampsCombined file within ±{JCFG.tolerance_seconds}s of {dt_sess} in {p.parent}"
            )

        allts_path, _dt, camera_name = match
        allts_df = pd.read_csv(allts_path, usecols=[JCFG.allts_system_ts_col])
        df = attach_frame_idx_column(df, allts_df, JCFG)

        s.metadata["alltimestamps_csv_path"] = str(allts_path)
        s.metadata["alltimestamps_camera"] = camera_name

        s.photometry["combined_filtered"] = df
        s.metadata["photometry_non_fiber_cols"] = non_fiber_cols
        s.metadata["signal_keys"] = [k for k, _ in signals]
        s.metadata["led_col_name"] = led_col
    return sessions


# --- per-session splitter (side-effect only) ---
def split_photometry_session(ctx, s, source_key: str = "combined_filtered", keep_led_col: bool = False) -> None:
    df = s.photometry.get(source_key)
    if df is None or getattr(df, "empty", True):
        return

    led_col = JCFG.led_col_name
    if led_col not in df.columns:
        return

    non_fiber   = set(s.metadata.get("photometry_non_fiber_cols", []))

    # ensure frame_idx is preserved as a non-fiber column
    fic = getattr(JCFG, "frame_index_col_name", "frame_idx")
    if fic in df.columns:
        non_fiber.add(fic)

    signal_keys = get_signal_cols(df)
    sig_meta    = s.metadata.get("signal_meta", {})

    for code, (letter_key, freq) in JCFG.led_state_map.items():
        part = df[df[led_col] == code]
        if part.empty:
            continue

        # OLD:
        # chosen = select_signals_for_state(letter_key, signal_keys, sig_meta)
        # NEW: keep ALL signals in every channel (415/470/560)
        chosen = set(signal_keys)

        keep = [
            c for c in part.columns
            if (c in non_fiber) or (keep_led_col and c == led_col) or (c in chosen)
        ]
        if not keep:
            continue

        out = part.loc[:, keep].copy()
        s.photometry[freq] = out.reset_index(drop=True)

    chans = [freq for _, (_, freq) in JCFG.led_state_map.items() if freq in s.photometry]
    if chans:
        s.metadata["photometry_channels"] = sorted(set(chans))

def split_photometry(ctx, sessions, source_key: str = "combined_filtered"):
    for s in sessions:
        split_photometry_session(ctx, s, source_key=source_key)
    return sessions


# --- per-session ---
def compute_dff_zscore_session(ctx, s, freqs=("470", "560"), iso="415") -> None:
    iso_df = s.photometry.get(str(iso))
    if iso_df is None or getattr(iso_df, "empty", True):
        return

    iso_signals = set(get_signal_cols(iso_df))
    ts_col = "SystemTimestamp"  # Ensure this matches your JCFG config
    
    # Estimate FPS for rolling calculations
    fps = estimate_fps(iso_df, ts_col)

    for freq in freqs:
        if str(freq) == str(iso):
            continue
        tgt = s.photometry.get(str(freq))
        if tgt is None or getattr(tgt, "empty", True):
            continue

        tgt_signals = set(get_signal_cols(tgt))
        common = sorted(iso_signals & tgt_signals)
        if not common:
            continue

        n = min(len(iso_df), len(tgt))
        if n == 0:
            continue

        # Get Timestamps for Interpolation
        # We want to map ISO (Reference) -> TGT (Signal) time
        can_interp = (ts_col in iso_df.columns) and (ts_col in tgt.columns)
        
        if can_interp:
            # Convert to numpy float for fast interpolation
            t_iso = iso_df[ts_col].iloc[:n].to_numpy(dtype=float)
            t_tgt = tgt[ts_col].iloc[:n].to_numpy(dtype=float)

        for sig in common:
            # Get raw traces
            tgt_trace = tgt[sig].iloc[:n]
            
            if can_interp:
                # --- OPTIMIZATION: Linear Interpolation ---
                # Estimate what Ref was doing at the exact moment Signal was recorded
                iso_vals = iso_df[sig].iloc[:n].to_numpy(dtype=float)
                iso_interp = np.interp(t_tgt, t_iso, iso_vals)
                iso_trace = pd.Series(iso_interp, index=tgt_trace.index)
            else:
                # Fallback: Row-by-row subtraction (The "Old" way)
                iso_trace = iso_df[sig].iloc[:n]

            # 1. Standard Global dF/F (Legacy)
            dff, z = compute_dff_and_z(iso_trace, tgt_trace)

            dff_col = f"{sig}_dff"
            z_col = f"{sig}_zscore"
            
            # 2. Datta-style Logic
            # A. Rolling dF/F (using Interpolated Reference)
            sig_dff_datta = calculate_rolling_dff(tgt_trace, fps, win_sec=5.0)
            ref_dff_datta = calculate_rolling_dff(iso_trace, fps, win_sec=5.0)
            
            # B. RANSAC Subtraction (filtered internally by the function)
            clean_datta = calculate_ransac_subtraction(sig_dff_datta, ref_dff_datta)
            
            # C. Rolling Z-Score
            z_datta = calculate_rolling_zscore(clean_datta, fps, win_sec=20.0)
            
            datta_col = f"{sig}_zscore_datta"

            # Assign
            if dff_col not in tgt.columns: tgt[dff_col] = pd.NA
            if z_col not in tgt.columns:   tgt[z_col] = pd.NA
            if datta_col not in tgt.columns: tgt[datta_col] = pd.NA
            
            tgt.loc[: n - 1, dff_col] = dff.to_numpy()
            tgt.loc[: n - 1, z_col] = z.to_numpy()
            tgt.loc[: n - 1, datta_col] = z_datta.to_numpy()

        s.photometry[str(freq)] = tgt

    # Metadata updates
    md = s.metadata
    used = set(map(str, freqs)); used.discard(str(iso))
    md["dff_iso_channel"] = str(iso)
    md["dff_computed_channels"] = sorted(used)
    md["datta_replication_enabled"] = True
    md["reference_interpolation"] = can_interp # Mark if we used interpolation

def compute_dff_zscore(ctx, sessions, freqs=("470", "560"), iso="415"):
    for s in sessions:
        compute_dff_zscore_session(ctx, s, freqs=freqs, iso=iso)
    return sessions