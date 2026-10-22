# syllable_generation/dialects/jonathan/fiberphotometry.py
# steps/fiberphotometry.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import JCFG
from ..utils.photometry import *
from ..utils.time import parse_session_ts


def attach_photometry_refs(ctx, sessions, tolerance_seconds=None):
    cfg = getattr(ctx, "cfg", JCFG)
    tol = cfg.tolerance_seconds if tolerance_seconds is None else float(tolerance_seconds)
    root = ctx.csv_root
    if not root:
        return sessions

    entries = list_combined_csv_entries(root, cfg)
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
    cfg = getattr(ctx, "cfg", JCFG)
    import gc
    from collections import defaultdict

    # 1. Group sessions by their shared photometry CSV
    by_path = defaultdict(list)
    for s in sessions:
        csv_path = s.metadata.get("photometry_csv_path")
        if csv_path:
            by_path[csv_path].append(s)

    # 2. Caches for this step
    # Cache for AllTimestamps DataFrames: (path, camera) -> df
    allts_cache = {}

    # 3. Process each unique CSV once
    for csv_path_str, group in by_path.items():
        p = Path(csv_path_str)
        if not p.exists():
            continue

        # Get header once for planning
        header_df = pd.read_csv(p, nrows=0)
        all_cols = list(header_df.columns)

        # Build union of columns needed for ALL sessions in this group
        union_cols = set()
        group_plans = {}
        for s in group:
            plan = plan_from_header(all_cols, s, cfg)
            group_plans[s.session_id] = plan

            sig_cols = [col for _, col in plan["signals"]]
            union_cols.update(plan["non_fiber_cols"])
            union_cols.update(sig_cols)

            s.metadata["effective_indices"] = plan.get("effective_indices", [])

        # Read the shared CSV ONCE
        usecols = dedup_preserve_order([c for c in all_cols if c in union_cols])
        # OPTIMIZATION: Use pyarrow engine for faster parsing
        try:
            shared_df = pd.read_csv(p, usecols=usecols, engine='pyarrow')
        except Exception:
            shared_df = pd.read_csv(p, usecols=usecols)

        # 4. Distribute and finalize each session
        for s in group:
            plan = group_plans[s.session_id]
            non_fiber_cols = plan["non_fiber_cols"]
            signals = plan["signals"]
            led_col = plan["led_col"]

            # Select and copy sub-columns for this session
            sig_cols = [col for _, col in signals]
            sess_cols = dedup_preserve_order(non_fiber_cols + sig_cols)
            df = shared_df[sess_cols].copy()

            # Metadata and renaming
            fiber_to_signal = update_signal_meta_from_signals(s, signals)
            if fiber_to_signal:
                df = df.rename(columns=fiber_to_signal)

            # --- attach frame_idx (with caching) ---
            entries = list_alltimestamps_entries(p.parent, cfg)
            ts_str = s.metadata.get("timestamp")
            if not isinstance(ts_str, str):
                continue
            dt_sess = parse_session_ts(ts_str)

            match = None
            if entries:
                match = best_csv_match(dt_sess, entries, cfg.tolerance_seconds)

            if not match and ctx.csv_root:
                # Fallback: search recursively in the csv_root
                import re
                csv_re = re.compile(cfg.alltimestamps_csv_regex, re.I)
                fallback_entries = []
                for p_any in Path(ctx.csv_root).rglob(cfg.alltimestamps_csv_glob):
                    m = csv_re.match(p_any.name)
                    if not m:
                        continue
                    try:
                        dt = parse_session_ts(m.group("ts"))
                        cam = m.group("camera")
                        fallback_entries.append((p_any, dt, cam))
                    except Exception:
                        pass
                fallback_entries.sort(key=lambda t: t[1])
                match = best_csv_match(dt_sess, fallback_entries, cfg.tolerance_seconds)

            if not match:
                raise FileNotFoundError(f"No matching AllTimestampsCombined for {dt_sess}")

            allts_path, _dt, camera_name = match
            cache_key = (str(allts_path), camera_name)

            if cache_key not in allts_cache:
                # OPTIMIZATION: Cache AllTimestamps
                allts_cache[cache_key] = pd.read_csv(allts_path, usecols=[cfg.allts_system_ts_col], engine='pyarrow')

            df = attach_frame_idx_column(df, allts_cache[cache_key], cfg)

            s.metadata["alltimestamps_csv_path"] = str(allts_path)
            s.metadata["alltimestamps_camera"] = camera_name
            s.photometry["combined_filtered"] = df
            s.metadata["photometry_non_fiber_cols"] = non_fiber_cols
            s.metadata["signal_keys"] = [k for k, _ in signals]
            s.metadata["led_col_name"] = led_col

        # Cleanup group data to keep memory usage stable
        del shared_df
        gc.collect()

    return sessions


# --- per-session splitter (side-effect only) ---
def split_photometry_session(ctx, s, source_key: str = "combined_filtered", keep_led_col: bool = False) -> None:
    cfg = getattr(ctx, "cfg", JCFG)
    df = s.photometry.get(source_key)
    if df is None or getattr(df, "empty", True):
        return

    led_col = cfg.led_col_name
    if led_col not in df.columns:
        return

    non_fiber   = set(s.metadata.get("photometry_non_fiber_cols", []))

    # ensure frame_idx is preserved as a non-fiber column
    fic = getattr(cfg, "frame_index_col_name", "frame_idx")
    if fic in df.columns:
        non_fiber.add(fic)

    signal_keys = get_signal_cols(df)
    sig_meta    = s.metadata.get("signal_meta", {})

    for code, (letter_key, freq) in cfg.led_state_map.items():
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

    chans = [freq for _, (_, freq) in cfg.led_state_map.items() if freq in s.photometry]
    if chans:
        s.metadata["photometry_channels"] = sorted(set(chans))

def split_photometry(ctx, sessions, source_key: str = "combined_filtered"):
    for s in sessions:
        split_photometry_session(ctx, s, source_key=source_key)
    return sessions


# --- per-session ---
def compute_dff_zscore_session(
    ctx, s, freqs=("470", "560"), iso="415", *, cfg=None
) -> None:
    iso_df = s.photometry.get(str(iso))
    if iso_df is None or getattr(iso_df, "empty", True):
        return

    if cfg is None:
        from .. import config as live_config
        cfg = getattr(ctx, "cfg", live_config.JCFG)

    def _assign_col(df, col, data_series):
        if col not in df.columns: df[col] = pd.NA
        # Slice both to length n just to be safe
        n = len(data_series)
        df.loc[: n - 1, col] = data_series.to_numpy()

    iso_signals = set(get_signal_cols(iso_df))
    ts_col = cfg.phot_system_ts_col

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

            # --- Standard Datta (RANSAC) ---
            clean_ransac = calculate_ransac_subtraction(sig_dff_datta, ref_dff_datta)
            z_ransac = calculate_rolling_zscore(clean_ransac, fps, win_sec=20.0)

            if cfg.include_ransac:
                _assign_col(tgt, f"{sig}_zscore_ransac", z_ransac)

            # --- Optional Wallace Method (Regression) ---
            # Compute if requested OR if needed as basis for prominence
            active_basis = getattr(cfg, "prominence_basis", "wallace")

            # 1. Wallace (Regression)
            needs_wallace = cfg.include_wallace or (cfg.include_prominence and active_basis == "wallace")
            z_wallace = None
            if needs_wallace:
                clean_wallace = calculate_wallace_dff(sig_dff_datta, ref_dff_datta)
                z_wallace = calculate_rolling_zscore(clean_wallace, fps, win_sec=20.0)

                if cfg.include_wallace:
                    _assign_col(tgt, f"{sig}_zscore_wallace", z_wallace)

            # 2. Manual Method (Simple Subtraction)
            needs_manual = cfg.include_manual or (cfg.include_prominence and active_basis == "manual")
            z_manual = None
            if needs_manual:
                # Use rolling dF/F inputs to account for signal drift, but apply simple subtraction
                # (no regression scaling) as per Wallace et al. (2025).
                clean_manual = calculate_manual_dff(sig_dff_datta, ref_dff_datta)
                z_manual = calculate_rolling_zscore(clean_manual, fps, win_sec=20.0)

                if cfg.include_manual:
                    _assign_col(tgt, f"{sig}_zscore_manual", z_manual)

            # --- Legacy/Global Outputs ---
            if cfg.include_dff:
                _assign_col(tgt, dff_col, dff)

            if cfg.include_zscore:
                _assign_col(tgt, z_col, z)

            # --- Prominence (Optional) ---
            if cfg.include_prominence:
                # Select basis based on config
                base_for_prom = None
                if active_basis == "wallace":
                    base_for_prom = z_wallace
                elif active_basis == "manual":
                    base_for_prom = z_manual
                else: # Default to RANSAC
                    base_for_prom = z_ransac

                if base_for_prom is not None:
                    prom = calculate_signal_prominence(base_for_prom, fps, win_sec=5.0)
                    _assign_col(tgt, f"{sig}_prominence", prom)

        s.photometry[str(freq)] = tgt

    # Metadata updates
    md = s.metadata
    used = set(map(str, freqs)); used.discard(str(iso))
    md["dff_iso_channel"] = str(iso)
    md["dff_computed_channels"] = sorted(used)
    md["datta_replication_enabled"] = True
    md["reference_interpolation"] = can_interp # Mark if we used interpolation

def compute_dff_zscore(
    ctx, sessions, freqs=("470", "560"), iso="415"
):
    from .. import config as live_config
    cfg = getattr(ctx, "cfg", live_config.JCFG)
    # OPTIMIZATION: Parallelize signal processing across sessions if joblib is available
    def _proc(s, selected_cfg):
        compute_dff_zscore_session(ctx, s, freqs=freqs, iso=iso, cfg=selected_cfg)
        return s

    try:
        from joblib import Parallel, delayed
        sessions = Parallel(n_jobs=-1)(delayed(_proc)(s, cfg) for s in sessions)
    except ImportError:
        sessions = [_proc(s, cfg) for s in sessions]
    return sessions
