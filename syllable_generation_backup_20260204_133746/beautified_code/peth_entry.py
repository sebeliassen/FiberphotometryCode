# peth_entry.py
import os
import re
import math
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from session import Session
from photometry_processor import (
    load_raw_photometry_and_merge_timestamps,
)

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# --------------------------- utilities ---------------------------

def normalize_hyphens(s: str) -> str:
    """Replace Unicode hyphens/dashes with ASCII '-' for robust plotting."""
    return re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2212]', '-', s)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_camera_timestamps_ns(center_timestamp_csv: str,
                              prefer: str = "Cam_Basler_Timestamp") -> Optional[np.ndarray]:
    """Returns per-frame camera timestamps in **ns** as numpy array, preferring Basler."""
    df = pd.read_csv(center_timestamp_csv)
    col = prefer if prefer in df.columns else ("Cam_Bonsai_Timestamp" if "Cam_Bonsai_Timestamp" in df.columns else None)
    if col is None:
        logger.error(f"No camera timestamp column found in {center_timestamp_csv}")
        return None
    # Basler appears to be integer ns; Bonsai here looks like days-seconds float; we keep numeric and coerce to ns
    ts = pd.to_numeric(df[col], errors="coerce").to_numpy()
    ts = ts[np.isfinite(ts)]
    if ts.size == 0:
        logger.error(f"Camera timestamps empty in {center_timestamp_csv}")
        return None

    # Heuristic: if values look like seconds (<= 1e8) or day-fraction, convert to ns;
    # if already in ns scale (~1e12+), leave as-is.
    tmax = float(np.nanmax(ts))
    if tmax < 1e9:                # seconds (or day seconds); convert sec->ns
        ts = (ts * 1e9).astype(np.float64)
    elif tmax < 1e12:             # milliseconds → ns
        ts = (ts * 1e6).astype(np.float64)
    else:
        ts = ts.astype(np.float64)  # already ns-ish
    return ts

def _safe_get_event_frames(obj) -> List[int]:
    """
    Try hard to extract a list of original frame indices from a variety of possible pickle structures.
    Accepts: list[int], list[tuple], tuple, dicts with keys like 'frames','frame','start','center_frame', etc.
    """
    if obj is None:
        return []

    # Direct list/array of ints
    if isinstance(obj, (list, np.ndarray)):
        vals = list(obj)
        if len(vals) and all(isinstance(x, (int, np.integer)) for x in vals):
            return [int(x) for x in vals]
        # list of tuples/ranges -> take first element as onset
        out = []
        for x in vals:
            if isinstance(x, (tuple, list)) and len(x) >= 1 and isinstance(x[0], (int, np.integer)):
                out.append(int(x[0]))
        if out:
            return out

    # Single tuple/range-like
    if isinstance(obj, tuple) and len(obj) >= 1 and isinstance(obj[0], (int, np.integer)):
        return [int(obj[0])]

    # Dict variants
    if isinstance(obj, dict):
        for key in ["frames", "frame_list", "onsets", "starts", "start_frames", "original_frames"]:
            if key in obj and isinstance(obj[key], (list, np.ndarray)):
                li = [int(x) for x in obj[key] if isinstance(x, (int, np.integer))]
                if li:
                    return li
        for key in ["center_frame", "start", "frame", "original_frame"]:
            if key in obj and isinstance(obj[key], (int, np.integer)):
                return [int(obj[key])]
        # sometimes each value is a dict with 'start' etc.
        if all(isinstance(v, dict) for v in obj.values()):
            out = []
            for v in obj.values():
                for key in ["center_frame", "start", "frame", "original_frame"]:
                    if key in v and isinstance(v[key], (int, np.integer)):
                        out.append(int(v[key]))
            if out:
                return out

    return []

def load_syllable_events_for_session(session: Session) -> Dict[str, List[int]]:
    """
    Returns: dict {syllable_name: [original_frame_indices, ...]} for this session.
    Strategy:
      1) Try global FRAME_RANGES_PKL (config.FRAME_RANGES_PKL).
      2) Fall back to per-session 'used_frames.pkl' next to the 3D pose video (if discoverable).
    """
    events: Dict[str, List[int]] = {}

    # 1) Global frame ranges
    if hasattr(config, "FRAME_RANGES_PKL") and config.FRAME_RANGES_PKL and Path(config.FRAME_RANGES_PKL).exists():
        try:
            with open(config.FRAME_RANGES_PKL, "rb") as f:
                fr = pickle.load(f)
            # We need a key for this session. Try several identifiers
            candidates = []
            if hasattr(session, "behavior_basename"):
                candidates.append(session.behavior_basename)
            if hasattr(session, "session_key"):
                candidates.append(session.session_key)
            if hasattr(session, "name"):
                candidates.append(session.name)
            # often the composite used a behavior video stem like "Arena1234_CenterTopCamVid_YYYY-..._<suffix>"
            # collect all dict entries whose key contains the date-stamp part of this session, if available
            matched = None
            for k in candidates:
                if k and k in fr:
                    matched = fr[k]
                    break
            if matched is None:
                # try fuzzy match by arena/date if available on session
                for k in fr.keys():
                    if hasattr(session, "arena") and str(session.arena) in k:
                        if hasattr(session, "date_str") and str(getattr(session, "date_str")) in k:
                            matched = fr[k]; break
                if matched is None and candidates:
                    # last resort: substring match any candidate in keys
                    for k in fr.keys():
                        if any(c and c in k for c in candidates):
                            matched = fr[k]; break

            if matched is not None and isinstance(matched, dict):
                for syllable, payload in matched.items():
                    frames = _safe_get_event_frames(payload)
                    if frames:
                        events[str(syllable)] = frames
        except Exception as e:
            logger.warning(f"FRAME_RANGES_PKL load failed: {e}")

    # 2) Per-session fallback: look for 'used_frames.pkl' near pose or syllable files
    if not events:
        try:
            # Guess a directory near pose or syllable video
            guess_dirs = []
            if hasattr(config, "POSE_3D_VIDEO_DIR"):
                guess_dirs.append(Path(config.POSE_3D_VIDEO_DIR))
            if hasattr(config, "SYLLABLE_VIDEO_DIR"):
                guess_dirs.append(Path(config.SYLLABLE_VIDEO_DIR))
            for gd in guess_dirs:
                if gd and gd.exists():
                    cand = gd / "used_frames.pkl"
                    if cand.exists():
                        with open(cand, "rb") as f:
                            payload = pickle.load(f)
                        # Expect something like {session_id: {syllable: [...]}}
                        key = None
                        for k in [getattr(session, "behavior_basename", None),
                                  getattr(session, "session_key", None),
                                  getattr(session, "name", None)]:
                            if k is None: continue
                            if isinstance(payload, dict) and k in payload:
                                key = k; break
                        if key is not None:
                            for syllable, val in payload[key].items():
                                frames = _safe_get_event_frames(val)
                                if frames:
                                    events[str(syllable)] = frames
                        break
        except Exception as e:
            logger.warning(f"used_frames.pkl probe failed: {e}")

    return events


# --------------------------- PETH core ---------------------------

def make_time_grid(window_s: Tuple[float, float], step_s: float) -> np.ndarray:
    t = np.arange(window_s[0], window_s[1] + 1e-12, step_s, dtype=np.float64)
    # ensure 0 is exactly represented if in range
    if window_s[0] < 0 < window_s[1]:
        t[np.argmin(np.abs(t))] = 0.0
    return t

def interp_trace_at_event(behavior_ts_ns: np.ndarray,
                          z_values: np.ndarray,
                          event_ts_ns: float,
                          t_rel_s: np.ndarray) -> np.ndarray:
    """
    Interpolate Z trace at times (event_ts + t_rel_s). Uses linear interp with edge NaNs.
    """
    x = behavior_ts_ns.astype(np.float64)
    y = z_values.astype(np.float64)

    # Make sure y has no all-NaN; if so return NaNs
    if np.all(~np.isfinite(y)):
        return np.full_like(t_rel_s, np.nan, dtype=np.float64)

    # Simple forward/back fill of short NaN runs to avoid gaps in interp
    y_work = pd.Series(y).interpolate(limit_direction="both").to_numpy()

    query_ns = event_ts_ns + t_rel_s * 1e9
    # Outside range -> NaN
    inside = (query_ns >= x[0]) & (query_ns <= x[-1])
    out = np.full_like(query_ns, np.nan, dtype=np.float64)
    if np.any(inside):
        out[inside] = np.interp(query_ns[inside], x, y_work)
    return out

def compute_peth_for_session(session: Session,
                             window_s: Tuple[float, float],
                             step_s: float,
                             use_filtered: bool = False) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    Returns:
        nested dict: {syllable: {region_name: [traces_for_each_event (len=tgrid)]}}
        (per single session)
    """
    logp = f"[{getattr(session, 'behavior_basename', getattr(session, 'name', 'session'))}] "
    # Load aligned photometry (unfiltered Z by default)
    df = load_raw_photometry_and_merge_timestamps(
        session.photometry_csv,
        session.center_timestamp_csv,
        session.arena,
        session.left_region,
        session.right_region,
        log_prefix=logp
    )
    if df is None or df.empty:
        logger.warning(f"{logp}photometry df empty; skipping")
        return {}

    cam_ts = load_camera_timestamps_ns(session.center_timestamp_csv, prefer="Cam_Basler_Timestamp")
    if cam_ts is None or cam_ts.size == 0:
        logger.warning(f"{logp}no camera timestamps; skipping")
        return {}

    # BehaviorTimestamp is in ns already (from merge), ensure numeric asc order
    bt = pd.to_numeric(df["BehaviorTimestamp"], errors="coerce").to_numpy(np.float64)
    ok = np.isfinite(bt)
    if not np.any(ok):
        logger.warning(f"{logp}BehaviorTimestamp invalid; skipping")
        return {}

    # Regions
    regions = [session.left_region, session.right_region]
    reg_cols = []
    for r in regions:
        r_norm = normalize_hyphens(r)
        # prefer filtered column only if requested & present
        col_f = f"{r_norm}_filtered"
        col_z = r_norm
        # Columns in df may have original unicode hyphen; try both
        candidates = [col_f, col_z,
                      f"{r}_filtered", r]
        found = None
        for c in candidates:
            if c in df.columns:
                found = c; break
        if found is None:
            logger.warning(f"{logp}region column missing: {r}")
            continue
        reg_cols.append((r_norm, found))

    if not reg_cols:
        logger.warning(f"{logp}no usable region columns")
        return {}

    # Events (syllables -> list of frames)
    syll2frames = load_syllable_events_for_session(session)
    if not syll2frames:
        logger.warning(f"{logp}no syllable events found")
        return {}

    # Time grid
    t = make_time_grid(window_s, step_s)

    out: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for syll, frames in syll2frames.items():
        traces_per_region: Dict[str, List[np.ndarray]] = {rname: [] for rname, _ in reg_cols}
        for f in frames:
            if f < 0 or f >= len(cam_ts):
                continue
            ev_ns = float(cam_ts[f])
            for rname, col in reg_cols:
                y = df[col].to_numpy(np.float64)
                tr = interp_trace_at_event(bt, y, ev_ns, t)
                traces_per_region[rname].append(tr)
        # Keep only if we collected something
        if any(len(v) for v in traces_per_region.values()):
            out[syll] = traces_per_region

    return out

def aggregate_by_genotype(
    sessions: List[Session],
    window_s: Tuple[float, float],
    step_s: float,
    use_filtered: bool = False
) -> Tuple[np.ndarray, Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Returns:
      tgrid,
      means: {syllable: {region: {genotype: mean_trace}}},
      sems:  {syllable: {region: {genotype: sem_trace}}}
    """
    t = make_time_grid(window_s, step_s)

    # Store all event-level traces per (syll, region, genotype)
    bucket: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for sess in sessions:
        sess_peth = compute_peth_for_session(sess, window_s, step_s, use_filtered=use_filtered)
        geno = getattr(sess, "genotype", "UNK")
        for syll, reg_map in sess_peth.items():
            for region, traces in reg_map.items():
                for tr in traces:
                    if tr is None or not np.any(np.isfinite(tr)):
                        continue
                    bucket[syll][region][geno].append(tr)

    means = defaultdict(lambda: defaultdict(dict))
    sems  = defaultdict(lambda: defaultdict(dict))

    for syll, reg_map in bucket.items():
        for region, geno_map in reg_map.items():
            for geno, traces in geno_map.items():
                if not traces:
                    continue
                arr = np.vstack(traces)  # (n_events, T)
                mean = np.nanmean(arr, axis=0)
                std  = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
                n = np.sum(np.any(np.isfinite(arr), axis=1))
                sem = std / math.sqrt(max(n, 1))
                means[syll][region][geno] = mean
                sems[syll][region][geno]  = sem

    return t, means, sems


# --------------------------- plotting/export ---------------------------

def _genotype_palette(genotypes: List[str]) -> Dict[str, str]:
    # cycle through tab10
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    pal = {}
    for i, g in enumerate(sorted(genotypes)):
        pal[g] = colors[i % len(colors)]
    return pal

def save_peth_pngs(
    t: np.ndarray,
    means: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    sems:  Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: Path
) -> List[Path]:
    ensure_dir(output_dir)
    saved: List[Path] = []

    # Gather all region names to set a consistent subplot order
    all_regions = set()
    for syll, reg_map in means.items():
        for region in reg_map.keys():
            all_regions.add(region)
    region_order = sorted(all_regions)

    # All genotypes present
    all_genos = set()
    for syll, reg_map in means.items():
        for region, geno_map in reg_map.items():
            for g in geno_map.keys():
                all_genos.add(g)
    pal = _genotype_palette(list(all_genos))

    for syll, reg_map in means.items():
        if not reg_map:
            continue
        nrows = len(region_order)
        fig_h = max(2.5 * nrows, 3.0)
        fig, axes = plt.subplots(nrows, 1, figsize=(8, fig_h), sharex=True)
        if nrows == 1:
            axes = [axes]

        for ax, region in zip(axes, region_order):
            geno_map = reg_map.get(region, {})
            # Plot each genotype mean ± SEM
            for g, mean in geno_map.items():
                sem = sems[syll][region].get(g)
                color = pal.get(g, None)
                ax.plot(t, mean, label=g, linewidth=1.8, color=color)
                if sem is not None and np.any(np.isfinite(sem)):
                    ax.fill_between(t, mean - sem, mean + sem, alpha=0.25, linewidth=0, color=color)
            ax.axvline(0, linestyle='--', linewidth=1, color='k', alpha=0.7)
            ax.set_ylabel(f"{normalize_hyphens(region)}\nZ-score", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.5)
        axes[-1].set_xlabel("Time (s)")

        fig.suptitle(f"PETH — Syllable: {syll}", fontsize=13, color="black")  # black title text
        axes[0].legend(loc="upper right", fontsize=9, ncol=min(3, len(all_genos)))
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        out_path = output_dir / f"peth_{syll}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)
        logger.info(f"Saved {out_path}")

    return saved


# --------------------------- public entry point ---------------------------

def run_peth_export(
    sessions: List[Session],
    window_s: Tuple[float, float] = (-2.0, 5.0),
    step_s: float = 0.02,
    use_filtered: bool = False,
    outdir: Optional[Path] = None
) -> List[Path]:
    """
    Generate PETH PNGs: one PNG per syllable; each PNG has |regions| subplots;
    each subplot shows |genotypes| traces (mean ± SEM).

    Args:
      sessions: list of Session objects (must include photometry_csv, center_timestamp_csv,
                left_region, right_region, genotype, mouse_id, arena).
      window_s: (pre, post) seconds relative to syllable onset.
      step_s:   sampling step for the PETH curve (seconds).
      use_filtered: if True and per-region *_filtered columns are present in the merged DF,
                    use them; otherwise use raw Z-score columns.
      outdir: where to dump PNGs. Defaults to config.OUTPUT_DIR / "peth_plots".
    """
    if outdir is None:
        outdir = Path(getattr(config, "OUTPUT_DIR", Path.cwd())) / "peth_plots"

    logger.info(f"PETH export: window={window_s}s, step={step_s}s, use_filtered={use_filtered}")
    t, means, sems = aggregate_by_genotype(sessions, window_s, step_s, use_filtered=use_filtered)
    return save_peth_pngs(t, means, sems, outdir)


# --------------------------- optional runnable main ---------------------------

def main_peth():
    """
    Alternative main you can run directly (no argparse).
    Option A (recommended): pass your existing sessions_to_process here.
    Option B: if you already have a discovery helper, import and use it.
    """
    try:
        # --- Option A: if you have sessions_to_process constructed elsewhere, import it:
        # from main import sessions_to_process
        # saved = run_peth_export(sessions_to_process, window_s=(-2, 5), step_s=0.02, use_filtered=False)
        # print(f"Saved {len(saved)} PETH PNGs.")
        # return

        # --- Option B: if you have a session discovery utility, call it here:
        from your_session_discovery_module import discover_sessions_to_process  # replace with your actual function
        sessions = discover_sessions_to_process()  # must return List[Session]
        saved = run_peth_export(sessions, window_s=(-2, 5), step_s=0.02, use_filtered=False)
        print(f"Saved {len(saved)} PETH PNGs.")
    except Exception as e:
        logger.error(
            "Could not discover sessions automatically. "
            "Import/build your 'sessions_to_process' list and call run_peth_export(sessions_to_process). "
            f"Details: {e}"
        )

if __name__ == "__main__":
    main_peth()
