
import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy.signal as signal
from tqdm import tqdm

# Add parent to path to find local lib
sys.path.append(str(Path(__file__).resolve().parent))
# Add Repo Parent to allow unpickling
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(repo_root.parent))

# Force Line Buffering
sys.stdout.reconfigure(line_buffering=True)

from lib import aggregator, config, data_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Syllable Trace Analysis (Online Aggregation)")
    parser.add_argument("--project", choices=["ifdn", "amph", "all"], required=True)
    parser.add_argument("--force", action="store_true", help="Force re-extraction even if intermediate exists")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top syllables to plot")
    parser.add_argument("--norm-times", type=str, default="-3.0,0.0", help="Comma-separated times (s) to zero traces at (e.g. -3.0,0.0). Use 'None' for raw.")
    parser.add_argument("--filter-1hz", action="store_true", help="Apply 1Hz Bessel High-pass filter to signal before extraction")
    return parser.parse_args()

def apply_bessel_filter(series, fs):
    """High-pass filter (1Hz Bessel) helper, borrowed from panel_1h logic."""
    nyquist = fs / 2.0
    if nyquist <= 1.0: return series
    Wn = 1.0 / nyquist
    # Using SOS for stability
    sos = signal.bessel(1, Wn, btype="high", output="sos")

    # Handle NaNs: filter only valid parts
    valid_mask = ~np.isnan(series)
    result = np.full_like(series, np.nan)
    if np.sum(valid_mask) > 10:
        # fillna with 0 for the filter, then put NaNs back
        # Actually filtfilt handles borders better if we fill with mean or something,
        # but let's stick to the verified Datta logic.
        result[valid_mask] = signal.sosfiltfilt(sos, series[valid_mask])
    return result

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return f"{mem_info.rss / 1e9:.2f} GB"

def save_aggregates_parquet(acc_map, path):
    """
    Converts acc_map to DataFrame and saves as Parquet.
    acc_map keys: (type, syll, reg, hemi, cond, norm_time, is_filtered)
    """
    rows = []
    for key, acc in acc_map.items():
        sig_type, syll, reg, hemi, cond, norm_t, is_filtered = key
        rows.append({
            "type": sig_type,
            "syllable": syll,
            "region": reg,
            "hemi": hemi,
            "condition": cond,
            "norm_time": str(norm_t),
            "is_filtered": bool(is_filtered),
            "n": acc.count,
            "sum": acc.sum.tolist(),
            "sum_sq": acc.sum_sq.tolist(),
            "window_size": acc.window_size
        })

    df = pd.DataFrame(rows)
    for c in ["type", "region", "hemi", "condition", "norm_time"]:
        df[c] = df[c].astype("category")

    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} aggregate rows to {path}")

def load_aggregates_parquet(path):
    """
    Loads Parquet and reconstructs acc_map.
    """
    df = pd.read_parquet(path)
    acc_map = {}

    # Check if 'is_filtered' exists in old parquet
    has_filter_col = "is_filtered" in df.columns

    for row in df.itertuples(index=False):
        norm_t_str = row.norm_time
        norm_t = float(norm_t_str) if norm_t_str != "None" else None

        is_filtered = getattr(row, "is_filtered", False) if has_filter_col else False

        acc = aggregator.TraceAccumulator(int(row.window_size))
        acc.count = int(row.n)
        acc.sum = np.array(row.sum, dtype=np.float64)
        acc.sum_sq = np.array(row.sum_sq, dtype=np.float64)

        key = (row.type, row.syllable, row.region, row.hemi, row.condition, norm_t, is_filtered)
        acc_map[key] = acc

    return acc_map

def run_analysis(project, force=False, top_n=20, norm_times_str="-3.0,0.0", use_filter=False):
    print(f"--- Starting Online Analysis for {project} (Filter: {use_filter}) ---")

    # Parse norm times
    norm_times = []
    for t in norm_times_str.split(","):
        t = t.strip()
        if t.lower() == "none": norm_times.append(None)
        else: norm_times.append(float(t))

    # 1. Load Sessions (needed early to get pkl_path for caching)
    print(f"Loading sessions for {project}...")
    try:
        sessions, pkl_path = data_loader.load_project_sessions(project)
    except Exception as e:
        print(f"Failed to load sessions: {e}")
        return

    # Path for intermediate (versioned by pkl filename)
    inter_dir = config.OUTPUT_DIR / project / "intermediates"
    inter_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = inter_dir / f"aggregates_{pkl_path.stem}.parquet"

    acc_map = {}

    # 2. Load or Extract
    if parquet_path.exists() and not force:
        print(f"Loading cached aggregates from {parquet_path}")
        try:
            acc_map = load_aggregates_parquet(parquet_path)
            # Check if all requested combinations (norm_times x use_filter) are present
            present_combos = set((k[5], k[6]) for k in acc_map.keys())
            if not all((t, use_filter) in present_combos for t in norm_times):
                print("Cache missing requested Norm/Filter combinations. Re-extracting.")
                force = True
                acc_map = {}
            else:
                print(f"Loaded {len(acc_map)} keys.")
        except Exception as e:
             print(f"Failed to load parquet ({e}). Re-extracting.")
             force = True

    if not acc_map or force:
        print(f"Extraction needed. Initial Memory: {get_memory_usage()}")
        # sessions already loaded above

        print(f"Memory after load: {get_memory_usage()}")

        stats = {"events": 0, "processed": 0, "skipped_nan": 0, "skipped_bounds": 0}

        pbar = tqdm(sessions, desc="Sessions")
        window_len = config.WINDOW_FRAMES * 2
        t_axis_raw = np.linspace(-config.WINDOW_SECONDS, config.WINDOW_SECONDS, window_len)

        for i, s in enumerate(pbar):
            try:
                df = s.all_data_df
                meta = s.metadata or {}
                genotype = meta.get("genotype", "UNK").upper() if meta.get("genotype") else "UNK"
                if genotype == "UNK": continue
                drug = meta.get("drug", "saline")
                inj_frame = meta.get("injection_frame")

                # Base condition (for Saline or IFDN)
                base_condition = genotype if project == "ifdn" else f"{drug}-{genotype}"

                sig_meta = data_loader.extract_signal_metadata(df, meta)
                valid_signals = [m for m in sig_meta if m["orig_letter"] == "G" and m["signal"] in df.columns]

                if not valid_signals or "syllable" not in df.columns: continue

                # Pre-filter all signals for this session if requested
                session_signals = {}
                for sig_info in valid_signals:
                    s_name = sig_info["signal"]
                    raw_sig = df[s_name].to_numpy()
                    if use_filter:
                        session_signals[s_name] = apply_bessel_filter(raw_sig, config.FPS)
                    else:
                        session_signals[s_name] = raw_sig

                syl_col = df["syllable"].fillna(-999).astype(int, copy=False)
                onsets_mask = (syl_col != syl_col.shift(1)) & (syl_col != -999)
                onset_indices = df.index[onsets_mask].tolist()
                half_window = config.WINDOW_FRAMES

                for idx in onset_indices:
                    syll_id = syl_col.loc[idx]

                    # Split Amphetamine into Pre/Post
                    condition = base_condition
                    if project == "amph" and drug.lower() == "amphetamine" and inj_frame is not None:
                        try:
                            loc = df.index.get_loc(idx)
                            if isinstance(loc, slice): loc = loc.start
                            elif hasattr(loc, "__len__"): loc = loc[0]

                            curr_frame = df["frame_index"].iloc[loc]
                            if curr_frame < inj_frame:
                                condition = f"Amph-Pre-{genotype}"
                            else:
                                condition = f"Amph-Post-{genotype}"
                        except:
                            # If we can't find the frame, skip or use base_condition
                            pass

                    try:
                        loc = df.index.get_loc(idx)
                        if isinstance(loc, slice): loc = loc.start
                        elif hasattr(loc, "__len__"): loc = loc[0]
                    except KeyError:
                        stats["skipped_bounds"] += len(valid_signals)
                        continue

                    start_loc = loc - half_window
                    end_loc = loc + half_window
                    if start_loc < 0 or end_loc > len(df):
                        stats["skipped_bounds"] += len(valid_signals)
                        continue

                    for sig_info in valid_signals:
                        raw_trace = session_signals[sig_info["signal"]][start_loc:end_loc]
                        if len(raw_trace) != window_len:
                            stats["skipped_bounds"] += 1
                            continue
                        if np.isnan(raw_trace).any():
                            stats["skipped_nan"] += 1
                            continue

                        stats["events"] += 1

                        # Add to multiple accumulators (one per norm_time)
                        for nt in norm_times:
                            processed_trace = raw_trace.copy()
                            if nt is not None:
                                # Find index closest to nt
                                b_idx = np.abs(t_axis_raw - nt).argmin()
                                processed_trace -= processed_trace[b_idx]

                            key = (sig_info["signal_type"], syll_id, sig_info["brain_reg"], sig_info["hemisphere"], condition, nt, use_filter)
                            if key not in acc_map:
                                acc_map[key] = aggregator.TraceAccumulator(window_len)
                            acc_map[key].add(processed_trace)

                        stats["processed"] += 1
            except Exception as e:
                print(f"Error session {s.session_id}: {e}")
                continue
            if i % 10 == 0:
                pbar.set_postfix({"Mem": get_memory_usage(), "Proc": stats["processed"]})
                gc.collect()

        print("\nProcessing Complete. Stats:", stats)
        save_aggregates_parquet(acc_map, parquet_path)
        del sessions
        gc.collect()

    # 3. Plotting with Top-N Filter
    print(f"Generating Top {top_n} plots for norm_times {norm_times}...")

    # Structure: group_map[norm_time][(type, reg, hemi)][syll][condition] = Accumulator
    grouped_for_plotting = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for key, acc in acc_map.items():
        sig_type, syll, reg, hemi, cond, norm_t, is_filtered = key
        if norm_t not in norm_times: continue # Skip cached but not requested
        if is_filtered != use_filter: continue # Focus on requested mode
        grouped_for_plotting[norm_t][(sig_type, reg, hemi)][syll][cond] = acc

    base_out = config.OUTPUT_DIR / project
    window_len = config.WINDOW_FRAMES * 2
    t_axis = np.linspace(-config.WINDOW_SECONDS, config.WINDOW_SECONDS, window_len)

    for norm_t, plots_map in grouped_for_plotting.items():
        norm_label = f"norm_{norm_t}" if norm_t is not None else "raw"
        if use_filter:
            norm_label += "_filtered"

        print(f"Processing folder: {norm_label}")

        # Determine all types in this norm
        all_types = set(k[0] for k in plots_map.keys())
        for stype in all_types:
            type_dir = base_out / str(stype) / norm_label
            if type_dir.exists():
                for f in type_dir.glob("*.png"): f.unlink()
            type_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        pbar_plots = tqdm(plots_map.items(), desc=f"Plots ({norm_label})")

        for group_key, syll_dict in pbar_plots:
            sig_type, reg, hemi = group_key
            type_dir = base_out / str(sig_type) / norm_label

            # Ranking per Group (Across ALL conditions for robustness)
            syll_scores = []
            all_conds = sorted(list(set(c for s_d in syll_dict.values() for c in s_d.keys())))
            for syll, cond_dict in syll_dict.items():
                counts = [cond_dict[c].count if c in cond_dict else 0 for c in all_conds]
                syll_scores.append((syll, min(counts) if counts else 0))
            syll_scores.sort(key=lambda x: x[1], reverse=True)
            top_sylls = [x[0] for x in syll_scores[:top_n]]

            # Determine Plot Sets
            # Default: one plot per syllable with all conditions
            plot_sets = [("", all_conds)]
            if project == "amph":
                # User wants split Pre and Post plots for AMPH
                pre_group = [c for c in all_conds if "Post" not in c]
                post_group = [c for c in all_conds if "Pre" not in c]
                plot_sets = [("_PRE", pre_group), ("_POST", post_group)]

            colors_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            for suffix, target_conds in plot_sets:
                for syll in top_sylls:
                    cond_dict = syll_dict[syll]
                    plt.figure(figsize=(6, 4))
                    has_data = False

                    # Sort target_conds to keep colors consistent
                    # (Saline-WT, Saline-KI, Amph-...)
                    sorted_target = sorted(target_conds)

                    for idx, cond in enumerate(sorted_target):
                        if cond not in cond_dict: continue
                        acc = cond_dict[cond]
                        res = acc.get_stats()
                        if res["n"] < 3: continue
                        has_data = True
                        label = f"{cond} (n={res['n']})"
                        color = colors_cycle[idx % len(colors_cycle)]
                        plt.plot(t_axis, res["mean"], label=label, color=color)
                        border = np.nan_to_num(res["ci_95"], 0)
                        plt.fill_between(t_axis, res["mean"] - border, res["mean"] + border, color=color, alpha=0.2)

                    if not has_data:
                        plt.close()
                        continue

                    plt.axvline(0, linestyle="--", color="black", alpha=0.5)
                    if norm_t is not None:
                        plt.axvline(norm_t, linestyle=":", color="red", alpha=0.3)

                    plt.title(f"Syl {syll} | {reg} {hemi} | {sig_type} {suffix}")
                    plt.xlabel("Time (s)")
                    plt.ylabel(f"Signal ({sig_type})")
                    plt.legend(fontsize='small', loc='upper right')
                    plt.tight_layout()
                    plt.savefig(type_dir / f"{syll}_{reg}_{hemi}{suffix}.png")
                    plt.close()
                    count += 1
        print(f"Generated {count} plots for {norm_label}")

def main():
    args = parse_args()
    projects = [args.project] if args.project != "all" else ["ifdn", "amph"]
    for p in projects:
        run_analysis(p, top_n=args.top_n, force=args.force, norm_times_str=args.norm_times, use_filter=args.filter_1hz)

if __name__ == "__main__":
    main()
