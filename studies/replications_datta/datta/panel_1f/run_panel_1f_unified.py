
"""
Unified Driver for Panel 1f (Event-Triggered Snippet Extraction)
Supports both IFDN (Baseline) and AMPH (Expansion) projects.
"""

import argparse
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed

from fpcode.behavior.kinematics import detect_movement_initiations as _movement_initiations
from fpcode.common.paths import get_repo_root

# --- Pathing ---
repo_root = get_repo_root()

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# --- Constants ---
safe_limit_pct = 0.80
LONG_WINDOW_SECONDS = (-10.0, 10.0)

def log_status(msg, log_path=None):
    print(msg)
    if log_path:
        with open(log_path, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def get_project_config(project_name):
    prj = project_name.lower()
    if prj == "ifdn":
        pkl_pattern = "*ifdn*w_sylls.pkl"
        output_subfolder = "panel_1f_ifdn"
    elif prj == "amph":
        pkl_pattern = "*amph*w_sylls_final.pkl"
        output_subfolder = "panel_1f_amph"
    else:
        raise ValueError(f"Unknown project: {project_name}")
    return {
        "pkl_pattern": pkl_pattern,
        "output_subfolder": output_subfolder,
        "fs": 80.0
    }

def detect_syllable_onsets(df, syllable_col="syllable"):
    if syllable_col not in df.columns:
        return np.array([], dtype=int)
    syl = df[syllable_col].astype(float)
    change = (syl != syl.shift(1)) & syl.notna()
    return np.where(change.to_numpy())[0]

def detect_movement_initiations(df, fs, vel_col="velocity_px_s"):
    if vel_col not in df.columns:
        return np.array([], dtype=int)
    return _movement_initiations(df[vel_col].to_numpy().astype(float), fs,
                                derivative_scale="per_second", missing="zero")

def process_session(s, fs, output_dir, use_signals, memory_limit_bytes=None):
    from studies.replications_datta.datta.lib import features
    if memory_limit_bytes:
        import resource
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        except ValueError:
            pass
    try:
        if not hasattr(s, "all_data_df") or s.all_data_df.empty:
            return None
        df = s.all_data_df.reset_index(drop=True)
        try:
             df = features.calculate_kinematics(df)
        except Exception:
             pass
        if "syllable" not in df.columns:
            return None
        vel_col = "velocity_px_s"
        if vel_col not in df.columns and "velocity_2d_mm" in df.columns:
            vel_col = "velocity_2d_mm"
        syl_onsets = detect_syllable_onsets(df)
        mov_onsets = detect_movement_initiations(df, fs, vel_col=vel_col)
        event_limit = getattr(s, "event_limit", None)
        if event_limit:
            # Sample from middle to avoid start/end padding clips
            if len(syl_onsets) > event_limit:
                syl_onsets = syl_onsets[500:500+event_limit]
            if len(mov_onsets) > event_limit:
                mov_onsets = mov_onsets[500:500+event_limit]

        if len(syl_onsets) == 0 and len(mov_onsets) == 0:
            return None
        meta = s.metadata or {}
        genotype = meta.get("genotype", "WT")
        animal = meta.get("animal_id", "UNK")
        drug = meta.get("drug", "baseline")
        if genotype == "UNK":
             return None
        signal_meta = meta.get("signal_meta", {})
        valid_signals = []
        sig_details = []
        for sig_name in use_signals:
            if sig_name not in df.columns:
                 continue
            match = re.search(r"(signal_\d+)", sig_name)
            base_sig = match.group(1) if match else None
            s_info = signal_meta.get(base_sig, {}) if base_sig else {}
            hemi_string = str(s_info.get("hemisphere", "UNK")).upper()
            reg_string = str(s_info.get("brain_reg", "UNK")).upper()
            orig_letter = s_info.get("orig_letter", "UNK")
            region = "UNK"
            for r_target in ["DMS", "VS", "DLS"]:
                if r_target in reg_string:
                    region = r_target
                    break
            hemi = "UNK"
            if "LEFT" in hemi_string or hemi_string == "L":
                hemi = "L"
            elif "RIGHT" in hemi_string or hemi_string == "R":
                hemi = "R"
            if region != "UNK" and hemi != "UNK":
                valid_signals.append(sig_name)
                sig_details.append({
                    "signal_source": sig_name, "hemisphere": hemi, "brain_reg": region, "orig_letter": orig_letter
                })
        if not valid_signals:
            return None
        lw_start, lw_stop = LONG_WINDOW_SECONDS
        win_size = int(lw_stop * fs) - int(lw_start * fs) + 1
        total_events = len(syl_onsets) + len(mov_onsets)
        total_rows = total_events * len(valid_signals) * win_size
        arr_val = np.zeros(total_rows, dtype=np.float32)
        arr_snip_id = np.zeros(total_rows, dtype=np.int32)
        snippet_map = []
        arr_init_type = np.zeros(total_rows, dtype=np.int8)
        arr_sig_idx = np.zeros(total_rows, dtype=np.int8)
        df_sigs = [df[s].to_numpy(dtype=np.float32) for s in valid_signals]
        start_offset = int(lw_start * fs)
        stop_offset = int(lw_stop * fs)
        cursor = 0
        snip_counter = 0
        max_len = len(df)
        for type_code, onsets in [(0, syl_onsets), (1, mov_onsets)]:
             type_str = "syllables_offline" if type_code == 0 else "movement"
             for i, pos in enumerate(onsets):
                 start = pos + start_offset
                 stop = pos + stop_offset + 1
                 if start < 0 or stop > max_len or (stop - start) != win_size:
                     continue
                 snip_str = f"{s.session_id}_{type_str}_{i}"
                 snippet_map.append(snip_str)
                 current_snip_idx = snip_counter
                 snip_counter += 1
                 for s_idx, sig_arr in enumerate(df_sigs):
                     dest_start = cursor
                     dest_end = cursor + win_size
                     arr_val[dest_start:dest_end] = sig_arr[start:stop]
                     arr_snip_id[dest_start:dest_end] = current_snip_idx
                     arr_init_type[dest_start:dest_end] = type_code
                     arr_sig_idx[dest_start:dest_end] = s_idx
                     cursor += win_size
        total_filled = cursor
        arr_val = arr_val[:total_filled]
        arr_snip_id = arr_snip_id[:total_filled]
        arr_init_type = arr_init_type[:total_filled]
        arr_sig_idx = arr_sig_idx[:total_filled]
        n_blocks = total_filled // win_size
        t_vec = (np.arange(start_offset, stop_offset + 1) / fs).astype(np.float32)
        arr_x = np.tile(t_vec, n_blocks)
        final_df = pd.DataFrame({"x": arr_x, "value": arr_val})
        final_df["snippet"] = pd.Categorical.from_codes(arr_snip_id, categories=snippet_map)
        final_df["init_type"] = pd.Categorical.from_codes(arr_init_type, categories=["syllables_offline", "movement"])
        final_df["signal_source"] = np.array([d["signal_source"] for d in sig_details])[arr_sig_idx]
        final_df["signal_source"] = final_df["signal_source"].astype("category")
        final_df["hemisphere"] = np.array([d["hemisphere"] for d in sig_details])[arr_sig_idx]
        final_df["hemisphere"] = final_df["hemisphere"].astype("category")
        final_df["brain_reg"] = np.array([d["brain_reg"] for d in sig_details])[arr_sig_idx]
        final_df["brain_reg"] = final_df["brain_reg"].astype("category")
        final_df["orig_letter"] = np.array([str(d["orig_letter"]) for d in sig_details])[arr_sig_idx]
        final_df["orig_letter"] = final_df["orig_letter"].astype("category")
        final_df["session_id"] = s.session_id
        final_df["animal_id"] = animal
        final_df["genotype"] = genotype
        final_df["drug"] = drug
        for c in ["animal_id", "genotype", "drug"]:
             final_df[c] = final_df[c].astype("category")
        vals_2d = final_df["value"].values.reshape(-1, win_size)
        m = np.nanmean(vals_2d, axis=1, keepdims=True)
        s_std = np.nanstd(vals_2d, axis=1, keepdims=True)
        s_std[s_std == 0] = 1.0
        final_df["value_renorm"] = ((vals_2d - m) / s_std).flatten().astype(np.float32)
        out_path = Path(output_dir) / f"{s.session_id}.parquet"
        final_df.to_parquet(out_path, index=False)
        return str(out_path)
    except Exception as e:
        print(f"Error session {s.session_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Unified Panel 1f Extraction")
    parser.add_argument("--project", type=str, required=True, choices=["ifdn", "amph"])
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--event-limit", type=int, default=None)
    args = parser.parse_args()
    # Config retains the historical syllable_generation/data and output roots.
    # Delay it until orchestration: imports and --help must not create outputs.
    from studies.replications_datta.datta.lib import config

    p_config = get_project_config(args.project)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = config.REPO_ROOT / "analysis/replications/datta/output" / p_config["output_subfolder"]
    snippets_dir = out_dir / "datta_1f_SNIPPETS.parquet"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir.parent / f"status_{args.project}_unified.log"
    log_status(f"--- Unified Extraction: {args.project.upper()} ---", log_path)
    data_dir = config.REPO_ROOT / "data"
    candidates = list(data_dir.glob(p_config["pkl_pattern"]))
    pkl_path = max(candidates, key=lambda p: p.stat().st_mtime)
    with open(pkl_path, "rb") as f:
        sessions = pickle.load(f)
    sessions = [s for s in sessions if hasattr(s, "all_data_df") and not s.all_data_df.empty]
    if args.test:
        sessions = sessions[:2]

    if args.event_limit:
        for s in sessions:
            s.event_limit = args.event_limit
    test_df = sessions[0].all_data_df
    potential = [c for c in test_df.columns if "signal" in c and "zscore" in c]
    if args.project == "ifdn":
         target = ["signal_0_zscore_datta", "signal_2_zscore_datta", "signal_0_zscore", "signal_2_zscore"]
         use_signals = [c for c in target if c in test_df.columns]
         if not use_signals: use_signals = potential[:2]
    else:
        use_signals = potential[:2]
    mem_total = psutil.virtual_memory().total
    worker_limit = int((mem_total * safe_limit_pct) / args.n_jobs)
    log_status(f"Processing {len(sessions)} sessions...", log_path)
    Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_session)(s, p_config["fs"], snippets_dir, use_signals, worker_limit)
        for s in sessions
    )
    log_status("Done.", log_path)

if __name__ == "__main__":
    main()
