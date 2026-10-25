# analysis/attach_kpms.py

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# === user-editable constants ===
PATH_PREFIX = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/"
SESSIONS_PKL = PATH_PREFIX + "data/jonathan_ifdn_sessions_w_sylls.pkl"
RESULTS_PATH = PATH_PREFIX + "syllabizers/keypoint_moseq_syll/demo_project/kappa-scout-full_data_sweep1-09-k3.728e+04/checkpoint.h5"
SYLLABLE_LOADER = "keypoint-moseq"
OVERWRITE_PICKLE = True

# Direct-file execution activates this checkout through imports and main().
REPO_ROOT = Path(__file__).resolve().parents[2]
_bootstrap_paths = []
if __name__ == "__main__" and not __package__:
    for _source_dir in (REPO_ROOT, REPO_ROOT / "src"):
        if str(_source_dir) not in sys.path:
            sys.path.insert(0, str(_source_dir))
            _bootstrap_paths.append(str(_source_dir))

try:
    from studies.session_model_exports.utils.sampling import flatten_results_to_timeline
    from studies.session_model_exports.utils.syll_results import (
        build_result_to_session_map,
        compute_actual_lengths_from_sessions,
        load_kpms_results_and_vfi,
    )
except BaseException:
    for _source_dir in _bootstrap_paths:
        if _source_dir in sys.path:
            sys.path.remove(_source_dir)
    raise


def get_sessions(pkl_path: str | Path):
    sess = pd.read_pickle(pkl_path)
    if not sess:
        raise ValueError(f"No sessions in {pkl_path}")
    return sess


def _attach_kpms_full_to_sessions(
    sessions,
    full_dfs,                      # dict[recording_key] -> full-length DataFrame
    *,
    result_to_session=None,        # optional dict[recording_key] -> session_index
    expected_lengths=None,         # optional dict[recording_key] -> int (true video length)
    key_prefix="kpms",
    single_df=True,                # If True, set session.syllables = DF
):
    if result_to_session is None:
        result_to_session = build_result_to_session_map(
            sessions, results_keys=list(full_dfs.keys()), strict=True
        )

    if expected_lengths is None:
        expected_lengths = compute_actual_lengths_from_sessions(sessions, result_to_session)

    # Check for duplicates if single_df
    if single_df:
        from collections import Counter
        counts = Counter(result_to_session.values())
        # We allow exactly 2 streams per session (normal + flipped) for Double Flip projects
        dup_sessions = [i for i, n in counts.items() if n > 2]
        if dup_sessions:
            ids = [sessions[i].session_id for i in dup_sessions]
            raise ValueError(f"single_df=True but >2 recs map to sessions: {ids}")

    attached = 0
    for rec_key, df in full_dfs.items():
        if rec_key not in result_to_session:
            continue
        si = result_to_session[rec_key]
        exp = expected_lengths.get(rec_key, None)

        # Length check (Skip if mismatch allows, e.g. upsampling)
        if exp is not None and len(df) != int(exp):
             # print(f"Warning: {rec_key} length mismatch. DF={len(df)}, Expected={exp}")
             pass

        if single_df:
            if rec_key.endswith('_flipped'):
                sessions[si].syllables_flipped = df
            else:
                sessions[si].syllables = df
        else:
            if not hasattr(sessions[si], "syllables") or not isinstance(sessions[si].syllables, dict):
                sessions[si].syllables = {}
            sessions[si].syllables[f"{key_prefix}:{rec_key}"] = df

        attached += 1

    return attached


def run_attach(
    sessions_pkl,
    results_path,
    model_name=None,
    *,
    overwrite=True,
    use_cpu=True,
    key_prefix="kpms",
    use_kpms_lib=False,
    downsample_rate=1.0,
    out_path=None
):
    """
    Standardized attachment logic.
    Supports both:
      1. Legacy Mode (use_kpms_lib=False): Uses internal 'load_kpms_results_and_vfi'
      2. Library Mode (use_kpms_lib=True): Uses 'kpms.compute_moseq_df' (Standard columns)

    Existing VFI mapping/conversion is preserved. 'downsample_rate' does not
    infer or rescale coordinates; it only prints a notice when greater than one.
    'use_cpu' is passed only to the legacy loader; library mode does not set a
    device policy. Loading results may extract and write model products.
    """
    sessions_pkl = Path(sessions_pkl)
    results_path = Path(results_path)

    sessions = pd.read_pickle(sessions_pkl)

    # 1. LOAD RESULTS & VFI
    moseq_df = None
    results = None
    vfi = None

    if use_kpms_lib:
        try:
            import keypoint_moseq as kpms
        except ImportError:
            raise ImportError("keypoint_moseq library not installed") from None

        print(f"Loading via KPMS lib from {results_path} / {model_name}...")
        moseq_df = kpms.compute_moseq_df(str(results_path), model_name, smooth_heading=True)

        vfi_path = results_path / model_name / "video_frame_indexes.npz"
        vf_data = np.load(vfi_path)
        vfi = {k: vf_data[k].astype(int) for k in vf_data.files}

    else:
        print(f"Loading raw results from {results_path}...")
        results, vfi = load_kpms_results_and_vfi(results_path, use_cpu=use_cpu)

    # 2. BUILD MAPPING & EXPECTED LENGTHS (Needed to decide on scaling)
    results_keys = list(set(moseq_df.name)) if use_kpms_lib else list(results.keys())
    r2s = build_result_to_session_map(sessions, results_keys=results_keys, strict=True)
    expected = compute_actual_lengths_from_sessions(sessions, r2s)

    # 3. APPLY UPSAMPLING (Conditional)
    if downsample_rate > 1.0:
        # VFI now inherently embeds its scaling and subsetting natively.
        # Length or max-based heuristics fail because active subsetting naturally creates arrays < 50% length.
        print(f"[attach] Downsample rate is {downsample_rate}. Assuming VFI is already Ground Truth mapped.")

    # 4. PREPARE DATAFRAMES (FLATTEN TO TIMELINE)
    if use_kpms_lib:
        # Convert moseq_df (tidy) -> Dict of DFs (flattened/filled)
        # We simulate what 'flatten_results_to_timeline' does but with moseq_df input
        print("Flattening Library Moseq DF to Timeline...")
        full_dfs = {}
        for rk in results_keys:
            if rk not in vfi: continue

            # Filter rows for this recording
            group = moseq_df[moseq_df.name == rk].copy()
            # Sort by frame_index just in case
            group = group.sort_values("frame_index")

            # Use VFI to map to absolute time (which we might have just upsampled)
            # CAUTION: 'group.frame_index' is 0..N-1. We need to map it via VFI.
            raw_indices = group["frame_index"].values.astype(int)

            # Clip if out of bounds (shouldn't happen if VFI matches)
            current_vfi = vfi[rk]
            valid_mask = raw_indices < len(current_vfi)
            raw_indices = raw_indices[valid_mask]
            group = group[valid_mask]

            abs_indices = current_vfi[raw_indices]

            # Create the full timeline DF
            # If scaling > 1, the timeline is defined by min(vfi)..max(vfi)
            # We reindex to filling gaps
            start, end = abs_indices.min(), abs_indices.max()
            full_index = np.arange(start, end + 1)

            # Set index to abs_indices
            group = group.set_index(pd.Index(abs_indices, name="frame_index"))

            # Reindex to full range and FFILL to handle upsampling (e.g. 40Hz -> 80Hz)
            # This ensures we don't have NaNs on the odd frames
            group = group.reindex(full_index).ffill()

            full_dfs[rk] = group

    else:
        full_dfs = flatten_results_to_timeline(results, vfi, actual_lengths=expected)

    # 5. ATTACH
    print(f"Attaching {len(full_dfs)} recordings...")
    attached = _attach_kpms_full_to_sessions(
        sessions,
        full_dfs,
        result_to_session=r2s,
        expected_lengths=expected,
        key_prefix=key_prefix,
        single_df=True # Always attach as main DF for this pipeline?
                       # Or use 'key_prefix' arg to decide?
                       # User script passes key_prefix="kpms"
                       # Let's use single_df=True if use_kpms_lib is True (AMPH mode)
                       # and single_df=False if Legacy keys are used?
                       # Actually, let's default single_df=True if overwrite is True (Standardizing)
    )

    save_path = Path(out_path) if out_path else sessions_pkl
    if save_path.exists() and not overwrite:
        raise FileExistsError(f"{save_path} exists and overwrite=False")

    with open(save_path, "wb") as f:
        pickle.dump(sessions, f)

    return {"attached": attached, "save_path": str(save_path)}

def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="Pass CPU preference to legacy loader (no effect with --use-lib)")
    ap.add_argument("--key-prefix", default="kpms")
    ap.add_argument("--use-lib", action="store_true")
    ap.add_argument("--downsample", type=float, default=1.0)
    ap.add_argument("--out", type=str, default=None, help="Output PKL path")

    args = ap.parse_args(argv)
    out = run_attach(
        args.sessions, args.results,
        model_name=args.model_name,
        overwrite=args.overwrite,
        use_cpu=args.cpu,
        key_prefix=args.key_prefix,
        use_kpms_lib=args.use_lib,
        downsample_rate=args.downsample,
        out_path=args.out
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        for _source_dir in _bootstrap_paths:
            if _source_dir in sys.path:
                sys.path.remove(_source_dir)
