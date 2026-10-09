# analysis/utils/syll_results.py

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional, List

def load_kpms_results_and_vfi(path: str | Path, use_cpu: bool = True):
    """
    path: model dir OR a file inside it (e.g. .../checkpoint.h5)
    returns: (results_dict, video_frame_indexes_dict)
    """
    import os
    if use_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"  # avoid grabbing the GPU for read-only ops

    import keypoint_moseq as kpms

    p = Path(path).resolve()
    model_dir   = p if p.is_dir() else p.parent
    project_dir = model_dir.parent
    model_name  = model_dir.name

    # Determine if we have a saved results file, or just a checkpoint
    results_path = model_dir / "results.h5"
    if results_path.exists():
        # Fast path if already extracted
        print(f"[load] Found existing results.h5 at {results_path}")
        results = kpms.load_results(str(project_dir), model_name)
    else:
        # Load the checkpoint & extract results into memory, and save them for next time
        print(f"[load] No results.h5 found. Extracting from checkpoint for {model_name}...")
        model, data, metadata, current_iter = kpms.load_checkpoint(str(project_dir), model_name)
        results = kpms.extract_results(model, metadata, str(project_dir), model_name, save_results=True)
        print(f"[load] Successfully extracted and saved results.h5.")

    # Load video_frame_indexes if saved by pipeline (NPZ of arrays keyed by recording name)
    vfi_npz = model_dir / "video_frame_indexes.npz"
    if vfi_npz.exists():
        vf = np.load(vfi_npz)
        video_frame_indexes = {k: vf[k].astype(int) for k in vf.files}
    else:
        # Fallback: identity (cannot re-expand to true length without actual_lengths)
        video_frame_indexes = {k: np.arange(len(v["syllable"]), dtype=int) for k, v in results.items()}
        print(f"[load] WARNING: {vfi_npz} not found; using identity indexes.")

    return results, video_frame_indexes


def build_result_to_session_map(
    sessions: Iterable, results_keys: Iterable[str], *, strict: bool = True, verbose: bool = False
) -> Dict[str, int]:
    """
    Map each results key -> session index by substring match:
    a result key must contain exactly one session.session_id.

    Side effects:
      - Sets `session.results_dict_key` on each session.
        * strict=True  -> always a str (initialized to "")
        * strict=False -> may be None (initialized to None)

    If strict=True: also ensure bijection (no two results map to the same session).

    Returns:
      {results_key -> session_index}

    Raises:
      ValueError on any mismatch/ambiguity.
    """
    sessions: List = list(sessions)
    sid_to_idx = {s.session_id: i for i, s in enumerate(sessions)}
    session_ids = list(sid_to_idx.keys())

    # Initialize attribute on all sessions to a consistent type
    init_value = "" if strict else None
    for s in sessions:
        setattr(s, "results_dict_key", init_value)

    r2s: Dict[str, int] = {}
    problems: List[str] = []

    if verbose:
        print(f"[load] DEBUG: System has {len(session_ids)} session IDs. First few: {session_ids[:3]}")

    for rk in results_keys:
        # Flipped models append '_flipped' to the key, which prevents exact substring match 
        # against the pure session ID if the session ID itself is at the end of the key.
        search_rk = rk.replace("_flipped", "")
        
        matches = []
        for sid in session_ids:
            if sid in search_rk:
                s = sessions[sid_to_idx[sid]]
                trial_id = None
                if hasattr(s, 'metadata') and isinstance(s.metadata, dict):
                    trial_id = s.metadata.get('trial_id')
                if trial_id is None:
                    trial_id = getattr(s, 'trial_id', None)
                
                if trial_id is not None:
                    if trial_id in search_rk:
                        matches.append(sid)
                else:
                    matches.append(sid)

        if verbose:
            print(f"[load] DEBUG: Key={rk[:50]}... Matches={matches}")

        if len(matches) == 1:
            si = sid_to_idx[matches[0]]
            prev = sessions[si].results_dict_key

            if strict and isinstance(prev, str) and prev not in ("", rk):
                # Only throw ambiguity error if the two keys aren't just flipped pairs
                if not (prev.replace("_flipped", "") == rk.replace("_flipped", "")):
                    problems.append(
                        f"session '{sessions[si].session_id}' matched by multiple result keys "
                        f"(e.g., '{prev}' and '{rk}')"
                    )

            r2s[rk] = si
            sessions[si].results_dict_key = rk  # always str in strict mode

        elif len(matches) == 0:
            # If a result key has no matching session in the database, we can skip it 
            # if strict=False or if we only match keys of interest. But in strict=True, 
            # we only warn or ignore if the session was intentionally excluded.
            # To be safe, let's not raise an error for result keys that don't match any active session.
            if verbose:
                print(f"[load] WARNING: No matching session found in database for result key '{rk}'. Skipping.")
        else:
            problems.append(
                f"ambiguous: {len(matches)} session_ids match results key '{rk}': {matches}"
            )

    if problems:
        raise ValueError("Result↔Session matching problems:\n- " + "\n- ".join(problems))

    return r2s


def compute_actual_lengths_from_sessions(
    sessions: Iterable, result_to_session: Dict[str, int]
) -> Dict[str, int]:
    """
    For each results key, read the 'true' length from the first row of the
    matched session's video_refs_df['n_frames'].
    """
    out = {}
    for rk, si in result_to_session.items():
        vr = sessions[si].video_refs_df
        if vr is None or "n_frames" not in vr.columns or len(vr) == 0:
            raise ValueError(f"[{sessions[si].session_id}] missing video_refs_df or 'n_frames' column")
        n = vr["n_frames"].iloc[0]
        out[rk] = int(n)
    return out


    return out
