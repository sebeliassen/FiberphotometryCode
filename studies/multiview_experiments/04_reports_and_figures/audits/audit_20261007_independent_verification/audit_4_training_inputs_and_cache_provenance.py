#!/usr/bin/env python3
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from fpcode.common.paths import get_repo_root, resolve_data_path

REPO = Path(str(get_repo_root()))
EXP = REPO / 'studies/multiview_experiments'
AUDIT_DIR = EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification"

COHORT_DIR = resolve_data_path('data/models/cohort_static_21kp')
MODEL_DIR = COHORT_DIR / "model_21kp_static_50ar_200slds_seed101"
FUSED_DIR = resolve_data_path('data/fused/multiview/cohort_fused_static_21kp')

def audit_cache_provenance():
    prep_path = COHORT_DIR / "preprocessed_data.pkl"
    pca_path = COHORT_DIR / "pca.p"
    vfi_path = COHORT_DIR / "video_frame_indexes.npz"
    cfg_path = COHORT_DIR / "config.yml"
    res_path = MODEL_DIR / "results.h5"
    ckpt_path = MODEL_DIR / "checkpoint.h5"

    report = {}

    # File Timestamps & Sizes
    files = {
        "preprocessed_data.pkl": prep_path,
        "pca.p": pca_path,
        "video_frame_indexes.npz": vfi_path,
        "config.yml": cfg_path,
        "results.h5": res_path,
        "checkpoint.h5": ckpt_path
    }

    file_info = {}
    for name, p in files.items():
        if p.exists():
            file_info[name] = {
                "exists": True,
                "size_bytes": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "mtime_iso": pd.Timestamp(p.stat().st_mtime, unit="s").isoformat()
            }
        else:
            file_info[name] = {"exists": False}
    report["files"] = file_info

    # 1. Inspect preprocessed_data.pkl
    print("Loading preprocessed_data.pkl...")
    with open(prep_path, "rb") as f:
        data, metadata = pickle.load(f)

    # Check structure
    Y = data["Y"] # array of sequences: (N_seq, T, K, 2)
    mask = data["mask"] # (N_seq, T)
    print(f"Data contains {len(Y)} continuous segments.")
    print(f"First segment shape: {Y[0].shape}, mask shape: {mask[0].shape}")

    # Bodyparts
    with open(cfg_path) as f:
        import yaml
        cfg = yaml.safe_load(f)
    bps = cfg["use_bodyparts"]
    report["use_bodyparts"] = bps
    report["n_bodyparts"] = len(bps)
    print(f"Config bodyparts ({len(bps)}): {bps}")

    # Inspect missingness across all frames
    total_valid_frames = int(sum(m.sum() for m in mask))
    total_padded_frames = int(sum(len(m) for m in mask))
    print(f"Total valid frames: {total_valid_frames:,} (padded: {total_padded_frames:,})")

    # Check per-paw missingness
    paw_names = ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"]
    paw_indices = [bps.index(p) for p in paw_names]

    paw_missing_counts = {p: 0 for p in paw_names}
    paw_nan_counts = {p: 0 for p in paw_names}

    for y_seq, m_seq in zip(Y, mask):
        valid_y = np.array(y_seq)[np.array(m_seq) > 0.5] # (T_valid, K, 2)
        for p_name, p_idx in zip(paw_names, paw_indices):
            # Missing in KPMS means NaN or masked
            nans = np.isnan(valid_y[:, p_idx]).any(axis=-1).sum()
            paw_nan_counts[p_name] += int(nans)

    paw_nan_pct = {p: float(paw_nan_counts[p] / total_valid_frames * 100) for p in paw_names}
    report["total_valid_frames"] = total_valid_frames
    report["total_padded_frames"] = total_padded_frames
    report["paw_nan_percentage"] = paw_nan_pct
    print(f"Per-paw NaN rate: {paw_nan_pct}")

    # 2. Check alignment with fused files
    fused_files = sorted(FUSED_DIR.glob("*_tandem_21kp.h5"))
    print(f"Total fused files in data/cohort_fused_static_21kp: {len(fused_files)}")
    report["n_fused_files"] = len(fused_files)

    # Compare sample session from fused H5 to preprocessed Y
    sample_sid = metadata[0][0].decode("utf-8") if isinstance(metadata[0][0], bytes) else str(metadata[0][0])
    base_sid = sample_sid.split("_seg")[0]
    sample_h5 = FUSED_DIR / f"{base_sid}_tandem_21kp.h5"

    mismatch_detected = False
    if sample_h5.exists():
        df_sample = pd.read_hdf(sample_h5)
        sc = df_sample.columns.levels[0][0]
        # Compare first valid frame
        t_sample_x = df_sample[sc, bps[0], "x"].to_numpy()
        # In preprocessed data, Y is centered or egocentric?
        # In KPMS preprocessed_data.pkl, coordinates are raw or centered?
        print(f"Sample H5 {bps[0]} x (first 3): {t_sample_x[:3]}")
        print(f"Preprocessed Y[0] {bps[0]} x (first 3): {Y[0][:3, 0, 0]}")
        report["sample_fused_vs_prep"] = {
            "h5_val": float(t_sample_x[0]),
            "prep_val": float(Y[0][0, 0, 0])
        }

    # Check VFI
    with np.load(vfi_path) as z:
        vfi_keys = z.files
    report["vfi_session_count"] = len(vfi_keys)
    print(f"VFI keys count: {len(vfi_keys)}")

    # Check Results H5 keys
    with h5py.File(res_path, "r") as res:
        res_keys = list(res.keys())
    report["results_session_count"] = len(res_keys)
    print(f"Results H5 keys count: {len(res_keys)}")

    # Check if VFI keys match Results keys
    vfi_set = set(vfi_keys)
    res_set = set(res_keys)
    vfi_res_diff = len(vfi_set.symmetric_difference(res_set))
    report["vfi_vs_results_diff_count"] = vfi_res_diff
    print(f"VFI vs Results symmetric difference: {vfi_res_diff}")

    # Check timestamp dependencies in manifests
    manifest_files = list(FUSED_DIR.glob("*.manifest.json"))
    report["fused_manifests_count"] = len(manifest_files)
    if manifest_files:
        sample_mf = json.loads(manifest_files[0].read_text())
        inputs_in_mf = [inp.get("name", "") for inp in sample_mf.get("inputs", [])]
        report["sample_manifest_inputs"] = inputs_in_mf
        has_ts = any("AllTimestampsCombined" in str(x) or ".csv" in str(x) for x in inputs_in_mf)
        report["manifest_tracks_timestamp_csvs"] = has_ts
        print(f"Manifest tracks timestamp CSVs: {has_ts}")

    out_json = AUDIT_DIR / "training_inputs_and_cache_provenance.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {out_json}")

if __name__ == "__main__":
    audit_cache_provenance()
