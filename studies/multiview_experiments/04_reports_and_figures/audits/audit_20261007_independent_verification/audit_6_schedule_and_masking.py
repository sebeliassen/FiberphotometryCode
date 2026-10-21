#!/usr/bin/env python3
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import adjusted_rand_score

from fpcode.common.paths import get_repo_root, resolve_data_path

REPO = Path(str(get_repo_root()))
EXP = REPO / 'studies/multiview_experiments'
AUDIT_DIR = EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification"

COHORT_DIR = resolve_data_path('data/models/cohort_static_21kp')
MODEL_DIR = COHORT_DIR / "model_21kp_static_50ar_200slds_seed101"

def audit_schedule_truncation():
    chunks = [
        (0, 50, True, "AR Phase (0 -> 50)"),
        (50, 100, False, "SLDS Phase 1 (50 -> 100)"),
        (100, 150, False, "SLDS Phase 2 (100 -> 150)"),
        (150, 200, False, "SLDS Phase 3 (150 -> 200)"),
        (200, 250, False, "SLDS Phase 4 (200 -> 250)"),
    ]
    # Test what happens when num_slds_iters is set to 300
    requested_ar = 50
    requested_slds = 300
    total_requested = requested_ar + requested_slds
    max_chunk_iter = chunks[-1][1]

    return {
        "hardcoded_chunks_max_iter": max_chunk_iter,
        "requested_total_iters_example": total_requested,
        "executed_iters": min(total_requested, max_chunk_iter),
        "unexecuted_iters": max(0, total_requested - max_chunk_iter),
        "is_schedule_truncated": bool(total_requested > max_chunk_iter)
    }

def audit_masking_impact():
    with open(COHORT_DIR / "preprocessed_data.pkl", "rb") as f:
        data, _ = pickle.load(f)

    mask_full = np.array(data["mask"]) > 0.5
    with h5py.File(MODEL_DIR / "checkpoint.h5", "r") as f:
        z225 = f["model_snapshots/225/states/z"][:]
        z250 = f["model_snapshots/250/states/z"][:]

    nlags = mask_full.shape[1] - z250.shape[1]
    mask = mask_full[:, nlags:]

    # Unmasked
    u_225 = z225.ravel()
    u_250 = z250.ravel()
    u_val = (u_225 >= 0) & (u_250 >= 0)
    u_ari = float(adjusted_rand_score(u_225[u_val], u_250[u_val]))
    u_flip = float(np.mean(u_225[u_val] != u_250[u_val]) * 100.0)

    # Masked
    m_225 = z225[mask]
    m_250 = z250[mask]
    m_val = (m_225 >= 0) & (m_250 >= 0)
    m_ari = float(adjusted_rand_score(m_225[m_val], m_250[m_val]))
    m_flip = float(np.mean(m_225[m_val] != m_250[m_val]) * 100.0)

    return {
        "unmasked_frames": int(len(u_250)),
        "masked_frames": int(len(m_250)),
        "padded_dummy_frames": int(len(u_250) - len(m_250)),
        "unmasked_iter_225_to_250_ari": u_ari,
        "masked_iter_225_to_250_ari": m_ari,
        "unmasked_label_flip_pct": u_flip,
        "masked_label_flip_pct": m_flip,
    }

def main():
    s_res = audit_schedule_truncation()
    m_res = audit_masking_impact()
    out = {
        "schedule_truncation": s_res,
        "masking_impact": m_res
    }
    out_file = AUDIT_DIR / "schedule_and_masking_audit.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"Saved schedule and masking audit to {out_file}")
    print(f"Padded dummy frames in evaluation: {m_res['padded_dummy_frames']:,}")

if __name__ == "__main__":
    main()
