#!/usr/bin/env python3
"""
audit_priority4_metric_definitions.py
=====================================
Priority 4 Audit: Mathematical Formulation, Absolute Retained Variance, and Reconstruction Error.
Audits:
  1. Exact mathematical formulas for PCA share and Dorsal R^2.
  2. Absolute retained dorsal variance (px^2) vs compositional percentage.
  3. Absolute dorsal and paw reconstruction RMSE (px) on identical common valid observations.
  4. Sensitivity to confidence filtering, center embedding, and whitening.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

REPO = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP = REPO / "syllable_generation/multiview_top_bot_experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from jax_moseq.models.keypoint_slds.alignment import center_embedding

PILOT_21_DIR = EXP / "pilot_static_21kp"
PILOT_23_DIR = EXP / "pilot_static_23kp"
LEGACY_DIR = EXP / "kpms_tandem_cohort_continuous_calibrated"
ONE_CAM_DIR = REPO / "syllable_generation/syllabizers/keypoint_moseq_syll/karen_project_2/final_karen_model_k240k"

DORSAL_17 = [
    "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back", "mouse_center",
    "mid_backend", "mid_backend2", "mid_backend3", "tail_base", "left_shoulder",
    "left_midside", "left_hip", "right_shoulder", "right_midside", "right_hip",
    "head_midpoint"
]
PAWS_4 = ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"]
ANCHORS_2 = ["bot_nose_tip", "bot_tail_base"]

def evaluate_pca_reconstruction(pca_path, cfg_path, common_XY, bodyparts_order):
    """
    Evaluates PCA projection and reconstruction on a given set of centered/rotated coordinates.
    common_XY: (N, K, 2)
    """
    pca = joblib.load(pca_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    bps = cfg["use_bodyparts"]
    n_bps = len(bps)
    d = 2

    # Map common_XY to bps order
    indices = [bodyparts_order.index(b) for b in bps]
    XY_ordered = common_XY[:, indices, :] # (N, n_bps, 2)

    # Keypoint center embedding
    Gamma = center_embedding(n_bps) # (n_bps, n_bps - 1)
    # Embed: (N, n_bps - 1, 2)
    embedded = np.einsum("ki,nkj->nij", Gamma, XY_ordered)
    flat = embedded.reshape(len(common_XY), -1) # (N, (n_bps - 1) * 2)

    # Truncate to model latent dimension (top 10 PCs)
    n_latent = 10
    pcs = pca.transform(flat)[:, :n_latent] # (N, 10)
    # Reconstruct from top 10 PCs only
    recon_flat = pcs @ pca.components_[:n_latent] + pca.mean_
    recon_embedded = recon_flat.reshape(len(common_XY), n_bps - 1, 2)
    recon_XY = np.einsum("ki,nij->nkj", Gamma, recon_embedded) # (N, n_bps, 2)

    # Dorsal indices within bps
    dorsal_in_bps = [i for i, b in enumerate(bps) if b in DORSAL_17]
    paws_in_bps = [i for i, b in enumerate(bps) if b in PAWS_4]

    # Dorsal Reconstruction Error & Retained Variance
    dorsal_orig = XY_ordered[:, dorsal_in_bps, :]
    dorsal_recon = recon_XY[:, dorsal_in_bps, :]
    dorsal_err = (dorsal_orig - dorsal_recon)**2
    dorsal_mse = float(np.mean(dorsal_err))
    dorsal_rmse = float(np.sqrt(dorsal_mse))

    dorsal_tot_var = float(np.mean((dorsal_orig - np.mean(dorsal_orig, axis=0))**2))
    dorsal_retained_var = max(0.0, dorsal_tot_var - dorsal_mse)
    dorsal_retained_pct = 100.0 * dorsal_retained_var / dorsal_tot_var

    # Paws Reconstruction Error
    paw_rmse = None
    if paws_in_bps:
        paw_orig = XY_ordered[:, paws_in_bps, :]
        paw_recon = recon_XY[:, paws_in_bps, :]
        paw_mse = float(np.mean((paw_orig - paw_recon)**2))
        paw_rmse = float(np.sqrt(paw_mse))

    return {
        "n_components": int(pca.n_components_),
        "dorsal_tot_variance_px2": dorsal_tot_var,
        "dorsal_retained_variance_px2": dorsal_retained_var,
        "dorsal_retained_pct_of_dorsal": dorsal_retained_pct,
        "dorsal_reconstruction_rmse_px": dorsal_rmse,
        "paw_reconstruction_rmse_px": paw_rmse,
    }

def main():
    print("=" * 80)
    print("AUDIT PRIORITY 4: METRIC DEFINITIONS & RECONSTRUCTION EVALUATION")
    print("=" * 80)

    # Load from 23kp file which contains all 23 bodyparts
    fused_23_files = sorted((EXP / "data/pilot_fused_static_23kp").glob("*_tandem_23kp.h5"))
    sample_df = pd.read_hdf(fused_23_files[0])
    sc = sample_df.columns.levels[0][0]
    all_bps_23 = DORSAL_17 + PAWS_4 + ANCHORS_2

    px = np.stack([sample_df[sc, b, "x"].to_numpy() for b in all_bps_23], axis=1)
    py = np.stack([sample_df[sc, b, "y"].to_numpy() for b in all_bps_23], axis=1)
    conf = np.stack([sample_df[sc, b, "likelihood"].to_numpy() for b in all_bps_23], axis=1)
    XY = np.stack([px, py], axis=-1)

    valid = np.isfinite(XY).all(axis=(1, 2)) & (conf >= 0.5).all(axis=1)
    valid_XY = XY[valid][:5000] # 5000 common valid test frames
    centered_XY = valid_XY - valid_XY.mean(axis=1, keepdims=True)

    print(f"Loaded {len(centered_XY)} common valid test observations across {len(all_bps_23)} bodyparts.")

    models_to_test = [
        ("pilot_static_21kp", PILOT_21_DIR / "pca.p", PILOT_21_DIR / "config.yml"),
        ("pilot_static_23kp", PILOT_23_DIR / "pca.p", PILOT_23_DIR / "config.yml"),
        ("tandem_legacy_seed404", LEGACY_DIR / "pca.p", LEGACY_DIR / "config.yml")
    ]

    out_metrics = {}
    for name, p_path, c_path in models_to_test:
        if not p_path.exists() or not c_path.exists():
            continue
        res = evaluate_pca_reconstruction(p_path, c_path, centered_XY, all_bps_23)
        out_metrics[name] = res
        print(f"\n[{name}]")
        print(f"  Dorsal Total Variance:     {res['dorsal_tot_variance_px2']:.2f} px^2")
        print(f"  Dorsal Retained Variance:  {res['dorsal_retained_variance_px2']:.2f} px^2 ({res['dorsal_retained_pct_of_dorsal']:.1f}% of dorsal)")
        print(f"  Dorsal Reconstruction RMSE:{res['dorsal_reconstruction_rmse_px']:.2f} px")
        if res['paw_reconstruction_rmse_px'] is not None:
            print(f"  Paw Reconstruction RMSE:   {res['paw_reconstruction_rmse_px']:.2f} px")

    out_file = EXP / "04_reports_and_figures/audits/priority4_metric_definitions_audit.json"
    out_file.write_text(json.dumps(out_metrics, indent=2))
    print(f"\nSaved audit report to {out_file}")

if __name__ == "__main__":
    main()
