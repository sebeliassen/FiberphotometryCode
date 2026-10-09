#!/usr/bin/env python3
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, h5py

REPO = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP = REPO / "syllable_generation/multiview_top_bot_experiments"
AUDIT_DIR = EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification"

VIZ_DIR = EXP / "04_reports_and_figures/visualizations_cohort_static_21kp_seed101"
GRID_PKL = VIZ_DIR / "density_sampled_instances.pkl" # contains bouts chosen by grid movie / density
TRAJ_PKL = VIZ_DIR / "sampled_median_trajectories.pkl"

def audit_heading_filter():
    arr = np.array([-3.13]*4 + [0.0] + [3.13]*4)
    scalar_med = float(np.median(arr))
    sin_med = np.median(np.sin(arr))
    cos_med = np.median(np.cos(arr))
    circ_med = float(np.arctan2(sin_med, cos_med))
    return {
        "input_angles_rad": arr.tolist(),
        "scalar_median_rad": scalar_med,
        "scalar_median_deg": float(np.degrees(scalar_med)),
        "circular_median_rad": circ_med,
        "circular_median_deg": float(np.degrees(circ_med)),
        "angular_error_deg": float(np.abs(np.degrees(circ_med) - np.degrees(scalar_med)))
    }

def audit_bouts_selection():
    # Load density_sampled_instances.pkl
    with open(GRID_PKL, "rb") as f:
        grid_bouts = pickle.load(f)
    with open(TRAJ_PKL, "rb") as f:
        traj_data = pickle.load(f)
    
    # Check overlap of bouts for syllables
    sylls = sorted(list(grid_bouts.keys()))[:10]
    comparison = {}
    for s in sylls:
        gb = set((b[0], b[1], b[2]) for b in grid_bouts[s])
        # Trajectories store 3D array (T, K, 2)
        comparison[f"syllable_{s}"] = {
            "n_grid_bouts": len(gb),
            "sample_grid_bouts": list(gb)[:3],
        }
    return comparison

def main():
    h_res = audit_heading_filter()
    b_res = audit_bouts_selection()
    out = {
        "heading_median_filter_defect": h_res,
        "bouts_comparison": b_res
    }
    out_file = AUDIT_DIR / "heading_and_bouts_audit.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"Saved heading and bouts audit to {out_file}")
    print(f"Heading scalar median: {h_res['scalar_median_deg']:.1f} deg, Circular: {h_res['circular_median_deg']:.1f} deg")

if __name__ == "__main__":
    main()
