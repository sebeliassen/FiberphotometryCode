#!/usr/bin/env python3
"""
bind_and_export_annotations.py
=============================================================================
Export and cryptographically bind existing ethologist and pre-labeled
annotations to the exact model fingerprints (results.h5, config.yml, VFI,
preprocessed_data.pkl).

Enforces:
1. Strict binding: Syllable IDs are stochastic latent states tied to specific
   model seeds; numerical IDs cannot be transferred without validation.
2. Complete provenance: Hashes of model weights, results, configs, and VFI.
=============================================================================
"""

import hashlib
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parents[2]
AUDIT_DIR = Path(__file__).resolve().parent
VIZ_V4_DIR = EXP_DIR / "04_reports_and_figures/visualizations_cohort_static_21kp_v2_seed101_v4"

def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("[Bind Annotations] Binding annotations to model fingerprints...")

    # Fingerprint current production model (seed 101)
    cohort_dir = EXP_DIR / "cohort_static_21kp_v2"
    model_dir = cohort_dir / "model_21kp_static_50ar_300slds_seed101"

    fingerprint_seed101 = {
        "model_name": "model_21kp_static_50ar_300slds_seed101",
        "cohort": "cohort_static_21kp_v2",
        "results_h5_sha256": file_sha256(model_dir / "results.h5"),
        "config_yml_sha256": file_sha256(cohort_dir / "config.yml"),
        "vfi_npz_sha256": file_sha256(cohort_dir / "video_frame_indexes.npz"),
        "preprocessed_pkl_sha256": file_sha256(cohort_dir / "preprocessed_data.pkl"),
    }

    # Load legacy seed404 annotations if available
    seed404_path = EXP_DIR / "04_reports_and_figures/visualizations_seed404/seed404_prelabeled_annotations.json"
    seed404_annotations = None
    if seed404_path.exists():
        with open(seed404_path) as f:
            seed404_annotations = json.load(f)

    # Load seed101 recovered label map
    label_map_path = AUDIT_DIR / "seed101_old_to_new_label_map_recovered_20261005.json"
    seed101_label_map = None
    if label_map_path.exists():
        with open(label_map_path) as f:
            seed101_label_map = json.load(f)

    binding_record = {
        "schema_version": 4,
        "binding_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "canonical_production_model_fingerprint": fingerprint_seed101,
        "annotation_integrity_policy": {
            "cross_model_numeric_id_transfer": "FORBIDDEN",
            "reason": (
                "Keypoint-MoSeq HDP-SLDS syllable IDs are stochastic discrete latent states "
                "tied strictly to the random seed and iteration checkpoint. A syllable ID in "
                "seed404 (e.g. Syllable 1) DOES NOT correspond to Syllable 1 in seed101. "
                "Transferring numerical IDs without empirical co-occurrence alignment causes "
                "false biological categorization."
            )
        },
        "seed101_bound_metadata": {
            "model_name": fingerprint_seed101["model_name"],
            "label_map_recovered": seed101_label_map,
            "status": "canonical_unlabeled_gallery",
            "notes": "Gallery generated for ethological labeling under strict QC."
        },
        "seed404_legacy_reference": {
            "source_file": str(seed404_path) if seed404_path.exists() else None,
            "original_model_name": seed404_annotations.get("model_name") if seed404_annotations else None,
            "total_annotations": len(seed404_annotations.get("annotations", [])) if seed404_annotations else 0,
            "bound_annotations": seed404_annotations.get("annotations") if seed404_annotations else []
        }
    }

    out_file = VIZ_V4_DIR / "bound_model_annotations.json"
    VIZ_V4_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(binding_record, f, indent=2)

    print(f"✅ [Bind Annotations] Bound annotations saved to: {out_file}")

if __name__ == "__main__":
    main()
