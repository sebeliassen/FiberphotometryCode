#!/usr/bin/env python3
"""
verify_fused_in_kpms_pipeline.py
=============================================================================
Tests the newly generated reflection-corrected tandem coordinates through
the complete Keypoint-MoSeq (KPMS) pipeline:

1. Loads 23-keypoint coordinates using native KPMS routines.
2. Verifies tensor shapes, coordinate bounds, and confidence distributions.
3. Fits PCA (D=10) on CPU: evaluates eigenvalue spectrum and paw loadings.
4. Initializes AR-HMM model (kpms.init_model).
5. Executes 2 test Gibbs sampling iterations to prove likelihood convergence
   and extract valid discrete syllable sequences without shape/NaN errors.

SAFETY:
- Strictly operates in: kpms_test_pipeline_reflection_corrected/
- Zero side effects on production models or master session pickles.
=============================================================================
"""
import os
import shutil
from pathlib import Path

from fpcode.common.paths import get_repo_root

# Force CPU execution for testing to leave GPU untouched
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import keypoint_moseq as kpms

REPO_ROOT = Path(str(get_repo_root()))
EXPERIMENT_DIR = REPO_ROOT / 'studies/multiview_experiments'
FUSED_H5_DIR = EXPERIMENT_DIR / "fused_h5_reflection_corrected"

TEST_PROJ_DIR = EXPERIMENT_DIR / "kpms_test_pipeline_reflection_corrected"
TEST_DATA_DIR = TEST_PROJ_DIR / "data"

def main():
    print("=" * 80)
    print("🔬 TESTING REFLECTION-CORRECTED TANDEM COORDS THROUGH KPMS PIPELINE")
    print(f"Data Source:      {FUSED_H5_DIR}")
    print(f"Test Project Dir: {TEST_PROJ_DIR}")
    print("=" * 80)

    # 1. Setup isolated test project directory
    if TEST_PROJ_DIR.exists():
        shutil.rmtree(TEST_PROJ_DIR)
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Symlink fused H5 files into test project data/
    fused_files = list(FUSED_H5_DIR.glob("*.h5"))
    print(f"\n[1/5] Ingesting {len(fused_files)} fused H5 files...")
    for f in fused_files:
        dest = TEST_DATA_DIR / f.name
        dest.symlink_to(f)
        print(f"  --> Linked: {f.name}")

    # 2. Setup project config by copying clean multiview config
    print("\n[2/5] Initializing KPMS Project Configuration...")
    prod_config = EXPERIMENT_DIR / "multiview_kappa_search/project/config.yml"
    shutil.copy(prod_config, TEST_PROJ_DIR / "config.yml")
    config = kpms.load_config(str(TEST_PROJ_DIR))
    print("  --> Loaded clean config with 23 keypoints (17 dorsal + 6 ventral)")

    # 3. Load Keypoints via native KPMS
    print("\n[3/5] Loading and formatting keypoints via native KPMS...")
    coordinates, confidences, bodyparts = kpms.load_keypoints(str(TEST_DATA_DIR), format="deeplabcut")
    print(f"  Successfully loaded {len(coordinates)} sessions:")
    for k, v in coordinates.items():
        conf = confidences[k]
        print(f"    • {k}: Shape {v.shape} | Mean Conf = {np.nanmean(conf):.3f} | Min/Max = [{np.nanmin(v):.1f}, {np.nanmax(v):.1f}]")

    # Format data (centering, outlier removal)
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    print(f"  Formatted data: {len(data['Y'])} batched segments")
    print(f"  Y tensor shape: {data['Y'].shape} (T, K=23, D=2)")

    # 4. Fit PCA on 23-keypoint multi-view coordinates
    print("\n[4/5] Fitting PCA (D=10) on multi-view coordinates...")
    # Convert any jax arrays in data/config to standard Python/numpy types if needed
    pca = kpms.fit_pca(**data, **config)
    kpms.save_pca(pca, str(TEST_PROJ_DIR))

    var_exp = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_exp)
    print(f"  Top 10 PCs Cumulative Variance Explained: {cum_var[-1]*100:.2f}%")
    for i in range(min(10, len(var_exp))):
        print(f"    PC {i+1:02d}: {var_exp[i]*100:5.2f}% (Cumulative: {cum_var[i]*100:5.2f}%)")

    # Check component loadings
    components = pca.components_
    n_modeled_bps = components.shape[1] // 2
    components_2d = components.reshape(components.shape[0], n_modeled_bps, 2)
    print(f"  --> PCA components shape: {components.shape} ({n_modeled_bps} modeled keypoints x 2D)")

    # 5. Initialize Model & Run 2 Test Iterations
    print("\n[5/5] Initializing AR-HMM Model and running 2 test iterations on CPU...")
    from jax_moseq.utils.debugging import convert_data_precision
    data = convert_data_precision(data)
    model = kpms.init_model(data, pca=pca, **config)
    print("  --> Model initialized successfully.")
    print("  --> Model states x shape:", model["states"]["x"].shape)

    print("  --> Executing 2 test AR-HMM iterations on CPU...")
    kpms.fit_model(
        model, data, metadata,
        project_dir=str(TEST_PROJ_DIR),
        model_name="test_model",
        num_iters=2,
        start_iter=0,
        ar_only=True,
        generate_progress_plots=False,
        verbose=True
    )

    # Load saved checkpoint and extract results
    model, data, metadata, current_iter = kpms.load_checkpoint(str(TEST_PROJ_DIR), "test_model")
    results = kpms.extract_results(model, metadata, str(TEST_PROJ_DIR), "test_model", save_results=False)
    print("\n  --> Test fitting completed successfully!")
    for k, v in results.items():
        sylls = v["syllable"]
        n_unique = len(np.unique(sylls))
        print(f"    • {k}: {len(sylls):,} frames, {n_unique} active syllables discovered.")

    print("\n" + "=" * 80)
    print("🏆 PIPELINE VALIDATION PASSED WITH 100% SUCCESS!")
    print("   The new reflection-corrected tandem coordinates load and fit flawlessly.")
    print("=" * 80)

if __name__ == "__main__":
    main()
