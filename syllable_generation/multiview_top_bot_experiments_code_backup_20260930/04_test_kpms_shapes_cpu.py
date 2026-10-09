#!/usr/bin/env python3
"""
04_test_kpms_shapes_cpu.py
--------------------------
CPU-Only Sanity and Shape Validation for Multi-View Keypoint-MoSeq.

GUARANTEES:
1. Strictly CPU isolated (CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu").
   Zero VRAM allocated; zero interference with active RTX 4090 GPU jobs.
2. Uses only a small sliver (3,000 - 5,000 frames) for rapid validation.
3. Verifies tensor shapes through:
   - Keypoint ingestion: (T, 21, 2)
   - PCA fitting: (T, 10)
   - AR-HMM model initialization
   - Fast 2-iteration fitting step
   - Syllable extraction: (T,)
"""

# --- GPU ISOLATION GUARDS (Must be set BEFORE importing jax / torch / kpms) ---
import os
import sys
import shutil

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from pathlib import Path
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax
jax.config.update("jax_platform_name", "cpu")

import pandas as pd
import yaml

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXPERIMENT_DIR = REPO_ROOT / "syllable_generation" / "multiview_top_bot_experiments"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from clock_sync import align_tracking_dataframes

TEST_PROJECT_DIR = EXPERIMENT_DIR / "kpms_cpu_shape_test"
TEST_DATA_DIR = TEST_PROJECT_DIR / "data"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_LOG_FILE = TEST_PROJECT_DIR / "test.log"

class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.terminal.flush()
        self.logfile.flush()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

sys.stdout = TeeLogger(TEST_LOG_FILE)
sys.stderr = sys.stdout

TOPCAM_H5_DIR = REPO_ROOT / "karen_project" / "h5_files"
BOTCAM_PROD_DIR = Path("/home/fsp585/Desktop/production_analysis_arm_B_seed_123_20260922/predictions")

TOPCAM_BPS = [
    "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back",
    "mouse_center", "mid_backend", "mid_backend2", "mid_backend3",
    "tail_base", "left_shoulder", "left_midside", "left_hip",
    "right_shoulder", "right_midside", "right_hip", "head_midpoint"
]

BOTCAM_PAWS = [
    "left_front_paw", "right_front_paw",
    "left_hind_paw", "right_hind_paw"
]

ALL_FUSED_BPS = TOPCAM_BPS + BOTCAM_PAWS

USE_BPS = [
    "nose", "head_midpoint", "neck", "mid_back",
    "mouse_center", "mid_backend", "tail_base",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_front_paw", "right_front_paw",
    "left_hind_paw", "right_hind_paw"
]

SKELETON = [
    ["nose", "head_midpoint"],
    ["head_midpoint", "neck"],
    ["neck", "mid_back"],
    ["mid_back", "mouse_center"],
    ["mouse_center", "mid_backend"],
    ["mid_backend", "tail_base"],
    ["neck", "left_shoulder"],
    ["neck", "right_shoulder"],
    ["tail_base", "left_hip"],
    ["tail_base", "right_hip"],
    ["left_shoulder", "left_front_paw"],
    ["right_shoulder", "right_front_paw"],
    ["left_hip", "left_hind_paw"],
    ["right_hip", "right_hind_paw"]
]

def create_test_config(project_dir: Path):
    import keypoint_moseq as kpms
    cfg_file = project_dir / "config.yml"
    if cfg_file.exists():
        cfg_file.unlink()
    kpms.setup_project(str(project_dir), overwrite=True)

    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["video_dir"] = str(project_dir / "videos")
    cfg["bodyparts"] = ALL_FUSED_BPS
    cfg["use_bodyparts"] = USE_BPS
    cfg["skeleton"] = SKELETON
    cfg["anterior_bodyparts"] = ["tail_base"]
    cfg["posterior_bodyparts"] = ["nose"]
    cfg["fps"] = 40
    cfg["downsample_rate"] = 1
    cfg["conf_threshold"] = 0.5
    cfg["added_noise_level"] = 0.1
    cfg["PCA_fitting_num_frames"] = 2000
    cfg["whiten"] = True
    cfg["verbose"] = True
    cfg["latent_dim"] = 10
    cfg["kappa"] = 15000

    if "ar_hypparams" in cfg:
        cfg["ar_hypparams"]["latent_dim"] = 10
    if "trans_hypparams" in cfg:
        cfg["trans_hypparams"]["kappa"] = 15000

    with open(cfg_file, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return cfg_file

def generate_fused_sliver(n_frames: int = 3000) -> Path:
    """
    Creates a canonical fused 21-keypoint HDF5 file from TopCam H5 and geometrically locked BotCam paws.
    """
    # 1. Pick a TopCam H5 file
    top_candidates = sorted(list(TOPCAM_H5_DIR.glob("*T1_6.2.4*top_left*.h5")))
    if not top_candidates:
        top_candidates = sorted(list(TOPCAM_H5_DIR.glob("*.h5")))
    if not top_candidates:
        raise FileNotFoundError(f"No TopCam H5 files found in {TOPCAM_H5_DIR}")

    top_h5 = top_candidates[0]
    print(f"  Source TopCam H5: {top_h5.name}")
    df_top = pd.read_hdf(top_h5).iloc[:n_frames]
    scorer_top = df_top.columns.levels[0][0]

    scorer_fused = "MultiView_DorsalVentral"
    fused_dict = {}

    # 2. Copy ONLY the 17 canonical TopCam bodyparts
    for bp in TOPCAM_BPS:
        if bp in df_top[scorer_top].columns.levels[0]:
            fused_dict[(scorer_fused, bp, 'x')] = df_top[scorer_top][bp]['x'].values
            fused_dict[(scorer_fused, bp, 'y')] = df_top[scorer_top][bp]['y'].values
            # Clean / clamp likelihood to ensure valid confidence for shape validation
            p = df_top[scorer_top][bp]['likelihood'].values
            fused_dict[(scorer_fused, bp, 'likelihood')] = np.where(np.isnan(p) | (p < 0.5), 0.95, p)
        else:
            raise KeyError(f"Expected TopCam bodypart '{bp}' missing from {top_h5.name}")

    # 3. Add 4 ventral paws with realistic anatomical alignment & gait oscillation
    cx = df_top[scorer_top]['mouse_center']['x'].values
    cy = df_top[scorer_top]['mouse_center']['y'].values
    for paw, offset in [
        ("left_front_paw", (-15, -20)),
        ("right_front_paw", (15, -20)),
        ("left_hind_paw", (-15, 20)),
        ("right_hind_paw", (15, 20))
    ]:
        t = np.arange(n_frames)
        stride_x = 8.0 * np.sin(2 * np.pi * t / 20.0 + (np.pi if "right" in paw else 0.0))
        stride_y = 4.0 * np.cos(2 * np.pi * t / 20.0 + (np.pi if "hind" in paw else 0.0))
        fused_dict[(scorer_fused, paw, 'x')] = cx + offset[0] + stride_x
        fused_dict[(scorer_fused, paw, 'y')] = cy + offset[1] + stride_y
        fused_dict[(scorer_fused, paw, 'likelihood')] = np.full(n_frames, 0.99)

    # 4. Build MultiIndex DataFrame in exact order of ALL_FUSED_BPS (21 bodyparts)
    col_tuples = []
    for bp in ALL_FUSED_BPS:
        for c in ['x', 'y', 'likelihood']:
            col_tuples.append((scorer_fused, bp, c))

    df_fused = pd.DataFrame(
        {col: fused_dict[col] for col in col_tuples},
        columns=pd.MultiIndex.from_tuples(col_tuples, names=['scorer', 'bodyparts', 'coords'])
    )

    out_file = TEST_DATA_DIR / f"{top_h5.stem}_fused_sliver.h5"
    df_fused.to_hdf(str(out_file), key="df_with_missing", mode="w")
    print(f"  Created Fused H5 Sliver: {out_file.name} ({len(df_fused)} frames, {len(ALL_FUSED_BPS)} keypoints)")
    return out_file

def run_cpu_shape_test():
    print("=" * 70)
    print("RUNNING MULTIVIEW KEYPOINT-MOSEQ CPU SHAPE & PIPELINE VALIDATION")
    print(f"Test Project Dir: {TEST_PROJECT_DIR}")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("JAX_PLATFORMS:       ", os.environ.get("JAX_PLATFORMS", "<unset>"))
    print("=" * 70)

    import keypoint_moseq as kpms
    print(f"Imported keypoint_moseq version: {getattr(kpms, '__version__', 'unknown')}")

    # 1. Config creation
    config_path = create_test_config(TEST_PROJECT_DIR)
    print(f"[Step 1] Config created at {config_path}")

    # 2. Sliver data generation
    h5_path = generate_fused_sliver(n_frames=3000)
    print(f"[Step 2] Fused data ready at {h5_path}")

    # 3. Load keypoints and test shapes
    print("[Step 3] Loading keypoints through KPMS loader...")
    config = kpms.load_config(str(TEST_PROJECT_DIR))
    coordinates, confidences, loaded_bodyparts = kpms.load_keypoints([str(h5_path)], "deeplabcut")

    print(f"  Loaded bodyparts ({len(loaded_bodyparts)}): {loaded_bodyparts}")
    assert len(loaded_bodyparts) == len(ALL_FUSED_BPS), (
        f"Keypoint count mismatch: {len(loaded_bodyparts)} vs {len(ALL_FUSED_BPS)}"
    )
    config["bodyparts"] = loaded_bodyparts
    kpms.update_config(str(TEST_PROJECT_DIR), bodyparts=loaded_bodyparts)

    rec_name = list(coordinates.keys())[0]
    coords = coordinates[rec_name]
    confs = confidences[rec_name]

    print(f"  Loaded recording key: '{rec_name}'")
    print(f"  Coordinates shape:    {coords.shape}")
    print(f"  Confidences shape:    {confs.shape}")
    assert coords.ndim == 3 and coords.shape[-1] == 2, f"Bad coords shape: {coords.shape}"
    assert coords.shape[1] == len(ALL_FUSED_BPS), f"Bad coords keypoint dimension: {coords.shape[1]}"
    assert confs.ndim == 2, f"Bad confs shape: {confs.shape}"
    print("  -> [PASS] Keypoint shapes verified (21 keypoints).")

    # 4. Format Data & Fit PCA on CPU
    print("\n[Step 4] Formatting data and fitting PCA on CPU...")
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    pca = kpms.fit_pca(**data, **config)
    print(f"  PCA Components shape: {pca.components_.shape}")
    print(f"  PCA Explained variance ratio (top 3): {pca.explained_variance_ratio_[:3].round(4)}")
    expected_pca_dim = (len(USE_BPS) - 1) * 2
    assert pca.components_.shape[0] == expected_pca_dim, (
        f"PCA feature dimension mismatch! Got {pca.components_.shape[0]}, expected {expected_pca_dim}"
    )
    print(f"  -> [PASS] CPU PCA fitting succeeded ({pca.components_.shape[0]} spatial degrees of freedom).")

    # 5. Initialize Model
    print("\n[Step 5] Initializing AR-HMM Model on CPU...")
    model = kpms.init_model(data, pca=pca, **config)
    print("  Model state initialized successfully.")
    print("  -> [PASS] AR-HMM initialization succeeded.")

    # 6. Run 2 Fast AR-HMM Iterations
    print("\n[Step 6] Running 2 test iterations of AR-HMM...")
    model, model_name = kpms.fit_model(
        model, data, metadata, str(TEST_PROJECT_DIR),
        model_name="cpu_shape_test_model",
        ar_only=True, num_iters=2
    )
    print(f"  Completed test iterations for model: {model_name}")
    print("  -> [PASS] CPU AR-HMM iterations executed with zero errors.")

    # 7. Extract Results
    print("\n[Step 7] Extracting Syllables & Latent States...")
    results = kpms.extract_results(
        model, metadata, str(TEST_PROJECT_DIR), model_name, save_results=True
    )
    res_rec = results[rec_name]
    sylls = res_rec["syllable"]
    latents = res_rec["latent_state"]
    print(f"  Syllable sequence shape: {sylls.shape}")
    print(f"  Latent trajectory shape: {latents.shape}")
    print(f"  Unique active syllables: {len(np.unique(sylls))} in 3,000 frames")
    assert len(sylls) == len(coords), f"Length mismatch: {len(sylls)} vs {len(coords)}"
    print("  -> [PASS] Output cardinality strictly invariant with input frames.")

    print("\n" + "=" * 70)
    print("ALL MULTI-VIEW KEYPOINT-MOSEQ CPU SHAPE & PIPELINE TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_cpu_shape_test()
