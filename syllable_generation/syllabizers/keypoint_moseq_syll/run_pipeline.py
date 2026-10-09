import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import sys 
import multiprocessing
if multiprocessing.current_process().name != 'MainProcess':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    logging.getLogger("jax").setLevel(logging.CRITICAL)

# Add repo root to path for unpickling artifacts that reference 'syllable_generation'
sys.path.insert(0, "/home/fsp585/Desktop/FiberphotometryCode")
# Add current directory to path to import kpms_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
np.bool8 = np.bool_

import json
from pathlib import Path
from kpms_pipeline import run_pipeline
import pandas as pd

# ========= constants (edit these) =========
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "karen_project_2")
DETECTOR = "deeplabcut"
CAM = "center_cam"
SUBROUTINE = "jonathan"
LATENT_DIM = 10
NUM_AR_ITERS = 50
TOTAL_ITERS = 125                    
VIDEO_EXTENSION_OVERRIDE = None
PCA_MODE = "load_or_fit"

# --- KAPPA & MIXING ---
INIT_KAPPA = int(os.environ.get("KPMS_INIT_KAPPA", 130000))  # Default yields kappa_slds = 13,000
DECREASE_KAPPA_FACTOR = int(os.environ.get("KPMS_DECREASE_KAPPA_FACTOR", 10))
MIXED_MAP_ITERS = 64

# --- DOWNSAMPLING ---
DOWNSAMPLE = 2
NEW_FPS = 40

# --- TOGGLES ---
USE_BATCHED_INIT = True    # For Initialization (Large datasets)
USE_MEMORY_PATCH = True    # For SLDS Fitting (Prevents OOM)

SESSIONS_PKL = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/sessions_karen_project_dirty.pkl"
MIN_RAW_FRAMES = 50_000  # Filter sessions with < 50k raw frames at 80 FPS (~10 min)


# ========= sessions provider =========
def get_sessions():
    import h5py
    sessions = pd.read_pickle(SESSIONS_PKL)
    if not sessions:
        raise ValueError(f"No sessions in {SESSIONS_PKL}")

    # Filter out sessions with too few raw frames
    filtered = []
    skipped = 0
    for s in sessions:
        h5_path = None
        if hasattr(s, 'tracking_refs_df') and s.tracking_refs_df is not None and not s.tracking_refs_df.empty:
            h5_path = s.tracking_refs_df['path'].iloc[0]
        if not h5_path or not os.path.exists(h5_path):
            skipped += 1
            continue
        try:
            with h5py.File(h5_path, 'r') as f:
                n = f['df_with_missing/table'].shape[0]
            if n >= MIN_RAW_FRAMES:
                filtered.append(s)
            else:
                print(f"[filter] Skipping {s.session_id}: only {n} raw frames (< {MIN_RAW_FRAMES})")
                skipped += 1
        except Exception:
            skipped += 1
    print(f"[filter] Kept {len(filtered)} sessions, skipped {skipped} (missing/short).")
    return filtered

# ========= tiny io helper =========
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32, np.float16)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)

def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NpEncoder)

# =====================================
if __name__ == "__main__":
    # ========= PARSE ARGUMENTS =========
    if len(sys.argv) > 1:
        MODEL_NAME = sys.argv[1]
    else:
        MODEL_NAME = "final-model-run1"

    if len(sys.argv) > 2:
        SEED = int(sys.argv[2])
    else:
        SEED = 42 

    if len(sys.argv) > 3:
        FLIPPING_MODE = sys.argv[3]
        if FLIPPING_MODE == "None": 
            FLIPPING_MODE = None
    else:
        FLIPPING_MODE = None

    if len(sys.argv) > 4:
        SUBSET_METHOD = sys.argv[4]
    else:
        SUBSET_METHOD = "spread"

    if len(sys.argv) > 5:
        PROJECT_DIR_OVERRIDE = sys.argv[5]
    else:
        PROJECT_DIR_OVERRIDE = PROJECT_DIR

    if len(sys.argv) > 6:
        PCA_MODE_OVERRIDE = sys.argv[6]
    else:
        PCA_MODE_OVERRIDE = PCA_MODE

    print(f"--- STARTING PIPELINE ---")
    print(f"Name: {MODEL_NAME}")
    print(f"Seed: {SEED}")
    print(f"Downsample: x{DOWNSAMPLE} (FPS -> {NEW_FPS})")
    print(f"Batched Init: {USE_BATCHED_INIT}")
    print(f"Memory Patch: {USE_MEMORY_PATCH}")
    
    sessions = get_sessions()
    
    # Auto-test mode: if model name starts with "test", only use 2 sessions
    if MODEL_NAME.startswith("test"):
        print("[TEST MODE] Detected 'test' prefix. Slicing to first 2 sessions.")
        sessions = sessions[:2]

    # Default to 50% if activity based, otherwise 100% (no subsetting)
    if SUBSET_METHOD == "activity":
        SUBSET_FRACTION = 0.5
        print(f"[config] Activity subsetting enabled. Keeping top {SUBSET_FRACTION*100}% of data.")
    else:
        SUBSET_FRACTION = 1.0

    pipeline_kwargs = {
        "project_dir": PROJECT_DIR_OVERRIDE,
        "detector": DETECTOR,
        "cam": CAM,
        "subroutine": SUBROUTINE,
        "latent_dim": LATENT_DIM,
        "num_ar_iters": NUM_AR_ITERS,
        "total_iters": TOTAL_ITERS,
        "override_video_extension": VIDEO_EXTENSION_OVERRIDE,
        "pca_mode": PCA_MODE_OVERRIDE,
        "model_name": MODEL_NAME,
        "init_kappa": INIT_KAPPA,
        "decrease_kappa_factor": DECREASE_KAPPA_FACTOR,
        "mixed_map_iters": MIXED_MAP_ITERS,
        "seed": SEED,
        "downsample_rate": DOWNSAMPLE,
        "new_fps": NEW_FPS,
        "use_batched_init": USE_BATCHED_INIT,
        "use_memory_patch": USE_MEMORY_PATCH,
        "flipping_mode": FLIPPING_MODE,
        "subset_method": SUBSET_METHOD,
        "subset_fraction": SUBSET_FRACTION
    }

    print(f"Executing pipeline with kwargs: {json.dumps({k:v for k,v in pipeline_kwargs.items() if v is not None}, indent=2)}")

    # Run pipeline
    results = run_pipeline(sessions, **pipeline_kwargs)

    # Combine configuration with results
    summary = {**pipeline_kwargs, **results}

    model_dir = os.path.join(PROJECT_DIR_OVERRIDE, summary["model_name"])
    model_summary_path = os.path.join(model_dir, "model_summary.json")
    _write_json(model_summary_path, summary)

    print("\n=== Run summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved model summary to: {model_summary_path}")