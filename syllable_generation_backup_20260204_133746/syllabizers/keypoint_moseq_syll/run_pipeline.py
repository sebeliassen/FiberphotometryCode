import os
import sys 
import json
from pathlib import Path
from kpms_pipeline import run_pipeline
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3])) 

# ========= constants (edit these) =========
PROJECT_DIR = "amph_project"        
DETECTOR = "deeplabcut"
CAM = "center_cam"
SUBROUTINE = "jonathan"
LATENT_DIM = 10
NUM_AR_ITERS = 50
TOTAL_ITERS = 200                    
VIDEO_EXTENSION_OVERRIDE = None
PCA_MODE = "load"

# --- KAPPA & MIXING ---
INIT_KAPPA = 120000                  
DECREASE_KAPPA_FACTOR = 10
MIXED_MAP_ITERS = 64

# --- DOWNSAMPLING ---
DOWNSAMPLE = 2
NEW_FPS = 40

# --- TOGGLES ---
USE_BATCHED_INIT = True    # For Initialization (Large datasets)
USE_MEMORY_PATCH = True    # For SLDS Fitting (Prevents OOM)

SESSIONS_PKL = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions.pkl"

# ========= PARSE ARGUMENTS =========
if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]
else:
    MODEL_NAME = "final-model-run1"

if len(sys.argv) > 2:
    SEED = int(sys.argv[2])
else:
    SEED = 42 

# ========= sessions provider =========
def get_sessions():
    sessions = pd.read_pickle(SESSIONS_PKL)
    if not sessions:
        raise ValueError(f"No sessions in {SESSIONS_PKL}")
    return sessions

# ========= tiny io helper =========
def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# =====================================
if __name__ == "__main__":
    print(f"--- STARTING PIPELINE ---")
    print(f"Name: {MODEL_NAME}")
    print(f"Seed: {SEED}")
    print(f"Downsample: x{DOWNSAMPLE} (FPS -> {NEW_FPS})")
    print(f"Batched Init: {USE_BATCHED_INIT}")
    print(f"Memory Patch: {USE_MEMORY_PATCH}")

    sessions = get_sessions()

    pipeline_kwargs = {
        "project_dir": PROJECT_DIR,
        "detector": DETECTOR,
        "cam": CAM,
        "subroutine": SUBROUTINE,
        "latent_dim": LATENT_DIM,
        "num_ar_iters": NUM_AR_ITERS,
        "total_iters": TOTAL_ITERS,
        "override_video_extension": VIDEO_EXTENSION_OVERRIDE,
        "pca_mode": PCA_MODE,
        "model_name": MODEL_NAME,
        "init_kappa": INIT_KAPPA,
        "decrease_kappa_factor": DECREASE_KAPPA_FACTOR,
        "mixed_map_iters": MIXED_MAP_ITERS,
        "seed": SEED,
        "downsample_rate": DOWNSAMPLE,
        "new_fps": NEW_FPS,
        "use_batched_init": USE_BATCHED_INIT,
        "use_memory_patch": USE_MEMORY_PATCH
    }

    print(f"Executing pipeline with kwargs: {json.dumps({k:v for k,v in pipeline_kwargs.items() if v is not None}, indent=2)}")
    
    # Run pipeline
    results = run_pipeline(sessions, **pipeline_kwargs)

    # Combine configuration with results
    summary = {**pipeline_kwargs, **results}

    model_dir = os.path.join(PROJECT_DIR, summary["model_name"])
    model_summary_path = os.path.join(model_dir, "model_summary.json")
    _write_json(model_summary_path, summary)

    print("\n=== Run summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved model summary to: {model_summary_path}")