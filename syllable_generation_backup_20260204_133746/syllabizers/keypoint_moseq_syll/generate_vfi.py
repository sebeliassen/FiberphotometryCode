import sys
from pathlib import Path
import numpy as np
import pandas as pd
import keypoint_moseq as kpms

# 1. SETUP PATHS (Match your environment)
# ---------------------------------------
# Add repo root to path so we can import 'utils'
sys.path.insert(0, str(Path(__file__).resolve().parents[3])) 
from utils.paths import ref_pairs_from_sessions
from utils.trimming import presence_mask_from_conf, filter_with_masks, clamp_conf_nonneg

# 2. CONFIGURATION (Must match your training parameters)
# ---------------------------------------
SESSIONS_PKL = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions.pkl"
PROJECT_DIR = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project" # Ensure this points to the actual folder
MODEL_NAME = "amph_vfi_test"          # The model folder name you intend to use
DETECTOR = "deeplabcut"
CAM = "center_cam"
SUBROUTINE = "jonathan"

# Downsampling params
DOWNSAMPLE_RATE = 1
NEW_FPS = 80

def main():
    print(f"--- Fast-Forward Index Generation for {PROJECT_DIR} ---")

    # 3. LOAD SESSIONS
    # ---------------------------------------
    if not Path(SESSIONS_PKL).exists():
        raise FileNotFoundError(f"Could not find sessions pickle: {SESSIONS_PKL}")
    
    print(f"Loading sessions from: {SESSIONS_PKL}")
    sessions = pd.read_pickle(SESSIONS_PKL)

    # 4. PRE-PROCESSING (Steps 0-3 of pipeline)
    # ---------------------------------------
    
    # A) Get file paths
    keypoint_paths, video_paths = ref_pairs_from_sessions(sessions, subroutine=SUBROUTINE, cam=CAM)
    
    # B) Load Keypoints
    print("Loading keypoints (this may take a moment)...")
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_paths, DETECTOR)

    # C) Downsampling Config Update
    if DOWNSAMPLE_RATE > 1:
        if NEW_FPS is not None:
            kpms.update_config(PROJECT_DIR, fps=NEW_FPS)

    # D) Create Presence Masks
    all_masks = {name: presence_mask_from_conf(conf) for name, conf in confidences.items()}

    # E) Filter (This generates the file we need!)
    print("Filtering and calculating indexes...")
    # Note: We load config here to ensure we use current project settings for thresholds
    cfg = kpms.load_config(PROJECT_DIR)
    
    filtered_coords, filtered_confs, video_frame_indexes = filter_with_masks(
        coordinates, confidences, all_masks, PROJECT_DIR,
        return_video_frame_indexes=True, downsample_rate=DOWNSAMPLE_RATE, **cfg
    )

    # 5. SAVE ARTIFACTS
    # ---------------------------------------
    # Ensure model directory exists
    model_dir = Path(PROJECT_DIR) / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = model_dir / "video_frame_indexes.npz"
    
    print(f"Saving video_frame_indexes.npz to: {output_path}")
    np.savez_compressed(
        output_path,
        **{k: np.asarray(v, dtype=np.int64) for k, v in video_frame_indexes.items()}
    )
    
    print("Success! You can now use the Cleaning GUI.")

if __name__ == "__main__":
    main()