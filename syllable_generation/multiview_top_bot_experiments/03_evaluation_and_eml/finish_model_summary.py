import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import json
from pathlib import Path

import keypoint_moseq as kpms
import numpy as np
from keypoint_moseq.viz import get_durations

PROJECT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/multiview_kappa_search/project")
MODEL_NAME = "fullmodel_multiview_23kp_k6e12"
TARGET_FPS = 60.0

print(f"Loading checkpoint for {MODEL_NAME}...")
model, data, metadata, current_iter = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)
print(f"Loaded successfully at iteration {current_iter}!")

# Extract results and save results.h5
print("Extracting results and saving results.h5...")
results = kpms.extract_results(model, metadata, str(PROJECT_DIR), MODEL_NAME, save_results=True)

# Calculate final stats
z_final = np.array(model["states"]["z"])
mask_final = np.array(data["mask"]).astype(bool)
valid_mask = mask_final[:, :z_final.shape[1]]

durations = get_durations(z_final, mask_final)
final_median_frames = float(np.median(durations))
final_median_ms = (final_median_frames / TARGET_FPS) * 1000.0

valid_z = z_final[valid_mask]
unique_sylls, syll_counts = np.unique(valid_z, return_counts=True)
num_active_sylls = len(unique_sylls)
total_frames = int(valid_mask.sum())

# Sort syllables by frequency
sort_idx = np.argsort(-syll_counts)
top_sylls = [
    {"syllable": int(unique_sylls[i]), "count": int(syll_counts[i]), "usage_pct": float(syll_counts[i] / total_frames * 100)}
    for i in sort_idx[:15]
]

summary_data = {
    "model_name": MODEL_NAME,
    "total_iters": int(current_iter),
    "num_ar_iters": 50,
    "num_slds_iters": 150,
    "init_kappa": 6.0e12,
    "decrease_kappa_factor": 10.0,
    "slds_kappa": 6.0e11,
    "target_fps": TARGET_FPS,
    "total_frames_modeled": total_frames,
    "final_median_frames": final_median_frames,
    "final_median_ms": final_median_ms,
    "num_active_syllables": int(num_active_sylls),
    "top_15_syllables": top_sylls,
}

model_save_dir = PROJECT_DIR / MODEL_NAME
summary_path = model_save_dir / "model_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary_data, f, indent=2)

print("\n" + "=" * 65)
print("FULL MODEL EVALUATION SUCCESS:")
print(f"  Model: {MODEL_NAME}")
print(f"  Total Iterations: {current_iter}")
print(f"  Median Dwell: {final_median_frames:.2f} frames ({final_median_ms:.1f} ms @ 60 FPS)")
print(f"  Active Syllables: {num_active_sylls}")
print(f"  Total Valid Frames Modeled: {total_frames:,}")
print("  Top 5 Syllables by Usage:")
for s in top_sylls[:5]:
    print(f"    Syllable {s['syllable']}: {s['usage_pct']:.2f}% ({s['count']:,} frames)")
print("=" * 65)
