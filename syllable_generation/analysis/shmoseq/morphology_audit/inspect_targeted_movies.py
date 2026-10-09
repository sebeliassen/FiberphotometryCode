import os
import cv2
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode")
MOVIE_DIR = BASE_DIR / "syllable_generation" / "analysis" / "shmoseq" / "morphology_audit" / "grid_movies_targeted_8kp"
ARTIFACT_DIR = Path("/home/fsp585/.gemini/antigravity-ide/brain/e3532183-ee20-47bf-86a2-d66c7d4e288d")
SNAP_DIR = ARTIFACT_DIR / "movie_snapshots"
SNAP_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SYLLS = [0, 1, 6, 10, 33, 41, 43, 49]

# 1. Capture middle-frame snapshots of each grid movie
for s in TARGET_SYLLS:
    vid_path = MOVIE_DIR / f"syllable{s}.mp4"
    if not vid_path.exists():
        continue
    cap = cv2.VideoCapture(str(vid_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
    ret, frame = cap.read()
    cap.release()
    if ret:
        out_png = SNAP_DIR / f"syllable_{s:02d}_mid.png"
        cv2.imwrite(str(out_png), frame)

print(f"Saved snapshots to {SNAP_DIR}")

# 2. Extract kinematics for these 8 syllables from results.h5
RES_PATH = BASE_DIR / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "amph_project_8kp" / "amph_8kp_k150k_run_03" / "results.h5"

kin_records = {s: {"vel": [], "head_vel": [], "count": 0} for s in TARGET_SYLLS}

with h5py.File(RES_PATH, "r") as f:
    for k in list(f.keys())[:15]: # Sample 15 sessions
        sylls = f[k]["syllable"][:]
        cents = f[k]["centroid"][:]
        heads = f[k]["heading"][:]
        
        vel = np.linalg.norm(np.diff(cents, axis=0), axis=1) * 40.0 # px/s
        h_diff = np.abs(np.diff(np.unwrap(heads))) * 40.0 # rad/s
        
        for s in TARGET_SYLLS:
            mask = (sylls[:-1] == s)
            if np.any(mask):
                kin_records[s]["vel"].extend(vel[mask].tolist())
                kin_records[s]["head_vel"].extend(h_diff[mask].tolist())
                kin_records[s]["count"] += int(mask.sum())

rows = []
for s in TARGET_SYLLS:
    v = np.array(kin_records[s]["vel"])
    hv = np.array(kin_records[s]["head_vel"])
    rows.append({
        "syllable": s,
        "sample_frames": kin_records[s]["count"],
        "mean_speed_px_s": float(np.mean(v)) if len(v) else 0.0,
        "median_speed_px_s": float(np.median(v)) if len(v) else 0.0,
        "mean_turn_rad_s": float(np.mean(hv)) if len(hv) else 0.0,
    })

df_kin = pd.DataFrame(rows)
df_kin.to_csv(SNAP_DIR / "target_syllables_kinematics.csv", index=False)
print("\nKinematics Summary:")
print(df_kin.to_string(index=False))
