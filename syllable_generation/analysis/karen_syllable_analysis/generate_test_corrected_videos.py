#!/usr/bin/env python3
"""
generate_test_corrected_videos.py

Generates test grid movies for Syllable 6, 72, and 0 with:
1. ANATOMICAL FORWARD ORIENTATION:
   anterior_idxs=[0] (nose) and posterior_idxs=[9] (tail_base),
   flipping heading by 180 degrees so the animal faces forward!
2. BALANCED CENTROID:
   Centered on the midpoint between neck and hips (true geometric torso center),
   preventing the posterior spine cluster from dragging the centroid to the rump.
"""

import sys
import pickle
from pathlib import Path
import numpy as np
np.bool8 = np.bool_
import cv2
from tqdm import tqdm
import jax.numpy as jnp

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from vidio.read import OpenCVReader
import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch
from keypoint_moseq.viz import write_video_clip, get_syllable_instances

# Fast-seek reader
_original_read = OpenCVReader.read
def _fast_read(self, framenum):
    if isinstance(framenum, slice):
        return _original_read(self, framenum)
    gap = framenum - self.fnum
    if 0 < gap <= 30:
        for _ in range(gap):
            ret, _ = self.file_object.read()
            if not ret:
                break
        self.fnum = framenum
    return _original_read(self, framenum)
OpenCVReader.read = _fast_read

PROJECT_DIR = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "karen_project_2"
MODEL_NAME = "final_karen_model_k240k"
DATA_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls_k240k.pkl"
OUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "karen_syllable_analysis" / "output" / "test_corrected_orientation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows, cols = 4, 6
num_samples = rows * cols
fps = 40
pre, post = 20, 40
window_size = 144
dot_color = (255, 255, 255)
dot_radius = 4

print("=" * 80)
print("🎬 GENERATING TEST VIDEOS WITH FORWARD HEADING & BALANCED CENTROID")
print(f"   Output: {OUT_DIR}")
print("=" * 80)

model, data, metadata, _ = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)
config = kpms.load_config(str(PROJECT_DIR))
results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)

bodyparts = config.get("use_bodyparts", config.get("bodyparts"))
config["bodyparts"] = bodyparts

Y_est = estimate_coordinates(
    jnp.array(model["states"]["x"]),
    jnp.array(model["states"]["v"]),
    jnp.array(model["states"]["h"]),
    jnp.array(model["params"]["Cd"])
)
coordinates = unbatch(Y_est, *metadata)

# Calculate CORRECTED forward heading and torso-balanced centroid
nose_idx = bodyparts.index("nose")
tail_idx = bodyparts.index("tail_base")
neck_idx = bodyparts.index("neck")
left_hip_idx = bodyparts.index("left_hip")
right_hip_idx = bodyparts.index("right_hip")

cfg_corr = config.copy()
cfg_corr["anterior_idxs"] = [nose_idx]
cfg_corr["posterior_idxs"] = [tail_idx]

_, headings_corr = kpms.get_centroids_headings(coordinates, **cfg_corr)

# Balanced torso centroid: midpoint between neck and hips
centroids_corr = {}
for k, coords in coordinates.items():
    torso_pts = coords[:, [neck_idx, left_hip_idx, right_hip_idx]]
    centroids_corr[k] = torso_pts.mean(axis=1)

# VFI and Video paths
vfi_path = PROJECT_DIR / MODEL_NAME / "video_frame_indexes.npz"
with np.load(vfi_path) as z:
    video_frame_indexes = {k: z[k] for k in z.files}

with open(DATA_PKL, "rb") as f:
    sessions = pickle.load(f)

stem_to_path = {}
for s in sessions:
    if hasattr(s, "video_refs_df") and s.video_refs_df is not None and len(s.video_refs_df) > 0:
        for _, row in s.video_refs_df.iterrows():
            p = row["path"]
            stem_to_path[Path(p).stem] = p
            stem_to_path[s.session_id] = p

video_paths = {}
for k in results.keys():
    base_k = k.split("_part")[0] if "_part" in k else k
    if k in stem_to_path:
        video_paths[k] = stem_to_path[k]
    elif base_k in stem_to_path:
        video_paths[k] = stem_to_path[base_k]
    else:
        for stem, path in stem_to_path.items():
            if stem in k or k in stem:
                video_paths[k] = path
                break

syllables_dict = {k: v["syllable"] for k, v in results.items() if "syllable" in v}
all_instances = get_syllable_instances(
    syllables_dict, pre=pre, post=post, min_duration=0, min_frequency=0, min_instances=1
)

TEST_SYLLABLES = [0, 6, 72]

for syl in TEST_SYLLABLES:
    insts = all_instances.get(syl, [])
    if len(insts) < num_samples:
        insts = insts * (num_samples // len(insts) + 1)
    
    np.random.seed(42)
    sub_idx = np.random.choice(len(insts), num_samples, replace=False)
    sampled = [insts[i] for i in sub_idx]
    
    tiles_clean = []
    for key, start, end in sampled:
        cs = centroids_corr[key][start - pre : start + post]
        h, c = headings_corr[key][start], cs[pre]
        r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])

        frame_ixs = video_frame_indexes[key][start - pre : start + post]
        reader = OpenCVReader(video_paths[key])
        frames = [reader[ix] for ix in frame_ixs]
        reader.close()

        c_rot = r @ c - window_size // 2
        M = np.float32([[np.cos(h), np.sin(h), -c_rot[0]], [-np.sin(h), np.cos(h), -c_rot[1]]])

        t_clean = []
        for ii, (raw_f, c_pt) in enumerate(zip(frames, cs)):
            f_clean = cv2.warpAffine(raw_f.copy(), M, (window_size, window_size))
            if 0 <= ii - pre <= end - start and dot_radius > 0:
                pos = tuple([int(x) for x in M @ np.append(c_pt, 1)])
                cv2.circle(f_clean, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
            t_clean.append(f_clean)

        tiles_clean.append(np.stack(t_clean))

    tc = np.stack(tiles_clean).reshape(rows, cols, post + pre, window_size, window_size, 3)
    clean_frames = np.concatenate(np.concatenate(tc, axis=2), axis=2)
    
    out_path = OUT_DIR / f"syllable{syl}_corrected.mp4"
    write_video_clip(clean_frames, str(out_path), fps=fps)
    print(f"  Generated corrected video: {out_path.name}")

print("\n" + "=" * 80)
print("✅ Test corrected videos complete!")
print(f"   Directory: {OUT_DIR}")
print("=" * 80)
