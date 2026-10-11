#!/usr/bin/env python3
"""
find_similar_syllables.py

High-speed Syllable Similarity Finder & Video Generator for Nominal Starred Hits.

1. Precomputes filtered centroids and headings once.
2. Extracts canonical egocentric trajectories across all syllables.
3. Computes exact pairwise cosine and Euclidean trajectory similarity matrices.
4. For each nominal starred syllable (6, 93, 36, 41), identifies the Top 2 most similar syllables.
5. Automatically renders single-pass dual grid movies (with/without trajectories) for any missing ones.
"""

import os
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pickle

import cv2
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
from collections import Counter

import jax.numpy as jnp
import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch
from keypoint_moseq.viz import (
    filter_centroids_headings,
    get_edges,
    get_syllable_instances,
    overlay_keypoints_on_image,
    write_video_clip,
)
from scipy.spatial.distance import pdist, squareform

# Fast-seek monkeypatch
from vidio.read import OpenCVReader

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
OUT_BASE = REPO_ROOT / "syllable_generation" / "analysis" / "karen_syllable_analysis" / "output"
OUT_WITH_TRAJ = OUT_BASE / "grid_movies_with_trajectories"
OUT_WITHOUT_TRAJ = OUT_BASE / "grid_movies_without_trajectories"

print(f"[INFO] Loading model {MODEL_NAME}...", flush=True)
model, data, metadata, current_iter = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)
config = kpms.load_config(str(PROJECT_DIR))
bodyparts = config.get("use_bodyparts", config.get("bodyparts"))
config["bodyparts"] = bodyparts

results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)

print("[INFO] Estimating coordinates, centroids, and headings...", flush=True)
Y_est = estimate_coordinates(
    jnp.array(model["states"]["x"]),
    jnp.array(model["states"]["v"]),
    jnp.array(model["states"]["h"]),
    jnp.array(model["params"]["Cd"])
)
coordinates = unbatch(Y_est, *metadata)
raw_centroids, raw_headings = kpms.get_centroids_headings(coordinates, **config)

print("[INFO] Precomputing filtered centroids and headings...", flush=True)
centroids, headings = filter_centroids_headings(raw_centroids, raw_headings, filter_size=config.get("filter_size", 9))

# ==============================================================================
# FAST EGOCENTRIC TRAJECTORY SIMILARITY
# ==============================================================================
fps = 40
pre = int(0.125 * fps)  # 5 frames
post = int(0.375 * fps) # 15 frames
total_len = pre + post

syllables_dict = {k: v["syllable"] for k, v in results.items() if "syllable" in v}
all_instances = get_syllable_instances(syllables_dict, pre=pre, post=post, min_duration=1, min_frequency=0.0001, min_instances=1)

print(f"[INFO] Extracting canonical egocentric trajectories across {len(all_instances)} syllables...", flush=True)

mean_trajectories = {}
for syl, insts in all_instances.items():
    if len(insts) == 0:
        continue
    # Sample up to 100 instances for robust mean
    np.random.seed(0)
    sample_n = min(len(insts), 100)
    idx = np.random.choice(len(insts), sample_n, replace=False)

    syl_trajs = []
    for i in idx:
        key, start, end = insts[i]
        cs = centroids[key][start - pre : start + post]
        h, c = headings[key][start], cs[pre]
        r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])

        # Egocentric keypoints aligned to heading at start and centered at centroid
        coords = coordinates[key][start - pre : start + post] # (T, K, 2)
        ego_coords = (coords - c) @ r.T # (T, K, 2)
        syl_trajs.append(ego_coords)

    mean_trajectories[syl] = np.mean(syl_trajs, axis=0) # (T, K, 2)

syllable_ixs = sorted(mean_trajectories.keys())
Xs = np.stack([mean_trajectories[s] for s in syllable_ixs])
flat_Xs = Xs.reshape(Xs.shape[0], -1)

# Pairwise cosine and euclidean similarity
cosine_dists = squareform(pdist(flat_Xs, metric="cosine"))
euclid_dists = squareform(pdist(flat_Xs, metric="euclidean"))
syll_to_idx = {s: i for i, s in enumerate(syllable_ixs)}

# Compute syllable metrics for summary
bout_counts = Counter()
frame_counts = Counter()
for k, v in results.items():
    if "syllable" in v:
        sylls = np.array(v["syllable"])
        diffs = np.diff(sylls, prepend=-999)
        bout_mask = (diffs != 0) & (sylls >= 0)
        for syl in sylls[bout_mask]:
            bout_counts[int(syl)] += 1
        for syl in sylls:
            if syl >= 0:
                frame_counts[int(syl)] += 1

STARRED_SYLLS = [6, 93, 36, 41]

print("\n" + "="*80, flush=True)
print("🔍 TOP 2 MOST SIMILAR SYLLABLES FOR NOMINAL STARRED HITS", flush=True)
print("="*80, flush=True)

missing_to_render = set()
similarity_results = {}

for star_syl in STARRED_SYLLS:
    if star_syl not in syll_to_idx:
        continue
    idx = syll_to_idx[star_syl]
    dists = cosine_dists[idx].copy()
    dists[idx] = np.inf # Exclude self
    sorted_order = np.argsort(dists)

    top2_idx = sorted_order[:2]
    top2_sylls = [syllable_ixs[i] for i in top2_idx]
    top2_dists = [dists[i] for i in top2_idx]
    top2_sims = [1.0 - d for d in top2_dists]

    similarity_results[star_syl] = [
        {"syllable": top2_sylls[0], "similarity": top2_sims[0], "distance": top2_dists[0], "bouts": bout_counts[top2_sylls[0]]},
        {"syllable": top2_sylls[1], "similarity": top2_sims[1], "distance": top2_dists[1], "bouts": bout_counts[top2_sylls[1]]}
    ]

    print(f"\n🌟 Target Syllable {star_syl} (Bouts: {bout_counts[star_syl]}):", flush=True)
    for rank, item in enumerate(similarity_results[star_syl], 1):
        s = item['syllable']
        sim = item['similarity']
        b = item['bouts']
        print(f"   Rank #{rank} Similar: Syllable {s:2d} | Cosine Sim: {sim:.4f} | Total Bouts: {b:6d}", flush=True)
        missing_to_render.add(s)

# Check which ones are already rendered
existing_traj = {int(p.stem.replace("syllable", "")) for p in OUT_WITH_TRAJ.glob("*.mp4")}
existing_clean = {int(p.stem.replace("syllable", "")) for p in OUT_WITHOUT_TRAJ.glob("*.mp4")}
already_rendered = existing_traj.intersection(existing_clean)

needed_sylls = [s for s in missing_to_render if s not in already_rendered]
print(f"\n[INFO] Already rendered ({len(already_rendered)}): {sorted(list(already_rendered))}", flush=True)
print(f"[INFO] Missing similar syllables needing rendering: {needed_sylls}", flush=True)

# Render missing videos
if needed_sylls:
    print(f"\n[INFO] Rendering {len(needed_sylls)} missing similar syllables in single pass...", flush=True)
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

    rows, cols = 4, 6
    num_samples = rows * cols
    fps = 40
    pre_sec, post_sec = 0.5, 1.0
    pre_mov = int(pre_sec * fps)
    post_mov = int(post_sec * fps)
    edges = get_edges(bodyparts, config.get("skeleton", []))
    window_size = 144

    movie_instances = get_syllable_instances(syllables_dict, pre=pre_mov, post=post_mov, min_duration=0, min_frequency=0, min_instances=1)

    sampled_instances = {}
    for syl in needed_sylls:
        insts = movie_instances.get(syl, [])
        if len(insts) == 0:
            continue
        if len(insts) < num_samples:
            insts = insts + [insts[0]] * (num_samples - len(insts))
        np.random.seed(0)
        idx = np.random.choice(len(insts), num_samples, replace=False)
        sampled_instances[syl] = [insts[i] for i in idx]

    for syl in needed_sylls:
        if syl not in sampled_instances:
            continue
        insts = sampled_instances[syl]

        dot_color = (255, 255, 255)
        dot_radius = 4
        scaled_window_size = window_size
        tiles_clean = []
        tiles_overlay = []

        for key, start, end in insts:
            cs = centroids[key][start - pre_mov : start + post_mov]
            h, c = headings[key][start], cs[pre_mov]
            r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
            syllable_coordinates = coordinates[key][start - pre_mov : start + post_mov].copy()

            frame_ixs = video_frame_indexes[key][start - pre_mov : start + post_mov]
            reader = OpenCVReader(video_paths[key])
            frames = [reader[ix] for ix in frame_ixs]
            reader.close()

            c_rot = r @ c - window_size // 2
            M = np.float32([[np.cos(h), np.sin(h), -c_rot[0]], [-np.sin(h), np.cos(h), -c_rot[1]]])

            t_clean = []
            t_overlay = []

            for ii, (raw_f, c_pt) in enumerate(zip(frames, cs)):
                f_clean = cv2.warpAffine(raw_f.copy(), M, (window_size, window_size))
                if 0 <= ii - pre_mov <= end - start and dot_radius > 0:
                    pos = tuple([int(x) for x in M @ np.append(c_pt, 1)])
                    cv2.circle(f_clean, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
                t_clean.append(f_clean)

                coords = syllable_coordinates[ii]
                f_ov = overlay_keypoints_on_image(raw_f.copy(), coords, edges=edges)
                f_ov = cv2.warpAffine(f_ov, M, (window_size, window_size))
                if 0 <= ii - pre_mov <= end - start and dot_radius > 0:
                    pos = tuple([int(x) for x in M @ np.append(c_pt, 1)])
                    cv2.circle(f_ov, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
                t_overlay.append(f_ov)

            tiles_clean.append(np.stack(t_clean))
            tiles_overlay.append(np.stack(t_overlay))

        tc = np.stack(tiles_clean).reshape(rows, cols, post_mov + pre_mov, scaled_window_size, scaled_window_size, 3)
        frames_clean = np.concatenate(np.concatenate(tc, axis=2), axis=2)

        to = np.stack(tiles_overlay).reshape(rows, cols, post_mov + pre_mov, scaled_window_size, scaled_window_size, 3)
        frames_overlay = np.concatenate(np.concatenate(to, axis=2), axis=2)

        clean_out = OUT_WITHOUT_TRAJ / f"syllable{syl}.mp4"
        traj_out = OUT_WITH_TRAJ / f"syllable{syl}.mp4"

        write_video_clip(frames_clean, str(clean_out), fps=fps)
        write_video_clip(frames_overlay, str(traj_out), fps=fps)
        print(f" Saved dual videos for Syllable {syl}: {clean_out.name} & {traj_out.name}", flush=True)

print("\n" + "="*80, flush=True)
print(" ALL SIMILAR SYLLABLES VERIFIED AND RENDERED!", flush=True)
print("="*80, flush=True)
