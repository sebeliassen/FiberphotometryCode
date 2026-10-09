#!/usr/bin/env python3
"""
select_cohort_static_21kp_v2_syllable_manifest.py
==================================================
Generates the Unified Syllable Selection Manifest for Cohort Static 21kp V2:
- Grid size: G = 12 (default) or G = 24.
- For each syllable with N qualifying, unique bouts:
  1. Tier 1: N > 50
     - Built-in KPMS density sampling in 4D PCA space.
     - Exemplar maximizing local/global density ratio placed first,
       followed by its G - 1 nearest neighbors in PCA space ordered by distance.
  2. Tier 2: G < N <= 50
     - Compact-neighborhood selection:
       a. For each bout i in {0..N-1}, form candidate set S_i containing i and its G - 1 nearest bouts.
       b. Compute mean pairwise trajectory distance within S_i.
       c. Choose candidate set S* with lowest score.
       d. Find medoid m* in S*, place m* first, order remaining G - 1 examples by distance from m*.
  3. Tier 3: N <= G
     - Include all N bouts.
     - Find medoid of the N bouts, place it first, order remaining N - 1 by distance from medoid.
     - Unused (G - N) slots are marked as blank.
- Representation:
  - Aligned observed 21kp coordinates (17 dorsal + 4 ventral paws) in ego-centric frame
    (centered at onset centroid, rotated by onset heading).
  - Fixed window [onset - pre, onset + post], pre=5, post=15 (20 frames = 333 ms @ 60 FPS).
- Output:
  - Saved to `unified_syllable_selection_manifest.json`
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
import numpy as np
import h5py
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXPERIMENT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from core.pair_matching import find_matched_pairs
from jax_moseq.utils import unbatch

COHORT_DIR = EXPERIMENT_DIR / "cohort_static_21kp_v2"
MODEL_DIR = COHORT_DIR / "model_21kp_static_50ar_300slds_seed101"
RESULTS_PATH = MODEL_DIR / "results.h5"
VFI_PATH = COHORT_DIR / "video_frame_indexes.npz"
PREPROCESSED_PATH = COHORT_DIR / "preprocessed_data.pkl"

TOP_VID_DIR = REPO_ROOT / "karen_project/video_files"
BOT_VID_DIR = REPO_ROOT / "karen_project/botcam_video_files"

CHAMBER_TO_ARENA = {
    "top_left": "Arena1",
    "bottom_left": "Arena2",
    "bottom_right": "Arena3",
    "top_right": "Arena4"
}


def get_session_files(sid, pairs_by_sid):
    """Retrieves paths for TopCam video, BotCam video, TopCam H5, and BotCam H5."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_([a-z]+_[a-z]+)", sid)
    if not m:
        return None
    ts, chamber = m.group(1), m.group(2)
    arena = CHAMBER_TO_ARENA.get(chamber)
    if not arena:
        return None

    top_vids = list(TOP_VID_DIR.glob(f"*{ts}*{chamber}*.avi"))
    bot_vids = list(BOT_VID_DIR.glob(f"*{arena}*BotCamVid*{ts}*.avi"))

    pair = pairs_by_sid.get(f"{ts}_{chamber}")
    if not pair:
        return None

    top_h5 = pair["top_h5"]
    bot_h5 = pair["bot_h5"]

    if top_vids and bot_vids and Path(top_h5).exists() and Path(bot_h5).exists():
        return {
            "top_vid": str(top_vids[0]),
            "bot_vid": str(bot_vids[0]),
            "top_h5": str(top_h5),
            "bot_h5": str(bot_h5),
        }
    return None


def extract_egocentric_trajectory(Y_slice, c0, h0):
    """
    Transforms Y_slice (T, K, 2) to ego-centric coordinates aligned at onset centroid c0 and heading h0.
    Returns flattened vector of shape (T * K * 2,).
    """
    Y_centered = Y_slice - c0
    cos_h = np.cos(h0)
    sin_h = np.sin(h0)
    x_rot = Y_centered[:, :, 0] * cos_h + Y_centered[:, :, 1] * sin_h
    y_rot = -Y_centered[:, :, 0] * sin_h + Y_centered[:, :, 1] * cos_h
    Y_ego = np.stack([x_rot, y_rot], axis=-1)
    return Y_ego.reshape(-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate Syllable Selection Manifest for Cohort Static 21kp V2")
    parser.add_argument("--grid_size", type=int, default=12, choices=[12, 24],
                        help="Number of tiles in the grid movie (default: 12)")
    parser.add_argument("--pre", type=int, default=5, help="Frames before onset for trajectory (default: 5)")
    parser.add_argument("--post", type=int, default=15, help="Frames after onset for trajectory (default: 15)")
    parser.add_argument("--min_frequency", type=float, default=0.0,
                        help="Minimum frequency fraction in cohort frames (default: 0.0 to cover all syllables)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory for manifest")
    args = parser.parse_args()

    G = args.grid_size
    pre, post = args.pre, args.post
    clip_len = pre + post

    out_dir = Path(args.output_dir) if args.output_dir else (
        EXPERIMENT_DIR / "04_reports_and_figures/visualizations_cohort_static_21kp_v2_seed101"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "unified_syllable_selection_manifest.json"

    print("=" * 80)
    print("🎯 GENERATING UNIFIED SYLLABLE SELECTION MANIFEST (V2 COHORT STATIC 21KP)")
    print(f"Grid Size: G = {G}")
    print(f"Trajectory Window: [-{pre}, +{post}] ({clip_len} frames)")
    print(f"Output Manifest:   {manifest_path}")
    print("=" * 80)

    # 1. Load VFI and pair mapping
    print("\n[1/5] Loading Video Frame Indexes (VFI) and matched pairs...", flush=True)
    with np.load(VFI_PATH) as z:
        vfi_dict = {k: z[k] for k in z.files}

    pairs = find_matched_pairs()
    pairs_by_sid = {p["sid"]: p for p in pairs}

    # 2. Load unbatched coordinates Y
    print("[2/5] Loading and unbatching preprocessed 21kp coordinates (8.9 GB)...", flush=True)
    t0 = time.time()
    with open(PREPROCESSED_PATH, "rb") as pf:
        data, metadata = pickle.load(pf)
    coords_dict = unbatch(data["Y"], *metadata)
    del data
    print(f"  Coordinates unbatched for {len(coords_dict)} segments in {time.time() - t0:.1f}s.", flush=True)

    # 3. Index all qualifying bouts and extract ego-centric trajectories
    print("\n[3/5] Indexing qualifying syllable bouts across results.h5...", flush=True)
    bouts_by_syllable = {}
    trajectories_by_syllable = {}
    all_trajectories = []
    total_cohort_frames = 0
    syll_total_frames = {}

    with h5py.File(RESULTS_PATH, "r") as res_file:
        for sid in sorted(res_file.keys()):
            if sid not in coords_dict or sid not in vfi_dict:
                continue
            paths = get_session_files(sid, pairs_by_sid)
            if not paths:
                continue

            grp = res_file[sid]
            sylls = grp["syllable"][:]
            centroids = grp["centroid"][:]
            headings = grp["heading"][:]
            Y_arr = coords_dict[sid]
            vfi = vfi_dict[sid]

            n_frames = len(sylls)
            total_cohort_frames += n_frames

            u, c = np.unique(sylls, return_counts=True)
            for val, cnt in zip(u, c):
                val = int(val)
                syll_total_frames[val] = syll_total_frames.get(val, 0) + cnt

            diffs = np.diff(sylls, prepend=-1, append=-1)
            starts = np.where(diffs[:-1] != 0)[0]
            ends = np.where(diffs[1:] != 0)[0]

            for s_idx, e_idx in zip(starts, ends):
                dur = e_idx - s_idx
                # Valid qualifying bout: dur >= 5, onset has enough room for trajectory window
                if dur >= 5 and s_idx >= pre and (s_idx + post) <= n_frames:
                    s_id = int(sylls[s_idx])
                    if s_id not in bouts_by_syllable:
                        bouts_by_syllable[s_id] = []
                        trajectories_by_syllable[s_id] = []

                    c0 = centroids[s_idx]
                    h0 = headings[s_idx]
                    Y_slice = Y_arr[s_idx - pre : s_idx + post]

                    if np.isnan(c0).any() or np.isnan(h0) or np.isnan(Y_slice).any():
                        continue

                    ego_traj = extract_egocentric_trajectory(Y_slice, c0, h0)

                    src_onset = int(vfi[s_idx]) if s_idx < len(vfi) else s_idx
                    src_offset = int(vfi[e_idx - 1]) + 1 if (e_idx - 1) < len(vfi) else e_idx

                    bouts_by_syllable[s_id].append({
                        "sid": sid,
                        "onset_segment_frame": int(s_idx),
                        "offset_segment_frame": int(e_idx),
                        "duration_frames": int(dur),
                        "source_top_onset": src_onset,
                        "source_top_offset": src_offset,
                        "top_vid": paths["top_vid"],
                        "bot_vid": paths["bot_vid"],
                        "top_h5": paths["top_h5"],
                        "bot_h5": paths["bot_h5"]
                    })
                    trajectories_by_syllable[s_id].append(ego_traj)
                    all_trajectories.append(ego_traj)

    del coords_dict
    print(f"  Indexed {len(all_trajectories)} qualifying bouts across {len(bouts_by_syllable)} syllables.")

    # Target syllables: all qualifying syllables (or >= min_frequency)
    min_freq = args.min_frequency
    target_sylls = [
        s for s, cnt in sorted(syll_total_frames.items(), key=lambda x: x[1], reverse=True)
        if (cnt / total_cohort_frames) >= min_freq and s in bouts_by_syllable and len(bouts_by_syllable[s]) > 0
    ]
    print(f"  Target syllables to process (>= {min_freq*100:.2f}% cohort frames): {len(target_sylls)} syllables.")

    # 4. Global PCA for Built-in KPMS Density Sampling
    print("\n[4/5] Fitting global 4D PCA for Tier 1 Built-in KPMS density sampling...", flush=True)
    all_trajs_arr = np.array(all_trajectories, dtype=np.float32)
    n_sample_pca = min(50000, len(all_trajs_arr))
    np.random.seed(42)
    sample_indices = np.random.choice(len(all_trajs_arr), n_sample_pca, replace=False)
    pca = PCA(n_components=4, random_state=42).fit(all_trajs_arr[sample_indices])
    all_pca = pca.transform(all_trajs_arr[sample_indices])
    all_nbrs = NearestNeighbors(n_neighbors=50).fit(all_pca)
    print("  Global PCA and background KD-Tree fitted successfully.")

    # 5. Execute 3-Tier Instance Selection per Syllable
    print(f"\n[5/5] Executing 3-Tier Prototype Selection for G={G} tiles across {len(target_sylls)} syllables...", flush=True)
    manifest = {
        "model_name": "model_21kp_static_50ar_300slds_seed101",
        "results_path": str(RESULTS_PATH),
        "total_iterations": 350,
        "grid_size": G,
        "trajectory_window": {"pre": pre, "post": post, "clip_len": clip_len},
        "tiers_definition": {
            "tier_1": f"N > 50: Built-in KPMS density sampling (4D PCA, exemplar mode)",
            "tier_2": f"{G} < N <= 50: Compact-neighborhood selection (lowest mean pairwise distance)",
            "tier_3": f"N <= {G}: Complete bout enumeration (medoid first, remaining ordered, unused blank)"
        },
        "syllables": {}
    }

    tier_counts = {"tier_1": 0, "tier_2": 0, "tier_3": 0}

    for s in target_sylls:
        bouts = bouts_by_syllable[s]
        trajs = np.array(trajectories_by_syllable[s], dtype=np.float32)
        N = len(bouts)

        if N > 50:
            # =================================================================
            # TIER 1: Built-in KPMS Density Sampling
            # =================================================================
            tier_counts["tier_1"] += 1
            tier_name = "tier_1_density"
            X_s_pca = pca.transform(trajs)
            nbrs = NearestNeighbors(n_neighbors=50).fit(X_s_pca)
            distances, indices = nbrs.kneighbors(X_s_pca)
            local_density = 1.0 / (distances.mean(axis=1) + 1e-8)

            distances_glob, _ = all_nbrs.kneighbors(X_s_pca)
            global_density = 1.0 / (distances_glob.mean(axis=1) + 1e-8)

            density_ratio = local_density / global_density
            exemplar_idx = int(np.argmax(density_ratio))

            # indices[exemplar_idx] has length 50, starting with exemplar_idx itself
            chosen_local_indices = list(indices[exemplar_idx][:G])
            medoid_local_idx = exemplar_idx

        elif N > G:
            # =================================================================
            # TIER 2: Compact-Neighborhood Selection (G < N <= 50)
            # 1. For each bout, candidate set of itself and G - 1 nearest bouts
            # 2. Calculate mean pairwise trajectory distance within candidate set
            # 3. Choose candidate set with lowest score
            # 4. Put set's actual medoid first, order remaining by distance
            # =================================================================
            tier_counts["tier_2"] += 1
            tier_name = "tier_2_compact_neighborhood"
            D = pairwise_distances(trajs)

            scores = []
            candidate_sets = []
            for i in range(N):
                nbr_set = np.argsort(D[i])[:G]
                candidate_sets.append(nbr_set)
                sub_D = D[np.ix_(nbr_set, nbr_set)]
                # Mean pairwise distance among distinct pairs
                score = float(sub_D[np.triu_indices(G, k=1)].mean())
                scores.append(score)

            best_cand_idx = int(np.argmin(scores))
            winning_set = candidate_sets[best_cand_idx]

            # Find actual medoid of the winning set
            sub_D_win = D[np.ix_(winning_set, winning_set)]
            medoid_in_win = int(np.argmin(sub_D_win.sum(axis=1)))
            medoid_local_idx = int(winning_set[medoid_in_win])

            # Order remaining by distance to medoid
            rem = [int(idx) for idx in winning_set if idx != medoid_local_idx]
            rem_sorted = sorted(rem, key=lambda idx: float(D[medoid_local_idx, idx]))
            chosen_local_indices = [medoid_local_idx] + rem_sorted

        else:
            # =================================================================
            # TIER 3: Complete Bout Enumeration (N <= G)
            # Show every bout once, medoid first, leave unused tiles blank
            # =================================================================
            tier_counts["tier_3"] += 1
            tier_name = "tier_3_all_bouts"
            if N > 1:
                D = pairwise_distances(trajs)
                medoid_local_idx = int(np.argmin(D.sum(axis=1)))
                rem = [int(idx) for idx in range(N) if idx != medoid_local_idx]
                rem_sorted = sorted(rem, key=lambda idx: float(D[medoid_local_idx, idx]))
                chosen_local_indices = [medoid_local_idx] + rem_sorted
            else:
                medoid_local_idx = 0
                chosen_local_indices = [0]

        # Assemble selected bout list with blank tile placeholders
        selected_bouts = []
        for rank, b_idx in enumerate(chosen_local_indices):
            bout = bouts[b_idx]
            selected_bouts.append({
                "tile_index": rank,
                "is_blank": False,
                "is_medoid": (b_idx == medoid_local_idx),
                "rank": rank,
                "segment_name": bout["sid"],
                "onset_segment_frame": bout["onset_segment_frame"],
                "offset_segment_frame": bout["offset_segment_frame"],
                "duration_frames": bout["duration_frames"],
                "source_top_onset": bout["source_top_onset"],
                "source_top_offset": bout["source_top_offset"],
                "top_vid": bout["top_vid"],
                "bot_vid": bout["bot_vid"],
                "top_h5": bout["top_h5"],
                "bot_h5": bout["bot_h5"]
            })

        # Add explicit blank tiles if N < G
        for blank_rank in range(len(chosen_local_indices), G):
            selected_bouts.append({
                "tile_index": blank_rank,
                "is_blank": True,
                "is_medoid": False,
                "rank": blank_rank,
                "segment_name": None,
                "duration_frames": 0
            })

        usage_pct = float((syll_total_frames.get(s, 0) / total_cohort_frames) * 100.0)
        manifest["syllables"][str(s)] = {
            "syllable_id": s,
            "total_cohort_bouts": N,
            "usage_pct": round(usage_pct, 4),
            "is_active_0p1pct": bool(usage_pct >= 0.1),
            "tier": tier_name,
            "num_active_tiles": len(chosen_local_indices),
            "num_blank_tiles": G - len(chosen_local_indices),
            "bouts": selected_bouts
        }

    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 80)
    print("✅ UNIFIED SYLLABLE SELECTION MANIFEST COMPLETE!")
    print(f"Total Processed Syllables: {len(target_sylls)}")
    print(f"  Tier 1 (N > 50):       {tier_counts['tier_1']} syllables (Built-in KPMS Density)")
    print(f"  Tier 2 (G < N <= 50):  {tier_counts['tier_2']} syllables (Compact-Neighborhood Selection)")
    print(f"  Tier 3 (N <= G):       {tier_counts['tier_3']} syllables (Full Enumeration + Blanks)")
    print(f"Saved to: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
