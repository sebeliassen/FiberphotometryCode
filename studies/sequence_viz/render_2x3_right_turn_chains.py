#!/usr/bin/env python3
"""
render_2x3_right_turn_chains.py
=============================================================================
Generates 2x3 (6-panel) Candidate Screening Grid Movies for each
Turn Right -> Run/Walk Right -> Turn Right chain:

For each chain:
- Video A: Anchored on the #1 Most Exemplar (argmax density ratio) + 5 Nearest Neighbors (distinct mice)
- Video B: Anchored on the #2 "Second Most Exemplar" + 5 Nearest Neighbors (distinct mice)

Chains evaluated:
1. Chain 1 (Pure Triplet): Turn_R -> Run_R -> Turn_R ([2, 0, 2])
2. Chain 2 (Full Dash): Turn_R -> Run_R -> Dash -> Turn_R ([2, 0, 13, 2])
3. Chain 3 (Short Dash): Turn_R -> Run_R -> Dash_Short -> Turn_R ([2, 0, 21, 2])

Layout: 2 rows x 3 columns (6 panels, read left to right: Panels 1-3 top row, 4-6 bottom row).
Display: show_trajectory_inset=False (clean animal view, no trajectory HUD).
=============================================================================
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

# NumPy compatibility
import numpy.core.numeric as _numeric

sys.modules['numpy._core.numeric'] = _numeric

from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from studies.behavior.sequence_viz.chain_grid_movie_wrapper import (
    generate_chain_grid_movie,
    load_chain_movie_resources,
)
from studies.behavior.sequence_viz.clustering_resolution_comparison.run_clustering_resolution_experiment import (
    get_verified_metadata,
)

OUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/movies_2x3_right_turn_chains"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHAINS = [
    {
        "chain_id": "chain1_triplet",
        "title": "Chain 1: Turn_R -> Run_R -> Turn_R",
        "target_seq": [2, 0, 2],
        "tokens": ["Turn_R", "Run_R", "Turn_R"],
        "max_dur": 75
    },
    {
        "chain_id": "chain2_dash",
        "title": "Chain 2: Turn_R -> Run_R -> Dash -> Turn_R",
        "target_seq": [2, 0, 13, 2],
        "tokens": ["Turn_R", "Run_R", "Dash", "Turn_R"],
        "max_dur": 75
    },
    {
        "chain_id": "chain3_dash21",
        "title": "Chain 3: Turn_R -> Run_R -> Short Dash -> Turn_R",
        "target_seq": [2, 0, 21, 2],
        "tokens": ["Turn_R", "Run_R", "Dash_S", "Turn_R"],
        "max_dur": 75
    }
]


def generate_thumbnail(video_path, thumb_path):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_f = max(0, min(total_frames - 1, int(total_frames * 0.4)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(str(thumb_path), frame)
    cap.release()


def extract_candidates(results, vfi, target_seq, stage_tokens, max_dur=75):
    n_stages = len(target_seq)
    candidates = []
    for rk, data in results.items():
        if "latent_state" not in data or "centroid" not in data or rk not in vfi:
            continue
        cents = data["centroid"]
        heads = data["heading"]
        latents = data["latent_state"]
        sylls = data["syllable"]

        mask = np.concatenate([[True], sylls[1:] != sylls[:-1]])
        c_seq = sylls[mask].tolist()
        c_starts = np.where(mask)[0].tolist()
        c_ends = c_starts[1:] + [len(sylls)]

        aid, geno, drug = get_verified_metadata(rk)

        for i in range(len(c_seq) - n_stages + 1):
            if c_seq[i : i + n_stages] == target_seq:
                sf = c_starts[i]
                ef = c_ends[i + n_stages - 1]
                bounds = [c_starts[i + k] for k in range(n_stages)] + [ef]
                durs = np.diff(bounds)

                if not (all(d >= 2 for d in durs) and all(d <= max_dur for d in durs)):
                    continue
                if ef >= len(cents) or ef >= len(latents):
                    continue

                sub_cents = cents[sf:ef]
                if len(sub_cents) < 6:
                    continue
                step_d = np.linalg.norm(np.diff(sub_cents, axis=0), axis=1)
                if np.any(step_d > 25.0):
                    continue

                t_orig = np.linspace(0, 1, ef - sf)
                t_targ = np.linspace(0, 1, 20)
                z_res = interp1d(t_orig, latents[sf:ef], axis=0)(t_targ).flatten()
                v_res = interp1d(np.linspace(0, 1, len(step_d)), step_d * 40.0)(t_targ)
                v_norm = (v_res - 60.0) / 45.0
                feat = np.concatenate([z_res, v_norm * 2.0])

                candidates.append({
                    "recording_key": rk,
                    "start_frame": int(sf),
                    "end_frame": int(ef),
                    "stage_boundaries": [int(b) for b in bounds],
                    "stage_tokens": stage_tokens,
                    "animal_id": aid,
                    "genotype": geno,
                    "drug": drug,
                    "duration_sec": float((ef - sf) / 40.0),
                    "path_length": float(np.sum(step_d)),
                    "net_displacement": float(np.linalg.norm(sub_cents[-1] - sub_cents[0])),
                    "avg_speed": float(np.mean(step_d) * 40.0),
                    "feat": feat
                })
    return candidates


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("🎬 RENDERING 2x3 VIDEOS FOR RIGHT-TURN CHAINS (RANK 1 & RANK 2 EXEMPLARS)", flush=True)
    print(f"   Output Directory: {OUT_DIR}", flush=True)
    print("=" * 80, flush=True)

    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]

    # Background Trajectory 4D PCA
    print("[PCA] Constructing multimodal background space...", flush=True)
    bg_trajs = []
    keys = [k for k in results.keys() if "latent_state" in results[k] and "centroid" in results[k]]
    rng = np.random.RandomState(42)

    for _ in range(500):
        rk = keys[rng.randint(len(keys))]
        latents = results[rk]["latent_state"]
        cents = results[rk]["centroid"]
        n = min(len(latents), len(cents))
        if n < 80:
            continue
        dur = rng.randint(30, 90)
        sf = rng.randint(0, n - dur)
        ef = sf + dur

        t_orig = np.linspace(0, 1, dur)
        t_targ = np.linspace(0, 1, 20)
        z_res = interp1d(t_orig, latents[sf:ef], axis=0)(t_targ).flatten()
        step_d = np.linalg.norm(np.diff(cents[sf:ef], axis=0), axis=1) * 40.0
        v_res = interp1d(np.linspace(0, 1, len(step_d)), step_d)(t_targ)
        v_norm = (v_res - 60.0) / 45.0
        bg_trajs.append(np.concatenate([z_res, v_norm * 2.0]))

    X_bg = np.array(bg_trajs)
    pca_bg = PCA(n_components=4, random_state=42).fit(X_bg)
    Xpca_bg = pca_bg.transform(X_bg)
    bg_nbrs = NearestNeighbors(n_neighbors=50, algorithm="kd_tree").fit(Xpca_bg)

    rendered_movies = []

    for ch in CHAINS:
        chain_id = ch["chain_id"]
        chain_title = ch["title"]
        target_seq = ch["target_seq"]
        stage_tokens = ch["tokens"]
        max_dur = ch["max_dur"]

        print("\n" + "=" * 80, flush=True)
        print(f"PROCESSING: {chain_title}", flush=True)
        print("=" * 80, flush=True)

        cands = extract_candidates(results, vfi, target_seq, stage_tokens, max_dur=max_dur)
        print(f"Total Candidate Occurrences: N = {len(cands)} across {len(set(c['animal_id'] for c in cands))} animals.", flush=True)

        if len(cands) < 6:
            print("Insufficient candidates, skipping.", flush=True)
            continue

        # Project into 4D PCA
        X_cand = np.array([c["feat"] for c in cands])
        Xpca_cand = pca_bg.transform(X_cand)

        # Density Ratio
        k_local = min(15, len(Xpca_cand) - 1)
        fam_nbrs = NearestNeighbors(n_neighbors=k_local + 1).fit(Xpca_cand)
        d_local, _ = fam_nbrs.kneighbors(Xpca_cand)
        local_dens = 1.0 / (np.mean(d_local[:, 1:], axis=1) + 1e-4)

        d_bg, _ = bg_nbrs.kneighbors(Xpca_cand)
        bg_dens = 1.0 / (np.mean(d_bg, axis=1) + 1e-4)
        ratios = local_dens / (bg_dens + 1e-4)

        # Identify Top 2 Distinct Animal Exemplars
        sort_ratios = np.argsort(-ratios)
        exemplar_indices = []
        seen_aids = set()
        for idx in sort_ratios:
            aid = cands[idx]["animal_id"]
            if aid not in seen_aids:
                seen_aids.add(aid)
                exemplar_indices.append(idx)
                if len(exemplar_indices) == 2:
                    break

        for rank_num, anchor_idx in enumerate(exemplar_indices):
            anchor_cand = cands[anchor_idx].copy()
            anchor_aid = anchor_cand["animal_id"]
            anchor_ratio = float(ratios[anchor_idx])
            role_label = f"EXEMPLAR_RANK_{rank_num+1}"

            # Find 5 nearest neighbors from distinct animals
            dists = np.linalg.norm(Xpca_cand - Xpca_cand[anchor_idx], axis=1)
            sort_dists = np.argsort(dists)

            suite_panels = [anchor_cand]
            suite_panels[0]["role"] = role_label
            suite_panels[0]["proto_cost"] = 0.0
            suite_panels[0]["kin_dist"] = anchor_ratio

            suite_seen_aids = {anchor_aid}
            for n_idx in sort_dists:
                if n_idx == anchor_idx:
                    continue
                c = cands[n_idx]
                if c["animal_id"] not in suite_seen_aids:
                    suite_seen_aids.add(c["animal_id"])
                    c_copy = c.copy()
                    c_copy["role"] = "NEAREST_NEIGHBOR"
                    c_copy["proto_cost"] = float(dists[n_idx])
                    c_copy["kin_dist"] = float(ratios[n_idx])
                    suite_panels.append(c_copy)
                    if len(suite_panels) == 6:
                        break

            # If fewer than 6 distinct animals, fill up to 6 with closest remaining
            if len(suite_panels) < 6:
                for n_idx in sort_dists:
                    if n_idx == anchor_idx:
                        continue
                    c = cands[n_idx]
                    if (c["recording_key"], c["start_frame"]) not in {(p["recording_key"], p["start_frame"]) for p in suite_panels}:
                        c_copy = c.copy()
                        c_copy["role"] = "NEAREST_NEIGHBOR"
                        c_copy["proto_cost"] = float(dists[n_idx])
                        c_copy["kin_dist"] = float(ratios[n_idx])
                        suite_panels.append(c_copy)
                        if len(suite_panels) == 6:
                            break

            for p_idx, p in enumerate(suite_panels):
                p["instance_id"] = f"Panel {p_idx+1}: M{p['animal_id']}"
                p["panel_label"] = f"Panel {p_idx+1}"

            video_filename = f"candidate_2x3_{chain_id}_rank{rank_num+1}_exemplar.mp4"
            out_v = OUT_DIR / video_filename
            out_t = OUT_DIR / video_filename.replace(".mp4", ".png")

            subtitle = f"{chain_title} [Exemplar Rank {rank_num+1} Suite - 2x3 Grid]"
            print(f"\n--- Video {rank_num+1}: {subtitle} ---", flush=True)
            for p_idx, p in enumerate(suite_panels):
                print(f"  Panel {p_idx+1}: Animal {p['animal_id']} ({p['genotype']}, {p['drug']}) | "
                      f"Role: {p['role']} | Dist: {p['proto_cost']:.2f} | Ratio: {p['kin_dist']:.2f}x | Dur: {p['duration_sec']:.2f}s", flush=True)

            generate_chain_grid_movie(
                instances=suite_panels,
                output_path=out_v,
                resources=res,
                centroid_mode="follow_centroid",
                sync_mode="nothing",
                rows=2,
                cols=3,
                window_size=240,
                scaled_window_size=320,
                pre_sec=0.5,
                post_sec=0.75,
                fps=40,
                dot_radius=5,
                show_hud=True,
                show_trajectory_inset=False,  # Trajectory inset removed per user request
                show_similarity_panel=True,
                use_gpu_encoder=True,
                suptitle_text=subtitle
            )
            generate_thumbnail(out_v, out_t)
            print(f"  Saved poster frame -> {out_t.name}", flush=True)
            rendered_movies.append(out_v)

    total_time = time.time() - t_start
    print("\n" + "=" * 80, flush=True)
    print(f"✅ ALL {len(rendered_movies)} 2x3 MOVIES GENERATED IN {total_time:.1f}s!", flush=True)
    for m in rendered_movies:
        print(f"  - {m}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
