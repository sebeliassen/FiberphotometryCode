#!/usr/bin/env python3
"""
render_2x3_friday_seqnmf_exemplars.py
=============================================================================
Generates 2x3 (6-panel) Candidate Screening Grid Movies matching Friday's exact
Level 1 seqNMF exemplar pipeline (run_clustering_resolution_experiment.py):

- Evaluated across the complete dataset (all 30 sessions: WT & KI, Saline & Amph)
- Multimodal 4D PCA space (continuous SLDS pose latents + instantaneous speed profile)
- Native contrastive density-ratio scoring: Local Density / Global Background Density
- Panel 1: Canonical Exemplar (argmax density ratio)
- Panels 2-6: Nearest Neighbors in continuous space, stratified across distinct animals
- Display preset from Friday: 40 FPS real-speed, follow centroid crop,
  trajectory inset in top right, similarity panel, and bottom token progression HUD.

Themes:
1. Theme A: Wall-Rear & Descent ([15, 14, 4, 3]: Sniff_Up -> Wall_Rear -> Desc_L -> Walk_L)
2. Theme B: Turn & Sprint ([2, 0, 13, 6]: Turn_R -> Run_R -> Dash -> Turn_L)
3. Theme B (Left-Turn variant): Turn Left & Sprint ([6, 17, 6]: Turn_L -> Run_L -> Turn_L)
4. Theme C: Olfactory Inspection & Rest ([8, 32, 8, 32]: Sniff -> Rest -> Sniff -> Rest)
=============================================================================
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

# NumPy compatibility
import numpy.core.numeric as _numeric
sys.modules['numpy._core.numeric'] = _numeric

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
    load_chain_movie_resources,
    generate_chain_grid_movie
)
from syllable_generation.analysis.sequence_viz.clustering_resolution_comparison.run_clustering_resolution_experiment import (
    get_verified_metadata
)

OUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/movies_2x3_friday_seqnmf"
OUT_DIR.mkdir(parents=True, exist_ok=True)


THEMES = [
    {
        "theme_id": "theme_A_rear_descent",
        "theme_title": "Theme A: Wall-Rear & Descent Locomotion",
        "target_seq": [15, 14, 4, 3],
        "tokens": ["Sniff_Up", "Wall_Rear", "Desc_L", "Walk_L"],
        "max_dur": 85,
        "filename": "candidate_2x3_theme_A_rear_descent.mp4"
    },
    {
        "theme_id": "theme_B_turn_slalom",
        "theme_title": "Theme B: Sprint & Turning Redirection",
        "target_seq": [2, 0, 13, 6],
        "tokens": ["Turn_R", "Run_R", "Dash", "Turn_L"],
        "max_dur": 75,
        "filename": "candidate_2x3_theme_B_turn_slalom.mp4"
    },
    {
        "theme_id": "theme_B_left_sprint",
        "theme_title": "Theme B (Left): Turn Left -> Run Left -> Turn Left",
        "target_seq": [6, 17, 6],
        "tokens": ["Turn_L", "Run_L", "Turn_L"],
        "max_dur": 75,
        "filename": "candidate_2x3_theme_B_left_turn_run.mp4"
    },
    {
        "theme_id": "theme_C_sensory_rest",
        "theme_title": "Theme C: Olfactory Inspection & Rest",
        "target_seq": [8, 32, 8, 32],
        "tokens": ["Sniff", "Rest", "Sniff", "Rest"],
        "max_dur": 110,
        "filename": "candidate_2x3_theme_C_sensory_rest.mp4"
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


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("🎬 RENDERING 2x3 (6-PANEL) EXEMPLAR MOVIES VIA FRIDAY's SEQNMF PIPELINE", flush=True)
    print(f"   Output Directory: {OUT_DIR}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load resources
    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]

    # 2. Build 4D PCA Background (same as Friday's run_clustering_resolution_experiment.py)
    print("[PCA] Constructing multimodal background trajectory space (L=20 timepoints)...", flush=True)
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
    pca = PCA(n_components=4, random_state=42).fit(X_bg)
    Xpca_bg = pca.transform(X_bg)
    bg_nbrs = NearestNeighbors(n_neighbors=50, algorithm="kd_tree").fit(Xpca_bg)
    print(f"[PCA] 4D PCA accounts for {np.sum(pca.explained_variance_ratio_)*100:.1f}% background variance.", flush=True)

    # 3. Process each theme
    rendered_videos = []

    for theme in THEMES:
        theme_id = theme["theme_id"]
        theme_title = theme["theme_title"]
        target_seq = theme["target_seq"]
        stage_tokens = theme["tokens"]
        n_stages = len(target_seq)
        max_dur = theme["max_dur"]
        fn = theme["filename"]

        print("\n" + "-" * 80, flush=True)
        print(f"PROFILING: {theme_title}", flush=True)
        print(f"Target Sequence: {target_seq} ({' -> '.join(stage_tokens)})", flush=True)

        candidates = []
        for rk, data in results.items():
            if "latent_state" not in data or "centroid" not in data or rk not in vfi:
                continue
            cents = data["centroid"]
            heads = data["heading"]
            latents = data["latent_state"]
            sylls = data["syllable"]

            # Collapsed syllable bout stream
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
                        continue  # tracking glitch

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
                        "avg_speed": float(np.mean(step_d) * 40.0)
                    })

        n_pop = len(candidates)
        n_animals = len(set(c["animal_id"] for c in candidates))
        print(f"Population: N = {n_pop} episodes across {n_animals} distinct animals (WT, KI, Saline, Amphetamine).", flush=True)

        if n_pop < 3:
            print(f"[WARNING] Insufficient candidates ({n_pop}), skipping.", flush=True)
            continue

        # Project candidates into 4D PCA
        c_feats = []
        for c in candidates:
            sf, ef = c["start_frame"], c["end_frame"]
            rk = c["recording_key"]
            z = results[rk]["latent_state"][sf:ef]
            cents_sub = results[rk]["centroid"][sf:ef]

            t_orig = np.linspace(0, 1, ef - sf)
            t_targ = np.linspace(0, 1, 20)
            z_res = interp1d(t_orig, z, axis=0)(t_targ).flatten()
            step_d = np.linalg.norm(np.diff(cents_sub, axis=0), axis=1) * 40.0
            v_res = interp1d(np.linspace(0, 1, len(step_d)), step_d)(t_targ)
            v_norm = (v_res - 60.0) / 45.0
            c_feats.append(np.concatenate([z_res, v_norm * 2.0]))

        X_cands = np.array(c_feats)
        Xpca_cands = pca.transform(X_cands)

        # Density Ratio: Local / Global
        k_local = min(15, len(Xpca_cands) - 1)
        fam_nbrs = NearestNeighbors(n_neighbors=k_local + 1).fit(Xpca_cands)
        d_local, _ = fam_nbrs.kneighbors(Xpca_cands)
        local_dens = 1.0 / (np.mean(d_local[:, 1:], axis=1) + 1e-4)

        d_bg, _ = bg_nbrs.kneighbors(Xpca_cands)
        bg_dens = 1.0 / (np.mean(d_bg, axis=1) + 1e-4)
        ratios = local_dens / (bg_dens + 1e-4)

        best_idx = int(np.argmax(ratios))
        exemplar_inst = candidates[best_idx].copy()
        exemplar_inst["role"] = "CANONICAL_EXEMPLAR"
        exemplar_inst["proto_cost"] = 0.0
        exemplar_inst["kin_dist"] = float(ratios[best_idx])

        # Pick 6 panels (1 canonical + 5 nearest neighbors across distinct animals)
        exemplar_panels = [exemplar_inst]
        seen_exemplar_aids = {exemplar_inst["animal_id"]}

        dists_to_ex = np.linalg.norm(Xpca_cands - Xpca_cands[best_idx], axis=1)
        sort_order = np.argsort(dists_to_ex)

        for idx in sort_order:
            if idx == best_idx:
                continue
            c = candidates[idx]
            aid = c["animal_id"]
            if aid not in seen_exemplar_aids:
                c_copy = c.copy()
                c_copy["role"] = "NEAREST_NEIGHBOR_EXEMPLAR"
                c_copy["proto_cost"] = float(dists_to_ex[idx])
                c_copy["kin_dist"] = float(ratios[idx])
                exemplar_panels.append(c_copy)
                seen_exemplar_aids.add(aid)
                if len(exemplar_panels) >= 6:
                    break

        # Fallback if fewer than 6 distinct animals in pool
        if len(exemplar_panels) < 6:
            for idx in sort_order:
                if idx == best_idx:
                    continue
                c = candidates[idx]
                if (c["recording_key"], c["start_frame"]) not in {(p["recording_key"], p["start_frame"]) for p in exemplar_panels}:
                    c_copy = c.copy()
                    c_copy["role"] = "NEAREST_NEIGHBOR_EXEMPLAR"
                    c_copy["proto_cost"] = float(dists_to_ex[idx])
                    c_copy["kin_dist"] = float(ratios[idx])
                    exemplar_panels.append(c_copy)
                    if len(exemplar_panels) >= 6:
                        break

        # Tag panel numbers for easy user cherry-picking
        for p_idx, p in enumerate(exemplar_panels):
            p["instance_id"] = f"Panel {p_idx+1}: M{p['animal_id']}"
            p["panel_label"] = f"Panel {p_idx+1}"

        print("\nSelected 6 Candidate Panels:", flush=True)
        for p_idx, p in enumerate(exemplar_panels):
            print(f"  Panel {p_idx+1}: Animal {p['animal_id']} ({p['genotype']}, {p['drug']}) | "
                  f"Role: {p['role']} | Dist: {p['proto_cost']:.2f} | Ratio: {p['kin_dist']:.2f}x | Dur: {p['duration_sec']:.2f}s", flush=True)

        out_video = OUT_DIR / fn
        out_thumb = OUT_DIR / fn.replace(".mp4", ".png")

        print(f"\n[RENDER] Rendering 2x3 Grid Movie -> {out_video.name}...", flush=True)
        generate_chain_grid_movie(
            instances=exemplar_panels,
            output_path=out_video,
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
            show_trajectory_inset=True,
            show_similarity_panel=True,
            use_gpu_encoder=True,
            suptitle_text=f"{theme_title} [2x3 Candidate Selection Grid]"
        )
        generate_thumbnail(out_video, out_thumb)
        print(f"  Saved poster frame -> {out_thumb.name}", flush=True)
        rendered_videos.append(out_video)

    total_time = time.time() - t_start
    print("\n" + "=" * 80, flush=True)
    print(f"✅ ALL 2x3 CANDIDATE MOVIES GENERATED IN {total_time:.1f}s!", flush=True)
    for v in rendered_videos:
        print(f"  - {v}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
