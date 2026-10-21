#!/usr/bin/env python3
"""
generate_3panel_whole_chain_exemplars.py
=============================================================================
Generates 1x3 (3-panel widescreen) High-Definition Diagnostic Grid Movies for
Clustered seqNMF Macro-Motifs using Multimodal Kinematic-Augmented Whole-Chain
Contrastive Density Ratio Exemplar Selection.

Solves the "Contortion in Place / Stationary Impostor" Problem:
1. 20th-Percentile Kinematic Consistency Filter:
   - For locomotor motifs (Sprints, Turns, Runs), strictly prunes candidate
     episodes falling below the 20th percentile of Total Path Length and
     Average Speed, plus requires >=25px net displacement.
   - For stationary motifs (Rest, Scanning), prunes hyperactive episodes (>80th percentile speed).
2. Multimodal Joint Manifold Exemplar Selection:
   - Evaluates both Egocentric Posture (10D continuous SLDS latents resampled to L=20 -> 200D)
     AND Allocentric Velocity Profile (instantaneous speed profile v(t) resampled to L=20 -> 20D).
   - Projects 220D joint trajectory into 4D PCA background space.
   - Exemplar = argmax(Local Family Density / Global Background Density).
   - Panels 2 & 3: Nearest neighbors in joint space, strictly stratified across distinct animals.
=============================================================================
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from studies.behavior.sequence_viz.chain_grid_movie_wrapper import (
    generate_chain_grid_movie,
    load_chain_movie_resources,
)
from studies.behavior.sequence_viz.clustered_seqnmf.prepare_clustered_sequences import (
    SYLL_TO_META_ARR,
)

OUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies_3panel_exemplars"
OUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_JSON = OUT_DIR / "whole_chain_exemplar_metadata.json"

MOTIF_SPECS = [
    {
        "id": "macro_motif_07",
        "title": "Macro-Motif 7: Kinetic Wall-Rear & Sprint Loop (KI-Enriched)",
        "target_seq": [3, 4, 6, 7, 3],
        "stage_tokens": ["Sprint", "Turn_R", "Rear", "Descent", "Sprint"],
        "filename": "exemplar_3panel_macro_motif_07_rear_sprint_loop.mp4",
        "is_locomotor": True,
        "max_dur": 85,
        "desc": "KI hyperkinetic stereotypy: ballistic approach, wall-rear climb, touchdown rebound."
    },
    {
        "id": "macro_motif_01",
        "title": "Macro-Motif 1: Rest & Maintenance Cycle (WT-Enriched)",
        "target_seq": [8, 11, 8, 11, 8],
        "stage_tokens": ["Sniff", "Rest", "Sniff", "Rest", "Sniff"],
        "filename": "exemplar_3panel_macro_motif_01_rest_maintenance_cycle.mp4",
        "is_locomotor": False,
        "max_dur": 120,
        "desc": "WT signature vigilance: resting pauses interspersed with sensory olfactory inspection."
    },
    {
        "id": "macro_motif_04",
        "title": "Macro-Motif 4: High-Speed Slalom Burst (KI-Enriched)",
        "target_seq": [8, 3, 5, 3],
        "stage_tokens": ["Sniff", "Sprint", "Turn_L", "Sprint"],
        "filename": "exemplar_3panel_macro_motif_04_slalom_burst.mp4",
        "is_locomotor": True,
        "max_dur": 65,
        "desc": "Sudden escape acceleration: olfactory sampling directly launched into lateral slalom sprint."
    },
    {
        "id": "macro_motif_08",
        "title": "Macro-Motif 8: Attentive Visual Scanning Loop (WT-Enriched)",
        "target_seq": [8, 9, 8, 9, 8],
        "stage_tokens": ["Sniff", "Look", "Sniff", "Look", "Sniff"],
        "filename": "exemplar_3panel_macro_motif_08_attentive_scanning_loop.mp4",
        "is_locomotor": False,
        "max_dur": 90,
        "desc": "WT attentive scanning: deliberate head-casting, lateral orienting, and local micro-sniffing."
    },
    {
        "id": "macro_motif_03",
        "title": "Macro-Motif 3: Rear Descent & Locomotor Launch",
        "target_seq": [6, 7, 1, 3],
        "stage_tokens": ["Rear", "Descent", "Walk_L", "Sprint"],
        "filename": "exemplar_3panel_macro_motif_03_descent_sprint_launch.mp4",
        "is_locomotor": True,
        "max_dur": 80,
        "desc": "Postural transition: vertical descent seamlessly transitioning into left-turn acceleration."
    },
    {
        "id": "macro_motif_05",
        "title": "Macro-Motif 5: Sprint, Turn & Scan",
        "target_seq": [3, 4, 8, 9],
        "stage_tokens": ["Sprint", "Turn_R", "Sniff", "Look"],
        "filename": "exemplar_3panel_macro_motif_05_sprint_turn_scan.mp4",
        "is_locomotor": True,
        "max_dur": 80,
        "desc": "Deceleration transition: fast approach curving into local ground sniffing and visual gaze."
    }
]


def build_background_pca_model(results, n_episodes=500, n_timepoints=20, pca_dim=4):
    """
    Constructs global background trajectory space and fits 4D PCA
    on joint [Latents (200D) + Speed Profile (20D)].
    """
    print(f"[PCA] Building multimodal background trajectory PCA model (N={n_episodes}, L={n_timepoints})...", flush=True)
    background_trajs = []
    keys = [k for k in results.keys() if "latent_state" in results[k] and "centroid" in results[k]]
    rng = np.random.RandomState(42)

    for _ in range(n_episodes):
        rk = keys[rng.randint(len(keys))]
        latents = results[rk]["latent_state"]
        cents = results[rk]["centroid"]
        n = min(len(latents), len(cents))
        if n < 100:
            continue
        dur = rng.randint(30, 150)
        sf = rng.randint(0, n - dur)
        ef = sf + dur

        chunk_z = latents[sf:ef]
        chunk_c = cents[sf:ef]

        # Resample posture latents
        t_orig = np.linspace(0, 1, len(chunk_z))
        t_target = np.linspace(0, 1, n_timepoints)
        z_res = interp1d(t_orig, chunk_z, axis=0)(t_target).flatten()

        # Resample speed profile
        step_d = np.linalg.norm(np.diff(chunk_c, axis=0), axis=1) * 40.0
        step_d_full = np.concatenate([[step_d[0]], step_d])
        v_res = interp1d(np.linspace(0, 1, len(step_d_full)), step_d_full)(t_target)
        v_norm = (v_res - 60.0) / 45.0  # standard scaling

        feat = np.concatenate([z_res, v_norm * 2.0])
        background_trajs.append(feat)

    X_bg = np.array(background_trajs)
    pca = PCA(n_components=pca_dim, random_state=42).fit(X_bg)
    Xpca_bg = pca.transform(X_bg)
    bg_nbrs = NearestNeighbors(n_neighbors=min(50, len(Xpca_bg)), algorithm="kd_tree").fit(Xpca_bg)

    exp_var = float(np.sum(pca.explained_variance_ratio_)) * 100.0
    print(f"[PCA] Multimodal Background 4D PCA explained variance: {exp_var:.1f}% across {len(X_bg)} background episodes.", flush=True)
    return pca, bg_nbrs


def get_animal_id_and_genotype(rk):
    parts = rk.split("_")
    tag = parts[2] if len(parts) > 2 else "unknown"
    quad_map = {'top_left': 0, 'top_right': 1, 'bottom_left': 2, 'bottom_right': 3}
    q_idx = 0
    for q, idx in quad_map.items():
        if q in rk:
            q_idx = idx
            break
    tag_parts = tag.split('.')
    animal_id = tag_parts[q_idx] if len(tag_parts) > q_idx and tag_parts[q_idx] != 'e' else tag
    if animal_id == 'e' or animal_id == tag:
        non_e = [x for x in tag.split('.') if x != 'e']
        animal_id = non_e[0] if non_e else tag[:8]
    genotype = "KI" if any(x in animal_id for x in ["1658", "1664", "1666", "1749", "1771", "1694"]) else "WT"
    return animal_id, genotype


def extract_motif_candidates(results, vfi, target_seq, stage_tokens, max_dur=85, is_locomotor=True):
    """
    Extracts candidates and applies:
    1. Tracking glitch filtering (<25px step)
    2. 20th-percentile Kinematic Consistency Filter:
       - If is_locomotor: filters out bouts with path_length < P20 OR avg_speed < P20 OR net_disp < 25px
       - If not is_locomotor: filters out bouts with avg_speed > P80
    """
    n_stages = len(target_seq)
    raw_cands = []

    for rk, data in results.items():
        if "latent_state" not in data or "centroid" not in data or rk not in vfi:
            continue
        sylls = data["syllable"]
        meta = SYLL_TO_META_ARR[sylls]
        c_idx = np.where(meta[1:] != meta[:-1])[0] + 1
        m_seq = [meta[0]] + meta[c_idx].tolist()
        b_starts = [0] + c_idx.tolist()
        b_ends = c_idx.tolist() + [len(meta)]

        animal_id, genotype = get_animal_id_and_genotype(rk)

        for i in range(len(m_seq) - n_stages):
            if m_seq[i : i + n_stages] == target_seq:
                st_frame = b_starts[i]
                end_frame = b_ends[i + n_stages - 1]
                boundaries = [b_starts[i + k] for k in range(n_stages)] + [end_frame]
                durs = np.diff(boundaries)

                # Duration bounds
                if not (all(d >= 3 for d in durs) and all(d <= max_dur for d in durs)):
                    continue

                if end_frame >= len(data["centroid"]) or end_frame >= len(data["latent_state"]):
                    continue

                cents = data["centroid"][st_frame:end_frame]
                if len(cents) < 8:
                    continue

                step_disps = np.linalg.norm(np.diff(cents, axis=0), axis=1)
                if np.any(step_disps > 25.0):  # Glitch
                    continue

                path_len = float(np.sum(step_disps))
                net_disp = float(np.linalg.norm(cents[-1] - cents[0]))
                avg_speed = float(np.mean(step_disps) * 40.0)
                max_speed = float(np.max(step_disps) * 40.0)

                raw_cands.append({
                    "recording_key": rk,
                    "start_frame": int(st_frame),
                    "end_frame": int(end_frame),
                    "stage_boundaries": [int(b) for b in boundaries],
                    "stage_tokens": stage_tokens,
                    "animal_id": animal_id,
                    "genotype": genotype,
                    "duration_sec": float((end_frame - st_frame) / 40.0),
                    "path_length": path_len,
                    "net_displacement": net_disp,
                    "avg_speed": avg_speed,
                    "max_speed": max_speed
                })

    if len(raw_cands) == 0:
        return []

    # Percentile Kinematic Consistency Filter
    path_lens = np.array([c["path_length"] for c in raw_cands])
    avg_speeds = np.array([c["avg_speed"] for c in raw_cands])

    p20_path = float(np.percentile(path_lens, 20))
    p20_speed = float(np.percentile(avg_speeds, 20))
    p80_speed = float(np.percentile(avg_speeds, 80))

    if is_locomotor:
        # Locomotor: discard bottom 20% in path length and speed, ensure net displacement
        filtered = [
            c for c in raw_cands
            if c["path_length"] >= p20_path
            and c["avg_speed"] >= p20_speed
            and c["net_displacement"] >= 25.0
        ]
        filter_summary = f"Locomotor P20 Filter: Kept {len(filtered)}/{len(raw_cands)} (Path >= {p20_path:.1f}px, Speed >= {p20_speed:.1f}px/s, Disp >= 25px)"
    else:
        # Stationary: discard hyperactive mice exceeding 80th percentile speed
        filtered = [c for c in raw_cands if c["avg_speed"] <= p80_speed]
        filter_summary = f"Stationary P80 Filter: Kept {len(filtered)}/{len(raw_cands)} (Speed <= {p80_speed:.1f}px/s)"

    print(f"  [KINEMATICS] {filter_summary}", flush=True)
    return filtered


def select_whole_chain_exemplars(candidates, results, pca, bg_nbrs, target_count=3, n_timepoints=20):
    """
    Selects 3 instances using Multimodal Kinematic-Augmented Whole-Chain Contrastive Density Ratio:
    - Feature: Resampled 10D continuous SLDS latents (200D) + Speed Profile (20D).
    - Evaluates in 4D PCA space.
    - Canonical Exemplar = argmax(Local Density / Global Density).
    - Panels 2 & 3: Nearest neighbors in 4D latent space from distinct animals.
    """
    N = len(candidates)
    if N <= target_count:
        return candidates, {"N": N, "density_ratio": "1.00x", "disp_reduction": "0.0%", "unique_animals": len(candidates)}

    trajectories = []
    valid_indices = []

    for idx, inst in enumerate(candidates):
        rk = inst["recording_key"]
        sf = inst["start_frame"]
        ef = inst["end_frame"]

        chunk_z = results[rk]["latent_state"][sf:ef]
        chunk_c = results[rk]["centroid"][sf:ef]

        if len(chunk_z) < 5 or len(chunk_c) < 5:
            continue

        # Posture latents (200D)
        t_orig = np.linspace(0, 1, len(chunk_z))
        t_target = np.linspace(0, 1, n_timepoints)
        z_res = interp1d(t_orig, chunk_z, axis=0)(t_target).flatten()

        # Speed profile (20D)
        step_d = np.linalg.norm(np.diff(chunk_c, axis=0), axis=1) * 40.0
        step_d_full = np.concatenate([[step_d[0]], step_d])
        v_res = interp1d(np.linspace(0, 1, len(step_d_full)), step_d_full)(t_target)
        v_norm = (v_res - 60.0) / 45.0

        feat = np.concatenate([z_res, v_norm * 2.0])
        trajectories.append(feat)
        valid_indices.append(idx)

    X_cand = np.array(trajectories)
    Xpca_cand = pca.transform(X_cand)

    k_pool = min(30, max(2, int(np.ceil(0.10 * len(Xpca_cand)))))
    if k_pool >= len(Xpca_cand):
        k_pool = max(2, len(Xpca_cand) - 1)

    # Local Density within Candidate Family
    nbrs = NearestNeighbors(n_neighbors=k_pool, algorithm="kd_tree").fit(Xpca_cand)
    local_dists, local_indices = nbrs.kneighbors(Xpca_cand)
    local_density = 1.0 / np.maximum(1e-6, local_dists.mean(axis=1))

    # Global Density against Background
    global_dists, _ = bg_nbrs.kneighbors(Xpca_cand)
    global_density = 1.0 / np.maximum(1e-6, global_dists.mean(axis=1))

    # Contrastive Density Ratio
    density_ratio = local_density / global_density
    exemplar_idx = int(np.argmax(density_ratio))
    exemplar_ratio_val = float(density_ratio[exemplar_idx])

    # Dispersion reduction relative to cluster center
    center_pca = np.mean(Xpca_cand, axis=0)
    all_dists_to_center = np.linalg.norm(Xpca_cand - center_pca, axis=1)
    mean_all_dist = float(np.mean(all_dists_to_center))
    mean_k_dist = float(np.mean(local_dists[exemplar_idx]))
    disp_reduction = max(0.0, (1.0 - (mean_k_dist / max(1e-6, mean_all_dist))) * 100.0)

    # Distances of all candidates to Canonical Exemplar in 4D latent PCA space
    dists_to_exemplar = np.linalg.norm(Xpca_cand - Xpca_cand[exemplar_idx], axis=1)
    sorted_order = np.argsort(dists_to_exemplar)

    # Panel 1: Canonical Exemplar
    canonical_inst = candidates[valid_indices[exemplar_idx]].copy()
    canonical_inst["role"] = "CANONICAL_EXEMPLAR"
    canonical_inst["proto_cost"] = 0.0
    canonical_inst["kin_dist"] = 0.05
    selected = [canonical_inst]
    seen_animals = {canonical_inst["animal_id"]}

    # Panels 2 & 3: Nearest neighbors in 4D latent PCA space from distinct animals
    for idx in sorted_order:
        if idx == exemplar_idx:
            continue
        c = candidates[valid_indices[idx]]
        aid = c["animal_id"]
        if aid not in seen_animals:
            c_copy = c.copy()
            c_copy["role"] = "NEAREST_NEIGHBOR_EXEMPLAR"
            c_copy["proto_cost"] = float(dists_to_exemplar[idx])
            c_copy["kin_dist"] = 0.10
            selected.append(c_copy)
            seen_animals.add(aid)
            if len(selected) >= target_count:
                break

    # If < 3 distinct animals available, select closest distinct non-duplicate episodes
    if len(selected) < target_count:
        selected_eps = {(x["recording_key"], x["start_frame"]) for x in selected}
        for idx in sorted_order:
            if idx == exemplar_idx:
                continue
            c = candidates[valid_indices[idx]]
            ep_key = (c["recording_key"], c["start_frame"])
            if ep_key not in selected_eps:
                c_copy = c.copy()
                c_copy["role"] = "NEAREST_NEIGHBOR_EXEMPLAR"
                c_copy["proto_cost"] = float(dists_to_exemplar[idx])
                c_copy["kin_dist"] = 0.15
                selected.append(c_copy)
                selected_eps.add(ep_key)
                if len(selected) >= target_count:
                    break

    stats = {
        "N_candidates": N,
        "k_pool": k_pool,
        "density_ratio": f"{exemplar_ratio_val:.2f}x",
        "disp_reduction": f"{disp_reduction:.1f}%",
        "unique_animals_selected": len(seen_animals),
        "canonical_animal": canonical_inst["animal_id"],
        "canonical_genotype": canonical_inst["genotype"],
        "canonical_path_length": f"{canonical_inst['path_length']:.1f}px",
        "canonical_net_disp": f"{canonical_inst['net_displacement']:.1f}px",
        "canonical_avg_speed": f"{canonical_inst['avg_speed']:.1f}px/s"
    }

    return selected, stats


def main():
    print("=" * 80)
    print("KINEMATIC-AUGMENTED WHOLE-CHAIN EXEMPLAR SELECTION (1x3 GRID MOVIES)")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 80)

    t0 = time.time()
    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]
    print(f"Resources loaded in {time.time() - t0:.1f}s.")

    pca_model, bg_nbrs = build_background_pca_model(results, n_episodes=500, n_timepoints=20, pca_dim=4)

    all_metadata = []

    for idx, spec in enumerate(MOTIF_SPECS, 1):
        print("\n" + "-" * 80)
        print(f"[{idx}/{len(MOTIF_SPECS)}] Processing: {spec['title']}")
        print(f"Target Sequence: {spec['target_seq']} ({' -> '.join(spec['stage_tokens'])})")

        candidates = extract_motif_candidates(
            results, vfi, spec["target_seq"], spec["stage_tokens"],
            max_dur=spec["max_dur"], is_locomotor=spec["is_locomotor"]
        )
        print(f"  Extracted {len(candidates)} clean, kinematically consistent candidate episodes.")

        if len(candidates) < 3:
            print(f"  [WARNING] Insufficient candidates ({len(candidates)}), skipping.")
            continue

        selected_instances, stats = select_whole_chain_exemplars(
            candidates, results, pca_model, bg_nbrs, target_count=3, n_timepoints=20
        )

        print(f"  Exemplar Audit: Density Ratio = {stats['density_ratio']} | Dispersion Reduction = {stats['disp_reduction']}")
        print(f"  Canonical Metrics: PathLen = {stats['canonical_path_length']} | NetDisp = {stats['canonical_net_disp']} | AvgSpeed = {stats['canonical_avg_speed']}")
        print(f"  Selected 3 panels from animals: {[inst['animal_id'] for inst in selected_instances]}")

        out_path = OUT_DIR / spec["filename"]
        print(f"  Rendering 1x3 Widescreen Movie -> {out_path.name}...")
        t_render = time.time()

        generate_chain_grid_movie(
            instances=selected_instances,
            output_path=out_path,
            resources=res,
            centroid_mode="follow_centroid",
            sync_mode="nothing",  # Fluid continuous natural speed
            rows=1,
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
            use_gpu_encoder=True
        )

        elapsed = time.time() - t_render
        size_mb = os.path.getsize(out_path) / 1e6 if out_path.exists() else 0.0
        print(f"  ✓ Successfully rendered {out_path.name} in {elapsed:.1f}s ({size_mb:.2f} MB)")

        all_metadata.append({
            "motif_id": spec["id"],
            "title": spec["title"],
            "tokens": spec["stage_tokens"],
            "sequence": spec["target_seq"],
            "is_locomotor": spec["is_locomotor"],
            "filename": spec["filename"],
            "filepath": str(out_path),
            "size_mb": round(size_mb, 2),
            "stats": stats,
            "panels": [
                {
                    "panel_index": p_idx + 1,
                    "animal_id": inst["animal_id"],
                    "genotype": inst["genotype"],
                    "role": inst.get("role", "EXEMPLAR"),
                    "duration_sec": inst["duration_sec"],
                    "path_length_px": round(inst["path_length"], 1),
                    "net_displacement_px": round(inst["net_displacement"], 1),
                    "avg_speed_px_s": round(inst["avg_speed"], 1),
                    "recording_key": inst["recording_key"],
                    "start_frame": inst["start_frame"],
                    "end_frame": inst["end_frame"]
                }
                for p_idx, inst in enumerate(selected_instances)
            ]
        })

    with open(METADATA_JSON, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print("\n" + "=" * 80)
    print(f"COMPLETED! Rendered {len(all_metadata)} Kinematic-Augmented 1x3 Exemplar Movies.")
    print(f"Metadata saved to: {METADATA_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
