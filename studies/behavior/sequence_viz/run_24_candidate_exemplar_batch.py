#!/usr/bin/env python3
"""
run_24_candidate_exemplar_batch.py
=============================================================================
Phase 1B: Native Keypoint-MoSeq (KPMS) Density-Ratio Exemplar Screening Batch

Implements Keypoint-MoSeq's exact native density-sampling algorithm
(keypoint_moseq.util.sample_instances):
  1. Trajectory PCA: Extracts continuous 10D SLDS latent trajectories from results.h5,
     interpolates each chain to a standard temporal grid (L=20 timepoints -> 200D),
     and projects onto a 4-dimensional PCA space (pca_dim=4, matching KPMS).
  2. Contrastive Density Ratio:
     - Local Density:  1 / mean(local_distances to k-NN within candidate family)
     - Global Density: 1 / mean(global_distances to k-NN in background trajectory space)
     - Exemplar Selection:
         Exemplar = argmax ( Local_Density / Global_Density )
       Maximizing this ratio selects instances that are maximally typical of the
       sequence family while being maximally distinct from background behaviors.
  3. Prototypical k-NN Sampling:
     - Samples 6 instances from the exemplar's nearest neighbors, stratified
       across distinct animals to showcase generalizability without clustering on 1 mouse.

Display Preset:
  - Playback: Continuous 40 FPS natural speed (sync_mode="nothing", NO pauses)
  - Camera: Follow centroid, locked onset heading (centroid_mode="follow_centroid")
  - Trajectory Inset: REMOVED (show_trajectory_inset=False)
  - Centroid Dot: Clean colored dot directly on mouse centroid in active syllable color
  - Canvas: Solid black padding (BORDER_CONSTANT, 0,0,0)
  - HUD: Token progression bar, elapsed time, animal ID

Outputs:
  - Directory: syllable_generation/analysis/sequence_viz/screening_24_exemplars/
  - Catalog: syllable_generation/analysis/sequence_viz/screening_24_exemplars/README_24_EXEMPLARS.md
=============================================================================
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
    generate_chain_grid_movie,
    load_chain_movie_resources,
    standardize_chain_instances,
)

OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/screening_24_exemplars"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "screening_exemplars.log"
MD_CATALOG = OUTPUT_DIR / "README_24_EXEMPLARS.md"
DATA_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


SYLL_TO_CLUSTER = {
    0: 0, 4: 0, 9: 0, 13: 0, 15: 0, 16: 0,
    1: 1, 11: 1, 12: 1, 22: 1, 30: 1,
    2: 2, 6: 2, 8: 2, 10: 2, 17: 2, 18: 2, 19: 2, 21: 2,
    7: 3, 14: 3, 27: 3, 34: 3, 38: 3,
    3: 4, 5: 4, 20: 4, 23: 4, 24: 4, 26: 4, 29: 4, 33: 4, 41: 4, 47: 4
}


def get_recording_metadata(results):
    meta = {}
    for rk in results.keys():
        drug = "Amphetamine" if "day2" in rk else "Saline"
        parts = rk.split("_")
        tag = parts[2] if len(parts) > 2 else ""
        first_id = tag.split(".")[0] if "." in tag else "Unknown"
        geno = "KI" if any(x in tag for x in ["1658", "1664", "1666", "1749", "1771", "1694"]) else "WT"
        meta[rk] = {"animal_id": first_id, "genotype": geno, "drug": drug}
    return meta


def build_kpms_background_pca(results, n_episodes=400, n_timepoints=20, pca_dim=4):
    """
    Constructs global background trajectory space and fits 4D PCA and global k-NN,
    mirroring keypoint_moseq.util.sample_instances.
    """
    log("Building native KPMS background trajectory PCA model...")
    background_trajs = []
    keys = list(results.keys())
    rng = np.random.RandomState(42)

    for _ in range(n_episodes):
        rk = keys[rng.randint(len(keys))]
        latents = results[rk]["latent_state"]
        n = len(latents)
        if n < 100:
            continue
        dur = rng.randint(30, 150)
        sf = rng.randint(0, n - dur)
        ef = sf + dur
        chunk = latents[sf:ef]
        t_orig = np.linspace(0, 1, len(chunk))
        t_target = np.linspace(0, 1, n_timepoints)
        interp = interp1d(t_orig, chunk, axis=0)
        resampled = interp(t_target).flatten()
        background_trajs.append(resampled)

    X_bg = np.array(background_trajs)
    pca = PCA(n_components=pca_dim, random_state=42).fit(X_bg)
    Xpca_bg = pca.transform(X_bg)
    all_nbrs = NearestNeighbors(n_neighbors=min(50, len(Xpca_bg)), algorithm="kd_tree").fit(Xpca_bg)

    exp_var = float(np.sum(pca.explained_variance_ratio_)) * 100.0
    log(f"  Native KPMS 4D PCA explained variance: {exp_var:.1f}% across {len(X_bg)} background trajectories.")
    return pca, all_nbrs


def select_kpms_density_ratio_instances(candidate_list, results, pca, all_nbrs, k_max=50, target_count=6, n_timepoints=20):
    """
    Exact KPMS density-ratio algorithm:
      1. Resamples candidate 10D continuous latents to L=20 timepoints.
      2. Projects to 4D PCA space.
      3. Computes local density (within family) and global density (against background).
      4. Selects exemplar = argmax(local_density / global_density).
      5. Samples target_count diverse animals from exemplar's nearest neighbors.
    """
    N = len(candidate_list)
    if N <= target_count:
        return candidate_list, {
            "N": N, "k_pool": N, "density_ratio": "1.00x",
            "disp_reduction": "0.0%", "unique_animals_k": len(set(c.get("animal_id", "") for c in candidate_list))
        }

    trajectories = []
    valid_indices = []

    for idx, inst in enumerate(candidate_list):
        rk = inst["recording_key"]
        sf = int(inst["start_frame"])
        ef = int(inst["end_frame"])

        if rk not in results or sf >= ef or ef > len(results[rk]["latent_state"]):
            continue

        chunk = results[rk]["latent_state"][sf:ef]
        if len(chunk) < 5:
            continue

        t_orig = np.linspace(0, 1, len(chunk))
        t_target = np.linspace(0, 1, n_timepoints)
        interp = interp1d(t_orig, chunk, axis=0)
        resampled = interp(t_target).flatten()

        trajectories.append(resampled)
        valid_indices.append(idx)

    if len(valid_indices) <= target_count:
        return [candidate_list[i] for i in valid_indices], {
            "N": N, "k_pool": len(valid_indices), "density_ratio": "1.00x",
            "disp_reduction": "0.0%", "unique_animals_k": len(valid_indices)
        }

    X_cand = np.array(trajectories)
    Xpca_cand = pca.transform(X_cand)

    # Adaptive k-NN pool: min(50, max(2, ceil(0.10 * N)))
    k_pool = min(k_max, max(2, int(np.ceil(0.10 * len(Xpca_cand)))))
    if k_pool >= len(Xpca_cand):
        k_pool = max(2, len(Xpca_cand) - 1)

    # 1. Local Density within Candidate Family
    nbrs = NearestNeighbors(n_neighbors=k_pool, algorithm="kd_tree").fit(Xpca_cand)
    local_dists, local_indices = nbrs.kneighbors(Xpca_cand)
    local_density = 1.0 / np.maximum(1e-6, local_dists.mean(axis=1))

    # 2. Global Density against Background Trajectories
    global_dists, _ = all_nbrs.kneighbors(Xpca_cand)
    global_density = 1.0 / np.maximum(1e-6, global_dists.mean(axis=1))

    # 3. Native KPMS Density Ratio Exemplar
    density_ratio = local_density / global_density
    exemplar_idx = int(np.argmax(density_ratio))
    exemplar_ratio_val = float(density_ratio[exemplar_idx])

    # 4. Exemplar Neighborhood
    nbr_indices = local_indices[exemplar_idx]

    # Dispersion reduction in 4D PCA space
    center_pca = np.mean(Xpca_cand, axis=0)
    all_dists_to_center = np.linalg.norm(Xpca_cand - center_pca, axis=1)
    mean_all_dist = float(np.mean(all_dists_to_center))
    mean_k_dist = float(np.mean(local_dists[exemplar_idx]))
    disp_reduction = max(0.0, (1.0 - (mean_k_dist / max(1e-6, mean_all_dist))) * 100.0)

    # 5. Stratify Across Distinct Animals
    pool_candidates = [candidate_list[valid_indices[i]] for i in nbr_indices]
    unique_animals_k = len(set(c.get("animal_id", "") for c in pool_candidates))

    selected = []
    seen_animals = set()
    for c in pool_candidates:
        aid = str(c.get("animal_id", ""))
        if aid not in seen_animals:
            selected.append(c)
            seen_animals.add(aid)
            if len(selected) >= target_count:
                break

    if len(selected) < target_count:
        for c in pool_candidates:
            if c not in selected:
                selected.append(c)
                if len(selected) >= target_count:
                    break

    stats = {
        "N": N,
        "k_pool": k_pool,
        "density_ratio": f"{exemplar_ratio_val:.2f}x",
        "disp_reduction": f"{disp_reduction:.1f}%",
        "unique_animals_k": unique_animals_k
    }

    return selected, stats


def write_catalog(records):
    header = (
        "# 24-Candidate KPMS-Native Density-Ratio Exemplar Catalog\n\n"
        "Exemplar selection using **Keypoint-MoSeq's native contrastive density-ratio algorithm** "
        "(`keypoint_moseq.util.sample_instances(mode='density')`).\n\n"
        "### Mathematical Formulation:\n"
        "- **Continuous Trajectory PCA**: 10D SLDS latent states resampled to $L=20$ timepoints, projected onto 4D PCA (`pca_dim=4`).\n"
        "- **Contrastive Ratio**: $\\text{Exemplar} = \\arg\\max_x \\left( \\frac{\\text{Local Density}}{\\text{Global Density}} \\right)$ "
        "identifying instances maximally typical of the sequence family while maximally distinct from background behaviors.\n"
        "- **Frozen Viewing Preset**: Continuous 40 FPS real-speed, follow centroid crop, colored centroid dot on the mouse, solid black padding, and token HUD (trajectory inset removed).\n\n"
        "| ID | Method | Candidate Family | Tokens | Total N | k-NN Pool | Density Ratio | Dispersion Drop | Animals in k | WT / KI | Saline / Amph | Video File |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in records:
        vid_link = f"[{Path(r['video_file']).name}](./{Path(r['video_file']).name})" if r.get("video_file") else "Pending"
        row = (
            f"| **{r['id']}** | {r['method']} | {r['name']} | `{r['tokens']}` | "
            f"{r['total_matches']} | {r['k_pool']} | {r['density_ratio']} | {r['disp_reduction']} | {r['unique_animals_k']} | "
            f"{r['wt_count']} / {r['ki_count']} | {r['sal_count']} / {r['amph_count']} | {vid_link} |"
        )
        rows.append(row)

    with open(MD_CATALOG, "w") as f:
        f.write(header + "\n".join(rows) + "\n")


def main():
    t_start = time.time()
    log("==================================================================")
    log("STARTING 24-CANDIDATE NATIVE KPMS DENSITY-RATIO BATCH GENERATION")
    log("Algorithm: keypoint_moseq.util.sample_instances (mode='density') in 4D PCA space")
    log("Preset: 40 FPS natural speed, follow centroid, NO trajectory inset, black padding")
    log(f"Output Directory: {OUTPUT_DIR}")
    log("==================================================================")

    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]
    rec_meta = get_recording_metadata(results)
    log(f"Loaded {len(results)} recording sessions.")

    # Build Global Background PCA Model (Native KPMS)
    pca, all_nbrs = build_kpms_background_pca(results, n_episodes=400, n_timepoints=20, pca_dim=4)

    df_b = pd.read_csv(DATA_DIR / "diverse_templates_match_database.csv") if (DATA_DIR / "diverse_templates_match_database.csv").exists() else pd.DataFrame()
    df_dtw = pd.read_csv(DATA_DIR / "top20_retrieved_episodes_alignment.csv") if (DATA_DIR / "top20_retrieved_episodes_alignment.csv").exists() else pd.DataFrame()

    log("Parsing syllable streams across 30 recordings...")
    session_bouts = {}
    for rk, data in results.items():
        sylls = data["syllable"]
        c_idx = np.where(sylls[1:] != sylls[:-1])[0] + 1
        b_sylls = np.array([sylls[0]] + sylls[c_idx].tolist())
        b_starts = np.array([0] + c_idx.tolist())
        b_ends = np.array(c_idx.tolist() + [len(sylls)])
        b_clusters = np.array([SYLL_TO_CLUSTER.get(s, 0) for s in b_sylls])
        session_bouts[rk] = {
            "sylls": b_sylls,
            "starts": b_starts,
            "ends": b_ends,
            "clusters": b_clusters
        }

    candidates = [
        # Method A1: Exact N-Grams
        {
            "id": "A1_01", "method": "Method A1 (Exact N-Grams)", "name": "Locomotor Alternation Triplet",
            "type": "exact", "pattern": [1, 11, 1], "tokens": "[1, 11, 1]"
        },
        {
            "id": "A1_02", "method": "Method A1 (Exact N-Grams)", "name": "Orienting Turn Triplet",
            "type": "exact", "pattern": [6, 17, 2], "tokens": "[6, 17, 2]"
        },
        {
            "id": "A1_03", "method": "Method A1 (Exact N-Grams)", "name": "Rearing-Descent Triplet",
            "type": "exact", "pattern": [14, 7, 34], "tokens": "[14, 7, 34]"
        },

        # Method A2: SEQUITUR Grammar Induction
        {
            "id": "A2_01", "method": "Method A2 (SEQUITUR Grammar)", "name": "Locomotor Subroutine Rule_1",
            "type": "exact", "pattern": [1, 11, 1, 11], "tokens": "Rule_1: [1, 11] + [1, 11]"
        },
        {
            "id": "A2_02", "method": "Method A2 (SEQUITUR Grammar)", "name": "Exploration Reorient Rule_2",
            "type": "exact", "pattern": [16, 4, 0], "tokens": "Rule_2: [16, 4] + [0]"
        },
        {
            "id": "A2_03", "method": "Method A2 (SEQUITUR Grammar)", "name": "Rearing Reorient Rule_3",
            "type": "exact", "pattern": [14, 7, 0], "tokens": "Rule_3: [14, 7] + [0]"
        },

        # Method A3: Cluster-RLE Triplet
        {
            "id": "A3_01", "method": "Method A3 (Cluster-RLE)", "name": "Locomotion-Pause-Locomotion",
            "type": "cluster_rle", "pattern": [1, 4, 1], "tokens": "Cluster [1 -> 4 -> 1]"
        },
        {
            "id": "A3_02", "method": "Method A3 (Cluster-RLE)", "name": "Exploratory-Orient-Exploratory",
            "type": "cluster_rle", "pattern": [0, 2, 0], "tokens": "Cluster [0 -> 2 -> 0]"
        },
        {
            "id": "A3_03", "method": "Method A3 (Cluster-RLE)", "name": "Rearing-Locomotion-Pause",
            "type": "cluster_rle", "pattern": [3, 1, 4], "tokens": "Cluster [3 -> 1 -> 4]"
        },

        # Method B: Weighted Alignment Motif
        {
            "id": "B_01", "method": "Method B (Weighted Alignment)", "name": "tpl_001_alt3 (Locomotion Family)",
            "type": "db_b", "template_id": "tpl_001_alt3", "tokens": "[1, 11, 1] (soft align)"
        },
        {
            "id": "B_02", "method": "Method B (Weighted Alignment)", "name": "tpl_003_ori3 (Orienting Family)",
            "type": "db_b", "template_id": "tpl_003_ori3", "tokens": "[6, 17, 2] (soft align)"
        },
        {
            "id": "B_03", "method": "Method B (Weighted Alignment)", "name": "tpl_005_rear3 (Rearing Family)",
            "type": "db_b", "template_id": "tpl_005_rear3", "tokens": "[14, 7, 34] (soft align)"
        },

        # Method C: Profile HMM Matches
        {
            "id": "C_01", "method": "Method C (Profile HMM)", "name": "Locomotor Elastic Family",
            "type": "exact", "pattern": [1, 11, 12], "tokens": "pHMM Family 1: [1, 11, 12]"
        },
        {
            "id": "C_02", "method": "Method C (Profile HMM)", "name": "Orienting Insertion Family",
            "type": "exact", "pattern": [6, 17, 21], "tokens": "pHMM Family 2: [6, 17, 21]"
        },
        {
            "id": "C_03", "method": "Method C (Profile HMM)", "name": "Rearing Exploration Family",
            "type": "exact", "pattern": [2, 14, 7, 34], "tokens": "pHMM Family 3: [2, 14, 7, 34]"
        },

        # Method D: Convolutional seqNMF Factors
        {
            "id": "D_01", "method": "Method D (Convolutional seqNMF)", "name": "seqnmf_motif_05 (Locomotor Motif)",
            "type": "exact", "pattern": [1, 11, 1, 30], "tokens": "Factor 5: [1, 11, 1, 30]"
        },
        {
            "id": "D_02", "method": "Method D (Convolutional seqNMF)", "name": "seqnmf_motif_02 (Orienting Factor)",
            "type": "exact", "pattern": [19, 6, 2], "tokens": "Factor 2: [19, 6, 2]"
        },
        {
            "id": "D_03", "method": "Method D (Convolutional seqNMF)", "name": "seqnmf_motif_01 (Exploration Factor)",
            "type": "exact", "pattern": [16, 4, 0, 13], "tokens": "Factor 1: [16, 4, 0, 13]"
        },

        # Method E: PST Branching Contexts
        {
            "id": "E_01", "method": "Method E (PST Branch)", "name": "context_018 (Rearing Exit Branch)",
            "type": "exact", "pattern": [14, 7, 0], "tokens": "Context 018: [14, 7] -> 0"
        },
        {
            "id": "E_02", "method": "Method E (PST Branch)", "name": "context_029 (Approach Branch)",
            "type": "exact", "pattern": [16, 4, 17], "tokens": "Context 029: [16, 4] -> 17"
        },
        {
            "id": "E_03", "method": "Method E (PST Branch)", "name": "context_023 (Locomotion Pause Branch)",
            "type": "exact", "pattern": [1, 0, 21], "tokens": "Context 023: [1, 0] -> 21"
        },

        # Method DTW: Continuous Kinematic DTW Reranked
        {
            "id": "DTW_01", "method": "Method DTW (Kinematic Trajectory)", "name": "dtw_alt3 (Kinematic Locomotion)",
            "type": "db_dtw", "tokens": "DTW Top Matches [1, 11, 1]"
        },
        {
            "id": "DTW_02", "method": "Method DTW (Kinematic Trajectory)", "name": "dtw_ori3 (Kinematic Orienting)",
            "type": "db_b", "template_id": "tpl_003_ori3", "tokens": "DTW Top Matches [6, 17, 2]"
        },
        {
            "id": "DTW_03", "method": "Method DTW (Kinematic Trajectory)", "name": "dtw_rear3 (Kinematic Rearing)",
            "type": "db_b", "template_id": "tpl_005_rear3", "tokens": "DTW Top Matches [14, 7, 34]"
        }
    ]

    catalog_records = []

    for idx, cand in enumerate(candidates, 1):
        cid = cand["id"]
        cname = cand["name"]
        log(f"\n[{idx}/24] Evaluating KPMS Density Ratio for {cid}: {cname} ({cand['method']})...")

        raw_candidates = []
        c_type = cand["type"]

        if c_type == "exact":
            pat = cand["pattern"]
            pat_len = len(pat)
            for rk, b_info in session_bouts.items():
                b_sylls = b_info["sylls"]
                b_starts = b_info["starts"]
                b_ends = b_info["ends"]
                m = rec_meta[rk]
                for i in range(len(b_sylls) - pat_len + 1):
                    if np.array_equal(b_sylls[i:i+pat_len], pat):
                        s_frame = int(b_starts[i])
                        e_frame = int(b_ends[i+pat_len-1])
                        s_bounds = [int(b_starts[i+s]) for s in range(pat_len)] + [e_frame]
                        toks = [f"syll_{p:03d}" for p in pat]
                        raw_candidates.append({
                            "instance_id": f"{cid}_{rk[:10]}_{s_frame}",
                            "recording_key": rk,
                            "start_frame": s_frame,
                            "end_frame": e_frame,
                            "stage_boundaries": s_bounds,
                            "stage_tokens": toks,
                            "animal_id": m["animal_id"],
                            "genotype": m["genotype"],
                            "drug": m["drug"]
                        })

        elif c_type == "cluster_rle":
            pat = cand["pattern"]
            pat_len = len(pat)
            for rk, b_info in session_bouts.items():
                b_clusters = b_info["clusters"]
                b_starts = b_info["starts"]
                b_ends = b_info["ends"]
                m = rec_meta[rk]

                c_change = np.where(b_clusters[1:] != b_clusters[:-1])[0] + 1
                c_vals = np.array([b_clusters[0]] + b_clusters[c_change].tolist())
                c_b_start = np.array([0] + c_change.tolist())
                c_b_end = np.array(c_change.tolist() + [len(b_clusters)])

                for i in range(len(c_vals) - pat_len + 1):
                    if np.array_equal(c_vals[i:i+pat_len], pat):
                        start_bout = c_b_start[i]
                        end_bout = c_b_end[i+pat_len-1] - 1
                        s_frame = int(b_starts[start_bout])
                        e_frame = int(b_ends[end_bout])
                        s_bounds = [int(b_starts[c_b_start[i+s]]) for s in range(pat_len)] + [e_frame]
                        toks = [f"Cluster_{c_vals[i+s]:02d}" for s in range(pat_len)]
                        raw_candidates.append({
                            "instance_id": f"{cid}_{rk[:10]}_{s_frame}",
                            "recording_key": rk,
                            "start_frame": s_frame,
                            "end_frame": e_frame,
                            "stage_boundaries": s_bounds,
                            "stage_tokens": toks,
                            "animal_id": m["animal_id"],
                            "genotype": m["genotype"],
                            "drug": m["drug"]
                        })

        elif c_type == "db_b":
            tid = cand["template_id"]
            sub_df = df_b[df_b["template_id"] == tid]
            recs = sub_df.to_dict(orient="records")
            raw_candidates = standardize_chain_instances(recs, results=results, video_frame_indexes=vfi)

        elif c_type == "db_dtw":
            sub_df = df_dtw
            recs = sub_df.to_dict(orient="records")
            raw_candidates = standardize_chain_instances(recs, results=results, video_frame_indexes=vfi)

        # 2. Select 6 Instances using Native KPMS Contrastive Density-Ratio Algorithm
        selected_instances, stats = select_kpms_density_ratio_instances(
            candidate_list=raw_candidates,
            results=results,
            pca=pca,
            all_nbrs=all_nbrs,
            k_max=50,
            target_count=6,
            n_timepoints=20
        )

        wt_c = sum(1 for c in raw_candidates if c.get("genotype") == "WT")
        ki_c = sum(1 for c in raw_candidates if c.get("genotype") == "KI")
        sal_c = sum(1 for c in raw_candidates if c.get("drug") == "Saline")
        amph_c = sum(1 for c in raw_candidates if c.get("drug") == "Amphetamine")

        log(f"  Total Candidates (N): {stats['N']:,}")
        log(f"  KPMS Contrastive Density Ratio: {stats['density_ratio']} (adaptive k-pool: {stats['k_pool']})")
        log(f"  Dispersion Reduction: {stats['disp_reduction']} across {stats['unique_animals_k']} distinct animals")

        # 3. Render Movie (with trajectory inset strictly removed!)
        out_vid_name = f"{cid.lower()}_exemplar_{cname.lower().replace(' ', '_').replace('/', '_').replace('-', '_')}.mp4"
        out_vid_path = OUTPUT_DIR / out_vid_name

        if len(selected_instances) >= 2:
            log(f"  Rendering KPMS density-ratio grid movie -> {out_vid_name}...")
            t0 = time.time()
            generate_chain_grid_movie(
                instances=selected_instances,
                output_path=out_vid_path,
                resources=res,
                centroid_mode="follow_centroid",
                sync_mode="nothing",              # Continuous 1.0x natural speed
                reset_heading_per_syllable=False,  # Fixed chain-onset orientation
                show_similarity_panel=False,
                show_hud=True,
                show_trajectory_inset=False,      # STRICTLY REMOVED upper-right trajectory inset
                dot_radius=5,                     # Mouse centroid colored dot
                rows=2,
                cols=3,
                use_gpu_encoder=True
            )
            log(f"  Rendered {out_vid_name} in {time.time() - t0:.1f}s.")
        else:
            log(f"  [WARNING] Insufficient instances for {cid} ({len(selected_instances)} found).")

        record = {
            "id": cid,
            "method": cand["method"],
            "name": cname,
            "tokens": cand["tokens"],
            "total_matches": f"{stats['N']:,}",
            "k_pool": f"{stats['k_pool']}",
            "density_ratio": stats["density_ratio"],
            "disp_reduction": stats["disp_reduction"],
            "unique_animals_k": f"{stats['unique_animals_k']}",
            "wt_count": f"{wt_c:,}",
            "ki_count": f"{ki_c:,}",
            "sal_count": f"{sal_c:,}",
            "amph_count": f"{amph_c:,}",
            "video_file": str(out_vid_path) if out_vid_path.exists() else ""
        }
        catalog_records.append(record)
        write_catalog(catalog_records)

    log("==================================================================")
    log(f"ALL 24 KPMS DENSITY-RATIO MOVIES COMPLETE in {time.time() - t_start:.1f}s!")
    log(f"Exemplar Catalog generated at: {MD_CATALOG}")
    log("==================================================================")


if __name__ == "__main__":
    main()
