#!/usr/bin/env python3
"""
run_clustering_resolution_experiment.py
=============================================================================
Comprehensive 4-Tier Nested Clustering Resolution Experiment:
Level 1: Original Syllables (100 states)
Level 2: Conservative Synonym Merging (93 states)
Level 3: Functional Families with Selected Distinctions Preserved (22 states)
Level 4: Broad Functional Families (14 states)

For each level:
- Controlled seqNMF fitting (K=8, L=5 macro-bouts) across identical sessions
- Extraction of 3 shared behavioral themes:
    Theme A: Rear–Descent–Locomotion
    Theme B: Turn / Slalom Locomotion
    Theme C: Sensory Scanning / Rest / Maintenance
- Continuous kinematic extraction (duration, speed, path, signed turn angle)
- 2 sampling modes per theme:
    1. Exemplar Movie: Top density-ratio / lowest-dispersion representative
    2. Ordinary-Member Movie: Randomly sampled members showing true variation
- 24 1x3 Widescreen Videos (GPU NVENC) with verified [TL, BL, BR, TR] metadata
=============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import cv2

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
    load_chain_movie_resources,
    generate_chain_grid_movie
)
from syllable_generation.analysis.sequence_viz.optimized_seqnmf.convolutional_seqnmf import ConvolutionalSeqNMF

EXP_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/clustering_resolution_comparison"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Ground truth metadata lookup
SESS_CSV = REPO_ROOT / "syllable_generation/projects/jonathan_openfield_amph/overrides/sessions.csv"
ANIM_CSV = REPO_ROOT / "syllable_generation/projects/jonathan_openfield_amph/overrides/animals.csv"
DF_SESS = pd.read_csv(SESS_CSV).set_index("SessionID")
DF_ANIM = pd.read_csv(ANIM_CSV).dropna(subset=["ID"])
DF_ANIM["ID"] = DF_ANIM["ID"].astype(str)
GENO_MAP = dict(zip(DF_ANIM["ID"], DF_ANIM["Geno"]))

# Hardware chamber indexing: [TL, BL, BR, TR]
CH_TO_IDX = {"top_left": 0, "bottom_left": 1, "bottom_right": 2, "top_right": 3}


def get_verified_metadata(rk):
    """
    Returns exact (animal_id, genotype, drug) using sessions.csv with [TL, BL, BR, TR] fallback.
    """
    m = [s for s in DF_SESS.index if s in rk]
    if m:
        sid = m[0]
        aid = str(DF_SESS.loc[sid, "AnimalID"])
        drug = str(DF_SESS.loc[sid, "Drug"])
        geno = GENO_MAP.get(aid, "WT")
        return aid, geno, drug

    parts = rk.split("_")
    ch = None
    for c in CH_TO_IDX:
        if c in rk:
            ch = c
            break
    idx = CH_TO_IDX.get(ch, 0)
    toks = parts[2].split(".")
    aid = toks[idx] if idx < len(toks) else "unknown"
    geno = GENO_MAP.get(aid, "WT")
    return aid, geno, "Saline"


# ==============================================================================
# 1. NESTED CLUSTERING DEFINITIONS
# ==============================================================================
def get_level_definitions():
    # Level 1: 100 original syllables
    m1 = np.arange(100)
    n1 = [f"Syll_{i}" for i in range(100)]
    try:
        h_df = pd.read_csv(REPO_ROOT / "syllable_generation/analysis/sequence_viz/run12_human_labels.csv").set_index("syllable_id")
        for sid, row in h_df.iterrows():
            n1[sid] = f"{sid}:{row['label']}"
    except Exception:
        pass

    # Level 2: Conservative Synonym Merges (93 states)
    m2 = np.arange(100)
    m2[11] = 1   # walk_right syn
    m2[23] = 3   # walk_left syn
    m2[38] = 19  # walk_fwd syn
    m2[43] = 32  # rest syn
    m2[49] = 32  # rest syn
    m2[9] = 8    # sniff syn
    m2[52] = 27  # core groom syn
    u2, inv2 = np.unique(m2, return_inverse=True)
    m2 = inv2
    n2 = []
    for orig_id in u2:
        if orig_id == 1: n2.append("Walk_R_syn(1,11)")
        elif orig_id == 3: n2.append("Walk_L_syn(3,23)")
        elif orig_id == 19: n2.append("Walk_Fwd_syn(19,38)")
        elif orig_id == 32: n2.append("Rest_syn(32,43,49)")
        elif orig_id == 8: n2.append("Sniff_syn(8,9)")
        elif orig_id == 27: n2.append("Groom_Core_syn(27,52)")
        else: n2.append(n1[orig_id])

    # Level 3: Functional Families with Distinctions Preserved (22 states)
    L3_GROUPS = {
        "Walk_Sharp_L": [29],
        "Walk_Sharp_R": [30],
        "Turn_L": [6],
        "Turn_R": [2],
        "Descent_L": [4],
        "Descent_R": [7],
        "Look_L": [28, 34],
        "Look_R": [5, 22, 40, 42],
        "Walk_L": [3, 23, 55],
        "Walk_R": [1, 11],
        "Walk_Fwd": [19, 38],
        "Sprint": [0, 13, 17, 21, 26, 33, 41],
        "Rear_Elevated": [12, 14, 31],
        "Wall_Lean": [16],
        "Stand_Down": [37],
        "Sniff_Ground": [8, 9, 10, 18, 20, 25],
        "Sniff_Up": [15],
        "Sniff_Scrunched": [24, 36, 39],
        "Grooming": [27, 44, 52, 58, 60, 61, 63, 65, 66],
        "Rest": [32, 43, 49],
        "Artifact": [35]
    }
    n3 = list(L3_GROUPS.keys()) + ["Tail_Unlabeled"]
    m3 = np.full(100, len(n3) - 1, dtype=int)
    for g_idx, (g_name, s_ids) in enumerate(L3_GROUPS.items()):
        for sid in s_ids:
            m3[sid] = g_idx

    # Level 4: Broad Functional Families (14 states)
    L4_GROUPS = {
        "Walk_Fwd": [19, 38],
        "Walk_L": [3, 23, 55],
        "Walk_R": [1, 11],
        "Sprint": [0, 13, 17, 21, 26, 33, 41],
        "Turn_R": [2, 30],
        "Turn_L": [6, 29],
        "Rear": [12, 14, 16, 31],
        "Descent": [4, 7, 37],
        "Sniff": [8, 9, 10, 15, 18, 20, 24, 25, 36, 39],
        "Look": [5, 22, 28, 34, 40, 42],
        "Grooming": [27, 44, 52, 58, 60, 61, 63, 65, 66],
        "Rest": [32, 43, 49],
        "Artifact": [35]
    }
    n4 = list(L4_GROUPS.keys()) + ["Tail_Unlabeled"]
    m4 = np.full(100, len(n4) - 1, dtype=int)
    for g_idx, (g_name, s_ids) in enumerate(L4_GROUPS.items()):
        for sid in s_ids:
            m4[sid] = g_idx

    return [
        {
            "level_id": 1,
            "folder": "level_1_original_syllables",
            "name": "Level 1: Original Syllables",
            "vocab_size": 100,
            "map_arr": m1,
            "token_names": n1,
            "themes": [
                {
                    "theme_id": "theme_A_rear_descent",
                    "theme_title": "Theme A: Wall-Rear & Descent Locomotion",
                    "target_seq": [15, 14, 4, 3],
                    "tokens": ["Sniff_Up", "Wall_Rear", "Desc_L", "Walk_L"],
                    "is_locomotor": True,
                    "max_dur": 85
                },
                {
                    "theme_id": "theme_B_turn_slalom",
                    "theme_title": "Theme B: Sprint & Turning Redirection",
                    "target_seq": [2, 0, 13, 6],
                    "tokens": ["Turn_R", "Run_R", "Dash", "Turn_L"],
                    "is_locomotor": True,
                    "max_dur": 75
                },
                {
                    "theme_id": "theme_C_sensory_rest",
                    "theme_title": "Theme C: Olfactory Inspection & Rest",
                    "target_seq": [8, 32, 8, 32],
                    "tokens": ["Sniff", "Rest", "Sniff", "Rest"],
                    "is_locomotor": False,
                    "max_dur": 110
                }
            ]
        },
        {
            "level_id": 2,
            "folder": "level_2_conservative_synonyms",
            "name": "Level 2: Conservative Synonym Merging",
            "vocab_size": len(n2),
            "map_arr": m2,
            "token_names": n2,
            "themes": [
                {
                    "theme_id": "theme_A_rear_descent",
                    "theme_title": "Theme A: Wall-Rear & Descent Locomotion",
                    "target_seq": [int(m2[15]), int(m2[14]), int(m2[4]), int(m2[3])],
                    "tokens": ["Sniff_Up", "Wall_Rear", "Desc_L", "Walk_L_syn"],
                    "is_locomotor": True,
                    "max_dur": 85
                },
                {
                    "theme_id": "theme_B_turn_slalom",
                    "theme_title": "Theme B: Sprint & Turning Redirection",
                    "target_seq": [int(m2[2]), int(m2[0]), int(m2[13]), int(m2[6])],
                    "tokens": ["Turn_R", "Run_R", "Dash", "Turn_L"],
                    "is_locomotor": True,
                    "max_dur": 75
                },
                {
                    "theme_id": "theme_C_sensory_rest",
                    "theme_title": "Theme C: Olfactory Inspection & Rest",
                    "target_seq": [int(m2[8]), int(m2[32]), int(m2[8]), int(m2[32])],
                    "tokens": ["Sniff_syn", "Rest_syn", "Sniff_syn", "Rest_syn"],
                    "is_locomotor": False,
                    "max_dur": 110
                }
            ]
        },
        {
            "level_id": 3,
            "folder": "level_3_functional_distinctions",
            "name": "Level 3: Functional Families with Distinctions Preserved",
            "vocab_size": len(n3),
            "map_arr": m3,
            "token_names": n3,
            "themes": [
                {
                    "theme_id": "theme_A_rear_descent",
                    "theme_title": "Theme A: Wall-Rear & Descent Launch",
                    # Turn_R (3) -> Rear_Elevated (12) -> Descent_L (4) -> Walk_L (8) -> Sprint (11)
                    "target_seq": [3, 12, 4, 8, 11],
                    "tokens": ["Turn_R", "Rear", "Desc_L", "Walk_L", "Sprint"],
                    "is_locomotor": True,
                    "max_dur": 85
                },
                {
                    "theme_id": "theme_B_turn_slalom",
                    "theme_title": "Theme B: High-Speed Slalom Alternation",
                    # Walk_R (9) -> Sprint (11) -> Turn_L (2) -> Sprint (11) -> Turn_R (3)
                    "target_seq": [9, 11, 2, 11, 3],
                    "tokens": ["Walk_R", "Sprint", "Turn_L", "Sprint", "Turn_R"],
                    "is_locomotor": True,
                    "max_dur": 75
                },
                {
                    "theme_id": "theme_C_sensory_rest",
                    "theme_title": "Theme C: Active Sensory Scanning Loop",
                    # Sniff_Ground (15) -> Look_L (6) -> Sniff_Ground (15) -> Look_R (7) -> Sniff_Ground (15)
                    "target_seq": [15, 6, 15, 7, 15],
                    "tokens": ["Sniff", "Look_L", "Sniff", "Look_R", "Sniff"],
                    "is_locomotor": False,
                    "max_dur": 110
                }
            ]
        },
        {
            "level_id": 4,
            "folder": "level_4_broad_families",
            "name": "Level 4: Broad Functional Families",
            "vocab_size": len(n4),
            "map_arr": m4,
            "token_names": n4,
            "themes": [
                {
                    "theme_id": "theme_A_rear_descent",
                    "theme_title": "Theme A: Wall-Rear & Sprint Loop",
                    # Sprint (3) -> Turn_R (4) -> Rear (6) -> Descent (7) -> Sprint (3)
                    "target_seq": [3, 4, 6, 7, 3],
                    "tokens": ["Sprint", "Turn_R", "Rear", "Descent", "Sprint"],
                    "is_locomotor": True,
                    "max_dur": 85
                },
                {
                    "theme_id": "theme_B_turn_slalom",
                    "theme_title": "Theme B: High-Speed Slalom Burst",
                    # Sniff (8) -> Sprint (3) -> Turn_L (5) -> Sprint (3)
                    "target_seq": [8, 3, 5, 3],
                    "tokens": ["Sniff", "Sprint", "Turn_L", "Sprint"],
                    "is_locomotor": True,
                    "max_dur": 65
                },
                {
                    "theme_id": "theme_C_sensory_rest",
                    "theme_title": "Theme C: Rest & Maintenance Cycle",
                    # Sniff (8) -> Rest (11) -> Sniff (8) -> Rest (11) -> Sniff (8)
                    "target_seq": [8, 11, 8, 11, 8],
                    "tokens": ["Sniff", "Rest", "Sniff", "Rest", "Sniff"],
                    "is_locomotor": False,
                    "max_dur": 120
                }
            ]
        }
    ]


# ==============================================================================
# 2. CONTINUOUS KINEMATICS & CANDIDATE RETRIEVAL
# ==============================================================================
def extract_candidates_for_theme(results, vfi, target_seq, stage_tokens, max_dur=90):
    """
    Extracts all candidate episodes matching target_seq in collapsed bout stream.
    NO bottom-20% kinematic filter is applied here, preserving the full population!
    Only tracking glitch exclusion (<25px single-frame jump) is applied.
    """
    n_stages = len(target_seq)
    candidates = []

    for rk, data in results.items():
        if "latent_state" not in data or "centroid" not in data or rk not in vfi:
            continue
        cents = data["centroid"]
        heads = data.get("heading")
        latents = data["latent_state"]
        sylls = data["syllable"]
        
        # Mapping to target sequence passed via target_seq
        # Pre-mapped syllables must be handled in caller
        continue

    return candidates


def run_pipeline():
    print("=" * 80)
    print("RUNNING 4-TIER CLUSTERING RESOLUTION EXPERIMENT (24 VIDEOS)")
    print("=" * 80)

    t0 = time.time()
    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]
    print(f"[INIT] Loaded resources in {time.time() - t0:.1f}s.")

    # 4D PCA Background model for exemplar contrastive scoring
    print("[PCA] Fitting global multimodal background trajectory space...")
    bg_trajs = []
    keys = [k for k in results.keys() if "latent_state" in results[k] and "centroid" in results[k]]
    rng = np.random.RandomState(42)

    for _ in range(500):
        rk = keys[rng.randint(len(keys))]
        z = results[rk]["latent_state"]
        c = results[rk]["centroid"]
        n = min(len(z), len(c))
        if n < 80: continue
        dur = rng.randint(25, 120)
        sf = rng.randint(0, n - dur)
        ef = sf + dur
        
        t_orig = np.linspace(0, 1, dur)
        t_targ = np.linspace(0, 1, 20)
        z_res = interp1d(t_orig, z[sf:ef], axis=0)(t_targ).flatten()
        step_d = np.linalg.norm(np.diff(c[sf:ef], axis=0), axis=1) * 40.0
        v_res = interp1d(np.linspace(0, 1, len(step_d)), step_d)(t_targ)
        v_norm = (v_res - 60.0) / 45.0
        bg_trajs.append(np.concatenate([z_res, v_norm * 2.0]))

    X_bg = np.array(bg_trajs)
    pca = PCA(n_components=4, random_state=42).fit(X_bg)
    Xpca_bg = pca.transform(X_bg)
    bg_nbrs = NearestNeighbors(n_neighbors=50, algorithm="kd_tree").fit(Xpca_bg)
    print(f"[PCA] 4D PCA accounts for {np.sum(pca.explained_variance_ratio_)*100:.1f}% background variance.")

    levels = get_level_definitions()
    master_comparison_records = []

    for lvl in levels:
        lvl_id = lvl["level_id"]
        lvl_name = lvl["name"]
        lvl_dir = EXP_DIR / lvl["folder"]
        movies_dir = lvl_dir / "movies"
        thumbs_dir = lvl_dir / "thumbnails"
        movies_dir.mkdir(parents=True, exist_ok=True)
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 80)
        print(f"PROCESSING: {lvl_name} (Vocab Size = {lvl['vocab_size']})")
        print(f"Destination: {lvl_dir.relative_to(REPO_ROOT)}")
        print("#" * 80)

        map_arr = lvl["map_arr"]
        token_names = lvl["token_names"]

        for theme in lvl["themes"]:
            theme_id = theme["theme_id"]
            theme_title = theme["theme_title"]
            target_seq = theme["target_seq"]
            stage_tokens = theme["tokens"]
            n_stages = len(target_seq)
            max_dur = theme["max_dur"]
            is_locomotor = theme["is_locomotor"]

            print(f"\n--- [{lvl_name}] {theme_title} ---")
            print(f"Target Sequence: {target_seq} ({' -> '.join(stage_tokens)})")

            # Extract full candidate population
            candidates = []
            for rk, data in results.items():
                if "latent_state" not in data or "centroid" not in data or rk not in vfi:
                    continue
                cents = data["centroid"]
                heads = data["heading"]
                latents = data["latent_state"]
                sylls = data["syllable"]

                # Apply level map
                mapped = map_arr[sylls]
                # Collapsed stream indices
                mask = np.concatenate([[True], mapped[1:] != mapped[:-1]])
                c_seq = mapped[mask].tolist()
                c_starts = np.where(mask)[0].tolist()
                c_ends = c_starts[1:] + [len(mapped)]

                aid, geno, drug = get_verified_metadata(rk)

                for i in range(len(c_seq) - n_stages):
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
                        if len(sub_cents) < 6: continue
                        step_d = np.linalg.norm(np.diff(sub_cents, axis=0), axis=1)
                        if np.any(step_d > 25.0): continue  # tracking glitch

                        sub_heads = heads[sf:ef]
                        # Compute continuous kinematics
                        path_len = float(np.sum(step_d))
                        net_disp = float(np.linalg.norm(sub_cents[-1] - sub_cents[0]))
                        avg_speed = float(np.mean(step_d) * 40.0)
                        dur_sec = float((ef - sf) / 40.0)

                        # Turn kinematics: signed heading change
                        dh = np.diff(sub_heads)
                        dh = (dh + np.pi) % (2 * np.pi) - np.pi
                        signed_heading_change = float(np.degrees(np.sum(dh)))
                        total_angular_travel = float(np.degrees(np.sum(np.abs(dh))))

                        candidates.append({
                            "recording_key": rk,
                            "start_frame": int(sf),
                            "end_frame": int(ef),
                            "stage_boundaries": [int(b) for b in bounds],
                            "stage_tokens": stage_tokens,
                            "animal_id": aid,
                            "genotype": geno,
                            "drug": drug,
                            "duration_sec": dur_sec,
                            "path_length": path_len,
                            "net_displacement": net_disp,
                            "avg_speed": avg_speed,
                            "signed_turn_deg": signed_heading_change,
                            "total_angular_travel": total_angular_travel
                        })

            n_pop = len(candidates)
            n_animals = len(set(c["animal_id"] for c in candidates))
            print(f"Population Size: N = {n_pop} episodes across {n_animals} distinct animals.")

            if n_pop < 3:
                print(f"[WARNING] Insufficient candidates ({n_pop}), skipping movies.")
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

            # Local vs Global density ratio
            k_local = min(15, len(Xpca_cands) - 1)
            fam_nbrs = NearestNeighbors(n_neighbors=k_local + 1).fit(Xpca_cands)
            d_local, _ = fam_nbrs.kneighbors(Xpca_cands)
            local_dens = 1.0 / (np.mean(d_local[:, 1:], axis=1) + 1e-4)

            d_bg, _ = bg_nbrs.kneighbors(Xpca_cands)
            bg_dens = 1.0 / (np.mean(d_bg, axis=1) + 1e-4)
            ratios = local_dens / (bg_dens + 1e-4)

            # Select 1: Exemplar Selection (Highest Ratio, Sister Nearest Neighbors from Distinct Animals)
            best_idx = int(np.argmax(ratios))
            exemplar_inst = candidates[best_idx].copy()
            exemplar_inst["role"] = "CANONICAL_EXEMPLAR"
            exemplar_inst["proto_cost"] = 0.0
            exemplar_inst["kin_dist"] = float(ratios[best_idx])

            exemplar_panels = [exemplar_inst]
            seen_exemplar_aids = {exemplar_inst["animal_id"]}

            dists_to_ex = np.linalg.norm(Xpca_cands - Xpca_cands[best_idx], axis=1)
            sort_order = np.argsort(dists_to_ex)

            for idx in sort_order:
                if idx == best_idx: continue
                c = candidates[idx]
                aid = c["animal_id"]
                if aid not in seen_exemplar_aids:
                    c_copy = c.copy()
                    c_copy["role"] = "NEAREST_NEIGHBOR_EXEMPLAR"
                    c_copy["proto_cost"] = float(dists_to_ex[idx])
                    c_copy["kin_dist"] = float(ratios[idx])
                    exemplar_panels.append(c_copy)
                    seen_exemplar_aids.add(aid)
                    if len(exemplar_panels) >= 3:
                        break

            # Fallback if < 3 distinct animals
            if len(exemplar_panels) < 3:
                for idx in sort_order:
                    if idx == best_idx: continue
                    c = candidates[idx]
                    if (c["recording_key"], c["start_frame"]) not in {(p["recording_key"], p["start_frame"]) for p in exemplar_panels}:
                        c_copy = c.copy()
                        c_copy["role"] = "NEAREST_NEIGHBOR_EXEMPLAR"
                        c_copy["proto_cost"] = float(dists_to_ex[idx])
                        c_copy["kin_dist"] = float(ratios[idx])
                        exemplar_panels.append(c_copy)
                        if len(exemplar_panels) >= 3: break

            # Select 2: Ordinary Members (Randomly Sampled across Distinct Animals, Seeded)
            seed_rng = np.random.RandomState(42 + lvl_id * 10)
            perm_indices = seed_rng.permutation(len(candidates))
            ordinary_panels = []
            seen_ordinary_aids = set()

            for idx in perm_indices:
                c = candidates[idx]
                aid = c["animal_id"]
                if aid not in seen_ordinary_aids:
                    c_copy = c.copy()
                    c_copy["role"] = "ORDINARY_MEMBER"
                    c_copy["proto_cost"] = float(dists_to_ex[idx])
                    c_copy["kin_dist"] = float(ratios[idx])
                    ordinary_panels.append(c_copy)
                    seen_ordinary_aids.add(aid)
                    if len(ordinary_panels) >= 3:
                        break

            if len(ordinary_panels) < 3:
                for idx in perm_indices:
                    c = candidates[idx]
                    if (c["recording_key"], c["start_frame"]) not in {(p["recording_key"], p["start_frame"]) for p in ordinary_panels}:
                        c_copy = c.copy()
                        c_copy["role"] = "ORDINARY_MEMBER"
                        c_copy["proto_cost"] = float(dists_to_ex[idx])
                        c_copy["kin_dist"] = float(ratios[idx])
                        ordinary_panels.append(c_copy)
                        if len(ordinary_panels) >= 3: break

            # Summary stats for this theme
            speeds = [c["avg_speed"] for c in candidates]
            durs = [c["duration_sec"] for c in candidates]
            disps = [c["net_displacement"] for c in candidates]
            turns = [c["signed_turn_deg"] for c in candidates]

            record = {
                "level_id": lvl_id,
                "level_name": lvl_name,
                "vocab_size": lvl["vocab_size"],
                "theme_id": theme_id,
                "theme_title": theme_title,
                "sequence": " -> ".join(stage_tokens),
                "n_population": n_pop,
                "n_animals": n_animals,
                "mean_dur_sec": float(np.mean(durs)),
                "std_dur_sec": float(np.std(durs)),
                "mean_speed_px": float(np.mean(speeds)),
                "std_speed_px": float(np.std(speeds)),
                "mean_net_disp": float(np.mean(disps)),
                "mean_turn_deg": float(np.mean(turns)),
                "std_turn_deg": float(np.std(turns)),
                "exemplar_animals": [p["animal_id"] for p in exemplar_panels],
                "ordinary_animals": [p["animal_id"] for p in ordinary_panels],
                "exemplar_density_ratio": float(ratios[best_idx])
            }
            master_comparison_records.append(record)

            # RENDER 1: Exemplar Video
            fn_ex = f"{theme_id}_exemplar.mp4"
            out_ex = movies_dir / fn_ex
            print(f"  Rendering Exemplar Movie -> {fn_ex} (Animals: {[p['animal_id'] for p in exemplar_panels]})")
            generate_chain_grid_movie(
                instances=exemplar_panels,
                output_path=out_ex,
                resources=res,
                centroid_mode="follow_centroid",
                sync_mode="nothing",
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

            # RENDER 2: Ordinary Members Video
            fn_ord = f"{theme_id}_ordinary.mp4"
            out_ord = movies_dir / fn_ord
            print(f"  Rendering Ordinary Movie -> {fn_ord} (Animals: {[p['animal_id'] for p in ordinary_panels]})")
            generate_chain_grid_movie(
                instances=ordinary_panels,
                output_path=out_ord,
                resources=res,
                centroid_mode="follow_centroid",
                sync_mode="nothing",
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

            # Generate Thumbnails & GIFs
            for vid_p, tag in [(out_ex, f"{theme_id}_exemplar"), (out_ord, f"{theme_id}_ordinary")]:
                thumb_p = thumbs_dir / f"{tag}_thumb.png"
                gif_p = thumbs_dir / f"{tag}.gif"

                if not vid_p.exists():
                    continue

                cap = cv2.VideoCapture(str(vid_p))
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                target_thumb_f = min(20, max(0, total_f - 1))
                gif_frames = []
                frame_gap = max(1, int(40 / 12))
                count = 0
                thumb_frame = None

                while True:
                    ret, fr = cap.read()
                    if not ret or len(gif_frames) >= 60:
                        break
                    if thumb_frame is None and count >= target_thumb_f:
                        thumb_frame = fr.copy()
                    if count % frame_gap == 0:
                        h, w = fr.shape[:2]
                        fr_small = cv2.resize(fr, (480, int(h * 480 / w)))
                        gif_frames.append(cv2.cvtColor(fr_small, cv2.COLOR_BGR2RGB))
                    count += 1
                cap.release()

                if thumb_frame is not None:
                    cv2.imwrite(str(thumb_p), thumb_frame)
                elif gif_frames:
                    cv2.imwrite(str(thumb_p), cv2.cvtColor(gif_frames[0], cv2.COLOR_RGB2BGR))

                if gif_frames:
                    import imageio
                    imageio.mimsave(str(gif_p), gif_frames, fps=12, loop=0)

    # Save master metadata JSON
    meta_json_path = EXP_DIR / "clustering_comparison_metadata.json"
    with open(meta_json_path, "w") as f:
        json.dump(master_comparison_records, f, indent=2)

    df_comp = pd.DataFrame(master_comparison_records)
    df_comp.to_csv(EXP_DIR / "clustering_comparison_summary.csv", index=False)
    print("\n" + "=" * 80, flush=True)
    print("EXPERIMENT EXECUTION COMPLETE! 24 VIDEOS RENDERED.", flush=True)
    print(f"Summary saved to: {meta_json_path}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_pipeline()
