#!/usr/bin/env python3
"""
audit_clustering_discrepancy.py
=============================================================================
Comprehensive Diagnostic and Audit of Clustering Resolution Comparisons:
1. Traces candidate counts at every filter step for all 4 levels and 3 themes.
2. Identifies exact sequence definitions, lengths, and reasons for Level 3 candidate pool sizes.
3. Tests matched action queries (apples-to-apples across all 4 levels).
4. Audits reference trajectory bias and tests pooled unbiased references.
5. Discovers each level's OWN top recurrent families.
6. Computes P80 (and P90) operating points and exports master per-episode CSV.
7. Generates comprehensive audit plots and renders P80 1x3 widescreen videos.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
    load_chain_movie_resources,
    generate_chain_grid_movie
)
from syllable_generation.analysis.sequence_viz.clustering_resolution_comparison.run_clustering_resolution_experiment import (
    get_level_definitions,
    get_verified_metadata
)

EXP_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/clustering_resolution_comparison"
AUDIT_DIR = EXP_DIR / "diagnostic_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def extract_continuous_features(results, rk, sf, ef, t_steps=20):
    cents = results[rk]["centroid"][sf:ef]
    heads = results[rk]["heading"][sf:ef]
    latents = results[rk]["latent_state"][sf:ef]

    t_orig = np.linspace(0, 1, ef - sf)
    t_targ = np.linspace(0, 1, t_steps)

    # 1. Pose latent states (10 dims)
    z_res = interp1d(t_orig, latents, axis=0)(t_targ).flatten()

    # 2. Centroid speed profile (px/s)
    step_d = np.linalg.norm(np.diff(cents, axis=0), axis=1) * 40.0
    v_res = interp1d(np.linspace(0, 1, len(step_d)), step_d)(t_targ)
    v_norm = (v_res - 60.0) / 45.0

    # 3. Heading angular velocity (deg/s)
    dh = np.diff(heads)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    dh_res = interp1d(np.linspace(0, 1, len(dh)), dh * 40.0)(t_targ)
    dh_norm = dh_res / 5.0

    return np.concatenate([z_res, v_norm * 2.0, dh_norm * 2.0])


def main():
    print("=" * 80)
    print("STARTING CLUSTERING DISCREPANCY AUDIT & P80 VIDEO PIPELINE")
    print("=" * 80)
    t0 = time.time()

    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]
    levels = get_level_definitions()

    # =========================================================================
    # 1. CANDIDATE TRACE AT EVERY FILTER STEP (Question 2)
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. TRACING CANDIDATE COUNTS AT EVERY FILTER STEP")
    print("=" * 80)

    trace_records = []
    all_theme_candidates = []  # Will hold all extracted candidate episodes

    for lvl in levels:
        lvl_id = lvl["level_id"]
        lvl_name = lvl["name"]
        m_arr = lvl["map_arr"]
        token_names = lvl["token_names"]

        for th in lvl["themes"]:
            th_id = th["theme_id"]
            th_title = th["theme_title"]
            tseq = th["target_seq"]
            stage_tokens = th["tokens"]
            max_dur = th["max_dur"]
            n_stages = len(tseq)

            raw_ngram_matches = 0
            pass_dur = 0
            pass_tracking = 0
            pass_len = 0

            animals_raw = set()
            animals_final = set()

            for rk, data in results.items():
                if "latent_state" not in data or "centroid" not in data or rk not in vfi:
                    continue
                cents = data["centroid"]
                heads = data["heading"]
                latents = data["latent_state"]
                sylls = data["syllable"]
                mapped = m_arr[sylls]

                mask = np.concatenate([[True], mapped[1:] != mapped[:-1]])
                c_seq = mapped[mask].tolist()
                c_starts = np.where(mask)[0].tolist()
                c_ends = c_starts[1:] + [len(mapped)]
                aid, geno, drug = get_verified_metadata(rk)

                for i in range(len(c_seq) - n_stages):
                    if c_seq[i : i + n_stages] == tseq:
                        raw_ngram_matches += 1
                        animals_raw.add(aid)

                        sf = c_starts[i]
                        ef = c_ends[i + n_stages - 1]
                        bounds = [c_starts[i + k] for k in range(n_stages)] + [ef]
                        durs = np.diff(bounds)

                        # Duration filter check
                        dur_ok = all(d >= 2 for d in durs) and all(d <= max_dur for d in durs)
                        if not dur_ok:
                            continue
                        pass_dur += 1

                        # Length / bounds check
                        if ef >= len(cents) or (ef - sf) < 8:
                            continue
                        pass_len += 1

                        # Tracking glitch check (<25px jump)
                        sub_cents = cents[sf:ef]
                        step_d = np.linalg.norm(np.diff(sub_cents, axis=0), axis=1)
                        if np.any(step_d > 25.0):
                            continue
                        pass_tracking += 1
                        animals_final.add(aid)

                        # Kinematic extraction
                        feat = extract_continuous_features(results, rk, sf, ef)
                        path_len = float(np.sum(step_d))
                        net_disp = float(np.linalg.norm(sub_cents[-1] - sub_cents[0]))
                        avg_speed = float(np.mean(step_d) * 40.0)
                        dur_sec = float((ef - sf) / 40.0)

                        dh = np.diff(heads[sf:ef])
                        dh = (dh + np.pi) % (2 * np.pi) - np.pi
                        signed_heading = float(np.degrees(np.sum(dh)))

                        all_theme_candidates.append({
                            "level_id": lvl_id,
                            "level_name": lvl_name,
                            "theme_id": th_id,
                            "theme_title": th_title,
                            "sequence_length": n_stages,
                            "target_seq": str(tseq),
                            "stage_tokens": stage_tokens,
                            "matched_tokens": stage_tokens,
                            "target_tokens": " -> ".join(stage_tokens),
                            "recording_key": rk,
                            "animal_id": aid,
                            "genotype": geno,
                            "drug": drug,
                            "start_frame": int(sf),
                            "end_frame": int(ef),
                            "stage_boundaries": [int(b) for b in bounds],
                            "duration_sec": dur_sec,
                            "avg_speed_px_s": avg_speed,
                            "path_length_px": path_len,
                            "net_displacement_px": net_disp,
                            "signed_turn_deg": signed_heading,
                            "feature_vector": feat
                        })

            trace_records.append({
                "Level": lvl_name,
                "Level_ID": lvl_id,
                "Theme": th_title,
                "Theme_ID": th_id,
                "Seq_Length": n_stages,
                "Target_Sequence": " -> ".join(stage_tokens),
                "Raw_Collapsed_Ngrams": raw_ngram_matches,
                "Animals_Raw": len(animals_raw),
                "Pass_Duration_Filter": pass_dur,
                "Pass_Length_Filter": pass_len,
                "Pass_Tracking_Quality": pass_tracking,
                "Animals_Final": len(animals_final),
                "Tracking_Exclusion_Pct": f"{(pass_len - pass_tracking) / max(1, pass_len) * 100:.1f}%",
                "Duration_Exclusion_Pct": f"{(raw_ngram_matches - pass_dur) / max(1, raw_ngram_matches) * 100:.1f}%"
            })

    df_trace = pd.DataFrame(trace_records)
    trace_csv = AUDIT_DIR / "candidate_filtering_trace_by_level.csv"
    df_trace.to_csv(trace_csv, index=False)
    print(f"[TRACE] Saved filtering trace to {trace_csv.relative_to(REPO_ROOT)}")
    print(df_trace[["Level_ID", "Theme_ID", "Seq_Length", "Raw_Collapsed_Ngrams", "Pass_Tracking_Quality", "Animals_Final"]].to_string())

    # =========================================================================
    # 2. AUDIT REFERENCE TRAJECTORY & COMPUTE SIMILARITIES
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. AUDITING REFERENCE BIAS & COMPUTING UNBIASED POOLED SIMILARITIES")
    print("=" * 80)

    # For each theme, compute:
    # A) Level 1 reference (previous method)
    # B) Pooled multi-level reference (mean across all valid candidates from all levels)
    # C) Per-level own centroid reference
    theme_ids = ["theme_A_rear_descent", "theme_B_turn_slalom", "theme_C_sensory_rest"]
    
    for th_id in theme_ids:
        th_cands = [c for c in all_theme_candidates if c["theme_id"] == th_id]
        l1_cands = [c for c in th_cands if c["level_id"] == 1]
        
        l1_feats = np.array([c["feature_vector"] for c in l1_cands])
        all_feats = np.array([c["feature_vector"] for c in th_cands])
        
        ref_l1 = np.mean(l1_feats, axis=0)
        ref_l1_norm = ref_l1 / (np.linalg.norm(ref_l1) + 1e-8)
        
        ref_pool = np.mean(all_feats, axis=0)
        ref_pool_norm = ref_pool / (np.linalg.norm(ref_pool) + 1e-8)
        
        for c in th_cands:
            f = c["feature_vector"]
            fn = f / (np.linalg.norm(f) + 1e-8)
            c["similarity_l1_ref"] = float(np.dot(fn, ref_l1_norm))
            c["similarity_pooled_ref"] = float(np.dot(fn, ref_pool_norm))

    # =========================================================================
    # 3. DIRECT APPLES-TO-APPLES MATCHED ACTION CHAINS ACROSS ALL 4 LEVELS
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. TESTING APPLES-TO-APPLES MATCHED ACTION CHAINS")
    print("=" * 80)

    # To eliminate length and vocabulary discrepancies, we query the exact same
    # core behavioral verbs across all 4 levels:
    # Motif 1: Rear -> Descent -> Walk
    # Motif 2: Sprint -> Turn_R (or Turn) -> Sprint
    # Motif 3: Sniff -> Rest -> Sniff
    matched_queries = [
        {
            "motif_id": "matched_rear_descent_walk",
            "motif_title": "Matched Motif 1: Rear -> Descent -> Walk",
            "l1_tokens": [[12, 14, 16, 31], [4, 7, 37], [1, 3, 11, 19, 23, 38]],
            "l2_tokens": [[12, 14, 16, 31], [4, 7, 37], [1, 3, 19]], # mapped syns
            "l3_tokens": [
                [12, 13], # Rear_Elevated (12), Wall_Lean (13)
                [4, 5, 14], # Descent_L (4), Descent_R (5), Stand_Down (14)
                [0, 1, 8, 9, 10] # Walk_Sharp_L/R, Walk_L/R, Walk_Fwd
            ],
            "l4_tokens": [
                [6], # Rear
                [7], # Descent
                [0, 1, 2] # Walk_Fwd, Walk_L, Walk_R
            ],
            "max_dur": 85
        },
        {
            "motif_id": "matched_sprint_turn_sprint",
            "motif_title": "Matched Motif 2: Sprint -> Turn -> Sprint",
            "l1_tokens": [[0, 13, 17, 21, 26, 33, 41], [2, 6, 29, 30], [0, 13, 17, 21, 26, 33, 41]],
            "l2_tokens": [[0, 13, 17, 21, 26, 33, 41], [2, 6, 29, 30], [0, 13, 17, 21, 26, 33, 41]],
            "l3_tokens": [
                [11], # Sprint
                [0, 1, 2, 3], # Walk_Sharp_L/R, Turn_L/R
                [11]  # Sprint
            ],
            "l4_tokens": [
                [3], # Sprint
                [4, 5], # Turn_R, Turn_L
                [3]  # Sprint
            ],
            "max_dur": 75
        },
        {
            "motif_id": "matched_sniff_rest_sniff",
            "motif_title": "Matched Motif 3: Sniff -> Rest -> Sniff",
            "l1_tokens": [[8, 9, 10, 15, 18, 20, 24, 25, 36, 39], [32, 43, 49], [8, 9, 10, 15, 18, 20, 24, 25, 36, 39]],
            "l2_tokens": [[8, 10, 15, 18, 20, 24, 25, 36, 39], [32], [8, 10, 15, 18, 20, 24, 25, 36, 39]],
            "l3_tokens": [
                [15, 16, 17], # Sniff_Ground, Sniff_Up, Sniff_Scrunched
                [19], # Rest
                [15, 16, 17]  # Sniff_Ground, Sniff_Up, Sniff_Scrunched
            ],
            "l4_tokens": [
                [8], # Sniff
                [11], # Rest
                [8]  # Sniff
            ],
            "max_dur": 110
        }
    ]

    matched_cands_by_motif = {}
    matched_summary_records = []

    for mq in matched_queries:
        m_id = mq["motif_id"]
        m_title = mq["motif_title"]
        max_dur = mq["max_dur"]
        matched_cands_by_motif[m_id] = {}

        print(f"\nEvaluating Matched Action Chain: {m_title}")

        for lvl in levels:
            lvl_id = lvl["level_id"]
            lvl_name = lvl["name"]
            m_arr = lvl["map_arr"]

            if lvl_id == 1: query_token_sets = mq["l1_tokens"]
            elif lvl_id == 2: query_token_sets = mq["l2_tokens"]
            elif lvl_id == 3: query_token_sets = mq["l3_tokens"]
            elif lvl_id == 4: query_token_sets = mq["l4_tokens"]

            cands = []
            for rk, data in results.items():
                if "latent_state" not in data or "centroid" not in data or rk not in vfi:
                    continue
                cents = data["centroid"]
                heads = data["heading"]
                sylls = data["syllable"]
                mapped = m_arr[sylls]

                mask = np.concatenate([[True], mapped[1:] != mapped[:-1]])
                c_seq = mapped[mask].tolist()
                c_starts = np.where(mask)[0].tolist()
                c_ends = c_starts[1:] + [len(mapped)]
                aid, geno, drug = get_verified_metadata(rk)

                n_stages = len(query_token_sets)
                for i in range(len(c_seq) - n_stages):
                    # Check if stage k matches any token in query_token_sets[k]
                    match = True
                    for k in range(n_stages):
                        if c_seq[i + k] not in query_token_sets[k]:
                            match = False
                            break
                    if match:
                        sf = c_starts[i]
                        ef = c_ends[i + n_stages - 1]
                        bounds = [c_starts[i + k] for k in range(n_stages)] + [ef]
                        durs = np.diff(bounds)

                        if not (all(d >= 2 for d in durs) and all(d <= max_dur for d in durs)):
                            continue
                        if ef >= len(cents) or (ef - sf) < 8:
                            continue

                        step_d = np.linalg.norm(np.diff(cents[sf:ef], axis=0), axis=1)
                        if np.any(step_d > 25.0):
                            continue

                        feat = extract_continuous_features(results, rk, sf, ef)
                        path_len = float(np.sum(step_d))
                        net_disp = float(np.linalg.norm(cents[ef - 1] - cents[sf]))
                        avg_speed = float(np.mean(step_d) * 40.0)
                        dur_sec = float((ef - sf) / 40.0)

                        dh = np.diff(heads[sf:ef])
                        dh = (dh + np.pi) % (2 * np.pi) - np.pi
                        signed_heading = float(np.degrees(np.sum(dh)))

                        cands.append({
                            "motif_id": m_id,
                            "motif_title": m_title,
                            "level_id": lvl_id,
                            "level_name": lvl_name,
                            "recording_key": rk,
                            "animal_id": aid,
                            "genotype": geno,
                            "drug": drug,
                            "start_frame": int(sf),
                            "end_frame": int(ef),
                            "stage_boundaries": [int(b) for b in bounds],
                            "duration_sec": dur_sec,
                            "avg_speed_px_s": avg_speed,
                            "path_length_px": path_len,
                            "net_displacement_px": net_disp,
                            "signed_turn_deg": signed_heading,
                            "feature_vector": feat
                        })

            print(f"  {lvl_name}: N={len(cands)} episodes across {len(set(c['animal_id'] for c in cands))} animals")
            matched_cands_by_motif[m_id][lvl_id] = cands

        # Compute common pooled reference for this matched action chain across all levels
        all_m_cands = []
        for lid in [1, 2, 3, 4]:
            all_m_cands.extend(matched_cands_by_motif[m_id][lid])

        if len(all_m_cands) > 0:
            m_feats = np.array([c["feature_vector"] for c in all_m_cands])
            m_ref = np.mean(m_feats, axis=0)
            m_ref_norm = m_ref / (np.linalg.norm(m_ref) + 1e-8)

            for lid in [1, 2, 3, 4]:
                cands = matched_cands_by_motif[m_id][lid]
                for c in cands:
                    f = c["feature_vector"]
                    fn = f / (np.linalg.norm(f) + 1e-8)
                    c["movement_similarity"] = float(np.dot(fn, m_ref_norm))

                sims = np.array([c["movement_similarity"] for c in cands])
                aids = [c["animal_id"] for c in cands]
                p80 = float(np.percentile(sims, 80)) if len(sims) > 0 else 0.0
                p90 = float(np.percentile(sims, 90)) if len(sims) > 0 else 0.0

                matched_summary_records.append({
                    "Motif": m_title,
                    "Level_ID": lid,
                    "Level_Name": levels[lid - 1]["name"],
                    "Total_Episodes": len(cands),
                    "Total_Animals": len(set(aids)),
                    "P80_Similarity": round(p80, 3),
                    "P90_Similarity": round(p90, 3),
                    "Episodes_Sim_ge_0.70": int(np.sum(sims >= 0.70)),
                    "Animals_Sim_ge_0.70": len(set(aid for aid, s in zip(aids, sims) if s >= 0.70)),
                    "Episodes_Sim_ge_0.75": int(np.sum(sims >= 0.75)),
                    "Animals_Sim_ge_0.75": len(set(aid for aid, s in zip(aids, sims) if s >= 0.75)),
                    "Episodes_Sim_ge_0.80": int(np.sum(sims >= 0.80)),
                    "Animals_Sim_ge_0.80": len(set(aid for aid, s in zip(aids, sims) if s >= 0.80))
                })

    df_matched = pd.DataFrame(matched_summary_records)
    matched_csv = AUDIT_DIR / "matched_action_chains_scaling_summary.csv"
    df_matched.to_csv(matched_csv, index=False)
    print(f"\n[MATCHED] Saved matched action scaling summary to {matched_csv.relative_to(REPO_ROOT)}")
    print(df_matched.to_string())

    # =========================================================================
    # 4. DISCOVER EACH LEVEL'S OWN TOP RECURRENT FAMILIES (Question 5)
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. DISCOVERING EACH LEVEL'S OWN TOP RECURRENT FAMILIES")
    print("=" * 80)

    # For each level, find the top 5 most recurrent 4-grams and 5-grams supported across >=8 animals
    own_family_records = []

    for lvl in levels:
        lvl_id = lvl["level_id"]
        lvl_name = lvl["name"]
        m_arr = lvl["map_arr"]
        token_names = lvl["token_names"]

        # Aggregate collapsed streams
        collapsed_sessions = []
        for rk, data in results.items():
            if "latent_state" not in data or "centroid" not in data or rk not in vfi:
                continue
            sylls = data["syllable"]
            mapped = m_arr[sylls]
            mask = np.concatenate([[True], mapped[1:] != mapped[:-1]])
            c_seq = mapped[mask].tolist()
            aid, geno, drug = get_verified_metadata(rk)
            collapsed_sessions.append((rk, aid, c_seq))

        # Count 4-grams and 5-grams
        for n_gram_len in [4, 5]:
            ngram_counts = {}
            ngram_animals = {}

            for rk, aid, c_seq in collapsed_sessions:
                for i in range(len(c_seq) - n_gram_len + 1):
                    gram = tuple(c_seq[i : i + n_gram_len])
                    ngram_counts[gram] = ngram_counts.get(gram, 0) + 1
                    if gram not in ngram_animals:
                        ngram_animals[gram] = set()
                    ngram_animals[gram].add(aid)

            # Filter for recurrent: >= 20 occurrences, >= 8 animals
            qual = [
                (gram, count, len(ngram_animals[gram]))
                for gram, count in ngram_counts.items()
                if count >= 20 and len(ngram_animals[gram]) >= 8
            ]
            qual.sort(key=lambda x: x[1], reverse=True)

            # Record top 4 distinct families
            for rank, (gram, count, n_anim) in enumerate(qual[:4], 1):
                gram_str = " -> ".join([token_names[t] if t < len(token_names) else f"Syll_{t}" for t in gram])
                own_family_records.append({
                    "Level_ID": lvl_id,
                    "Level_Name": lvl_name,
                    "N_Gram_Length": n_gram_len,
                    "Rank": rank,
                    "Gram_Tokens": gram_str,
                    "Total_Occurrences": count,
                    "Animal_Coverage": n_anim,
                    "Gram_IDs": str(gram)
                })

    df_own = pd.DataFrame(own_family_records)
    own_csv = AUDIT_DIR / "each_level_own_top_recurrent_families.csv"
    df_own.to_csv(own_csv, index=False)
    print(f"[OWN FAMILIES] Saved top recurrent families to {own_csv.relative_to(REPO_ROOT)}")
    print(df_own[["Level_ID", "N_Gram_Length", "Rank", "Total_Occurrences", "Animal_Coverage", "Gram_Tokens"]].head(16).to_string())

    # =========================================================================
    # 5. EXPORT MASTER PER-EPISODE CSV
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. EXPORTING MASTER PER-EPISODE CSV")
    print("=" * 80)

    # Annotate all_theme_candidates with P80 and P90 thresholds per (level, theme)
    for lvl in levels:
        lid = lvl["level_id"]
        for th_id in theme_ids:
            subset = [c for c in all_theme_candidates if c["level_id"] == lid and c["theme_id"] == th_id]
            if len(subset) == 0: continue
            sims = np.array([c["similarity_pooled_ref"] for c in subset])
            p80_val = float(np.percentile(sims, 80))
            p90_val = float(np.percentile(sims, 90))

            for c in subset:
                c["p80_operating_point"] = p80_val
                c["p90_operating_point"] = p90_val
                c["is_p80_band"] = abs(c["similarity_pooled_ref"] - p80_val) <= 0.03
                c["is_p90_band"] = abs(c["similarity_pooled_ref"] - p90_val) <= 0.03

    # Build per-episode dataframe
    clean_episodes = []
    for c in all_theme_candidates:
        clean_episodes.append({
            "level_id": c["level_id"],
            "level_name": c["level_name"],
            "theme_id": c["theme_id"],
            "theme_title": c["theme_title"],
            "sequence_length": c["sequence_length"],
            "target_tokens": c["target_tokens"],
            "recording_key": c["recording_key"],
            "animal_id": c["animal_id"],
            "genotype": c["genotype"],
            "drug": c["drug"],
            "start_frame": c["start_frame"],
            "end_frame": c["end_frame"],
            "duration_sec": round(c["duration_sec"], 3),
            "avg_speed_px_s": round(c["avg_speed_px_s"], 2),
            "path_length_px": round(c["path_length_px"], 2),
            "net_displacement_px": round(c["net_displacement_px"], 2),
            "signed_turn_deg": round(c["signed_turn_deg"], 2),
            "similarity_to_l1_ref": round(c.get("similarity_l1_ref", 0.0), 3),
            "similarity_to_pooled_ref": round(c.get("similarity_pooled_ref", 0.0), 3),
            "is_p80_band": c.get("is_p80_band", False),
            "is_p90_band": c.get("is_p90_band", False)
        })

    df_episodes = pd.DataFrame(clean_episodes)
    episodes_csv = AUDIT_DIR / "master_per_episode_candidate_audit.csv"
    df_episodes.to_csv(episodes_csv, index=False)
    # Also save to the main comparison directory for direct user access
    df_episodes.to_csv(EXP_DIR / "master_per_episode_candidate_audit.csv", index=False)
    print(f"[EPISODES] Saved {len(df_episodes)} candidate episodes to {episodes_csv.relative_to(REPO_ROOT)}")

    # =========================================================================
    # 6. PLOT COMPREHENSIVE REPAIRED CUMULATIVE CURVES (WITH P80 & P90)
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. GENERATING REPAIRED SIMILARITY PLOTS (POOLED REFERENCE + APPLES-TO-APPLES)")
    print("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(19, 10), sharex="col")
    thresholds = np.linspace(0.10, 0.95, 86)
    lvl_colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728"}
    lvl_labels = {
        1: "Level 1: 100 Raw States",
        2: "Level 2: 93 Conservative",
        3: "Level 3: 22 Preserved Asymmetry",
        4: "Level 4: 14 Broad Families"
    }

    theme_meta = [
        {"idx": 0, "id": "theme_A_rear_descent", "title": "Theme A: Wall-Rear & Descent Launch"},
        {"idx": 1, "id": "theme_B_turn_slalom", "title": "Theme B: Sprint & Turning Redirection"},
        {"idx": 2, "id": "theme_C_sensory_rest", "title": "Theme C: Olfactory Inspection & Rest"}
    ]

    for col_idx, tm in enumerate(theme_meta):
        th_id = tm["id"]
        th_title = tm["title"]
        ax_n = axes[0, col_idx]
        ax_anim = axes[1, col_idx]

        ax_n.set_title(th_title, fontsize=13, fontweight="bold", pad=12)

        for lvl_id in [1, 2, 3, 4]:
            cands = [c for c in all_theme_candidates if c["level_id"] == lvl_id and c["theme_id"] == th_id]
            sims = np.array([c["similarity_pooled_ref"] for c in cands])
            aids = [c["animal_id"] for c in cands]

            n_qual = [np.sum(sims >= t) for t in thresholds]
            anim_qual = [len(set(aid for aid, s in zip(aids, sims) if s >= t)) for t in thresholds]

            p80 = float(np.percentile(sims, 80)) if len(sims) > 0 else 0.0
            p90 = float(np.percentile(sims, 90)) if len(sims) > 0 else 0.0

            # Line plot
            ax_n.plot(
                thresholds, n_qual,
                color=lvl_colors[lvl_id],
                linewidth=2.5,
                label=f"{lvl_labels[lvl_id]} (N={len(cands)}, P80={p80:.2f})"
            )
            # Mark P80 (diamond) and P90 (circle)
            n_at_p80 = np.sum(sims >= p80)
            anim_at_p80 = len(set(aid for aid, s in zip(aids, sims) if s >= p80))
            ax_n.scatter([p80], [n_at_p80], color=lvl_colors[lvl_id], marker="D", s=65, zorder=6, edgecolor="black")

            n_at_p90 = np.sum(sims >= p90)
            anim_at_p90 = len(set(aid for aid, s in zip(aids, sims) if s >= p90))
            ax_n.scatter([p90], [n_at_p90], color=lvl_colors[lvl_id], marker="o", s=65, zorder=6, edgecolor="black")

            # Animal coverage
            ax_anim.plot(thresholds, anim_qual, color=lvl_colors[lvl_id], linewidth=2.5)
            ax_anim.scatter([p80], [anim_at_p80], color=lvl_colors[lvl_id], marker="D", s=65, zorder=6, edgecolor="black")
            ax_anim.scatter([p90], [anim_at_p90], color=lvl_colors[lvl_id], marker="o", s=65, zorder=6, edgecolor="black")

        ax_n.set_ylabel("Qualifying Episodes (N)", fontsize=11, fontweight="bold")
        ax_n.set_yscale("log")
        ax_n.set_ylim(bottom=0.8, top=1000)
        ax_n.grid(True, alpha=0.3, linestyle="--")
        ax_n.legend(fontsize=9, loc="upper right")

        ax_anim.set_xlabel("Pooled Movement Similarity Threshold", fontsize=11, fontweight="bold")
        ax_anim.set_ylabel("Distinct Animals (Max 15)", fontsize=11, fontweight="bold")
        ax_anim.set_ylim(-0.5, 16)
        ax_anim.set_yticks([0, 3, 6, 9, 12, 15])
        ax_anim.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("Audited Cumulative Movement Similarity Scaling Across Resolutions (Unbiased Pooled Reference)\nDiamonds = P80 Operating Points | Circles = P90 Operating Points", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    repaired_fig_path = EXP_DIR / "fig_similarity_vs_qualifying_episodes_repaired.png"
    plt.savefig(repaired_fig_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved repaired cumulative plot to {repaired_fig_path.relative_to(REPO_ROOT)}")

    # Plot Matched Action Chains Figure (Apples-to-Apples)
    fig_m, axes_m = plt.subplots(2, 3, figsize=(19, 10), sharex="col")
    for col_idx, mq in enumerate(matched_queries):
        m_id = mq["motif_id"]
        m_title = mq["motif_title"]
        ax_n = axes_m[0, col_idx]
        ax_anim = axes_m[1, col_idx]

        ax_n.set_title(m_title, fontsize=13, fontweight="bold", pad=12)

        for lvl_id in [1, 2, 3, 4]:
            cands = matched_cands_by_motif[m_id][lvl_id]
            sims = np.array([c["movement_similarity"] for c in cands])
            aids = [c["animal_id"] for c in cands]

            n_qual = [np.sum(sims >= t) for t in thresholds]
            anim_qual = [len(set(aid for aid, s in zip(aids, sims) if s >= t)) for t in thresholds]

            p80 = float(np.percentile(sims, 80)) if len(sims) > 0 else 0.0
            p90 = float(np.percentile(sims, 90)) if len(sims) > 0 else 0.0

            ax_n.plot(thresholds, n_qual, color=lvl_colors[lvl_id], linewidth=2.5,
                      label=f"{lvl_labels[lvl_id]} (N={len(cands)}, P80={p80:.2f})")
            n_at_p80 = np.sum(sims >= p80)
            anim_at_p80 = len(set(aid for aid, s in zip(aids, sims) if s >= p80))
            ax_n.scatter([p80], [n_at_p80], color=lvl_colors[lvl_id], marker="D", s=65, zorder=6, edgecolor="black")

            ax_anim.plot(thresholds, anim_qual, color=lvl_colors[lvl_id], linewidth=2.5)
            ax_anim.scatter([p80], [anim_at_p80], color=lvl_colors[lvl_id], marker="D", s=65, zorder=6, edgecolor="black")

        ax_n.set_ylabel("Qualifying Episodes (N)", fontsize=11, fontweight="bold")
        ax_n.set_yscale("log")
        ax_n.set_ylim(bottom=0.8, top=3000)
        ax_n.grid(True, alpha=0.3, linestyle="--")
        ax_n.legend(fontsize=9, loc="upper right")

        ax_anim.set_xlabel("Continuous Movement Similarity Threshold", fontsize=11, fontweight="bold")
        ax_anim.set_ylabel("Distinct Animals (Max 15)", fontsize=11, fontweight="bold")
        ax_anim.set_ylim(-0.5, 16)
        ax_anim.set_yticks([0, 3, 6, 9, 12, 15])
        ax_anim.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("Strict Apples-to-Apples Matched Action Chains Across Resolutions (Identical 3-Step Physical Grammar)\nDiamonds = P80 Operating Points", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    matched_fig_path = EXP_DIR / "fig_matched_action_chains_scaling.png"
    plt.savefig(matched_fig_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved matched action chains plot to {matched_fig_path.relative_to(REPO_ROOT)}")

    # =========================================================================
    # 7. RENDER 12 P80 1x3 WIDESCREEN VIDEOS ACROSS ALL LEVELS
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. RENDERING 12 P80 1x3 WIDESCREEN VIDEOS ACROSS ALL LEVELS (NO PAUSING)")
    print("=" * 80)

    p80_video_metadata = []

    for lvl in levels:
        lid = lvl["level_id"]
        lvl_folder = EXP_DIR / lvl["folder"]
        movies_dir = lvl_folder / "movies"
        thumbs_dir = lvl_folder / "thumbnails"

        for tm in theme_meta:
            th_id = tm["id"]
            th_title = tm["title"]

            subset = [c for c in all_theme_candidates if c["level_id"] == lid and c["theme_id"] == th_id]
            if len(subset) == 0: continue

            sims = np.array([c["similarity_pooled_ref"] for c in subset])
            p80_val = float(np.percentile(sims, 80))

            # Find episodes closest to P80 from 3 distinct animals
            sorted_by_dist = sorted(subset, key=lambda x: abs(x["similarity_pooled_ref"] - p80_val))

            selected_triplet = []
            seen_animals = set()
            for cand in sorted_by_dist:
                if cand["animal_id"] not in seen_animals:
                    selected_triplet.append(cand)
                    seen_animals.add(cand["animal_id"])
                    if len(selected_triplet) == 3:
                        break

            # Fallback if fewer than 3 distinct animals
            if len(selected_triplet) < 3:
                for cand in sorted_by_dist:
                    if cand not in selected_triplet:
                        selected_triplet.append(cand)
                        if len(selected_triplet) == 3:
                            break

            print(f"[{lvl['name']} - {th_id}] P80 Target: {p80_val:.3f}")
            for sc in selected_triplet:
                print(f"  Mouse {sc['animal_id']} ({sc['genotype']}): Sim={sc['similarity_pooled_ref']:.3f}, Dur={sc['duration_sec']:.2f}s, Turn={sc['signed_turn_deg']:.1f}deg")

            # Prepare instances for chain grid movie generator
            p80_instances = []
            for p_idx, sc in enumerate(selected_triplet):
                c_copy = sc.copy()
                c_copy["role"] = "P80_REPRESENTATIVE"
                c_copy["proto_cost"] = float(abs(sc["similarity_pooled_ref"] - p80_val))
                c_copy["kin_dist"] = float(sc["similarity_pooled_ref"])
                c_copy["stage_tokens"] = sc["stage_tokens"]
                c_copy["matched_tokens"] = sc["stage_tokens"]
                p80_instances.append(c_copy)

            mp4_path = movies_dir / f"{th_id}_p80.mp4"
            gif_path = thumbs_dir / f"{th_id}_p80.gif"
            png_thumb = thumbs_dir / f"{th_id}_p80_thumb.png"

            print(f"[{lvl['name']} | {th_id}] Rendering P80 Movie -> {mp4_path.name}")
            print(f"   P80 Target Sim = {p80_val:.3f}, Panels Sim = {[round(p['similarity_pooled_ref'], 3) for p in p80_instances]}")
            print(f"   Animals: {[p['animal_id'] for p in p80_instances]}")

            generate_chain_grid_movie(
                instances=p80_instances,
                output_path=mp4_path,
                resources=res,
                centroid_mode="follow_centroid",
                sync_mode="nothing",
                template_tokens=p80_instances[0]["stage_tokens"],
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

            # Generate Thumbnail & GIF
            cap = cv2.VideoCapture(str(mp4_path))
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
                cv2.imwrite(str(png_thumb), thumb_frame)
            elif gif_frames:
                cv2.imwrite(str(png_thumb), cv2.cvtColor(gif_frames[0], cv2.COLOR_RGB2BGR))

            if gif_frames:
                import imageio
                imageio.mimsave(str(gif_path), gif_frames, fps=12, loop=0)

            p80_video_metadata.append({
                "level_id": lid,
                "level_name": lvl["name"],
                "theme_id": th_id,
                "theme_title": th_title,
                "p80_similarity": round(p80_val, 3),
                "mp4_file": str(mp4_path.relative_to(REPO_ROOT)),
                "gif_file": str(gif_path.relative_to(REPO_ROOT)),
                "animals": [sc["animal_id"] for sc in selected_triplet],
                "sims": [round(sc["similarity_pooled_ref"], 3) for sc in selected_triplet]
            })

    with open(AUDIT_DIR / "p80_video_metadata.json", "w") as f:
        json.dump(p80_video_metadata, f, indent=2)

    print("\n" + "=" * 80)
    print(f"AUDIT & P80 PIPELINE COMPLETE IN {time.time() - t0:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
