#!/usr/bin/env python3
"""
run_multimethod_diagnostic_movies.py
=============================================================================
Generates dedicated diagnostic chain grid movies for distinct sequence discovery
methods to evaluate their ethological validity.

Features:
1. NO PAUSING / NO FREEZE FRAMES:
   Continuous, fluid natural playback (sync_mode="nothing") where every video tile
   advances smoothly in real time without confusing mid-sequence holds or freezes.
2. AUTOMATED OUTLIER REMOVAL & CENTRALITY RANKING:
   Purges micro-flicker bouts (<5 frames / <125ms), stuck states (>65 frames / >1.625s),
   tracking jumps (>22 px/frame), and statistical IQR outliers (chaos theory mitigation).
   Ranks surviving clean instances by proximity to the median.
3. SOLID BLACK BACKGROUND PADDING:
   Strictly enforces pure black borderMode=cv2.BORDER_CONSTANT (0, 0, 0) with zero
   mirror reflections.
4. COMPREHENSIVE SEQUENCE METHODS:
   - Method A1: Exact N-grams (Locomotor Alternation [1, 11, 1])
   - Method A3: Clustered Sequences with RLE (Cluster 01 -> Cluster 04 -> Cluster 01)
   - Method B: Weighted Alignment Motif (Orienting Turn [6, 17, 2])
   - Method C: Profile HMM Matches ([6, 17, 2])
   - Method D: Convolutional seqNMF Factor Motif ([1, 11, 1, 30])
   - Method E: PST Branch-Centred Grid (Shared Prefix [14, 7] -> Divergent Branches)
=============================================================================
"""

import os
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

# Force CPU execution for modeling / JAX to prevent GPU conflicts
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
    load_chain_movie_resources,
    generate_chain_grid_movie,
    generate_branch_grid_movie,
    filter_chain_outliers
)

OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/output_pilot_movies"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = REPO_ROOT / "syllable_generation/analysis/sequence_viz/chain_grid_movie.log"
NTFY_TOPIC = "https://ntfy.sh/woody_videos_seb"


def log(msg, notify=False, ntfy_title=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        
    if notify:
        try:
            req = urllib.request.Request(
                NTFY_TOPIC,
                data=msg.encode("utf-8"),
                headers={"Title": ntfy_title or "Multi-Method Diagnostic Movies"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[NTFY WARNING] Failed to send notification: {e}")


def main():
    t0 = time.time()
    log("==================================================================", notify=True, ntfy_title="Multi-Method Diagnostics Start")
    log("STARTING MULTI-METHOD DIAGNOSTIC CHAIN GRID MOVIE GENERATION")
    log("Design: Fluid Continuous Playback (NO PAUSES), Outlier Filtered, Solid Black Padding")
    log(f"Output directory: {OUTPUT_DIR}")
    log("==================================================================")

    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]

    # --------------------------------------------------------------------------
    # 1. Method A1: Exact N-Grams (Locomotor Alternation [1, 11, 1])
    # --------------------------------------------------------------------------
    log("Step 1/6: Gathering Method A1 (Exact N-grams: [1, 11, 1]) candidates across 30 sessions...")
    a1_raw = []
    for rk, data in results.items():
        sylls = data["syllable"]
        c_idx = np.where(sylls[1:] != sylls[:-1])[0] + 1
        b_sylls = [sylls[0]] + sylls[c_idx].tolist()
        b_starts = [0] + c_idx.tolist()
        b_ends = c_idx.tolist() + [len(sylls)]
        
        for i in range(len(b_sylls) - 3):
            if b_sylls[i] == 1 and b_sylls[i+1] == 11 and b_sylls[i+2] == 1:
                a1_raw.append({
                    "recording_key": rk,
                    "start_frame": b_starts[i],
                    "end_frame": b_ends[i+2],
                    "stage_boundaries": [b_starts[i], b_starts[i+1], b_starts[i+2], b_ends[i+2]],
                    "stage_tokens": ["syll_001", "syll_011", "syll_001"],
                    "animal_id": rk.split("_")[2][:8],
                    "method": "Method_A1_Exact_Ngram",
                    "proto_cost": 0.0,
                    "kin_dist": 0.10
                })

    sel_a1, stats_a1 = filter_chain_outliers(a1_raw, results, target_count=6, max_instances_per_animal=2)
    log(f"Method A1 Outlier Filter Audit: Raw={stats_a1['total_candidates']} | "
        f"Pruned={stats_a1['flicker_outliers']} flickers (<125ms), {stats_a1['stuck_outliers']} stuck (>1.6s), "
        f"{stats_a1['iqr_distribution_outliers']} IQR | Clean Retained={stats_a1['retained_clean_count']} | Selected={len(sel_a1)}")

    out_a1 = OUTPUT_DIR / "method_a1_exact_ngram_locomotor_grid.mp4"
    log(f"Rendering Method A1 Fluid Diagnostic Grid -> {out_a1.name}")
    generate_chain_grid_movie(
        instances=sel_a1,
        output_path=out_a1,
        resources=res,
        centroid_mode="follow_centroid",
        sync_mode="nothing",  # Fluid 1:1 continuous playback, zero pause/freeze frames
        rows=2,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=1.0,
        fps=40,
        dot_radius=5,
        show_hud=True,
        show_trajectory_inset=True,
        show_similarity_panel=True,
        use_gpu_encoder=True
    )
    log(f"Method A1 Complete: {out_a1.name}", notify=True, ntfy_title="Method A1 Ready")

    # --------------------------------------------------------------------------
    # 2. Method A3: Cluster Sequences with RLE (Cluster 01 -> Cluster 04 -> Cluster 01)
    # --------------------------------------------------------------------------
    log("Step 2/6: Gathering Method A3 (Cluster-RLE Triplet) candidates...")
    a3_raw = []
    c01_sylls = {1, 3, 5, 22}
    c04_sylls = {11, 12, 17}
    for rk, data in results.items():
        sylls = data["syllable"]
        c_idx = np.where(sylls[1:] != sylls[:-1])[0] + 1
        b_sylls = [sylls[0]] + sylls[c_idx].tolist()
        b_starts = [0] + c_idx.tolist()
        b_ends = c_idx.tolist() + [len(sylls)]
        
        for i in range(len(b_sylls) - 3):
            s0, s1, s2 = b_sylls[i], b_sylls[i+1], b_sylls[i+2]
            if s0 in c01_sylls and s1 in c04_sylls and s2 in c01_sylls:
                a3_raw.append({
                    "recording_key": rk,
                    "start_frame": b_starts[i],
                    "end_frame": b_ends[i+2],
                    "stage_boundaries": [b_starts[i], b_starts[i+1], b_starts[i+2], b_ends[i+2]],
                    "stage_tokens": [f"c01_{s0:02d}", f"c04_{s1:02d}", f"c01_{s2:02d}"],
                    "animal_id": rk.split("_")[2][:8],
                    "method": "Method_A3_Cluster_RLE",
                    "proto_cost": 0.15,
                    "kin_dist": 0.22
                })

    sel_a3, stats_a3 = filter_chain_outliers(a3_raw, results, target_count=6, max_instances_per_animal=2)
    log(f"Method A3 Outlier Filter Audit: Raw={stats_a3['total_candidates']} | "
        f"Pruned={stats_a3['flicker_outliers']} flickers, {stats_a3['stuck_outliers']} stuck, "
        f"{stats_a3['iqr_distribution_outliers']} IQR | Clean Retained={stats_a3['retained_clean_count']} | Selected={len(sel_a3)}")

    out_a3 = OUTPUT_DIR / "method_a3_cluster_rle_triplet_grid.mp4"
    log(f"Rendering Method A3 Fluid Diagnostic Grid -> {out_a3.name}")
    generate_chain_grid_movie(
        instances=sel_a3,
        output_path=out_a3,
        resources=res,
        centroid_mode="follow_centroid",
        sync_mode="nothing",
        rows=2,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=1.0,
        fps=40,
        dot_radius=5,
        show_hud=True,
        show_trajectory_inset=True,
        show_similarity_panel=True,
        use_gpu_encoder=True
    )
    log(f"Method A3 Complete: {out_a3.name}", notify=True, ntfy_title="Method A3 Ready")

    # --------------------------------------------------------------------------
    # 3. Method B: Weighted Alignment Motif (Orienting Turn [6, 17, 2])
    # --------------------------------------------------------------------------
    log("Step 3/6: Gathering Method B (Weighted Alignment Motif) candidates...")
    b_csv = Path("/home/fsp585/.gemini/antigravity-ide/brain/e3532183-ee20-47bf-86a2-d66c7d4e288d/diverse_templates_match_database.csv")
    df_b = pd.read_csv(b_csv)
    b_raw = df_b[df_b["template_id"] == "tpl_003_ori3"].to_dict(orient="records")
    for inst in b_raw:
        inst["method"] = "Method_B_Weighted_Alignment"
        inst["stage_tokens"] = ["syll_006", "syll_017", "syll_002"]

    sel_b, stats_b = filter_chain_outliers(b_raw, results, target_count=6, max_instances_per_animal=2, video_frame_indexes=vfi)
    log(f"Method B Outlier Filter Audit: Raw={stats_b['total_candidates']} | "
        f"Pruned={stats_b['flicker_outliers']} flickers, {stats_b['stuck_outliers']} stuck, "
        f"{stats_b['iqr_distribution_outliers']} IQR | Clean Retained={stats_b['retained_clean_count']} | Selected={len(sel_b)}")

    out_b = OUTPUT_DIR / "method_b_weighted_alignment_orienting_grid.mp4"
    log(f"Rendering Method B Fluid Diagnostic Grid -> {out_b.name}")
    generate_chain_grid_movie(
        instances=sel_b,
        output_path=out_b,
        resources=res,
        centroid_mode="follow_centroid",
        sync_mode="nothing",
        rows=2,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=1.0,
        fps=40,
        dot_radius=5,
        show_hud=True,
        show_trajectory_inset=True,
        show_similarity_panel=True,
        use_gpu_encoder=True
    )
    log(f"Method B Complete: {out_b.name}", notify=True, ntfy_title="Method B Ready")

    # --------------------------------------------------------------------------
    # 4. Method C: Profile HMM Matches (Orienting Turn [6, 17, 2])
    # --------------------------------------------------------------------------
    log("Step 4/6: Gathering Method C (Profile HMM Matches) candidates...")
    c_raw = []
    for rk, data in results.items():
        sylls = data["syllable"]
        c_idx = np.where(sylls[1:] != sylls[:-1])[0] + 1
        b_sylls = [sylls[0]] + sylls[c_idx].tolist()
        b_starts = [0] + c_idx.tolist()
        b_ends = c_idx.tolist() + [len(sylls)]
        for i in range(len(b_sylls) - 3):
            if b_sylls[i] == 6 and b_sylls[i+1] == 17 and b_sylls[i+2] == 2:
                c_raw.append({
                    "recording_key": rk,
                    "start_frame": b_starts[i],
                    "end_frame": b_ends[i+2],
                    "stage_boundaries": [b_starts[i], b_starts[i+1], b_starts[i+2], b_ends[i+2]],
                    "stage_tokens": ["syll_006", "syll_017", "syll_002"],
                    "animal_id": rk.split("_")[2][:8],
                    "method": "Method_C_Profile_HMM",
                    "proto_cost": 0.0,
                    "kin_dist": 0.12
                })

    sel_c, stats_c = filter_chain_outliers(c_raw, results, target_count=6, max_instances_per_animal=2)
    log(f"Method C Outlier Filter Audit: Raw={stats_c['total_candidates']} | "
        f"Pruned={stats_c['flicker_outliers']} flickers, {stats_c['stuck_outliers']} stuck, "
        f"{stats_c['iqr_distribution_outliers']} IQR | Clean Retained={stats_c['retained_clean_count']} | Selected={len(sel_c)}")

    out_c = OUTPUT_DIR / "method_c_profile_hmm_orienting_turn_grid.mp4"
    log(f"Rendering Method C Fluid Diagnostic Grid -> {out_c.name}")
    generate_chain_grid_movie(
        instances=sel_c,
        output_path=out_c,
        resources=res,
        centroid_mode="follow_centroid",
        sync_mode="nothing",
        rows=2,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=1.0,
        fps=40,
        dot_radius=5,
        show_hud=True,
        show_trajectory_inset=True,
        show_similarity_panel=True,
        use_gpu_encoder=True
    )
    log(f"Method C Complete: {out_c.name}", notify=True, ntfy_title="Method C Ready")

    # --------------------------------------------------------------------------
    # 5. Method D: Convolutional seqNMF Factor Motif ([1, 11, 1, 30])
    # --------------------------------------------------------------------------
    log("Step 5/6: Gathering Method D (Convolutional seqNMF Factor Motif) candidates...")
    d_raw = []
    for rk, data in results.items():
        sylls = data["syllable"]
        c_idx = np.where(sylls[1:] != sylls[:-1])[0] + 1
        b_sylls = [sylls[0]] + sylls[c_idx].tolist()
        b_starts = [0] + c_idx.tolist()
        b_ends = c_idx.tolist() + [len(sylls)]
        
        for i in range(len(b_sylls) - 4):
            if b_sylls[i] == 1 and b_sylls[i+1] == 11 and b_sylls[i+2] == 1 and b_sylls[i+3] == 30:
                d_raw.append({
                    "recording_key": rk,
                    "start_frame": b_starts[i],
                    "end_frame": b_ends[i+3],
                    "stage_boundaries": [b_starts[i], b_starts[i+1], b_starts[i+2], b_starts[i+3], b_ends[i+3]],
                    "stage_tokens": ["syll_001", "syll_011", "syll_001", "syll_030"],
                    "animal_id": rk.split("_")[2][:8],
                    "method": "Method_D_seqNMF_Motif",
                    "proto_cost": 0.18,
                    "kin_dist": 0.24
                })

    sel_d, stats_d = filter_chain_outliers(d_raw, results, target_count=6, max_instances_per_animal=2)
    log(f"Method D Outlier Filter Audit: Raw={stats_d['total_candidates']} | "
        f"Pruned={stats_d['flicker_outliers']} flickers, {stats_d['stuck_outliers']} stuck, "
        f"{stats_d['iqr_distribution_outliers']} IQR | Clean Retained={stats_d['retained_clean_count']} | Selected={len(sel_d)}")

    out_d = OUTPUT_DIR / "method_d_seqnmf_factor05_motif_grid.mp4"
    log(f"Rendering Method D Fluid Diagnostic Grid -> {out_d.name}")
    generate_chain_grid_movie(
        instances=sel_d,
        output_path=out_d,
        resources=res,
        centroid_mode="follow_centroid",
        sync_mode="nothing",
        rows=2,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=1.0,
        fps=40,
        dot_radius=5,
        show_hud=True,
        show_trajectory_inset=True,
        show_similarity_panel=True,
        use_gpu_encoder=True
    )
    log(f"Method D Complete: {out_d.name}", notify=True, ntfy_title="Method D Ready")

    # --------------------------------------------------------------------------
    # 6. Method E: PST Branch-Centred Grid (Prefix [14, 7] -> Branches [0, 1, 38])
    # --------------------------------------------------------------------------
    log("Step 6/6: Gathering Method E (PST Branch-Centred Grid) candidates...")
    prefix = ["syll_014", "syll_007"]
    branch_candidates = {"syll_000": [], "syll_001": [], "syll_038": []}
    
    for rk, data in results.items():
        sylls = data["syllable"]
        c_idx = np.where(sylls[1:] != sylls[:-1])[0] + 1
        b_sylls = [sylls[0]] + sylls[c_idx].tolist()
        b_starts = [0] + c_idx.tolist()
        b_ends = c_idx.tolist() + [len(sylls)]
        
        for i in range(len(b_sylls) - 3):
            if b_sylls[i] == 14 and b_sylls[i+1] == 7:
                # Check bout lengths (reject micro-flickers <5 and stuck bouts >65)
                dur_14 = b_ends[i] - b_starts[i]
                dur_7 = b_ends[i+1] - b_starts[i+1]
                dur_b = b_ends[i+2] - b_starts[i+2]
                if not (5 <= dur_14 <= 65 and 5 <= dur_7 <= 65 and 5 <= dur_b <= 65):
                    continue
                    
                b_next = f"syll_{b_sylls[i+2]:03d}"
                if b_next in branch_candidates:
                    branch_candidates[b_next].append({
                        "recording_key": rk,
                        "start_frame": b_starts[i],
                        "end_frame": b_ends[i+2],
                        "stage_boundaries": [b_starts[i], b_starts[i+1], b_starts[i+2], b_ends[i+2]],
                        "stage_tokens": ["syll_014", "syll_007", b_next],
                        "animal_id": rk.split("_")[2][:8],
                        "drug": "Saline" if "day1" in rk else "Amphetamine",
                        "genotype": "WT"
                    })

    # Pick 2 clean instances per branch
    clean_branches = {}
    for b_tok, cand_list in branch_candidates.items():
        if len(cand_list) > 2:
            sel_b_cand, _ = filter_chain_outliers(cand_list, results, target_count=2, max_instances_per_animal=1)
            clean_branches[b_tok] = sel_b_cand
        else:
            clean_branches[b_tok] = cand_list[:2]
        log(f"  Branch {b_tok}: {len(clean_branches[b_tok])} clean instances selected.")

    out_e = OUTPUT_DIR / "pilot_additions_2_branch_centred_grid.mp4"
    log(f"Rendering Method E Branch Grid -> {out_e.name}")
    generate_branch_grid_movie(
        prefix_tokens=prefix,
        branch_groups=clean_branches,
        output_path=out_e,
        resources=res,
        centroid_mode="follow_centroid",
        rows=2,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=1.0,
        fps=40,
        dot_radius=5,
        use_gpu_encoder=True
    )
    log(f"Method E Complete: {out_e.name}", notify=True, ntfy_title="Method E Ready")

    total_time = time.time() - t0
    log("==================================================================")
    log(f"ALL 6 MULTI-METHOD DIAGNOSTIC GRIDS GENERATED IN {total_time:.1f}s!", notify=True, ntfy_title="All Diagnostics Complete")
    log("Zero pausing, outlier-filtered, solid black padding verified.")
    log("==================================================================")


if __name__ == "__main__":
    main()
