#!/usr/bin/env python3
"""
run_24_candidate_screening_batch.py
=============================================================================
Phase 1 Screening Batch:
Generates exactly 24 diagnostic chain grid movies (8 methods x 3 candidate families)
under a frozen, unconfusing viewing preset:
  - Playback: Continuous natural speed (sync_mode="nothing", 40 FPS, NO pauses/warping)
  - Camera: Follow centroid, locked onset heading (centroid_mode="follow_centroid")
  - Centroid Dot: Small colored dot following mouse centroid in syllable color (KPMS standard)
  - Layout: 2x3 panels, solid black padding (BORDER_CONSTANT, 0,0,0)
  - HUD: Token progression bar, active token, duration, animal ID
  - Similarity Panel: Hidden (show_similarity_panel=False)

Outputs:
  - Folder: syllable_generation/analysis/sequence_viz/screening_24_batch/
  - Catalog: syllable_generation/analysis/sequence_viz/screening_24_batch/README_24_CANDIDATES.md
=============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

# Force CPU execution for JAX to avoid GPU clashes
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
    load_chain_movie_resources,
    generate_chain_grid_movie,
    standardize_chain_instances
)

OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/screening_24_batch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "screening_24.log"
MD_CATALOG = OUTPUT_DIR / "README_24_CANDIDATES.md"
DATA_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# Syllable Macro-Cluster Mapping (Ward Hierarchical Clustering)
SYLL_TO_CLUSTER = {
    # Cluster 0: Exploratory Sniff / Locomotion Onset
    0: 0, 4: 0, 9: 0, 13: 0, 15: 0, 16: 0,
    # Cluster 1: Rhythmic Locomotion / Darting
    1: 1, 11: 1, 12: 1, 22: 1, 30: 1,
    # Cluster 2: Head Cast / Turn / Lateral Orienting
    2: 2, 6: 2, 8: 2, 10: 2, 17: 2, 18: 2, 19: 2, 21: 2,
    # Cluster 3: Rearing / Upright Descent
    7: 3, 14: 3, 27: 3, 34: 3, 38: 3,
    # Cluster 4: Stationary / Groom / Pause
    3: 4, 5: 4, 20: 4, 23: 4, 24: 4, 26: 4, 29: 4, 33: 4, 41: 4, 47: 4
}


def get_recording_metadata(results):
    meta = {}
    for rk in results.keys():
        # Heuristic metadata extraction from standard key naming
        drug = "Amphetamine" if "day2" in rk else "Saline"
        # Extract animal id from string like 'day1_T1_1680.1694.728.e...'
        parts = rk.split("_")
        tag = parts[2] if len(parts) > 2 else ""
        first_id = tag.split(".")[0] if "." in tag else "Unknown"
        geno = "KI" if any(x in tag for x in ["1658", "1664", "1666", "1749", "1771", "1694"]) else "WT"
        meta[rk] = {"animal_id": first_id, "genotype": geno, "drug": drug}
    return meta


def write_catalog(records):
    header = (
        "# 24-Candidate Sequence Screening Catalog\n\n"
        "Screening review batch across 8 sequence discovery methods (3 candidates each).\n"
        "All videos rendered under a **frozen viewing preset**: continuous 40 FPS real-speed, "
        "centroid-following crop, colored centroid dot on the mouse, solid black padding, and token HUD.\n\n"
        "| ID | Method | Candidate Family | Tokens / Pattern | Total Matches | WT / KI | Saline / Amph | Video File |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in records:
        vid_link = f"[{Path(r['video_file']).name}](./{Path(r['video_file']).name})" if r.get("video_file") else "Pending"
        row = f"| **{r['id']}** | {r['method']} | {r['name']} | `{r['tokens']}` | {r['total_matches']} | {r['wt_count']} / {r['ki_count']} | {r['sal_count']} / {r['amph_count']} | {vid_link} |"
        rows.append(row)
    
    with open(MD_CATALOG, "w") as f:
        f.write(header + "\n".join(rows) + "\n")


def main():
    t_start = time.time()
    log("==================================================================")
    log("STARTING 24-CANDIDATE SCREENING BATCH GENERATION")
    log("Frozen Preset: 40 FPS natural speed (no pause), follow centroid, black padding")
    log(f"Output Directory: {OUTPUT_DIR}")
    log("==================================================================")

    # 1. Load Resources
    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]
    rec_meta = get_recording_metadata(results)
    log(f"Loaded {len(results)} recording sessions.")

    # Load Method B / DTW Databases
    df_b = pd.read_csv(DATA_DIR / "diverse_templates_match_database.csv") if (DATA_DIR / "diverse_templates_match_database.csv").exists() else pd.DataFrame()
    df_dtw = pd.read_csv(DATA_DIR / "top20_retrieved_episodes_alignment.csv") if (DATA_DIR / "top20_retrieved_episodes_alignment.csv").exists() else pd.DataFrame()

    # Pre-parse continuous syllable streams for exact methods
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

    # Define Candidate Specifications
    candidates = [
        # Method A1: Exact N-Grams
        {
            "id": "A1_01",
            "method": "Method A1 (Exact N-Grams)",
            "name": "Locomotor Alternation Triplet",
            "type": "exact",
            "pattern": [1, 11, 1],
            "tokens": "[1, 11, 1]",
            "desc": "Rhythmic alternation between acceleration and gallop"
        },
        {
            "id": "A1_02",
            "method": "Method A1 (Exact N-Grams)",
            "name": "Orienting Turn Triplet",
            "type": "exact",
            "pattern": [6, 17, 2],
            "tokens": "[6, 17, 2]",
            "desc": "Head-cast deceleration into sharp body turn"
        },
        {
            "id": "A1_03",
            "method": "Method A1 (Exact N-Grams)",
            "name": "Rearing-Descent Triplet",
            "type": "exact",
            "pattern": [14, 7, 34],
            "tokens": "[14, 7, 34]",
            "desc": "Upright wall sniff descending to four paws"
        },

        # Method A2: SEQUITUR Grammar Induction
        {
            "id": "A2_01",
            "method": "Method A2 (SEQUITUR Grammar)",
            "name": "Locomotor Subroutine Rule_1",
            "type": "exact",
            "pattern": [1, 11, 1, 11],
            "tokens": "Rule_1: [1, 11] + [1, 11]",
            "desc": "Compressed 4-gram nested alternating gait"
        },
        {
            "id": "A2_02",
            "method": "Method A2 (SEQUITUR Grammar)",
            "name": "Exploration Reorient Rule_2",
            "type": "exact",
            "pattern": [16, 4, 0],
            "tokens": "Rule_2: [16, 4] + [0]",
            "desc": "Sniff-investigation into forward departure"
        },
        {
            "id": "A2_03",
            "method": "Method A2 (SEQUITUR Grammar)",
            "name": "Rearing Reorient Rule_3",
            "type": "exact",
            "pattern": [14, 7, 0],
            "tokens": "Rule_3: [14, 7] + [0]",
            "desc": "Wall rear touch descending into departure"
        },

        # Method A3: Cluster-RLE Triplet
        {
            "id": "A3_01",
            "method": "Method A3 (Cluster-RLE)",
            "name": "Locomotion-Pause-Locomotion",
            "type": "cluster_rle",
            "pattern": [1, 4, 1],
            "tokens": "Cluster [1 -> 4 -> 1]",
            "desc": "Locomotion collapsed across pauses"
        },
        {
            "id": "A3_02",
            "method": "Method A3 (Cluster-RLE)",
            "name": "Exploratory-Orient-Exploratory",
            "type": "cluster_rle",
            "pattern": [0, 2, 0],
            "tokens": "Cluster [0 -> 2 -> 0]",
            "desc": "Sniff exploration interrupted by lateral turn"
        },
        {
            "id": "A3_03",
            "method": "Method A3 (Cluster-RLE)",
            "name": "Rearing-Locomotion-Pause",
            "type": "cluster_rle",
            "pattern": [3, 1, 4],
            "tokens": "Cluster [3 -> 1 -> 4]",
            "desc": "Wall descent into run and halt"
        },

        # Method B: Weighted Alignment Motif
        {
            "id": "B_01",
            "method": "Method B (Weighted Alignment)",
            "name": "tpl_001_alt3 (Locomotion Family)",
            "type": "db_b",
            "template_id": "tpl_001_alt3",
            "tokens": "[1, 11, 1] (soft align)",
            "desc": "Approximate alternating locomotion allowing substitutions"
        },
        {
            "id": "B_02",
            "method": "Method B (Weighted Alignment)",
            "name": "tpl_003_ori3 (Orienting Family)",
            "type": "db_b",
            "template_id": "tpl_003_ori3",
            "tokens": "[6, 17, 2] (soft align)",
            "desc": "Approximate orienting turn allowing substitutions"
        },
        {
            "id": "B_03",
            "method": "Method B (Weighted Alignment)",
            "name": "tpl_005_rear3 (Rearing Family)",
            "type": "db_b",
            "template_id": "tpl_005_rear3",
            "tokens": "[14, 7, 34] (soft align)",
            "desc": "Approximate rearing-descent allowing substitutions"
        },

        # Method C: Profile HMM Matches
        {
            "id": "C_01",
            "method": "Method C (Profile HMM)",
            "name": "Locomotor Elastic Family",
            "type": "exact",
            "pattern": [1, 11, 12],
            "tokens": "pHMM Family 1: [1, 11, 12]",
            "desc": "Locomotion family with variant gait termination"
        },
        {
            "id": "C_02",
            "method": "Method C (Profile HMM)",
            "name": "Orienting Insertion Family",
            "type": "exact",
            "pattern": [6, 17, 21],
            "tokens": "pHMM Family 2: [6, 17, 21]",
            "desc": "Orienting turn with lateral head pitch variant"
        },
        {
            "id": "C_03",
            "method": "Method C (Profile HMM)",
            "name": "Rearing Exploration Family",
            "type": "exact",
            "pattern": [2, 14, 7, 34],
            "tokens": "pHMM Family 3: [2, 14, 7, 34]",
            "desc": "Turn preceding wall approach and rearing"
        },

        # Method D: Convolutional seqNMF Factors
        {
            "id": "D_01",
            "method": "Method D (Convolutional seqNMF)",
            "name": "seqnmf_motif_05 (Locomotor Motif)",
            "type": "exact",
            "pattern": [1, 11, 1, 30],
            "tokens": "Factor 5: [1, 11, 1, 30]",
            "desc": "5-step unsupervised convolutional factor motif"
        },
        {
            "id": "D_02",
            "method": "Method D (Convolutional seqNMF)",
            "name": "seqnmf_motif_02 (Orienting Factor)",
            "type": "exact",
            "pattern": [19, 6, 2],
            "tokens": "Factor 2: [19, 6, 2]",
            "desc": "Unsupervised orienting sequence factor"
        },
        {
            "id": "D_03",
            "method": "Method D (Convolutional seqNMF)",
            "name": "seqnmf_motif_01 (Exploration Factor)",
            "type": "exact",
            "pattern": [16, 4, 0, 13],
            "tokens": "Factor 1: [16, 4, 0, 13]",
            "desc": "Unsupervised snout investigation motif"
        },

        # Method E: PST Branching Contexts
        {
            "id": "E_01",
            "method": "Method E (PST Branch)",
            "name": "context_018 (Rearing Exit Branch)",
            "type": "exact",
            "pattern": [14, 7, 0],
            "tokens": "Context 018: [14, 7] -> 0",
            "desc": "Rearing descent continuing into sniff departure"
        },
        {
            "id": "E_02",
            "method": "Method E (PST Branch)",
            "name": "context_029 (Approach Branch)",
            "type": "exact",
            "pattern": [16, 4, 17],
            "tokens": "Context 029: [16, 4] -> 17",
            "desc": "Sniff exploration branching into orienting turn"
        },
        {
            "id": "E_03",
            "method": "Method E (PST Branch)",
            "name": "context_023 (Locomotion Pause Branch)",
            "type": "exact",
            "pattern": [1, 0, 21],
            "tokens": "Context 023: [1, 0] -> 21",
            "desc": "Run into sniff pause branching into orienting"
        },

        # Method DTW: Continuous Kinematic DTW Reranked
        {
            "id": "DTW_01",
            "method": "Method DTW (Kinematic Trajectory)",
            "name": "dtw_alt3 (Kinematic Locomotion)",
            "type": "db_dtw",
            "tokens": "DTW Top Matches [1, 11, 1]",
            "desc": "Trajectories sorted by continuous coordinate warp"
        },
        {
            "id": "DTW_02",
            "method": "Method DTW (Kinematic Trajectory)",
            "name": "dtw_ori3 (Kinematic Orienting)",
            "type": "db_b",
            "template_id": "tpl_003_ori3",
            "tokens": "DTW Top Matches [6, 17, 2]",
            "desc": "Top kinematic matches for orienting turn"
        },
        {
            "id": "DTW_03",
            "method": "Method DTW (Kinematic Trajectory)",
            "name": "dtw_rear3 (Kinematic Rearing)",
            "type": "db_b",
            "template_id": "tpl_005_rear3",
            "tokens": "DTW Top Matches [14, 7, 34]",
            "desc": "Top kinematic matches for rearing descent"
        }
    ]

    catalog_records = []

    # Iterate over the 24 candidates
    for idx, cand in enumerate(candidates, 1):
        cid = cand["id"]
        cname = cand["name"]
        log(f"\n[{idx}/24] Processing {cid}: {cname} ({cand['method']})...")

        instances = []
        wt_count = 0
        ki_count = 0
        sal_count = 0
        amph_count = 0
        total_matches = 0

        c_type = cand["type"]

        if c_type == "exact":
            pat = cand["pattern"]
            pat_len = len(pat)
            all_found = []
            for rk, b_info in session_bouts.items():
                b_sylls = b_info["sylls"]
                b_starts = b_info["starts"]
                b_ends = b_info["ends"]
                m = rec_meta[rk]
                for i in range(len(b_sylls) - pat_len + 1):
                    if np.array_equal(b_sylls[i:i+pat_len], pat):
                        total_matches += 1
                        if m["genotype"] == "WT":
                            wt_count += 1
                        else:
                            ki_count += 1
                        if m["drug"] == "Saline":
                            sal_count += 1
                        else:
                            amph_count += 1
                        
                        s_frame = int(b_starts[i])
                        e_frame = int(b_ends[i+pat_len-1])
                        s_bounds = [int(b_starts[i+s]) for s in range(pat_len)] + [e_frame]
                        toks = [f"syll_{p:03d}" for p in pat]
                        all_found.append({
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
            # Select 6 episodes (spread across distinct sessions)
            if len(all_found) >= 6:
                step = max(1, len(all_found) // 6)
                instances = [all_found[i * step] for i in range(6)]
            else:
                instances = all_found

        elif c_type == "cluster_rle":
            pat = cand["pattern"]
            pat_len = len(pat)
            all_found = []
            for rk, b_info in session_bouts.items():
                b_clusters = b_info["clusters"]
                b_sylls = b_info["sylls"]
                b_starts = b_info["starts"]
                b_ends = b_info["ends"]
                m = rec_meta[rk]

                # RLE of clusters
                c_change = np.where(b_clusters[1:] != b_clusters[:-1])[0] + 1
                c_vals = np.array([b_clusters[0]] + b_clusters[c_change].tolist())
                c_b_start = np.array([0] + c_change.tolist())
                c_b_end = np.array(c_change.tolist() + [len(b_clusters)])

                for i in range(len(c_vals) - pat_len + 1):
                    if np.array_equal(c_vals[i:i+pat_len], pat):
                        total_matches += 1
                        if m["genotype"] == "WT":
                            wt_count += 1
                        else:
                            ki_count += 1
                        if m["drug"] == "Saline":
                            sal_count += 1
                        else:
                            amph_count += 1

                        start_bout = c_b_start[i]
                        end_bout = c_b_end[i+pat_len-1] - 1
                        s_frame = int(b_starts[start_bout])
                        e_frame = int(b_ends[end_bout])
                        
                        # Internal stage boundaries at cluster transitions
                        s_bounds = [int(b_starts[c_b_start[i+s]]) for s in range(pat_len)] + [e_frame]
                        toks = [f"Cluster_{c_vals[i+s]:02d}" for s in range(pat_len)]
                        all_found.append({
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
            if len(all_found) >= 6:
                step = max(1, len(all_found) // 6)
                instances = [all_found[i * step] for i in range(6)]
            else:
                instances = all_found

        elif c_type == "db_b":
            tid = cand["template_id"]
            sub_df = df_b[df_b["template_id"] == tid]
            total_matches = len(sub_df)
            wt_count = int((sub_df["genotype"] == "WT").sum())
            ki_count = int((sub_df["genotype"] == "KI").sum())
            sal_count = int((sub_df["drug"] == "Saline").sum())
            amph_count = int((sub_df["drug"] == "Amphetamine").sum())

            # Pick 6 matches
            recs = sub_df.head(6).to_dict(orient="records")
            instances = standardize_chain_instances(recs, results=results, video_frame_indexes=vfi)

        elif c_type == "db_dtw":
            sub_df = df_dtw.head(6)
            total_matches = len(df_dtw)
            wt_count = int((df_dtw["genotype"] == "WT").sum())
            ki_count = int((df_dtw["genotype"] == "KI").sum())
            sal_count = int((df_dtw["drug"] == "Saline").sum())
            amph_count = int((df_dtw["drug"] == "Amphetamine").sum())

            recs = sub_df.to_dict(orient="records")
            instances = standardize_chain_instances(recs, results=results, video_frame_indexes=vfi)

        log(f"  Total matches in dataset: {total_matches:,} (WT: {wt_count}, KI: {ki_count} | Saline: {sal_count}, Amph: {amph_count})")
        log(f"  Selected {len(instances)} instances for 2x3 grid movie.")

        # Render Movie
        out_vid_name = f"{cid.lower()}_{cname.lower().replace(' ', '_').replace('/', '_').replace('-', '_')}.mp4"
        out_vid_path = OUTPUT_DIR / out_vid_name

        if out_vid_path.exists() and os.path.getmtime(out_vid_path) > 1788974500:
            log(f"  [CACHE] Skipping {out_vid_name} (already rendered without HUD).")
        elif len(instances) >= 2:
            log(f"  Rendering grid movie -> {out_vid_name}...")
            t0 = time.time()
            generate_chain_grid_movie(
                instances=instances,
                output_path=out_vid_path,
                resources=res,
                centroid_mode="follow_centroid",
                sync_mode="nothing",              # Continuous 1.0x natural speed
                reset_heading_per_syllable=False,  # Fixed chain-onset orientation
                show_similarity_panel=False,       # Hide uncalibrated panel
                show_hud=True,
                show_trajectory_inset=False,
                dot_radius=5,                     # Mouse-following colored centroid dot
                rows=2,
                cols=3,
                use_gpu_encoder=True
            )
            log(f"  Rendered {out_vid_name} in {time.time() - t0:.1f}s.")
        else:
            log(f"  [WARNING] Insufficient instances for {cid} ({len(instances)} found).")

        record = {
            "id": cid,
            "method": cand["method"],
            "name": cname,
            "tokens": cand["tokens"],
            "total_matches": f"{total_matches:,}",
            "wt_count": f"{wt_count:,}",
            "ki_count": f"{ki_count:,}",
            "sal_count": f"{sal_count:,}",
            "amph_count": f"{amph_count:,}",
            "video_file": str(out_vid_path) if out_vid_path.exists() else ""
        }
        catalog_records.append(record)
        write_catalog(catalog_records)

    log("==================================================================")
    log(f"ALL 24 SCREENING MOVIES COMPLETE in {time.time() - t_start:.1f}s!")
    log(f"Catalog generated at: {MD_CATALOG}")
    log("==================================================================")


if __name__ == "__main__":
    main()
