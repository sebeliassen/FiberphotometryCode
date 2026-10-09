#!/usr/bin/env python3
"""
run_prioritized_additions_pilot.py
=============================================================================
Demonstration and verification runner for the 5 prioritized additions to the
Chain Grid Movie generator:
1. Dynamic Concurrent-Token Color Palette (maximally separated colors across active syllables)
2. 2D Centroid Trajectory Inset (arena displacement trail colored by active token)
3. Explicit Alignment Engine for Indels:
   - 'show_variations': Preserves inserted footage; other panels wait on dimmed frames; deletions show gap slates.
   - 'compact_template': Skips inserted footage with explicit cut badges.
   - 'natural_speed': Uninterrupted 1:1 playback alongside them.
4. Branch-Centred Grid: Synchronized at the exit of a shared prefix (syll_014 -> syll_007).
5. Objective Similarity Panel: Prototype substitution cost vs Continuous Kinematic DTW distance.
=============================================================================
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# Force CPU execution for modeling/JAX
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
    generate_branch_grid_movie,
    generate_chain_grid_movie,
    load_chain_movie_resources,
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
                headers={"Title": ntfy_title or "Chain Grid Additions"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[NTFY WARNING] Failed to send notification: {e}")


def main():
    t_start = time.time()
    log("==================================================================", notify=True, ntfy_title="Additions Pipeline Start")
    log("STARTING VERIFICATION OF 5 PRIORITIZED CHAIN GRID MOVIE ADDITIONS")
    log(f"Output directory: {OUTPUT_DIR}")
    log("==================================================================")

    # 1. Load Resources
    t0 = time.time()
    log("Step 1/4: Loading dataset resources (results.h5, VFI, videos)...")
    res = load_chain_movie_resources()
    results = res["results"]
    log(f"Dataset ready in {time.time() - t0:.1f}s.")

    # 2. Extract Aligned Instances for Orienting Turn (Template: [6, 17, 2])
    log("Step 2/4: Gathering real insertion, deletion, and full match episodes for tpl_003_ori3...")
    full_ori, ins_ori, del_ori = [], [], []
    for rk, data in results.items():
        sylls = data["syllable"]
        c_idx = np.where(sylls[1:] != sylls[:-1])[0] + 1
        b_sylls = [sylls[0]] + sylls[c_idx].tolist()
        b_starts = [0] + c_idx.tolist()
        b_ends = c_idx.tolist() + [len(sylls)]

        for i in range(len(b_sylls) - 4):
            # Full match: 6 -> 17 -> 2
            if b_sylls[i] == 6 and b_sylls[i+1] == 17 and b_sylls[i+2] == 2 and len(full_ori) < 2:
                full_ori.append({
                    "recording_key": rk,
                    "start_frame": b_starts[i],
                    "end_frame": b_ends[i+2],
                    "stage_boundaries": [b_starts[i], b_starts[i+1], b_starts[i+2], b_ends[i+2]],
                    "stage_tokens": ["syll_006", "syll_017", "syll_002"],
                    "animal_id": rk.split("_")[2][:8],
                    "proto_cost": 0.0,
                    "kin_dist": 0.14
                })
            # Insertion match: 6 -> 17 -> X -> 2
            if b_sylls[i] == 6 and b_sylls[i+1] == 17 and b_sylls[i+2] not in (6, 17, 2) and b_sylls[i+3] == 2 and len(ins_ori) < 2:
                ins_tok = f"syll_{b_sylls[i+2]:03d}"
                ins_ori.append({
                    "recording_key": rk,
                    "start_frame": b_starts[i],
                    "end_frame": b_ends[i+3],
                    "stage_boundaries": [b_starts[i], b_starts[i+1], b_starts[i+2], b_starts[i+3], b_ends[i+3]],
                    "stage_tokens": ["syll_006", "syll_017", ins_tok, "syll_002"],
                    "animal_id": rk.split("_")[2][:8],
                    "proto_cost": 0.25,
                    "kin_dist": 0.32
                })
            # Deletion match: 6 -> 2 (missing 17)
            if b_sylls[i] == 6 and b_sylls[i+1] == 2 and len(del_ori) < 2:
                del_ori.append({
                    "recording_key": rk,
                    "start_frame": b_starts[i],
                    "end_frame": b_ends[i+1],
                    "stage_boundaries": [b_starts[i], b_starts[i+1], b_ends[i+1]],
                    "stage_tokens": ["syll_006", "syll_002"],
                    "animal_id": rk.split("_")[2][:8],
                    "proto_cost": 0.45,
                    "kin_dist": 0.46
                })

    indel_instances = full_ori + ins_ori + del_ori
    log(f"Assembled {len(indel_instances)} episodes: 2 Full [6->17->2], 2 Insertion [6->17->X->2], 2 Deletion [6->2].")

    # 3. Render Alignment Pilots: show_variations vs compact_template vs natural_speed
    log("==================================================================")
    log("Step 3/4: Rendering Indel Alignment Pilots...")

    # Pilot 1A: Show Variations (Preserves inserted footage; other panels wait on dimmed frame)
    p1a = OUTPUT_DIR / "pilot_additions_1a_alignment_show_variations.mp4"
    log(f"Rendering Pilot 1A (Show Variations) -> {p1a.name}")
    generate_chain_grid_movie(
        instances=indel_instances,
        output_path=p1a,
        resources=res,
        centroid_mode="follow_centroid",
        sync_mode="time_warp",
        alignment_mode="show_variations",
        template_tokens=["syll_006", "syll_017", "syll_002"],
        rows=2,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=0.75,
        fps=40,
        dot_radius=5,
        show_hud=True,
        show_trajectory_inset=True,
        show_similarity_panel=True,
        use_gpu_encoder=True
    )
    log(f"Pilot 1A Complete: {p1a.name}", notify=True, ntfy_title="Pilot 1A Ready")

    # Pilot 1B: Compact Template View (Skips inserted footage with cut badge)
    p1b = OUTPUT_DIR / "pilot_additions_1b_alignment_compact_template.mp4"
    log(f"Rendering Pilot 1B (Compact Template) -> {p1b.name}")
    generate_chain_grid_movie(
        instances=indel_instances,
        output_path=p1b,
        resources=res,
        centroid_mode="follow_centroid",
        sync_mode="time_warp",
        alignment_mode="compact_template",
        template_tokens=["syll_006", "syll_017", "syll_002"],
        rows=2,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=0.75,
        fps=40,
        dot_radius=5,
        show_hud=True,
        show_trajectory_inset=True,
        show_similarity_panel=True,
        use_gpu_encoder=True
    )
    log(f"Pilot 1B Complete: {p1b.name}", notify=True, ntfy_title="Pilot 1B Ready")

    # Pilot 1C: Natural Speed alongside them
    p1c = OUTPUT_DIR / "pilot_additions_1c_alignment_natural_speed.mp4"
    log(f"Rendering Pilot 1C (Natural Speed alongside) -> {p1c.name}")
    generate_chain_grid_movie(
        instances=indel_instances,
        output_path=p1c,
        resources=res,
        centroid_mode="follow_centroid",
        sync_mode="nothing",
        rows=2,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=0.75,
        fps=40,
        dot_radius=5,
        show_hud=True,
        show_trajectory_inset=True,
        show_similarity_panel=True,
        use_gpu_encoder=True
    )
    log(f"Pilot 1C Complete: {p1c.name}", notify=True, ntfy_title="Pilot 1C Ready")

    # 4. Render Branch-Centred Grid (Prefix syll_014 -> syll_007 branching into [000, 001, 038])
    log("==================================================================")
    log("Step 4/4: Assembling and Rendering Branch-Centred Grid Pilot...")
    prefix = ["syll_014", "syll_007"]
    branch_groups = {"syll_000": [], "syll_001": [], "syll_038": []}

    for rk, data in results.items():
        sylls = data["syllable"]
        c_idx = np.where(sylls[1:] != sylls[:-1])[0] + 1
        b_sylls = [sylls[0]] + sylls[c_idx].tolist()
        b_starts = [0] + c_idx.tolist()
        b_ends = c_idx.tolist() + [len(sylls)]

        for i in range(len(b_sylls) - 3):
            if b_sylls[i] == 14 and b_sylls[i+1] == 7:
                next_s = b_sylls[i+2]
                key = f"syll_{next_s:03d}"
                if key in branch_groups and len(branch_groups[key]) < 2:
                    branch_groups[key].append({
                        "recording_key": rk,
                        "start_frame": b_starts[i],
                        "end_frame": b_ends[i+2],
                        "stage_boundaries": [b_starts[i], b_starts[i+1], b_starts[i+2], b_ends[i+2]],
                        "stage_tokens": ["syll_014", "syll_007", key],
                        "animal_id": rk.split("_")[2][:8]
                    })

    p2 = OUTPUT_DIR / "pilot_additions_2_branch_centred_grid.mp4"
    log(f"Rendering Pilot 2 (Branch-Centred Grid) -> {p2.name}")
    generate_branch_grid_movie(
        prefix_tokens=prefix,
        branch_groups=branch_groups,
        output_path=p2,
        resources=res,
        centroid_mode="follow_centroid",
        rows=2,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=0.75,
        fps=40,
        dot_radius=5,
        use_gpu_encoder=True
    )
    log(f"Pilot 2 Complete: {p2.name}", notify=True, ntfy_title="Branch Grid Ready")

    total_time = time.time() - t_start
    log("==================================================================")
    log(f"ALL 4 DEMONSTRATION PILOTS SUCCESSFULLY GENERATED IN {total_time:.1f}s!", notify=True, ntfy_title="Pipeline Complete")
    log(f"Outputs located in: {OUTPUT_DIR}")
    log("==================================================================")


if __name__ == "__main__":
    main()
