#!/usr/bin/env python3
"""
run_relational_topology_and_heading_reset_movies.py
=============================================================================
Implementation and diagnostic movie renderer for:
1. Dvorkin et al. (2008) Relational Excursion Topology &
   Alexander et al. (2020) Egocentric Boundary Vectors (EBVs).
2. Per-Syllable Heading Reset (stage-relative camera alignment) to decouple
   motor performance from Cartesian trajectory heading drift.

Generates two dedicated diagnostic grid movies:
- relational_topology_dvorkin_alexander_grid.mp4
- syllable_heading_reset_comparison_grid.mp4
=============================================================================
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# CPU execution for JAX / Keypoint-MoSeq
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from studies.behavior.sequence_viz.bout_quality_and_spatial_verifier import ChainQualityAndContextVerifier
from studies.behavior.sequence_viz.chain_grid_movie_wrapper import (
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
                headers={"Title": ntfy_title or "Relational Topology Diagnostics"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[NTFY WARNING] Failed to send notification: {e}")


def extract_frame_from_video(video_path, target_frame_idx, output_png_path):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        cv2.imwrite(str(output_png_path), frame)
        return True
    return False


def main():
    t0 = time.time()
    log("==================================================================", notify=True, ntfy_title="Relational Topology Start")
    log("STARTING DVORKIN-ALEXANDER RELATIONAL TOPOLOGY & HEADING RESET RUN")
    log("Grounding: Dvorkin et al. 2008 & Alexander et al. 2020")
    log(f"Output directory: {OUTPUT_DIR}")
    log("==================================================================")

    res = load_chain_movie_resources()
    results = res["results"]
    verifier = ChainQualityAndContextVerifier(results)
    log("Trained baseline reference profiles and initialized spatial context verifier.")

    # --------------------------------------------------------------------------
    # 1. Search candidate chains for [1, 11, 1, 30]
    # --------------------------------------------------------------------------
    log("Searching for multi-stage episodes of seqNMF Motif [1, 11, 1, 30]...")
    candidates = []
    for rk, data in results.items():
        sylls = data["syllable"]
        c_idx = np.where(sylls[1:] != sylls[:-1])[0] + 1
        b_sylls = [sylls[0]] + sylls[c_idx].tolist()
        b_starts = [0] + c_idx.tolist()
        b_ends = c_idx.tolist() + [len(sylls)]

        for i in range(len(b_sylls) - 4):
            if b_sylls[i] == 1 and b_sylls[i+1] == 11 and b_sylls[i+2] == 1 and b_sylls[i+3] == 30:
                candidates.append({
                    "recording_key": rk,
                    "start_frame": b_starts[i],
                    "end_frame": b_ends[i+3],
                    "stage_boundaries": [b_starts[i], b_starts[i+1], b_starts[i+2], b_starts[i+3], b_ends[i+3]],
                    "stage_tokens": ["syll_001", "syll_011", "syll_001", "syll_030"],
                    "animal_id": rk.split("_")[2][:8],
                    "genotype": "WT",
                    "drug": "Saline" if "day1" in rk else "Amphetamine"
                })

    log(f"Retrieved {len(candidates)} raw candidates. Running weakest-stage and relational topology analysis...")
    verified_chains = []
    for c in candidates:
        v = verifier.verify_chain(c)
        c_dict = dict(c)
        c_dict.update(v)
        verified_chains.append(c_dict)

    df = pd.DataFrame(verified_chains)
    log("Classification distribution:")
    log(str(df["classification"].value_counts().to_dict()))

    clean_df = df[df["classification"].isin(["TRUSTED_CLEAN", "MOTOR_VARIANT"])]
    log(f"Clean sound episodes: {len(clean_df)} / {len(df)}")
    log("Dvorkin Relational Topology distribution on clean episodes:")
    log(str(clean_df["relational_topology"].value_counts().to_dict()))
    log("Alexander EBV distribution on clean episodes:")
    log(str(clean_df["start_ebv"].value_counts().to_dict()))

    # --------------------------------------------------------------------------
    # 2. Movie 1: Relational Topology Grid (Dvorkin & Alexander)
    # --------------------------------------------------------------------------
    # Pick Row 1: Boundary Relocations (RELOCATION_BOUNDARY along perimeter with flank EBVs)
    # Pick Row 2: Origin Loops (RETURN_TO_ORIGIN) or Cross-Arena Relocations (RELOCATION_CROSS_ARENA)
    row1_candidates = clean_df[clean_df["relational_topology"] == "RELOCATION_BOUNDARY"].sort_values(by="quality_min", ascending=False)
    selected_row1 = []
    seen_animals = set()
    for _, r in row1_candidates.iterrows():
        aid = r["animal_id"]
        if aid not in seen_animals:
            d = r.to_dict()
            d["animal_id"] = f"{aid} [Reloc Bnd | {r['start_ebv'][:7]}]"
            d["proto_cost"] = 1.0 - r["similarity_mean"]
            d["kin_dist"] = r["net_displacement"] / 100.0
            selected_row1.append(d)
            seen_animals.add(aid)
            if len(selected_row1) >= 3:
                break

    # If fewer than 3, take top cross arena with flank EBV
    if len(selected_row1) < 3:
        alt = clean_df[clean_df["start_ebv"].isin(["LEFT_FLANK", "RIGHT_FLANK"])].sort_values(by="quality_min", ascending=False)
        for _, r in alt.iterrows():
            aid = r["animal_id"]
            if aid not in seen_animals:
                d = r.to_dict()
                d["animal_id"] = f"{aid} [Reloc Bnd | {r['start_ebv'][:7]}]"
                d["proto_cost"] = 1.0 - r["similarity_mean"]
                d["kin_dist"] = r["net_displacement"] / 100.0
                selected_row1.append(d)
                seen_animals.add(aid)
                if len(selected_row1) >= 3:
                    break

    # Row 2: Cross Arena Relocations or Return to Origin
    row2_candidates = clean_df[clean_df["relational_topology"].isin(["RELOCATION_CROSS_ARENA", "RETURN_TO_ORIGIN", "RELOCATION_INTERIOR"])].sort_values(by="quality_min", ascending=False)
    selected_row2 = []
    for _, r in row2_candidates.iterrows():
        aid = r["animal_id"]
        if aid not in seen_animals or len(selected_row2) < 3:
            d = r.to_dict()
            topo_tag = "Loop A->A" if r["relational_topology"] == "RETURN_TO_ORIGIN" else "Traverse A->B"
            d["animal_id"] = f"{aid} [{topo_tag} | {r['start_ebv'][:7]}]"
            d["proto_cost"] = 1.0 - r["similarity_mean"]
            d["kin_dist"] = r["net_displacement"] / 100.0
            selected_row2.append(d)
            seen_animals.add(aid)
            if len(selected_row2) >= 3:
                break

    relational_instances = selected_row1[:3] + selected_row2[:3]
    out_relational = OUTPUT_DIR / "relational_topology_dvorkin_alexander_grid.mp4"
    log(f"Rendering Movie 1: Relational Topology Grid -> {out_relational.name}...")
    generate_chain_grid_movie(
        instances=relational_instances,
        output_path=out_relational,
        resources=res,
        centroid_mode="follow_centroid",
        sync_mode="nothing", # Pure continuous fluid real speed
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
    log(f"Movie 1 Ready: {out_relational.name}", notify=True, ntfy_title="Movie 1 Generated")

    # --------------------------------------------------------------------------
    # 3. Movie 2: Syllable Heading Reset Comparison Grid
    # --------------------------------------------------------------------------
    # Compare:
    # Row 1: reset_heading_per_syllable = False (Standard Chain-Onset Alignment)
    # Row 2: reset_heading_per_syllable = True (Reset Heading at each syllable onset)
    log("Selecting episodes with prominent turning angles for Syllable Heading Reset comparison...")
    # Find episodes where max turn angle across stages >= 35 deg
    turn_episodes = []
    for _, r in clean_df.iterrows():
        turns = [abs(a) for a in r["stage_turn_angles"][1:]]
        if max(turns) >= 35.0:
            turn_episodes.append(r.to_dict())

    turn_df = pd.DataFrame(turn_episodes).sort_values(by="quality_min", ascending=False)
    selected_episodes = []
    seen_turn_animals = set()
    for _, r in turn_df.iterrows():
        aid = r["animal_id"]
        if aid not in seen_turn_animals:
            d = r.to_dict()
            selected_episodes.append(d)
            seen_turn_animals.add(aid)
            if len(selected_episodes) >= 3:
                break

    log(f"Selected 3 distinct turning episodes: {[e['animal_id'] for e in selected_episodes]}")

    # Build comparison instances:
    # Row 1 = standard heading (False)
    # Row 2 = per-syllable heading reset (True)
    out_row1 = OUTPUT_DIR / "temp_row1_standard_heading.mp4"
    out_row2 = OUTPUT_DIR / "temp_row2_reset_heading.mp4"
    out_comparison = OUTPUT_DIR / "syllable_heading_reset_comparison_grid.mp4"

    # Row 1 instances
    row1_insts = []
    for e in selected_episodes:
        d = dict(e)
        d["animal_id"] = f"{e['animal_id']} [Chain-Onset Fixed]"
        row1_insts.append(d)

    # Row 2 instances
    row2_insts = []
    for e in selected_episodes:
        d = dict(e)
        d["animal_id"] = f"{e['animal_id']} [Reset Per Syllable]"
        row2_insts.append(d)

    log("Rendering Row 1 (Standard Chain Onset Alignment)...")
    generate_chain_grid_movie(
        instances=row1_insts,
        output_path=out_row1,
        resources=res,
        centroid_mode="follow_centroid",
        sync_mode="nothing",
        rows=1,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=1.0,
        fps=40,
        dot_radius=5,
        show_hud=True,
        show_trajectory_inset=True,
        show_similarity_panel=False,
        reset_heading_per_syllable=False,
        use_gpu_encoder=True
    )

    log("Rendering Row 2 (Reset Heading Per Syllable)...")
    generate_chain_grid_movie(
        instances=row2_insts,
        output_path=out_row2,
        resources=res,
        centroid_mode="follow_centroid",
        sync_mode="nothing",
        rows=1,
        cols=3,
        window_size=240,
        scaled_window_size=320,
        pre_sec=0.75,
        post_sec=1.0,
        fps=40,
        dot_radius=5,
        show_hud=True,
        show_trajectory_inset=True,
        show_similarity_panel=False,
        reset_heading_per_syllable=True,
        use_gpu_encoder=True
    )

    # Vertically stack Row 1 and Row 2 into out_comparison using FFmpeg H.264
    log("Stacking Row 1 and Row 2 into unified 2x3 comparison movie using FFmpeg H.264...")
    import subprocess
    cmd_nvenc = [
        "ffmpeg", "-y",
        "-i", str(out_row1),
        "-i", str(out_row2),
        "-filter_complex", "vstack",
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        str(out_comparison)
    ]
    cmd_cpu = [
        "ffmpeg", "-y",
        "-i", str(out_row1),
        "-i", str(out_row2),
        "-filter_complex", "vstack",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_comparison)
    ]
    try:
        subprocess.run(cmd_nvenc, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        subprocess.run(cmd_cpu, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Clean up temp files
    if out_row1.exists(): out_row1.unlink()
    if out_row2.exists(): out_row2.unlink()

    log(f"Movie 2 Ready: {out_comparison.name}", notify=True, ntfy_title="Movie 2 Generated")

    # --------------------------------------------------------------------------
    # 4. Extract Diagnostic Screenshot Frames
    # --------------------------------------------------------------------------
    log("Extracting diagnostic screenshot frames for walkthrough...")
    brain_dir = Path("/home/fsp585/.gemini/antigravity-ide/brain/6b2f8bd9-6fe5-4797-8e3f-dfeea193e3f8")

    # From Movie 1: Relational Topology
    extract_frame_from_video(out_relational, 60, brain_dir / "relational_topology_frame60.png")
    extract_frame_from_video(out_relational, 110, brain_dir / "relational_topology_frame110.png")

    # From Movie 2: Heading Reset Comparison
    extract_frame_from_video(out_comparison, 55, brain_dir / "syllable_heading_reset_frame55.png")
    extract_frame_from_video(out_comparison, 95, brain_dir / "syllable_heading_reset_frame95.png")

    total_time = time.time() - t0
    log("==================================================================")
    log(f"ALL DVORKIN-ALEXANDER & HEADING RESET MOVIES COMPLETED IN {total_time:.1f}s!", notify=True, ntfy_title="All Movies Complete")
    log("==================================================================")


if __name__ == "__main__":
    main()
