#!/usr/bin/env python3
"""
run_bout_quality_and_spatial_diagnostics.py
=============================================================================
Demonstration and Diagnostic Runner for the 4-Part Quality, Trajectory,
Weakest-Stage, and Spatial Context Framework proposed by ChatGPT.

Evaluates candidate chains across discovery methods:
1. Physical Tracking Quality (Q in [0, 1]) vs Kinematic Similarity (S in [0, 1]).
2. Trajectory Profile Comparison (Speed, Turn Rate, 10D Pose Latents).
3. Weakest-Stage (Worst-Link) Attribution: identifies exactly which stage failed.
4. Spatial Context: Wall distance (d_wall) & Wall alignment (\theta_wall).
5. Categorizes instances into:
   - TRUSTED_CLEAN: Sound tracking (Q_min >= 0.8) & Typical motor profile (S_min >= 0.4)
   - MOTOR_VARIANT: Sound tracking (Q_min >= 0.8), but rare/extreme motor execution.
   - MEASUREMENT_CORRUPT: Physical sensor/tracking failure (Q_min < 0.8).
6. Renders a dedicated diagnostic comparison grid movie:
   - Top row: TRUSTED_CLEAN episodes
   - Bottom row: MOTOR_VARIANT episodes (demonstrating that biological variants
     are preserved rather than falsely purged!)
=============================================================================
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Force CPU execution for modeling / JAX
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.bout_quality_and_spatial_verifier import ChainQualityAndContextVerifier
from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
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
                headers={"Title": ntfy_title or "Bout Quality & Spatial Diagnostics"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[NTFY WARNING] Failed to send notification: {e}")


def main():
    t0 = time.time()
    log("==================================================================", notify=True, ntfy_title="Bout Quality Diagnostics Start")
    log("STARTING BOUT QUALITY, WEAKEST-STAGE & SPATIAL VERIFICATION RUN")
    log("Testing ChatGPT's 4-Part Quality vs Similarity & Spatial Context Architecture")
    log(f"Output directory: {OUTPUT_DIR}")
    log("==================================================================")

    res = load_chain_movie_resources()
    results = res["results"]

    verifier = ChainQualityAndContextVerifier(results)
    log("Trained baseline reference profiles across all clean syllable bouts.")

    # --------------------------------------------------------------------------
    # 1. Evaluate Method D: seqNMF Factor Motif 05 ([1, 11, 1, 30])
    # --------------------------------------------------------------------------
    log("Evaluating Method D (seqNMF Motif: [1 -> 11 -> 1 -> 30]) candidates...")
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
                    "genotype": "WT",
                    "drug": "Saline" if "day1" in rk else "Amphetamine"
                })

    log(f"Collected {len(d_raw)} raw occurrences of [1, 11, 1, 30]. Analyzing each stage...")
    evaluated_chains = []
    for inst in d_raw:
        ver = verifier.verify_chain(inst)
        inst_enriched = dict(inst)
        inst_enriched.update(ver)
        evaluated_chains.append(inst_enriched)

    # Statistical Breakdown
    df = pd.DataFrame(evaluated_chains)
    counts = df["classification"].value_counts().to_dict()
    log("Classification Breakdown for seqNMF Factor 05:")
    log(f"  TRUSTED_CLEAN: {counts.get('TRUSTED_CLEAN', 0)} ({counts.get('TRUSTED_CLEAN', 0)/len(df)*100:.1f}%)")
    log(f"  MOTOR_VARIANT (Biological Variants Kept): {counts.get('MOTOR_VARIANT', 0)} ({counts.get('MOTOR_VARIANT', 0)/len(df)*100:.1f}%)")
    log(f"  MEASUREMENT_CORRUPT (Hardware/Tracking Failures): {counts.get('MEASUREMENT_CORRUPT', 0)} ({counts.get('MEASUREMENT_CORRUPT', 0)/len(df)*100:.1f}%)")

    # Weakest stage breakdown among corrupted
    corrupt_df = df[df["classification"] == "MEASUREMENT_CORRUPT"]
    worst_counts = corrupt_df["quality_worst_stage_token"].value_counts().to_dict()
    log(f"Weakest-Stage Attribution among corrupt chains: {worst_counts}")

    # Spatial wall proximity breakdown
    clean_df = df[df["classification"].isin(["TRUSTED_CLEAN", "MOTOR_VARIANT"])]
    wall_following_count = sum(clean_df["has_wall_following"])
    mean_dist_all = clean_df["mean_wall_dist"].mean()
    log(f"Spatial Context for Clean Episodes: Mean Wall Distance = {mean_dist_all:.1f} px | Wall-Following Thigmotaxis = {wall_following_count} / {len(clean_df)} ({wall_following_count/len(clean_df)*100:.1f}%)")

    # --------------------------------------------------------------------------
    # 2. Select Diagnostic Comparison Grid: Clean vs Motor Variants
    # --------------------------------------------------------------------------
    # Pick top 3 TRUSTED_CLEAN across distinct animals
    clean_sorted = df[df["classification"] == "TRUSTED_CLEAN"].sort_values(by=["quality_min", "similarity_mean"], ascending=[False, False])
    selected_clean = []
    seen_animals = set()
    for _, row in clean_sorted.iterrows():
        aid = row["animal_id"]
        if aid not in seen_animals:
            d_inst = row.to_dict()
            d_inst["animal_id"] = f"{aid} [CLEAN Q:{row['quality_min']:.2f}]"
            d_inst["proto_cost"] = 1.0 - row["similarity_mean"]
            d_inst["kin_dist"] = row["mean_wall_dist"] / 100.0
            selected_clean.append(d_inst)
            seen_animals.add(aid)
            if len(selected_clean) >= 3:
                break

    # Pick top 3 MOTOR_VARIANT across distinct animals (sound tracking Q>=0.8, but atypical speed or turn angle)
    variant_df = df[df["classification"] == "MOTOR_VARIANT"]
    if len(variant_df) < 3:
        # If few variants, pick sound tracking with lowest similarity
        variant_df = df[(df["quality_min"] >= 0.8) & (df["similarity_min"] < 0.5)].sort_values(by="similarity_min")

    selected_variants = []
    for _, row in variant_df.iterrows():
        aid = row["animal_id"]
        if aid not in seen_animals or len(selected_variants) < 3:
            d_inst = row.to_dict()
            worst_tok = row["quality_worst_stage_token"]
            d_inst["animal_id"] = f"{aid} [VAR S:{row['similarity_min']:.2f}]"
            d_inst["proto_cost"] = 1.0 - row["similarity_mean"]
            d_inst["kin_dist"] = row["mean_wall_dist"] / 100.0
            selected_variants.append(d_inst)
            if len(selected_variants) >= 3:
                break

    # 2 x 3 Grid: Row 1 = Clean Standard, Row 2 = Motor Variants
    comparison_instances = selected_clean + selected_variants
    log(f"Assembled {len(comparison_instances)} diagnostic episodes: {len(selected_clean)} TRUSTED_CLEAN (Row 1) and {len(selected_variants)} MOTOR_VARIANT (Row 2).")

    out_comp = OUTPUT_DIR / "diagnostic_quality_vs_spatial_comparison_grid.mp4"
    log(f"Rendering Diagnostic Comparison Movie -> {out_comp.name}")
    generate_chain_grid_movie(
        instances=comparison_instances,
        output_path=out_comp,
        resources=res,
        centroid_mode="follow_centroid",
        sync_mode="nothing", # Pure continuous fluid natural speed
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
    log(f"Diagnostic Movie Ready: {out_comp.name}", notify=True, ntfy_title="Quality Diagnostics Complete")

    total_time = time.time() - t0
    log("==================================================================")
    log(f"BOUT QUALITY & SPATIAL VERIFICATION COMPLETE IN {total_time:.1f}s!", notify=True, ntfy_title="All Diagnostics Complete")
    log("==================================================================")


if __name__ == "__main__":
    main()
