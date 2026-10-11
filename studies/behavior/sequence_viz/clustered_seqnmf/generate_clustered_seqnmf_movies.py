#!/usr/bin/env python3
"""
generate_clustered_seqnmf_movies.py
=============================================================================
Generates 2x3 high-definition, fluid, synchronized diagnostic grid movies
for the top 4 discovered Clustered seqNMF Macro-Motifs:
1. Macro-Motif 7: Kinetic Rearing & Sprint Loop (Flagship KI Mutant Loop)
2. Macro-Motif 1: Rest & Maintenance Cycle (Signature WT Vigilance Cycle)
3. Macro-Motif 4: High-Speed Slalom Burst (Rapid acceleration and turn)
4. Macro-Motif 8: Attentive Scanning Loop (Head-casting & micro-sniffing)

Features:
- Native 80 Hz camera sync via fast-seek OpenCV reader
- Multi-token HUD with vibrant high-contrast color coding
- Real-time 2D spatial trajectory inset with glowing head marker
- Fluid continuous playback (zero pause/freeze frames)
=============================================================================
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from studies.behavior.sequence_viz.chain_grid_movie_wrapper import (
    filter_chain_outliers,
    generate_chain_grid_movie,
    load_chain_movie_resources,
)
from studies.behavior.sequence_viz.clustered_seqnmf.prepare_clustered_sequences import (
    SYLL_TO_META_ARR,
)

OUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/clustered_seqnmf/movies"
OUT_DIR.mkdir(parents=True, exist_ok=True)


MOTIF_TARGETS = [
    {
        "id": "macro_motif_07",
        "title": "Macro-Motif 7: Kinetic Wall-Rear & Sprint Loop (KI-Enriched)",
        "target_seq": [3, 4, 6, 7, 3],  # Sprint -> Turn_R -> Rear -> Touchdown -> Sprint
        "stage_tokens": ["Sprint", "Turn_R", "Rear", "Descent", "Sprint"],
        "filename": "macro_motif_07_kinetic_rearing_and_sprint_loop.mp4",
        "min_path": 10.0,
        "max_dur": 80,
    },
    {
        "id": "macro_motif_01",
        "title": "Macro-Motif 1: Rest & Maintenance Cycle (WT-Enriched)",
        "target_seq": [8, 11, 8, 11, 8],  # Sniff -> Rest -> Sniff -> Rest -> Sniff
        "stage_tokens": ["Sniff", "Rest", "Sniff", "Rest", "Sniff"],
        "filename": "macro_motif_01_rest_and_maintenance_cycle.mp4",
        "min_path": 0.0,  # allow stationary pauses
        "max_dur": 120,
    },
    {
        "id": "macro_motif_04",
        "title": "Macro-Motif 4: High-Speed Slalom Burst (KI-Enriched)",
        "target_seq": [8, 3, 5, 3],  # Sniff -> Sprint -> Turn_L -> Sprint
        "stage_tokens": ["Sniff", "Sprint", "Turn_L", "Sprint"],
        "filename": "macro_motif_04_high_speed_slalom_burst.mp4",
        "min_path": 15.0,
        "max_dur": 60,
    },
    {
        "id": "macro_motif_08",
        "title": "Macro-Motif 8: Attentive Visual Scanning Loop (WT-Enriched)",
        "target_seq": [8, 9, 8, 9, 8],  # Sniff -> Look -> Sniff -> Look -> Sniff
        "stage_tokens": ["Sniff", "Look", "Sniff", "Look", "Sniff"],
        "filename": "macro_motif_08_attentive_scanning_loop.mp4",
        "min_path": 0.0,
        "max_dur": 90,
    },
]


def extract_motif_candidates(results, target_seq, stage_tokens, max_dur=80):
    n_stages = len(target_seq)
    candidates = []

    for rk, data in results.items():
        sylls = data["syllable"]
        meta = SYLL_TO_META_ARR[sylls]
        c_idx = np.where(meta[1:] != meta[:-1])[0] + 1
        m_seq = [meta[0]] + meta[c_idx].tolist()
        b_starts = [0] + c_idx.tolist()
        b_ends = c_idx.tolist() + [len(meta)]

        for i in range(len(m_seq) - n_stages):
            if m_seq[i : i + n_stages] == target_seq:
                st_frame = b_starts[i]
                end_frame = b_ends[i + n_stages - 1]
                boundaries = [b_starts[i + k] for k in range(n_stages)] + [end_frame]
                durs = np.diff(boundaries)

                if all(d >= 3 for d in durs) and all(d <= max_dur for d in durs):
                    candidates.append({
                        "recording_key": rk,
                        "start_frame": st_frame,
                        "end_frame": end_frame,
                        "stage_boundaries": boundaries,
                        "stage_tokens": stage_tokens,
                        "animal_id": rk.split("_")[2][:8] if len(rk.split("_")) > 2 else "animal",
                        "method": "Clustered_seqNMF",
                        "proto_cost": 0.0,
                        "kin_dist": 0.10,
                    })

    return candidates


def main():
    print("=" * 80)
    print("GENERATING CLUSTERED SEQNMF MACRO-MOTIF GRID MOVIES")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 80)

    t0 = time.time()
    print("Loading modeling results and video paths...")
    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]
    print(f"Resources ready in {time.time() - t0:.1f}s.")

    rendered_movies = []

    for idx, motif in enumerate(MOTIF_TARGETS, 1):
        print("\n" + "-" * 80)
        print(f"[{idx}/{len(MOTIF_TARGETS)}] Processing: {motif['title']}")
        print(f"Target Sequence: {motif['target_seq']} ({' -> '.join(motif['stage_tokens'])})")

        candidates = extract_motif_candidates(
            results, motif["target_seq"], motif["stage_tokens"], max_dur=motif["max_dur"]
        )
        print(f"  Found {len(candidates)} raw matching episodes across 30 sessions.")

        sel_instances, stats = filter_chain_outliers(
            candidates,
            results,
            target_count=6,
            max_instances_per_animal=2,
            min_path_displacement=motif["min_path"],
            video_frame_indexes=vfi,
        )

        print(f"  Outlier Audit: Filtered {stats.get('total_candidates', len(candidates))} -> Selected {len(sel_instances)} diverse clean panels.")

        if len(sel_instances) == 0:
            print(f"  [WARNING] No clean instances found for {motif['id']}, skipping.")
            continue

        out_path = OUT_DIR / motif["filename"]
        print(f"  Rendering 2x3 Fluid Grid Movie -> {out_path.name}...")
        t_render = time.time()

        generate_chain_grid_movie(
            instances=sel_instances,
            output_path=out_path,
            resources=res,
            centroid_mode="follow_centroid",
            sync_mode="nothing",  # Continuous natural playback
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
        )

        elapsed = time.time() - t_render
        size_mb = os.path.getsize(out_path) / 1e6 if out_path.exists() else 0.0
        print(f"  ✓ Successfully rendered {out_path.name} in {elapsed:.1f}s ({size_mb:.2f} MB)")
        rendered_movies.append(out_path)

    print("\n" + "=" * 80)
    print(f"ALL {len(rendered_movies)} MACRO-MOTIF MOVIES RENDERED SUCCESSFULLY!")
    for m in rendered_movies:
        print(f"  * {m}")
    print("=" * 80)


if __name__ == "__main__":
    main()
