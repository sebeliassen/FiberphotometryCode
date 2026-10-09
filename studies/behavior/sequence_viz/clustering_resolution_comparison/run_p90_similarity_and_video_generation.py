#!/usr/bin/env python3
"""
run_p90_similarity_and_video_generation.py
=============================================================================
Implements the P90 Movement Similarity & Sample-Size Scaling Framework:
1. Standardized continuous kinematic feature trajectory representation:
   - Continuous 10D pose latent states z(t) interpolated to T=20
   - Centroid speed v(t) interpolated to T=20
   - Heading angular velocity dh(t) interpolated to T=20
2. Common reference trajectory for each theme across all resolutions.
3. Computes continuous movement similarity for all candidates across the 4 levels.
4. Generates the Cumulative Similarity Threshold Curve:
   fig_similarity_vs_qualifying_episodes.png
5. Identifies P90 representatives (clips near the 90th percentile of similarity)
   across distinct animals.
6. Renders 12 P90 1x3 widescreen videos (4 levels x 3 themes) with animated GIFs
   at native 40 fps without pauses.
7. Saves comprehensive similarity metadata JSON and summary CSV.
=============================================================================
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
    generate_chain_grid_movie,
    load_chain_movie_resources,
)
from syllable_generation.analysis.sequence_viz.clustering_resolution_comparison.run_clustering_resolution_experiment import (
    get_level_definitions,
    get_verified_metadata,
)

EXP_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/clustering_resolution_comparison"


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

    # Combine into standardized feature vector
    return np.concatenate([z_res, v_norm * 2.0, dh_norm * 2.0])


def main():
    print("=" * 80)
    print("P90 MOVEMENT SIMILARITY & SAMPLE SIZE SCALING PIPELINE")
    print("=" * 80)
    t0 = time.time()

    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]
    levels = get_level_definitions()

    # 1. Harvest candidates and compute standardized features
    theme_meta = [
        {"idx": 0, "id": "theme_A_rear_descent", "title": "Theme A: Wall-Rear & Descent Launch"},
        {"idx": 1, "id": "theme_B_turn_slalom", "title": "Theme B: Sprint & Turning Redirection"},
        {"idx": 2, "id": "theme_C_sensory_rest", "title": "Theme C: Olfactory Inspection & Rest"}
    ]

    theme_data = {}

    for tm in theme_meta:
        th_idx = tm["idx"]
        th_id = tm["id"]
        th_title = tm["title"]
        print(f"\n[HARVESTING] {th_title}...")

        level_cands = {}

        for lvl in levels:
            lvl_id = lvl["level_id"]
            lvl_name = lvl["name"]
            th = lvl["themes"][th_idx]
            tseq = th["target_seq"]
            stage_tokens = th["tokens"]
            max_dur = th["max_dur"]
            m_arr = lvl["map_arr"]
            n_stages = len(tseq)

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

                for i in range(len(c_seq) - n_stages):
                    if c_seq[i : i + n_stages] == tseq:
                        sf = c_starts[i]
                        ef = c_ends[i + n_stages - 1]
                        bounds = [c_starts[i + k] for k in range(n_stages)] + [ef]
                        durs = np.diff(bounds)
                        if not (all(d >= 2 for d in durs) and all(d <= max_dur for d in durs)):
                            continue
                        if ef >= len(cents) or ef - sf < 8:
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
                            "signed_turn_deg": signed_heading,
                            "feature_vector": feat
                        })

            print(f"  {lvl['name']}: {len(cands)} candidates ({len(set(c['animal_id'] for c in cands))} animals)")
            level_cands[lvl_id] = cands

        # 2. Build common reference trajectory for this theme
        # We use the mean continuous trajectory of Level 1 candidates as the fixed reference
        l1_feats = np.array([c["feature_vector"] for c in level_cands[1]])
        ref_vec = np.mean(l1_feats, axis=0)
        ref_norm = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)

        # Compute cosine similarity to common reference for all candidates across all levels
        for lvl_id, cands in level_cands.items():
            for c in cands:
                f = c["feature_vector"]
                sim = float(np.dot(f, ref_norm) / (np.linalg.norm(f) + 1e-8))
                c["movement_similarity"] = sim

        theme_data[th_id] = {
            "title": th_title,
            "ref_norm": ref_norm,
            "level_cands": level_cands
        }

    # =========================================================================
    # 3. GENERATE CUMULATIVE SIMILARITY THRESHOLD PLOT
    # =========================================================================
    print("\n" + "=" * 80)
    print("PLOTTING CUMULATIVE SIMILARITY THRESHOLD CURVES")
    print("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(19, 10), sharex="col")
    thresholds = np.linspace(0.10, 0.95, 86)
    lvl_colors = {
        1: "#1f77b4",  # Blue (Level 1)
        2: "#2ca02c",  # Green (Level 2)
        3: "#ff7f0e",  # Orange (Level 3)
        4: "#d62728"   # Red (Level 4)
    }
    lvl_labels = {
        1: "Level 1: 100 Raw States",
        2: "Level 2: 93 Conservative",
        3: "Level 3: 22 Preserved Asymmetry",
        4: "Level 4: 14 Broad Families"
    }

    scaling_summary_table = []

    for col_idx, tm in enumerate(theme_meta):
        th_id = tm["id"]
        th_title = tm["title"]
        ax_n = axes[0, col_idx]
        ax_anim = axes[1, col_idx]

        ax_n.set_title(th_title, fontsize=14, fontweight="bold", pad=12)

        for lvl_id in [1, 2, 3, 4]:
            cands = theme_data[th_id]["level_cands"][lvl_id]
            sims = np.array([c["movement_similarity"] for c in cands])
            aids = [c["animal_id"] for c in cands]

            n_qual = [np.sum(sims >= t) for t in thresholds]
            anim_qual = [len(set(aid for aid, s in zip(aids, sims) if s >= t)) for t in thresholds]

            p90 = float(np.percentile(sims, 90)) if len(sims) > 0 else 0.0
            p50 = float(np.percentile(sims, 50)) if len(sims) > 0 else 0.0

            # Plot Episode Count
            ax_n.plot(
                thresholds, n_qual,
                color=lvl_colors[lvl_id],
                linewidth=2.5,
                label=f"{lvl_labels[lvl_id]} (N={len(cands)}, P90={p90:.2f})"
            )
            # Mark P90 point on curve
            n_at_p90 = np.sum(sims >= p90)
            anim_at_p90 = len(set(aid for aid, s in zip(aids, sims) if s >= p90))
            ax_n.scatter([p90], [n_at_p90], color=lvl_colors[lvl_id], s=70, zorder=5, edgecolor="black")

            # Plot Animal Coverage
            ax_anim.plot(
                thresholds, anim_qual,
                color=lvl_colors[lvl_id],
                linewidth=2.5
            )
            ax_anim.scatter([p90], [anim_at_p90], color=lvl_colors[lvl_id], s=70, zorder=5, edgecolor="black")

            # Record stats at benchmark thresholds
            scaling_summary_table.append({
                "Theme": tm["title"],
                "Level": lvl_labels[lvl_id],
                "Total_Retrieved_N": len(cands),
                "Total_Animals": len(set(aids)),
                "Median_Similarity": round(p50, 3),
                "P90_Similarity": round(p90, 3),
                "N_exceed_070": int(np.sum(sims >= 0.70)),
                "Anim_exceed_070": len(set(aid for aid, s in zip(aids, sims) if s >= 0.70)),
                "N_exceed_075": int(np.sum(sims >= 0.75)),
                "Anim_exceed_075": len(set(aid for aid, s in zip(aids, sims) if s >= 0.75)),
                "N_exceed_080": int(np.sum(sims >= 0.80)),
                "Anim_exceed_080": len(set(aid for aid, s in zip(aids, sims) if s >= 0.80)),
                "N_exceed_085": int(np.sum(sims >= 0.85)),
                "Anim_exceed_085": len(set(aid for aid, s in zip(aids, sims) if s >= 0.85)),
            })

        ax_n.set_ylabel("Qualifying Episodes (N ≥ threshold)", fontsize=11, fontweight="semibold")
        ax_n.grid(True, linestyle="--", alpha=0.5)
        ax_n.legend(fontsize=9, loc="upper right")
        ax_n.set_xlim(0.10, 0.95)

        ax_anim.set_xlabel("Movement Similarity Threshold to Reference (Cosine Sim)", fontsize=11, fontweight="semibold")
        ax_anim.set_ylabel("Distinct Animal Coverage (max 15)", fontsize=11, fontweight="semibold")
        ax_anim.set_ylim(0, 16)
        ax_anim.set_xlim(0.10, 0.95)
        ax_anim.axhline(15, color="gray", linestyle=":", alpha=0.6)
        ax_anim.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = EXP_DIR / "fig_similarity_vs_qualifying_episodes.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Successfully saved scaling plot to: {plot_path}")

    # Copy to brain artifact directory for ide visibility
    brain_fig = Path("/home/fsp585/.gemini/antigravity-ide/brain/b03aa115-6afd-40d7-a4d1-a025411e6b82/fig_similarity_vs_qualifying_episodes.png")
    import shutil
    shutil.copy2(plot_path, brain_fig)
    print(f"[PLOT] Copied figure to brain artifact directory: {brain_fig}")

    df_scaling = pd.DataFrame(scaling_summary_table)
    scaling_csv = EXP_DIR / "p90_similarity_scaling_summary.csv"
    df_scaling.to_csv(scaling_csv, index=False)
    print(f"[CSV] Saved scaling summary table to: {scaling_csv}")

    if "--plot-only" in sys.argv:
        print("[EXIT] --plot-only flag specified. Skipping video rendering.")
        return

    # =========================================================================
    # 4. SELECT P90 REPRESENTATIVES & RENDER VIDEOS (12 VIDEOS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SELECTING P90 REPRESENTATIVES (90TH PERCENTILE) ACROSS DISTINCT ANIMALS")
    print("=" * 80)

    p90_records = []

    for tm in theme_meta:
        th_id = tm["id"]
        th_title = tm["title"]

        for lvl in levels:
            lvl_id = lvl["level_id"]
            lvl_dir = EXP_DIR / lvl["folder"]
            movies_dir = lvl_dir / "movies"
            thumbs_dir = lvl_dir / "thumbnails"
            movies_dir.mkdir(parents=True, exist_ok=True)
            thumbs_dir.mkdir(parents=True, exist_ok=True)

            cands = theme_data[th_id]["level_cands"][lvl_id]
            if len(cands) < 3:
                print(f"[SKIP] Insufficient candidates for {lvl['name']} {th_title} ({len(cands)})")
                continue

            sims = np.array([c["movement_similarity"] for c in cands])
            p90_val = float(np.percentile(sims, 90))

            # Select clips *near* the 90th percentile of similarity
            # Order candidates by absolute distance to p90_val
            diff_to_p90 = np.abs(sims - p90_val)
            rank_order = np.argsort(diff_to_p90)

            p90_panels = []
            seen_aids = set()

            for idx in rank_order:
                c = cands[idx]
                aid = c["animal_id"]
                if aid not in seen_aids:
                    c_copy = c.copy()
                    c_copy["role"] = "P90_REPRESENTATIVE"
                    c_copy["proto_cost"] = float(np.abs(c["movement_similarity"] - p90_val))
                    c_copy["kin_dist"] = float(c["movement_similarity"])
                    p90_panels.append(c_copy)
                    seen_aids.add(aid)
                    if len(p90_panels) >= 3:
                        break

            # Fallback if fewer than 3 distinct animals
            if len(p90_panels) < 3:
                for idx in rank_order:
                    c = cands[idx]
                    if (c["recording_key"], c["start_frame"]) not in {(p["recording_key"], p["start_frame"]) for p in p90_panels}:
                        c_copy = c.copy()
                        c_copy["role"] = "P90_REPRESENTATIVE"
                        c_copy["proto_cost"] = float(np.abs(c["movement_similarity"] - p90_val))
                        c_copy["kin_dist"] = float(c["movement_similarity"])
                        p90_panels.append(c_copy)
                        if len(p90_panels) >= 3:
                            break

            rec = {
                "level_id": lvl_id,
                "level_name": lvl["name"],
                "theme_id": th_id,
                "theme_title": th_title,
                "p90_target_similarity": round(p90_val, 3),
                "actual_panel_similarities": [round(p["movement_similarity"], 3) for p in p90_panels],
                "panel_animals": [p["animal_id"] for p in p90_panels],
                "total_candidates_N": len(cands),
                "total_animals_N": len(set(c["animal_id"] for c in cands))
            }
            p90_records.append(rec)

            fn_p90 = f"{th_id}_p90.mp4"
            out_p90 = movies_dir / fn_p90
            print(f"[{lvl['name']} | {th_id}] Rendering P90 Movie -> {fn_p90}")
            print(f"   P90 Target Sim = {p90_val:.3f}, Panels Sim = {[round(p['movement_similarity'], 3) for p in p90_panels]}")
            print(f"   Animals: {[p['animal_id'] for p in p90_panels]}")

            generate_chain_grid_movie(
                instances=p90_panels,
                output_path=out_p90,
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

            # Generate Thumbnail & GIF
            thumb_p = thumbs_dir / f"{th_id}_p90_thumb.png"
            gif_p = thumbs_dir / f"{th_id}_p90.gif"

            cap = cv2.VideoCapture(str(out_p90))
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

    p90_meta_json = EXP_DIR / "p90_similarity_metadata.json"
    with open(p90_meta_json, "w") as f:
        json.dump(p90_records, f, indent=2)

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE in {time.time() - t0:.1f}s!")
    print(f"Plot saved to: {plot_path}")
    print(f"Metadata saved to: {p90_meta_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
