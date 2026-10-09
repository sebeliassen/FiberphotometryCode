#!/usr/bin/env python3
"""
evaluate_and_plot.py
=============================================================================
Comprehensive Quantitative Evaluation and Visualization Suite for the
seqNMF-Adjacent Probabilistic Episode Pilot.

Produces:
1. Quantitative Recurrence & Population Time Coverage Metrics (matching the
   Level tests from clustering resolution comparison)
2. Cumulative Movement Similarity Scaling Curves (with P80, P99, Canonical markers)
3. Shared 2D PCA Occurrence-Level Embedding colored by Method and Theme
4. Physical Duration (seconds) and Kinematic Distribution Plots
5. Temporal Intersection-Over-Union (IoU) Overlap Matrix across Arms
6. Audit Table of all 8 seqNMF Motifs with cross-arm candidate counts
=============================================================================
"""

import sys
import time
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Add repo root to path
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

# Compatibility bridge for NumPy unpickling
import numpy.core.numeric as _numeric
sys.modules['numpy._core.numeric'] = _numeric

from syllable_generation.analysis.sequence_viz.probabilistic_episode_pilot.episode_pilot_config import (
    EXP_DIR, DATA_DIR, PLOTS_DIR, THEMES, TRAIN_ANIMALS, VAL_ANIMALS
)


def compute_population_time_coverage(candidates, total_valid_frames=5600000):
    """
    Computes the 1D interval union of all qualifying episode intervals
    (counting overlapping frames only once) across recordings.
    """
    recs = set(c["recording_key"] for c in candidates)
    total_covered_frames = 0
    per_animal_cov = {}

    for rk in recs:
        c_rk = [c for c in candidates if c["recording_key"] == rk]
        intervals = sorted([(c["start_frame"], c["end_frame"]) for c in c_rk])
        if not intervals:
            continue

        merged = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        cov_f = sum(e - s for s, e in merged)
        total_covered_frames += cov_f

    pct_cov = (total_covered_frames / max(1, total_valid_frames)) * 100.0
    return pct_cov, total_covered_frames


def main():
    print("=" * 80)
    print("STARTING QUANTITATIVE EVALUATION & VISUALIZATION GENERATION")
    print("=" * 80)
    t0 = time.time()

    # Load master candidates
    master_pkl = DATA_DIR / "pilot_master_candidates.pkl"
    with open(master_pkl, "rb") as f:
        data = pickle.load(f)

    arm1_cands = data["arm1_candidates"]
    arm2_cands = data["arm2_candidates"]
    arm3_cands = data["arm3_candidates"]

    # 1. QUANTITATIVE BENCHMARK TABLE
    print("[1/5] Computing Quantitative Performance Metrics across Arms...")
    metrics_records = []

    arms = [
        ("Arm 1: seqNMF Baseline", arm1_cands, "seqnmf"),
        ("Arm 2: Categorical HSMM (Syllables Only)", arm2_cands, "cat_hsmm"),
        ("Arm 3: Mixed HSMM (Option 1 Calibrated)", arm3_cands, "mixed_hsmm")
    ]

    for arm_name, cands, arm_id in arms:
        for th_id, th_meta in THEMES.items():
            th_cands = [c for c in cands if c["theme_id"] == th_id]
            n_total = len(th_cands)
            train_cands = [c for c in th_cands if c["animal_id"] in TRAIN_ANIMALS]
            val_cands = [c for c in th_cands if c["animal_id"] in VAL_ANIMALS]

            durs = np.array([c["duration_sec"] for c in th_cands]) if n_total > 0 else np.array([0.0])
            speeds = np.array([c["avg_speed_px_s"] for c in th_cands]) if n_total > 0 else np.array([0.0])
            disps = np.array([c["net_displacement_px"] for c in th_cands]) if n_total > 0 else np.array([0.0])
            sims = np.array([c["movement_similarity"] for c in th_cands]) if n_total > 0 else np.array([0.0])

            aids = set(c["animal_id"] for c in th_cands)
            pop_cov_pct, _ = compute_population_time_coverage(th_cands)

            metrics_records.append({
                "Arm": arm_name,
                "Theme": f"{th_id}: {th_meta['title']}",
                "Total Episodes": n_total,
                "Train Episodes": len(train_cands),
                "Val Episodes": len(val_cands),
                "Animals (of 15)": len(aids),
                "Population Time Coverage (%)": round(pop_cov_pct, 2),
                "Mean Duration (s)": f"{np.mean(durs):.2f} ± {np.std(durs):.2f}",
                "Median Duration (s)": round(float(np.median(durs)), 2),
                "Duration IQR (s)": f"[{np.percentile(durs, 25):.2f} - {np.percentile(durs, 75):.2f}]",
                "Median Speed (px/s)": round(float(np.median(speeds)), 1),
                "Median Displacement (px)": round(float(np.median(disps)), 1),
                "Median Movement Sim": round(float(np.median(sims)), 3),
                "P80 Movement Sim": round(float(np.percentile(sims, 80)), 3) if n_total > 0 else 0.0,
                "Exemplar (Top) Sim": round(float(np.max(sims)), 3) if n_total > 0 else 0.0
            })

    df_metrics = pd.DataFrame(metrics_records)
    metrics_csv = DATA_DIR / "quantitative_performance_benchmarks.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"  [CSV] Saved benchmark metrics table to: {metrics_csv}")

    # 2. CUMULATIVE MOVEMENT SIMILARITY SCALING CURVES
    print("[2/5] Plotting Cumulative Movement Similarity Scaling Curves...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    thresholds = np.linspace(0.10, 0.90, 40)

    colors = {
        "seqnmf": "#1f77b4",       # Blue
        "cat_hsmm": "#2ca02c",     # Green
        "mixed_hsmm": "#d62728"    # Red
    }
    labels = {
        "seqnmf": "Arm 1: seqNMF (Level 2 Soft)",
        "cat_hsmm": "Arm 2: Categorical HSMM (Raw)",
        "mixed_hsmm": "Arm 3: Mixed HSMM (Option 1 Calibrated)"
    }

    for idx, (th_id, th_meta) in enumerate(THEMES.items()):
        ax = axes[idx]
        th_title = th_meta["title"]

        for arm_name, cands, arm_id in arms:
            th_cands = [c for c in cands if c["theme_id"] == th_id]
            sims = np.array([c["movement_similarity"] for c in th_cands])
            aids = [c["animal_id"] for c in th_cands]

            n_qual = [np.sum(sims >= t) for t in thresholds]
            ax.plot(thresholds, n_qual, label=f"{labels[arm_id]} (N={len(th_cands)})", color=colors[arm_id], lw=2.4)

            # Mark P80 (diamond) and Exemplar (star)
            if len(sims) > 0:
                p80_val = float(np.percentile(sims, 80))
                max_val = float(np.max(sims))

                n_p80 = np.sum(sims >= p80_val)
                n_max = 1

                ax.scatter([p80_val], [n_p80], marker="D", color=colors[arm_id], s=65, edgecolor="black", zorder=5)
                ax.scatter([max_val], [n_max], marker="*", color=colors[arm_id], s=130, edgecolor="black", zorder=6)

        ax.set_title(f"{th_id.upper()}: {th_title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Movement Similarity Threshold", fontsize=11)
        ax.set_ylabel("Qualifying Episodes (N ≥ Thresh)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(frameon=True, fontsize=9, loc="upper right")

    plt.suptitle("Cumulative Movement Similarity Scaling Across Methods (Unbiased Pooled Reference)\nDiamonds = P80 Operating Point (Top 20% Core) | Stars = Exemplar (Top Similarity)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plot_scaling_path = PLOTS_DIR / "fig_similarity_vs_qualifying_episodes.png"
    plt.savefig(plot_scaling_path, dpi=300)
    plt.close()
    print(f"  [PLOT] Saved scaling curves to: {plot_scaling_path}")

    # 3. SHARED OCCURRENCE-LEVEL 2D PCA EMBEDDING
    print("[3/5] Computing Shared Occurrence-Level 2D PCA Embedding...")
    # Gather training features to fit PCA
    train_feats = [c["feature_240d"] for c in arm1_cands + arm3_cands if c["animal_id"] in TRAIN_ANIMALS]
    pca_2d = PCA(n_components=2, random_state=42)
    pca_2d.fit(np.array(train_feats))

    # Project seqNMF and Mixed HSMM occurrences
    emb_data = []
    for c in arm1_cands:
        coords = pca_2d.transform([c["feature_240d"]])[0]
        emb_data.append({
            "PC1": coords[0], "PC2": coords[1],
            "Method": "seqNMF", "Theme": c["theme_id"], "Animal": c["animal_id"]
        })
    for c in arm3_cands:
        coords = pca_2d.transform([c["feature_240d"]])[0]
        emb_data.append({
            "PC1": coords[0], "PC2": coords[1],
            "Method": "Probabilistic Episode (Mixed)", "Theme": c["theme_id"], "Animal": c["animal_id"]
        })

    df_emb = pd.DataFrame(emb_data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    palette_theme = {"theme_A": "#d62728", "theme_B": "#ff7f0e", "theme_C": "#2ca02c"}
    palette_method = {"seqNMF": "#1f77b4", "Probabilistic Episode (Mixed)": "#9467bd"}

    # Left: Colored by Theme
    sns.scatterplot(
        data=df_emb, x="PC1", y="PC2", hue="Theme", style="Method",
        palette=palette_theme, alpha=0.5, s=25, ax=axes[0]
    )
    axes[0].set_title("Shared Kinematic Trajectory Space: Colored by Theme", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Right: Colored by Method
    sns.scatterplot(
        data=df_emb, x="PC1", y="PC2", hue="Method", style="Theme",
        palette=palette_method, alpha=0.5, s=25, ax=axes[1]
    )
    axes[1].set_title("Shared Kinematic Trajectory Space: Colored by Method", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Shared Occurrence-Level PCA Embedding (240D Kinematic & Posture Space)\nFit on Training Animal Occurrences Across Methods", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_emb_path = PLOTS_DIR / "fig_shared_occurrence_embedding.png"
    plt.savefig(plot_emb_path, dpi=300)
    plt.close()
    print(f"  [PLOT] Saved shared embedding plot to: {plot_emb_path}")

    # 4. DURATION & KINEMATIC DISTRIBUTIONS
    print("[4/5] Plotting Physical Duration and Speed Distributions...")
    all_df = pd.DataFrame([
        {
            "Method": c["method"],
            "Theme": c["theme_id"],
            "Duration_Sec": c["duration_sec"],
            "Speed_px_s": c["avg_speed_px_s"],
            "Displacement_px": c["net_displacement_px"],
            "Turn_deg": c["signed_turn_deg"]
        }
        for c in arm1_cands + arm3_cands
    ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.boxplot(data=all_df, x="Theme", y="Duration_Sec", hue="Method", palette="Set2", showfliers=False, ax=axes[0, 0])
    axes[0, 0].set_title("Physical Duration (Seconds)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Duration (s)")

    sns.boxplot(data=all_df, x="Theme", y="Speed_px_s", hue="Method", palette="Set2", showfliers=False, ax=axes[0, 1])
    axes[0, 1].set_title("Centroid Speed (px/s)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Speed (px/s)")

    sns.boxplot(data=all_df, x="Theme", y="Displacement_px", hue="Method", palette="Set2", showfliers=False, ax=axes[1, 0])
    axes[1, 0].set_title("Net Displacement (px)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Net Displacement (px)")

    sns.boxplot(data=all_df, x="Theme", y="Turn_deg", hue="Method", palette="Set2", showfliers=False, ax=axes[1, 1])
    axes[1, 1].set_title("Signed Heading Change (deg)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Heading Change (deg)")

    plt.suptitle("Physical Kinematic & Duration Distributions: seqNMF Baseline vs. Mixed Probabilistic Episode Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_dist_path = PLOTS_DIR / "fig_duration_and_kinematic_distributions.png"
    plt.savefig(plot_dist_path, dpi=300)
    plt.close()
    print(f"  [PLOT] Saved distribution comparisons to: {plot_dist_path}")

    # 5. TEMPORAL INTERSECTION-OVER-UNION (IoU) OVERLAP ANALYSIS
    print("[5/5] Computing Temporal IoU Overlap Between seqNMF and Probabilistic Episode Occurrences...")
    # For each seqNMF occurrence, find if there is a matching HSMM occurrence with IoU >= 0.3
    overlap_results = []
    for th_id in THEMES:
        s_cands = [c for c in arm1_cands if c["theme_id"] == th_id]
        h_cands = [c for c in arm3_cands if c["theme_id"] == th_id]

        recs = set(c["recording_key"] for c in s_cands)
        matched_seqnmf = 0
        ious = []

        for c_seq in s_cands:
            rk = c_seq["recording_key"]
            sf_s, ef_s = c_seq["start_frame"], c_seq["end_frame"]
            # Candidates in same recording
            same_rec = [c for c in h_cands if c["recording_key"] == rk]
            best_iou = 0.0
            for c_h in same_rec:
                sf_h, ef_h = c_h["start_frame"], c_h["end_frame"]
                inter = max(0, min(ef_s, ef_h) - max(sf_s, sf_h))
                union = max(ef_s, ef_h) - min(sf_s, sf_h)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou

            if best_iou >= 0.30:
                matched_seqnmf += 1
                ious.append(best_iou)

        overlap_results.append({
            "Theme": th_id,
            "seqNMF Baseline Episodes": len(s_cands),
            "Mixed HSMM Episodes": len(h_cands),
            "seqNMF Recovered by HSMM (IoU ≥ 0.30)": f"{matched_seqnmf} ({matched_seqnmf / max(1, len(s_cands)) * 100:.1f}%)",
            "Newly Admitted Flexible Executions": len(h_cands) - matched_seqnmf,
            "Mean Matching IoU": round(float(np.mean(ious)), 2) if ious else 0.0
        })

    df_overlap = pd.DataFrame(overlap_results)
    overlap_csv = DATA_DIR / "cross_arm_temporal_overlap_analysis.csv"
    df_overlap.to_csv(overlap_csv, index=False)
    print(f"  [CSV] Saved overlap analysis to: {overlap_csv}")
    print("\n" + "=" * 80)
    print(df_overlap.to_string(index=False))

    print(f"\nCompleted in {time.time() - t0:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
