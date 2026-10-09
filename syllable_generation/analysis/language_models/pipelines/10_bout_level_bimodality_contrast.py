"""
Bout-Level Kinematic + Postural (Syllable PCA) Bimodality & Contrastive Audit
Located at: syllable_generation/analysis/language_models/pipelines/10_bout_level_bimodality_contrast.py

This script performs an Enriched Bout Audit:
1. Summarizes each strategy bout into bout-level mean kinematic parameters
   (mean speed, mean dist to center, mean angular velocity, mean cadence).
2. Computes the 100-element substate syllable frequency vector P(syllable | bout)
   for each bout, and projects it into 2 principal components (Syllables_PC1, Syllables_PC2).
3. Performs GMM BIC comparison (1-comp vs 2-comp) on ENRICHED 6D BOUT VECTORS
   [speed, dist_center, ang_vel, cadence, Syllable_PC1, Syllable_PC2] to test if any vibe
   genuinely contains posture/syllable sub-modes or kinematic sub-types.
"""

import os
import pickle
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")

def main():
    print("=" * 75)
    print("ENRICHED BOUT AUDIT: Kinematics + Syllable Composition PCA")
    print("=" * 75)

    # Load raw sessions
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        sessions = pickle.load(f)

    session_lookup = {s.session_id: s for s in sessions}

    intervals_csv = os.path.join(OUT_DIR, "session_strategy_intervals.csv")
    df_intervals = pd.read_csv(intervals_csv)

    # Determine global max syllable index for histogram vector
    max_syll_id = 0
    for s in sessions:
        if "syllable" in s.all_data_df.columns:
            m = s.all_data_df["syllable"].max()
            if pd.notna(m) and m > max_syll_id:
                max_syll_id = int(m)
    n_syll_dim = max(100, max_syll_id + 1)

    print(f"Loaded {len(df_intervals)} total strategy bouts across {len(session_lookup)} sessions (Syllable Dim: {n_syll_dim})")

    # Compute bout-level kinematic + syllable histogram features
    bout_summaries = []
    bout_syll_hists = []

    for idx, row in df_intervals.iterrows():
        sid = row["session_id"]
        session = session_lookup.get(sid)
        if session is None:
            base_sid = sid.replace("_pre", "").replace("_post", "")
            session = session_lookup.get(base_sid)
        if session is None:
            continue

        df = session.all_data_df
        s_frame = int(row["start_frame"])
        e_frame = min(int(row["end_frame"]), len(df))

        if e_frame <= s_frame:
            continue

        sub = df.iloc[s_frame:e_frame]

        vel = sub["velocity_px_s"].values if "velocity_px_s" in sub.columns else np.zeros(len(sub))
        cx = sub["centroid_x"].values if "centroid_x" in sub.columns else np.zeros(len(sub))
        cy = sub["centroid_y"].values if "centroid_y" in sub.columns else np.zeros(len(sub))
        c_mean_x, c_mean_y = np.nanmean(cx), np.nanmean(cy)
        dist_c = np.sqrt((cx - c_mean_x)**2 + (cy - c_mean_y)**2)
        ang_v = np.abs(sub["angular_velocity"].values) if "angular_velocity" in sub.columns else np.zeros(len(sub))

        sylls = sub["syllable"].values if "syllable" in sub.columns else np.zeros(len(sub))
        cadence = np.zeros(len(sub))
        if len(sylls) > 1:
            cadence[1:] = (sylls[1:] != sylls[:-1]).astype(float)

        # Compute 100-dim syllable frequency vector
        syll_hist = np.zeros(n_syll_dim, dtype=float)
        sylls_clean = pd.to_numeric(pd.Series(sylls), errors="coerce").dropna().values.astype(int)
        for s_id in sylls_clean:
            if 0 <= s_id < n_syll_dim:
                syll_hist[s_id] += 1.0
        sum_h = np.sum(syll_hist)
        if sum_h > 0:
            syll_hist /= sum_h

        bout_syll_hists.append(syll_hist)

        bout_summaries.append({
            "vibe_id": int(row["strategy_id"]),
            "session_id": sid,
            "start_frame": s_frame,
            "duration_sec": row["end_time_sec"] - row["start_time_sec"],
            "mean_speed": np.nanmean(vel),
            "std_speed": np.nanstd(vel),
            "mean_dist_center": np.nanmean(dist_c),
            "mean_angular_vel": np.nanmean(ang_v),
            "mean_cadence": np.nanmean(cadence),
        })

    bout_df = pd.DataFrame(bout_summaries)
    syll_matrix = np.array(bout_syll_hists)

    # Fit PCA on Syllable Histograms (extract top 2 PCs)
    pca = PCA(n_components=2, random_state=42)
    syll_pcs = pca.fit_transform(syll_matrix)
    bout_df["syllable_pc1"] = syll_pcs[:, 0]
    bout_df["syllable_pc2"] = syll_pcs[:, 1]

    var_exp = pca.explained_variance_ratio_
    print(f"PCA on Syllable Histograms: PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}% (Total: {(var_exp[0]+var_exp[1])*100:.1f}%)")

    # Run GMM BIC audit on ENRICHED 6D BOUT VECTORS
    print("\n--- GMM Bimodality Test on ENRICHED 6D BOUT VECTORS (Kinematics + Syllable PCs) ---")
    bimodality_results = []

    feature_cols = ["mean_speed", "mean_dist_center", "mean_angular_vel", "mean_cadence", "syllable_pc1", "syllable_pc2"]

    for vibe_id in range(9):
        v_df = bout_df[bout_df["vibe_id"] == vibe_id]
        n_bouts = len(v_df)

        if n_bouts < 6:
            print(f"  Vibe {vibe_id}: Skipped (Too few bouts: N={n_bouts})")
            continue

        features = v_df[feature_cols].values
        valid_mask = ~np.isnan(features).any(axis=1)
        X = features[valid_mask]

        if len(X) < 6:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        gmm1 = GaussianMixture(n_components=1, random_state=42)
        gmm2 = GaussianMixture(n_components=2, random_state=42)

        gmm1.fit(X_scaled)
        gmm2.fit(X_scaled)

        bic1 = gmm1.bic(X_scaled)
        bic2 = gmm2.bic(X_scaled)
        delta_bic = bic1 - bic2 # Positive => 2 components preferred

        is_bimodal = delta_bic > 6.0 # Standard BIC strong evidence threshold
        status = "⚠️ BOUT-BIMODAL" if is_bimodal else "✅ UNIMODAL BOUTS"

        weights = gmm2.weights_ if is_bimodal else [1.0, 0.0]

        print(f"  Vibe {vibe_id} (N={n_bouts:2d} bouts): {status} | ΔBIC={delta_bic:>6.2f} | weights=[{weights[0]:.2f}, {weights[1]:.2f}] | Speed={v_df['mean_speed'].mean():.1f} px/s")

        bimodality_results.append({
            "vibe_id": vibe_id,
            "n_bouts": n_bouts,
            "delta_bic_6d": delta_bic,
            "is_bout_bimodal_6d": is_bimodal,
            "weight_mode_a": weights[0],
            "weight_mode_b": weights[1],
            "grand_mean_speed": v_df["mean_speed"].mean(),
            "grand_mean_dist_center": v_df["mean_dist_center"].mean(),
            "grand_mean_ang_vel": v_df["mean_angular_vel"].mean(),
            "grand_mean_cadence": v_df["mean_cadence"].mean(),
        })

    res_df = pd.DataFrame(bimodality_results)
    out_csv = os.path.join(OUT_DIR, "enriched_bout_bimodality_summary.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved enriched bout-level audit summary to: {out_csv}")

    # Save enriched bout kinematics dataframe with PCs
    bout_df.to_csv(os.path.join(OUT_DIR, "enriched_bout_features_w_pcs.csv"), index=False)

    # Plot Enriched 2D PCA Syllable Space per Vibe
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle("Enriched Bout Syllable Space (PC1 vs PC2 per Vibe)", fontsize=15, fontweight="bold")

    for vibe_id in range(9):
        ax = axes[vibe_id // 3, vibe_id % 3]
        v_df = bout_df[bout_df["vibe_id"] == vibe_id]
        if v_df.empty:
            continue

        pc1 = v_df["syllable_pc1"].values
        pc2 = v_df["syllable_pc2"].values
        speeds = v_df["mean_speed"].values

        sc = ax.scatter(pc1, pc2, c=speeds, cmap="plasma", s=60, alpha=0.8, edgecolors="k")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Speed (px/s)", fontsize=8)

        is_b = res_df.loc[res_df["vibe_id"] == vibe_id, "is_bout_bimodal_6d"].values
        b_str = "BIMODAL" if (len(is_b) > 0 and is_b[0]) else "UNIMODAL"
        color_title = "darkred" if b_str == "BIMODAL" else "darkblue"

        ax.set_title(f"Vibe {vibe_id} ({b_str})\nN={len(v_df)} bouts", fontsize=11, fontweight="bold", color=color_title)
        ax.set_xlabel("Syllable PC1", fontsize=9)
        ax.set_ylabel("Syllable PC2", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(OUT_DIR, "enriched_syllable_pc_space_per_vibe.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved enriched Syllable PC diagnostic plot to: {plot_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
