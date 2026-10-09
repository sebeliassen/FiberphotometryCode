"""
Vibe Bimodality Audit: Are any of the 9 vibes secretly bimodal?

For each vibe, we test whether the distribution of frame-level kinematics
(speed, distance-to-center, angular velocity, cadence) is unimodal or bimodal.

Methods:
  1. GMM BIC comparison: fit 1-component vs 2-component Gaussian Mixture Model.
     If BIC(2) << BIC(1), the vibe has two distinct kinematic sub-modes.
  2. Per-bout syllable histogram cosine similarity to cluster prototype.
     Identifies which bouts are most/least representative.

Output:
  - Console summary table
  - Diagnostic scatter + marginal density plots per vibe
  - CSV with per-bout representativeness scores
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine as cosine_dist

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")


def load_frame_level_data():
    """Load sessions and reconstruct frame-level vibe assignments with kinematics."""
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        sessions = pickle.load(f)

    intervals_csv = os.path.join(OUT_DIR, "session_strategy_intervals.csv")
    df_intervals = pd.read_csv(intervals_csv)

    # Build session lookup
    session_lookup = {}
    for s in sessions:
        session_lookup[s.session_id] = s

    all_rows = []
    for _, interval in df_intervals.iterrows():
        sid = interval["session_id"]
        # Try to find session (handle _pre suffix)
        session = session_lookup.get(sid)
        if session is None:
            base_sid = sid.replace("_pre", "").replace("_post", "")
            session = session_lookup.get(base_sid)
        if session is None:
            continue

        df = session.all_data_df
        s_frame = int(interval["start_frame"])
        e_frame = min(int(interval["end_frame"]), len(df))
        vibe_id = int(interval["strategy_id"])

        sub = df.iloc[s_frame:e_frame].copy()
        sub["vibe_id"] = vibe_id
        sub["session_id"] = sid
        sub["bout_start_frame"] = s_frame
        sub["bout_end_frame"] = e_frame
        sub["bout_duration_sec"] = interval["end_time_sec"] - interval["start_time_sec"]

        # Compute kinematics
        vel = sub["velocity_px_s"].values if "velocity_px_s" in sub.columns else np.zeros(len(sub))
        cx = sub["centroid_x"].values if "centroid_x" in sub.columns else np.zeros(len(sub))
        cy = sub["centroid_y"].values if "centroid_y" in sub.columns else np.zeros(len(sub))
        c_mean_x, c_mean_y = np.nanmean(cx), np.nanmean(cy)
        dist_center = np.sqrt((cx - c_mean_x)**2 + (cy - c_mean_y)**2)
        ang_vel = np.abs(sub["angular_velocity"].values) if "angular_velocity" in sub.columns else np.zeros(len(sub))
        sylls = sub["syllable"].values if "syllable" in sub.columns else np.zeros(len(sub))
        cadence = np.zeros(len(sub))
        if len(sylls) > 1:
            cadence[1:] = (sylls[1:] != sylls[:-1]).astype(float)

        for i in range(len(sub)):
            all_rows.append({
                "vibe_id": vibe_id,
                "session_id": sid,
                "bout_start_frame": s_frame,
                "velocity": vel[i],
                "dist_center": dist_center[i],
                "angular_vel": ang_vel[i],
                "cadence": cadence[i],
                "syllable": sylls[i] if i < len(sylls) else -1,
                "bout_duration_sec": interval["end_time_sec"] - interval["start_time_sec"],
            })

    print(f"Loaded {len(all_rows)} frame-level kinematic observations across all vibes")
    return pd.DataFrame(all_rows)


def test_bimodality_gmm(data_2d, vibe_id):
    """
    Fit 1-component vs 2-component GMM on standardized kinematic features.
    Returns BIC difference and component means/weights.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(data_2d)

    gmm1 = GaussianMixture(n_components=1, random_state=42, covariance_type="full")
    gmm2 = GaussianMixture(n_components=2, random_state=42, covariance_type="full")

    gmm1.fit(X)
    gmm2.fit(X)

    bic1 = gmm1.bic(X)
    bic2 = gmm2.bic(X)
    delta_bic = bic1 - bic2  # positive => 2 components preferred

    # Get component weights for 2-component model
    weights = gmm2.weights_
    means_scaled = scaler.inverse_transform(gmm2.means_)

    return {
        "bic_1comp": bic1,
        "bic_2comp": bic2,
        "delta_bic": delta_bic,
        "bimodal": delta_bic > 10,  # standard BIC threshold
        "comp_weights": weights,
        "comp_means": means_scaled,
    }


def compute_bout_representativeness(frame_df):
    """
    For each bout within each vibe, compute cosine similarity of the bout's
    syllable frequency histogram to the overall cluster prototype histogram.
    """
    results = []
    # Pre-compute global syllable vocabulary (drop NA)
    all_sylls_series = frame_df["syllable"].dropna()
    all_sylls = np.unique(all_sylls_series.values.astype(float))

    for vibe_id in sorted(frame_df["vibe_id"].unique()):
        vibe_mask = frame_df["vibe_id"] == vibe_id
        vibe_sylls_raw = frame_df.loc[vibe_mask, "syllable"].values
        # Convert to float numpy, coerce NA to NaN
        vibe_sylls = pd.to_numeric(pd.Series(vibe_sylls_raw), errors="coerce").values
        vibe_sylls_clean = vibe_sylls[~np.isnan(vibe_sylls)]
        
        # Cluster prototype histogram
        proto_hist = np.zeros(len(all_sylls))
        for i, s in enumerate(all_sylls):
            proto_hist[i] = np.sum(vibe_sylls_clean == s)
        proto_hist = proto_hist / (proto_hist.sum() + 1e-12)

        # Per-bout histograms
        vibe_df = frame_df[vibe_mask]
        bout_keys = vibe_df.groupby(["session_id", "bout_start_frame"]).groups
        for (sid, bsf), idxs in bout_keys.items():
            bout_sylls_raw = vibe_df.loc[idxs, "syllable"].values
            bout_sylls = pd.to_numeric(pd.Series(bout_sylls_raw), errors="coerce").values
            bout_sylls_clean = bout_sylls[~np.isnan(bout_sylls)]
            bout_hist = np.zeros(len(all_sylls))
            for i, s in enumerate(all_sylls):
                bout_hist[i] = np.sum(bout_sylls_clean == s)
            bout_hist = bout_hist / (bout_hist.sum() + 1e-12)

            cos_sim = 1.0 - cosine_dist(proto_hist, bout_hist)
            bout_dur = vibe_df.loc[idxs, "bout_duration_sec"].iloc[0]
            bout_vel_mean = vibe_df.loc[idxs, "velocity"].mean()

            results.append({
                "vibe_id": vibe_id,
                "session_id": sid,
                "bout_start_frame": bsf,
                "bout_duration_sec": bout_dur,
                "cosine_sim_to_prototype": cos_sim,
                "mean_velocity": bout_vel_mean,
                "n_frames": len(idxs),
            })

    return pd.DataFrame(results)


def plot_bimodality_diagnostics(frame_df, bimodality_results, out_path):
    """
    Create a 3x3 grid of scatter/density plots (one per vibe) showing
    Speed vs Distance-to-Center with GMM contours if bimodal.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle("Vibe Bimodality Audit: Speed vs Distance-to-Center", fontsize=16, fontweight="bold")

    for vibe_id in range(9):
        ax = axes[vibe_id // 3, vibe_id % 3]
        vibe_mask = frame_df["vibe_id"] == vibe_id
        vel = frame_df.loc[vibe_mask, "velocity"].values
        dist = frame_df.loc[vibe_mask, "dist_center"].values

        # Subsample for plotting (max 5000 points)
        n = len(vel)
        if n > 5000:
            idx = np.random.RandomState(42).choice(n, 5000, replace=False)
            vel_plot, dist_plot = vel[idx], dist[idx]
        else:
            vel_plot, dist_plot = vel, dist

        result = bimodality_results[vibe_id]
        color = "red" if result["bimodal"] else "steelblue"
        label = "BIMODAL" if result["bimodal"] else "UNIMODAL"

        ax.scatter(vel_plot, dist_plot, s=1, alpha=0.15, c=color)
        ax.set_title(f"Vibe {vibe_id} ({label})\n"
                     f"ΔBIC={result['delta_bic']:.0f}  "
                     f"w=[{result['comp_weights'][0]:.2f}, {result['comp_weights'][1]:.2f}]",
                     fontsize=10, color="darkred" if result["bimodal"] else "darkblue")
        ax.set_xlabel("Speed (px/s)", fontsize=8)
        ax.set_ylabel("Dist to Center (px)", fontsize=8)

        # Mark component centers
        means = result["comp_means"]
        ax.plot(means[0, 0], means[0, 1], "k*", markersize=14, label=f"Mode A ({result['comp_weights'][0]:.0%})")
        ax.plot(means[1, 0], means[1, 1], "w*", markersize=14, markeredgecolor="black",
                label=f"Mode B ({result['comp_weights'][1]:.0%})")
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bimodality diagnostic plot to: {out_path}")


def main():
    print("=" * 70)
    print("VIBE BIMODALITY AUDIT: Testing 9 vibes for internal sub-modes")
    print("=" * 70)

    # 1. Load frame-level data
    frame_df = load_frame_level_data()

    # 2. Test bimodality per vibe (Speed × Distance-to-Center)
    print("\n--- GMM Bimodality Test (Speed × Dist-to-Center) ---")
    bimodality_results = {}
    summary_rows = []

    for vibe_id in range(9):
        vibe_mask = frame_df["vibe_id"] == vibe_id
        vel = frame_df.loc[vibe_mask, "velocity"].values
        dist = frame_df.loc[vibe_mask, "dist_center"].values

        # Remove NaN
        valid = ~(np.isnan(vel) | np.isnan(dist))
        data_2d = np.column_stack([vel[valid], dist[valid]])

        result = test_bimodality_gmm(data_2d, vibe_id)
        bimodality_results[vibe_id] = result

        status = "⚠️ BIMODAL" if result["bimodal"] else "✅ UNIMODAL"
        w0, w1 = result["comp_weights"]
        m0, m1 = result["comp_means"]
        print(f"  Vibe {vibe_id}: {status}  ΔBIC={result['delta_bic']:>10.0f}  "
              f"weights=[{w0:.2f}, {w1:.2f}]  "
              f"Mode A: speed={m0[0]:.1f} dist={m0[1]:.1f}  "
              f"Mode B: speed={m1[0]:.1f} dist={m1[1]:.1f}")

        summary_rows.append({
            "vibe_id": vibe_id,
            "n_frames": int(vibe_mask.sum()),
            "delta_bic": result["delta_bic"],
            "bimodal": result["bimodal"],
            "weight_mode_a": w0,
            "weight_mode_b": w1,
            "mode_a_speed": m0[0],
            "mode_a_dist_center": m0[1],
            "mode_b_speed": m1[0],
            "mode_b_dist_center": m1[1],
        })

    # 3. Also test on Angular Velocity × Cadence
    print("\n--- GMM Bimodality Test (Angular Vel × Cadence) ---")
    bimodality_results_ang = {}
    for vibe_id in range(9):
        vibe_mask = frame_df["vibe_id"] == vibe_id
        ang = frame_df.loc[vibe_mask, "angular_vel"].values
        cad = frame_df.loc[vibe_mask, "cadence"].values
        valid = ~(np.isnan(ang) | np.isnan(cad))
        data_2d = np.column_stack([ang[valid], cad[valid]])
        result = test_bimodality_gmm(data_2d, vibe_id)
        bimodality_results_ang[vibe_id] = result
        status = "⚠️ BIMODAL" if result["bimodal"] else "✅ UNIMODAL"
        print(f"  Vibe {vibe_id}: {status}  ΔBIC={result['delta_bic']:>10.0f}")

        summary_rows[vibe_id]["delta_bic_ang_cad"] = result["delta_bic"]
        summary_rows[vibe_id]["bimodal_ang_cad"] = result["bimodal"]

    # 4. Compute bout-level representativeness
    print("\n--- Computing per-bout cosine similarity to cluster prototypes ---")
    bout_df = compute_bout_representativeness(frame_df)
    bout_csv_path = os.path.join(OUT_DIR, "vibe_bout_representativeness.csv")
    bout_df.to_csv(bout_csv_path, index=False)
    print(f"Saved bout representativeness to: {bout_csv_path}")

    # Print top 3 most representative bouts per vibe
    print("\n--- Top 3 Most Representative Bouts Per Vibe ---")
    for vibe_id in range(9):
        vibe_bouts = bout_df[bout_df["vibe_id"] == vibe_id].sort_values("cosine_sim_to_prototype", ascending=False)
        print(f"\n  Vibe {vibe_id} (N={len(vibe_bouts)} bouts):")
        for _, row in vibe_bouts.head(3).iterrows():
            print(f"    Session: {row['session_id'][:40]}  "
                  f"start_frame={int(row['bout_start_frame'])}  "
                  f"dur={row['bout_duration_sec']:.0f}s  "
                  f"cos_sim={row['cosine_sim_to_prototype']:.3f}  "
                  f"avg_speed={row['mean_velocity']:.1f} px/s")

    # 5. Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(OUT_DIR, "vibe_bimodality_audit.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved bimodality summary to: {summary_csv_path}")

    # 6. Plot diagnostics
    plot_path = os.path.join(OUT_DIR, "vibe_bimodality_diagnostic.png")
    plot_bimodality_diagnostics(frame_df, bimodality_results, plot_path)

    # 7. Final summary
    print("\n" + "=" * 70)
    n_bimodal_speed = sum(1 for r in bimodality_results.values() if r["bimodal"])
    n_bimodal_ang = sum(1 for r in bimodality_results_ang.values() if r["bimodal"])
    print(f"SUMMARY: {n_bimodal_speed}/9 vibes bimodal on Speed×Dist, "
          f"{n_bimodal_ang}/9 vibes bimodal on AngVel×Cadence")
    print("=" * 70)


if __name__ == "__main__":
    main()
