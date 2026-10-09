"""
Multi-Scale Exemplar Strategy Mining & Modality Audit (100% -> 50% -> 25% -> 10% -> 5% -> 2% -> 1%)
File: syllable_generation/analysis/language_models/pipelines/23_exemplar_modality_and_clustering_audit.py

Evaluates whether filtering strategy windows to their top archetypes/exemplars
(from 100% down to 1%) sharpens attractor basins, eliminates boundary noise,
amplifies cluster separation (Silhouette, Davies-Bouldin), and increases
between-strategy physical distinction (ANOVA eta^2 on 4 pillars).

Models evaluated:
  - KMeans (K=8)
  - KMeans (K=16)
  - GMM (K=8)
  - GMM (K=16)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis, f_oneway

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation"))
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models"))

OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
MODEL_DIR = os.path.join(OUT_DIR, "production_models")
AUDIT_DIR = os.path.join(OUT_DIR, "exemplar_audit")
os.makedirs(AUDIT_DIR, exist_ok=True)

# 1. Load Datasets
print("Loading session datasets and window scalar cache...")
pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
with open(pkl_path, "rb") as f:
    raw_sessions = pickle.load(f)

with open(os.path.join(OUT_DIR, "window_scalars_cache.pkl"), "rb") as f:
    wdata = pickle.load(f)
raw_trans_inputs = wdata["trans_inputs"]
win_meta = wdata["meta"]
win_scalars = wdata["scalars"]

# Pre-extract session kinematics
print("Pre-indexing session kinematic arrays in memory...")
session_kinematics = {}
for session in raw_sessions:
    sid = session.session_id
    df = session.all_data_df
    
    vel = df["velocity_px_s"].values if "velocity_px_s" in df.columns else np.zeros(len(df))
    cx = df["centroid_x"].values if "centroid_x" in df.columns else np.zeros(len(df))
    cy = df["centroid_y"].values if "centroid_y" in df.columns else np.zeros(len(df))
    c_mean_x, c_mean_y = np.nanmean(cx), np.nanmean(cy)
    dist_center = np.sqrt((cx - c_mean_x)**2 + (cy - c_mean_y)**2)
    
    h = df["heading"].values if "heading" in df.columns else np.zeros(len(df))
    dh = np.diff(h, prepend=h[0])
    dh_wrapped = (dh + np.pi) % (2 * np.pi) - np.pi
    ang_vel = np.clip(np.abs(dh_wrapped) * 80.025, 0.0, 25.0)
    
    session_kinematics[sid] = {
        "velocity": vel.astype(np.float32),
        "dist_center": dist_center.astype(np.float32),
        "angular_vel": ang_vel.astype(np.float32),
        "len": len(df)
    }

# Extract 4-Pillar Features per window
print("Extracting 4-pillar behavioral metrics for all 14,092 windows...")
window_records = []
for w_i, (w, s) in enumerate(zip(win_meta, win_scalars)):
    sid = w["session_id"]
    sf = int(w["start_frame"])
    ef = min(int(w["end_frame"]), session_kinematics[sid]["len"])
    
    if ef <= sf:
        continue
        
    v_arr = session_kinematics[sid]["velocity"][sf:ef:2]
    dc_arr = session_kinematics[sid]["dist_center"][sf:ef:2]
    av_arr = session_kinematics[sid]["angular_vel"][sf:ef:2]
    
    valid = ~np.isnan(v_arr) & ~np.isnan(dc_arr)
    if np.sum(valid) < 10:
        continue
        
    v_val = v_arr[valid]
    dc_val = dc_arr[valid]
    av_val = av_arr[valid]
    
    mean_v = float(np.mean(v_val))
    std_v = float(np.std(v_val))
    speed_cv = float(std_v / (mean_v + 1e-3))
    pause_frac = float(np.mean(v_val < 5.0))
    
    mean_av = float(np.mean(av_val))
    mean_dc = float(np.mean(dc_val))
    
    window_records.append({
        "window_idx": w_i,
        "session_id": sid,
        "mean_speed": mean_v,
        "speed_cv": speed_cv,
        "pause_frac": pause_frac,
        "angular_vel": mean_av,
        "dist_center": mean_dc,
        "spatial_entropy": float(s["spatial_entropy"]),
        "cadence": float(s["cadence"]),
        "sequence_entropy": float(s["sequence_entropy"]),
        "chirality": float(s["chirality"])
    })

df_win_features = pd.DataFrame(window_records)
valid_win_indices = df_win_features["window_idx"].values
print(f"Loaded {len(df_win_features):,} valid windows.")

def calc_bimodality_coeff(x):
    """Sarle's Bimodality Coefficient: BC > 0.555 (5/9) indicates genuine bimodality."""
    n = len(x)
    if n < 4:
        return 0.0
    g = skew(x)
    k = kurtosis(x, fisher=False)
    if k <= 0:
        return 0.0
    bc = (g**2 + 1.0) / (k + 3.0 * ((n - 1)**2) / ((n - 2) * (n - 3)))
    return float(bc)

def calc_eta2(groups_data):
    """Compute ANOVA effect size eta^2 = SS_between / SS_total."""
    valid_groups = [g for g in groups_data if len(g) > 1]
    if len(valid_groups) < 2:
        return 0.0
    all_data = np.concatenate(valid_groups)
    grand_mean = np.mean(all_data)
    ss_total = np.sum((all_data - grand_mean)**2)
    if ss_total <= 1e-9:
        return 0.0
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in valid_groups)
    return float(ss_between / ss_total)

TIERS = [1.00, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01]
TIER_LABELS = ["100% (All)", "50%", "25%", "10%", "5%", "2%", "1%"]
MODELS = ["kmeans_k8", "kmeans_k16", "gmm_k8", "gmm_k16"]

all_tier_results = []
archetype_profiles = {}

for model_name in MODELS:
    print(f"\n=======================================================")
    print(f"Exemplar Mining & Coherence Sweep on: {model_name}")
    print(f"=======================================================")
    
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    with open(model_path, "rb") as f:
        m_data = pickle.load(f)
        
    clf = m_data["model"]
    svd = m_data["svd"]
    cfg = m_data["config"]
    full_labels = m_data["window_labels"][valid_win_indices]
    full_probs = m_data["window_probs"][valid_win_indices]
    
    # Re-obtain latent SVD representations
    mat = raw_trans_inputs[valid_win_indices] + cfg["laplace_smooth"]
    if cfg["norm_type"] == "log1p":
        mat = np.log1p(mat)
    elif cfg["norm_type"] == "l1":
        from sklearn.preprocessing import normalize
        mat = normalize(mat, norm="l1")
    elif cfg["norm_type"] == "l2":
        from sklearn.preprocessing import normalize
        mat = normalize(mat, norm="l2")
    latent_z = svd.transform(mat)
    
    K = cfg["k"]
    unique_strats = np.sort(np.unique(full_labels))
    
    # Compute ranking metric per window within its assigned cluster
    # KMeans: Euclidean distance to centroid (ascending)
    # GMM: Posterior probability / margin (descending)
    point_scores = np.zeros(len(latent_z), dtype=np.float32)
    
    if cfg["family"] == "KMeans":
        centroids = clf.cluster_centers_
        for i in range(len(latent_z)):
            st = full_labels[i]
            point_scores[i] = np.linalg.norm(latent_z[i] - centroids[st]) # lower is better
    else: # GMM
        for i in range(len(latent_z)):
            st = full_labels[i]
            # Posterior confidence
            point_scores[i] = -full_probs[i, st] # negative so lower is better
            
    # For each tier, extract exemplar mask
    for tier_pct, tier_label in zip(TIERS, TIER_LABELS):
        selected_mask = np.zeros(len(latent_z), dtype=bool)
        strat_radii = []
        strat_bc_pc1 = []
        strat_bc_speed = []
        strat_bc_cv = []
        strat_bc_turn = []
        strat_bc_center = []
        strat_bc_cadence = []
        
        # 4-pillar data collectors for ANOVA eta^2
        pillar_speed = {s_id: [] for s_id in unique_strats}
        pillar_cv = {s_id: [] for s_id in unique_strats}
        pillar_turn = {s_id: [] for s_id in unique_strats}
        pillar_center = {s_id: [] for s_id in unique_strats}
        pillar_cadence = {s_id: [] for s_id in unique_strats}
        pillar_seq = {s_id: [] for s_id in unique_strats}
        
        for s_id in unique_strats:
            c_indices = np.where(full_labels == s_id)[0]
            n_c = len(c_indices)
            c_scores = point_scores[c_indices]
            
            # Sort ascending (lowest distance or highest confidence)
            sort_order = np.argsort(c_scores)
            n_keep = max(int(np.ceil(n_c * tier_pct)), 4)
            keep_indices = c_indices[sort_order[:n_keep]]
            selected_mask[keep_indices] = True
            
            # Latent radius for kept points
            z_sub = latent_z[keep_indices]
            c_mean = np.mean(z_sub, axis=0)
            radius = np.mean(np.linalg.norm(z_sub - c_mean, axis=1))
            strat_radii.append(radius)
            
            # Internal PC1 Modality
            if len(z_sub) >= 6:
                pca_internal = PCA(n_components=1, random_state=42)
                pc1_proj = pca_internal.fit_transform(z_sub).ravel()
                bc_pc1 = calc_bimodality_coeff(pc1_proj)
            else:
                bc_pc1 = 0.20
            strat_bc_pc1.append(bc_pc1)
            
            # 4-Pillar metrics for kept windows
            sub_feats = df_win_features.iloc[keep_indices]
            v_vals = sub_feats["mean_speed"].values
            cv_vals = sub_feats["speed_cv"].values
            t_vals = sub_feats["angular_vel"].values
            dc_vals = sub_feats["dist_center"].values
            cad_vals = sub_feats["cadence"].values
            seq_vals = sub_feats["sequence_entropy"].values
            
            strat_bc_speed.append(calc_bimodality_coeff(v_vals))
            strat_bc_cv.append(calc_bimodality_coeff(cv_vals))
            strat_bc_turn.append(calc_bimodality_coeff(t_vals))
            strat_bc_center.append(calc_bimodality_coeff(dc_vals))
            strat_bc_cadence.append(calc_bimodality_coeff(cad_vals))
            
            pillar_speed[s_id] = v_vals
            pillar_cv[s_id] = cv_vals
            pillar_turn[s_id] = t_vals
            pillar_center[s_id] = dc_vals
            pillar_cadence[s_id] = cad_vals
            pillar_seq[s_id] = seq_vals
            
        n_selected = np.sum(selected_mask)
        z_sel = latent_z[selected_mask]
        lbl_sel = full_labels[selected_mask]
        
        # Latent Clustering Metrics
        # Subsample for fast silhouette computation if large
        if len(z_sel) > 5000:
            sub_idx = np.random.choice(len(z_sel), size=5000, replace=False)
            sil = float(silhouette_score(z_sel[sub_idx], lbl_sel[sub_idx]))
        else:
            sil = float(silhouette_score(z_sel, lbl_sel))
            
        db_score = float(davies_bouldin_score(z_sel, lbl_sel))
        ch_score = float(calinski_harabasz_score(z_sel, lbl_sel))
        mean_radius = float(np.mean(strat_radii))
        
        # 4-Pillar ANOVA eta^2
        eta2_speed = calc_eta2(list(pillar_speed.values()))
        eta2_cv = calc_eta2(list(pillar_cv.values()))
        eta2_turn = calc_eta2(list(pillar_turn.values()))
        eta2_center = calc_eta2(list(pillar_center.values()))
        eta2_cadence = calc_eta2(list(pillar_cadence.values()))
        eta2_seq = calc_eta2(list(pillar_seq.values()))
        mean_eta2 = float(np.mean([eta2_speed, eta2_cv, eta2_turn, eta2_center, eta2_cadence, eta2_seq]))
        
        # Modality summary
        mean_bc_pc1 = float(np.mean(strat_bc_pc1))
        mean_bc_speed = float(np.mean(strat_bc_speed))
        mean_bc_center = float(np.mean(strat_bc_center))
        bimodal_strats_count = sum(1 for bc in strat_bc_speed if bc > 0.555)
        
        res_row = {
            "model": model_name,
            "tier_label": tier_label,
            "tier_pct": tier_pct,
            "n_windows": int(n_selected),
            "mean_radius": mean_radius,
            "silhouette": sil,
            "davies_bouldin": db_score,
            "calinski_harabasz": ch_score,
            "mean_bc_pc1": mean_bc_pc1,
            "mean_bc_speed": mean_bc_speed,
            "mean_bc_center": mean_bc_center,
            "bimodal_speed_states": bimodal_strats_count,
            "eta2_speed": eta2_speed,
            "eta2_cv": eta2_cv,
            "eta2_turn": eta2_turn,
            "eta2_center": eta2_center,
            "eta2_cadence": eta2_cadence,
            "eta2_seq": eta2_seq,
            "mean_eta2": mean_eta2
        }
        all_tier_results.append(res_row)
        
        print(f"Tier {tier_label:10s} | N={n_selected:5d} | Radius={mean_radius:.4f} | Sil={sil:+.3f} | DB={db_score:.2f} | Mean η²={mean_eta2:.3f} | PC1 BC={mean_bc_pc1:.3f}")

df_all_results = pd.DataFrame(all_tier_results)
csv_results_path = os.path.join(AUDIT_DIR, "exemplar_metrics_summary.csv")
df_all_results.to_csv(csv_results_path, index=False)
print(f"\nSaved full multi-tier exemplar audit CSV to: {csv_results_path}")

# ==============================================================================
# 2. Plot Multi-Panel Progression Figure Across All 4 Models
# ==============================================================================
print("\nGenerating publication progression curves...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=200)

tier_x = np.array([100, 50, 25, 10, 5, 2, 1])

colors = {
    "kmeans_k8": "#2563eb",
    "kmeans_k16": "#059669",
    "gmm_k8": "#d97706",
    "gmm_k16": "#7c3aed"
}
labels_map = {
    "kmeans_k8": "KMeans (K=8)",
    "kmeans_k16": "KMeans (K=16) [Discrete Champion]",
    "gmm_k8": "GMM (K=8)",
    "gmm_k16": "GMM (K=16) [Continuous Champion]"
}

# Panel 1: Latent Silhouette Score vs Exemplar Tier
ax = axes[0, 0]
for m in MODELS:
    sub = df_all_results[df_all_results["model"] == m]
    ax.plot(tier_x, sub["silhouette"].values, marker="o", linewidth=2.2, label=labels_map[m], color=colors[m])
ax.set_title("A. Latent Space Cluster Separation (Silhouette Score)", fontsize=11, fontweight="bold")
ax.set_xlabel("Exemplar Population Tier (% Purest Windows)", fontsize=9)
ax.set_ylabel("Silhouette Score (Higher = Better Separated)", fontsize=9)
ax.set_xscale("log")
ax.set_xticks(tier_x)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.invert_xaxis()
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(fontsize=8, loc="best")

# Panel 2: Mean Intra-Cluster Radius (Cluster Tightness)
ax = axes[0, 1]
for m in MODELS:
    sub = df_all_results[df_all_results["model"] == m]
    # Normalize radius relative to 100% baseline for fair comparison
    base_r = sub["mean_radius"].values[0]
    norm_r = (sub["mean_radius"].values / base_r) * 100.0
    ax.plot(tier_x, norm_r, marker="s", linewidth=2.2, label=labels_map[m], color=colors[m])
ax.set_title("B. Cluster Radius Compression (% of 100% Baseline)", fontsize=11, fontweight="bold")
ax.set_xlabel("Exemplar Population Tier (% Purest Windows)", fontsize=9)
ax.set_ylabel("Mean Cluster Radius (% of Baseline)", fontsize=9)
ax.set_xscale("log")
ax.set_xticks(tier_x)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.invert_xaxis()
ax.grid(True, linestyle="--", alpha=0.3)

# Panel 3: ANOVA Physical Variance Explained (Mean η² across 4 Pillars)
ax = axes[1, 0]
for m in MODELS:
    sub = df_all_results[df_all_results["model"] == m]
    ax.plot(tier_x, sub["mean_eta2"].values, marker="^", linewidth=2.2, label=labels_map[m], color=colors[m])
ax.set_title("C. Physical Distinctiveness (Mean ANOVA η² across 4 Pillars)", fontsize=11, fontweight="bold")
ax.set_xlabel("Exemplar Population Tier (% Purest Windows)", fontsize=9)
ax.set_ylabel("ANOVA η² Effect Size (Between-Strategy Variance)", fontsize=9)
ax.set_xscale("log")
ax.set_xticks(tier_x)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.invert_xaxis()
ax.grid(True, linestyle="--", alpha=0.3)

# Panel 4: Sarle's Bimodality on Internal PC1
ax = axes[1, 1]
for m in MODELS:
    sub = df_all_results[df_all_results["model"] == m]
    ax.plot(tier_x, sub["mean_bc_pc1"].values, marker="D", linewidth=2.2, label=labels_map[m], color=colors[m])
ax.axhline(0.333, color="red", linestyle=":", label="Gaussian Unimodal Limit (BC=0.333)")
ax.axhline(0.555, color="black", linestyle="--", alpha=0.5, label="Bimodality Threshold (BC=0.555)")
ax.set_title("D. Internal PC1 Attractor Unimodality (Sarle's BC)", fontsize=11, fontweight="bold")
ax.set_xlabel("Exemplar Population Tier (% Purest Windows)", fontsize=9)
ax.set_ylabel("Sarle's Bimodality Coefficient (BC)", fontsize=9)
ax.set_xscale("log")
ax.set_xticks(tier_x)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.invert_xaxis()
ax.set_ylim(0.10, 0.60)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(fontsize=8, loc="lower right")

plt.suptitle("Multi-Scale Strategy Exemplar Mining: 100% → 50% → 25% → 10% → 5% → 2% → 1%", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
plot_progression_path = os.path.join(AUDIT_DIR, "exemplar_cohesion_progression.png")
plt.savefig(plot_progression_path, dpi=200)
plt.close()
print(f"Saved multi-panel progression figure to: {plot_progression_path}")

# ==============================================================================
# 3. Latent Space Scatter Comparison: 100% vs 25% vs 5% vs 1% (KMeans-16 & GMM-16)
# ==============================================================================
print("\nGenerating 2D Latent Projection Scatters...")
for champ_model in ["kmeans_k16", "gmm_k16"]:
    model_path = os.path.join(MODEL_DIR, f"{champ_model}.pkl")
    with open(model_path, "rb") as f:
        m_data = pickle.load(f)
    clf = m_data["model"]
    svd = m_data["svd"]
    cfg = m_data["config"]
    full_labels = m_data["window_labels"][valid_win_indices]
    full_probs = m_data["window_probs"][valid_win_indices]
    
    mat = raw_trans_inputs[valid_win_indices] + cfg["laplace_smooth"]
    if cfg["norm_type"] == "l1":
        from sklearn.preprocessing import normalize
        mat = normalize(mat, norm="l1")
    latent_z = svd.transform(mat)
    
    # 2D PCA projection for visualization
    pca2 = PCA(n_components=2, random_state=42)
    z_2d = pca2.fit_transform(latent_z)
    
    point_scores = np.zeros(len(latent_z), dtype=np.float32)
    if cfg["family"] == "KMeans":
        centroids = clf.cluster_centers_
        for i in range(len(latent_z)):
            point_scores[i] = np.linalg.norm(latent_z[i] - centroids[full_labels[i]])
    else:
        for i in range(len(latent_z)):
            point_scores[i] = -full_probs[i, full_labels[i]]
            
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=200)
    compare_tiers = [(1.00, "100% (All Windows)"), (0.25, "Top 25% Exemplars"), 
                     (0.05, "Top 5% Pure Exemplars"), (0.01, "Top 1% Ultra Archetypes")]
    
    for ax_i, (t_pct, t_name) in enumerate(compare_tiers):
        ax = axes[ax_i]
        sel_mask = np.zeros(len(latent_z), dtype=bool)
        for s_id in np.unique(full_labels):
            c_idx = np.where(full_labels == s_id)[0]
            s_scores = point_scores[c_idx]
            n_k = max(int(np.ceil(len(c_idx) * t_pct)), 4)
            sel_mask[c_idx[np.argsort(s_scores)[:n_k]]] = True
            
        z_plot = z_2d[sel_mask]
        l_plot = full_labels[sel_mask]
        
        ax.scatter(z_plot[:, 0], z_plot[:, 1], c=l_plot, cmap="tab20", alpha=0.75 if t_pct <= 0.05 else 0.4, s=15 if t_pct <= 0.05 else 6)
        ax.set_title(f"{t_name}\n(N={np.sum(sel_mask):,} windows)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Latent Strategy PC1", fontsize=9)
        ax.set_ylabel("Latent Strategy PC2", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        
    plt.suptitle(f"Attractor Basin Condensation: {champ_model.upper()} (K=16)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    scatter_out = os.path.join(AUDIT_DIR, f"latent_condensation_scatter_{champ_model}.png")
    plt.savefig(scatter_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved 2D condensation scatter to: {scatter_out}")

print("\n✅ Multi-Scale Exemplar Audit Pipeline Finished Successfully!")
