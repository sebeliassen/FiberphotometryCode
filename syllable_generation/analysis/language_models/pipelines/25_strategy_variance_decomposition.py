"""
Strategy Variance Decomposition & Residual Eigenspace Discovery
File: syllable_generation/analysis/language_models/pipelines/25_strategy_variance_decomposition.py

Quantifies:
1. How much of the Strategy Latent Space variance (24-D / 32-D) is explained
   by our current 9 physical & information-theoretic scalar metrics.
2. Per-PC variance explained (which dimensions of strategy are fully captured vs uncaptured).
3. Residual Eigenspace Analysis (PCA on unexplained strategy variance).
4. Ethological interpretation of the top Residual PCs by correlating with:
   - 100 Syllable Usage Frequencies
   - Syllable Bigram Transitions
   - Temporal Bout Durations
   - Non-linear Feature Interactions
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation"))
    sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models"))

OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
MODEL_DIR = os.path.join(OUT_DIR, "production_models")
VAR_DIR = os.path.join(OUT_DIR, "strategy_variance_audit")
os.makedirs(VAR_DIR, exist_ok=True)

# 1. Load Data
print("Loading production models and window cache...")
with open(os.path.join(MODEL_DIR, "kmeans_k16.pkl"), "rb") as f:
    m16 = pickle.load(f)
svd = m16["svd"]
cfg = m16["config"]
clf = m16["model"]

with open(os.path.join(OUT_DIR, "window_scalars_cache.pkl"), "rb") as f:
    wdata = pickle.load(f)
raw_trans = wdata["trans_inputs"]
meta = wdata["meta"]
scalars_raw = wdata["scalars"]
raw_sylls = wdata["raw_sylls"] # (14092, 100)

pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
with open(pkl_path, "rb") as f:
    raw_sessions = pickle.load(f)

# Load continuous bouts for duration metrics
with open(os.path.join(OUT_DIR, "continuous_bout_cache.pkl"), "rb") as f:
    bdata = pickle.load(f)
session_bouts = bdata["session_bouts"]

FPS = 80.025
session_kin = {}
for s in raw_sessions:
    sid = s.session_id
    df = s.all_data_df
    vel = df["velocity_px_s"].values if "velocity_px_s" in df.columns else np.zeros(len(df))
    cx = df["centroid_x"].values if "centroid_x" in df.columns else np.zeros(len(df))
    cy = df["centroid_y"].values if "centroid_y" in df.columns else np.zeros(len(df))
    dist_center = np.sqrt((cx - np.nanmean(cx))**2 + (cy - np.nanmean(cy))**2)
    h = df["heading"].values if "heading" in df.columns else np.zeros(len(df))
    dh = np.diff(h, prepend=h[0])
    dh_wrapped = (dh + np.pi) % (2 * np.pi) - np.pi
    ang_vel = np.clip(np.abs(dh_wrapped) * FPS, 0.0, 25.0)
    session_kin[s.session_id] = {
        "vel": vel.astype(np.float32),
        "dist_center": dist_center.astype(np.float32),
        "ang_vel": ang_vel.astype(np.float32),
        "len": len(df)
    }

# 2. Build 9 Current Physical & Scalar Predictors
print("Extracting current physical & scalar predictors...")
X_list = []
valid_indices = []

for i, (w, s) in enumerate(zip(meta, scalars_raw)):
    sid = w["session_id"]
    sf, ef = int(w["start_frame"]), min(int(w["end_frame"]), session_kin[sid]["len"])
    if ef <= sf:
        continue
    v_sub = session_kin[sid]["vel"][sf:ef:2]
    dc_sub = session_kin[sid]["dist_center"][sf:ef:2]
    av_sub = session_kin[sid]["ang_vel"][sf:ef:2]
    
    valid = ~np.isnan(v_sub) & ~np.isnan(dc_sub)
    if np.sum(valid) < 10:
        continue
        
    v_vals = v_sub[valid]
    v_mean = float(np.mean(v_vals))
    v_std = float(np.std(v_vals))
    v_cv = float(v_std / (v_mean + 1e-3))
    pause = float(np.mean(v_vals < 5.0))
    dc_mean = float(np.mean(dc_sub[valid]))
    av_mean = float(np.mean(av_sub[valid]))
    
    spat_ent = float(s["spatial_entropy"]) if not np.isnan(s["spatial_entropy"]) else 0.0
    cad = float(s["cadence"]) if not np.isnan(s["cadence"]) else 0.0
    seq_ent = float(s["sequence_entropy"]) if not np.isnan(s["sequence_entropy"]) else 0.0
    chir = float(s["chirality"]) if not np.isnan(s["chirality"]) else 0.0
    
    X_list.append([
        v_mean, v_cv, pause, av_mean, dc_mean,
        spat_ent, cad, seq_ent, chir
    ])
    valid_indices.append(i)

valid_indices = np.array(valid_indices)
X_scalars = np.array(X_list)
scalar_names = ["Mean Speed", "Speed CV", "Pause Fraction", "Turn Rate", "Distance-to-Center", "Spatial Entropy", "Cadence", "Sequence Entropy", "Chirality"]

# Impute any edge NaNs
imputer = SimpleImputer(strategy="median")
X_scalars = imputer.fit_transform(X_scalars)

# 3. Compute Strategy Latent Representation Z
mat = normalize(raw_trans[valid_indices] + cfg["laplace_smooth"], norm="l1")
Z = svd.transform(mat) # (N, 24)
raw_sylls_sub = raw_sylls[valid_indices] # (N, 100)

print(f"Dataset: {len(Z):,} windows | Z space: {Z.shape[1]}-D SVD | Predictors: {X_scalars.shape[1]} Scalars")

# 4. Strategy Latent Eigenspectrum
pca_Z = PCA(n_components=24, random_state=42).fit(Z)
cum_var = np.cumsum(pca_Z.explained_variance_ratio_)
Z_pca = pca_Z.transform(Z)

# 5. Fit Multivariate Ridge Regression to Predict Strategy Space from Current 9 Scalars
scaler_X = StandardScaler()
X_norm = scaler_X.fit_transform(X_scalars)

ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
ridge.fit(X_norm, Z)
Z_hat = ridge.predict(X_norm)
Z_hat_pca = pca_Z.transform(Z_hat)

# Total Variance Explained
ss_total = np.sum((Z - np.mean(Z, axis=0))**2)
ss_res = np.sum((Z - Z_hat)**2)
r2_linear = float(1.0 - (ss_res / ss_total))

# Non-linear Random Forest check on top 5 PCs
print("Evaluating non-linear variance capture via Random Forest on Top PCs...")
rf_r2_list = []
for k in range(5):
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_norm[::2], Z_pca[::2, k]) # 50% train
    rf_pred = rf.predict(X_norm[1::2]) # 50% test
    var_test = np.var(Z_pca[1::2, k])
    var_err = np.var(Z_pca[1::2, k] - rf_pred)
    rf_r2_list.append(float(1.0 - var_err / var_test))

print(f"\n=======================================================")
print(f"LINEAR VARIANCE OF STRATEGY SPACE EXPLAINED BY 9 SCALARS: {r2_linear*100:.2f}%")
print(f"TOP PC NON-LINEAR VARIANCE EXPLAINED (Random Forest):     {np.mean(rf_r2_list)*100:.2f}%")
print(f"=======================================================")

# Per-PC Breakdown
pc_rows = []
for k in range(12):
    var_true = np.var(Z_pca[:, k])
    var_res = np.var(Z_pca[:, k] - Z_hat_pca[:, k])
    r2_pc = float(1.0 - (var_res / var_true))
    pc_rows.append({
        "strategy_pc": f"PC{k+1}",
        "eigenspace_weight_pct": float(pca_Z.explained_variance_ratio_[k] * 100.0),
        "cumulative_weight_pct": float(cum_var[k] * 100.0),
        "r2_explained_by_scalars_pct": float(r2_pc * 100.0),
        "residual_unexplained_pct": float((1.0 - max(r2_pc, 0.0)) * 100.0)
    })

df_pc_summary = pd.DataFrame(pc_rows)
print("\n--- Strategy Eigenspace Variance Explained Breakdown ---")
print(df_pc_summary.to_string(index=False))

# 6. Residual Eigenspace Analysis (PCA on Unexplained Strategy Space)
print("\nPerforming PCA on Residual (Unexplained) Strategy Space...")
Z_res = Z - Z_hat # Residual matrix (14092, 24)
pca_res = PCA(n_components=10, random_state=42).fit(Z_res)
Z_res_pca = pca_res.transform(Z_res)

res_cum = np.cumsum(pca_res.explained_variance_ratio_)
print(f"Top 5 Residual PCs account for {res_cum[4]*100:.2f}% of unexplained variance:")
for r_i in range(5):
    print(f"  Residual PC{r_i+1}: {pca_res.explained_variance_ratio_[r_i]*100:.2f}% (Cumulative: {res_cum[r_i]*100:.2f}%)")

# 7. Discovering the Ethological Identity of Residual PCs
print("\nCorrelating Residual PCs with 100 Syllables to find missing behavioral dimensions...")
residual_corrs = []

for r_i in range(5):
    r_vec = Z_res_pca[:, r_i]
    
    # Syllable correlations
    syll_corrs = np.array([np.corrcoef(r_vec, raw_sylls_sub[:, s_idx])[0, 1] for s_idx in range(100)])
    
    top_pos_sylls = np.argsort(syll_corrs)[-3:][::-1]
    top_neg_sylls = np.argsort(syll_corrs)[:3]
    
    pos_desc = ", ".join([f"Syllable {s} (r={syll_corrs[s]:+.2f})" for s in top_pos_sylls])
    neg_desc = ", ".join([f"Syllable {s} (r={syll_corrs[s]:+.2f})" for s in top_neg_sylls])
    
    residual_corrs.append({
        "residual_pc": f"Residual PC{r_i+1}",
        "variance_pct": float(pca_res.explained_variance_ratio_[r_i] * 100.0),
        "top_positive_syllables": pos_desc,
        "top_negative_syllables": neg_desc
    })

df_res_id = pd.DataFrame(residual_corrs)
print("\n--- Missing / Residual Strategy Dimensions Identity ---")
print(df_res_id.to_string(index=False))

# Export Data
csv_pc_path = os.path.join(VAR_DIR, "strategy_pc_variance_explained.csv")
df_pc_summary.to_csv(csv_pc_path, index=False)
csv_res_path = os.path.join(VAR_DIR, "residual_pc_identities.csv")
df_res_id.to_csv(csv_res_path, index=False)

# 8. Multi-Panel Figure Generation
print("\nGenerating Variance Decomposition Figure...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

# Panel A: Strategy Eigenspace vs. Explained Variance
ax = axes[0, 0]
pcs_idx = np.arange(1, 13)
ax.bar(pcs_idx - 0.2, df_pc_summary["eigenspace_weight_pct"], width=0.4, label="Strategy Eigenspace Weight (%)", color="#2563eb", alpha=0.85)
ax.bar(pcs_idx + 0.2, df_pc_summary["r2_explained_by_scalars_pct"], width=0.4, label="R² Explained by Current 9 Scalars (%)", color="#059669", alpha=0.85)
ax.set_title("A. Strategy PC Weights vs. Variance Explained by Scalars", fontsize=11, fontweight="bold")
ax.set_xlabel("Strategy Latent Principal Component", fontsize=9)
ax.set_ylabel("Percentage (%)", fontsize=9)
ax.set_xticks(pcs_idx)
ax.set_xticklabels([f"PC{i}" for i in pcs_idx])
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(fontsize=8, loc="upper right")

# Panel B: Residual Strategy Eigenspectrum (What are we missing?)
ax = axes[0, 1]
ax.plot(np.arange(1, 11), pca_res.explained_variance_ratio_ * 100, marker="o", color="#d97706", linewidth=2.2, label="Individual Residual PC Var (%)")
ax.plot(np.arange(1, 11), res_cum * 100, marker="s", color="#7c3aed", linewidth=2.0, linestyle="--", label="Cumulative Unexplained Var (%)")
ax.set_title("B. Unexplained Strategy Variance (Residual Eigenspectrum)", fontsize=11, fontweight="bold")
ax.set_xlabel("Residual Principal Component", fontsize=9)
ax.set_ylabel("Unexplained Variance (%)", fontsize=9)
ax.set_xticks(np.arange(1, 11))
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(fontsize=8, loc="center right")

# Panel C: Feature Importance of Current 9 Scalars (Ridge Weights on Top 3 PCs)
ax = axes[1, 0]
coef_mat = np.abs(ridge.coef_[:3]) # Top 3 SVD dimensions
y_pos = np.arange(len(scalar_names))
for p_i, color, lbl in zip(range(3), ["#2563eb", "#059669", "#d97706"], ["Strategy Dim 1", "Strategy Dim 2", "Strategy Dim 3"]):
    ax.barh(y_pos + (p_i - 1)*0.25, coef_mat[p_i], height=0.25, color=color, alpha=0.8, label=lbl)
ax.set_title("C. Current Scalar Importance across Top Strategy Dimensions", fontsize=11, fontweight="bold")
ax.set_yticks(y_pos)
ax.set_yticklabels(scalar_names, fontsize=8)
ax.set_xlabel("Absolute Ridge Regression Weight", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(fontsize=8, loc="lower right")

# Panel D: Residual PC1 vs Residual PC2 Scatter (The Hidden Behavioral Dimensions)
ax = axes[1, 1]
scatter = ax.scatter(Z_res_pca[:, 0], Z_res_pca[:, 1], c=X_scalars[:, 0], cmap="viridis", s=6, alpha=0.4)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Mean Speed (px/s)", fontsize=8)
ax.set_title("D. Residual Space PC1 vs PC2 (Orthogonal to All 9 Scalars)", fontsize=11, fontweight="bold")
ax.set_xlabel(f"Residual PC1 ({pca_res.explained_variance_ratio_[0]*100:.1f}% unexp. var)", fontsize=9)
ax.set_ylabel(f"Residual PC2 ({pca_res.explained_variance_ratio_[1]*100:.1f}% unexp. var)", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.3)

plt.suptitle("Strategy Variance Decomposition & Residual Eigenspace Audit", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
fig_out = os.path.join(VAR_DIR, "strategy_variance_decomposition.png")
plt.savefig(fig_out, dpi=200)
plt.close()
print(f"Saved variance decomposition figure to: {fig_out}")

print("\n✅ Strategy Variance Decomposition Complete!")
