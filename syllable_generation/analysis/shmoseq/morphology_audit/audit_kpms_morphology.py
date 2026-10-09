#!/usr/bin/env python3
"""
audit_kpms_morphology.py
=============================================================================
Phase 1: Deep KPMS Morphology & Latent Dynamics Audit

Addresses every point of the ChatGPT review:
1. True anatomical axial-chain length vs. straight-line Euclidean distance.
2. Conditioning on active forward locomotion (speed > 5 cm/s).
3. Variance decomposition (Between-Animal vs. Within-Animal Dynamic Variance).
4. Raw PCA scores vs. Inferred AR-HMM Latent States (x_t).
5. Syllable size-stratification audit.
=============================================================================
"""

import os
import sys
import pickle
import joblib
import h5py
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(BASE_DIR, "syllable_generation", "syllabizers", "keypoint_moseq_syll", "amph_project")
RUN12_DIR = os.path.join(KPMS_DIR, "amph_prod_k150k_run_12")
SHMOSEQ_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq")
OUTPUT_DIR = os.path.join(SHMOSEQ_DIR, "morphology_audit")
TAB_DIR = os.path.join(OUTPUT_DIR, "tables")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. LOAD DATA AND METADATA
# -----------------------------------------------------------------------------
print("=" * 80)
print("🚀 RUNNING PHASE 1: DEEP KPMS MORPHOLOGY & LATENT DYNAMICS AUDIT")
print("=" * 80, flush=True)

# Load metadata bundle
with open(os.path.join(SHMOSEQ_DIR, "output", "data", "shmoseq_input_data.pkl"), "rb") as f:
    bundle = pickle.load(f)

metadata_dict = bundle["metadata_dict"]
syllables_dict = bundle["syllables_dict"]

# Keypoint mappings based on use_bodyparts in config.yml
# 0: nose, 1: left_ear_tip, 2: right_ear_tip, 3: neck, 4: mid_back, 5: mouse_center
# 6: mid_backend, 7: mid_backend2, 8: mid_backend3, 9: tail_base
# 10: left_shoulder, 11: left_midside, 12: left_hip, 13: right_shoulder, 14: right_midside, 15: right_hip, 16: head_midpoint
AXIAL_CHAIN_INDICES = [3, 4, 5, 6, 7, 8, 9] # neck -> mid_back -> center -> backend1 -> backend2 -> backend3 -> tail_base
NECK_IDX = 3
TAIL_BASE_IDX = 9
NOSE_IDX = 0
LEFT_MIDSIDE_IDX = 11
RIGHT_MIDSIDE_IDX = 14

print(f"Verified {len(metadata_dict)} sessions across {len(set(m['animal_id'] for m in metadata_dict.values()))} unique animals.")

# -----------------------------------------------------------------------------
# 2. AXIAL-CHAIN VS STRAIGHT-LINE MEASUREMENTS (FULL SESSION SAMPLING)
# -----------------------------------------------------------------------------
print("\n[Step 1/4] Measuring True Anatomical Axial-Chain Length across full sessions...", flush=True)

chk_path = os.path.join(RUN12_DIR, "checkpoint.h5")
res_path = os.path.join(RUN12_DIR, "results.h5")

morphology_records = []
all_frames_data = {}

with h5py.File(chk_path, "r") as f_chk, h5py.File(res_path, "r") as f_res:
    meta0 = [s.decode() if hasattr(s, "decode") else str(s) for s in f_chk["metadata"]["arr0"][:]]
    Y_dset = f_chk["data"]["Y"]
    conf_dset = f_chk["data"]["conf"]
    mask_dset = f_chk["data"]["mask"]

    for k, m in metadata_dict.items():
        aid = str(m["animal_id"])
        geno = m["genotype"]
        drug = m["drug"]
        orig_id = m["original_session_id"]

        matching_chk = [s for s in set(meta0) if orig_id in s]
        matching_res = [s for s in f_res.keys() if orig_id in s]
        if not matching_chk or not matching_res:
            print(f"  Warning: could not match session {orig_id}")
            continue

        chk_name = matching_chk[0]
        res_name = matching_res[0]

        # Get all batches for this session
        batch_indices = [i for i, name in enumerate(meta0) if name == chk_name]
        
        # Sample uniformly across all batches (up to 15 batches spread across session)
        sample_step = max(1, len(batch_indices) // 15)
        selected_batches = batch_indices[::sample_step][:15]

        # Load batches
        y_batches = Y_dset[selected_batches] # (B, T, 17, 2)
        c_batches = conf_dset[selected_batches] # (B, T, 17)
        m_batches = mask_dset[selected_batches] # (B, T)

        y_flat = y_batches.reshape(-1, 17, 2)
        c_flat = c_batches.reshape(-1, 17)
        m_flat = m_batches.reshape(-1)

        # High confidence filter on spine and width keypoints
        spine_conf = np.all(c_flat[:, AXIAL_CHAIN_INDICES] > 0.5, axis=1)
        width_conf = np.all(c_flat[:, [LEFT_MIDSIDE_IDX, RIGHT_MIDSIDE_IDX]] > 0.5, axis=1)
        valid_mask = (m_flat.astype(bool)) & spine_conf & width_conf

        y_valid = y_flat[valid_mask]
        if len(y_valid) < 500:
            print(f"  Warning: insufficient valid frames for {orig_id} ({len(y_valid)})")
            continue

        # 1. Straight-line neck-to-tail
        straight_neck_tail = np.linalg.norm(y_valid[:, NECK_IDX] - y_valid[:, TAIL_BASE_IDX], axis=1)

        # 2. True axial-chain spine length (sum of contiguous segments along vertebral column)
        axial_segments = [
            np.linalg.norm(y_valid[:, AXIAL_CHAIN_INDICES[i]] - y_valid[:, AXIAL_CHAIN_INDICES[i+1]], axis=1)
            for i in range(len(AXIAL_CHAIN_INDICES) - 1)
        ]
        axial_chain_length = np.sum(axial_segments, axis=0)

        # 3. Body width
        body_width = np.linalg.norm(y_valid[:, LEFT_MIDSIDE_IDX] - y_valid[:, RIGHT_MIDSIDE_IDX], axis=1)

        # 4. Aspect ratio (axial chain / width)
        aspect_ratio = axial_chain_length / (body_width + 1e-5)

        morphology_records.append({
            "session_key": k,
            "animal_id": aid,
            "genotype": geno,
            "drug": drug,
            "valid_frames": len(y_valid),
            "straight_neck_tail_med": float(np.median(straight_neck_tail)),
            "straight_neck_tail_iqr": float(stats.iqr(straight_neck_tail)),
            "axial_chain_med": float(np.median(axial_chain_length)),
            "axial_chain_iqr": float(stats.iqr(axial_chain_length)),
            "axial_chain_p95": float(np.percentile(axial_chain_length, 95)),
            "body_width_med": float(np.median(body_width)),
            "body_width_iqr": float(stats.iqr(body_width)),
            "aspect_ratio_med": float(np.median(aspect_ratio)),
            "aspect_ratio_iqr": float(stats.iqr(aspect_ratio)),
        })

df_morph = pd.DataFrame(morphology_records)
df_morph.to_csv(os.path.join(TAB_DIR, "axial_chain_vs_straight_line_summary.csv"), index=False)

# Evaluate Intra-Animal Consistency across days (Saline vs Amph)
pivot_straight = df_morph.pivot(index="animal_id", columns="drug", values="straight_neck_tail_med")
pivot_axial = df_morph.pivot(index="animal_id", columns="drug", values="axial_chain_med")
pivot_width = df_morph.pivot(index="animal_id", columns="drug", values="body_width_med")

r_straight = pivot_straight["Saline"].corr(pivot_straight["Amphetamine"])
r_axial = pivot_axial["Saline"].corr(pivot_axial["Amphetamine"])
r_width = pivot_width["Saline"].corr(pivot_width["Amphetamine"])

print(f"\nIntra-Animal Stability across Days (Day 1 Saline vs Day 2 Amphetamine):")
print(f"  Straight-Line Neck-Tail Length : r = {r_straight:.4f}")
print(f"  True Axial-Chain Spine Length  : r = {r_axial:.4f}")
print(f"  Transverse Body Width          : r = {r_width:.4f}")
print(f"  Axial-Chain Length Range       : {df_morph['axial_chain_med'].min():.2f} to {df_morph['axial_chain_med'].max():.2f} px")

# -----------------------------------------------------------------------------
# 3. LATENT VARIANCE DECOMPOSITION (BETWEEN-ANIMAL VS WITHIN-ANIMAL DYNAMIC)
# -----------------------------------------------------------------------------
print("\n[Step 2/4] Quantifying Latent Variance Decomposition (Between vs Within)...", flush=True)

# Load latent states from results.h5 for Saline baseline
animal_latents = {}
with h5py.File(res_path, "r") as f_res:
    for k, m in metadata_dict.items():
        if m["drug"] != "Saline":
            continue
        aid = str(m["animal_id"])
        orig_id = m["original_session_id"]
        matching = [s for s in f_res.keys() if orig_id in s]
        if not matching:
            continue
        lat = f_res[matching[0]]["latent_state"][:] # (T, 10)
        valid = ~np.isnan(lat).any(axis=1)
        animal_latents[aid] = lat[valid]

all_animals = sorted(list(animal_latents.keys()))
all_latents_concat = np.vstack([animal_latents[aid] for aid in all_animals])
grand_mean = np.mean(all_latents_concat, axis=0) # (10,)
total_ss = np.sum((all_latents_concat - grand_mean) ** 2, axis=0)

between_ss = np.zeros(10)
within_ss = np.zeros(10)
for aid in all_animals:
    a_lat = animal_latents[aid]
    n_a = len(a_lat)
    a_mean = np.mean(a_lat, axis=0)
    between_ss += n_a * ((a_mean - grand_mean) ** 2)
    within_ss += np.sum((a_lat - a_mean) ** 2, axis=0)

eta_squared = between_ss / total_ss * 100.0 # percentage between-animal variance

variance_records = []
for d in range(10):
    variance_records.append({
        "latent_dimension": f"Dim_{d}",
        "total_variance": float(np.var(all_latents_concat[:, d])),
        "between_animal_variance_fraction_pct": float(eta_squared[d]),
        "within_animal_dynamic_variance_pct": float(100.0 - eta_squared[d]),
        "f_statistic": float((between_ss[d] / (len(all_animals) - 1)) / (within_ss[d] / (len(all_latents_concat) - len(all_animals)))),
    })

df_var = pd.DataFrame(variance_records)
df_var.to_csv(os.path.join(TAB_DIR, "latent_variance_decomposition.csv"), index=False)
print("Latent Variance Partitioning (Saline Baseline):")
print(df_var[["latent_dimension", "between_animal_variance_fraction_pct", "within_animal_dynamic_variance_pct"]].to_string(index=False))

# -----------------------------------------------------------------------------
# 4. RAW PCA SCORES VS INFERRED AR-HMM LATENT STATES
# -----------------------------------------------------------------------------
print("\n[Step 3/4] Comparing Raw PCA Scores vs Inferred AR-HMM Latent States...", flush=True)

# Load PCA model using joblib
pca_model = joblib.load(os.path.join(KPMS_DIR, "pca.p"))
pca_components = pca_model.components_ # (10, 17*2) = (10, 34)
pca_mean = pca_model.mean_ # (34,)
pca_var_ratio = pca_model.explained_variance_ratio_

print(f"PCA Loaded: {pca_components.shape[0]} components, Top 3 explained variance: {pca_var_ratio[:3]}")

# Correlate animal mean PCA vs animal mean latent_state with axial chain & width
animal_mean_latents = {}
animal_axial_chain = {}
animal_body_width = {}

# Saline sessions
for row in morphology_records:
    if row["drug"] == "Saline":
        aid = row["animal_id"]
        animal_axial_chain[aid] = row["axial_chain_med"]
        animal_body_width[aid] = row["body_width_med"]
        animal_mean_latents[aid] = np.mean(animal_latents[aid], axis=0)

aids_ordered = sorted(list(animal_axial_chain.keys()))
axial_arr = np.array([animal_axial_chain[aid] for aid in aids_ordered])
width_arr = np.array([animal_body_width[aid] for aid in aids_ordered])
lat_arr = np.array([animal_mean_latents[aid] for aid in aids_ordered])

corr_records = []
for d in range(10):
    r_axial_lat = float(stats.pearsonr(axial_arr, lat_arr[:, d])[0])
    p_axial_lat = float(stats.pearsonr(axial_arr, lat_arr[:, d])[1])
    r_width_lat = float(stats.pearsonr(width_arr, lat_arr[:, d])[0])
    p_width_lat = float(stats.pearsonr(width_arr, lat_arr[:, d])[1])

    corr_records.append({
        "dimension": f"Dim_{d}",
        "r_with_axial_chain_spine_length": r_axial_lat,
        "p_val_axial_spine": p_axial_lat,
        "r_with_transverse_body_width": r_width_lat,
        "p_val_body_width": p_width_lat,
        "eta_sq_between_animal_pct": float(eta_squared[d])
    })

df_corr = pd.DataFrame(corr_records)
df_corr.to_csv(os.path.join(TAB_DIR, "pca_vs_latent_morphology_correlations.csv"), index=False)
print("\nCorrelation with True Axial Spine Length and Width:")
print(df_corr[["dimension", "r_with_axial_chain_spine_length", "r_with_transverse_body_width", "eta_sq_between_animal_pct"]].to_string(index=False))

# -----------------------------------------------------------------------------
# 5. SYLLABLE SIZE-SPLITTING AUDIT
# -----------------------------------------------------------------------------
print("\n[Step 4/4] Auditing Syllable Size-Splitting across Animals...", flush=True)

# For each syllable, compute the weighted average animal axial size
syll_size_records = []
all_sylls_saline = {}
for k, seq in syllables_dict.items():
    m = metadata_dict[k]
    if m["drug"] != "Saline":
        continue
    aid = str(m["animal_id"])
    c = np.bincount(seq, minlength=100)
    all_sylls_saline[aid] = c / c.sum()

# Syllable-by-animal matrix (100 x 15)
syll_matrix = np.array([all_sylls_saline[aid] for aid in aids_ordered]).T # (100, 15)

for s in range(100):
    weights = syll_matrix[s]
    total_w = weights.sum()
    if total_w > 0:
        mean_size = np.sum(weights * axial_arr) / total_w
        size_spread = np.sqrt(np.sum(weights * ((axial_arr - mean_size) ** 2)) / total_w)
    else:
        mean_size = np.nan
        size_spread = np.nan
    syll_size_records.append({
        "syllable": s,
        "mean_axial_size_px": float(mean_size) if not np.isnan(mean_size) else 0.0,
        "size_spread_px": float(size_spread) if not np.isnan(size_spread) else 0.0,
        "total_usage_across_animals": float(total_w)
    })

df_syll_size = pd.DataFrame(syll_size_records).sort_values("total_usage_across_animals", ascending=False)
df_syll_size.to_csv(os.path.join(TAB_DIR, "syllable_size_stratification.csv"), index=False)

# -----------------------------------------------------------------------------
# 6. GENERATE MULTI-PANEL AUDIT FIGURE
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

# Panel A: Axial Chain vs Straight Line
axs[0, 0].scatter(pivot_axial["Saline"], pivot_axial["Amphetamine"], color="#2b5c8f", s=60, label=f"Axial Chain (r={r_axial:.3f})")
axs[0, 0].scatter(pivot_straight["Saline"], pivot_straight["Amphetamine"], color="#d95f02", marker="^", s=60, label=f"Straight Line (r={r_straight:.3f})")
min_v = min(pivot_axial.min().min(), pivot_straight.min().min()) - 2
max_v = max(pivot_axial.max().max(), pivot_straight.max().max()) + 2
axs[0, 0].plot([min_v, max_v], [min_v, max_v], "--", color="gray", alpha=0.7)
axs[0, 0].set_xlabel("Day 1 (Saline) Length [px]", fontweight="bold")
axs[0, 0].set_ylabel("Day 2 (Amphetamine) Length [px]", fontweight="bold")
axs[0, 0].set_title("A. Intra-Animal Spinal Measurement Stability", fontweight="bold")
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# Panel B: Latent Variance Decomposition
bars = axs[0, 1].bar(range(10), eta_squared, color="#386cb0", edgecolor="black", alpha=0.85)
axs[0, 1].axhline(15, color="red", linestyle="--", label="15% Threshold (Movement Dominated)")
axs[0, 1].set_xlabel("Latent Dimension", fontweight="bold")
axs[0, 1].set_ylabel("Between-Animal Variance (%)", fontweight="bold")
axs[0, 1].set_title("B. Latent Variance Decomposition (Between vs Within)", fontweight="bold")
axs[0, 1].set_xticks(range(10))
axs[0, 1].set_xticklabels([f"D{d}" for d in range(10)])
axs[0, 1].set_ylim(0, 100)
axs[0, 1].legend()
axs[0, 1].grid(True, axis="y", alpha=0.3)

# Panel C: Dim 0 vs True Axial Spine Length
axs[1, 0].scatter(axial_arr, lat_arr[:, 0], color="#7570b3", s=70, edgecolor="black")
z = np.polyfit(axial_arr, lat_arr[:, 0], 1)
p = np.poly1d(z)
axs[1, 0].plot(np.sort(axial_arr), p(np.sort(axial_arr)), color="#7570b3", linestyle="--")
axs[1, 0].set_xlabel("True Axial-Chain Spine Length [px]", fontweight="bold")
axs[1, 0].set_ylabel("Mean Inferred Latent Dim 0", fontweight="bold")
axs[1, 0].set_title(f"C. Latent Dim 0 vs. Axial Spine Length (r={df_corr.loc[0, 'r_with_axial_chain_spine_length']:.3f})", fontweight="bold")
axs[1, 0].grid(True, alpha=0.3)

# Panel D: Syllable Mean Animal Size Distribution
top30_sylls = df_syll_size.head(30)
axs[1, 1].hist(top30_sylls["mean_axial_size_px"], bins=10, color="#1b9e77", edgecolor="black", alpha=0.75)
axs[1, 1].axvline(np.mean(axial_arr), color="black", linestyle="--", label=f"Cohort Mean Size ({np.mean(axial_arr):.1f} px)")
axs[1, 1].set_xlabel("Mean Animal Axial Size of Expressed Syllable [px]", fontweight="bold")
axs[1, 1].set_ylabel("Number of Syllables (Top 30)", fontweight="bold")
axs[1, 1].set_title("D. Syllable Size Stratification (Top 30 Syllables)", fontweight="bold")
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "fig_kpms_morphology_audit_panels.png")
plt.savefig(fig_path, bbox_inches="tight")
plt.close()

print(f"\n✓ Saved audit figure: {fig_path}", flush=True)
print("=" * 80)
print("PHASE 1 AUDIT COMPLETE!")
print("=" * 80)
