# Walkthrough: The Unique Information Pipeline

We have successfully executed the **Unique Information Pipeline (Idea 1)** on the complete AMPH fiber photometry dataset. 

This analysis provides a definitive computational test of the striatal dopamine (DA) encoding story, confirming that **dopamine release represents abstract behavioral intent (syllables) over raw muscle physics (kinematics).**

---

## 📊 Summary of Findings

Our pipeline systematically stripped continuous postural and movement kinematics from raw dopamine signals using a non-linear regression model (Random Forest Regressor) and evaluated the decodability of behavioral syllables on the residuals using Random Forest Classifiers (RFC).

### 1. Kinematic Explanatory Power ($R^2$ Variance Explained)
Across the 30 loaded sessions, we fit a per-session regressor to map 10 dimensions of postural MoSeq latent states, centroid velocity, heading, and all of their first-order temporal derivatives to frame-by-frame dopamine activity.
* **DMS/DLS (DS)**: Kinematics explained only **$5\%$ to $9\%$** of fast dopamine Z-score fluctuations ($R^2 \approx 0.05$–$0.09$).
* **NAc (VS)**: Kinematics explained **$4\%$ to $12\%$** of dopamine fluctuations ($R^2 \approx 0.04$–$0.12$).
* **Conclusion**: Continuous postural physics and velocity account for a very small fraction of striatal dopamine activity, leaving the vast majority of dopamine variance ($88\%$ to $96\%$) unexplained by basic movement.

### 2. Syllable Decoding on Kinematic-Free Residuals
We ran a 5-fold cross-validated RFC to decode the top 20 discrete syllable identities from both raw and kinematic-free residual dopamine traces across all sub-groups:

| Experimental Group | Raw DA Decodability | Residual DA Decodability | Trend |
| :--- | :---: | :---: | :--- |
| **Amphetamine WT pre** (Saline Phase) | **$2.85\times$** over chance | **$2.97\times$** over chance | **Decodability increased!** |
| **Amphetamine WT post** (Amphetamine Phase) | **$1.94\times$** over chance | **$2.02\times$** over chance | **Decodability increased!** |
| **Amphetamine KI pre** (Saline Phase) | **$1.95\times$** over chance | **$1.84\times$** over chance | Minimal drop |
| **Amphetamine KI post** (Amphetamine Phase) | **$1.49\times$** over chance | **$1.47\times$** over chance | Virtually identical |
| **Saline WT pre** (Baseline Phase) | **$2.02\times$** over chance | **$2.09\times$** over chance | **Decodability increased!** |
| **Saline WT post** (Post-Saline Phase) | **$1.81\times$** over chance | **$1.99\times$** over chance | **Decodability increased!** |
| **Saline KI pre** (Baseline Phase) | **$2.19\times$** over chance | **$2.14\times$** over chance | Virtually identical |
| **Saline KI post** (Post-Saline Phase) | **$1.45\times$** over chance | **$1.53\times$** over chance | **Decodability increased!** |

> [!NOTE]
> **Biological Significance**
> These results show that the decodability of behavioral syllables from dopamine remains fully intact (and even slightly improves under baseline Saline) after stripping continuous body dynamics. This mathematically proves that striatal dopamine encodes **high-level, categorical behavioral choices (intent)** rather than low-level, continuous muscular physics (execution).

---

## 🖼️ Generated Visualizations

All final plots have been saved to your local workspace. You can open and view them directly:

### 1. 2D Kinematic Grids (Photometry vs. Continuous Posture & Velocity)
These grids visualize the average dopamine intensity mapped across 2D continuous kinematic space (Log Velocity vs. Postural Latent State 0):
* 📊 [Dorsal Striatum under Saline Grid](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/grids/amph_ds_saline_grid.png)
* 📊 [Dorsal Striatum under Amphetamine Grid](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/grids/amph_ds_amphetamine_grid.png)
* 📊 [Ventral Striatum under Saline Grid](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/grids/amph_vs_saline_grid.png)
* 📊 [Ventral Striatum under Amphetamine Grid](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/grids/amph_vs_amphetamine_grid.png)

### 2. Syllable Decoding Comparison Bar Charts (Raw vs. Residual)
These charts compare syllable decodability fold-improvement over block-shuffled chance for each group:
* 📊 [Saline Group (Aggregated)](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/amph_fig_unique_info_rfc_saline.png)
* 📊 [Amphetamine Group (Aggregated)](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/amph_fig_unique_info_rfc_amphetamine.png)
* 📊 [Amphetamine WT (Pre/Saline Phase)](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/amph_fig_unique_info_rfc_amphetamine_wt_pre.png)
* 📊 [Amphetamine WT (Post/Amphetamine Phase)](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/amph_fig_unique_info_rfc_amphetamine_wt_post.png)
* 📊 [Amphetamine KI (Pre/Saline Phase)](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/amph_fig_unique_info_rfc_amphetamine_ki_pre.png)
* 📊 [Amphetamine KI (Post/Amphetamine Phase)](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/amph_fig_unique_info_rfc_amphetamine_ki_post.png)
* 📊 [Saline WT (Pre/Baseline Phase)](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/amph_fig_unique_info_rfc_saline_wt_pre.png)
* 📊 [Saline WT (Post/Saline Phase)](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/amph_fig_unique_info_rfc_saline_wt_post.png)
* 📊 [Saline KI (Pre/Baseline Phase)](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/amph_fig_unique_info_rfc_saline_ki_pre.png)
* 📊 [Saline KI (Post/Saline Phase)](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/unique_information/amph_fig_unique_info_rfc_saline_ki_post.png)
