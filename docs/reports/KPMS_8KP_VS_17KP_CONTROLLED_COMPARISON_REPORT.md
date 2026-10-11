# Controlled Pipeline Evaluation: 8-Keypoint vs. 17-Keypoint KPMS

**Date**: September 08, 2026  
**Evaluation Scope**: Controlled head-to-head comparison of two complete Keypoint-MoSeq pipelines:
- **17KP Candidate**: `amph_prod_k150k_run_12` (17 keypoints, $D=10$, Seed 41)
- **8KP Working Candidate**: `amph_8kp_k150k_run_03` (8 informative keypoints, $D=6$, Seed 202)  
**Cohort**: 30 recordings across 15 unique animals (8 WT, 7 DAT-Het KI; paired Saline and Amphetamine sessions)  
**Total Frames**: 5,516,954 frames @ 40 FPS  
**Figure Reference**: [`fig_controlled_8kp_vs_17kp_comparison.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/morphology_audit/figures/fig_controlled_8kp_vs_17kp_comparison.png)  

---

## 1. Executive Summary & Working Decision

This controlled evaluation compares the two complete pipelines under matched experimental conditions. Based on these quantitative metrics:
- **`amph_8kp_k150k_run_03` is adopted as our working input for downstream shMoSeq exploration**. We are developing an effective shMoSeq modeling pipeline, not establishing that 8KP is scientifically superior to 17KP.
- Removing perimeter keypoints and reducing latent dimensions ($D=10 \to D=6$) **did not eliminate body size information** from the continuous latent space. Both pipelines retain substantial predictability of mouse physical dimensions.
- However, persistent body size predictability in continuous latent space **does not establish that the discrete syllables are compromised** or unsuitable for behavioral modeling.
- Both pipelines yield plausible, comparable syllable duration timescales (16–19 frames, ~400–475 ms) and reliable genotype-associated differences under exact animal-level permutation testing.

---

## 2. Quantitative Results Summary

| Metric Category | Evaluation Metric | 17KP Pipeline (`run_12`, D=10) | 8KP Pipeline (`run_03`, D=6) | Measured Association & Context |
| :--- | :--- | :---: | :---: | :--- |
| **Continuous Latents** | Mean Between-Animal Variance ($\eta^2$) | **15.01%** | **14.76%** | Difference of 0.24% in animal variance partitioning |
| | Max Between-Animal Variance ($\eta^2$) | **32.49%** (Dim 4) | **32.76%** (Dim 4) | Animal-associated variance concentrates in Dim 4 in both models |
| | Max Univariate Correlation (Spine Length) | **r = 0.9636** | **r = 0.9547** | Single-axis collinearity with spine length |
| | Max Univariate Correlation (Body Width) | **r = 0.9440** | **r = 0.9821** | Single-axis collinearity with body width |
| **Multivariate Prediction** | **LOO Ridge $R^2$ (Spine Length)** | **$R^2$ = 0.8508** | **$R^2$ = 0.8241** | Full latent space decodability (MAE: 1.29 px vs. 1.53 px) |
| | **LOO Ridge $R^2$ (Body Width)** | **$R^2$ = 0.8759** | **$R^2$ = 0.9231** | Full latent space decodability (MAE: 0.53 px vs. 0.40 px) |
| **Syllable Consistency** | Within-Animal Cross-Session $r$ (Saline vs Amph) | **r = 0.7633** | **r = 0.8060** | Cross-session consistency for the same individual across days |
| | Between-Animal Cross-Session $r$ | **r = 0.3370** | **r = 0.3069** | Unpaired between-animal baseline similarity |
| | Animal Individuality Index ($\Delta r$) | **0.4262** | **0.4991** | Measured difference between paired and unpaired correlations |
| **Size Stratification** | Top 30 Syllable Mean Size Spread (SD) | **1.50 px** | **2.43 px** | Dispersion of mean body sizes across top expressed syllables |
| **Timescales** | Median Syllable Duration | **16.0 frames** (400 ms) | **19.0 frames** (475 ms) | Within natural ethological range (15–20 frames, ~375–500 ms) |
| | Duration IQR | **25.0 frames** | **31.0 frames** | Breadth of syllable duration distribution |
| | Short Flickering (<75 ms) | **8.25%** | **6.88%** | Percentage of bouts lasting < 3 frames |
| **Genotype Separation** | Exact Permutation $p$ (6,435 combos) | **p = 0.0059** (F=2.64) | **p = 0.0026** (F=2.80) | Genotype-associated separation preserving paired animal sessions |

---

## 3. Multi-Seed Stability in 8-Keypoint Ensemble

To assess whether these metrics are specific to Seed 202, we evaluated the other completed 8KP production models:

| Model Run | Seed | Latent LOO $R^2$ (Spine Length) | Median Duration | Individuality Index ($\Delta r$) | Exact Genotype $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`amph_8kp_k150k_run_03`** | `202` | **0.8241** | **19.0 frames** | **0.4991** | **p = 0.0026** |
| `amph_8kp_k150k_run_01` | `42` | **0.8086** | **17.0 frames** | **0.4870** | **p = 0.0061** |
| `amph_8kp_k150k_run_02` | `101` | **0.8180** | **21.0 frames** | **0.5117** | **p = 0.0068** |

Summary metrics are broadly consistent across seeds, indicating stable convergence behavior in the 8KP formulation.

---

## 4. Key Measured Findings vs. Hypotheses

1. **Multivariate Morphology Decodability**:
   - Evaluating out-of-fold cross-validated Ridge regression confirms that physical body dimensions remain predictable from the continuous latent coordinates in both models ($R^2 \approx 0.81\text{--}0.85$ for length; $R^2 \approx 0.88\text{--}0.93$ for width).
   - Changing the skeleton and latent dimensionality did not isolate pose from scale. However, this shared physical footprint does not demonstrate that discrete syllable assignments are compromised or invalid.
2. **Kinematic Timescales**:
   - Both pipelines maintain median syllable durations in the expected 16–19 frame range (~400–475 ms).
   - The proportion of rapid flickering (<75 ms) is slightly lower in 8KP (6.88% vs. 8.25%).
3. **Cross-Session Consistency and Individuality**:
   - Animals exhibit higher cross-session profile correlation across paired Saline and Amphetamine days ($r \approx 0.76\text{--}0.81$) than unpaired animals exhibit to each other ($r \approx 0.31\text{--}0.34$).
   - This individual-associated consistency represents a measured association. Whether it stems from persistent individual behavioral tendencies, anatomical differences, or animal-specific tracking features remains unresolved.
4. **Genotype-Associated Differences**:
   - Both pipelines exhibit statistically significant separation between DAT-Het KI and WT cohorts under exact paired permutation testing ($p = 0.0059$ in 17KP; $p = 0.0026$ in 8KP). The underlying neurobiological vs. physical drivers of this separation require further investigation.

---

## 5. Visual Check: Targeted Grid Movies

Before fitting higher-order shMoSeq models, a targeted visual check of common syllables (e.g. top frequent syllables 0, 1, 6, 10) and syllables exhibiting extreme size associations (e.g. syllables 41, 33, 43, 49) will confirm:
1. Absence of obvious tracking artifacts.
2. Whether syllables represent coherent actions across animals of differing body sizes rather than size-dependent splits.
