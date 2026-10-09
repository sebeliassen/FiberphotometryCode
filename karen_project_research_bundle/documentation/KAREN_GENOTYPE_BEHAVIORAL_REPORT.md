# Karen Dataset Genotype Behavioral Analysis Report
**Comparison**: PFF ($N=56$) vs. Monomer (Control, $N=43$)

*For complete mathematical definitions, pedagogical descriptions, and permutation validations, see **[EXPLANATION_OF_6_UNIFIED_METRICS.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/EXPLANATION_OF_6_UNIFIED_METRICS.md)** and **[KAREN_BEHAVIORAL_PRESENTATION.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/KAREN_BEHAVIORAL_PRESENTATION.md)**.*

## Executive Summary
- **Global Kinematics**: Found **0 / 10** uncorrected global locomotor/spatial metrics showing significant differences at p < 0.05 on raw non-parametric MWU (significant when tested with Student's $t$ and 10,000 label-shuffling permutations).
- **Syllable Repertoire**: Found **6 / 100** syllables with significant differential occupancy ($p < 0.05$).
- **Within-Syllable Micro-Kinematics**: Found **5** syllables with significant execution speed modulation.
- **Behavioral Syntax & Transitions**: Found **46** bigram transitions significantly altered between genotypes.

## 1. Global Locomotor & Spatial Kinematics
| Metric | Monomer (Mean ± Std) | PFF (Mean ± Std) | % Change | Cohen's d | p-value (MWU) | FDR q-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **duration_min** | 37.06 ± 3.53 | 38.36 ± 3.67 | +3.5% | +0.36 | **0.1654** | 0.2757 |
| **mean_speed_px_s** | 74.77 ± 38.05 | 61.16 ± 14.99 | -18.2% | -0.50 | **0.0713** | 0.2593 |
| **median_speed_px_s** | 51.99 ± 23.16 | 45.22 ± 7.02 | -13.0% | -0.42 | **0.1354** | 0.2709 |
| **p95_speed_px_s** | 232.47 ± 155.90 | 175.16 ± 80.48 | -24.7% | -0.48 | **0.1037** | 0.2593 |
| **immobility_fraction** | 0.01 ± 0.00 | 0.01 ± 0.00 | +11.1% | +0.42 | **0.0950** | 0.2593 |
| **active_mean_speed_px_s** | 75.38 ± 38.13 | 61.75 ± 15.03 | -18.1% | -0.50 | **0.0713** | 0.2593 |
| **mean_accel_px_s2** | 970.62 ± 119.34 | 963.64 ± 98.57 | -0.7% | -0.06 | **0.6540** | 0.7323 |
| **abs_ang_vel_mean_rad_s** | 2.05 ± 2.41 | 1.77 ± 2.46 | -13.5% | -0.11 | **0.5556** | 0.7323 |
| **center_occupancy_fraction** | 0.13 ± 0.10 | 0.13 ± 0.09 | -2.1% | -0.03 | **0.9916** | 0.9916 |
| **total_distance_px** | 172302.16 ± 50538.38 | 166132.12 ± 33632.91 | -3.6% | -0.15 | **0.6591** | 0.7323 |


## 2. Top Differential Behavioral Syllables (p < 0.05)
| Syllable ID | Monomer Usage | PFF Usage | Log2 Fold Change | p-value (MWU) | Direction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Syllable 74** | 0.49% | 0.76% | +0.65 | **0.0109** | 🔺 Upregulated in PFF |
| **Syllable 68** | 0.42% | 0.64% | +0.63 | **0.0153** | 🔺 Upregulated in PFF |
| **Syllable 83** | 0.11% | 0.26% | +1.27 | **0.0210** | 🔺 Upregulated in PFF |
| **Syllable 31** | 1.86% | 2.60% | +0.49 | **0.0294** | 🔺 Upregulated in PFF |
| **Syllable 40** | 0.98% | 0.66% | -0.56 | **0.0327** | 🔻 Downregulated in PFF |
| **Syllable 14** | 1.43% | 1.86% | +0.38 | **0.0351** | 🔺 Upregulated in PFF |


## 3. Top Altered Behavioral Syntax / Transitions
| Transition | Monomer Prob | PFF Prob | Log2 Fold Change | p-value (MWU) |
| :--- | :--- | :--- | :--- | :--- |
| **Syllable 14 → 65** | 0.038% | 0.097% | +1.32 | **0.0002** |
| **Syllable 31 → 48** | 0.019% | 0.043% | +1.16 | **0.0010** |
| **Syllable 65 → 39** | 0.003% | 0.027% | +2.79 | **0.0019** |
| **Syllable 29 → 22** | 0.015% | 0.034% | +1.12 | **0.0030** |
| **Syllable 31 → 45** | 0.005% | 0.020% | +1.88 | **0.0031** |
| **Syllable 3 → 8** | 0.094% | 0.067% | -0.48 | **0.0072** |
| **Syllable 74 → 14** | 0.049% | 0.071% | +0.53 | **0.0073** |
| **Syllable 42 → 83** | 0.030% | 0.067% | +1.11 | **0.0081** |
| **Syllable 14 → 2** | 0.232% | 0.281% | +0.28 | **0.0087** |
| **Syllable 45 → 14** | 0.038% | 0.078% | +1.00 | **0.0100** |

