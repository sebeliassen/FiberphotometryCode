# Hierarchical Shared-Subroutine Behavioral Model: Repaired Feasibility & Progression Audit

## Executive Summary of Repairs & Discoveries
This package contains the complete implementation, quantitative metrics, and diagnostic figures from the **repaired, boundary-aware Hierarchical Shared-Subroutine Model** incorporating all reviewer feedback:

1. **Multi-Sequence Baum-Welch**:
   - Recordings are fitted as independent sequences ($N=30$ sessions, 45,000 total bouts, 15 mice).
   - E-step accumulates expected counts $\xi$ and $\gamma$ across sessions before updating shared parameters.
   - Forward-backward in float32 operates on $\sim 1,500$ bouts per sequence, eliminating numerical underflow/drift. Log-likelihood now increases strictly monotonically on every iteration.
2. **Boundary-Safe Segmentation**:
   - Zero routine occurrences or macro-state visits cross recording joins.
   - Boundary-truncated units (touching session start $t=0$ or end $t=T$) are explicitly flagged and excluded from completion dwell statistics.
   - Longest true macro-state visit within a single recording lasts **14.9 minutes** (893.8 s), not 51.6 minutes.
   - True contributing mice are counted directly from metadata ($13\text{--}15$ unique mice per routine, eliminating `"mixed"`).
3. **The Progression Experiment (Free vs. Tied Stage Emissions)**:
   - Evaluated 5-fold cross-validation on held-out recordings comparing:
     * **Model A (Free Stage Progressions)**: $\boldsymbol{\theta}_{k, s}$ independent per stage.
     * **Model B (Tied Stage Mixtures)**: $\boldsymbol{\theta}_{k, 0} = \boldsymbol{\theta}_{k, 1} = \boldsymbol{\theta}_{k, 2} = \boldsymbol{\theta}_k$.
   - **Result**: Model A beats Model B across **all 5 folds** with a mean held-out advantage of **$+0.1019 \pm 0.0479$ log-likelihood per bout**.
   - **Conclusion**: Routines contain **genuine, reproducible sequential progressions across stages**, rather than merely grouping stationary behavioral mixtures.
4. **Control A (Unbiased 1st-Order Markov Surrogates)**:
   - Preserves exact session transition sparsity with zero self-transitions ($s \to s$), using the identical 25-iteration budget.
   - Real data is significantly more predictable than the 1st-order null: $\text{LL/bout} = -3.1977$ vs. $-3.2242 \pm 0.0111$ ($\Delta = +0.0265$).
5. **Control B (Initialization Sensitivity in Multi-Sequence)**:
   - Initial 4, 6, and 10 bouts lead to learned means of **6.69, 8.05, and 8.82 bouts**.
   - Short routines (R2, R4, R5) consistently dwell around $3.2\text{--}4.3\text{ bouts}$, while long routines (R3, R6) dwell around $11.3\text{--}17.9\text{ bouts}$.

---

## Metric Tables (Boundary-Clean)

### 1. Level 1: Subroutines (Non-Truncated Completed Occurrences)

| Routine ID | Learned $p_{\text{adv}}$ | $\mathbb{E}[D_k]$ | Clean Occurrences | Rate (/min) | Median Bouts | Mean Bouts | 90th % Bouts | Median Duration | Bout Share | Animals |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Routine 1** | 0.452 | 6.63 b | 392 | 0.81 | 6.0 b | 7.03 b | 13.0 b | 3.74 s | 6.2% | 13 / 15 |
| **Routine 2** | 0.770 | 3.90 b | 2,086 | 4.36 | 3.0 b | 3.78 b | 5.0 b | 1.53 s | 17.8% | 15 / 15 |
| **Routine 3** | 0.185 | 16.19 b | 369 | 0.77 | 11.0 b | 17.88 b | 37.0 b | 7.53 s | 15.0% | 14 / 15 |
| **Routine 4** | 0.649 | 4.62 b | 2,321 | 4.78 | 4.0 b | 4.28 b | 6.0 b | 2.48 s | 22.1% | 15 / 15 |
| **Routine 5** | 0.917 | 3.27 b | 2,975 | 6.12 | 3.0 b | 3.20 b | 4.0 b | 1.60 s | 21.2% | 15 / 15 |
| **Routine 6** | 0.290 | 10.34 b | 702 | 1.46 | 8.0 b | 11.34 b | 20.9 b | 4.72 s | 17.8% | 15 / 15 |

### 2. Level 2: Upper Macro-State Visits (Within Single Recordings)

| Macro State | Total Visits | Clean Visits | Median Routines / Visit | 90th % Routines / Visit | Median Visit Duration | Mean Visit Duration | Max Visit Duration | Total Time Occupied |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$U_1$** | 59 | 26 | 39.0 routines | 127.0 routines | 109.6 s (1.8 min) | 152.5 s (2.5 min) | 631.9 s (10.5 min) | 275.4 min (56.6%) |
| **$U_2$** | 48 | 33 | 17.0 routines | 45.8 routines | 103.8 s (1.7 min) | 169.4 s (2.8 min) | 893.8 s (14.9 min) | 169.5 min (34.8%) |
| **$U_3$** | 5 | 1 | 200.0 routines | 200.0 routines | 558.2 s (9.3 min) | 558.2 s (9.3 min) | 558.2 s (9.3 min) | 41.8 min (8.6%) |

### 3. Progression Experiment: 5-Fold Cross-Validation

| Fold | Test Recordings | Free Progression LL/bout | Tied Mixture LL/bout | $\Delta$ Advantage | Winning Architecture |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 6 sessions | -3.2381 | -3.3902 | **+0.1521** | Free Progression |
| **Fold 2** | 6 sessions | -3.3799 | -3.4022 | **+0.0223** | Free Progression |
| **Fold 3** | 6 sessions | -3.2947 | -3.4106 | **+0.1159** | Free Progression |
| **Fold 4** | 6 sessions | -3.2045 | -3.3121 | **+0.1076** | Free Progression |
| **Fold 5** | 6 sessions | -3.4047 | -3.5165 | **+0.1117** | Free Progression |
| **Mean $\pm$ Std** | | **$-3.3044$** | **$-3.4063$** | **$+0.1019 \pm 0.0479$** | **Free Progression (5/5 Folds)** |

---

## Package File Manifest
- `hierarchical_subroutine_model.py`: Multi-sequence GPU JAX model engine supporting Free and Tied stage emissions.
- `run_hierarchical_feasibility_experiment.py`: Master experiment pipeline with boundary-safe multi-sequence fitting, progression CV, and unbiased surrogates.
- `cluster_duration_baseline.py`: Per-cluster baseline run duration breakdown (Cluster 5 active repertoire audit).
- `plot_hierarchical_diagnostics.py`: 6-panel diagnostic plotter.
- `fig_hierarchical_pilot_diagnostics_repaired.png`: High-resolution 6-panel diagnostic plot.
- `progression_free_vs_tied_cv_results.csv`: Complete fold-by-fold results of the progression experiment.
- `real_level1_routine_summary_boundary_clean.csv`: Level 1 summary table.
- `real_level2_macro_summary_boundary_clean.csv`: Level 2 summary table.
- `control_a_surrogate_summary_repaired.csv`: Null surrogate test results ($N=10$).
- `control_b_initialization_sensitivity_repaired.csv`: Multi-sequence prior sensitivity sweep table.
- `real_routine_occurrences_boundary_clean.csv`: Individual routine occurrences with boundary truncation flags.
- `hierarchical_pilot.log`: Complete execution log.
