# Final Resolution Decision & Dual-Stream Presentation Framework

## 1. Executive Decision: Level 4 for Behavioral Families + Level 1 for Behavioral Paths

Based on the quantitative out-of-sample cross-validation results across all 15 mice and 86,411 continuous syllable bouts, **Level 4 (14 broad functional states)** is adopted as the primary working vocabulary for macro-scale sequence discovery and presentation, while **Level 1 (100 raw states)** is retained to inspect fine-grained postural "paths."

This decision represents a **principled ethological and analytical trade-off**, rather than asserting one universally correct resolution.

---

## 2. Decisive Quantitative Comparison (The Trade-Off)

| Metric / Evaluation | Level 3 (22 Functional States) | Level 4 (14 Broad States) | Statistical Difference | Ethological Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Five-step population time coverage** | **46.1%** *(942 families)* | **76.2%** *(1,524 families)* | **$+30.1\%$ coverage gain** | Over three-quarters of all behavior is organized into recurring multi-animal programs. |
| **Six-step population time coverage** | **9.4%** *(collapse)* | **46.0%** *(robust)* | **Nearly $5\times$ greater recurrence** | Level 3 suffers catastrophic combinatorial fragmentation at 6 steps; Level 4 remains stable. |
| **Held-out heading prediction error ($\Delta\theta$)** | **$21.02^\circ \pm 0.96^\circ$** | **$21.58^\circ \pm 1.06^\circ$** | **$+0.56^\circ$ difference** ($+2.7\%$, $p = 6.3 \times 10^{-4}$) | Modest increase in out-of-sample heading error across 15 animal folds. |
| **Held-out speed prediction error ($v$)** | **$27.85 \pm 2.32\text{ px/s}$** | **$28.49 \pm 2.46\text{ px/s}$** | **$+0.65\text{ px/s}$ difference** ($+2.3\%$, $p = 1.8 \times 10^{-3}$) | Within-state speed prediction remains virtually identical between levels. |
| **Held-out duration prediction error ($d$)** | **$0.429 \pm 0.021\text{ s}$** | **$0.434 \pm 0.022\text{ s}$** | **$+0.005\text{ s}$ difference** ($+1.2\%$, $p = 6.4 \times 10^{-6}$) | Difference is approximately $5\text{ milliseconds}$. |
| **Held-out displacement prediction error ($\Delta r$)** | **$24.10 \pm 2.63\text{ px}$** | **$24.54 \pm 2.76\text{ px}$** | **$+0.45\text{ px}$ difference** ($+1.8\%$, $p = 1.1 \times 10^{-2}$) | Sub-pixel difference in spatial translation per bout. |

---

## 3. Presentation Framing & Guidelines

### The Core Presentation Claim:
> **"Broader grouping substantially increases recurrent-sequence coverage, with a modest cost in predicting the measured bout-level kinematics."**

### Critical Nuances:
1. **Avoid the phrase "only 3% information loss":**
   - These cross-validation evaluations measure **four specific bout-level kinematic summaries** (heading change, centroid speed, duration, and net displacement), not every fine aspect of 3D skeletal posture and coordination.
   - Claiming "only 3% information loss" overstates what was tested. The accurate statement is that prediction errors on these key movement summaries increase by only $1.2\%$ to $2.7\%$ ($0.56^\circ$ heading, $0.65\text{ px/s}$ speed, $5\text{ ms}$ duration).
2. **Level 4 buys massive recurrence support:**
   - In Level 4, 5-step chains structure **$76.2\%$** of all recording time across 15 mice.
   - At 6 steps, Level 4 structures **$46.0\%$** of behavior, whereas Level 3 collapses to **$9.4\%$** and Level 1 collapses to **$1.7\%$**.
   - This fits our primary scientific goal: discovering recognizable, recurring behavioral programs with meaningful natural variation across animals.

---

## 4. The Dual-Stream Architecture: "Families" + "Paths" + "Kinematics"

Rather than treating resolution selection as mutually exclusive, we adopt a multi-scale behavioral framework:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STREAM 1: MACRO-SYNTAX & BEHAVIORAL FAMILIES (Level 4, 14 States)                       │
│ - Unsupervised seqNMF sequence discovery discovers recurring multi-second themes       │
│ - Overarching behavioral "verbs": [Rear -> Descent -> Sprint]                           │
│ - 76.2% population time coverage at 5 steps; robust multi-animal statistical support    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STREAM 2: MICRO-STRUCTURE & BEHAVIORAL PATHS (Level 1, 100 States)                     │
│ - Traces the exact constituent syllable sequence through high-dimensional posture space│
│ - Specific mechanical "paths": e.g., syll_014 -> syll_004 -> syll_003                   │
│ - Resolves fine-grained biomechanical execution and pose nuances                       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STREAM 3: CONTINUOUS KINEMATIC EXECUTION (Physical Dynamics)                           │
│ - Measures continuous "adverbs" describing execution vigor and trajectory geometry     │
│ - Peak/mean velocity (px/s), net displacement (px), bout duration (s)                  │
│ - Signed heading change Δθ (deg): distinguishes sharp turns from gentle veers           │
│ - Completely eliminates categorical blur while preserving massive sequence support     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Specific Role of Level 3

Level 3 (22 states) remains fully supported in the codebase and is recommended when hypotheses specifically focus on:
- **Vertical descent laterality:** `Descent_L` vs. `Descent_R` vs. `Stand_Down` (within-state heading dispersion is $9.91\%$ lower in L3).
- **Sensory scanning laterality:** `Look_L` vs. `Look_R` (within-state speed dispersion is $10.25\%$ lower in L3).
- **Fine sharp gait adjustments:** `Walk_Sharp` vs. forward locomotion.

For general cohort sequence discovery and baseline genotype comparisons, **Level 4** is the primary choice.

---

## 6. Registry of Verified Scripts, Figures, and Tables

| Asset | Path | Description |
| :--- | :--- | :--- |
| **LOAO Evaluation Script** | [`evaluate_kinematic_information_retention.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/evaluate_kinematic_information_retention.py) | Leave-One-Animal-Out cross-validation script across 15 mouse folds and 86,411 bouts |
| **Audit Script** | [`audit_clustering_discrepancy.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/audit_clustering_discrepancy.py) | Reproducible script for matched action chains and discrepancy audit |
| **LOAO Summary CSV** | [`loao_kinematic_information_retention_summary.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/diagnostic_audit/loao_kinematic_information_retention_summary.csv) | MAE, SEM, 95% CI, and $R^2$ across all 4 levels |
| **Per-Animal Fold Errors CSV** | [`loao_cross_validation_per_animal_errors.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/diagnostic_audit/loao_cross_validation_per_animal_errors.csv) | Test errors on held-out mice for each fold |
| **Action Dispersion CSV** | [`action_class_kinematic_dispersion_breakdown.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/diagnostic_audit/action_class_kinematic_dispersion_breakdown.csv) | Within-state dispersion inflation from L3 to L4 |
| **Matched Benchmark CSV** | [`matched_action_chains_per_episode_audit.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/diagnostic_audit/matched_action_chains_per_episode_audit.csv) | 24,983 candidate episodes across all 3 themes |
| **LOAO Figure** | [`fig_kinematic_information_retention_loao.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_kinematic_information_retention_loao.png) | 4-panel publication figure of LOAO-CV prediction errors |
| **Complete Package Zip** | `/home/fsp585/Desktop/chatgpt_ethological_seqnmf_package.zip` | Self-contained archive with all scripts, tables, figures, and reports |
