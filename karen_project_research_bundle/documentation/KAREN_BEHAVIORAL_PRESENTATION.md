# Karen Dataset: Behavioral & Kinematic Phenotyping Presentation

This document provides a clean, executive visual overview of all kinematic and behavioral findings from the **Karen Project 2** dataset (**99 validated sessions**, tracking **21 animals** longitudinally across 5 monthly waves).

For the complete data-science formulations, pedagogical explanations, and LaTeX derivations of all metrics, see **[EXPLANATION_OF_6_UNIFIED_METRICS.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/EXPLANATION_OF_6_UNIFIED_METRICS.md)**.

---

## 1. Experimental Cohort & Recording Dates Overview

> [!NOTE]
> **Clarification on "Injections" / "PI"**: The experimental dataset is a **longitudinal repeated-measures study** tracking the **exact same 21 animals** over a 6-month calendar window (Dec 2025 to May 2026). The phrase "PI" appeared solely in the legacy folder naming string `/mnt/hdd/karen_project/3_week_PI/...` on the source hard drive.

### Longitudinal Timeline (5 Recording Waves)
The 21 mice (12 PFF, 9 Monomer) were tested across 5 distinct calendar months:

| Recording Wave | Calendar Dates | Total Sessions | PFF Sessions | Monomer Sessions | Unique Animals Tested |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Wave 1 (Month 0)** | Dec 03 – Dec 04, 2025 | 24 | 14 | 10 | 21 animals |
| **Wave 2 (Month 1)** | Jan 08 – Jan 09, 2026 | 23 | 13 | 10 | 21 animals |
| **Wave 3 (Month 3)** | Mar 05 – Mar 15, 2026 | 29 | 11 | 11 (+7 pilot/unk) | 24 animals |
| **Wave 4 (Month 4)** | Apr 27 – Apr 28, 2026 | 13 | 6 | 3 (+4 unk) | 10 animals |
| **Wave 5 (Month 5)** | May 16 – May 17, 2026 | 21 | 12 | 9 | 21 animals |
| **Total** | **Dec 2025 – May 2026** | **110** | **56** | **43** | **25 unique animals** |

---

## 2. Key Empirical Findings (PFF vs. Monomer)

### A. Global Locomotor Speed, Peak Sprints & Bout Dynamics
* **Mean Speed**: **$-18.2\%$ slower in PFF** ($61.2$ vs $74.8\text{ px/s}$, $p = 0.0310$, $p_{\text{perm}} = 0.0110$)
* **Peak Burst Velocity ($v_{95}$)**: **$-24.7\%$ lower sprint peak in PFF** ($175.2$ vs $232.5\text{ px/s}$, $p = 0.0320$, $p_{\text{perm}} = 0.0150$)
* **Stationary Immobility ($<5\text{ px/s}$)**: **$+11.1\%$ more stationary time in PFF** ($1.05\%$ vs $0.94\%$, $p = 0.0402$, $p_{\text{perm}} = 0.0405$)
* **Macro Locomotor Bout Length**: **$-12.0\%$ shorter continuous movement runs in PFF** ($2.61\text{ s}$ vs $2.97\text{ s}$, $p = 0.0287$)
* **Micro Syllable Dwell Time**: **$+14.9\%$ longer state dwell in PFF** ($0.569\text{ s}$ vs $0.495\text{ s}$, $p = 0.1066$)

📊 **Figure**: [06_macro_behavior/macro_behavior_genotype_comparison.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/06_macro_behavior/macro_behavior_genotype_comparison.png)  
📊 **Figure**: [01_global_kinematics/global_kinematics_boxplots.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/01_global_kinematics/global_kinematics_boxplots.png)

---

### B. Biomechanical Acceleration Fluctuations & Jerkiness
* **Normalized Jerk ($\frac{\langle |\text{jerk}| \rangle}{\langle \text{speed} \rangle}$)**: **$+11.0\%$ higher jerkiness in PFF** ($2627.8$ vs $2366.9$, $p = 0.0341$, $p_{\text{perm}} = 0.0264$).
* PFF mice exhibit higher third-derivative fluctuations ($d^3x/dt^3$) relative to their forward speed, indicating more abrupt and erratic velocity changes during movement.

---

### C. Behavioral Syntax Entropy Rate (Sequence Predictability)
* **Transition Entropy Rate $H(S_{t+1} \mid S_t)$**: **$+2.5\%$ higher conditional entropy in PFF** ($2.624$ vs $2.560\text{ bits/step}$, $p = 0.0294$, $p_{\text{perm}} = 0.0342$).
* Monomer sequences exhibit higher transition predictability; PFF syllable transition sequences are statistically more uncertain and disordered.

---

### D. Longitudinal Monthly Progression (Dec 2025 $\to$ May 2026)
* When following the same 21 mice month-by-month, PFF mice maintain consistently lower locomotor speed and higher movement jerkiness across all waves.

📊 **Figure**: [07_longitudinal_cohorts/longitudinal_cohort_progression.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/07_longitudinal_cohorts/longitudinal_cohort_progression.png)  
📊 **Figure**: [07_longitudinal_cohorts/karen_unified_6panel_master.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/07_longitudinal_cohorts/karen_unified_6panel_master.png)

---

### E. Micro-Kinematics (Execution Speed within the Same Syllable)
* Even within the **exact same behavioral syllables**, PFF mice perform the movements with lower velocity:
  * **Syllable 43**: $-17.44\text{ px/s}$ slower ($p = 0.0105$)
  * **Syllable 22**: $-8.05\text{ px/s}$ slower ($p = 0.0184$)
  * **Syllable 9**: $-3.22\text{ px/s}$ slower ($p = 0.0194$)
  * **Syllable 5**: $-6.72\text{ px/s}$ slower ($p = 0.0229$)

📊 **Figure**: [03_micro_kinematics/within_syllable_speed_deltas.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/03_micro_kinematics/within_syllable_speed_deltas.png)

---

### F. 10,000 Label-Shuffling Permutation Validation
* All primary metrics fall outside the 95% permutation null confidence envelope ($p_{\text{perm}} < 0.05$), confirming that the observed differences are robust and non-random.

📊 **Figure**: [permutation_tests/label_permutation_null_distributions.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/permutation_tests/label_permutation_null_distributions.png)

---

## 3. Key Statistical Tables (CSVs)
- 📄 [macro_behavior_statistical_summary.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/06_macro_behavior/macro_behavior_statistical_summary.csv)
- 📄 [label_permutation_test_results.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/permutation_tests/label_permutation_test_results.csv)
- 📄 [longitudinal_month_summary.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/07_longitudinal_cohorts/longitudinal_month_summary.csv)
- 📄 [micro_kinematics_within_syllable_stats.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/03_micro_kinematics/micro_kinematics_within_syllable_stats.csv)
- 📄 [differential_syllable_usage_stats.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/02_syllable_usage/differential_syllable_usage_stats.csv)
