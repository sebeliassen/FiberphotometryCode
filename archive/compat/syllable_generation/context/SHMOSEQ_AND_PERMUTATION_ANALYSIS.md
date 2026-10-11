# SHMoSeq & Animal-Level Permutation Testing: Theoretical Foundations & Empirical Findings

**Authoritative reference on hierarchical behavioral modeling (SHMoSeq), exact animal-level permutation tests, and statistical rigor for the WT vs. KI (and Amph vs. Saline) dataset.**

---

## 1. Executive Summary

This document synthesizes:
1. **The Exact Animal-Level Permutation Test** on baseline Saline sessions across 15 animals (8 WT vs. 7 KI), demonstrating statistically significant baseline genotype separation.
2. **Statistical Corrections & Reporting Rigor**: Addressing FDR / Benjamini-Hochberg multi-test corrections across discrete states.
3. **SHMoSeq (Supervised / Semi-Hidden Hierarchical MoSeq)**: Multi-timescale decomposition, non-Markovian behavioral persistence, baseline drift removal, injection-handling artifacts, and dopamine photometry cross-validation.
4. **Catalogue of Key Generated Figures**: Master visual artifacts documenting model fitting, convergence, and biological findings.

---

## 2. Animal-Level Permutation Testing on Baseline (Saline) Data

### A. Experimental Design & Permutation Invariants
- **Cohort**: 15 animals total: **8 Wildtype (WT)** and **7 Knock-In (KI)**.
- **Data Scope**: Baseline **Saline sessions only** to test whether intrinsic behavioral phenotypes separate WT and KI in the absence of drug perturbation.
- **Unit of Randomization**: Shuffling is strictly performed **between animals** (preserving the 8/7 split), **NOT across frames or sessions**. Shuffling frames or treating repeated sessions from the same animal as independent violates exchangeability and leads to severe false-positive inflation.
- **Enumeration**: With 15 animals, the total number of unique assignments is exactly:
  $$\binom{15}{7} = 6,435 \text{ permutations}$$
  Antigravity enumerated all 6,435 assignments for exact, non-parametric p-values.

### B. Group Separation Metrics & Results
For each animal, its global state-occupancy profile was computed across fitted states. Group separation between WT and KI mean vectors was measured using three complementary distance metrics:

| Metric | Measured Statistic | Permutation $p$-value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Euclidean Distance** | $L_2$ geometric distance between group mean vectors | **$p \approx 0.010$** | Significant separation in Euclidean occupancy space |
| **Jensen–Shannon Distance** | Symmetrized KL divergence between probability distributions | **$p \approx 0.022$** | Significant distributional divergence |
| **Total Variation Distance** | Maximum difference in assigned probability across states | **$p \approx 0.044$** | Significant absolute probability shift |

> [!NOTE]
> These three metrics represent related checks on the same dataset rather than independent confirmations. Together, they rigorously confirm that the observed baseline differences between WT and KI animals have a **statistically significant genotype-associated component** that cannot be explained by random animal partitioning.

### C. Multiple Comparisons & FDR Rigor
When testing individual states for genotype-specific occupancy changes (e.g., states $Z_1 \dots Z_8$):
- **Raw vs. Adjusted $p$-values**: For State $Z_6$, the raw permutation p-value was $p = 0.01072$ (the lowest among the 8 tests).
- **Benjamini–Hochberg Correction**:
  $$q_{Z_6} = 8 \times \frac{0.01072}{1} \approx 0.086$$
- **Rigorous Reporting Rule**: A $q$-value of $0.086$ is a strong statistical trend, but does **not** meet the nominal $\alpha = 0.05$ threshold for FDR significance. Always report both raw $p$ and Benjamini-Hochberg FDR $q$ values on presentations and manuscripts to maintain strict scientific integrity.

---

## 3. SHMoSeq (Hierarchical MoSeq) Architecture & Diagnostics

Keypoint-MoSeq discovers sub-second syllables ($200\text{--}400\text{ ms}$). However, motivational states, pharmacological actions (Amphetamine), and genotypic phenotypes operate across **minutes to hours**. SHMoSeq models this hierarchy.

### A. Multi-Timescale Decomposition & Real vs. Markov Comparison
1. **The Non-Markovian Signature**:
   - A standard 1st-order Markov chain assumes state transitions depend only on the immediate preceding syllable:
     $$P(S_{t+1} \mid S_t, S_{t-1}, \dots) = P(S_{t+1} \mid S_t)$$
   - Real behavioral syllable sequences exhibit long-range persistence, mutual information decay spanning tens of seconds, and higher-order clusters.
   - We compared real syllable sequence autocovariance against synthetic Markov simulations, demonstrating that animals stay within macro-behavioral states (locomotor runs, grooming clusters, paused exploration) significantly longer than predicted by 1st-order Markov dynamics.
2. **Baseline Drift vs. State Dynamics**:
   - Long recordings (1 hour) undergo slow non-stationary drift due to habituation or drug pharmacokinetics ($C_{\max}$ and elimination).
   - Baseline-trend removal isolates true discrete state occupancy shifts from continuous drift.

### B. Injection-Handling Gaps & Confound Dissection
- **The Artifact**: Physical handling, syringe insertion, and returning the animal to the chamber produce a temporary disruption (handling gap).
- **Validation**: Cross-validation against DeepLabCut tracking timestamps proved that this transient behavior is an experimental artifact, distinct from true drug-induced states. Models must window out the handling interval to prevent false state discovery.

### C. Dopamine Photometry Cross-Validation
- Aligned continuous fiber photometry dF/F signals (dopamine sensor in striatum/nucleus accumbens) with hierarchical state entries and exits.
- Certain high-mobility states ($Z_1, Z_5$) coincide with sustained dopamine elevation, while paused/resting states ($Z_6$) show dopamine suppression, cross-validating the biological reality of the mathematical states.

---

## 4. Key Figures & Diagnostic Artifacts

All figures generated during this analysis are stored in the artifact directory (`/home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579/`):

| Figure Name | Content & Purpose |
| :--- | :--- |
| [`fig_exact_animal_permutation_test.png`](file:///home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579/fig_exact_animal_permutation_test.png) | Null permutation distribution (6,435 shuffles) vs. observed WT/KI separation across Euclidean, JS, and TV metrics |
| [`fig1_real_vs_markov_3panel.png`](file:///home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579/fig1_real_vs_markov_3panel.png) | Autocovariance and transition persistence comparing empirical behavioral sequences vs. 1st-order Markov simulations |
| [`fig1_real_vs_markov_baseline_removed.png`](file:///home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579/fig1_real_vs_markov_baseline_removed.png) | Real vs. Markov persistence after detrending slow habituation nonstationarity |
| [`fig1_timescale_by_group_comparison.png`](file:///home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579/fig1_timescale_by_group_comparison.png) | Timescale comparison between WT vs. KI and Saline vs. Amphetamine |
| [`fig2_shmoseq_fitting_convergence.png`](file:///home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579/fig2_shmoseq_fitting_convergence.png) | ELBO / log-likelihood convergence trajectories across MCMC / EM fitting iterations |
| [`fig3_shmoseq_state_characteristics.png`](file:///home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579/fig3_shmoseq_state_characteristics.png) | Kinematic speed, turning rate, syllable composition, and dwell times per hierarchical state |
| [`fig4_shmoseq_genotype_drug_dissection.png`](file:///home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579/fig4_shmoseq_genotype_drug_dissection.png) | Genotype (WT vs. KI) $\times$ Treatment (Saline vs. Amph) 2-way interaction on state occupancies |
| [`fig_session_state_timelines_k8.png`](file:///home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579/fig_session_state_timelines_k8.png) | Full 60-minute ethogram raster plots showing macro-state transitions across all sessions |
| [`fig_photometry_handling_cross_validation.png`](file:///home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579/fig_photometry_handling_cross_validation.png) | Dopamine dF/F transients aligned to hierarchical state onset/offset events |
