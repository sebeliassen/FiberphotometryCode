# Methodology Review: Two-Branch Behavioral Phenotyping (NMF & HMM)

This review package provides the verified codebase, configuration parameters, exact non-parametric permutation tests, and output tables derived from the 30-session Jonathan AMPH cohort (`jonathan_amph_sessions_run12_w_sylls.pkl`, Keypoint-MoSeq 17KP model, 229,579 continuous bouts, 80.025 Hz, $N=15$ mice: 8 WT, 7 DAT-Het KI).

---

## 1. Architectural Clarity: Separation of Two Complementary Branches

As clarified in our method audit, our framework consists of **two distinct, complementary analytical branches** operating at different levels of the behavioral hierarchy:

| Dimension | Branch A: Sliding-Window NMF | Branch B: Block HMM Macro-States |
| :--- | :--- | :--- |
| **Scientific Objective** | Characterize continuous behavioral repertoire diversity and dimensionality | Model discrete behavioral phase shifts, dwell times, and transition dynamics |
| **Mathematical Formulation** | Non-Negative Matrix Factorization: $X_{windows \times 100} \approx W H$ | Gaussian Hidden Markov Model: $Z_t \mid Q_t \sim \mathcal{N}(\mu_q, \Sigma_q)$ |
| **Temporal Discretization** | Continuous sliding window ($W = 20\text{s}$, 1-bout stride) | Non-overlapping blocks of 20 consecutive bouts |
| **Primary Metric** | Participation Ratio (PR) rank, component activation vectors $W$ | Physical session duration (minutes & %), entry rates, exit hazards, routing |
| **Key Finding** | KI Saline exhibits collapsed dimensionality (rank 8.4 vs 15.4) | KI Saline exhibits severe deficit in State $Q_4$ ($p = 0.0300$) driven by entry failure |

Neither branch relies on unwarranted causal assumptions. Both describe objective, reproducible regularities in behavioral structure.

---

## 2. Verified Data Provenance & Cleaning
- **Raw Source:** `syllable_generation/data/jonathan_amph_sessions_run12_w_sylls.pkl`
- **Session Records:** 30 complete sessions across 15 distinct mice (8 WT, 7 KI; each mouse tested under Saline and Amphetamine conditions).
- **Tracking Gap Filtering:** Merging is strictly terminated on tracking dropouts exceeding 2 frames ($>25\text{ms}$), ensuring no false bout bridges across camera skips.
- **Continuous Bout Cache:** 229,579 clean bouts extracted and verified (`run12_continuous_bout_cache.pkl`, 59.4 MB).

---

## 3. Methodological & Statistical Audits Implemented

### A. Resolution of the Double-Shuffle Permutation Bug
- In earlier exploratory drafts, certain Monte Carlo tests called `np.random.permutation` twice in calculating group differences.
- **Corrected Standard:** All mouse-level comparisons now evaluate an **exact combinatorial permutation test** across all $\binom{15}{7} = 6,435$ possible animal label partitions using a single complementary index split:
  $$p_{\text{exact}} = \frac{1 + \sum_{b=1}^{B-1} \mathbb{I}(|T_{\text{perm}}^{(b)}| \ge |T_{\text{obs}}|)}{B}, \quad B = 6,435.$$
- Where support is smaller (e.g. state exit events with $N_{\text{valid}} < 15$), exact $\binom{N}{n_1}$ combinations are enumerated directly.

### B. Physical Duration vs. Block Fraction
- Block fraction (proportion of 20-bout segments) is no longer conflated with continuous elapsed time.
- Both metrics are calculated and reported in parallel:
  - **State $Q_4$ Physical Time:** WT mice spend $26.8\%$ ($15.88\text{ min}$) in State $Q_4$ vs. KI mice spending $3.0\%$ ($1.78\text{ min}$) ($p_{\text{exact}} = 0.0300$, Cohen's $d = -1.11$, $95\%\text{ CI } [-44.0\%, -6.4\%]$).
  - **State $Q_4$ Block Fraction:** WT = $22.1\%$ vs. KI = $2.7\%$ ($p_{\text{exact}} = 0.0283$).

### C. Entry vs. Exit Decomposition
- The $Q_4$ deficit in DAT-Het KI is driven predominantly by a **6.2-fold reduction in entry rate**:
  $$\lambda_{\text{enter}}(Q_4): \text{ WT } = 0.0864 \ (8.6\%\text{/blk}) \text{ vs. KI } = 0.0140 \ (1.4\%\text{/blk}), \quad p_{\text{exact}} = 0.0179.$$
- Once inside $Q_4$, KI mice also exhibit an elevated exit hazard ($67.8\%$ vs. $38.4\%$, $p_{\text{exact}} = 0.0518$).

### D. Conserved Within-State Fine Motor Syntax
- When animals are in the same macro-state, their sub-second syllable execution is conserved:
  - Within-state syllable composition (JSD): $p = 0.0712$ ($Q_1$), $p = 0.3648$ ($Q_2$), $p = 0.0928$ ($Q_4$).
  - Syntax cross-transfer penalty: $p = 0.8321$ ($Q_1$), $p = 0.9685$ ($Q_2$), $p = 0.3551$ ($Q_4$).
- This confirms that genotype divergence is **hierarchical**: fine motor control is preserved, but state selection and dwell duration are altered.

---

## 4. Master Summary of Results (`run_12` Dataset)

| Metric | Source Table | WT Saline ($N=8$) | DAT-Het KI Saline ($N=7$) | Difference (KI - WT) | Statistical Test | Exact $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$Q_4$ Physical Time Occupancy** | `table_comp1_plain_occupancy.csv` | $26.8\% \ (15.9\text{m})$ | $3.0\% \ (1.8\text{m})$ | $-23.8\% \ (d = -1.11)$ | Exact Animal Permutation | **$p = 0.0300$** |
| **$Q_4$ Block Occupancy** | `table_comp1_plain_occupancy.csv` | $22.1\%$ | $2.7\%$ | $-19.4\%$ | Exact Animal Permutation | **$p = 0.0283$** |
| **$Q_4$ Entry Rate** | `table_comp3_entry_exit_decomposition.csv` | $8.64\%$ | $1.40\%$ | $-7.25\%$ | Exact Animal Permutation | **$p = 0.0179$** |
| **$Q_4$ Exit Hazard** | `table_comp3_entry_exit_decomposition.csv` | $38.39\%$ | $67.83\%$ | $+29.44\%$ | Exact Animal Permutation | **$p = 0.0518$** |
| **$Q_2$ Exit Hazard** | `table_comp3_entry_exit_decomposition.csv` | $48.35\%$ | $19.99\%$ | $-28.36\%$ | Exact Animal Permutation | **$p = 0.0393$** |
| **$Q_4 \to Q_2$ Routing** | `table_comp4_routing_after_exit.csv` | $3.4\%$ | $20.4\%$ | $+17.0\%$ | Exact Animal Permutation | **$p = 0.0402$** |
| **$Q_2$ Within-State Syntax JSD** | `table_comp5_within_state_syllables.csv` | $0.2762$ | $0.2547$ | Between: $0.2682$ | Animal Permutation | $p = 0.3648$ (Conserved) |
| **$Q_2$ Syntax Cross Penalty** | `table_comp6_within_state_local_transitions.csv`| $-0.0012\text{ b}$ | - | $p = 0.9685$ | Model Transfer | Conserved |
| **Decoder Accuracy (Occ + Entry/Exit)** | `table_comp8_genotype_decoder.csv` | $\text{Acc} = 73.3\%$ | $\text{BalAcc} = 74.1\%$ | $\text{AUC} = 0.750$ | LOAO Permutation | $p = 0.0680$ |

---

## 5. File Manifest

```
nmf_review_package/
├── code/
│   ├── build_run12_clean_bout_cache.py
│   ├── run_simple_genotype_comparison_suite.py
│   ├── run_balanced_shared_nmf.py
│   ├── run_nmf_adversarial_audits.py
│   ├── render_slide7_component_interpretation_figure.py
│   ├── render_slide7_nmf_retention_figure.py
│   ├── render_slide8_trajectory_figure.py
│   └── bout_features.py
├── tables/
│   ├── table_comp1_plain_occupancy.csv
│   ├── table_comp2_time_resolved_occupancy.csv
│   ├── table_comp3_entry_exit_decomposition.csv
│   ├── table_comp4_routing_after_exit.csv
│   ├── table_comp5_within_state_syllables.csv
│   ├── table_comp6_within_state_local_transitions.csv
│   ├── table_comp7_state_age_trajectories.csv
│   ├── table_comp8_genotype_decoder.csv
│   ├── table_balanced_shared_nmf_occupancy.csv
│   ├── audit_leakage_free_lambda_sweep.csv
│   └── audit_permutation_control.csv
├── figures/
│   ├── fig_simple_genotype_comparisons.png
│   └── fig_balanced_shared_nmf_occupancy.png
└── docs/
    ├── 01_professor_pitch_narrative_and_slides.md
    ├── 06_nmf_retention_investigation_and_adversarial_audit.md
    └── 10_hierarchical_genotype_dissection_and_network_routing.md
```
