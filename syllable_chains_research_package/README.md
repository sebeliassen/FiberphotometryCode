# Syllable Chain Dynamics & Auto-Discovery Framework (Project B)

> **Briefing & Research Package for AI Research Models**  
> **Authors:** Antigravity & Behavioral Neurobiology Team  
> **Repository Scope:** Unsupervised motif mining, soft-aligned sequence tracking, inter-chain syntax organization, intra-chain biomechanical execution fidelity, and context-dependent syllable kinematics.

---

## 1. Executive Purpose & Scientific Objective

Modern neuroethology tools like **Keypoint-MoSeq (KPMS)** decompose animal behavior into discrete, sub-second movement primitives called **syllables** (typically lasting 200–500 ms). While powerful, analyzing behavior purely at the level of individual syllables assumes animals operate as memoryless Markov chains, obscuring higher-order behavioral organization.

In reality, animals string syllables into recurring multi-second routines or **behavioral chains** (e.g., *rear $\to$ sniff $\to$ descend*, or *dash $\to$ wall-bound $\to$ steer*). 

**Project B: Syllable Chains** provides an automated, cohort-agnostic computational engine that:
1. **Discovers** recurring multi-stage behavioral routines from continuous syllable streams (via unsupervised n-gram frequency mining or seqNMF matrix factorization).
2. **Aligns** empirical executions with soft-matching tolerances (accounting for insertions, omitted steps, stutters, and early truncations).
3. **Quantifies Inter-Chain Organization:** How routines repeat, transition, continue, and initiate.
4. **Quantifies Intra-Chain Execution Fidelity:** How stereotyped the kinematics are (via Dynamic Time Warping), completion rates, abort hazard curves, and temporal stage pacing.
5. **Measures Context-Dependent Kinematics:** How the velocity and duration of individual syllables change depending on the higher-order chain in which they are embedded.

This package contains the complete production codebase, empirical results from two major mouse cohorts (**Amphetamine WT vs. DAT-Het KI** and **Parkinson's PFF vs. Monomer**), full documentation, and raw data outputs.

---

## 2. Package Architecture & File Map

```
syllable_chains_research_package/
├── README.md                            # This comprehensive master research briefing
├── syllable_chains/                     # Core Python Analysis Package
│   ├── __init__.py                      # Package exports
│   ├── core.py                          # Data models, CohortDataAdapter, ChainFinder, auto_discover_chains
│   ├── inter_chain.py                   # Inter-chain organization metrics (repeats, hazard, transitions, lead vs rep)
│   ├── intra_chain.py                   # Intra-chain execution metrics (DTW consistency, completion, variations, pacing)
│   ├── visualizer.py                    # High-density, space-efficient multi-panel publication figures
│   └── pipeline.py                      # Unified CLI runner (supports KPMS HDF5, CSV, session pickles)
├── docs_and_context/                    # Historical Neuroethological & Theoretical Notes
│   ├── PROJECT_B_BACKGROUND_AND_PLANS.md# Original project motivation & analytical roadmap
│   ├── PROJECT_B_COMPREHENSIVE_REPORT.md# Full analytical report with mathematical formulations
│   ├── DECODED_SEQNMF_CHAINS_WITH_HUMAN_LABELS.md # Human-curated chain semantics & behavioral themes
│   ├── COMPREHENSIVE_SEQUENCE_STATISTICS_REPORT.md# Initial baseline sequence statistics
│   ├── SUPER_CHAIN_EPISODE_ANALYSIS_PLAN.md       # Hierarchical super-chain structuring plan
│   ├── HIERARCHICAL_MODEL_WALKTHROUGH.md          # Multi-scale hierarchical model walkthrough
│   ├── monday_14_7_2026_presentation.md          # Lab meeting briefing notes
│   ├── WHOLE_CHAIN_EXEMPLAR_MACRO_MOTIFS.md       # Macro-motif exemplar catalog
│   ├── SEQNMF_LEVEL1_LEVEL4_RECOVERED_CONTEXT.md  # Multi-level seqNMF recovered context
│   ├── FUTURE_INTRA_SYLLABLE_ANALYSIS_FRAMEWORK.md# Frame-level micro-movement architectural blueprint
│   └── PRESENTATION_LLM_BRIEFING_AND_DATA_PACKAGE.md
└── results_reports/                     # Empirical Validation Outputs & Figures
    ├── amph_results_v2_corrected/       # Amphetamine Cohort Corrected V2 (WT vs DAT-Het KI; N=30 sessions)
    │   ├── CHAIN_ANALYSIS_REPORT.md     # Markdown statistical breakdown
    │   ├── inter_chain_organization_summary.png
    │   ├── intra_chain_execution_summary.png
    │   ├── context_dependent_syllable_kinematics.png
    │   └── csv/                         # Corrected quantitative CSV exports
    ├── karen_21kp_results_v2_corrected/ # Parkinson's Cohort Corrected V2 (Monomer vs PFF; N=110 sessions)
    │   ├── CHAIN_ANALYSIS_REPORT.md     # Markdown statistical breakdown
    │   ├── inter_chain_organization_summary.png
    │   ├── intra_chain_execution_summary.png
    │   ├── context_dependent_syllable_kinematics.png
    │   └── csv/                         # Corrected quantitative CSV exports
    ├── amph_results/                    # Original V1 baseline run for audit comparison
    └── karen_21kp_results/              # Original V1 baseline run for audit comparison
```

---

### Audit & Algorithmic Enhancements in Version 2 (V2):
Following an extensive code and biological verification audit, Version 2 implements six critical algorithmic corrections:
1. **Continuation Latency Enforcement:** Strictly enforces non-negative continuation latencies ($\text{gap\_s} \in [0.0, 1.0\text{ s}]$) between completed occurrences, eliminating self-transition artifacts where a chain counted as its own follow-on repetition.
2. **Disambiguation of Simultaneous Occurrences:** Added `ChainFinder.resolve_non_overlapping_episodes()` using temporal non-overlap heuristics, eliminating artificial $100\%$ Markov transitions between candidate chains sharing identical start frames.
3. **Repeat Probability Normalization:** Fixed repeat probability denominators to strictly normalize by total outgoing chain transitions ($\sum_j N(C_i \to C_j)$) rather than total occurrences minus 1.
4. **Conditional Progression Rate:** Re-framed completion rate as *Conditional Progression Rate* ($P(\text{full chain} \mid \text{prefix})$), accurately quantifying the probability that an initiated prefix successfully progresses to the complete multi-stage sequence.
5. **Exact Syllable Kinematic Slicing & Outside Baseline:** Context-dependent kinematics now extract precise frame intervals of target syllables within occurrences (handling palindromes like $A \to B \to A$) and benchmark them against outside-chain baseline bouts via speed and duration modulation ratios.
6. **Correct Biological Ground-Truth Semantics:** Re-aligned motif definitions with ground truth; Motif 10 (`[20, 34, 49]`) is correctly designated as `M10_Pause_Scan` (*Sniff Right $\to$ Look Left $\to$ Stationary Pause*).

---

## 3. The Analytical Pipeline: Step-by-Step

### Stage 1: Data Ingestion & Normalization (`core.CohortDataAdapter`)
The pipeline is strictly cohort-agnostic and ingests behavioral data from either:
- **KPMS HDF5 + CSV:** Reads `results.h5` containing `'syllable'`, `'centroid'`, `'heading'`, computing continuous velocity $\vec{v}(t) = \|\Delta \text{centroid}\| \cdot \text{FPS}$.
- **Session Pickles:** Dataclass objects containing `all_data_df` with aligned kinematics, photometry, and syllable tokens.
- **Data Representation:** Run-Length Encoded (RLE) syllable bouts:
  $$\vec{S} = [s_1, s_1, s_1, s_2, s_2, s_3, \dots] \xrightarrow{\text{RLE}} \text{Tokens: } [s_1, s_2, s_3], \quad \text{Starts: } [t_0, t_3, t_5], \quad \text{Ends: } [t_3, t_5, t_7]$$

### Stage 2: Unsupervised Motif Discovery (`core.auto_discover_chains`)
If canonical motifs are not predefined, the pipeline extracts the empirical distribution of $L$-gram sequences ($L=3$ by default) across the cohort, filtering trivial self-loops and ranking candidates by frequency:
$$\mathcal{M}_k = \arg\max_{c} \sum_{i=1}^{N_\text{sessions}} \text{Count}(c \in \text{Session}_i)$$
This generates standard `ChainDefinition` objects ready for downstream quantification.

### Stage 3: Flexible Multi-Stage Alignment (`core.ChainFinder`)
Exact string matching fails in biology because animals stutter, hesitate, or inject exploratory glances into routines. `ChainFinder` implements a greedy soft-alignment search:
- **Canonical Target:** $[T_0, T_1, \dots, T_{L-1}]$
- **Insertions:** Allows bounded extraneous tokens ($+1$ insertion threshold) if the animal immediately returns to the canonical sequence.
- **Stutters / Repetitions:** Syllable repeats extend the current stage duration without terminating the routine.
- **Early Abort & Hazard Tracking:** If an incompatible token occurs, the execution is marked as aborted at stage $k$, recording the exact exit syllable.
- **Edit Distance:** Computes Levenshtein edit distance:
  $$\text{ED} = N_\text{insertions} + N_\text{repetitions} + (L - \text{Stages Reached})$$

### Stage 4: Inter-Chain Organization Analysis (`inter_chain.InterChainAnalyzer`)
Quantifies organization across recurring routines:
1. **Perseveration Index ($P_\text{persev}$):** Probability of immediate self-transition:
   $$P_\text{persev} = \frac{1}{T-1} \sum_{t=1}^{T-1} \mathbb{I}[C_{t+1} = C_t]$$
2. **Continuation Hazard ($H_k$):** Given that a mouse executed cycle $k$ of a routine within latency gap $\Delta t \le 1.0\text{ s}$, what is the probability of continuing to cycle $k+1$?
   $$H_k = P(\text{Cycle } k+1 \mid \text{Cycle } k) = \frac{N(\text{Bouts with } \ge k+1 \text{ cycles})}{N(\text{Bouts with } \ge k \text{ cycles})}$$
3. **Markov Transition Matrix & Syntax Entropy Rate:**
   $$H(C_{t+1} \mid C_t) = -\sum_{i} P(C_i) \sum_{j} P(C_j \mid C_i) \log_2 P(C_j \mid C_i)$$
4. **Initiations (Lead vs. Repetition Contrast):** Compares kinematics of Cycle 1 (Lead initiation) against Cycles $k \ge 2$ (Follow-on repetitions) within the same sustained bout.

### Stage 5: Intra-Chain Execution Fidelity (`intra_chain.IntraChainAnalyzer`)
Quantifies biomechanical execution quality within each routine:
1. **Dynamic Time Warping (DTW) Consistency:** Velocity profiles $\vec{v}(t)$ of completed executions are resampled to $N=50$ points. The pairwise DTW distance is computed via dynamic programming:
   $$D(n, m) = |v_i[n] - v_j[m]| + \min\{D(n-1, m), D(n, m-1), D(n-1, m-1)\}$$
   Lower DTW distance denotes higher motor stereotypy.
2. **Completion Rate & Stage Hazard:**
   - Overall completion rate: $R_\text{comp} = (N_\text{complete} / N_\text{started}) \times 100\%$
   - Stage-by-stage exit hazard $h(s) = P(\text{Abort at stage } s \mid \text{Reached stage } s)$.
3. **Temporal Pacing Fractions:** Relative fraction of total chain duration allocated to stage $k$:
   $$f_k = \frac{\bar{d}_k}{\sum_{j=1}^L \bar{d}_j}$$
   Reveals whether specific genotypes stretch or compress specific phases of an action.
4. **Context-Dependent Movement:** Compares duration, mean speed, and peak sprint speed ($v_{95}$) of syllable $S$ when executed inside Chain A vs. standalone.

---

## 4. Empirical Ground-Truth Findings in Two Benchmark Cohorts

### Benchmark 1: Amphetamine Cohort (WT vs. DAT-Het KI $\pm$ Amphetamine)
- **Sessions:** 30 ($N=15$ animals) | **Detected Chain Occurrences:** 32,060
- **Motifs Analyzed:**
  - `M1_Rear_Descent` (`[14, 4, 3]`), `M5_Slalom_Sprint` (`[6, 17, 2]`), `M8_Turn_Wall_Turn` (`[11, 1, 30]`), `M10_Groom_Rest_Groom` (`[20, 34, 49]`), `M6_Dash_Redirect` (`[21, 2, 0]`), `M7_Wall_Exploration` (`[16, 14, 7]`).
- **Core Results:**
  1. **Locomotion Routine Completion:** DAT-Het KI mice complete locomotion chains at nearly double the rate of WT mice (`M5`: **16.2%** in KI vs. **9.7%** in WT; `M6`: **11.6%** in KI vs. **6.0%** in WT).
  2. **Grooming Routine Suppression:** WT mice execute **3,145** grooming bouts compared to only **905** in KI mice (3.5-fold suppression in KI), with WT completing 8.3% vs. KI completing only 2.5%.
  3. **Initiation Surge:** In wall-turn routines (`M8`), lead initiations are performed significantly faster ($127.7\text{ px/frame}$) than subsequent repetition cycles ($114.8\text{ px/frame}$, $p = 2.20 \times 10^{-8}$).

### Benchmark 2: Karen 21kp Parkinson's Cohort (Monomer vs. PFF Longitudinal)
- **Sessions:** 110 | **Detected Chain Occurrences:** 76,542
- **Empirical Discovered Motifs:**
  - `K1_Rest_Micro_Rest` (`[29, 1, 29]`), `K2_Rest_Shift_Rest` (`[29, 30, 29]`), `K3_Dash_Reorient` (`[15, 25, 15]`), `K4_Turn_Loop` (`[56, 58, 56]`), `K5_Sprint_Burst` (`[15, 53, 15]`), `K6_Bound_GroundSniff` (`[42, 71, 42]`).
- **Core Results:**
  1. **Parkinsonian Completion Impairment:** In the active dash-reorient routine (`K3`), PFF mice exhibit a **34% reduction in completion rate** (2.86% in PFF vs. 4.33% in Monomer).
  2. **Kinematic Degradation in Exploration:** In compound locomotion sniffing (`K6`), PFF mice display significantly higher DTW execution distance ($18.01$ vs. $17.09$) and prolonged duration ($3.85\text{ s}$ vs. $3.51\text{ s}$), capturing hesitation and bradykinesia during compound sequences.
  3. **Lead Cycle Extension:** For locomotion sniffing bursts, lead cycles are significantly longer in duration ($1.533\text{ s}$) than follow-on repetitions ($1.200\text{ s}$, $p = 2.99 \times 10^{-17}$).

---

## 5. How to Run the Pipeline

### CLI Execution Modes
```bash
# 1. Run on Amphetamine KPMS HDF5 + CSV dataset:
python -m syllable_chains.pipeline \
    --cohort amph \
    --h5-path path/to/results.h5 \
    --meta-csv path/to/metadata.csv \
    --out-dir results_amph

# 2. Run on Karen Parkinson's Session Pickle:
python -m syllable_chains.pipeline \
    --cohort karen \
    --pkl-path path/to/sessions.pkl \
    --syllable-col static_21kp_v2_syllable \
    --out-dir results_karen

# 3. Fully Automated Auto-Discovery on Any New Cohort:
python -m syllable_chains.pipeline \
    --cohort pickle \
    --pkl-path path/to/any_cohort.pkl \
    --auto-discover \
    --auto-discover-n 6 \
    --out-dir results_autodiscover
```

---

## 6. Strategic Questions & Frontiers for AI Research Models

If you are an AI research model analyzing this codebase, consider exploring the following open neuroethological challenges:

1. **Hidden Semi-Markov Modeling (HSMM):** How can we transition from empirical n-gram matching to a generative Hierarchical HSMM where chains are hidden macro-states emitting syllable duration distributions?
2. **Dopamine Transients & Chain Boundaries:** Using the synchronized photometry channels (`signal_0` = dSTR-DA), does dopamine release preferentially align with **Chain Initiations (Lead cycles)** versus **Repetitions (Follow-ons)**?
3. **Probabilistic Context-Free Grammars (PCFGs):** Can we infer a formal grammar of mouse behavior where syllables are terminals, chains are non-terminals, and genotype differences map to production rule probabilities?
4. **Disease Phenotyping Classifiers:** Can we train a linear SVM or XGBoost classifier purely on the inter-chain and intra-chain metrics (DTW distance, continuation hazard, completion rates) to predict PFF vs. Monomer or DAT-KI vs. WT status with high AUC?

---

## 7. Future Capability Roadmap: Intra-Syllable Micro-Movement Dynamics

See full architectural blueprint: **[`docs_and_context/FUTURE_INTRA_SYLLABLE_ANALYSIS_FRAMEWORK.md`](docs_and_context/FUTURE_INTRA_SYLLABLE_ANALYSIS_FRAMEWORK.md)**.

While the current package quantifies higher-order behavioral organization across syllables, the **exact same computational logic can be applied recursively within individual syllables** by treating each syllable as a sequence of video frames or pose micro-movements:

| Chain-Level Dimension (Current Package) | Intra-Syllable Micro-Movement Dimension (Future Roadmap) |
|---|---|
| **Syllables as Tokens:** Discrete states $s_k$ | **Frames as Tokens:** Vector-Quantized pose clusters or continuous keypoints $\mathbf{x}_t \in \mathbb{R}^D$ |
| **Consistency:** Pairwise DTW on 50-pt velocity curves | **Micro-Consistency:** High-dimensional trajectory DTW & Frechet distance across syllable instances |
| **Completion & Aborts:** Early exit before finishing 3 stages | **Micro-Completion:** Frame hazard curves $h_\text{micro}(t)$ — aborting action before peak elevation/foot-strike |
| **Variations:** Syllable insertions, omissions, stutters | **Micro-Variations:** Micro-stutters (oscillatory tremors) vs skipped action sub-phases |
| **Pacing:** Relative duration fractions per chain stage | **Micro-Pacing:** Non-linear temporal warping derivative $w'(t)$ detecting phase-specific bradykinesia |
| **Context-Dependent Movement:** Syllable speed inside vs outside chains | **Micro-Context:** Frame kinematics modulated by preceding syllable velocity |

This opens an unprecedented fractal description of movement kinematics across neurodegenerative and pharmacological cohorts.

