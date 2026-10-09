# Project B: Higher-Order Syllable Chain Dynamics Framework

**Author/Agent:** Antigravity  
**Project Scope:** Syllable Chains & Sequence Organization (Project B)  
**Code Location:** [`syllable_generation/analysis/syllable_chains/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/syllable_chains/)  
**Validated Cohorts:** Amph Project (WT vs. DAT-Het KI $\pm$ Amph) & Karen 21kp Parkinson's Project (Monomer vs. PFF Longitudinal)

---

## 1. Executive Summary & Architecture

Individual Keypoint-MoSeq (KPMS) syllables describe discrete behavioral primitives on sub-second timescales (typically 200–500 ms). However, natural behavior is organized hierarchically: animals chain primitives into stereotyped action sequences or routines (e.g., rearing $\to$ sniffing $\to$ descending, or sprint $\to$ turn $\to$ decelerate). 

To investigate whether and how mice organize these sequences across genotypes, disease models, and pharmacological perturbations, we developed **Project B: Syllable Chain Dynamics**, a decoupled, cohort-agnostic analysis framework.

```
                           Syllable Stream (Frame-by-Frame)
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
                  KPMS HDF5 / CSV              Session Pickles
               (e.g., Amph Project)       (e.g., Karen 21kp Project)
                         │                           │
                         └─────────────┬─────────────┘
                                       ▼
                             CohortDataAdapter
                                       │
                                       ▼
                       ChainFinder / auto_discover_chains
                      (RLE tokenization, greedy multi-stage
                       flexible matching, edit distance)
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
         InterChainAnalyzer                          IntraChainAnalyzer
  ┌─────────────────────────────┐             ┌─────────────────────────────┐
  │ • Repeats & Perseveration   │             │ • Consistency (DTW Dist & r)│
  │ • Continuation Hazard Rate  │             │ • Completion & Exit Hazard  │
  │ • Markov Transition Entropy │             │ • Variations (Ins/Del/Stut) │
  │ • Lead vs. Repeat Contrast  │             │ • Temporal Pacing Fractions │
  └─────────────────────────────┘             │ • Context-Dependent Speed   │
                 │                            └─────────────────────────────┘
                 └─────────────────────┬─────────────────────┘
                                       ▼
                                ChainVisualizer
                   (Consolidated Multi-Panel Publication Figures)
```

### Module Structure
- [`core.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/syllable_chains/core.py): Data models (`ChainDefinition`, `ChainOccurrence`, `ChainBout`), unified dataset loader (`CohortDataAdapter`), flexible alignment engine (`ChainFinder`), and motif discovery (`auto_discover_chains`).
- [`inter_chain.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/syllable_chains/inter_chain.py): Organization across routines (perseveration index, continuation hazard, Markov transition matrix, transition entropy, lead vs. follow-on kinematics).
- [`intra_chain.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/syllable_chains/intra_chain.py): Execution fidelity within routines (Dynamic Time Warping distance, velocity profile Pearson $r$, completion %, stage hazard, insertions/deletions/stutters, temporal pacing fractions, and context-dependent syllable modulation).
- [`visualizer.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/syllable_chains/visualizer.py): Space-efficient multi-panel publication figures that consolidate hundreds of metrics into high-density plots.
- [`pipeline.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/syllable_chains/pipeline.py): Unified CLI runner supporting `--cohort`, `--auto-discover`, `--gap-threshold`, `--syllable-col`, and `--out-dir`.

---

## 2. Mathematical Formulations & Analytical Metrics

### 2.1 Inter-Chain Organization
1. **Perseveration Index ($P_\text{persev}$):**
   $$P_\text{persev} = \frac{\sum_{t=1}^{T-1} \mathbb{I}[C_{t+1} = C_t]}{T - 1}$$
   Quantifies the probability that the animal immediately re-executes the same behavioral routine upon finishing.
2. **Continuation Hazard ($H_k$):**
   $$H_k = P(\text{Cycle } k+1 \mid \text{Cycle } k) = \frac{N(\text{Bouts with } \ge k+1 \text{ cycles})}{N(\text{Bouts with } \ge k \text{ cycles})}$$
   Distinguishes sustained execution of a routine (continuation within a threshold $\Delta t_\text{gap} \le 1.0\text{ s}$) from genuine new initiations.
3. **Syntax Transition Entropy Rate ($H_\text{syntax}$):**
   $$H_\text{syntax} = -\sum_{i} P(C_i) \sum_{j} P(C_j \mid C_i) \log_2 P(C_j \mid C_i)$$
   Measures the predictability and diversity of higher-order transitions between behavioral routines.
4. **Lead vs. Follow-On Repetition Kinematic Contrast:**
   Compares the kinematics (mean velocity $\bar{v}$, peak sprint velocity $v_{95}$, duration $d$) of Cycle 1 (Lead initiation) against Cycles $k \ge 2$ (Follow-on repetitions) within the same sustained burst:
   $$R_\text{speed} = \frac{\bar{v}_\text{lead}}{\bar{v}_\text{repeat}}, \quad R_\text{dur} = \frac{\bar{d}_\text{lead}}{\bar{d}_\text{repeat}}$$

### 2.2 Intra-Chain Execution Fidelity
1. **Biomechanical Execution Consistency (DTW Distance):**
   Velocity profiles $v_i(t)$ of complete chain executions are resampled to $N=50$ points. The pairwise Dynamic Time Warping distance is computed via dynamic programming:
   $$D(n, m) = |v_i[n] - v_j[m]| + \min\{D(n-1, m), D(n, m-1), D(n-1, m-1)\}$$
   Lower mean pairwise DTW distance indicates higher execution stereotypy.
2. **Completion Rate & Stage Hazard:**
   - **Completion Rate:** $R_\text{comp} = \frac{N_\text{complete}}{N_\text{started}} \times 100\%$
   - **Stage Abort Hazard:** For a chain of length $L$, the conditional probability of terminating after executing stage $s \in \{0, \dots, L-2\}$:
     $$h(s) = \frac{N(\text{Aborted at stage } s)}{N(\text{Reached stage } \ge s)}$$
3. **Structural Variations (Edit Distance):**
   Quantifies insertions (extraneous syllables injected), omissions/deletions (canonical steps skipped), and stutters (immediate repetition of an internal syllable) using Levenshtein alignment against the canonical chain template.
4. **Temporal Pacing Fractions:**
   For a chain with stages $s_1, \dots, s_L$ of mean durations $\bar{d}_1, \dots, \bar{d}_L$:
   $$f_k = \frac{\bar{d}_k}{\sum_{j=1}^L \bar{d}_j}$$
   Reveals whether specific animal groups stretch or contract specific phases of an action (temporal elasticity).
5. **Context-Dependent Movement:**
   Quantifies whether syllable $S$ has identical kinematics when performed in isolation versus embedded in specific recurring routines.

---

## 3. Validation Cohort 1: Amph Project (WT vs. DAT-Het KI)

- **Dataset:** 30 sessions ($N=15$ mice: 8 WT, 7 DAT-Het KI; Saline vs. Amphetamine)
- **Total Chain Executions Detected:** 32,060
- **Target Motifs:**
  - `M1_Rear_Descent`: `[14, 4, 3]` (Vertical Rearing $\to$ Sniffing $\to$ Descending)
  - `M5_Slalom_Sprint`: `[6, 17, 2]` (High-Speed Slalom Sprint)
  - `M8_Turn_Wall_Turn`: `[11, 1, 30]` (Right Turn $\to$ Wall Run $\to$ Right Turn)
  - `M10_Groom_Rest_Groom`: `[20, 34, 49]` (Groom Burst $\to$ Rest $\to$ Face-Wash Groom)
  - `M6_Dash_Redirect`: `[21, 2, 0]` (Dash $\to$ Steer $\to$ Orient)
  - `M7_Wall_Exploration`: `[16, 14, 7]` (Wall Sniff $\to$ Unsupported Rear $\to$ Settle)

### 3.1 Inter-Chain Organization Summary
![Amph Inter-Chain Organization](/home/fsp585/.gemini/antigravity-ide/brain/52486481-bdd4-4a73-9265-ee0c03656687/amph_inter_chain_organization_summary.png)

| Genotype | Perseveration Index | Syntax Entropy (bits) |
|:---|:---:|:---:|
| **WT** | $0.327 \pm 0.090$ | $2.181 \pm 0.230$ |
| **DAT-Het KI** | $0.260 \pm 0.047$ | $2.279 \pm 0.067$ |

**Key Findings:**
- **Perseveration Index:** WT mice show higher overall perseveration ($0.327$ vs $0.260$) due to heavy looping in grooming and resting chains (`M10`). In contrast, DAT-Het KI mice display increased transition entropy ($2.279$ vs $2.181$), reflecting hyperactive switching between exploratory and locomotive routines.
- **Continuation Hazard:**
  - In `M5_Slalom_Sprint`, KI mice continue to cycle 2 in **14.2%** of bouts (vs. 11.4% in WT).
  - In `M10_Groom_Rest_Groom`, WT mice executed **3,145** bouts compared to only **905** bouts in KI mice (a **3.5-fold suppression of grooming routines** in KI mice).
- **Initiations (Lead vs. Repetition Contrast):**
  - In `M8_Turn_Wall_Turn`, lead initiations are performed at **$127.7\text{ px/frame}$** compared to **$114.8\text{ px/frame}$** for follow-on repetitions ($p = 2.20 \times 10^{-8}$). Mice initiate the wall routine with a high-speed surge before settling into steady pace!
  - In `M7_Wall_Exploration`, lead duration is significantly shorter ($0.73\text{ s}$) than follow-on cycles ($0.92\text{ s}$, $p = 3.14 \times 10^{-5}$).

### 3.2 Intra-Chain Execution Fidelity Summary
![Amph Intra-Chain Execution](/home/fsp585/.gemini/antigravity-ide/brain/52486481-bdd4-4a73-9265-ee0c03656687/amph_intra_chain_execution_summary.png)

| Chain ID | Genotype | Attempts | Completed | Completion Rate (%) | DTW Dist | Dur CV |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `M5_Slalom_Sprint` | **KI** | 4,015 | 651 | **16.21%** | 24.51 | 0.536 |
| `M5_Slalom_Sprint` | **WT** | 3,318 | 322 | **9.70%** | 19.36 | 0.575 |
| `M6_Dash_Redirect` | **KI** | 2,149 | 250 | **11.63%** | 23.46 | 0.561 |
| `M6_Dash_Redirect` | **WT** | 2,043 | 122 | **5.97%** | 23.41 | 0.513 |
| `M10_Groom_Rest` | **KI** | 986 | 25 | **2.54%** | 8.76 | 0.543 |
| `M10_Groom_Rest` | **WT** | 3,539 | 293 | **8.28%** | 8.39 | 0.639 |

**Key Findings:**
- **Completion Divergence:** KI mice complete locomotion routines at almost **double the rate** of WT mice (`M5`: 16.2% vs 9.7%; `M6`: 11.6% vs 6.0%). Conversely, WT mice complete grooming routines at more than **triple the rate** of KI mice (8.3% vs 2.5%).
- **Execution Consistency:** Grooming routines (`M10`) exhibit the highest biomechanical consistency (lowest DTW distance: $\approx 8.4$–$8.8$), whereas sprinting routines exhibit higher variability ($\approx 19.4$–$28.8$).

### 3.3 Context-Dependent Movement (Amph)
![Amph Context Dependent Movement](/home/fsp585/.gemini/antigravity-ide/brain/52486481-bdd4-4a73-9265-ee0c03656687/amph_context_dependent_syllable_kinematics.png)

---

## 4. Validation Cohort 2: Karen 21kp Parkinson's Project (Monomer vs. PFF)

- **Dataset:** 110 sessions ($N=99$ longitudinal sessions across 3w, 8w, 16w, 24w, L-dopa waves)
- **Total Chain Executions Detected:** 76,542
- **Empirical Motifs:**
  - `K1_Rest_Micro_Rest`: `[29, 1, 29]` (Rest $\to$ Micro-movement $\to$ Rest)
  - `K2_Rest_Shift_Rest`: `[29, 30, 29]` (Rest $\to$ Posture Shift $\to$ Rest)
  - `K3_Dash_Reorient`: `[15, 25, 15]` (Fast Dash $\to$ Directional Reorient $\to$ Fast Dash)
  - `K4_Turn_Loop`: `[56, 58, 56]` (Steering Turn Left $\leftrightarrow$ Turn Right)
  - `K5_Sprint_Burst`: `[15, 53, 15]` (Sprint Stride $\to$ Gait Acceleration $\to$ Sprint)
  - `K6_Bound_GroundSniff`: `[42, 71, 42]` (Bound $\to$ Low Ground Sniff $\to$ Bound)

### 4.1 Inter-Chain Organization Summary
![Karen Inter-Chain Organization](/home/fsp585/.gemini/antigravity-ide/brain/52486481-bdd4-4a73-9265-ee0c03656687/karen_inter_chain_organization_summary.png)

| Genotype | Sessions | Perseveration Index | Syntax Entropy (bits) |
|:---|:---:|:---:|:---:|
| **Monomer** | 43 | $0.147 \pm 0.087$ | $1.082 \pm 0.121$ |
| **PFF** | 56 | $0.141 \pm 0.089$ | $1.072 \pm 0.160$ |

**Key Findings:**
- **Bout Continuations:** Multi-cycle bouts in Parkinson's mice are heavily concentrated in turning loops (`K4_Turn_Loop`, median duration $\approx 1.7\text{ s}$) and locomotion sniffing (`K6_Bound_GroundSniff`, median duration $\approx 1.2\text{ s}$).
- **Initiations Contrast:** In `K6_Bound_GroundSniff` ($N=1,755$ multi-cycle bouts), lead cycles are significantly longer in duration ($1.533\text{ s}$) than follow-on repetitions ($1.200\text{ s}$, $p = 2.99 \times 10^{-17}$)!

### 4.2 Intra-Chain Execution Fidelity Summary
![Karen Intra-Chain Execution](/home/fsp585/.gemini/antigravity-ide/brain/52486481-bdd4-4a73-9265-ee0c03656687/karen_intra_chain_execution_summary.png)

| Chain ID | Genotype | Attempts | Completed | Completion Rate (%) | DTW Dist | Mean Dur (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `K3_Dash_Reorient` | **Monomer** | 5,677 | 246 | **4.33%** | 5.66 | 6.07 |
| `K3_Dash_Reorient` | **PFF** | 7,857 | 225 | **2.86%** | 5.58 | 5.79 |
| `K6_Bound_Sniff` | **Monomer** | 7,372 | 196 | **2.66%** | **17.09** | 3.51 |
| `K6_Bound_Sniff` | **PFF** | 9,184 | 198 | **2.16%** | **18.01** | 3.85 |
| `K1_Rest_Micro` | **Monomer** | 4,704 | 300 | **6.38%** | 10.13 | 1.30 |
| `K1_Rest_Micro` | **PFF** | 5,730 | 348 | **6.07%** | 10.21 | 1.23 |

**Key Findings:**
- **Parkinsonian Execution Impairment:** In the active dash-reorient routine (`K3_Dash_Reorient`), PFF mice suffer a **34% reduction in completion rate** (2.86% in PFF vs. 4.33% in Monomer).
- **Kinematic Variability in Locomotion Sniffing:** In `K6_Bound_GroundSniff`, PFF mice exhibit elevated DTW distance ($18.01$ vs $17.09$) and prolonged duration ($3.85\text{ s}$ vs $3.51\text{ s}$), reflecting bradykinetic hesitation during compound exploratory bounds.

### 4.3 Context-Dependent Movement (Karen 21kp)
![Karen Context Dependent Movement](/home/fsp585/.gemini/antigravity-ide/brain/52486481-bdd4-4a73-9265-ee0c03656687/karen_context_dependent_syllable_kinematics.png)

---

## 5. How to Run on Any Behavioral Cohort

The framework is completely cohort-agnostic and supports zero-configuration auto-discovery of motifs for any novel dataset:

### Option A: Running on Session Pickles (Generic or Karen)
```bash
/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python -m syllable_generation.analysis.syllable_chains.pipeline \
    --cohort karen \
    --pkl-path path/to/sessions.pkl \
    --syllable-col static_21kp_v2_syllable \
    --out-dir results_my_cohort \
    --cohort-name "MyCohort"
```

### Option B: Running on KPMS HDF5 + CSV (Generic or Amph)
```bash
/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python -m syllable_generation.analysis.syllable_chains.pipeline \
    --cohort amph \
    --h5-path path/to/results.h5 \
    --meta-csv path/to/metadata.csv \
    --out-dir results_my_amph \
    --cohort-name "AmphProject"
```

### Option C: Fully Automated Motif Discovery (`--auto-discover`)
If canonical motifs are not yet known, pass `--auto-discover` to mine the top $N$ empirical sequences directly from data:
```bash
/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python -m syllable_generation.analysis.syllable_chains.pipeline \
    --cohort pickle \
    --pkl-path path/to/any_cohort.pkl \
    --auto-discover \
    --auto-discover-n 6 \
    --out-dir results_autodiscovered
```
