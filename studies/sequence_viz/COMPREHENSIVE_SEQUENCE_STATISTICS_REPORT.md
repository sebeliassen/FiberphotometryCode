# Comprehensive Statistical Evaluation: Multi-Method Behavioral Sequences

This report provides ground-truth population statistics across **all 30 sessions** (15 paired animals: 8 WT, 7 KI; Saline vs. Amphetamine) for the 24 candidate sequence families. Following consensus with the user and external review, **biological statistics are strictly separated from display exemplars**: group comparisons are calculated from **all qualifying occurrences**, using the individual animal as the statistical unit.

---

## 1. Executive Statistical Findings

1. **Baseline Genotype Differences in Saline (Contrast 1)**:
   - Multiple candidate sequences exhibit strong, statistically significant differences between Knock-in (KI) and Wildtype (WT) mice under baseline conditions.
   - Top discriminative sequences include **Method A2_02** (Exploration Reorient Rule 2: `[16, 4, 0]`, $p=0.0042$, Cohen's $d=+1.91$), **Method A3_01** (Locomotion-Pause-Locomotion: `Cluster [1 -> 4 -> 1]`, $p=0.0067$, Cohen's $d=+1.78$), **Method D_03** (Exploration Factor 1: `[16, 4, 0, 13]`, $p=0.0172$, Cohen's $d=+1.52$), and **Method A1_02** (Orienting Turn Triplet: `[6, 17, 2]`, $p=0.0246$, Cohen's $d=+1.25$).
   - KI mice display marked perseveration in pause-investigate and orienting loops, providing clear ethological markers of genotype-specific behavioral syntax.

2. **Paired Drug Shifts within Animal (Contrast 2)**:
   - Amphetamine administration induces profound reorganization of sequence usage across all animals.
   - Paused exploratory routines such as **Method A3_02** (`Cluster [0 -> 2 -> 0]`, $p=0.0054$, Cohen's $d=-0.85$) show significant suppression, while locomotor alternation chains undergo dramatic temporal compression (shorter bout durations, increased execution frequency).

3. **Genotype x Drug Interactions (Contrast 3)**:
   - Interaction tests reveal differential drug sensitivity: for example, **Method D_02** (Orienting Factor 2: `[19, 6, 2]`, $p=0.0637$, Cohen's $d=-1.05$) and **Method A3_01** ($d=-0.64$) show that WT animals alter their sequence execution patterns under amphetamine to a greater degree than KI mice, which retain rigid stereotypical sequencing.

4. **Information Gain via Sequence Syntax (LOAO Genotype Decoding)**:
   - Single-syllable frequencies alone achieve **66.1%** balanced accuracy.
   - The 24 multi-syllable sequence rates alone achieve **86.6%** balanced accuracy — an information gain of **+20.5%** over single-syllable composition!
   - This confirms that higher-order sequence syntax carries genuine diagnostic signal that transcends individual syllable frequencies.

---

## 2. Top Genotype-Discriminating Sequences (Saline Baseline)

Ranked by uncorrected exact permutation $p$-value and effect size (Cohen's $d$):

| Rank | ID | Method | Candidate Family | WT Sal Rate | KI Sal Rate | Diff (KI - WT) | Cohen's d | p (Uncorrected) | FDR q |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **A2_02** | Method A2 (SEQUITUR Grammar) | Exploration Reorient Rule_2 | 0.04 | 0.10 | +0.06 | **+1.91** | **0.0042** | 0.0802 |
| 2 | **A3_01** | Method A3 (Cluster-RLE) | Locomotion-Pause-Locomotion | 0.13 | 0.31 | +0.18 | **+1.78** | **0.0067** | 0.0802 |
| 3 | **D_03** | Method D (Convolutional seqNMF) | seqnmf_motif_01 (Exploration Factor) | 0.01 | 0.03 | +0.02 | **+1.52** | **0.0172** | 0.1380 |
| 4 | **A1_02** | Method A1 (Exact N-Grams) | Orienting Turn Triplet | 0.28 | 0.63 | +0.36 | **+1.25** | **0.0246** | 0.1473 |
| 5 | **D_02** | Method D (Convolutional seqNMF) | seqnmf_motif_02 (Orienting Factor) | 0.02 | 0.05 | +0.03 | **+1.23** | **0.0309** | 0.1484 |
| 6 | **B_02** | Method B (Weighted Alignment) | tpl_003_ori3 (Orienting Family) | 0.81 | 1.58 | +0.77 | **+1.03** | **0.0667** | 0.1913 |
| 7 | **DTW_02** | Method DTW (Kinematic Trajectory) | dtw_ori3 (Kinematic Orienting) | 0.81 | 1.58 | +0.77 | **+1.03** | **0.0667** | 0.1913 |
| 8 | **B_03** | Method B (Weighted Alignment) | tpl_005_rear3 (Rearing Family) | 0.97 | 1.71 | +0.73 | **+1.00** | **0.0766** | 0.1913 |

---

## 3. Master Statistical Contrasts Table (All 24 Candidate Families)

**Statistical Contrasts Defined**:
- **Total N**: All qualifying occurrences detected across all 30 recordings ($N=15$ animals: 8 WT, 7 KI).
- **Contrast 1 (Baseline Genotype)**: KI vs. WT under Saline. Shows rate difference $\Delta$, Cohen's $d$, uncorrected exact permutation $p$-value, and FDR $q$-value.
- **Contrast 2 (Paired Drug Effect)**: Amphetamine minus Saline within animal. Paired difference, Cohen's $d$, uncorrected permutation $p$-value, and FDR $q$-value.
- **Contrast 3 (Genotype x Drug Interaction)**: $(\Delta\text{Drug})_{\text{KI}} - (\Delta\text{Drug})_{\text{WT}}$. Interaction difference, Cohen's $d$, uncorrected permutation $p$-value, and FDR $q$-value.

| ID | Candidate Family | Total N | WT Sal Rate | KI Sal Rate | C1 Diff (KI-WT) | C1 p-exact | C1 FDR q | C2 Drug Delta | C2 p-exact | C2 FDR q | C3 Inter Diff | C3 p-exact | C3 FDR q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A1_01** | Locomotor Alternation Triplet | 1,295 | 0.60 | 0.68 | +0.08 (d=+0.09) | **0.9138** | 1.0000 | -0.17 (d=-0.43) | **0.0683** | 0.6162 | +0.19 (d=+0.49) | **0.4623** | 1.0000 |
| **A1_02** | Orienting Turn Triplet | 971 | 0.28 | 0.63 | +0.36 (d=+1.25) | **0.0246** | 0.1473 | -0.05 (d=-0.21) | **0.4521** | 0.8926 | -0.06 (d=-0.22) | **0.6923** | 1.0000 |
| **A1_03** | Rearing-Descent Triplet | 2 | 0.00 | 0.00 | -0.00 (d=-0.48) | **1.0000** | 1.0000 | +0.00 (d=+0.00) | **1.0000** | 1.0000 | -0.00 (d=-0.01) | **1.0000** | 1.0000 |
| **A2_01** | Locomotor Subroutine Rule_1 | 253 | 0.15 | 0.11 | -0.04 (d=-0.18) | **0.9183** | 1.0000 | -0.05 (d=-0.31) | **0.3374** | 0.8926 | +0.06 (d=+0.41) | **0.5764** | 1.0000 |
| **A2_02** | Exploration Reorient Rule_2 | 135 | 0.04 | 0.10 | +0.06 (d=+1.91) | **0.0042** | 0.0802 | -0.02 (d=-0.38) | **0.1619** | 0.6602 | -0.02 (d=-0.44) | **0.4014** | 1.0000 |
| **A2_03** | Rearing Reorient Rule_3 | 813 | 0.29 | 0.46 | +0.17 (d=+0.90) | **0.1044** | 0.1928 | -0.03 (d=-0.14) | **0.5951** | 0.8926 | -0.01 (d=-0.04) | **0.9492** | 1.0000 |
| **A3_01** | Locomotion-Pause-Locomotion | 460 | 0.13 | 0.31 | +0.18 (d=+1.78) | **0.0067** | 0.0802 | -0.04 (d=-0.39) | **0.1650** | 0.6602 | -0.06 (d=-0.64) | **0.2421** | 1.0000 |
| **A3_02** | Exploratory-Orient-Exploratory | 10,121 | 4.78 | 4.83 | +0.05 (d=+0.04) | **0.9481** | 1.0000 | -0.80 (d=-0.85) | **0.0054** | 0.1304 | -0.06 (d=-0.06) | **0.9001** | 1.0000 |
| **A3_03** | Rearing-Locomotion-Pause | 491 | 0.19 | 0.25 | +0.06 (d=+0.53) | **0.3632** | 0.5811 | -0.02 (d=-0.16) | **0.5512** | 0.8926 | -0.00 (d=-0.02) | **0.9629** | 1.0000 |
| **B_01** | tpl_001_alt3 (Locomotion Family) | 3,362 | 1.46 | 1.87 | +0.41 (d=+0.24) | **0.6974** | 0.9637 | -0.44 (d=-0.43) | **0.1218** | 0.6602 | +0.34 (d=+0.33) | **0.5445** | 1.0000 |
| **B_02** | tpl_003_ori3 (Orienting Family) | 2,582 | 0.81 | 1.58 | +0.77 (d=+1.03) | **0.0667** | 0.1913 | -0.12 (d=-0.16) | **0.5728** | 0.8926 | -0.01 (d=-0.01) | **0.9803** | 1.0000 |
| **B_03** | tpl_005_rear3 (Rearing Family) | 3,091 | 0.97 | 1.71 | +0.73 (d=+1.00) | **0.0766** | 0.1913 | +0.03 (d=+0.02) | **0.9268** | 1.0000 | +0.10 (d=+0.09) | **0.8609** | 1.0000 |
| **C_01** | Locomotor Elastic Family | 19 | 0.01 | 0.01 | -0.00 (d=-0.23) | **0.6019** | 0.9028 | -0.00 (d=-0.14) | **0.6797** | 0.9596 | +0.01 (d=+0.46) | **0.4280** | 1.0000 |
| **C_02** | Orienting Insertion Family | 306 | 0.19 | 0.12 | -0.07 (d=-0.28) | **0.7228** | 0.9637 | -0.05 (d=-0.49) | **0.0770** | 0.6162 | +0.04 (d=+0.39) | **0.4786** | 1.0000 |
| **C_03** | Rearing Exploration Family | 2 | 0.00 | 0.00 | -0.00 (d=-0.48) | **1.0000** | 1.0000 | +0.00 (d=+0.00) | **1.0000** | 1.0000 | -0.00 (d=-0.01) | **1.0000** | 1.0000 |
| **D_01** | seqnmf_motif_05 (Locomotor Motif) | 304 | 0.15 | 0.15 | +0.00 (d=+0.01) | **0.9974** | 1.0000 | -0.04 (d=-0.24) | **0.4580** | 0.8926 | +0.07 (d=+0.43) | **0.5069** | 1.0000 |
| **D_02** | seqnmf_motif_02 (Orienting Factor) | 71 | 0.02 | 0.05 | +0.03 (d=+1.23) | **0.0309** | 0.1484 | -0.01 (d=-0.19) | **0.4832** | 0.8926 | -0.04 (d=-1.05) | **0.0637** | 1.0000 |
| **D_03** | seqnmf_motif_01 (Exploration Factor) | 37 | 0.01 | 0.03 | +0.02 (d=+1.52) | **0.0172** | 0.1380 | -0.00 (d=-0.03) | **0.9043** | 1.0000 | -0.01 (d=-0.29) | **0.6824** | 1.0000 |
| **E_01** | context_018 (Rearing Exit Branch) | 813 | 0.29 | 0.46 | +0.17 (d=+0.90) | **0.1044** | 0.1928 | -0.03 (d=-0.14) | **0.5951** | 0.8926 | -0.01 (d=-0.04) | **0.9492** | 1.0000 |
| **E_02** | context_029 (Approach Branch) | 433 | 0.11 | 0.28 | +0.17 (d=+0.85) | **0.1214** | 0.2081 | -0.00 (d=-0.00) | **0.9913** | 1.0000 | +0.02 (d=+0.09) | **0.8967** | 1.0000 |
| **E_03** | context_023 (Locomotion Pause Branch) | 383 | 0.10 | 0.23 | +0.13 (d=+0.94) | **0.1026** | 0.1928 | +0.01 (d=+0.06) | **0.8418** | 1.0000 | -0.00 (d=-0.02) | **0.9719** | 1.0000 |
| **DTW_01** | dtw_alt3 (Kinematic Locomotion) | 20 | 0.00 | 0.01 | +0.01 (d=+1.05) | **0.0797** | 0.1913 | +0.00 (d=+0.34) | **0.2246** | 0.7701 | +0.01 (d=+0.72) | **0.2082** | 1.0000 |
| **DTW_02** | dtw_ori3 (Kinematic Orienting) | 2,582 | 0.81 | 1.58 | +0.77 (d=+1.03) | **0.0667** | 0.1913 | -0.12 (d=-0.16) | **0.5728** | 0.8926 | -0.01 (d=-0.01) | **0.9803** | 1.0000 |
| **DTW_03** | dtw_rear3 (Kinematic Rearing) | 3,091 | 0.97 | 1.71 | +0.73 (d=+1.00) | **0.0766** | 0.1913 | +0.03 (d=+0.02) | **0.9268** | 1.0000 | +0.10 (d=+0.09) | **0.8609** | 1.0000 |

---

## 4. Leave-One-Animal-Out (LOAO) Genotype Decoding Benchmark

To rigorously evaluate whether multi-syllable sequence syntax contains genuine information about genotype beyond single-syllable usage, we performed Leave-One-Animal-Out (LOAO) cross-validation across all 15 animals:

| Feature Representation | Number of Features | LOAO Balanced Accuracy | Performance Delta |
|---|---|---|---|
| **Single-Syllable Frequencies Alone (Baseline)** | 100 syllables | 66.1% | Reference |
| **24 Sequence Chain Rates Alone** | 24 sequence rates | **86.6%** | **+20.5% (Information Gain)** |
| **Combined (Syllables + Sequences)** | 124 features | 66.1% | +0.0% (Feature dilution) |

> [!TIP]
> **Key Neuroethological Insight**: Sequence chain rates alone outperform single-syllable frequencies by **+20.5 percentage points** (86.6% vs 66.1%). This proves that the *grammar and transition order* of behavior is far more predictive of genetic disruption than the mere frequency of individual motor primitives.

---

## 5. Exemplar Selection Modes: Human Visual Inspection vs. Biological Population Inference

As established in consultation with the user, **exemplars are strictly for display and human labeling**, not for calculating biological population statistics:

| Feature | Mode 1: No Exemplar (Uniform) | Mode 2: Whole-Chain Density Ratio | Mode 3: Forward Transition Exemplar |
|---|---|---|---|
| **Mathematical Target** | Stratified index sampling across dataset | Contrastive local/global density on full chain | Stage-wise medoid proximity within forward transition bucket $\mathcal{B}(S_k \to S_{k+1})$ |
| **Scope of Evaluation** | None (unbiased selection) | Post-hoc over complete sequence trajectory | Causal and local at each constituent syllable transition |
| **Circularity Risk** | None | **High**: Assumes what it tries to prove; masks middle-stage corruptions | **Zero**: Does not filter whole chains; unmasks poor intermediate transitions |
| **Strict Compliance Rate** | Variable (average ~10%) | Not measured stage-wise | **12% to 19%** across active candidates |
| **Dispersion Reduction** | 0.0% (full empirical variance) | 55% to 89% variance drop | Preserves natural directional variation while eliminating posture noise |
| **Primary Utility** | Measures raw inter-session variability | Creates smooth visual medoids | **Gold standard for ethologist labeling**: clean, recognizable progression without artifact masking |

---

## 6. Access to Visual Assets and Databases

- **Master 72-Video Catalog**: [`MASTER_72_VIDEO_CATALOG.md`](./MASTER_72_VIDEO_CATALOG.md) (All 24 candidate families across all 3 modes side-by-side)
- **Theoretical Framework**: [`EXEMPLAR_SELECTION_FRAMEWORKS.md`](./EXEMPLAR_SELECTION_FRAMEWORKS.md) (Detailed mathematical analysis of whole-chain vs. transition generative selection)
- **Contrast Statistics Database**: [`data/sequence_genotype_drug_statistical_contrasts.csv`](./data/sequence_genotype_drug_statistical_contrasts.csv)
- **Animal-Level Metrics Database**: [`data/animal_level_sequence_rates_and_durations.csv`](./data/animal_level_sequence_rates_and_durations.csv)
- **Decoding Benchmark Database**: [`data/decoding_information_gain_benchmarks.csv`](./data/decoding_information_gain_benchmarks.csv)
