# 24-Candidate Constituent Transition-Exemplar Catalog

Exemplar selection using **Forward Transition-Bucketed Constituent Scoring**.
Each constituent syllable is evaluated locally and causally within its forward transition bucket ($S_k \to S_{k+1}$ or $S_k \to C_{k+1}$), completely eliminating post-hoc whole-chain trajectory filtering.

### Mathematical Formulation:
- **Forward Transition Bucketing**: For every constituent bout $k$, its 10D continuous latents are resampled to $L=15$ timepoints ($150\text{D}$) and projected onto 3D PCA fitted on all bouts in that forward transition bucket $\mathcal{B}(S_k \to S_{k+1})$.
- **Constituent Percentile**: Stage quality $p_k \in [0, 1]$ represents proximity rank to the transition medoid exemplar within that bucket ($1.0 = \text{exemplar}$). Chains are ranked by geometric mean $\bar{p} = (\prod p_k)^{1/M}$.
- **Strict Constituent Chains**: Occurrences where *every single constituent stage* ranks in the top 50% ($p_{\min} \ge 0.50$) of its forward transition bucket.
- **Frozen Viewing Preset**: Continuous 40 FPS real-speed, follow centroid crop, colored centroid dot on the mouse, solid black padding, and token HUD (trajectory inset removed).

| ID | Method | Candidate Family | Tokens | Total Raw N | Strict Chains (≥50%) | Mod Chains (≥33%) | Top-6 Mean Pct | Top-6 Min Stage Pct | Animals | WT / KI | Saline / Amph | Video File |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A1_01** | Method A1 (Exact N-Grams) | Locomotor Alternation Triplet | `[1, 11, 1]` | 1,295 | 228 (17.6%) | 413 (31.9%) | 87.9% | 78.7% | 6 | 426 / 869 | 747 / 548 | [a1_01_trans_exemplar_locomotor_alternation_triplet.mp4](./a1_01_trans_exemplar_locomotor_alternation_triplet.mp4) |
| **A1_02** | Method A1 (Exact N-Grams) | Orienting Turn Triplet | `[6, 17, 2]` | 971 | 165 (17.0%) | 280 (28.8%) | 86.7% | 82.1% | 6 | 154 / 817 | 516 / 455 | [a1_02_trans_exemplar_orienting_turn_triplet.mp4](./a1_02_trans_exemplar_orienting_turn_triplet.mp4) |
| **A1_03** | Method A1 (Exact N-Grams) | Rearing-Descent Triplet | `[14, 7, 34]` | 2 | 0 (0.0%) | 1 (50.0%) | 27.3% | 25.2% | 2 | 1 / 1 | 1 / 1 | [a1_03_trans_exemplar_rearing-descent_triplet.mp4](./a1_03_trans_exemplar_rearing-descent_triplet.mp4) |
| **A2_01** | Method A2 (SEQUITUR Grammar) | Locomotor Subroutine Rule_1 | `Rule_1: [1, 11] + [1, 11]` | 253 | 32 (12.6%) | 60 (23.7%) | 69.5% | 45.8% | 6 | 104 / 149 | 154 / 99 | [a2_01_trans_exemplar_locomotor_subroutine_rule_1.mp4](./a2_01_trans_exemplar_locomotor_subroutine_rule_1.mp4) |
| **A2_02** | Method A2 (SEQUITUR Grammar) | Exploration Reorient Rule_2 | `Rule_2: [16, 4] + [0]` | 135 | 7 (5.2%) | 18 (13.3%) | 64.9% | 51.3% | 6 | 19 / 116 | 79 / 56 | [a2_02_trans_exemplar_exploration_reorient_rule_2.mp4](./a2_02_trans_exemplar_exploration_reorient_rule_2.mp4) |
| **A2_03** | Method A2 (SEQUITUR Grammar) | Rearing Reorient Rule_3 | `Rule_3: [14, 7] + [0]` | 813 | 74 (9.1%) | 126 (15.5%) | 90.0% | 79.5% | 6 | 193 / 620 | 425 / 388 | [a2_03_trans_exemplar_rearing_reorient_rule_3.mp4](./a2_03_trans_exemplar_rearing_reorient_rule_3.mp4) |
| **A3_01** | Method A3 (Cluster-RLE) | Locomotion-Pause-Locomotion | `Cluster [1 -> 4 -> 1]` | 460 | 79 (17.2%) | 145 (31.5%) | 87.1% | 75.3% | 6 | 121 / 339 | 252 / 208 | [a3_01_trans_exemplar_locomotion-pause-locomotion.mp4](./a3_01_trans_exemplar_locomotion-pause-locomotion.mp4) |
| **A3_02** | Method A3 (Cluster-RLE) | Exploratory-Orient-Exploratory | `Cluster [0 -> 2 -> 0]` | 10,121 | 1,636 (16.2%) | 3,129 (30.9%) | 98.0% | 95.6% | 6 | 2136 / 7985 | 5519 / 4602 | [a3_02_trans_exemplar_exploratory-orient-exploratory.mp4](./a3_02_trans_exemplar_exploratory-orient-exploratory.mp4) |
| **A3_03** | Method A3 (Cluster-RLE) | Rearing-Locomotion-Pause | `Cluster [3 -> 1 -> 4]` | 491 | 76 (15.5%) | 143 (29.1%) | 89.5% | 79.3% | 6 | 120 / 371 | 256 / 235 | [a3_03_trans_exemplar_rearing-locomotion-pause.mp4](./a3_03_trans_exemplar_rearing-locomotion-pause.mp4) |
| **B_01** | Method B (Weighted Alignment) | tpl_001_alt3 (Locomotion Family) | `[1, 11, 1] (soft align)` | 3,362 | 454 (13.5%) | 754 (22.4%) | 99.3% | 99.3% | 6 | 1448 / 1914 | 1941 / 1421 | [b_01_trans_exemplar_tpl_001_alt3_(locomotion_family).mp4](./b_01_trans_exemplar_tpl_001_alt3_(locomotion_family).mp4) |
| **B_02** | Method B (Weighted Alignment) | tpl_003_ori3 (Orienting Family) | `[6, 17, 2] (soft align)` | 2,582 | 441 (17.1%) | 692 (26.8%) | 98.0% | 97.5% | 6 | 921 / 1661 | 1367 / 1215 | [b_02_trans_exemplar_tpl_003_ori3_(orienting_family).mp4](./b_02_trans_exemplar_tpl_003_ori3_(orienting_family).mp4) |
| **B_03** | Method B (Weighted Alignment) | tpl_005_rear3 (Rearing Family) | `[14, 7, 34] (soft align)` | 3,091 | 451 (14.6%) | 768 (24.8%) | 98.8% | 98.8% | 6 | 1185 / 1906 | 1535 / 1556 | [b_03_trans_exemplar_tpl_005_rear3_(rearing_family).mp4](./b_03_trans_exemplar_tpl_005_rear3_(rearing_family).mp4) |
| **C_01** | Method C (Profile HMM) | Locomotor Elastic Family | `pHMM Family 1: [1, 11, 12]` | 19 | 2 (10.5%) | 3 (15.8%) | 49.9% | 27.6% | 5 | 3 / 16 | 11 / 8 | [c_01_trans_exemplar_locomotor_elastic_family.mp4](./c_01_trans_exemplar_locomotor_elastic_family.mp4) |
| **C_02** | Method C (Profile HMM) | Orienting Insertion Family | `pHMM Family 2: [6, 17, 21]` | 306 | 58 (19.0%) | 103 (33.7%) | 71.2% | 59.5% | 6 | 148 / 158 | 186 / 120 | [c_02_trans_exemplar_orienting_insertion_family.mp4](./c_02_trans_exemplar_orienting_insertion_family.mp4) |
| **C_03** | Method C (Profile HMM) | Rearing Exploration Family | `pHMM Family 3: [2, 14, 7, 34]` | 2 | 0 (0.0%) | 1 (50.0%) | 34.4% | 25.2% | 2 | 1 / 1 | 1 / 1 | [c_03_trans_exemplar_rearing_exploration_family.mp4](./c_03_trans_exemplar_rearing_exploration_family.mp4) |
| **D_01** | Method D (Convolutional seqNMF) | seqnmf_motif_05 (Locomotor Motif) | `Factor 5: [1, 11, 1, 30]` | 304 | 36 (11.8%) | 94 (30.9%) | 69.2% | 56.2% | 6 | 132 / 172 | 176 / 128 | [d_01_trans_exemplar_seqnmf_motif_05_(locomotor_motif).mp4](./d_01_trans_exemplar_seqnmf_motif_05_(locomotor_motif).mp4) |
| **D_02** | Method D (Convolutional seqNMF) | seqnmf_motif_02 (Orienting Factor) | `Factor 2: [19, 6, 2]` | 71 | 14 (19.7%) | 23 (32.4%) | 72.3% | 58.1% | 6 | 12 / 59 | 40 / 31 | [d_02_trans_exemplar_seqnmf_motif_02_(orienting_factor).mp4](./d_02_trans_exemplar_seqnmf_motif_02_(orienting_factor).mp4) |
| **D_03** | Method D (Convolutional seqNMF) | seqnmf_motif_01 (Exploration Factor) | `Factor 1: [16, 4, 0, 13]` | 37 | 0 (0.0%) | 1 (2.7%) | 51.4% | 18.2% | 5 | 4 / 33 | 19 / 18 | [d_03_trans_exemplar_seqnmf_motif_01_(exploration_factor).mp4](./d_03_trans_exemplar_seqnmf_motif_01_(exploration_factor).mp4) |
| **E_01** | Method E (PST Branch) | context_018 (Rearing Exit Branch) | `Context 018: [14, 7] -> 0` | 813 | 74 (9.1%) | 126 (15.5%) | 90.0% | 79.5% | 6 | 193 / 620 | 425 / 388 | [e_01_trans_exemplar_context_018_(rearing_exit_branch).mp4](./e_01_trans_exemplar_context_018_(rearing_exit_branch).mp4) |
| **E_02** | Method E (PST Branch) | context_029 (Approach Branch) | `Context 029: [16, 4] -> 17` | 433 | 84 (19.4%) | 164 (37.9%) | 86.0% | 74.9% | 6 | 90 / 343 | 217 / 216 | [e_02_trans_exemplar_context_029_(approach_branch).mp4](./e_02_trans_exemplar_context_029_(approach_branch).mp4) |
| **E_03** | Method E (PST Branch) | context_023 (Locomotion Pause Branch) | `Context 023: [1, 0] -> 21` | 383 | 66 (17.2%) | 115 (30.0%) | 83.8% | 73.0% | 6 | 52 / 331 | 188 / 195 | [e_03_trans_exemplar_context_023_(locomotion_pause_branch).mp4](./e_03_trans_exemplar_context_023_(locomotion_pause_branch).mp4) |
| **DTW_01** | Method DTW (Kinematic Trajectory) | dtw_alt3 (Kinematic Locomotion) | `DTW Top Matches [1, 11, 1]` | 20 | 3 (15.0%) | 4 (20.0%) | 58.5% | 39.2% | 6 | 4 / 16 | 8 / 12 | [dtw_01_trans_exemplar_dtw_alt3_(kinematic_locomotion).mp4](./dtw_01_trans_exemplar_dtw_alt3_(kinematic_locomotion).mp4) |
| **DTW_02** | Method DTW (Kinematic Trajectory) | dtw_ori3 (Kinematic Orienting) | `DTW Top Matches [6, 17, 2]` | 2,582 | 441 (17.1%) | 692 (26.8%) | 98.0% | 97.5% | 6 | 921 / 1661 | 1367 / 1215 | [dtw_02_trans_exemplar_dtw_ori3_(kinematic_orienting).mp4](./dtw_02_trans_exemplar_dtw_ori3_(kinematic_orienting).mp4) |
| **DTW_03** | Method DTW (Kinematic Trajectory) | dtw_rear3 (Kinematic Rearing) | `DTW Top Matches [14, 7, 34]` | 3,091 | 451 (14.6%) | 768 (24.8%) | 98.8% | 98.8% | 6 | 1185 / 1906 | 1535 / 1556 | [dtw_03_trans_exemplar_dtw_rear3_(kinematic_rearing).mp4](./dtw_03_trans_exemplar_dtw_rear3_(kinematic_rearing).mp4) |

---

## 3-Question Screening Review Rubric

When reviewing each 6-tile video, evaluate the sequence against this 3-question rubric:

1. **Understandable Progression**:
   - `Clear`: Consistent, recognizable behavioral transition shared across all 6 tiles.
   - `Possible`: Shared motor theme is visible, but timing or subtle variations make boundaries loose.
   - `No`: Tiles appear qualitatively discordant despite sharing token strings.
   - `Corrupt`: Tracking artifacts, occlusion, or boundary chatter obscure interpretation.
2. **What Appears Consistent?**
   - Note invariant motor elements (e.g., body elongation, head yaw direction, gait cadence, wall proximity).
3. **What Makes Interpretation Difficult?**
   - Note confounding factors (e.g., varying departure angles, duration stretching, different wall orientations).

---

## Comparison Across the 3 Paradigms

- **Uniform Screening Batch** ([`../screening_24_batch/README_24_CANDIDATES.md`](../screening_24_batch/README_24_CANDIDATES.md)): Samples evenly across the dataset to measure raw inter-session variability.
- **Whole-Chain Density-Ratio Batch** ([`../screening_24_exemplars/README_24_EXEMPLARS.md`](../screening_24_exemplars/README_24_EXEMPLARS.md)): Evaluates the entire multi-syllable sequence as a single global trajectory via KPMS contrastive density ratio (post-hoc selection on complete chains).
- **Constituent Transition-Exemplar Batch** (This Catalog): Evaluates each constituent syllable locally within its forward transition bucket, selecting chains composed entirely of canonical forward movements without whole-chain post-hoc bias.
