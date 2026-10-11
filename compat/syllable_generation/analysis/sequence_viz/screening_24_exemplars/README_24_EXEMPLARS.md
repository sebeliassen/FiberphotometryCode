# 24-Candidate KPMS-Native Density-Ratio Exemplar Catalog

Exemplar selection using **Keypoint-MoSeq's native contrastive density-ratio algorithm** (`keypoint_moseq.util.sample_instances(mode='density')`).

### Mathematical Formulation:
- **Continuous Trajectory PCA**: 10D SLDS latent states resampled to $L=20$ timepoints, projected onto 4D PCA (`pca_dim=4`).
- **Contrastive Ratio**: $\text{Exemplar} = \arg\max_x \left( \frac{\text{Local Density}}{\text{Global Density}} \right)$ identifying instances maximally typical of the sequence family while maximally distinct from background behaviors.
- **Frozen Viewing Preset**: Continuous 40 FPS real-speed, follow centroid crop, colored centroid dot on the mouse, solid black padding, and token HUD (trajectory inset removed).

| ID | Method | Candidate Family | Tokens | Total N | k-NN Pool | Density Ratio | Dispersion Drop | Animals in k | WT / KI | Saline / Amph | Video File |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A1_01** | Method A1 (Exact N-Grams) | Locomotor Alternation Triplet | `[1, 11, 1]` | 1,295 | 50 | 3.53x | 67.8% | 5 | 426 / 869 | 747 / 548 | [a1_01_exemplar_locomotor_alternation_triplet.mp4](./a1_01_exemplar_locomotor_alternation_triplet.mp4) |
| **A1_02** | Method A1 (Exact N-Grams) | Orienting Turn Triplet | `[6, 17, 2]` | 971 | 50 | 3.02x | 74.8% | 5 | 154 / 817 | 516 / 455 | [a1_02_exemplar_orienting_turn_triplet.mp4](./a1_02_exemplar_orienting_turn_triplet.mp4) |
| **A1_03** | Method A1 (Exact N-Grams) | Rearing-Descent Triplet | `[14, 7, 34]` | 2 | 2 | 1.00x | 0.0% | 2 | 1 / 1 | 1 / 1 | [a1_03_exemplar_rearing_descent_triplet.mp4](./a1_03_exemplar_rearing_descent_triplet.mp4) |
| **A2_01** | Method A2 (SEQUITUR Grammar) | Locomotor Subroutine Rule_1 | `Rule_1: [1, 11] + [1, 11]` | 253 | 26 | 3.20x | 64.7% | 2 | 104 / 149 | 154 / 99 | [a2_01_exemplar_locomotor_subroutine_rule_1.mp4](./a2_01_exemplar_locomotor_subroutine_rule_1.mp4) |
| **A2_02** | Method A2 (SEQUITUR Grammar) | Exploration Reorient Rule_2 | `Rule_2: [16, 4] + [0]` | 135 | 14 | 2.38x | 65.0% | 3 | 19 / 116 | 79 / 56 | [a2_02_exemplar_exploration_reorient_rule_2.mp4](./a2_02_exemplar_exploration_reorient_rule_2.mp4) |
| **A2_03** | Method A2 (SEQUITUR Grammar) | Rearing Reorient Rule_3 | `Rule_3: [14, 7] + [0]` | 813 | 50 | 2.14x | 64.9% | 4 | 193 / 620 | 425 / 388 | [a2_03_exemplar_rearing_reorient_rule_3.mp4](./a2_03_exemplar_rearing_reorient_rule_3.mp4) |
| **A3_01** | Method A3 (Cluster-RLE) | Locomotion-Pause-Locomotion | `Cluster [1 -> 4 -> 1]` | 460 | 46 | 1.92x | 70.6% | 5 | 121 / 339 | 252 / 208 | [a3_01_exemplar_locomotion_pause_locomotion.mp4](./a3_01_exemplar_locomotion_pause_locomotion.mp4) |
| **A3_02** | Method A3 (Cluster-RLE) | Exploratory-Orient-Exploratory | `Cluster [0 -> 2 -> 0]` | 10,121 | 50 | 4.44x | 73.3% | 5 | 2,136 / 7,985 | 5,519 / 4,602 | [a3_02_exemplar_exploratory_orient_exploratory.mp4](./a3_02_exemplar_exploratory_orient_exploratory.mp4) |
| **A3_03** | Method A3 (Cluster-RLE) | Rearing-Locomotion-Pause | `Cluster [3 -> 1 -> 4]` | 491 | 50 | 1.90x | 55.3% | 6 | 120 / 371 | 256 / 235 | [a3_03_exemplar_rearing_locomotion_pause.mp4](./a3_03_exemplar_rearing_locomotion_pause.mp4) |
| **B_01** | Method B (Weighted Alignment) | tpl_001_alt3 (Locomotion Family) | `[1, 11, 1] (soft align)` | 3,362 | 50 | 3.52x | 54.5% | 3 | 1,448 / 1,914 | 1,941 / 1,421 | [b_01_exemplar_tpl_001_alt3_(locomotion_family).mp4](./b_01_exemplar_tpl_001_alt3_(locomotion_family).mp4) |
| **B_02** | Method B (Weighted Alignment) | tpl_003_ori3 (Orienting Family) | `[6, 17, 2] (soft align)` | 2,582 | 50 | 3.66x | 85.2% | 3 | 921 / 1,661 | 1,367 / 1,215 | [b_02_exemplar_tpl_003_ori3_(orienting_family).mp4](./b_02_exemplar_tpl_003_ori3_(orienting_family).mp4) |
| **B_03** | Method B (Weighted Alignment) | tpl_005_rear3 (Rearing Family) | `[14, 7, 34] (soft align)` | 3,091 | 50 | 4.45x | 68.6% | 3 | 1,185 / 1,906 | 1,535 / 1,556 | [b_03_exemplar_tpl_005_rear3_(rearing_family).mp4](./b_03_exemplar_tpl_005_rear3_(rearing_family).mp4) |
| **C_01** | Method C (Profile HMM) | Locomotor Elastic Family | `pHMM Family 1: [1, 11, 12]` | 19 | 2 | 4.36x | 75.1% | 1 | 3 / 16 | 11 / 8 | [c_01_exemplar_locomotor_elastic_family.mp4](./c_01_exemplar_locomotor_elastic_family.mp4) |
| **C_02** | Method C (Profile HMM) | Orienting Insertion Family | `pHMM Family 2: [6, 17, 21]` | 306 | 31 | 2.65x | 69.3% | 2 | 148 / 158 | 186 / 120 | [c_02_exemplar_orienting_insertion_family.mp4](./c_02_exemplar_orienting_insertion_family.mp4) |
| **C_03** | Method C (Profile HMM) | Rearing Exploration Family | `pHMM Family 3: [2, 14, 7, 34]` | 2 | 2 | 1.00x | 0.0% | 2 | 1 / 1 | 1 / 1 | [c_03_exemplar_rearing_exploration_family.mp4](./c_03_exemplar_rearing_exploration_family.mp4) |
| **D_01** | Method D (Convolutional seqNMF) | seqnmf_motif_05 (Locomotor Motif) | `Factor 5: [1, 11, 1, 30]` | 304 | 31 | 3.11x | 67.3% | 4 | 132 / 172 | 176 / 128 | [d_01_exemplar_seqnmf_motif_05_(locomotor_motif).mp4](./d_01_exemplar_seqnmf_motif_05_(locomotor_motif).mp4) |
| **D_02** | Method D (Convolutional seqNMF) | seqnmf_motif_02 (Orienting Factor) | `Factor 2: [19, 6, 2]` | 71 | 8 | 2.37x | 64.6% | 3 | 12 / 59 | 40 / 31 | [d_02_exemplar_seqnmf_motif_02_(orienting_factor).mp4](./d_02_exemplar_seqnmf_motif_02_(orienting_factor).mp4) |
| **D_03** | Method D (Convolutional seqNMF) | seqnmf_motif_01 (Exploration Factor) | `Factor 1: [16, 4, 0, 13]` | 37 | 4 | 3.92x | 80.0% | 2 | 4 / 33 | 19 / 18 | [d_03_exemplar_seqnmf_motif_01_(exploration_factor).mp4](./d_03_exemplar_seqnmf_motif_01_(exploration_factor).mp4) |
| **E_01** | Method E (PST Branch) | context_018 (Rearing Exit Branch) | `Context 018: [14, 7] -> 0` | 813 | 50 | 2.14x | 64.9% | 4 | 193 / 620 | 425 / 388 | [e_01_exemplar_context_018_(rearing_exit_branch).mp4](./e_01_exemplar_context_018_(rearing_exit_branch).mp4) |
| **E_02** | Method E (PST Branch) | context_029 (Approach Branch) | `Context 029: [16, 4] -> 17` | 433 | 44 | 2.82x | 64.9% | 3 | 90 / 343 | 217 / 216 | [e_02_exemplar_context_029_(approach_branch).mp4](./e_02_exemplar_context_029_(approach_branch).mp4) |
| **E_03** | Method E (PST Branch) | context_023 (Locomotion Pause Branch) | `Context 023: [1, 0] -> 21` | 383 | 39 | 2.22x | 58.6% | 3 | 52 / 331 | 188 / 195 | [e_03_exemplar_context_023_(locomotion_pause_branch).mp4](./e_03_exemplar_context_023_(locomotion_pause_branch).mp4) |
| **DTW_01** | Method DTW (Kinematic Trajectory) | dtw_alt3 (Kinematic Locomotion) | `DTW Top Matches [1, 11, 1]` | 20 | 2 | 4.52x | 89.4% | 1 | 4 / 16 | 8 / 12 | [dtw_01_exemplar_dtw_alt3_(kinematic_locomotion).mp4](./dtw_01_exemplar_dtw_alt3_(kinematic_locomotion).mp4) |
| **DTW_02** | Method DTW (Kinematic Trajectory) | dtw_ori3 (Kinematic Orienting) | `DTW Top Matches [6, 17, 2]` | 2,582 | 50 | 3.66x | 85.2% | 3 | 921 / 1,661 | 1,367 / 1,215 | [dtw_02_exemplar_dtw_ori3_(kinematic_orienting).mp4](./dtw_02_exemplar_dtw_ori3_(kinematic_orienting).mp4) |
| **DTW_03** | Method DTW (Kinematic Trajectory) | dtw_rear3 (Kinematic Rearing) | `DTW Top Matches [14, 7, 34]` | 3,091 | 50 | 4.45x | 68.6% | 3 | 1,185 / 1,906 | 1,535 / 1,556 | [dtw_03_exemplar_dtw_rear3_(kinematic_rearing).mp4](./dtw_03_exemplar_dtw_rear3_(kinematic_rearing).mp4) |

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

## Metric Guide

- **$k$-NN Pool**: Adaptive neighborhood size $k = \min\left(50, \max\left(6, \lceil 0.10 \times N \rceil\right)\right)$ surrounding the contrastive exemplar.
- **Density Ratio**: Contrastive ratio $\bar{d}_{\text{global}} / \bar{d}_{\text{local}}$ measured in 4D latent PCA space. Ratios $> 2.0\times$ indicate strong clustering tightly separated from background behavior.
- **Dispersion Drop %**: Kinematic variance reduction relative to the full unconstrained sequence pool. A $60\%\text{--}89\%$ reduction means the grid isolates the core motor stereotypy of that sequence.
- **Animals in $k$**: Diversity of distinct animal subjects contributing to the neighborhood pool. The 6 displayed tiles are stratified across distinct mice.

---

## Next Steps: Selecting the 6 Phase 2 Finalists

1. Complete the 3-question scoring across these 24 videos (and compare against the [Uniform Screening Batch](../screening_24_batch/README_24_CANDIDATES.md)).
2. Select **up to 6 candidate families** (e.g., 2 clear locomotion, 2 orienting/turning, 2 rearing/exploratory).
3. For those 6 finalists, apply targeted diagnostic visual probes:
   - **Heading Reset Mode** (`reset_heading_per_syllable=True`) for turning sequences where trajectory departure direction varies.
   - **PST Branch-Centred Alignment** for choice/fork points.
   - **Side-by-Side Genotype/Drug Contrast** (WT vs KI, Saline vs Amphetamine).
