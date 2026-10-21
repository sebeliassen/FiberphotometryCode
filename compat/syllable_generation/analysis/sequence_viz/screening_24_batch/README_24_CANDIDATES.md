# 24-Candidate Sequence Screening Catalog

Screening review batch across 8 sequence discovery methods (3 candidates each).
All videos rendered under a **frozen viewing preset**: continuous 40 FPS real-speed, centroid-following crop, colored centroid dot on the mouse, solid black padding, and token HUD.

| ID | Method | Candidate Family | Tokens / Pattern | Total Matches | WT / KI | Saline / Amph | Video File |
|---|---|---|---|---|---|---|---|
| **A1_01** | Method A1 (Exact N-Grams) | Locomotor Alternation Triplet | `[1, 11, 1]` | 1,295 | 426 / 869 | 747 / 548 | [a1_01_locomotor_alternation_triplet.mp4](./a1_01_locomotor_alternation_triplet.mp4) |
| **A1_02** | Method A1 (Exact N-Grams) | Orienting Turn Triplet | `[6, 17, 2]` | 971 | 154 / 817 | 516 / 455 | [a1_02_orienting_turn_triplet.mp4](./a1_02_orienting_turn_triplet.mp4) |
| **A1_03** | Method A1 (Exact N-Grams) | Rearing-Descent Triplet | `[14, 7, 34]` | 2 | 1 / 1 | 1 / 1 | [a1_03_rearing_descent_triplet.mp4](./a1_03_rearing_descent_triplet.mp4) |
| **A2_01** | Method A2 (SEQUITUR Grammar) | Locomotor Subroutine Rule_1 | `Rule_1: [1, 11] + [1, 11]` | 253 | 104 / 149 | 154 / 99 | [a2_01_locomotor_subroutine_rule_1.mp4](./a2_01_locomotor_subroutine_rule_1.mp4) |
| **A2_02** | Method A2 (SEQUITUR Grammar) | Exploration Reorient Rule_2 | `Rule_2: [16, 4] + [0]` | 135 | 19 / 116 | 79 / 56 | [a2_02_exploration_reorient_rule_2.mp4](./a2_02_exploration_reorient_rule_2.mp4) |
| **A2_03** | Method A2 (SEQUITUR Grammar) | Rearing Reorient Rule_3 | `Rule_3: [14, 7] + [0]` | 813 | 193 / 620 | 425 / 388 | [a2_03_rearing_reorient_rule_3.mp4](./a2_03_rearing_reorient_rule_3.mp4) |
| **A3_01** | Method A3 (Cluster-RLE) | Locomotion-Pause-Locomotion | `Cluster [1 -> 4 -> 1]` | 460 | 121 / 339 | 252 / 208 | [a3_01_locomotion_pause_locomotion.mp4](./a3_01_locomotion_pause_locomotion.mp4) |
| **A3_02** | Method A3 (Cluster-RLE) | Exploratory-Orient-Exploratory | `Cluster [0 -> 2 -> 0]` | 10,121 | 2,136 / 7,985 | 5,519 / 4,602 | [a3_02_exploratory_orient_exploratory.mp4](./a3_02_exploratory_orient_exploratory.mp4) |
| **A3_03** | Method A3 (Cluster-RLE) | Rearing-Locomotion-Pause | `Cluster [3 -> 1 -> 4]` | 491 | 120 / 371 | 256 / 235 | [a3_03_rearing_locomotion_pause.mp4](./a3_03_rearing_locomotion_pause.mp4) |
| **B_01** | Method B (Weighted Alignment) | tpl_001_alt3 (Locomotion Family) | `[1, 11, 1] (soft align)` | 3,362 | 1,448 / 1,914 | 1,941 / 1,421 | [b_01_tpl_001_alt3_(locomotion_family).mp4](./b_01_tpl_001_alt3_(locomotion_family).mp4) |
| **B_02** | Method B (Weighted Alignment) | tpl_003_ori3 (Orienting Family) | `[6, 17, 2] (soft align)` | 2,582 | 921 / 1,661 | 1,367 / 1,215 | [b_02_tpl_003_ori3_(orienting_family).mp4](./b_02_tpl_003_ori3_(orienting_family).mp4) |
| **B_03** | Method B (Weighted Alignment) | tpl_005_rear3 (Rearing Family) | `[14, 7, 34] (soft align)` | 3,091 | 1,185 / 1,906 | 1,535 / 1,556 | [b_03_tpl_005_rear3_(rearing_family).mp4](./b_03_tpl_005_rear3_(rearing_family).mp4) |
| **C_01** | Method C (Profile HMM) | Locomotor Elastic Family | `pHMM Family 1: [1, 11, 12]` | 19 | 3 / 16 | 11 / 8 | [c_01_locomotor_elastic_family.mp4](./c_01_locomotor_elastic_family.mp4) |
| **C_02** | Method C (Profile HMM) | Orienting Insertion Family | `pHMM Family 2: [6, 17, 21]` | 306 | 148 / 158 | 186 / 120 | [c_02_orienting_insertion_family.mp4](./c_02_orienting_insertion_family.mp4) |
| **C_03** | Method C (Profile HMM) | Rearing Exploration Family | `pHMM Family 3: [2, 14, 7, 34]` | 2 | 1 / 1 | 1 / 1 | [c_03_rearing_exploration_family.mp4](./c_03_rearing_exploration_family.mp4) |
| **D_01** | Method D (Convolutional seqNMF) | seqnmf_motif_05 (Locomotor Motif) | `Factor 5: [1, 11, 1, 30]` | 304 | 132 / 172 | 176 / 128 | [d_01_seqnmf_motif_05_(locomotor_motif).mp4](./d_01_seqnmf_motif_05_(locomotor_motif).mp4) |
| **D_02** | Method D (Convolutional seqNMF) | seqnmf_motif_02 (Orienting Factor) | `Factor 2: [19, 6, 2]` | 71 | 12 / 59 | 40 / 31 | [d_02_seqnmf_motif_02_(orienting_factor).mp4](./d_02_seqnmf_motif_02_(orienting_factor).mp4) |
| **D_03** | Method D (Convolutional seqNMF) | seqnmf_motif_01 (Exploration Factor) | `Factor 1: [16, 4, 0, 13]` | 37 | 4 / 33 | 19 / 18 | [d_03_seqnmf_motif_01_(exploration_factor).mp4](./d_03_seqnmf_motif_01_(exploration_factor).mp4) |
| **E_01** | Method E (PST Branch) | context_018 (Rearing Exit Branch) | `Context 018: [14, 7] -> 0` | 813 | 193 / 620 | 425 / 388 | [e_01_context_018_(rearing_exit_branch).mp4](./e_01_context_018_(rearing_exit_branch).mp4) |
| **E_02** | Method E (PST Branch) | context_029 (Approach Branch) | `Context 029: [16, 4] -> 17` | 433 | 90 / 343 | 217 / 216 | [e_02_context_029_(approach_branch).mp4](./e_02_context_029_(approach_branch).mp4) |
| **E_03** | Method E (PST Branch) | context_023 (Locomotion Pause Branch) | `Context 023: [1, 0] -> 21` | 383 | 52 / 331 | 188 / 195 | [e_03_context_023_(locomotion_pause_branch).mp4](./e_03_context_023_(locomotion_pause_branch).mp4) |
| **DTW_01** | Method DTW (Kinematic Trajectory) | dtw_alt3 (Kinematic Locomotion) | `DTW Top Matches [1, 11, 1]` | 20 | 4 / 16 | 8 / 12 | [dtw_01_dtw_alt3_(kinematic_locomotion).mp4](./dtw_01_dtw_alt3_(kinematic_locomotion).mp4) |
| **DTW_02** | Method DTW (Kinematic Trajectory) | dtw_ori3 (Kinematic Orienting) | `DTW Top Matches [6, 17, 2]` | 2,582 | 921 / 1,661 | 1,367 / 1,215 | [dtw_02_dtw_ori3_(kinematic_orienting).mp4](./dtw_02_dtw_ori3_(kinematic_orienting).mp4) |
| **DTW_03** | Method DTW (Kinematic Trajectory) | dtw_rear3 (Kinematic Rearing) | `DTW Top Matches [14, 7, 34]` | 3,091 | 1,185 / 1,906 | 1,535 / 1,556 | [dtw_03_dtw_rear3_(kinematic_rearing).mp4](./dtw_03_dtw_rear3_(kinematic_rearing).mp4) |
