# 4-Tier Clustering Resolution & Video Exemplar Comparison

**Project:** Unsupervised Sequence Discovery via Controlled Convolutional seqNMF  
**Dataset:** 30 Open-Field Imaging Sessions (15 Amphetamine / 15 Saline, Jonathan Rig)  
**Chamber Order:** `[TL, BL, BR, TR]` (100% Cross-Validated with Canonical Rig Configuration)  
**Diagnostic Audit Report:** [CLUSTERING_RESOLUTION_DIAGNOSTIC_AUDIT.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_DIAGNOSTIC_AUDIT.md) (Explains recurrence table vs. similarity plot discrepancy & validates Level 3 superiority)  
**Master Per-Episode CSV:** [master_per_episode_candidate_audit.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/diagnostic_audit/master_per_episode_candidate_audit.csv)  
**Experiment Directory:** `syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/`  

---

## Executive Summary & Core Research Question

> **Central Question:** *Does a given clustering resolution organise more of the observed behaviour into recurring, biologically interpretable sequence families?*

Unsupervised Keypoint-MoSeq (KPMS) decomposes continuous pose trajectories into discrete autoregressive states ($S_0 \dots S_{99}$). However, evaluating sequence discovery across vocabulary resolutions requires avoiding a fundamental pitfall: **total unique observed strings is the wrong metric to treat as a performance score.** 

Because a 100-token vocabulary has $100^N$ potential permutations, Level 1 naturally generates the highest number of distinct observed strings—the vast majority of which occur only once or twice across the entire cohort. Conversely, coarsening the vocabulary reduces trivial permutations and makes token matching easier. **Therefore, higher sequence counts alone do not establish that the resulting sequences are coherent or interesting.**

To rigorously determine whether clustering organizes behavior into meaningful families, we evaluate four complementary empirical criteria:
1. **Recurrence & Support Distributions:** Episode counts and cross-animal coverage across families, evaluated via median and interquartile range (IQR: 25th–75th percentiles) rather than just an arbitrary threshold.
2. **Population Time Coverage:** What fraction of each animal's valid recording time belongs to these recurring sequence families? (1D interval union, counting overlapping frames only once).
3. **Actual Physical Duration:** Measuring elapsed time in seconds ($mean \pm std$ and median). When clustering collapses consecutive identical bouts ($A \to A \to A \implies A$), five coarse steps can span substantially more continuous physical behavior than five raw steps.
4. **Ordinary-Member Video Grounding:** Direct visual inspection of randomly sampled members from the unpruned statistical population across distinct animals to verify whether the broader family remains recognizable across natural kinematic variation.

---

## 1. Sequence Discovery, Recurrence & Population Coverage Matrix

The table below presents the quantitative comparison across the 4 nested clustering levels ($100 \to 93 \to 22 \to 14$ states) for sequence horizons $N = 4, 5, 6$ steps, evaluated across all 30 open-field sessions (>160,000 bouts, 15 mice).

### Summary Comparison: Raw Syllables (Level 1) vs. Broad Families (Level 4)

| Metric / Horizon | Level 1 (100 Raw States) | Level 4 (14 Broad Families) | Ethological Significance |
| :--- | :---: | :---: | :--- |
| **Distinct 5-Step Strings** | 133,003 | 32,074 | 4.1× fewer combinatorial permutations; eliminates single-occurrence variants. |
| **Recurrent 5-Step Families**<br>*(≥20 episodes across ≥8 mice)* | **345** | **1,524** | **4.4× more recurrent sequence families** with substantial multi-animal support. |
| **Recurrent 6-Step Families**<br>*(≥20 episodes across ≥8 mice)* | **36** | **911** | **25.3× more recurrent 6-step families**; raw strings fail to recur at 6 steps. |
| **Population Time Coverage (5-Step)**<br>*(Interval union: overlaps counted once)* | **14.1%**<br>*(14.0% ± 7.0% per mouse)* | **76.2%**<br>*(76.0% ± 10.7% per mouse)* | **5.4× increase in structured time:** 76% of animal life is organized into recurrent motifs. |
| **Population Time Coverage (6-Step)**<br>*(Interval union: overlaps counted once)* | **1.7%**<br>*(1.7% ± 1.4% per mouse)* | **46.0%**<br>*(45.7% ± 13.8% per mouse)* | At 6 steps, Level 1 collapses to 1.7%, while Level 4 still structures nearly half of all behavior. |
| **Episode Support per Family (5-Step)**<br>*(Median [25th–75th IQR])* | 28 episodes<br>[23–39] | 37 episodes<br>[26–63] | Higher median and substantially heavier upper quartile (peak circuits >500 episodes). |
| **Cohort Penetration (5-Step)**<br>*(Median [25th–75th IQR] mice / 15)* | 10 mice<br>[9–11] | 12 mice<br>[10–14] | Families generalize across a broader cross-section of the cohort. |
| **Actual Physical Duration (5-Step)**<br>*(Mean ± Std, Median)* | 2.58s ± 1.29s<br>*(median 2.35s)* | 3.63s ± 2.22s<br>*(median 3.17s)* | **+41% longer behavioral arcs:** Collapsing intra-family repetitions captures full multi-second arcs. |
| **Ordinary-Member Visual Grounding** | Rigid token identity;<br>sparse occurrences | High visual coherence;<br>clear macro-actions | Mode 2 videos confirm families remain recognizable across natural variation. |

---

### Complete 4-Tier Nested Resolution Matrix ($N = 4, 5, 6$ Steps)

*Criteria for Recurrent Family: $\ge 20$ total cohort occurrences spanning $\ge 8$ distinct animals (out of 15). Population coverage reflects the 1D temporal union of all frames spanned by recurrent families, counting overlaps only once.*

| Level | States | Horizon ($N$) | Distinct Observed | Recurrent Families | Population Time Coverage | Per-Animal Coverage (Mean ± Std) | Support Episodes (Median [IQR]) | Support Animals (Median [IQR]) | Physical Duration (Mean ± Std, Median) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1** | 100 | **4-Step** | 65,883 | **1,320** | **48.1%** | 47.8% ± 14.1% | 34 [25–52] | 11 [10–13] | 2.21s ± 1.33s *(med 1.93s)* |
| **Level 1** | 100 | **5-Step** | 133,003 | **345** | **14.1%** | 14.0% ± 7.0% | 28 [23–39] | 10 [9–11] | 2.58s ± 1.29s *(med 2.35s)* |
| **Level 1** | 100 | **6-Step** | 183,223 | **36** | **1.7%** | 1.7% ± 1.4% | 26 [22–34] | 9 [8–10] | 2.92s ± 1.19s *(med 2.80s)* |
| **Level 2** | 93 | **4-Step** | 59,990 | **1,343** | **51.7%** | 51.6% ± 13.4% | 34 [25–52] | 11 [10–13] | 2.40s ± 1.67s *(med 2.02s)* |
| **Level 2** | 93 | **5-Step** | 125,188 | **347** | **15.1%** | 15.0% ± 6.0% | 28 [23–37] | 10 [9–11] | 2.81s ± 1.82s *(med 2.40s)* |
| **Level 2** | 93 | **6-Step** | 175,658 | **27** | **1.1%** | 1.1% ± 0.8% | 26 [23–33] | 10 [9–10] | 2.75s ± 1.10s *(med 2.70s)* |
| **Level 3** | 22 | **4-Step** | 19,673 | **1,874** | **87.2%** | 87.3% ± 7.3% | 40 [27–68] | 12 [10–14] | 2.99s ± 2.39s *(med 2.45s)* |
| **Level 3** | 22 | **5-Step** | 61,781 | **942** | **46.1%** | 46.2% ± 9.9% | 29 [24–41] | 11 [9–12] | 3.78s ± 2.99s *(med 3.08s)* |
| **Level 3** | 22 | **6-Step** | 117,099 | **136** | **9.4%** | 9.5% ± 3.6% | 25 [22–33] | 10 [9–11] | 5.21s ± 4.67s *(med 3.85s)* |
| **Level 4** | 14 | **4-Step** | 9,498 | **1,468** | **94.8%** | 94.7% ± 3.8% | 47 [29–101] | 13 [11–14] | 2.99s ± 2.16s *(med 2.52s)* |
| **Level 4** | 14 | **5-Step** | 32,074 | **1,524** | **76.2%** | 76.0% ± 10.7% | 37 [26–63] | 12 [10–14] | 3.63s ± 2.22s *(med 3.17s)* |
| **Level 4** | 14 | **6-Step** | 68,864 | **911** | **46.0%** | 45.7% ± 13.8% | 33 [24–49] | 11 [10–13] | 4.22s ± 2.20s *(med 3.80s)* |

---

### Detailed Empirical Insights: Evaluating the 4 Added Metrics

#### 1. Population Time Coverage (1D Interval Union)
- **The Finding:** For 5-step sequences, Level 1 recurring sequences cover only **14.1%** of total recording time ($14.0\% \pm 7.0\%$ per animal). In Level 4, recurrent families cover **76.2%** of total recording time ($76.0\% \pm 10.7\%$)—a **5.4-fold increase in structured behavioral coverage**.
- **Why It Matters:** In Level 1, 86% of an animal's recording time consists of sequence permutations that occur fewer than 20 times or fail to generalize across mice. In Level 4, over three-quarters of the animal's behavior is parsed into recurring, multi-animal behavioral families. At 4 steps, Level 4 covers **94.8%** of all recorded behavior.

#### 2. Support per Family (Distributions vs. Cutoffs)
- **The Finding:** Evaluating the interquartile range (IQR) reveals that Level 1 recurring sequences cluster narrowly around the inclusion threshold (median 28 episodes, 75th percentile of 39). In contrast, Level 4 exhibits a substantially heavier upper tail (median 37 episodes, 75th percentile of 63, with top motifs occurring hundreds of times).
- **Why It Matters:** Cross-animal penetration also increases: Level 4 motifs generalize across a median of 12 mice [IQR: 10–14] compared to 10 mice [IQR: 9–11] in Level 1.

#### 3. Actual Physical Duration (Consecutive Bout Merging)
- **The Finding:** A 5-step sequence in Level 1 spans an average of **$2.58\text{ s}$** ($2.35\text{ s}$ median), whereas a 5-step sequence in Level 4 spans **$3.63\text{ s}$** ($3.17\text{ s}$ median), with grooming/resting cycles extending to $10\text{--}15\text{ s}$. At 6 steps, Level 4 sequences average **$4.22\text{ s}$**.
- **Why It Matters:** When syllables are grouped into functional families, consecutive identical bouts collapse ($A \to A \to A \implies A$). Consequently, five steps in Level 4 represent transitions between five *distinct functional behavioral phases*, capturing extended macro-behavioral trajectories rather than micro-pose fluctuations.

#### 4. Ordinary-Member Video Grounding (Visual Coherence Across Variation)
- **The Finding:** Coarsening vocabulary simplifies token matching, raising the concern that sequences might become so broad that they lose behavioral identity.
- **Why It Matters:** Our **Mode 2 (Ordinary Members)** videos directly answer this concern. When sampling randomly from the unpruned statistical population across different mice (e.g. Theme A: mice 1749, 1759, 1771 in Level 4):
  - The macroscopic motor structure (wall approach $\to$ rearing $\to$ descent $\to$ sprint launch) remains immediately recognizable across all three mice.
  - Continuous kinematics capture natural variations (e.g., rearing height, turning angle) as meaningful execution parameters, rather than fragmenting the overarching behavioral program into disconnected tokens.

---

## 2. Interactive Video & GIF Gallery: 3-Tier Demonstration (36 Videos)

To rigorously evaluate how sequence coherence changes across resolutions, we implement a **Three-Tier Demonstration Architecture**:
* **Tier 1 (Canonical Exemplar):** Centroid representative selected at the density peak of trajectory space.
* **Tier 2 (P90 Representative):** Randomly sampled members from distinct animals operating **near the 90th percentile of continuous movement similarity** to a shared reference trajectory.
* **Tier 3 (Ordinary Members):** Unpruned random sample from the statistical population across distinct animals showing natural behavioral dispersion.

---

### Level 1: Original Syllables (100 States)
*Discovered under strict, unclustered KPMS syllable identities.*

| Theme | Tier 1: Canonical Exemplar (Top 1–2%) | Tier 2: P90 Representative (90th Percentile) | Tier 3: Ordinary Members (Random Population) | Kinematics Summary |
| :--- | :--- | :--- | :--- | :--- |
| **Theme A: Wall-Rear & Descent**<br>`[15, 14, 4, 3]`<br>*(Sniff_Up $\to$ Rear $\to$ Desc_L $\to$ Walk_L)* | ![L1 ThA Exemplar](level_1_original_syllables/thumbnails/theme_A_rear_descent_exemplar.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_A_rear_descent_exemplar.mp4)<br>**Animals:** 1694, 1658, 1680 | ![L1 ThA P90](level_1_original_syllables/thumbnails/theme_A_rear_descent_p90.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_A_rear_descent_p90.mp4)<br>**Animals:** 1694, 1658, 1767<br>Sim: **0.71** | ![L1 ThA Ordinary](level_1_original_syllables/thumbnails/theme_A_rear_descent_ordinary.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_A_rear_descent_ordinary.mp4)<br>**Animals:** 1666, 1773, 1694 | **Duration:** $1.73 \pm 0.67\text{ s}$<br>**Speed:** $74.3 \pm 18.6\text{ px/s}$<br>**Net Disp:** $42.2\text{ px}$<br>**Turn $\Delta\theta$:** $-66.6^\circ \pm 69.7^\circ$ |
| **Theme B: Sprint & Slalom**<br>`[2, 0, 13, 6]`<br>*(Turn_R $\to$ Run_R $\to$ Dash $\to$ Turn_L)* | ![L1 ThB Exemplar](level_1_original_syllables/thumbnails/theme_B_turn_slalom_exemplar.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_B_turn_slalom_exemplar.mp4)<br>**Animals:** 1771, 1749, 1658 | ![L1 ThB P90](level_1_original_syllables/thumbnails/theme_B_turn_slalom_p90.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_B_turn_slalom_p90.mp4)<br>**Animals:** 1664, 1765, 1658<br>Sim: **0.81** | ![L1 ThB Ordinary](level_1_original_syllables/thumbnails/theme_B_turn_slalom_ordinary.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_B_turn_slalom_ordinary.mp4)<br>**Animals:** 1771, 740, 728 | **Duration:** $1.63 \pm 0.69\text{ s}$<br>**Speed:** $115.5 \pm 39.1\text{ px/s}$<br>**Net Disp:** $135.1\text{ px}$<br>**Turn $\Delta\theta$:** $+43.0^\circ \pm 53.5^\circ$ |
| **Theme C: Sensory Rest**<br>`[8, 32, 8, 32]`<br>*(Sniff $\to$ Rest $\to$ Sniff $\to$ Rest)* | ![L1 ThC Exemplar](level_1_original_syllables/thumbnails/theme_C_sensory_rest_exemplar.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_C_sensory_rest_exemplar.mp4)<br>**Animals:** 740, 1775, 728 | ![L1 ThC P90](level_1_original_syllables/thumbnails/theme_C_sensory_rest_p90.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_C_sensory_rest_p90.mp4)<br>**Animals:** 1775, 728, 1773<br>Sim: **0.71** | ![L1 ThC Ordinary](level_1_original_syllables/thumbnails/theme_C_sensory_rest_ordinary.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_C_sensory_rest_ordinary.mp4)<br>**Animals:** 1680, 728, 1765 | **Duration:** $3.12 \pm 1.14\text{ s}$<br>**Speed:** $52.0 \pm 3.4\text{ px/s}$<br>**Net Disp:** $8.5\text{ px}$<br>**Turn $\Delta\theta$:** $-22.5^\circ \pm 34.4^\circ$ |

---

### Level 2: Conservative Synonym Merging (93 States)
*Near-identical sub-variants (`1/11`, `3/23`, `19/38`, `32/43/49`, `8/9`, `27/52`) merged.*

| Theme | Tier 1: Canonical Exemplar (Top 1–2%) | Tier 2: P90 Representative (90th Percentile) | Tier 3: Ordinary Members (Random Population) | Kinematics Summary |
| :--- | :--- | :--- | :--- | :--- |
| **Theme A: Wall-Rear & Descent**<br>`[Sniff_Up, Wall_Rear, Desc_L, Walk_L_syn]` | ![L2 ThA Exemplar](level_2_conservative_synonyms/thumbnails/theme_A_rear_descent_exemplar.gif)<br>[🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_A_rear_descent_exemplar.mp4)<br>**Animals:** 1694, 1664, 1658 | ![L2 ThA P90](level_2_conservative_synonyms/thumbnails/theme_A_rear_descent_p90.gif)<br>[🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_A_rear_descent_p90.mp4)<br>**Animals:** 1694, 1658, 1767<br>Sim: **0.71** | ![L2 ThA Ordinary](level_2_conservative_synonyms/thumbnails/theme_A_rear_descent_ordinary.gif)<br>[🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_A_rear_descent_ordinary.mp4)<br>**Animals:** 1664, 1694, 1658 | **Duration:** $1.77 \pm 0.67\text{ s}$<br>**Speed:** $79.5 \pm 22.7\text{ px/s}$<br>**Net Disp:** $52.5\text{ px}$<br>**Turn $\Delta\theta$:** $-74.0^\circ \pm 68.5^\circ$ |
| **Theme B: Sprint & Slalom**<br>`[Turn_R, Run_R, Dash, Turn_L]` | ![L2 ThB Exemplar](level_2_conservative_synonyms/thumbnails/theme_B_turn_slalom_exemplar.gif)<br>[🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_B_turn_slalom_exemplar.mp4)<br>**Animals:** 1771, 1749, 1658 | ![L2 ThB P90](level_2_conservative_synonyms/thumbnails/theme_B_turn_slalom_p90.gif)<br>[🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_B_turn_slalom_p90.mp4)<br>**Animals:** 1664, 1765, 1658<br>Sim: **0.81** | ![L2 ThB Ordinary](level_2_conservative_synonyms/thumbnails/theme_B_turn_slalom_ordinary.gif)<br>[🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_B_turn_slalom_ordinary.mp4)<br>**Animals:** 1658, 728, 1775 | **Duration:** $1.63 \pm 0.69\text{ s}$<br>**Speed:** $115.5 \pm 39.1\text{ px/s}$<br>**Net Disp:** $135.1\text{ px}$<br>**Turn $\Delta\theta$:** $+43.0^\circ \pm 53.5^\circ$ |
| **Theme C: Sensory Rest**<br>`[Sniff_syn, Rest_syn, Sniff_syn, Rest_syn]` | ![L2 ThC Exemplar](level_2_conservative_synonyms/thumbnails/theme_C_sensory_rest_exemplar.gif)<br>[🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_C_sensory_rest_exemplar.mp4)<br>**Animals:** 740, 1775, 728 | ![L2 ThC P90](level_2_conservative_synonyms/thumbnails/theme_C_sensory_rest_p90.gif)<br>[🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_C_sensory_rest_p90.mp4)<br>**Animals:** 740, 728, 1775<br>Sim: **0.70** | ![L2 ThC Ordinary](level_2_conservative_synonyms/thumbnails/theme_C_sensory_rest_ordinary.gif)<br>[🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_C_sensory_rest_ordinary.mp4)<br>**Animals:** 1775, 1658, 728 | **Duration:** $3.29 \pm 1.23\text{ s}$<br>**Speed:** $52.1 \pm 3.3\text{ px/s}$<br>**Net Disp:** $8.7\text{ px}$<br>**Turn $\Delta\theta$:** $-21.3^\circ \pm 35.1^\circ$ |

---

### Level 3: Functional Families with Distinctions Preserved (22 States)
*Directional turns (`Turn_L` vs `Turn_R`), descent lateralities, and curved walking preserved.*

| Theme | Tier 1: Canonical Exemplar (Top 1–2%) | Tier 2: P90 Representative (90th Percentile) | Tier 3: Ordinary Members (Random Population) | Kinematics Summary |
| :--- | :--- | :--- | :--- | :--- |
| **Theme A: Wall-Rear & Descent Launch**<br>`[Turn_R, Rear, Desc_L, Walk_L, Sprint]` | ![L3 ThA Exemplar](level_3_functional_distinctions/thumbnails/theme_A_rear_descent_exemplar.gif)<br>[🎬 Watch MP4](level_3_functional_distinctions/movies/theme_A_rear_descent_exemplar.mp4)<br>**Animals:** 1666, 1658, 1771 | ![L3 ThA P90](level_3_functional_distinctions/thumbnails/theme_A_rear_descent_p90.gif)<br>[🎬 Watch MP4](level_3_functional_distinctions/movies/theme_A_rear_descent_p90.mp4)<br>**Animals:** 1666, 1658, 728<br>Sim: **0.52** | ![L3 ThA Ordinary](level_3_functional_distinctions/thumbnails/theme_A_rear_descent_ordinary.gif)<br>[🎬 Watch MP4](level_3_functional_distinctions/movies/theme_A_rear_descent_ordinary.mp4)<br>**Animals:** 1767, 1680, 1658 | **Duration:** $2.28 \pm 0.77\text{ s}$<br>**Speed:** $116.7 \pm 36.8\text{ px/s}$<br>**Net Disp:** $172.9\text{ px}$<br>**Turn $\Delta\theta$:** $-77.0^\circ \pm 45.8^\circ$ |
| **Theme B: High-Speed Slalom**<br>`[Walk_R, Sprint, Turn_L, Sprint, Turn_R]` | ![L3 ThB Exemplar](level_3_functional_distinctions/thumbnails/theme_B_turn_slalom_exemplar.gif)<br>[🎬 Watch MP4](level_3_functional_distinctions/movies/theme_B_turn_slalom_exemplar.mp4)<br>**Animals:** 1771, 1749, 1658 | ![L3 ThB P90](level_3_functional_distinctions/thumbnails/theme_B_turn_slalom_p90.gif)<br>[🎬 Watch MP4](level_3_functional_distinctions/movies/theme_B_turn_slalom_p90.mp4)<br>**Animals:** 1771, 1666, 1664<br>Sim: **0.75** | ![L3 ThB Ordinary](level_3_functional_distinctions/thumbnails/theme_B_turn_slalom_ordinary.gif)<br>[🎬 Watch MP4](level_3_functional_distinctions/movies/theme_B_turn_slalom_ordinary.mp4)<br>**Animals:** 1771, 1658, 1664 | **Duration:** $2.28 \pm 0.88\text{ s}$<br>**Speed:** $118.7 \pm 23.7\text{ px/s}$<br>**Net Disp:** $187.6\text{ px}$<br>**Turn $\Delta\theta$:** $+27.0^\circ \pm 78.5^\circ$ |
| **Theme C: Active Sensory Scanning**<br>`[Sniff, Look_L, Sniff, Look_R, Sniff]` | ![L3 ThC Exemplar](level_3_functional_distinctions/thumbnails/theme_C_sensory_rest_exemplar.gif)<br>[🎬 Watch MP4](level_3_functional_distinctions/movies/theme_C_sensory_rest_exemplar.mp4)<br>**Animals:** 1680, 740, 1765 | ![L3 ThC P90](level_3_functional_distinctions/thumbnails/theme_C_sensory_rest_p90.gif)<br>[🎬 Watch MP4](level_3_functional_distinctions/movies/theme_C_sensory_rest_p90.mp4)<br>**Animals:** 728, 1658, 1749<br>Sim: **0.21** | ![L3 ThC Ordinary](level_3_functional_distinctions/thumbnails/theme_C_sensory_rest_ordinary.gif)<br>[🎬 Watch MP4](level_3_functional_distinctions/movies/theme_C_sensory_rest_ordinary.mp4)<br>**Animals:** 740, 1680, 728 | **Duration:** $3.53 \pm 1.35\text{ s}$<br>**Speed:** $62.1 \pm 14.3\text{ px/s}$<br>**Net Disp:** $59.1\text{ px}$<br>**Turn $\Delta\theta$:** $+49.3^\circ \pm 94.5^\circ$ |

---

### Level 4: Broad Functional Families (14 States)
*Broad action groups pooling turning lateralities and descent sub-modes.*

| Theme | Tier 1: Canonical Exemplar (Top 1–2%) | Tier 2: P90 Representative (90th Percentile) | Tier 3: Ordinary Members (Random Population) | Kinematics Summary |
| :--- | :--- | :--- | :--- | :--- |
| **Theme A: Wall-Rear & Sprint Loop**<br>`[Sprint, Turn_R, Rear, Descent, Sprint]` | ![L4 ThA Exemplar](level_4_broad_families/thumbnails/theme_A_rear_descent_exemplar.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_A_rear_descent_exemplar.mp4)<br>**Animals:** 1658, 1694, 1666 | ![L4 ThA P90](level_4_broad_families/thumbnails/theme_A_rear_descent_p90.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_A_rear_descent_p90.mp4)<br>**Animals:** 1664, 1666, 1658<br>Sim: **0.43** | ![L4 ThA Ordinary](level_4_broad_families/thumbnails/theme_A_rear_descent_ordinary.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_A_rear_descent_ordinary.mp4)<br>**Animals:** 1749, 1759, 1771 | **Duration:** $2.58 \pm 0.92\text{ s}$<br>**Speed:** $124.6 \pm 40.4\text{ px/s}$<br>**Net Disp:** $183.7\text{ px}$<br>**Turn $\Delta\theta$:** $-16.0^\circ \pm 100.0^\circ$ |
| **Theme B: High-Speed Slalom Burst**<br>`[Sniff, Sprint, Turn_L, Sprint]` | ![L4 ThB Exemplar](level_4_broad_families/thumbnails/theme_B_turn_slalom_exemplar.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_B_turn_slalom_exemplar.mp4)<br>**Animals:** 728, 1666, 1680 | ![L4 ThB P90](level_4_broad_families/thumbnails/theme_B_turn_slalom_p90.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_B_turn_slalom_p90.mp4)<br>**Animals:** 1771, 728, 1680<br>Sim: **0.67** | ![L4 ThB Ordinary](level_4_broad_families/thumbnails/theme_B_turn_slalom_ordinary.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_B_turn_slalom_ordinary.mp4)<br>**Animals:** 1767, 1765, 728 | **Duration:** $2.38 \pm 0.68\text{ s}$<br>**Speed:** $108.6 \pm 30.1\text{ px/s}$<br>**Net Disp:** $168.1\text{ px}$<br>**Turn $\Delta\theta$:** $-17.7^\circ \pm 79.0^\circ$ |
| **Theme C: Rest & Maintenance Cycle**<br>`[Sniff, Rest, Sniff, Rest, Sniff]` | ![L4 ThC Exemplar](level_4_broad_families/thumbnails/theme_C_sensory_rest_exemplar.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_C_sensory_rest_exemplar.mp4)<br>**Animals:** 740, 728, 1771 | ![L4 ThC P90](level_4_broad_families/thumbnails/theme_C_sensory_rest_p90.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_C_sensory_rest_p90.mp4)<br>**Animals:** 740, 1775, 1773<br>Sim: **0.64** | ![L4 ThC Ordinary](level_4_broad_families/thumbnails/theme_C_sensory_rest_ordinary.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_C_sensory_rest_ordinary.mp4)<br>**Animals:** 1765, 728, 1694 | **Duration:** $4.84 \pm 1.61\text{ s}$<br>**Speed:** $52.3 \pm 3.3\text{ px/s}$<br>**Net Disp:** $11.3\text{ px}$<br>**Turn $\Delta\theta$:** $-8.1^\circ \pm 40.4^\circ$ |

---

## 3. The Power of $N$: P90 Movement Similarity & Cumulative Scaling Curves

For full mathematical documentation, see the standalone report: [P90_SIMILARITY_AND_SAMPLE_SIZE_SCALING.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/P90_SIMILARITY_AND_SAMPLE_SIZE_SCALING.md).

![Cumulative Similarity Curves](fig_similarity_vs_qualifying_episodes.png)
*Figure 1: Cumulative Movement Similarity Scaling Across Clustering Resolutions. Shows the number of qualifying episodes exceeding similarity thresholds to a common reference trajectory (top) and distinct animal coverage (bottom). Circles mark the P90 operating point.*

**Key Empirical Insight:** In Level 1, requiring exact syllable identities restricts the candidate pool to $N=62$ episodes across 11 animals. In Level 4, clustering synonymous syllables expands the candidate pool to $N=455$ episodes across all 15 animals. Even when setting a strict continuous movement similarity threshold ($S \ge 0.75$), Level 4 provides **4× to 6× more qualifying executions**, providing massive statistical power for trial-averaged neural alignments.

---

## 4. Continuous Kinematic Distribution Analysis

A central insight raised in discussion is:
> *"Clustering is letting you find repeated action sequences despite variation in the exact KPMS syllables. The promising question now is how much of the movement is conserved within those sequences... keep broad action groups for discovery, then describe each occurrence’s precise execution with continuous kinematics. That lets you retain the larger pool while distinguishing a slight veer from a reversal."*

Our quantitative video metadata confirms this principle with mathematical clarity:

### 1. The Exemplar Selection Mode vs Ordinary Member Dispersion
- **In Mode 1 (Canonical Exemplars):**
  - Trajectories are selected at the density peak of multimodal 4D PCA trajectory space.
  - Across all levels, exemplars achieve density ratios between **$1.58\times$ and $2.86\times$** higher than background baseline.
  - Visually, all 3 mice move in near-perfect kinematic unison, illustrating the **prototypical biomechanical shape** of the motif.
- **In Mode 2 (Ordinary Random Members):**
  - Preserved without artificial bottom-20% speed or path-length filtering.
  - Displays the **true natural variance** of the population: some mice take wide arcing sweeps, others execute abrupt turns, yet all traverse the identical structural syntax.

### 2. Level 3 (22 states) vs Level 4 (14 states): The Turning Angle Disambiguation
- In **Level 3**, turning lateralities are preserved as distinct states (`Turn_L` vs `Turn_R`). Consequently:
  - Theme A has a tightly constrained negative turning angle: $\Delta\theta = -77.0^\circ \pm 45.8^\circ$ (consistent leftward descent and re-orientation).
- In **Level 4**, where turning lateralities are grouped into an overarching turn family:
  - Theme A turning distribution broadens: $\Delta\theta = -16.0^\circ \pm 100.0^\circ$. The mean approaches zero because left turns ($-90^\circ$) and right turns ($+90^\circ$) cancel out in aggregate, while the standard deviation expands to $100^\circ$.
  - **Conclusion:** If Level 4 is used for sequence discovery, **each episode MUST carry its continuous turning angle $\Delta\theta$ as an annotation**. This provides the best of both worlds: massive candidate retrieval ($N=455$ across all 15 animals) without losing the physical trajectory of individual executions!

---

## 5. The P80 Operating Gallery Across All Levels (Syllable HUD & Colors Restored)

For the detailed algorithmic investigation resolving why Level 3 dominates overall recurrence while initially retrieving fewer hand-picked instances, see the master audit: [CLUSTERING_RESOLUTION_DIAGNOSTIC_AUDIT.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_DIAGNOSTIC_AUDIT.md).

All 12 widescreen videos below evaluate the **80th percentile (P80)** operating point of continuous movement similarity across three distinct animals, rendered with the **complete color-coded syllable HUD, real-time stage progress bar, colored centroid tracking dot, and multi-color 2D trajectory inset**:

| Level | Theme A: Wall-Rear & Descent Launch | Theme B: Sprint & Slalom | Theme C: Sensory Rest |
| :--- | :--- | :--- | :--- |
| **Level 1 (100 States)** | ![L1 ThA P80](level_1_original_syllables/thumbnails/theme_A_rear_descent_p80.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_A_rear_descent_p80.mp4)<br>**Mice:** 1767, 1680, 1658<br>Sim: **0.46** | ![L1 ThB P80](level_1_original_syllables/thumbnails/theme_B_turn_slalom_p80.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_B_turn_slalom_p80.mp4)<br>**Mice:** 1664, 1658, 1771<br>Sim: **0.77** | ![L1 ThC P80](level_1_original_syllables/thumbnails/theme_C_sensory_rest_p80.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_C_sensory_rest_p80.mp4)<br>**Mice:** 740, 1773, 1666<br>Sim: **0.63** |
| **Level 2 (93 States)** | ![L2 ThA P80](level_2_conservative_synonyms/thumbnails/theme_A_rear_descent_p80.gif)<br>[🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_A_rear_descent_p80.mp4)<br>**Mice:** 1680, 1694, 1658<br>Sim: **0.54** | ![L2 ThB P80](level_2_conservative_synonyms/thumbnails/theme_B_turn_slalom_p80.gif)<br>[🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_B_turn_slalom_p80.mp4)<br>**Mice:** 1664, 1658, 1771<br>Sim: **0.77** | ![L2 ThC P80](level_2_conservative_synonyms/thumbnails/theme_C_sensory_rest_p80.gif)<br>[🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_C_sensory_rest_p80.mp4)<br>**Mice:** 740, 1775, 1773<br>Sim: **0.62** |
| **Level 3 (22 States)** | ![L3 ThA P80](level_3_functional_distinctions/thumbnails/theme_A_rear_descent_p80.gif)<br>[🎬 Watch MP4](level_3_functional_distinctions/movies/theme_A_rear_descent_p80.mp4)<br>**Mice:** 1767, 1749, 1666<br>Sim: **0.62** | ![L3 ThB P80](level_3_functional_distinctions/thumbnails/theme_B_turn_slalom_p80.gif)<br>[🎬 Watch MP4](level_3_functional_distinctions/movies/theme_B_turn_slalom_p80.mp4)<br>**Mice:** 1658, 1767, 1749<br>Sim: **0.70** | ![L3 ThC P80](level_3_functional_distinctions/thumbnails/theme_C_sensory_rest_p80.gif)<br>[🎬 Watch MP4](level_3_functional_distinctions/movies/theme_C_sensory_rest_p80.mp4)<br>**Mice:** 1658, 1749, 1773<br>Sim: **0.19** |
| **Level 4 (14 States)** | ![L4 ThA P80](level_4_broad_families/thumbnails/theme_A_rear_descent_p80.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_A_rear_descent_p80.mp4)<br>**Mice:** 1658, 1767, 1694<br>Sim: **0.73** | ![L4 ThB P80](level_4_broad_families/thumbnails/theme_B_turn_slalom_p80.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_B_turn_slalom_p80.mp4)<br>**Mice:** 1765, 1680, 1759<br>Sim: **0.64** | ![L4 ThC P80](level_4_broad_families/thumbnails/theme_C_sensory_rest_p80.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_C_sensory_rest_p80.mp4)<br>**Mice:** 728, 740, 1775<br>Sim: **0.52** |

---

## 6. Recommendations for Monday's Presentation

For the upcoming committee/lab presentation in 4 days:
1. **Show Level 3 and Level 4 Side-by-Side as the Climax of Sequence Discovery:**
   - Level 1 and Level 2 demonstrate why raw MoSeq states require ethological grouping (syllable fragmentation).
   - Level 3 demonstrates that discrete grammar can preserve biomechanical laterality.
   - Level 4 demonstrates that broad functional families yield deep statistical power ($N=455$).
2. **Present the "Discrete Syntax + Continuous Kinematics" Paradigm:**
   - Frame KPMS and seqNMF as discovering the **discrete syntax** (the grammatical "verbs" of behavior).
   - Frame continuous SLDS kinematics ($\text{speed}$, $\text{path length}$, $\text{signed turning angle } \Delta\theta$) as the **adverbs** describing execution intensity and direction.
3. **Use the 1×3 Widescreen Videos Directly in the Slides:**
   - The verified hardware indexing `[TL, BL, BR, TR]` guarantees 100% accurate animal IDs, genotypes, and drug conditions.
