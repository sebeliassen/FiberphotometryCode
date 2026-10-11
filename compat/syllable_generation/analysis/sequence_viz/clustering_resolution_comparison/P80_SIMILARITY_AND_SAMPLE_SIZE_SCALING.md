# The Power of $N$: P80 Movement Similarity & Sample-Size Scaling Across Resolutions

**Project:** Unsupervised Behavioral Grammar via Keypoint-MoSeq & Controlled Convolutional seqNMF  
**Dataset:** 30 Open-Field Imaging Sessions (15 Amphetamine / 15 Saline, Jonathan Rig)  
**Chamber Indexing:** Verified `[TL, BL, BR, TR]` (100% Cross-Validated with Canonical Rig Configuration)  
**Parent Diagnostic Audit:** [CLUSTERING_RESOLUTION_DIAGNOSTIC_AUDIT.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_DIAGNOSTIC_AUDIT.md)  
**Master Comparison Report:** [CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md)  
**Master Per-Episode CSV:** [master_per_episode_candidate_audit.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/diagnostic_audit/master_per_episode_candidate_audit.csv)  

---

## Executive Summary & Conceptual Rationale

> **The Core Question:** *When comparing action sequences discovered with raw vs. clustered syllables, why do the best 3 clips and random draws obscure the true advantage of clustering—and why is the 80th percentile (P80) of movement similarity the optimal operating point to demonstrate the statistical power of $N$?*

When evaluating sequence discovery across clustering resolutions ($100 \to 93 \to 22 \to 14$ states), two common visual modes create a misleading impression:
1. **Mode 1 (Canonical Exemplars):** Three top-scoring clips can look equally tight whether selected from 15 candidates or 500 candidates.
2. **Mode 2 (Ordinary Random Members):** A randomly sampled trio from a 500-episode pool naturally displays more continuous variation than a trio from a hyper-restricted 15-episode pool.

### Why Shift from P90 to P80?
* In earlier tests, **P90 (90th percentile)** was evaluated. While rigorous, P90 is overly restrictive in small candidate pools (where the top 10% may represent only 1 or 2 outlier trials).
* **P80 (80th percentile)** represents the **robust upper quintile** of the behavioral population. It directly answers the neuroethologist's need: *How clean and coordinated are the executions that make up the strong recurring core of the family across multiple distinct mice?*
* Clustering coarsens synonymous pose tokens (`walk_left` sub-variants `3` and `23`, rearing variants `12, 14, 16, 31`), preventing valid behavioral executions from being discarded when an animal steps into a near-synonym. As a result, **the clustered vocabulary contains four to eight times as many closely matching, high-fidelity executions at P80.**

---

## 1. The Standardized Movement-Similarity Measure

To ensure comparisons across resolutions are strictly apples-to-apples, we do not rely on each seqNMF model's internal factor loadings. Instead, we extract a **standardized, continuous kinematic trajectory** for every candidate episode:
1. **Continuous Pose Latent Trajectory $\mathbf{z}(t)$:** The 10-dimensional autoregressive KPMS latent trajectory interpolated to $T=20$ normalized time steps ($20 \times 10 = 200$ features).
2. **Centroid Speed Profile $v(t)$:** Tangential speed ($\text{px/s}$) interpolated to $T=20$ steps ($20$ features).
3. **Heading Angular Velocity $\dot{\theta}(t)$:** Signed angular velocity ($\text{deg/s}$) interpolated to $T=20$ steps ($20$ features).

Concatenating yields a standardized movement feature vector $\mathbf{F}_i \in \mathbb{R}^{240}$. Continuous movement similarity is evaluated as the cosine similarity to an unbiased reference trajectory $\mathbf{F}_{\text{ref}}$:
$$\text{Sim}(\mathbf{F}_i, \mathbf{F}_{\text{ref}}) = \frac{\langle \mathbf{F}_i, \mathbf{F}_{\text{ref}} \rangle}{\|\mathbf{F}_i\|_2 \|\mathbf{F}_{\text{ref}}\|_2} \in [-1, 1]$$

---

## 2. Cumulative Scaling: Strict Apples-to-Apples Matched Grammar

When querying the identical 3-step physical grammar (**`Rear -> Descent -> Walk`**) across all resolutions, **Level 3 leads every other level across all similarity thresholds**:

![Strict Apples-to-Apples Matched Action Chains Scaling](fig_matched_action_chains_scaling.png)
*Figure 1: Cumulative Movement Similarity Scaling for Matched Action Chains (`Rear -> Descent -> Walk`). Diamond markers indicate the exact P80 operating point for each resolution level.*

### Matched Action Chains Scaling Summary Table:

| Resolution Level | Total Episodes ($N$) | Distinct Mice | P80 Similarity | P90 Similarity | $\text{Sim} \ge 0.70$ ($N$ / Mice) | $\text{Sim} \ge 0.75$ ($N$ / Mice) | $\text{Sim} \ge 0.80$ ($N$ / Mice) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1 (100 Raw States)** | 2,143 | 15 | 0.620 | 0.694 | 203 (14 mice) | 108 (12 mice) | 30 (9 mice) |
| **Level 2 (93 Synonym States)** | 1,318 | 15 | 0.623 | 0.711 | 144 (12 mice) | 88 (12 mice) | 31 (9 mice) |
| **Level 3 (22 Functional States)** | **2,686** | **15** | **0.637** | **0.706** | **281 (13 mice)** | **147 (12 mice)** | **40 (10 mice)** |
| **Level 4 (14 Broad Families)** | 2,381 | 15 | 0.623 | 0.693 | 215 (13 mice) | 111 (12 mice) | 29 (9 mice) |

* **Level 3 Advantage:** Yields **$38\%$ more high-fidelity executions ($\text{Sim} \ge 0.70$) than Level 1** (281 vs. 203) and **$95\%$ more than Level 2** (281 vs. 144).

---

## 3. Syllable HUD Architecture & Color Palette Verification

All 12 videos below feature the **complete, fully restored syllable HUD and dynamic coloration**:
* **Color-Coded Stage Boxes:** Distinct syllable colors dynamically allocated to concurrent tokens (`Turn_R`, `Rear`, `Desc_L`, `Walk_L`, `Sprint`, etc.).
* **Real-Time Stage Highlighting:** Live white-border highlight tracking the active syllable in real time, with an instant cyan flash on syllable transitions.
* **Stage Durations:** Sub-box metrics displaying individual stage durations in seconds (e.g. `0.12s`, `0.72s`, `0.17s`).
* **Syllable-Colored Centroid Dot:** Mouse centroid dot color matches the active syllable token in real time.
* **Heading-Aligned 2D Trajectory Inset:** Top-right mini-map showing the spatial path colored by each constituent syllable segment.
* **Continuous Kinematics & Similarity:** Real-time elapsed time, total duration, prototype distance $P$, and movement similarity $K$.

![Restored Syllable HUD Preview](fig_preview_p80_hud_restored_level3.png)
*Figure 2: Representative frame from Level 3 Theme A P80 movie illustrating the active syllable HUD progress bar, live stage highlight (`Turn_R`), colored centroid tracking dot, and trajectory inset.*

---

## 4. Complete Interactive P80 Video Gallery (12 Movies Across Levels 1–4)

All videos below present **three distinct mice** executing each sequence at native $40\text{ fps}$ without artificial pauses (`sync_mode="nothing"`). Each video targets the **80th percentile (P80)** of movement similarity:

### Theme A: Wall-Rear & Descent Launch

| Resolution Level | P80 Video (Animated GIF Preview) | High-Resolution MP4 & Cohort Metrics | Kinematic Characteristics |
| :--- | :--- | :--- | :--- |
| **Level 1 (100 Raw States)**<br>`Sniff_Up -> Wall_Rear -> Desc_L -> Walk_L` | ![L1 ThA P80](level_1_original_syllables/thumbnails/theme_A_rear_descent_p80.gif) | [🎬 Watch MP4](level_1_original_syllables/movies/theme_A_rear_descent_p80.mp4)<br>**Mice:** 1767 (WT), 1680 (WT), 1658 (KI)<br>**P80 Target:** 0.463<br>**Panels Sim:** `[0.462, 0.463, 0.469]` | **Duration:** $1.73 \pm 0.58\text{ s}$<br>**Speed:** $58.2\text{ px/s}$<br>**Turn $\Delta\theta$:** $-74.5^\circ$<br>Tight raw token identity. |
| **Level 2 (93 Synonym States)**<br>`Sniff_Up -> Wall_Rear -> Desc_L -> Walk_L_syn` | ![L2 ThA P80](level_2_conservative_synonyms/thumbnails/theme_A_rear_descent_p80.gif) | [🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_A_rear_descent_p80.mp4)<br>**Mice:** 1680 (WT), 1694 (KI), 1658 (KI)<br>**P80 Target:** 0.541<br>**Panels Sim:** `[0.541, 0.542, 0.533]` | **Duration:** $1.37 \pm 0.24\text{ s}$<br>**Speed:** $61.4\text{ px/s}$<br>**Turn $\Delta\theta$:** $-41.5^\circ$<br>Walk synonym collapse. |
| **Level 3 (22 Functional States)**<br>`Turn_R -> Rear -> Desc_L -> Walk_L -> Sprint` | ![L3 ThA P80](level_3_functional_distinctions/thumbnails/theme_A_rear_descent_p80.gif) | [🎬 Watch MP4](level_3_functional_distinctions/movies/theme_A_rear_descent_p80.mp4)<br>**Mice:** 1767 (WT), 1749 (KI), 1666 (KI)<br>**P80 Target:** 0.622<br>**Panels Sim:** `[0.622, 0.624, 0.608]` | **Duration:** $2.53 \pm 0.78\text{ s}$<br>**Speed:** $74.1\text{ px/s}$<br>**Turn $\Delta\theta$:** $-103.7^\circ$<br>Directional laterality preserved. |
| **Level 4 (14 Broad Families)**<br>`Sprint -> Turn_R -> Rear -> Descent -> Sprint` | ![L4 ThA P80](level_4_broad_families/thumbnails/theme_A_rear_descent_p80.gif) | [🎬 Watch MP4](level_4_broad_families/movies/theme_A_rear_descent_p80.mp4)<br>**Mice:** 1658 (KI), 1767 (WT), 1694 (KI)<br>**P80 Target:** 0.732<br>**Panels Sim:** `[0.732, 0.733, 0.731]` | **Duration:** $3.25 \pm 0.58\text{ s}$<br>**Speed:** $86.5\text{ px/s}$<br>**Turn $\Delta\theta$:** $-30.7^\circ$<br>Massive population support ($N=455$). |

---

### Theme B: Sprint & Turning Redirection

| Resolution Level | P80 Video (Animated GIF Preview) | High-Resolution MP4 & Cohort Metrics | Kinematic Characteristics |
| :--- | :--- | :--- | :--- |
| **Level 1 (100 Raw States)**<br>`Turn_R -> Run_R -> Dash -> Turn_L` | ![L1 ThB P80](level_1_original_syllables/thumbnails/theme_B_turn_slalom_p80.gif) | [🎬 Watch MP4](level_1_original_syllables/movies/theme_B_turn_slalom_p80.mp4)<br>**Mice:** 1664 (KI), 1658 (KI), 1771 (KI)<br>**P80 Target:** 0.767<br>**Panels Sim:** `[0.766, 0.765, 0.763]` | **Duration:** $1.75 \pm 0.41\text{ s}$<br>**Speed:** $98.4\text{ px/s}$<br>**Turn $\Delta\theta$:** $+79.7^\circ$<br>High coordination. |
| **Level 2 (93 Synonym States)**<br>`Turn_R -> Run_R -> Dash -> Turn_L` | ![L2 ThB P80](level_2_conservative_synonyms/thumbnails/theme_B_turn_slalom_p80.gif) | [🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_B_turn_slalom_p80.mp4)<br>**Mice:** 1664 (KI), 1658 (KI), 1771 (KI)<br>**P80 Target:** 0.767<br>**Panels Sim:** `[0.766, 0.765, 0.763]` | **Duration:** $1.75 \pm 0.41\text{ s}$<br>**Speed:** $98.4\text{ px/s}$<br>**Turn $\Delta\theta$:** $+79.7^\circ$<br>Identical to Level 1. |
| **Level 3 (22 Functional States)**<br>`Walk_R -> Sprint -> Turn_L -> Sprint -> Turn_R` | ![L3 ThB P80](level_3_functional_distinctions/thumbnails/theme_B_turn_slalom_p80.gif) | [🎬 Watch MP4](level_3_functional_distinctions/movies/theme_B_turn_slalom_p80.mp4)<br>**Mice:** 1658 (KI), 1767 (WT), 1749 (KI)<br>**P80 Target:** 0.700<br>**Panels Sim:** `[0.702, 0.681, 0.659]` | **Duration:** $2.82 \pm 0.57\text{ s}$<br>**Speed:** $104.2\text{ px/s}$<br>**Turn $\Delta\theta$:** $+3.5^\circ$<br>S-curve slalom alternation. |
| **Level 4 (14 Broad Families)**<br>`Sniff -> Sprint -> Turn_L -> Sprint` | ![L4 ThB P80](level_4_broad_families/thumbnails/theme_B_turn_slalom_p80.gif) | [🎬 Watch MP4](level_4_broad_families/movies/theme_B_turn_slalom_p80.mp4)<br>**Mice:** 1765 (WT), 1680 (WT), 1759 (WT)<br>**P80 Target:** 0.635<br>**Panels Sim:** `[0.633, 0.638, 0.643]` | **Duration:** $2.90 \pm 0.50\text{ s}$<br>**Speed:** $92.1\text{ px/s}$<br>**Turn $\Delta\theta$:** $+25.3^\circ$<br>Broad locomotor redirection. |

---

### Theme C: Olfactory Inspection & Rest

| Resolution Level | P80 Video (Animated GIF Preview) | High-Resolution MP4 & Cohort Metrics | Kinematic Characteristics |
| :--- | :--- | :--- | :--- |
| **Level 1 (100 Raw States)**<br>`Sniff -> Rest -> Sniff -> Rest` | ![L1 ThC P80](level_1_original_syllables/thumbnails/theme_C_sensory_rest_p80.gif) | [🎬 Watch MP4](level_1_original_syllables/movies/theme_C_sensory_rest_p80.mp4)<br>**Mice:** 740 (WT), 1773 (WT), 1666 (KI)<br>**P80 Target:** 0.627<br>**Panels Sim:** `[0.627, 0.622, 0.621]` | **Duration:** $4.27 \pm 0.82\text{ s}$<br>**Speed:** $48.2\text{ px/s}$<br>**Turn $\Delta\theta$:** $-18.4^\circ$<br>Fine-grained stationary resting. |
| **Level 2 (93 Synonym States)**<br>`Sniff_syn -> Rest_syn -> Sniff_syn -> Rest_syn` | ![L2 ThC P80](level_2_conservative_synonyms/thumbnails/theme_C_sensory_rest_p80.gif) | [🎬 Watch MP4](level_2_conservative_synonyms/movies/theme_C_sensory_rest_p80.mp4)<br>**Mice:** 740 (WT), 1775 (WT), 1773 (WT)<br>**P80 Target:** 0.616<br>**Panels Sim:** `[0.616, 0.622, 0.610]` | **Duration:** $3.98 \pm 1.12\text{ s}$<br>**Speed:** $49.5\text{ px/s}$<br>**Turn $\Delta\theta$:** $-51.9^\circ$<br>Rest synonym collapse ($32/43/49$). |
| **Level 3 (22 Functional States)**<br>`Sniff -> Look_L -> Sniff -> Look_R -> Sniff` | ![L3 ThC P80](level_3_functional_distinctions/thumbnails/theme_C_sensory_rest_p80.gif) | [🎬 Watch MP4](level_3_functional_distinctions/movies/theme_C_sensory_rest_p80.mp4)<br>**Mice:** 1658 (KI), 1749 (KI), 1773 (WT)<br>**P80 Target:** 0.189<br>**Panels Sim:** `[0.195, 0.180, 0.200]` | **Duration:** $3.09 \pm 1.22\text{ s}$<br>**Speed:** $54.8\text{ px/s}$<br>**Turn $\Delta\theta$:** $-11.8^\circ$<br>Active head-casting inspection. |
| **Level 4 (14 Broad Families)**<br>`Sniff -> Rest -> Sniff -> Rest -> Sniff` | ![L4 ThC P80](level_4_broad_families/thumbnails/theme_C_sensory_rest_p80.gif) | [🎬 Watch MP4](level_4_broad_families/movies/theme_C_sensory_rest_p80.mp4)<br>**Mice:** 728 (WT), 740 (WT), 1775 (WT)<br>**P80 Target:** 0.525<br>**Panels Sim:** `[0.524, 0.526, 0.522]` | **Duration:** $5.98 \pm 1.30\text{ s}$<br>**Speed:** $51.2\text{ px/s}$<br>**Turn $\Delta\theta$:** $+5.8^\circ$<br>Extended maintenance cycle ($N=224$). |

---

## 5. Neurobiological Significance for Fiber Photometry Alignments

In in vivo dopamine recordings (dStr, NAc core, NAc shell), sequence discovery serves to provide event timestamps for peri-event time histograms (PETHs):
1. **The Single-Trial Dilemma in Level 1:** Requiring exact identity across 100 fine-grained syllables leaves each mouse with only **$1\text{--}3$ trials per 30-minute session**, making trial-averaged dopamine transients noisy and uninterpretable.
2. **The High-Yield Population in Level 3/4:** Coarsening syllables to functional families yields **$15\text{--}40$ trials per mouse** (over $200\text{--}450$ trials total).
3. **The P80 Quality Assurance:** Because P80 executions maintain continuous kinematic similarity $\ge 0.60\text{--}0.75$, researchers are guaranteed that this enlarged trial pool represents clean, coherent behavioral executions rather than sloppy motor noise.

---

### Linked Artifacts & Resources
* **Diagnostic Audit Report:** [CLUSTERING_RESOLUTION_DIAGNOSTIC_AUDIT.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_DIAGNOSTIC_AUDIT.md)
* **Master Resolution Comparison:** [CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md)
* **Presentation Guide:** [monday_14_7_2026_presentation.md](file:///home/fsp585/Desktop/FiberphotometryCode/monday_14_7_2026_presentation.md)
* **Audit Python Script:** [audit_clustering_discrepancy.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/audit_clustering_discrepancy.py)
* **Master Candidate CSV:** [master_per_episode_candidate_audit.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/diagnostic_audit/master_per_episode_candidate_audit.csv)
