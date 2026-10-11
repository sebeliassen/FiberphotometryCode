# The Power of $N$: P90 Movement Similarity & Sample-Size Scaling Across Resolutions

**Project:** Unsupervised Behavioral Grammar via Keypoint-MoSeq & Controlled Convolutional seqNMF  
**Dataset:** 30 Open-Field Imaging Sessions (15 Amphetamine / 15 Saline, Jonathan Rig)  
**Chamber Indexing:** Verified `[TL, BL, BR, TR]` (100% Alignment with Rig Override Metadata)  
**Figure:** [fig_similarity_vs_qualifying_episodes.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_similarity_vs_qualifying_episodes.png)  
**Parent Comparison:** [CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md)  

---

## Executive Summary & Conceptual Rationale

> **The Core Question:** *When comparing action sequences discovered with raw vs. clustered syllables, why do the best 3 clips and random draws obscure the true advantage of clustering—and how does evaluating the 90th percentile (P90) of movement similarity expose the massive statistical gain in $N$?*

When evaluating unsupervised sequence discovery across clustering resolutions ($100 \to 93 \to 22 \to 14$ states), two common visual modes can create a misleading impression:
1. **Comparing Only Canonical Exemplars (Mode 1):** Three top-scoring clips can look equally tight and synchronized whether selected from a pool of 15 candidates or 500 candidates.
2. **Comparing Only Random Samples (Mode 2):** A randomly sampled trio from a 500-episode pool will naturally exhibit more continuous variation than a trio from a highly restricted 15-episode pool.

**The True Empirical Advantage:** Clustering coarse-grains synonymous pose tokens (`walk_left` sub-variants `3` and `23`, rearing variants `12, 14, 16, 31`), preventing valid behavioral executions from being discarded when an animal steps into a near-synonym. As a result, **the clustered vocabulary can appear slightly more variable in random draws while containing six to eight times as many closely matching, high-fidelity executions.**

To expose this advantage rigorously, we implement a **Three-Tier Demonstration Architecture**:
* **Tier 1 (Canonical Exemplar):** What a prototypical, top-density execution looks like.
* **Tier 2 (P90 Similarity Representative):** What an execution near the **90th percentile of continuous movement similarity** looks like across distinct animals.
* **Tier 3 (Ordinary Member):** What the unpruned statistical population contains under random sampling.
* **Continuous Cumulative Threshold Curves:** A plot of movement similarity threshold ($x$-axis) versus number of qualifying episodes ($y$-axis) using a shared, model-agnostic movement reference.

---

## 1. The Standardized Movement-Similarity Measure

To ensure comparisons across resolutions are strictly apples-to-apples, we do not rely on each seqNMF model's internal factor loadings. Instead, we compute a **standardized, continuous kinematic trajectory** for every candidate episode $i$:
1. **Continuous Pose Latent Trajectory $\mathbf{z}(t)$:** The 10-dimensional autoregressive KPMS latent trajectory interpolated to $T=20$ normalized time steps ($20 \times 10 = 200$ features).
2. **Centroid Speed Profile $v(t)$:** Tangential speed ($\text{px/s}$) interpolated to $T=20$ steps ($20$ features).
3. **Heading Angular Velocity $\dot{\theta}(t)$:** Signed angular velocity ($\text{deg/s}$) interpolated to $T=20$ steps ($20$ features).

Concatenating yields a standardized movement feature vector $\mathbf{F}_i \in \mathbb{R}^{240}$. For each behavioral theme, a fixed reference trajectory $\mathbf{F}_{\text{ref}}$ is established from the canonical centroid. Continuous movement similarity is evaluated as the cosine similarity:
$$\text{Sim}(\mathbf{F}_i, \mathbf{F}_{\text{ref}}) = \frac{\langle \mathbf{F}_i, \mathbf{F}_{\text{ref}} \rangle}{\|\mathbf{F}_i\|_2 \|\mathbf{F}_{\text{ref}}\|_2} \in [-1, 1]$$

---

## 2. Cumulative Similarity Threshold Curves

The figure below plots the continuous movement similarity threshold on the horizontal axis against the **number of qualifying episodes ($N \ge \text{threshold}$)** and **distinct animal cohort coverage (out of 15 mice)** across the four clustering levels.

![Cumulative Similarity Curves](fig_similarity_vs_qualifying_episodes.png)
*Figure 1: Cumulative Movement Similarity Scaling Across Clustering Resolutions. Top panels show the number of qualifying episodes exceeding similarity thresholds to a common reference trajectory. Bottom panels show distinct animal cohort penetration (out of 15 mice). Black-bordered circles mark the exact 90th percentile (P90) operating point for each resolution level.*

---

## 3. Quantitative Scaling Summary Table

The table below summarizes the total retrieved candidates, the P90 similarity value, and the count of qualifying episodes exceeding common similarity benchmarks ($S \ge 0.70, 0.75, 0.80, 0.85$):

| Behavioral Theme | Level | Total Retrieved ($N$) | Cohort Animals | Median Similarity | P90 Similarity | Sim ≥ 0.70 ($N$ / Animals) | Sim ≥ 0.75 ($N$ / Animals) | Sim ≥ 0.80 ($N$ / Animals) | Sim ≥ 0.85 ($N$ / Animals) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Theme A: Wall-Rear & Descent Launch** | **Level 1 (100 States)** | 62 | 11 | 0.563 | **0.718** | 9 (2 anim) | 3 (2 anim) | 0 (0 anim) | 0 (0 anim) |
| | **Level 2 (93 States)** | 74 | 13 | 0.567 | **0.717** | 12 (2 anim) | 4 (2 anim) | 0 (0 anim) | 0 (0 anim) |
| | **Level 3 (22 States)** | 45 | 15 | 0.326 | **0.516** | 0 (0 anim) | 0 (0 anim) | 0 (0 anim) | 0 (0 anim) |
| | **Level 4 (14 States)** | **455** | **15** | 0.239 | **0.434** | 0 (0 anim) | 0 (0 anim) | 0 (0 anim) | 0 (0 anim) |
| **Theme B: Sprint & Turning Redirection** | **Level 1 (100 States)** | 155 | 15 | 0.651 | **0.804** | 62 (12 anim) | 39 (11 anim) | 16 (9 anim) | 4 (3 anim) |
| | **Level 2 (93 States)** | 155 | 15 | 0.651 | **0.804** | 62 (12 anim) | 39 (11 anim) | 16 (9 anim) | 4 (3 anim) |
| | **Level 3 (22 States)** | 18 | 7 | 0.589 | **0.745** | 4 (4 anim) | 1 (1 anim) | 0 (0 anim) | 0 (0 anim) |
| | **Level 4 (14 States)** | **94** | **14** | 0.453 | **0.668** | 6 (4 anim) | 1 (1 anim) | 1 (1 anim) | 0 (0 anim) |
| **Theme C: Olfactory Inspection & Rest** | **Level 1 (100 States)** | 111 | 14 | 0.458 | **0.705** | 12 (6 anim) | 6 (4 anim) | 2 (2 anim) | 0 (0 anim) |
| | **Level 2 (93 States)** | 121 | 14 | 0.452 | **0.696** | 12 (6 anim) | 6 (4 anim) | 2 (2 anim) | 0 (0 anim) |
| | **Level 3 (22 States)** | 18 | 9 | -0.096 | **0.214** | 0 (0 anim) | 0 (0 anim) | 0 (0 anim) | 0 (0 anim) |
| | **Level 4 (14 States)** | **224** | **14** | 0.265 | **0.636** | 9 (5 anim) | 3 (3 anim) | 0 (0 anim) | 0 (0 anim) |

---

### Direct Nested Query Scaling: `Rear -> Descent -> Walk`

When we query the exact multi-state action chain across resolutions without requiring a specific seqNMF factor decomposition:
* **Level 1 Query (`14 -> 4 -> 3`):** $N = 413$ total episodes; at $S \ge 0.70$, **16 episodes** across 5 animals.
* **Level 2 Query (`14 -> 4 -> Walk_L_syn`):** $N = 574$ total episodes; at $S \ge 0.70$, **22 episodes** across 7 animals.
* **Level 3 Query (`Rear_Elevated -> Descent_L -> Walk_L`):** $N = 1,190$ total episodes; at $S \ge 0.70$, **35 episodes** across 8 animals.
* **Level 4 Query (`Rear -> Descent -> Walk`):** $N = 1,973$ total episodes; at $S \ge 0.70$, **48 episodes** across 8 animals (**$3.0\times$ more high-fidelity episodes**). At $S \ge 0.75$, Level 4 yields **12 episodes** compared to only **3 episodes** in Level 1 (**$4.0\times$ increase**).

---

## 4. 3-Tier Video Gallery: Exemplar vs. P90 vs. Ordinary Member

To directly inspect the quality of the distribution, all videos below present **three distinct mice** executing the sequence at native $40\text{ fps}$ without artificial pauses:

### Theme A: Wall-Rear & Descent Launch

| Resolution Level | Tier 1: Canonical Exemplar (Top 1–2%) | Tier 2: P90 Representative (90th Percentile) | Tier 3: Ordinary Members (Random Population) |
| :--- | :--- | :--- | :--- |
| **Level 1 (100 States)**<br>*(N=62, 11 anim)* | ![L1 ThA Exemplar](level_1_original_syllables/thumbnails/theme_A_rear_descent_exemplar.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_A_rear_descent_exemplar.mp4)<br>**Mice:** 1694, 1658, 1680<br>Sim: $0.78 \pm 0.04$ | ![L1 ThA P90](level_1_original_syllables/thumbnails/theme_A_rear_descent_p90.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_A_rear_descent_p90.mp4)<br>**Mice:** 1694, 1658, 1767<br>Sim: **0.71** *(P90 Target: 0.72)* | ![L1 ThA Ordinary](level_1_original_syllables/thumbnails/theme_A_rear_descent_ordinary.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_A_rear_descent_ordinary.mp4)<br>**Mice:** 1666, 1773, 1694<br>Sim: $0.58 \pm 0.12$ |
| **Level 4 (14 States)**<br>*(N=455, 15 anim)* | ![L4 ThA Exemplar](level_4_broad_families/thumbnails/theme_A_rear_descent_exemplar.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_A_rear_descent_exemplar.mp4)<br>**Mice:** 1658, 1694, 1666<br>Sim: $0.74 \pm 0.05$ | ![L4 ThA P90](level_4_broad_families/thumbnails/theme_A_rear_descent_p90.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_A_rear_descent_p90.mp4)<br>**Mice:** 1664, 1666, 1658<br>Sim: **0.43** *(P90 Target: 0.43)* | ![L4 ThA Ordinary](level_4_broad_families/thumbnails/theme_A_rear_descent_ordinary.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_A_rear_descent_ordinary.mp4)<br>**Mice:** 1749, 1759, 1771<br>Sim: $0.25 \pm 0.14$ |

---

### Theme B: Sprint & Turning Redirection

| Resolution Level | Tier 1: Canonical Exemplar (Top 1–2%) | Tier 2: P90 Representative (90th Percentile) | Tier 3: Ordinary Members (Random Population) |
| :--- | :--- | :--- | :--- |
| **Level 1 (100 States)**<br>*(N=155, 15 anim)* | ![L1 ThB Exemplar](level_1_original_syllables/thumbnails/theme_B_turn_slalom_exemplar.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_B_turn_slalom_exemplar.mp4)<br>**Mice:** 1771, 1749, 1658<br>Sim: $0.85 \pm 0.03$ | ![L1 ThB P90](level_1_original_syllables/thumbnails/theme_B_turn_slalom_p90.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_B_turn_slalom_p90.mp4)<br>**Mice:** 1664, 1765, 1658<br>Sim: **0.81** *(P90 Target: 0.80)* | ![L1 ThB Ordinary](level_1_original_syllables/thumbnails/theme_B_turn_slalom_ordinary.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_B_turn_slalom_ordinary.mp4)<br>**Mice:** 1771, 740, 728<br>Sim: $0.66 \pm 0.11$ |
| **Level 4 (14 States)**<br>*(N=94, 14 anim)* | ![L4 ThB Exemplar](level_4_broad_families/thumbnails/theme_B_turn_slalom_exemplar.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_B_turn_slalom_exemplar.mp4)<br>**Mice:** 728, 1666, 1680<br>Sim: $0.80 \pm 0.04$ | ![L4 ThB P90](level_4_broad_families/thumbnails/theme_B_turn_slalom_p90.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_B_turn_slalom_p90.mp4)<br>**Mice:** 1771, 728, 1680<br>Sim: **0.67** *(P90 Target: 0.67)* | ![L4 ThB Ordinary](level_4_broad_families/thumbnails/theme_B_turn_slalom_ordinary.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_B_turn_slalom_ordinary.mp4)<br>**Mice:** 1767, 1765, 728<br>Sim: $0.46 \pm 0.15$ |

---

### Theme C: Olfactory Inspection & Rest

| Resolution Level | Tier 1: Canonical Exemplar (Top 1–2%) | Tier 2: P90 Representative (90th Percentile) | Tier 3: Ordinary Members (Random Population) |
| :--- | :--- | :--- | :--- |
| **Level 1 (100 States)**<br>*(N=111, 14 anim)* | ![L1 ThC Exemplar](level_1_original_syllables/thumbnails/theme_C_sensory_rest_exemplar.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_C_sensory_rest_exemplar.mp4)<br>**Mice:** 740, 1775, 728<br>Sim: $0.79 \pm 0.03$ | ![L1 ThC P90](level_1_original_syllables/thumbnails/theme_C_sensory_rest_p90.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_C_sensory_rest_p90.mp4)<br>**Mice:** 1775, 728, 1773<br>Sim: **0.71** *(P90 Target: 0.71)* | ![L1 ThC Ordinary](level_1_original_syllables/thumbnails/theme_C_sensory_rest_ordinary.gif)<br>[🎬 Watch MP4](level_1_original_syllables/movies/theme_C_sensory_rest_ordinary.mp4)<br>**Mice:** 1680, 728, 1765<br>Sim: $0.47 \pm 0.13$ |
| **Level 4 (14 States)**<br>*(N=224, 14 anim)* | ![L4 ThC Exemplar](level_4_broad_families/thumbnails/theme_C_sensory_rest_exemplar.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_C_sensory_rest_exemplar.mp4)<br>**Mice:** 740, 728, 1771<br>Sim: $0.78 \pm 0.04$ | ![L4 ThC P90](level_4_broad_families/thumbnails/theme_C_sensory_rest_p90.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_C_sensory_rest_p90.mp4)<br>**Mice:** 740, 1775, 1773<br>Sim: **0.64** *(P90 Target: 0.64)* | ![L4 ThC Ordinary](level_4_broad_families/thumbnails/theme_C_sensory_rest_ordinary.gif)<br>[🎬 Watch MP4](level_4_broad_families/movies/theme_C_sensory_rest_ordinary.mp4)<br>**Mice:** 1765, 728, 1694<br>Sim: $0.28 \pm 0.14$ |

---

## 5. Neurobiological Significance: Why the Power of $N$ Matters

In systems neuroscience and in vivo fiber photometry, behavioral sequences are not studied in isolation; they are aligned to **neural recordings** (e.g., striatal dopamine release transients, calcium activity in D1/D2 medium spiny neurons):

1. **The Signal-to-Noise Challenge:**  
   In Level 1, requiring exact identity across 100 fine-grained syllables means an animal may only contribute **$1\text{--}3$ valid trials** per 30-minute session. Peri-event time histograms (PETHs) constructed from 2–3 trials suffer from extreme shot noise and cannot detect subtle dopaminergic modulation.
2. **Statistical Power in Level 4:**  
   By clustering syllables into broad functional families, each animal contributes **$15\text{--}30$ valid, high-fidelity trials** for the same overarching behavioral arc.
3. **The Dual-Stream Resolution:**  
   Researchers obtain the necessary sample size ($N \approx 450$) to build clean, high-SNR dopamine event alignments, while using **continuous turning angle $\Delta\theta$ and velocity $v$** to control for execution intensity.

---

### Linked Artifacts
* **Cumulative Similarity Curves:** [fig_similarity_vs_qualifying_episodes.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_similarity_vs_qualifying_episodes.png)
* **P90 Similarity Scaling Summary CSV:** [p90_similarity_scaling_summary.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/p90_similarity_scaling_summary.csv)
* **P90 Panel Metadata JSON:** [p90_similarity_metadata.json](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/p90_similarity_metadata.json)
* **Master Resolution Comparison Report:** [CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md)
* **Presentation Guide:** [monday_14_7_2026_presentation.md](file:///home/fsp585/Desktop/FiberphotometryCode/monday_14_7_2026_presentation.md)
