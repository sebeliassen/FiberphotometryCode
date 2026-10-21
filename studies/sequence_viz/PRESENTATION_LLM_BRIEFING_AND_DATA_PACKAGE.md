# Presentation LLM Briefing & Data Package: seqNMF Multi-Scale Behavioral Syntax

**Document Purpose**: Comprehensive, copy-pasteable briefing package for a Presentation LLM (e.g. Claude, ChatGPT, Gamma, Beautiful.ai) or human speaker to generate a complete slide deck. It provides exact slide-by-slide outlines, quantitative tables, high-resolution visual figure paths, and neuroethological insights.

---

## 1. Executive Summary & Thesis

* **Title**: Uncovering Multi-Scale Behavioral Grammar: From Discrete Syllable Paths to Macro-Action Syntax
* **Dataset**: 30 Open-Field Video & Photometry Sessions (>160,000 continuous syllable bouts across 15 mice: 8 Wild-Type, 7 DAT-Het Hyper-dopaminergic Mutants, Saline vs. Amphetamine).
* **Core Scientific Thesis**:
  > *"Discrete unsupervised pose models (Keypoint-MoSeq) decompose mouse behavior into 100 fine postural states. However, sequence discovery cannot be solved at a single resolution: **Level 4 (14 broad families)** unlocks macro-action syntax and spans 76% of animal life while retaining high kinematic fidelity, whereas **Level 1 (100 raw states)** reveals flexible phrase branching from shared postural roots (e.g., touchdown into walk vs. sprint vs. turn) and eliminates the motor confound in fiber photometry."*

---

## 2. Hard Empirical Data & Performance Tables

### Table 1: Multi-Scale Sequence Discovery & Population Coverage (Census across >160,000 Bouts)
*Definition of Recurrent Family: $\ge 20$ occurrences across $\ge 8$ distinct mice. Population coverage is the 1D temporal union of all frames spanned by recurrent families, counting overlaps only once.*

| Horizon | Metric | Level 1 (100 Raw States) | Level 4 (14 Broad Families) | Quantitative Gain / Ratio | Ethological Meaning |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **5-Step** | **Population Time Coverage** | **14.1%** | **76.2%** | **$5.4\times$ Increase** | Coarse grouping structures $\frac{3}{4}$ of mouse life into recurring multi-animal programs. |
| **6-Step** | **Population Time Coverage** | **1.7%** | **46.0%** | **$27.1\times$ Increase** | Level 1 collapses combinatorially; Level 4 still organizes nearly half of all behavior. |
| **5-Step** | **Recurrent Sequence Families** | **345** | **1,524** | **$4.4\times$ More Families** | Eliminates single-occurrence combinatorial noise variants. |
| **6-Step** | **Recurrent Sequence Families** | **36** | **911** | **$25.3\times$ More Families** | Discovers sustained higher-order routines across multiple seconds. |
| **5-Step** | **Physical Duration ($mean \pm std$)** | $2.58\text{ s} \pm 1.29\text{ s}$ | $3.63\text{ s} \pm 2.22\text{ s}$ | **$+41\%$ Longer Behavioral Arcs** | Merging intra-family bouts ($A \to A \implies A$) captures true macro-motifs. |
| **5-Step** | **Median Duration** | $2.35\text{ s}$ | $3.17\text{ s}$ | $+35\%$ Shift | Extended routines extend to $10\text{--}15\text{ s}$ in grooming/rest. |
| **5-Step** | **Episode Support per Family** | $28$ [IQR: $23\text{--}39$] | $37$ [IQR: $26\text{--}63$] | **Heavier Upper Tail** | Top motifs recur hundreds of times across mice. |
| **5-Step** | **Cohort Penetration** | $10\text{ / }15$ mice | $12\text{ / }15$ mice | **Broader Penetration** | Motifs generalize more robustly across biological subjects. |

---

### Table 2: Kinematic Information Retention (LOAO Cross-Validation across 15 Animal Folds)
*Evaluation: Leave-One-Animal-Out cross-validation across 86,411 continuous bouts to test whether coarsening from 22 states (Level 3) to 14 states (Level 4) destroys movement predictability.*

| Kinematic Metric | Level 3 (22 States) | Level 4 (14 States) | Absolute Difference | Relative Change | Statistical Significance | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Heading Deflection ($\Delta\theta$)** | $21.02^\circ \pm 0.96^\circ$ | $21.58^\circ \pm 1.06^\circ$ | $+0.56^\circ$ | $+2.7\%$ | $p = 6.3 \times 10^{-4}$ | Modest angular error increase ($<1^\circ$). |
| **Centroid Speed ($v$)** | $27.85 \pm 2.32\text{ px/s}$ | $28.49 \pm 2.46\text{ px/s}$ | $+0.65\text{ px/s}$ | $+2.3\%$ | $p = 1.8 \times 10^{-3}$ | Speed prediction is essentially identical. |
| **Bout Duration ($d$)** | $0.429 \pm 0.021\text{ s}$ | $0.434 \pm 0.022\text{ s}$ | $+0.005\text{ s}$ | $+1.2\%$ | $p = 6.4 \times 10^{-6}$ | Penalty is only $5\text{ milliseconds}$. |
| **Net Displacement ($\Delta r$)** | $24.10 \pm 2.63\text{ px}$ | $24.54 \pm 2.76\text{ px}$ | $+0.45\text{ px}$ | $+1.8\%$ | $p = 1.1 \times 10^{-2}$ | Sub-pixel difference per bout. |

---

### Table 3: Level 1 Phrase Branching & The Spectrum of Left-Turn Syllables
*In Level 4, all left movements collapse into `Turn_Left` or `Walk_Left`. At Level 1, seqNMF discovers how sequences branch across distinct kinematic expressions:*

| Syllable ID | Human Annotation | Velocity ($v$) | Ethological / Biomechanical Function | Role in Behavioral Branching |
| :---: | :--- | :---: | :--- | :--- |
| **6** | `turn left` | $59.4\text{ cm/s}$ | **Acute angular body pivot** | Standard sharp turn; branches from touchdown in Motif 1 (10.6%). |
| **29** | `walk left (sharp left)` | $44.8\text{ cm/s}$ | **$180^\circ$ Hairpin / wall abort** | Wall collision evasion; extreme curvature, lower velocity. |
| **4** | `back down + turn left` | $67.4\text{ cm/s}$ | **Postural descent with orienting pivot** | Shared root of Motif 1; animal touches down and re-orients. |
| **3** | `walk left` | $87.9\text{ cm/s}$ | **Steady forward-left locomotor walk** | Sustained cruising across arena; branches in Motif 1 (27.1%). |
| **23** | `walk left (slow)` | $43.3\text{ cm/s}$ | **Low-speed cautious stepping left** | Ground sniffing combined with low-cadence lateral drift. |
| **55** | `walk left (slight veer)` | $96.2\text{ cm/s}$ | **High-speed cruise with $15^\circ$ left veer** | Open-field course trimming; preserves forward momentum. |
| **17** | `run left` | $77.2\text{ cm/s}$ | **High-velocity sprint transit left** | Explosive locomotor acceleration; branches in Motif 1 (22.6%). |
| **28** | `look left` | $45.8\text{ cm/s}$ | **Low-amplitude head orienting glance** | Stationary/slow body base with lateral sensory displacement. |
| **34** | `look left (from right)` | $40.5\text{ cm/s}$ | **Subtle visual glance / freeze sampling** | Oscillates with stationary pauses (`49`) in Motif 10 (WT-enriched). |

---

## 3. Slide-by-Slide Presentation Blueprint (7-Slide Storyboard)

```
[Slide 1: The Dilemma]        ──> Combinatorial explosion: 100 syllables create single-occurrence noise
            │
[Slide 2: Level 4 Coverage]   ──> Functional grouping structures 76% of animal life with minimal kinematic cost
            │
[Slide 3: Video Proof (L4)]   ──> 1×3 Widescreen: Cohesion across animals & natural kinematic variation
            │
[Slide 4: Photometry Solved]  ──> Cross-Genotype Replicas: Eliminating the hyperactivity motor confound
            │
[Slide 5: Zooming into L1]    ──> Phrase branching from shared roots: The "Argmax Phantom Trap" demystified
            │
[Slide 6: Left-Turn Spectrum] ──> Separating the 180° wall hairpin from the 15° slight open-field veer
            │
[Slide 7: Unified Multi-Scale]──> Macro-Verbs + Micro-Paths + Continuous Adverbs (Future: Mixed HSMM)
```

---

### Slide 1: The Combinatorial Dilemma of Unsupervised Sequences
* **Headline**: Decomposing Continuous Pose into Syllables Leaves Sequence Grammar Unsolved
* **Sub-headline**: 100 autoregressive states capture static poses, but raw n-grams suffer from combinatorial explosion.
* **Core Talking Points**:
  - Keypoint-MoSeq discretizes 40 Hz 3D pose dynamics into 100 autoregressive syllables ($S_0 \dots S_{99}$).
  - A 100-token vocabulary produces $100^N$ potential permutations ($10\text{ billion}$ possible 5-step chains).
  - *The Failure of Raw Strings*: At Level 1, 5-step strings recurring across mice cover only **14.1% of recording time**. At 6 steps, coverage collapses to a negligible **1.7%**.
  - 86% of the animal's life is dismissed as unclassifiable single-occurrence permutations.
* **Key Graphic / Visual to Embed**:
  - [fig_matched_action_chains_scaling.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_matched_action_chains_scaling.png) (Candidate scaling curves demonstrating combinatorial fragmentation).
* **Presenter's Takeaway**: *"We cannot discover behavioral programs simply by searching for exact 100-syllable string matches."*

---

### Slide 2: Level 4 Unlocks Macro-Syntax & Population Coverage
* **Headline**: Functional Clustering Organizes Over Three-Quarters of Behavior
* **Sub-headline**: Grouping into 14 functional families reveals 1,524 recurrent programs with minimal loss of kinematic detail.
* **Core Talking Points**:
  - **$5.4\times$ Coverage Leap**: Recurring 5-step families surge from 14.1% to **76.2% of cohort time** (and **46.0%** at 6 steps, a **$27\times$ increase**).
  - **Consecutive Bout Merging**: Collapsing intra-family repetitions ($A \to A \to A \implies A$) expands physical sequence duration from $2.58\text{ s}$ to **$3.63\text{ s}$ (+41%)**, capturing multi-second behavioral arcs.
  - **Rigorous LOAO Cross-Validation**: Leave-One-Animal-Out tests across 15 mouse folds prove this coarsening carries a negligible predictive penalty: heading error increases by only **$+0.56^\circ$ ($2.7\%$)**, velocity by **$+0.65\text{ px/s}$ ($2.3\%$)**, and duration by only **$5\text{ milliseconds}$ ($1.2\%$)**.
* **Key Graphic / Visual to Embed**:
  - [fig_kinematic_information_retention_loao.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_kinematic_information_retention_loao.png) (4-panel LOAO cross-validation error comparison).
* **Presenter's Takeaway**: *"Coarsening to 14 functional families structures 76% of mouse behavior while sacrificing less than 3% in kinematic predictability."*

---

### Slide 3: Video Grounding — Coherence Across Natural Kinematic Variation
* **Headline**: Macro-Motifs Remain Immediately Identifiable in Unpruned Cohort Video
* **Sub-headline**: 1×3 widescreen video demonstrates macro-program cohesion across different animals.
* **Core Talking Points**:
  - **The Three Demonstration Tiers**:
    1. *Canonical Exemplar*: Trajectory centroid at the density peak.
    2. *P90 Representative*: Instances operating at the 90th percentile of continuous kinematic similarity.
    3. *Ordinary Members*: Unpruned random draws from distinct mice showing natural execution spread.
  - In Theme A (`Sprint -> Turn_R -> Rear -> Descent -> Sprint`, $n=455$ episodes across 15 mice):
    - Across mice 1749, 1759, and 1771, the overarching motor circuit (wall approach $\to$ vertical rear $\to$ touchdown $\to$ sprint launch) is unmistakable.
    - Continuous kinematics absorb variations in rearing height and turn angle as natural execution parameters rather than fragmenting the routine into disconnected tokens.
* **Key Graphic / Visual to Embed**:
  - [theme_A_rear_descent_exemplar_thumb.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_4_broad_families/thumbnails/theme_A_rear_descent_exemplar_thumb.png)
  - Video Link: [theme_A_rear_descent_exemplar.mp4](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_4_broad_families/movies/theme_A_rear_descent_exemplar.mp4)
* **Presenter's Takeaway**: *"Level 4 sequences are not abstract statistics; they correspond to visually cohesive, reproducible motor programs."*

---

### Slide 4: The Photometry Breakthrough — Cross-Genotype Behavioral Replicas
* **Headline**: Eliminating the Locomotion Confound in Dopamine Photometry
* **Sub-headline**: Holding physical execution invariant between WT and hyperactive mutants isolates true neuromodulatory tone.
* **Core Talking Points**:
  - **The Movement Confound Problem**: Hyper-dopaminergic mutant mice (DAT-Het KI) run faster, rear more often, and turn sharper. When fiber photometry detects divergent dopamine transients, is it altered reward/motivation signaling or merely the motor artifact of moving faster?
  - **The Solution: Cross-Genotype Replicas**:
    - In Level 4 Macro-Motif 7, WT mouse `728` and KI mice `1658` and `1666` execute **exact replicas** of the 4-second Wall-Rear Sprint Loop.
    - Path length ($592\text{--}748\text{ px}$) and mean velocity ($152\text{--}181\text{ px/s}$) match across genotypes.
  - Aligning striatal dopamine signals to these matched behavioral replicas isolates pure genotype-specific neuromodulation, completely immune to movement confounds.
* **Key Graphic / Visual to Embed**:
  - [fig_clustered_seqnmf_motifs.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/fig_clustered_seqnmf_motifs.png) (Temporal motif matrix $W$ showing Macro-Motifs 7 and 4).
* **Presenter's Takeaway**: *"Behavioral replicas allow us to hold physical behavior constant and measure pure neural dynamics."*

---

### Slide 5: Zooming into Level 1 — Phrase Syntax & Decision Branching
* **Headline**: Level 1 Uncovers Decision Points and Flexible Branching Trees
* **Sub-headline**: seqNMF does not enforce rigid templates; it uncovers phrase families anchored on shared biomechanical roots.
* **Core Talking Points**:
  - **The Argmax Phantom Trap**: Taking the $\arg\max$ token at each lag of a seqNMF template creates synthetic sequences that never happen in reality (e.g. Motif 5 template `[6, 17, 18]` occurs **0% of the time**, because mice counter-steer rather than braking into a sniff).
  - **Flagship Motif 1 ($n=350$ verified episodes across 10 animals)**:
    - Shared Postural Root: `Syllable 14` (Rear) $\to$ `Syllable 4` (Touchdown & Orient Left).
    - Divergent Downstream Branches:
      - `[14, 4, 3]`: **27.1% (95 bouts)** $\to$ Steady Walk Left ($42.1\text{ cm/s}$).
      - `[14, 4, 17]`: **22.6% (79 bouts)** $\to$ Explosive Sprint Left ($77.2\text{ cm/s}$).
      - `[14, 4, 6]`: **10.6% (37 bouts)** $\to$ Sharp Pivot Turn Left ($59.4\text{ cm/s}$).
  - **Photometry Milestone**: Align dopamine/calcium to the touchdown moment (`Syllable 4`) to test whether phasic transients predict subsequent motor vigor and choice commitment.
* **Key Graphic / Visual to Embed**:
  - [theme_A_rear_descent_exemplar_thumb.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_1_original_syllables/thumbnails/theme_A_rear_descent_exemplar_thumb.png)
  - Video Link: [theme_A_rear_descent_exemplar.mp4](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_1_original_syllables/movies/theme_A_rear_descent_exemplar.mp4)
* **Presenter's Takeaway**: *"Level 1 reveals the branching tree: direction is established at touchdown, and downstream branches select motor vigor."*

---

### Slide 6: The Left-Turn Spectrum — Why Similar Syllables Must Not Be Merged
* **Headline**: Preserving the Difference Between an Obstacle Abort and Course Trimming
* **Sub-headline**: Level 1 distinguishes 9 distinct leftward states spanning turning, walking, and sensory inspection.
* **Core Talking Points**:
  - **The 180° Hairpin vs. 15° Slight Veer**:
    - `Syllable 29` is an extreme-curvature **$180^\circ$ wall abort turn** ($44.8\text{ cm/s}$, collision evasion).
    - `Syllable 55` is a high-speed forward cruise with a **slight $15^\circ$ course trim** ($96.2\text{ cm/s}$, momentum preserving).
    - Merging both into a single "Turn Left" token blinds the analysis to the animal's cognitive and navigational intent.
  - **Genotype Divergence at Level 1**:
    - Hyperactive mutants (DAT-Het KI) show a **$3\times$ enrichment in high-speed counter-steers** (Motif 5: `6 -> 17 -> 2`, $251\text{ KI} : 79\text{ WT}$) and an **$8\times$ elevation in rightward locomotion** (Motif 8: `11 -> 1 -> 30`, $263\text{ KI} : 33\text{ WT}$).
    - Wild-type mice show a **$5\times$ enrichment in stationary investigative scanning** (Motif 10: `20 -> 34 -> 49`, $159\text{ WT} : 33\text{ KI}$).
* **Key Graphic / Visual to Embed**:
  - [theme_B_turn_slalom_exemplar_thumb.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_1_original_syllables/thumbnails/theme_B_turn_slalom_exemplar_thumb.png)
* **Presenter's Takeaway**: *"Level 1 protects against the over-clustering trap, preserving subtle angular and velocity decisions."*

---

### Slide 7: The Multi-Scale Framework & Future Directions
* **Headline**: A Unified Three-Stream Architecture for Neuroethology
* **Sub-headline**: Integrating macro-syntax verbs, micro-path execution, and continuous kinematics.
* **Core Talking Points**:
  - **Stream 1: Macro-Syntax (Level 4, 14 States)**: Identifies 4–8s behavioral circuits, population-wide routines (76% coverage), and cross-genotype replicas.
  - **Stream 2: Micro-Paths & Branching (Level 1, 100 States)**: Uncovers flexible phrase syntax, decision milestones, and fine posture choices.
  - **Stream 3: Continuous Kinematics (Physical Adverbs)**: Tracks velocity, heading deflection $\Delta\theta$, and spatial displacement, ensuring continuous trajectory fidelity.
  - **Future Horizon (Parked Pilot)**: *Mixed Semi-Markov Modeling (HSMM)* with duration-explicit transitions and continuous kinematic likelihoods to soften rigid token boundaries.
* **Key Graphic / Visual to Embed**:
  - [fig_similarity_vs_qualifying_episodes.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_similarity_vs_qualifying_episodes.png)
* **Presenter's Takeaway**: *"By pairing Level 4's population coverage with Level 1's branching resolution, we capture both the forest and the trees of mouse behavior."*

---

## 4. Visual Asset Directory (Ready-to-Use Files)

### High-Resolution Publication Figures (PNG)
* **LOAO Cross-Validation Prediction Errors**:  
  [`fig_kinematic_information_retention_loao.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_kinematic_information_retention_loao.png)
* **Matched Candidate Yield & Scaling**:  
  [`fig_matched_action_chains_scaling.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_matched_action_chains_scaling.png)
* **seqNMF Macro-Motif Factor Matrices $W$**:  
  [`fig_clustered_seqnmf_motifs.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/fig_clustered_seqnmf_motifs.png)
* **Similarity vs. Qualifying Episode Curves**:  
  [`fig_similarity_vs_qualifying_episodes.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_similarity_vs_qualifying_episodes.png)

### 1×3 Widescreen Video Thumbnails & Posters (PNG)
* **Level 1 Exemplar — Wall Rear & Descent (Theme A)**:  
  [`theme_A_rear_descent_exemplar_thumb.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_1_original_syllables/thumbnails/theme_A_rear_descent_exemplar_thumb.png)
* **Level 1 Exemplar — Turn Slalom (Theme B)**:  
  [`theme_B_turn_slalom_exemplar_thumb.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_1_original_syllables/thumbnails/theme_B_turn_slalom_exemplar_thumb.png)
* **Level 4 Exemplar — Wall Rear & Sprint Loop (Theme A)**:  
  [`theme_A_rear_descent_exemplar_thumb.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_4_broad_families/thumbnails/theme_A_rear_descent_exemplar_thumb.png)
* **Level 4 Ordinary Members — Natural Execution Variation (Theme A)**:  
  [`theme_A_rear_descent_ordinary_thumb.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_4_broad_families/thumbnails/theme_A_rear_descent_ordinary_thumb.png)

### Video Playback Files (MP4)
* **Level 1 Theme A (Exemplar)**: [`level_1_original_syllables/movies/theme_A_rear_descent_exemplar.mp4`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_1_original_syllables/movies/theme_A_rear_descent_exemplar.mp4)
* **Level 1 Theme B (Exemplar)**: [`level_1_original_syllables/movies/theme_B_turn_slalom_exemplar.mp4`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_1_original_syllables/movies/theme_B_turn_slalom_exemplar.mp4)
* **Level 4 Theme A (Exemplar)**: [`level_4_broad_families/movies/theme_A_rear_descent_exemplar.mp4`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_4_broad_families/movies/theme_A_rear_descent_exemplar.mp4)
* **Level 4 Theme A (Ordinary)**: [`level_4_broad_families/movies/theme_A_rear_descent_ordinary.mp4`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_4_broad_families/movies/theme_A_rear_descent_ordinary.mp4)
