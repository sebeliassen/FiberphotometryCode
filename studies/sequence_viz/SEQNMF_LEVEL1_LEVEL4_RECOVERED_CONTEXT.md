# seqNMF Behavioral Resolution Story: Level 1 (Branching Paths) vs. Level 4 (Population Coverage)

**Project:** Unsupervised Sequence Discovery & Behavioral Grammar via Keypoint-MoSeq & Controlled Convolutional seqNMF  
**Dataset:** 30 Open-Field Imaging Sessions (15 Amphetamine / 15 Saline; 15 WT, 15 DAT-KI Animals)  
**Corpus Scale:** >160,000 Continuous Syllable Bouts, 54 Expert-Annotated Syllables (>90% Cohort Bouts & Time)  
**Document Purpose:** Recovered historical synthesis and evidence-backed presentation narrative establishing the complementarity between **Level 1 (Micro-Structure & Branching Paths)** and **Level 4 (Macro-Syntax & Population Coverage)**. Mixed HSMM work is officially parked as "Future Work".

---

## Executive Summary & Presentation Narrative

When evaluating sequence discovery across clustering resolutions, the question is **not which resolution is "better" in isolation**, but how multi-scale behavioral organization serves distinct neuroethological questions:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STREAM 1: MACRO-SYNTAX & BEHAVIORAL FAMILIES (Level 4: 14 Broad Functional States)     │
│  - Structures 76.2% of all cohort time into recurring 5-step multi-animal families     │
│  - Yields 1,524 recurrent 5-step families (4.4× more than L1) & 911 6-step families   │
│  - Captures extended 3.6–4.8s behavioral arcs via consecutive bout collapsing          │
│  - Discovers Cross-Genotype Behavioral Replicas to eliminate photometric motor confounds│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STREAM 2: MICRO-STRUCTURE & BEHAVIORAL PATHS (Level 1: 100 Raw Autoregressive States)  │
│  - Uncovers flexible phrase branching from shared postural roots (e.g., 14 -> 4)       │
│  - Demystifies the "Argmax Phantom Trap" (argmax templates vs. empirical syntax)       │
│  - Pinpoints precise neural alignment milestones (e.g., touchdown at syllable 4)      │
│  - Resolves fine biomechanical divergence (walk vs. sprint vs. counter-steer)          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STREAM 3: CONTINUOUS PHYSICAL KINEMATICS (Continuous Execution "Adverbs")              │
│  - Continuous velocity (px/s), net displacement (px), bout duration (s)               │
│  - Signed angular deflection Δθ (deg): separates subtle veers from sharp hairpins      │
│  - Retains high kinematic fidelity: L4 suffers only 1.2%–2.7% LOAO prediction penalty  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Master Evidence Index

All quantitative statements and visual records are cross-validated against the following repository artifacts:

| Resource / Asset | Repository Path | Core Subject / Evidence Provided |
| :--- | :--- | :--- |
| **Decoded Chains Report** | [DECODED_SEQNMF_CHAINS_WITH_HUMAN_LABELS.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/DECODED_SEQNMF_CHAINS_WITH_HUMAN_LABELS.md) | Ground-truth breakdown of empirical phrase families, argmax phantom traps, and WT vs. KI motif enrichments. |
| **Final Resolution Decision** | [FINAL_RESOLUTION_DECISION_AND_FRAMEWORK.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/FINAL_RESOLUTION_DECISION_AND_FRAMEWORK.md) | Dual-stream framework rationale, LOAO cross-validation metrics, and presentation guidelines. |
| **Clustering Video Comparison** | [CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md) | Complete 4-tier matrix ($N=4,5,6$), population coverage integrals, and 3-tier video manifest. |
| **Diagnostic Audit Report** | [CLUSTERING_RESOLUTION_DIAGNOSTIC_AUDIT.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_DIAGNOSTIC_AUDIT.md) | Resolves recurrence vs. similarity discrepancy; explains query length and structural bias. |
| **N-Step Statistics CSV** | [n_step_sequence_statistics.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/n_step_sequence_statistics.csv) | Full census of observed permutations, animal penetration, and recurrence thresholds. |
| **Rigorous Coverage CSV** | [clustering_rigorous_coverage_statistics.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/clustering_rigorous_coverage_statistics.csv) | 1D temporal union coverage, per-animal distributions, episode IQRs, and physical durations. |
| **LOAO Kinematics Script** | [evaluate_kinematic_information_retention.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/evaluate_kinematic_information_retention.py) | Leave-One-Animal-Out cross-validation script for out-of-sample kinematic predictability. |
| **LOAO Summary CSV** | [loao_kinematic_information_retention_summary.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/diagnostic_audit/loao_kinematic_information_retention_summary.csv) | MAE, SEM, and $R^2$ values for heading, velocity, duration, and translation. |
| **Macro-Motifs Exemplars** | [WHOLE_CHAIN_EXEMPLAR_MACRO_MOTIFS.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/WHOLE_CHAIN_EXEMPLAR_MACRO_MOTIFS.md) | Whole-chain macro-motifs (4–8s), cross-genotype behavioral replicas, and dispersion gating. |
| **Master Ethology Guide** | [README_FOR_CHATGPT.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/README_FOR_CHATGPT.md) | System architecture, Goldilocks clustering dilemma, and fiber photometry confound solutions. |

---

## 2. Level 1 Strength: Fine Behavioral Paths & Phrase Branching

At Level 1 (100 autoregressive states), seqNMF does not recover rigid, single-track deterministic chains. Instead, it recovers **probabilistic phrase families**: behavioral subroutines anchored on a shared biomechanical root that branch into divergent functional continuations.

### A. The "Argmax Phantom Trap"
A naive approach to seqNMF takes the $\arg\max$ syllable at each temporal lag of the factor matrix $W_k$. However, our empirical audit proved that **argmax templates often construct synthetic phantom sequences that never occur in reality**:
* **Motif 5:** The raw argmax sequence was `[6, 17, 18]` (`Turn_L -> Run_L -> Sniff`). In reality, this exact sequence occurs **0 times (0.0%)**. Mice virtually never brake into a sniff after a high-speed left run; instead, 50.0% of episodes violently counter-steer into `2` (`Turn_R`).
* **Motif 10:** The raw argmax sequence was `[20, 34, 23]`. Its empirical occurrence was **0 times (0.0%)**. Instead, the actual behavior is `[20, 34, 49]` (sniff $\to$ subtle head glance $\to$ stationary freeze).

### B. Flagship Branching Motif Families (Level 1)

```
                       ┌──> [3]  Walk Left (27.1%, 95 ep, 10 mice) [Low Vigor]
                       │
[14] Rear/Lean ──> [4] Descend/Pivot ──> [17] Run Left  (22.6%, 79 ep, 10 mice) [High Vigor Sprint]
                       │
                       └──> [6]  Turn Left (10.6%, 37 ep, 9 mice)  [Sharp Deflection]
```

#### 1. Motif 1: The Rear–Descent Locomotor Family (Flagship Presentation Example)
* **Shared Root:** `Syllable 14` (Rear / Lean Wall) $\to$ `Syllable 4` (Touchdown & Orient Left).
* **Empirical Divergent Branches ($n=350$ verified episodes across 10 animals):**
  * `[14, 4, 3]`: **95 episodes (27.1%, 10 animals)** $\to$ Transition into `Walk Left` ($42.1\text{ cm/s}$).
  * `[14, 4, 17]`: **79 episodes (22.6%, 10 animals)** $\to$ Transition into `Run Left` ($77.2\text{ cm/s}$).
  * `[14, 4, 6]`: **37 episodes (10.6%, 9 animals)** $\to$ Transition into sharp `Turn Left` (deflection).
* **Neuroethological / Photometry Milestone:**
  Aligning striatal dopamine (DA) or calcium transients to the touchdown moment (`Syllable 4`) provides an unconfounded test of whether phasic neuromodulation predicts the subsequent motor vigor commitment (`Walk` vs. `Run` vs. `Turn`).

#### 2. Motif 5: High-Speed Slalom / Counter-Steer Family ($3\times$ KI Enriched)
* **Shared Root:** `Syllable 6` (Turn Left) $\to$ `Syllable 17` (Run Left at $77.2\text{ cm/s}$).
* **Empirical Divergent Branches ($n=330$ episodes; 79 WT, 251 KI):**
  * `[6, 17, 2]`: **165 episodes (50.0%, 10 animals)** $\to$ Rapid counter-steer right (`Turn_R`).
  * `[6, 17, 6]`: **44 episodes (13.3%, 9 animals)** $\to$ Continued leftward arc (`Turn_L`).
  * `[6, 17, 21]`: **32 episodes (9.7%, 8 animals)** $\to$ Forward acceleration dash (`Dash`).
* **Ethological Meaning:** A high-speed slalom maneuver strongly elevated in hyper-dopaminergic DAT-KI mutants ($251\text{ KI} : 79\text{ WT}$).

#### 3. Motif 8: Rightward Locomotion & Divergent Redirection ($8\times$ KI Enriched)
* **Shared Root:** `Syllable 11` (Walk Right) $\to$ `Syllable 1` (Walk Right).
* **Empirical Divergent Branches ($n=296$ episodes; 33 WT, 263 KI):**
  * `[11, 1, 30]`: **129 episodes (43.6%, 9 animals)** $\to$ Sharp rightward angular turn (`30`, $71.3\text{ cm/s}$).
  * `[11, 1, 0]`: **103 episodes (34.8%, 9 animals)** $\to$ High-speed forward sprint (`0`, $76.1\text{ cm/s}$).
* **Ethological Meaning:** Distinguishes forward acceleration from acute angular redirection during persistent locomotion.

#### 4. Motif 10: Stationary Investigative Scanning ($5\times$ WT Enriched)
* **Shared Root:** `Syllable 20` (Sniff Right) $\to$ `Syllable 34` (Look Left Subtle).
* **Empirical Divergent Branches ($n=192$ episodes; 159 WT, 33 KI):**
  * `[20, 34, 49]`: **72 episodes (37.5%, 8 animals)** $\to$ Stationary arrest / pause (`49`).
  * `[20, 34, 8]`: **49 episodes (25.5%, 7 animals)** $\to$ Low-velocity re-sniff left (`8`).

### C. The Left-Turn & Lateral Steer Spectrum: Why "Similar but Not Identical" Syllables Matter

A cornerstone of our ethological discussion was why Level 1 is indispensable despite its lower raw coverage: **it separates similar-looking movements that actually express distinct biomechanical, curvature, and vigor decisions**.

In coarse clustering (Level 4), all left-turning and left-locomoting actions are collapsed into broad classes like `Turn_Left` (`6, 29`) or `Walk_Left` (`3, 23, 55`). But in **Level 1**, Keypoint-MoSeq discovers an entire spectrum of distinct left-directed syllables:

| Syllable ID | Human Annotation | Velocity ($v$) | Kinematic & Ethological Role | How Level 1 Uncovers It |
| :---: | :--- | :---: | :--- | :--- |
| **6** | `turn left` | $59.4\text{ cm/s}$ | **Acute angular body deflection / pivot** | Standard sharp turn; branches in Motif 1 and initiates Motif 5. |
| **29** | `walk left (sharp left)` | $44.8\text{ cm/s}$ | **Extreme-curvature hairpin / wall abort** | $180^\circ$ wall evasion; high body torque at lower speed. |
| **3** | `walk left` | $87.9\text{ cm/s}$ | **Steady forward-left locomotor cruising** | Rhythmic translation across open field; moderate vigor. |
| **23** | `walk left (slow)` | $43.3\text{ cm/s}$ | **Low-speed cautious left stepping** | Ground sniffing combined with low-tempo left drift. |
| **55** | `walk left (slight veer)` | $96.2\text{ cm/s}$ | **High-speed forward cruise with slight left look** | $15^\circ$ course trim preserving momentum; NOT a sharp pivot. |
| **17** | `run left` | $77.2\text{ cm/s}$ | **High-velocity sprint transit left** | Explosive locomotor acceleration; paired counterpart to `0`. |
| **4** | `back down + turn left` | $67.4\text{ cm/s}$ | **Postural descent with orienting touchdown** | Wall exploration exit; biomechanical inflection point. |
| **28** | `look left` | $45.8\text{ cm/s}$ | **Low-amplitude head orienting glance** | Stationary/slow base with sensory head displacement. |
| **34** | `look left (from right)` | $40.5\text{ cm/s}$ | **Subtle visual glance / freeze sampling** | Coupled tightly with stationary pauses (`49`) in Motif 10. |

#### Why This Branching Structure Is Scientifically Powerful:
1. **The 180° Hairpin vs. 15° Slight Veer Trap**:
   - If an algorithm clusters `6`, `29`, and `55` into a monolithic "Turn" verb, it conflates an **obstacle collision abort** (`29`) with a **minor 15° open-field course adjustment** (`55`).
   - At Level 1, seqNMF reveals that upstream behavioral contexts branch specifically into either the sharp hairpin or the gentle veer, exposing the animal's true environmental steering strategy.
2. **Post-Descent Branching in Motif 1**:
   - Following `14 -> 4` (Rear $\to$ Descent Touchdown), the animal does *not* branch randomly across the 100 syllables; it branches almost exclusively across **three left-directed variants**:
     - `3` (steady walk, 27.1%)
     - `17` (explosive run, 22.6%)
     - `6` (sharp pivot deflection, 10.6%)
   - This proves that the mouse has committed to a leftward trajectory, but **branches across motor vigor and curvature expressions**.
3. **The "Blind Substitution" Dilemma**:
   - A rigid discrete token matcher considers replacing `17` with `55` as a complete error, even though both execute high-speed forward locomotion with leftward yaw.
   - At Level 1, analyzing empirical phrase trees reveals how these near-synonyms act as alternate branches of the same underlying behavioral routine.

---


## 3. Level 4 Strength: Population Coverage & Macro-Syntax

While Level 1 excels at revealing branching micro-mechanics, it suffers catastrophic fragmentation when tasked with structuring extended behavioral time. **Level 4 (14 broad functional states)** solves this by organizing the majority of mouse life into recurring behavioral families.

### A. Rigorous Quantitative Benchmark (Census across >160,000 Bouts, 15 Mice)

| Metric | Level 1 (100 Raw States) | Level 4 (14 Broad Families) | Quantitative Ratio / Gain | Ethological Significance |
| :--- | :---: | :---: | :---: | :--- |
| **5-Step Population Time Coverage** | **14.1%** | **76.2%** | **$5.4\times$ Increase** | Over $\frac{3}{4}$ of total recording time is parsed into recurrent multi-animal motifs. |
| **6-Step Population Time Coverage** | **1.7%** | **46.0%** | **$27.1\times$ Increase** | Level 1 completely collapses at 6 steps; Level 4 still structures nearly half of all behavior. |
| **Recurrent 5-Step Families**<br>*(≥20 ep, ≥8 mice)* | **345** | **1,524** | **$4.4\times$ More Families** | Broad functional grouping eliminates single-occurrence noise permutations. |
| **Recurrent 6-Step Families**<br>*(≥20 ep, ≥8 mice)* | **36** | **911** | **$25.3\times$ More Families** | Sustained higher-order syntax discovery across multi-second spans. |
| **Actual Physical Duration (5-Step)** | $2.58\text{ s} \pm 1.29\text{ s}$<br>*(median 2.35s)* | $3.63\text{ s} \pm 2.22\text{ s}$<br>*(median 3.17s)* | **$+41\%$ Longer Behavioral Arcs** | Collapsing intra-family repetitions ($A \to A \implies A$) captures true macro-actions. |
| **Episode Support per Family (5-Step)** | 28 episodes<br>[IQR: 23–39] | 37 episodes<br>[IQR: 26–63] | **Heavier Upper Tail** | High-support circuits occur hundreds of times across mice. |
| **Cohort Penetration per Family** | 10 mice<br>[IQR: 9–11] | 12 mice<br>[IQR: 10–14] | **Broader Penetration** | Motifs generalize more reliably across independent biological subjects. |

### B. Kinematic Information Retention (LOAO Cross-Validation)
A critical concern when coarsening vocabulary is whether functional clustering destroys kinematic information. Our **Leave-One-Animal-Out Cross-Validation (LOAO-CV)** across all 15 animal folds proved that **Level 4 retains high bout-level kinematic fidelity**:

* **Heading Change ($\Delta\theta$):** Error increases by only **$+0.56^\circ$** ($+2.7\%$, from $21.02^\circ$ in L3 to $21.58^\circ$ in L4; $p = 6.3 \times 10^{-4}$).
* **Centroid Velocity ($v$):** Error increases by only **$+0.65\text{ px/s}$** ($+2.3\%$, from $27.85$ to $28.49\text{ px/s}$; $p = 1.8 \times 10^{-3}$).
* **Bout Duration ($d$):** Error increases by only **$+5\text{ milliseconds}$** ($+1.2\%$, from $0.429\text{ s}$ to $0.434\text{ s}$; $p = 6.4 \times 10^{-6}$).
* **Net Displacement ($\Delta r$):** Error increases by only **$+0.45\text{ px}$** ($+1.8\%$, from $24.10$ to $24.54\text{ px}$; sub-pixel difference).

> **Presentation Rule:** Avoid claiming "only 3% information loss" (which implies all postural dimensions were tested). State accurately: *"Broader grouping increases 5-step recurrent coverage by $5.4\times$ while incurring only a $1.2\%\text{--}2.7\%$ prediction penalty on bout-level heading, velocity, and duration."*

---

## 4. Cross-Genotype Behavioral Replicas: Motor-Invariant Photometry

In behavioral neuroscience, comparing dopamine or calcium transients between wild-type and hyperactive mutant animals is plagued by **movement confounds** (hyperactive animals run faster, turn sharper, or rear more frequently).

Level 4 macro-motifs solve this via **Cross-Genotype Behavioral Replicas**:
* In **Macro-Motif 7 (Wall-Rear & Sprint Loop)**:
  * WT `728`, KI `1658`, and KI `1666` execute the exact same multi-second behavioral sequence (`Sprint -> Turn_R -> Rear -> Descent -> Sprint`).
  * Total path length ($592\text{--}748\text{ px}$), mean velocity ($152\text{--}181\text{ px/s}$), and rearing trajectory match across animals.
* **The Neuroethological Payoff:** Holding the gross physical motor program constant isolates pure genotype-specific neuromodulatory signals (baseline dopamine tone, reward expectation, or transition vigor), eliminating physical locomotion confounds.

---

## 5. Curated Video Asset Shortlist for Presentations

The following verified 1×3 widescreen video assets are pre-rendered and ready for slides or publication supplements:

### A. Level 1 Videos: Micro-Structure & Branching Demonstration
Located in `syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_1_original_syllables/movies/`:
* `theme_A_rear_descent_exemplar.mp4`: Wall-rear, touchdown, and leftward locomotion ($n=62$, 11 mice).
* `theme_A_rear_descent_p90.mp4`: High-fidelity execution near the 90th percentile of continuous kinematic similarity.
* `theme_B_turn_slalom_exemplar.mp4`: Rapid turn-sprint-counter-steer sequence ($n=155$, 15 mice).
* `theme_C_sensory_rest_exemplar.mp4`: Low-velocity sniff and pause cycles ($n=111$, 14 mice).

### B. Level 4 Videos: Broad Behavioral Families & Macro-Syntax
Located in `syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/level_4_broad_families/movies/`:
* `theme_A_rear_descent_exemplar.mp4`: Full macro-loop (Sprint $\to$ Turn $\to$ Rear $\to$ Descent $\to$ Sprint; $n=455$, 15 mice).
* `theme_A_rear_descent_ordinary.mp4`: Random unpruned members from cohort (mice 1749, 1759, 1771) demonstrating macro-program coherence despite natural kinematic variation.
* `theme_B_turn_slalom_exemplar.mp4`: High-speed slalom burst ($n=94$, 14 mice).
* `theme_C_sensory_rest_exemplar.mp4`: Extended grooming and rest maintenance cycle ($4.84\text{ s}$ average duration; $n=224$, 14 mice).

### C. Presentation Figures
* [fig_kinematic_information_retention_loao.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_kinematic_information_retention_loao.png): 4-panel LOAO cross-validation error curves demonstrating minimal prediction cost across resolutions.
* [fig_matched_action_chains_scaling.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_matched_action_chains_scaling.png): Log-scale scaling of candidate yield across resolutions under identical action queries.

---

## 6. Slide-by-Slide Presentation Outline Recommendation

1. **Slide 1: The Dilemma of Unsupervised Behavioral Sequences**
   * Keypoint-MoSeq discovers 100 fine autoregressive states.
   * Challenge: Combinatorial explosion ($100^N$) means 5-step raw strings cover only 14.1% of behavior, collapsing to 1.7% at 6 steps.
2. **Slide 2: Level 4 Unlocks Macro-Syntax & High Population Coverage**
   * Grouping into 14 functional families reveals recurrent sequences covering **76.2% of cohort life** at 5 steps and **46.0%** at 6 steps ($5.4\times$ and $27\times$ gains).
   * Consecutive bout merging captures full multi-second arcs ($3.6\text{--}4.8\text{ s}$).
   * Cross-validation confirms this coverage comes at minimal cost to bout kinematics ($+0.56^\circ$ heading, $+0.65\text{ px/s}$ speed).
3. **Slide 3: Level 4 Yields Cross-Genotype Replicas for Photometry**
   * Show 1×3 video: WT and DAT-KI animals executing identical 4-second Wall-Rear-Sprint circuits.
   * Explaining how motor-invariant matching solves the hyperactivity confound in fiber photometry.
4. **Slide 4: Zooming into Level 1: Phrase Syntax & Decision Branching**
   * Transition from family level to fine mechanics. Show that seqNMF does not enforce rigid templates.
   * Demystify the "Argmax Phantom Trap" (argmax vs. empirical syntax).
   * Present Motif 1 ($14 \to 4 \to \{3, 17, 6\}$): Postural touchdown branches into walking, sprinting, or turning.
   * Pristine photometry question: Aligning striatal signals to Syllable 4 touchdown to decode motor choice vigor.
5. **Slide 5: Summary & Future Outlook**
   * Multi-scale framework: Macro-verbs (Level 4), micro-paths (Level 1), continuous adverbs (kinematics).
   * *Future Work:* Probabilistic duration and continuous-time semi-Markov modeling (Mixed HSMM) to eliminate hard token boundaries.
