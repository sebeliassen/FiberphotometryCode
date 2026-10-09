# Presentation: Decoding Hierarchical Behavioral Organization in Dopamine Transporter Dysfunction
**Date:** Monday, July 14, 2026  
**Topic:** From Kinematics to Hierarchical shMoSeq: Proving Non-Markovian Behavioral Memory and Dopaminergic Disruption

---

## Slide 1: The Evolution of Behavioral Tracking — From Trajectories to Syllables

### Visuals / Layout
- **Left:** Centroid tracking (2D trajectory in open field, heatmaps of position and velocity).
- **Middle:** DeepLabCut / 3D Pose Estimation (skeletal keypoints on head, spine, paws, tail).
- **Right:** Keypoint-MoSeq (unsupervised segmentation of continuous kinematics into discrete sub-second "syllables").

### Speaking Points
1. **Where We Started (Trajectory Analysis):**
   - For decades, behavioral neuroscience relied on macroscopic centroid tracking: total distance traveled, mean velocity, center vs. periphery occupancy.
   - *The limitation:* Two mice can travel the exact same distance at the exact same velocity, yet one is engaged in focused exploratory rearing while the other is performing compulsive stereotypic circling.
2. **The Pose Revolution (DeepLabCut & 3D Kinematics):**
   - Computer vision advances enabled tracking high-dimensional skeletal keypoints across time.
   - This provided unprecedented resolution of the animal's physical posture.
3. **The Syllable Representation (Keypoint-MoSeq):**
   - Using autoregressive hidden Markov models on keypoints, Keypoint-MoSeq segments continuous motion into discrete, stereotyped, sub-second building blocks (~300–400 ms behavioral "syllables" like *rear-up*, *sniff-floor*, *left-turn*, *paw-lick*).
   - This transformed continuous motion into a symbolic "language of behavior," allowing us to construct transition matrices and syntax graphs.

---

## Slide 2: The Critical Question — *Is a 1-Step Transition Matrix Enough?*

### Visuals / Layout
- **Diagram:** A 1st-order Markov chain ($S_t \to S_{t+1}$) with a robotic coin-toss icon.
- **Dilemma Graphic:** Same syllable (e.g. *Rear-Up*) leading to two completely different outcomes depending on context.

### Speaking Points
1. **The Standard Approach (1st-Order Markov Chains):**
   - Most behavioral syntax models assume behavior is a 1st-order Markov process: the probability of the next syllable depends solely on the single syllable just executed:
     $$P(S_{t+1} \mid S_t, S_{t-1}, \dots) = P(S_{t+1} \mid S_t)$$
2. **The Fatal Flaw — Memorylessness:**
   - A 1st-order Markov model is memoryless. It assumes the mouse resets its decision history every 300 milliseconds.
   - *In reality:* A "Rear-Up" syllable in the middle of active exploration leads to forward darting; the exact same "Rear-Up" syllable in the middle of grooming leads to face-washing and scratching.
   - Transitions also drift over time (habituation across a 60-minute session).
3. **The Concept of Latent State:**
   - Syllables are not selected in isolation. They are generated under the influence of **underlying behavioral states** that persist for seconds to minutes.
   - To capture real behavior, we need a hierarchical model that infers these latent macro-states.

---

## Slide 3: Validating Behavioral Memory — The Weinreb Lagged MI Experiment

### Visuals / Layout
- **Figure:** [fig1_real_vs_markov_baseline_removed.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig1_real_vs_markov_baseline_removed.png)
  - **Panel A:** Raw Lagged Mutual Information (Real Data vs. 1st-Order Markov Null).
  - **Panel B:** Baseline Removed with Excess Memory Shaded ($\Delta I > 0$).

```
        A. Raw Lagged MI (Real vs. Markov)             B. Baseline Removed (Floor Corrected)
  Mutual Info (bits)                            Net Mutual Info (bits)
   1.4 ┼                                         1.0 ┼
   1.0 ┼ ╲  Real Data (Orange)                   0.6 ┼ ╲  Real Data (Solid)
   0.6 ┼  ╲                                          ┼  ╲ ░░ Excess Memory (Shaded)
   0.3 ┼   ─── 1st-Order Markov (Black)          0.2 ┼   - - 1st-Order Markov (Dashed)
   0.0 ┴───────┬─────────┬─────────┬─────────    0.0 ┼═════════════════════════════════
      1s      10s       100s      1000s             1s      10s       100s      1000s
                    Lag τ (seconds)                               Lag τ (seconds)
```

### Speaking Points
1. **The Literature Benchmark (Weinreb et al., *Neuron* 2026):**
   - Luckily, we are not the first to consider this. The Datta Lab recently introduced **shMoSeq** (semi-hidden Markov model on MoSeq) specifically to address this hierarchical organization.
   - To prove that syllables have long-timescale memory, they introduced the **Lagged Mutual Information diagnostic**:
     $$\text{If we observe syllable } S_t \text{ right now, how much information (in bits) does that provide about syllable } S_{t+\tau} \text{ performed } \tau \text{ seconds later?}$$
2. **Applying the Diagnostic to Our Dataset:**
   - We applied Weinreb's exact official algorithm across our 30-session cohort ($>4.1$ million continuous frames).
   - **The 1st-Order Markov Null (Black Dashed):** Decays precipitously to zero within 10–20 seconds. It forgets everything.
   - **Real Mouse Data (Orange Solid):** Decays far more slowly, maintaining a prominent shoulder of excess memory out past 60–100 seconds.
3. **The Peak Memory Timescale:**
   - Subtracting the baseline reveals that excess behavioral memory peaks at **$\tau \approx 6.3\text{ seconds}$ ($10^{0.8}\text{s}$, $+0.11\text{ bits}$)**.
   - This 6-second peak represents the naturalistic duration of animal **behavioral bouts**. Behavior is non-Markovian at the syllable level because it is governed by latent states.

---

## Slide 4: Ruling Out the Confound — Is State Just a Proxy for Time in Arena?

### Visuals / Layout
- **Figure:** [fig1_timescale_state_vs_time_decomposition.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig1_timescale_state_vs_time_decomposition.png)
  - 4-Way Information Decomposition: 1st-Order Markov, Time in Arena ($t$), State ($Z_t$), and Time + State ($t, Z_t$).

### Speaking Points
1. **The Skeptic's Question:**
   - *"Could this multi-second memory just be a trivial artifact of session time (arena habituation across the 60 minutes)?"*
2. **The 4-Way Information Decomposition:**
   - **Time Alone (Purple):** Sits completely flat at ~0.48 bits across all lags. Session time is a 60-minute variable—it has zero temporal resolution on whether a mouse will rear or sniff 6 seconds from now.
   - **State Beyond Time (Gold):** When we condition on session time, the inferred macro-state ($Z_t$) still provides **$+0.18\text{ bits}$ of independent, non-redundant predictive power**.
   - **The Memory Peak Remains:** Even after removing the time baseline, the excess memory peak at $\tau = 6.2\text{s}$ is fully preserved.
3. **Conclusion:**
   - shMoSeq macro-states are **not** a clock of arena time; they capture true internal, recurring behavioral organization.

---

## Slide 5: The Biological Headline — Dopaminergic Control of Behavioral Memory

### Visuals / Layout
- **Figure:** [fig1_timescale_by_group_comparison.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig1_timescale_by_group_comparison.png)
  - **Panel A:** WT Saline vs. WT Amphetamine.
  - **Panel B:** DAT-KI Saline vs. DAT-KI Amphetamine.
  - **Panel C:** Excess Memory Curves ($\Delta I$) across all 4 groups.
  - **Panel D:** Per-mouse bout memory at the 6.2s peak.

```
       Excess Behavioral Memory Across Groups (ΔI = Real − Markov)
  ΔI (bits)
   0.3 ┼ ╲
       │  ╲──────────────── WT Amphetamine (Red: Massive Stereotypic Inertia)
   0.2 ┼   ╲
   0.1 ┼    ▲ Peak at ~6s ──── WT Saline (Blue) & DAT-KI Saline (Green)
   0.0 ┼══════════════════════════════════════════════════════════════════
  -0.1 ┼ ─ ─ ─ ─ ─ ─ ─ ─ ──── DAT-KI Amphetamine (Orange: Severe Disruption)
       ┴───────┬─────────┬─────────┬─────────
              1s        6s        60s       300s
                               Lag τ (seconds)
```

### Speaking Points
1. **Normative Baseline (WT Saline & DAT-KI Saline):**
   - Both wild-type and DAT-KI mice at baseline exhibit an identical, highly consistent excess memory peak at **$\sim 5\text{--}6\text{ seconds}$** ($+0.14\text{ bits}$ in WT, $+0.10\text{ bits}$ in KI).
   - Every single animal in both baseline cohorts shows strictly positive bout memory.
2. **WT Amphetamine — Pathological Persistence:**
   - Under amphetamine, wild-type mice show massive elevation in behavioral memory across all lags ($+0.25\text{ bits}$ at 1s, $+0.12\text{ bits}$ persisting past 60s!).
   - Psychostimulants lock wild-type mice into persistent, stereotypic locomotive loops.
3. **DAT-KI Amphetamine — Severe Disruption & Alternating Loops:**
   - In DAT-KI mutants lacking normal dopamine reuptake, amphetamine triggers a catastrophic breakdown of normal bout organization.
   - Subsets of mutant mice collapse into **extreme 2-syllable alternating loops** ($A \leftrightarrow B$ circling in place), where 1-step Markov transitions become deterministic while naturalistic bout structure is completely erased.

---

## Slide 6: How shMoSeq States Re-Route Behavioral Syntax (Level 1 Mechanism)

### Visuals / Layout
- **Figure:** [fig3_shmoseq_state_characteristics.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig3_shmoseq_state_characteristics.png) & [fig6_statistical_state_similarity_heatmaps.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/capacity/fig6_statistical_state_similarity_heatmaps.png)
- **Syntax Graph:** Comparing destination transitions out of Syllable 42 in State 6 vs. State 1.

### Speaking Points
1. **What the States Actually Do:**
   - States do not just alter the frequencies of individual syllables; they **re-wire the transition probabilities**.
   - In **State 6 (Normative WT Exploration):** A rearing bout transitions cleanly into forward locomotion and exploratory sniffing.
   - In **State 1 (DAT-KI Stereotypy):** The exact same rearing syllable transitions into repetitive head-bobbing and stationary pawing.
2. **Statistical Distance:**
   - Jensen-Shannon divergence across state-dependent transition matrices ($D_{\text{trans}} > 0.5$) confirms that each macro-state operates as a distinct grammatical mode.

---

## Slide 7: Genotype & Drug Phenotypes — The Loss of Normative Exploration

### Visuals / Layout
- **Figure:** [fig4_shmoseq_genotype_drug_dissection.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig4_shmoseq_genotype_drug_dissection.png)
- **Timeline Strip:** [fig_session_state_timelines_k8.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/timelines/fig_session_state_timelines_k8.png)

### Speaking Points
1. **The 11-Fold Baseline Deficit in DAT-KI:**
   - **State 6** occupies **$48.3\%$** of baseline exploration in Wild-Type mice.
   - In DAT-KI mutants, State 6 is virtually eliminated, dropping to **$4.4\%$** ($p = 0.015$).
   - Instead, DAT-KI mice are trapped in mutant-dominant states $Z_0, Z_1, Z_3$.
2. **The Amphetamine Shock:**
   - Amphetamine causes an acute collapse of the mutant baseline state and drives animals into high-entropy fragmented states.
3. **Model Robustness:**
   - This phenotype is completely preserved when expanding capacity from $K=8$ to $K=12$ (WT = 28.7% vs. KI = 3.5%, matching cross-model overlap).

---

---

## Slide 8: Technical Validation — Ground-Truthing Injection Timing via DLC Morphology

### Visuals / Layout
- **Figure:** [fig_dlc_injection_timing_accuracy.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig_dlc_injection_timing_accuracy.png)
  - **Panel A:** Continuous 95-minute DLC tracking confidence across 27 keypoints with color-coded behavioral epochs (pre-entry, baseline, injection handling, drug response, post-session).
  - **Panel B:** High-resolution zoom ($\pm 90\text{s}$) on physical injection handling gap, comparing raw confidence collapse, morphological presence step-function, and inferred metadata.
  - **Panel C:** Cohort scatter of Inferred vs. Morphologically Detected injection time ($R > 0.99$).
  - **Panel D:** Cohort error distribution showing sub-minute precision (mean $\Delta = -30.7\text{s}$ to midpoint).
  - **Panel E:** Experimental epoch durations (20.0 min baseline, 61.7s handling gap, 59.3 min response) and 40 Hz $\to$ 80 Hz master alignment invariant.

### Speaking Points
1. **The Question of Ground Truth:**
   - How accurate is our inferred `injection_frame`? Could timing errors confound our pre/post-injection behavioral dynamics?
2. **1D Mathematical Morphology on DLC Tracking:**
   - Because the mouse is physically lifted out of the open field for intraperitoneal injection, DLC confidence across all 27 keypoints plummets to zero for ~60 seconds.
   - We applied 1D mathematical morphology (closing with 2.0s structuring element to bridge transient tracking dropouts during presence, opening with 5.0s structuring element on absence to isolate sustained empty-chamber periods).
3. **Rigorous Accuracy Findings (n=30 Sessions):**
   - **Chamber Entry Accuracy:** Metadata `entry_frame` matches physical arena placement to **$+0.62 \pm 0.93\text{ seconds}$** (sub-second precision).
   - **Handling Window Alignment:** Inferred `injection_frame` lands **directly inside the physical handling window** in **$46.7\%$** of sessions, and within $\sim 30\text{ seconds}$ prior to animal removal in the remainder (mean deviation: **$-30.7\text{ seconds}$**).
   - **Epoch Consistency:** Pre-injection baseline is **$20.00 \pm 0.40\text{ minutes}$** (target 20m), physical handling duration is **$61.7 \pm 38.1\text{ seconds}$** (median 52.5s), and post-injection response is **$59.31 \pm 0.38\text{ minutes}$** (target 60m).
4. **Master Timeline Harmonization (40 Hz vs. 80 Hz):**
   - Raw video, photometry, and DLC operate at **80.025 Hz**.
   - Keypoint-MoSeq models downsample by $2\times$ to **40.0125 Hz**.
   - In our master dataset, syllable streams are weaved back to the 80 Hz master timeline via 2-frame forward-fill, ensuring that row indices match native video frames 1-to-1 without risk of scale drift.

---

## Slide 9: Multimodal Ground-Truth — Optical Motion Artifacts During Physical Handling

### Visuals / Layout
- **Figure:** [fig_photometry_handling_cross_validation.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig_photometry_handling_cross_validation.png)
  - **Panel A:** Multimodal zoom on representative session (S0: 1694 Saline, 65.9s handling) comparing tracking confidence drop vs. raw photometry baseline drift and rolling standard deviation burst.
  - **Panel B:** Longest absence case (S24: 1771 Amphetamine, 225.6s handling) showing persistent optical instability throughout the entire 3.76-minute window.
  - **Panel C:** Cohort-wide variance amplification distribution showing 76.7% of sessions exhibit >1.25x variance increase during handling (median 1.77x, mean 22.43x).
  - **Panel D:** Variance amplification across WT Saline, WT Amph, DAT-KI Saline, and DAT-KI Amph boxplots.

### Speaking Points
1. **The Skeptic's Challenge: Is the Empty Chamber a Tracking Glitch?**
   - *"Could the zero-confidence DLC period just be an optical occlusion or computer vision tracking failure rather than physical absence?"*
2. **The Fiber Patch Cable as an Accelerometer:**
   - The mouse is tethered by an optical patch cable. During physical animal pickup, scruffing, intraperitoneal injection, and return, the cable is violently bent, twisted, and shaken.
   - This mechanical strain induces pronounced baseline shifts, optical motion transients, and high rolling standard deviation across optical channels.
3. **Double Verification Across Modalities:**
   - In both standard and extreme sessions, the optical variance surge **switches on at the exact frame of animal removal** and **quenches to resting baseline at the exact frame of animal return**.
   - Cohort-wide, **76.7% of sessions show elevated optical noise** during the handling gap (median $1.77\times$, mean $22.43\times$).
4. **Scientific Rigor:**
   - Optical signal physics provides an independent, non-DLC multimodal proof that our inferred handling window corresponds strictly to physical animal removal from the chamber, confirming our exclusion boundaries.

---

---

## Slide 10: Statistical Rigor — Exact Combinatorial Permutation Testing at the Animal Level

### Visuals / Layout
- **Figure:** [fig_exact_animal_permutation_test.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig_exact_animal_permutation_test.png)
  - **Panel A:** Animal-Level Saline Occupancy (15 individual mice plotted over WT and DAT-KI mean vectors).
  - **Panel B:** Exact Null Permutation Distribution across all 6,435 combinatorial partitions ($d_{L2} = 0.586, p = 0.0099$; $d_{\text{JS}} = 0.561, p = 0.0218$; $d_{\text{TV}} = 0.626, p = 0.0443$).
  - **Panel C & D:** Animal-Level Drug Response Vector ($\Delta = \text{Amph} - \text{Saline}$) and Null Distribution ($p = 0.6239$).
  - **Panel E:** Leave-One-Animal-Out (LOAO) Check across all 15 mice showing descriptive stability ($d_{\text{TV}} \in [0.601, 0.738]$) and sample-specific null distributions.
  - **Panel F:** Per-State Exact Permutation Tests showing State $Z_6$ as the largest difference ($p = 0.0107$, BH $q = 0.0858$).

```
   Exact Animal Permutation Null: Saline Baseline        Leave-One-Animal-Out (LOAO) Check
 Partitions (n=6,435)                               TV Distance (d_TV)
   500 ┼           ┌─┐                                0.8 ┼
   400 ┼       ┌─┐ │ │                                0.7 ┼ ─── Jackknife Range [0.601, 0.738]
   300 ┼     ┌─┤ └─┤ └─┐                              0.6 ┼ ◇ ◇ Sample-Specific 95th Nulls
   200 ┼   ┌─┘         └─┐  Observed TV: 0.626        0.5 ┼
   100 ┼ ┌─┘             │  (Exact p = 0.0443)        0.4 ┼
     0 ┴─┴───────────────┴──▼─────────────            0.0 ┴─────────────────────────────────────
        0.2   0.3   0.4   0.5   0.6   0.7                    All 15 Mice Tested Individually
                    TV Distance                                (Reduced L2 p < 0.05 in 15/15)
```

### Speaking Points
1. **The Core Statistical Challenge — Animal Independence:**
   - *"Do the 8 shMoSeq states represent true genotype differences, or are they an artifact of individual animal quirks?"*
   - In behavioral modeling, frame-shuffling creates massive pseudo-replication. To be statistically valid, **the unit of permutation must be the animal ($N=15$)**, preserving paired within-animal session structure.
2. **Exact Combinatorial Enumeration ($\binom{15}{8} = 6,435$ Partitions):**
   - With 8 WT and 7 DAT-KI mice, there are exactly 6,435 ways to partition the animals. We exhaustively evaluated every single partition.
   - **Baseline Saline Separation:**
     - Euclidean distance: $d_{L2} = 0.5856$, **exact $p = 0.00995$** (only 64 of 6,435 random groupings matched or exceeded observed).
     - Jensen-Shannon distance: $d_{\text{JS}} = 0.5612$, **exact $p = 0.0218$**.
     - Total Variation distance: $d_{\text{TV}} = 0.6257$, **exact $p = 0.0443$**.
   - *Note on Interpretation:* These are three related geometric checks on the same dataset rather than three independent proofs. Together, they confirm that the observed saline occupancy profiles separate WT and KI more than most arbitrary animal groupings do.
3. **Leave-One-Animal-Out (LOAO) Descriptive Stability:**
   - Removing any one animal does not cause the separation to collapse:
     - Full cohort $d_{\text{TV}} = 0.6257$; Jackknife mean: $d_{\text{TV}} = \mathbf{0.6375 \pm 0.0415}$ (range: $0.6011$ to $0.7377$).
     - Re-running the exact permutation test on each reduced 14-animal sample reveals that **Euclidean distance remains significant in 15 out of 15 iterations** ($p \in [0.002, 0.036]$), and TV distance remains at $p < 0.093$ in all 15.
4. **Drug Response vs. Baseline:**
   - The drug response vector ($\Delta_{\text{drug}} = \text{Amph} - \text{Saline}$) yields $p = 0.6239$.
   - This test did not detect a genotype difference in the *occupancy response* to amphetamine (due to individual heterogeneity), confirming that the primary statistical divergence lies in baseline organization.
5. **State $Z_6$ and Multiple Comparisons:**
   - State $Z_6$ shows the largest observed state-specific difference ($+43.9\%$ WT occupancy, raw exact $p = 0.01072$).
   - After Benjamini–Hochberg correction across the 8 states, $q_{Z_6} = 8 \times 0.01072 = 0.0858$, so it **does not survive strict FDR 0.05 correction**.
   - Importantly, the omnibus multivariate test across all 8 states remains statistically significant ($p = 0.0099$).
6. **Defensible Conclusion:**
   - *"State occupancy differs between genotypes at baseline in this exploratory analysis. The separation remains substantial when individual animals are omitted, while a genotype difference in the occupancy response to amphetamine was not detected."*

---

## Slide 12: Ground-Truthing Behavioral Sequences — The seqNMF & Clustering Resolution Tradeoff

### Visuals / Layout
- **Diagram:** Controlled Convolutional seqNMF workflow ($K=8, L=5$ macro-bouts, collapsed-bout rule $A \to A \to A \implies A$).
- **Resolution Spectrum Graphic:** 
  $$\text{Level 1 (100 Raw States)} \longrightarrow \text{Level 2 (93 Conservative)} \longrightarrow \text{Level 3 (22 Preserved Asymmetry)} \longrightarrow \text{Level 4 (14 Broad Families)}$$
- **Summary Matrix Table:**
  | Metric (5-Step Chains) | Level 1 (100 States) | Level 2 (93 States) | Level 3 (22 States) | Level 4 (14 States) |
  | :--- | :---: | :---: | :---: | :---: |
  | **Distinct Observed Strings** | 133,003 | 125,188 | 61,781 | 32,074 |
  | **Recurrent Families (≥20 ep, ≥8 mice)** | **345** | **347** | **942** | **1,524** *(4.4× increase)* |
  | **Population Time Coverage (%)** | **14.1%** *(14.0±7.0%)* | **15.1%** *(15.0±6.0%)* | **46.1%** *(46.2±9.9%)* | **76.2%** *(76.0±10.7%)* *(5.4× structured time)* |
  | **Episode Support (Median [IQR])** | 28 [23–39] | 28 [23–37] | 29 [24–41] | **37 [26–63]** *(heavier upper tail)* |
  | **Cohort Penetration (Mice / 15)** | 10 [9–11] | 10 [9–11] | 11 [9–12] | **12 [10–14]** |
  | **Physical Duration (Mean ± Std)** | 2.58s ± 1.29s | 2.81s ± 1.82s | 3.78s ± 2.99s | **3.63s ± 2.22s** *(captures full behavioral arcs)* |

### Speaking Points
1. **The Core Question:**
   - *"Does this resolution organise more of the observed behaviour into recurring, biologically interpretable families?"*
   - Animals do not experience behavior as static occupancy percentages; they execute **dynamic, multi-second action sequences** (e.g., *sniff wall $\to$ rear $\to$ lateral descent $\to$ sprint*).
2. **Avoiding the "Total Unique Sequences" Fallacy:**
   - Level 1 produces the largest number of distinct token strings (133,003 at 5 steps), but these are overwhelmingly single-occurrence variants. Coarsening vocabulary naturally reduces unique permutations while boosting recurring support.
   - Therefore, the score is not "total unique strings"—it is **population time coverage** and **within-family coherence**.
3. **The 5.4-Fold Population Coverage Jump:**
   - In Level 1, recurrent 5-step sequences cover only **14.1%** of an animal's recording time. 86% of the animal's life consists of sequence permutations that fail to recur robustly across mice.
   - In Level 4, recurrent families cover **76.2%** of all recording time. Over three-quarters of the animal's behavior is organized into recurring, multi-animal behavioral families. At 6 steps, Level 1 collapses to 1.7%, while Level 4 still organizes 46.0% of behavior.
4. **Physical Duration & Intra-Family Continuity:**
   - When syllables are grouped, consecutive identical bouts collapse ($A \to A \to A \implies A$). 
   - Consequently, 5 steps in Level 4 average **3.63 seconds** (and up to 10–15s for grooming/resting cycles), compared to 2.58s in Level 1, capturing complete multi-second behavioral arcs.
5. **The Dual-Lens Ethological Framework: "Paths" vs. "Families":**
   - **Level 1 (Original 100 States) = Behavioral "Paths":** Captures the precise, high-dimensional trajectory through posture space. Essential for inspecting fine-grained biomechanical execution and motor nuances.
   - **Level 4 (14 Broad States) = Behavioral "Families":** Clusters synonymous postures into functional action categories. Uncovers the overarching behavioral syntax and robust multi-step programs shared across mice.
6. **The 4-Tier Systematic Resolution Experiment:**
   - Level 1 (100 Raw), Level 2 (93 Conservative), Level 3 (22 Functional Distinctions), Level 4 (14 Broad Families), evaluated across all 30 sessions with verified hardware chamber mapping `[TL, BL, BR, TR]`.

---

## Slide 13: Video Evidence Across Resolutions — 3-Tier Demonstration (Exemplars, P90, Ordinary)

### Visuals / Layout
- **Video Demonstration Gallery:** Side-by-side comparison of 1×3 widescreen movies across resolutions:
  - **Tier 1 (Canonical Exemplars):** Top density-ratio centroid flanked by nearest-neighbor sister episodes from distinct animals (multimodal 4D PCA trajectory matching).
  - **Tier 2 (P90 Representatives):** Episodes operating near the 90th percentile of continuous movement similarity ($P90 \pm 2\%$) across distinct animals, exposing the high-quality upper distribution.
  - **Tier 3 (Ordinary Random Members):** Seed-controlled random samples from the unpruned statistical population (**without** bottom-20% speed filtering) across distinct animals.
- **Master Video Report:** [CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md) (contains all 36 videos and animated GIFs).
- **P90 Scaling Report:** [P90_SIMILARITY_AND_SAMPLE_SIZE_SCALING.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/P90_SIMILARITY_AND_SAMPLE_SIZE_SCALING.md).

```
   Resolution Comparison across Shared Behavioral Themes:
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │ Theme A: Wall-Rear & Descent Launch (Locomotor Transition)                      │
   │ Theme B: Sprint & Turning Redirection (High-Speed Slalom)                        │
   │ Theme C: Olfactory Inspection & Rest (Stationary Maintenance)                    │
   └─────────────────────────────────────────────────────────────────────────────────┘
   [Level 1: 100 States] ──> [Level 2: 93 States] ──> [Level 3: 22 States] ──> [Level 4: 14 States]
   RelErr: 0.8848            RelErr: 0.8800           RelErr: 0.7891           RelErr: 0.7280
   Theme A: N=62 (11 anim)   Theme A: N=74 (13 anim)  Theme A: N=45 (15 anim)  Theme A: N=455 (15 anim)
```

### Speaking Points
1. **Ground-Truthing Mathematical Sequences in Real Video:**
   - Rather than displaying abstract matrices, we rendered **36 synchronized 1×3 widescreen video experiments** demonstrating exactly how these sequences look in behaving mice across all 4 resolution levels.
2. **The Three Complementary Video Sampling Tiers:**
   - **Tier 1 (Canonical Exemplars):** Identifies the *ideal archetype* of the motif using density-ratio maximization ($1.58\times$ to $2.86\times$ above background). All 3 mice move in biomechanical synchrony across distinct animals and genotypes.
   - **Tier 2 (P90 Representatives):** Evaluates a consistent upper slice of the distribution ($P90 \pm 2\%$). Demonstrates that high-fidelity executions remain cohesive across different mice without being dominated by exceptional replicas.
   - **Tier 3 (Ordinary Random Members):** Reflects the *true population variance*. Mice take wider arcs or vary speeds, yet faithfully execute the identical structural sequence.
3. **What the Video Comparison Reveals:**
   - **Level 1 (Raw):** Visually sharp, but seqNMF factors capture only a fraction of animal sessions ($N=62$) due to state fragmentation.
   - **Level 2 (93 States):** Fixes true synonym splits without losing any geometry ($N=74$, 13 animals).
   - **Level 3 (22 States):** Optimal ethological coherence. Left vs right turns and lateralized descents remain visually and kinematically distinct ($N=45$, all 15 animals).
   - **Level 4 (14 States):** Massive statistical power ($N=455$, all 15 animals), capturing broad behavioral themes across entire cohorts.

---

## Slide 13b: The Power of $N$ — P80/P90 Similarity Scaling & The Level 3 Diagnostic Resolution

### Visuals / Layout
- **Core Figures:**
  - **Matched Action Chains Scaling (Apples-to-Apples):** [fig_matched_action_chains_scaling.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_matched_action_chains_scaling.png)
  - **Repaired Cumulative Scaling Curves (Pooled Reference):** [fig_similarity_vs_qualifying_episodes_repaired.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_similarity_vs_qualifying_episodes_repaired.png)
- **Diagnostic Audit Document:** [CLUSTERING_RESOLUTION_DIAGNOSTIC_AUDIT.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_DIAGNOSTIC_AUDIT.md)
- **Apples-to-Apples Matched Scaling Table (`Rear -> Descent -> Walk`):**
  | Resolution Level | Total Episodes ($N$) | Cohort Animals | P80 Similarity | Sim $\ge 0.70$ ($N$ / Animals) | Sim $\ge 0.75$ ($N$ / Animals) | Sim $\ge 0.80$ ($N$ / Animals) |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Level 1 (100 Raw)** | 2,143 | 15 | 0.620 | 203 (14 mice) | 108 (12 mice) | 30 (9 mice) |
  | **Level 2 (93 Conservative)** | 1,318 | 15 | 0.623 | 144 (12 mice) | 88 (12 mice) | 31 (9 mice) |
  | **Level 3 (22 Functional)** | **2,686** | **15** | **0.637** | **281 (13 mice)** | **147 (12 mice)** | **40 (10 mice)** |
  | **Level 4 (14 Broad)** | 2,381 | 15 | 0.623 | 215 (13 mice) | 111 (12 mice) | 29 (9 mice) |

### Speaking Points
1. **The Discrepancy Resolved: "Why Did Level 3 Show Small Candidate Pools in Earlier Plots?"**
   - The recurrence table measured *all recurrent 5-step families across the entire cohort*, where Level 3 organizes $46.1\%$ of behavior (942 recurrent families vs. 15.1% in Level 2).
   - In contrast, the original similarity plot used **mismatched query lengths**: Levels 1 & 2 queried 4-step sequences, while Level 3 was assigned a **5-step over-constrained query** requiring rigid intermediate transitions (`Descent_L -> Walk_L -> Sprint` and rigid alternating turns).
2. **Strict Apples-to-Apples Matched Query Proves Level 3 Leads:**
   - When querying the exact same 3-step physical grammar (`Rear -> Descent -> Walk`), **Level 3 achieves the highest sample size ($N=2,686$) and the highest number of high-fidelity trials ($281$ at $\text{Sim} \ge 0.70$)** across all 15 animals!
3. **Why P80 Over P90?**
   - P90 can be overly restrictive in smaller candidate pools. The **80th percentile (P80)** captures the consistent upper quintile of natural executions while providing ample trials for multi-animal video evaluation without pausing.
4. **The Recommendation:**
   - **Level 3 (22 States) is the optimal ethological resolution for fine lateralities:** It preserves descent laterality (`Descent_L` vs. `Descent_R`) and head-casting (`Look_L` vs. `Look_R`). For longer chains ($N \ge 5\text{--}6$), Level 4 provides massive $76.2\%$ coverage with minimal kinematic loss ($+2.7\%$).

---

## Slide 14: The Dual-Resolution Framework — Level 4 Behavioral Families + Level 1 Trajectory Paths

### Visuals / Layout
- **Figure / Conceptual Model:** Dual-Stream Architecture:
  - **Macro-Syntax (Level 4, 14 States):** Discovers recurring multi-second **Behavioral Families** (*Rear $\to$ Descent $\to$ Sprint*).
  - **Micro-Paths (Level 1, 100 States):** Resolves the specific postural **Paths** and biomechanical transitions taken through each family.
  - **Continuous Execution (Kinematics):** Measures the continuous **"adverbs"** (velocity, duration, net displacement, turning angle $\Delta\theta$).
- **Leave-One-Animal-Out Cross-Validation Figure:** `fig_kinematic_information_retention_loao.png`
- **Quantitative Trade-Off Table (The Decisive Comparison):**
  | Measure / Evaluation | Level 3 (22 Functional States) | Level 4 (14 Broad States) | Trade-Off & Interpretation |
  | :--- | :---: | :---: | :--- |
  | **Five-step population time coverage** | **46.1%** *(942 families)* | **76.2%** *(1,524 families)* | **$+30.1\%$ coverage gain** ($1.65\times$ more structured time) |
  | **Six-step population time coverage** | **9.4%** *(collapse)* | **46.0%** *(robust)* | **Nearly $5\times$ greater long-chain recurrence** (avoids combinatorial collapse) |
  | **Held-out heading prediction error ($\Delta\theta$)** | **$21.02^\circ \pm 0.96^\circ$** | **$21.58^\circ \pm 1.06^\circ$** | **$+0.56^\circ$ difference** ($p = 6.3 \times 10^{-4}$, LOAO-CV across 15 mice) |
  | **Held-out speed prediction error ($v$)** | **$27.85 \pm 2.32\text{ px/s}$** | **$28.49 \pm 2.46\text{ px/s}$** | **$+0.65\text{ px/s}$ difference** ($p = 1.8 \times 10^{-3}$) |
  | **Held-out duration prediction error ($d$)** | **$0.429 \pm 0.021\text{ s}$** | **$0.434 \pm 0.022\text{ s}$** | **$+0.005\text{ s}$ difference** ($5\text{ ms}$, $p = 6.4 \times 10^{-6}$) |
  | **Held-out displacement prediction error ($\Delta r$)** | **$24.10 \pm 2.63\text{ px}$** | **$24.54 \pm 2.76\text{ px}$** | **$+0.45\text{ px}$ difference** ($p = 1.1 \times 10^{-2}$) |

### Speaking Points
1. **A Useful Trade-Off Rather Than One Universally Correct Resolution:**
   - Choosing the presentation resolution is an ethological and analytical trade-off.
   - **Level 4 buys substantially greater recurrence coverage for small average increases in prediction error on measured bout kinematics.**
   - This fits our primary scientific goal: discovering broad, recognizable behavioral families with meaningful natural variation across animals.
2. **Strict & Grounded Scientific Claim:**
   - **Avoid claiming "only 3% information loss":** These cross-validation tests measure four specific bout-level kinematic summaries (heading change, speed, duration, displacement), not every fine detail of 3D skeletal movement.
   - The honest, accurate statement is: **Broader grouping substantially increases recurrent-sequence coverage, with a modest cost in predicting the measured bout-level kinematics.**
3. **How Level 1 and Level 4 Complement Each Other in the Presentation:**
   - **Level 4 for Behavioral Families (Macro-Grammar):** By grouping synonymous pose variants, Level 4 captures over three-quarters ($76.2\%$) of all behavior into recurring multi-step programs and avoids sequence fragmentation.
   - **Level 1 for Behavioral Paths (Micro-Structure):** We retain all original 100 syllable labels so we can trace the fine-grained postural "paths" mice take through each Level 4 motif.
   - **Continuous Kinematics for Physical Precision:** Annotating each motif instance with continuous parameters ($\Delta\theta$, peak speed, displacement) completely eliminates categorical blur.
4. **When to Retain Level 3:**
   - Level 3 remains valuable for specific hypotheses that require descent direction (`Descent_L` vs. `Descent_R`) or sensory head-casting (`Look_L` vs. `Look_R`), where within-state dispersion is lower. For general cohort sequence discovery, Level 4 is the clear presentation choice.

---

## Slide 15: Grand Summary & Neurobiological Roadmap

### Visuals / Layout
- **Comprehensive Hierarchical Diagram:**
  $$\begin{aligned}
  \text{Layer 1: Sub-Second Primitives} &\quad\longrightarrow\quad \text{KPMS Syllables (~300 ms, Posture \& Pose)} \\
  \text{Layer 2: Multi-Second Grammar} &\quad\longrightarrow\quad \text{seqNMF Sequential Motifs (~2–5 s, Behavioral Syntax)} \\
  \text{Layer 3: Continuous Execution} &\quad\longrightarrow\quad \text{Kinematic Metrics (Velocity, Net Displacement, Turning Angle } \Delta\theta\text{)} \\
  \text{Layer 4: Dopaminergic State} &\quad\longrightarrow\quad \text{Fiber Photometry Transients \& Tonic Baselines}
  \end{aligned}$$

### Key Takeaways for the Audience
1. **Non-Markovian Structure is Real:** Behavioral memory peaks at **$\sim 6.3\text{ seconds}$**, proving that mice operate via structured, multi-second action programs rather than memoryless coin-tosses.
2. **Robust Genotype Differences:** Baseline state occupancy separates WT from DAT-KI across exact animal-level combinatorial permutations ($p = 0.0099$), stable under leave-one-out Jackknife testing.
3. **Empirical Sequence Validation:** 36 video experiments across 3 tiers (Canonical Exemplars, P90 Representatives, and Ordinary Population Members) prove that discovered motifs represent genuine physical behaviors preserved across mice, validated with verified hardware chamber mapping.
4. **Discrete Syntax + Continuous Kinematics:** The combination of functional state clustering and continuous kinematic annotation resolves the fragmentation vs. blurriness dilemma.
5. **Immediate Next Step:** Aligning multi-second sequence motif onset and completion boundaries with **striatal dopamine release transients** to reveal how dopamine orchestrates the execution of behavioral programs.

---

### Linked Artifacts & Tables
- **Master 3-Tier Video Comparison Report (36 Movies):** [CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md)
- **P90 Similarity & Sample-Size Scaling Report:** [P90_SIMILARITY_AND_SAMPLE_SIZE_SCALING.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/P90_SIMILARITY_AND_SAMPLE_SIZE_SCALING.md)
- **Cumulative Movement Similarity Scaling Curves Figure:** [fig_similarity_vs_qualifying_episodes.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/fig_similarity_vs_qualifying_episodes.png)
- **P90 Similarity Scaling Summary CSV:** [p90_similarity_scaling_summary.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/p90_similarity_scaling_summary.csv)
- **Population Coverage & Duration Statistics CSV:** [clustering_rigorous_coverage_statistics.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/clustering_rigorous_coverage_statistics.csv)
- **N-Step Sequence Discovery Statistics CSV:** [n_step_sequence_statistics.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/n_step_sequence_statistics.csv)
- **Comparison Metadata JSON:** [clustering_comparison_metadata.json](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/clustering_comparison_metadata.json)
- **Comparison Summary CSV:** [clustering_comparison_summary.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/clustering_comparison_summary.csv)
- **Whole-Chain Exemplar Motifs Report:** [WHOLE_CHAIN_EXEMPLAR_MACRO_MOTIFS.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/clustered_seqnmf/WHOLE_CHAIN_EXEMPLAR_MACRO_MOTIFS.md)
- **Exact Animal Permutation Figure:** [fig_exact_animal_permutation_test.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig_exact_animal_permutation_test.png)
- **Omnibus Permutation Summary CSV:** [exact_permutation_omnibus_summary.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/tables/exact_permutation_omnibus_summary.csv)
- **Leave-One-Animal-Out Robustness CSV:** [leave_one_animal_out_robustness.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/tables/leave_one_animal_out_robustness.csv)
- **Per-State Permutation Tests CSV:** [per_state_exact_permutation_tests.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/tables/per_state_exact_permutation_tests.csv)
- **State Episode Video Audit CSV:** [state_episode_video_audit_table.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/tables/state_episode_video_audit_table.csv)
- **Photometry Handling Cross-Validation Figure:** [fig_photometry_handling_cross_validation.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig_photometry_handling_cross_validation.png)
- **DLC Injection Timing & Accuracy Figure:** [fig_dlc_injection_timing_accuracy.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig_dlc_injection_timing_accuracy.png)
- **Handling Gaps & Case Studies Figure:** [fig_handling_gaps_and_case_studies.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig_handling_gaps_and_case_studies.png)
- **Verified Event Frames Table:** [session_event_frames_verified.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/tables/session_event_frames_verified.csv)
- **Group Timescale Diagnostic Figure:** [fig1_timescale_by_group_comparison.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig1_timescale_by_group_comparison.png)
- **Baseline-Subtracted Timescale Figure:** [fig1_real_vs_markov_baseline_removed.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig1_real_vs_markov_baseline_removed.png)
- **State vs. Time Decomposition Figure:** [fig1_timescale_state_vs_time_decomposition.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/shmoseq/output/figures/fig1_timescale_state_vs_time_decomposition.png)



