# The Behavioral Chain Anomaly Atlas: A Breadth-First Framework for Neuroethology & Movement Disorders

**Document Path:** `docs/SEQNMF_CHAIN_ANOMALY_ATLAS_FRAMEWORK.md`  
**Purpose:** Foundational specification for discovering, quantifying, and validating multi-timescale behavioral sequence anomalies using `seqNMF`, multi-view tracking (TopCam + BotCam), striatal fiber photometry, and ethologist-in-the-loop review.  
**Target Application:** Longitudinal $\alpha$-Synuclein PFF Parkinson's Disease progression (`3w_PI` $\to$ `24w_PI`) and L-DOPA therapeutic pharmacodynamics.

---

## 1. Executive Philosophy: The Breadth-First Strategy

### 1.1 Moving Beyond Averages to Behavioral Sequences
Traditional behavioral assays reduce complex longitudinal recordings to summary statistics: mean velocity, distance traveled, or time spent in the center. Even standard Keypoint-MoSeq often stops at global syllable usage counts or flat first-order transition matrices.

However, motor programs in vertebrates are organized hierarchically:
$$\text{Sub-second Syllable } (200\text{--}400\text{ ms}) \longrightarrow \text{Action Chain / Motif } (1\text{--}3\text{ s}) \longrightarrow \text{Macro-Behavioral State } (10\text{--}60\text{ s})$$

Nigrostriatal dopamine depletion in Parkinson's Disease disrupts **motor chunking, action switching, and sequence stamina**. To uncover this pathology, we employ a **Breadth-First Search (BFS) Anomaly Atlas**:
1. Use unsupervised convolutional matrix factorization (**`seqNMF`**) to discover recurring action motifs ($W_k, H_k$) across the entire multi-session cohort without imposing rigid human boundaries.
2. Build an **Occurrence-Level Database** where every single detected instance of a chain is systematically scored across dozens of potential structural, kinematic, contextual, and neural anomalies.
3. Automatically render **curated video galleries** comparing canonical vs. anomalous executions for expert ethologists.
4. Use ethologist validation to triage computational patterns, discarding tracking artifacts and "zooming in" with targeted quantitative hypothesis tests.

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Unsupervised Motif Discovery (seqNMF)                               │
│    Syllable Matrix X (S x T) ──► Templates W (S x K x L) + Activations H│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Breadth-First Anomaly Extraction                                    │
│    Construct Master Occurrence Table (Every chain instance as 1 row)   │
│    - Intra-Chain: Dropouts, loops, reversals, sequence effect, paws    │
│    - Inter-Chain: Perseveration, gap latencies, abnormal branching     │
│    - Contextual: Predecessor bias, macro-states, spatial arena zones   │
│    - Photometry: Striatal dLight transients (start, join, abort)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Automated Video Gallery Generation                                  │
│    Export synchronized 3-panel clips (TopCam + BotCam + Trajectory)    │
│    Contrasting Top-Ranked Canonical vs. Top-Ranked Anomalous Bouts     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Ethologist-in-the-Loop Validation & "Zooming In"                    │
│    Ethologists triage: Real motor breakdown vs. variant vs. artifact   │
│    Downstream targeted modeling (LMMs, survival curves, causality)     │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Bidirectional Scanning Principle
Anomalies must be evaluated in **both directions**:
* **Hypo-Motor / Fragmentation:** Premature aborts, phase dropouts, hesitation pauses, loss of sustained locomotion, and decreased chain initiation.
* **Hyper-Motor / Rigidity:** Excessive repetition ($A \to A \to A$), pathological stereotypy, perseveration loops, inability to exit a state, or drug-induced dyskinesias.

Neither direction is inherently "healthier": wildtype behavior is flexible, whereas both fragmentation and excessive rigidity indicate circuit dysfunction.

---

## 2. The 5 Tiers of Chain & Syllable Anomalies

The framework applies a **fractal, multi-timescale hierarchy**: just as behavioral chains are composed of discrete syllables, syllables themselves are continuous movement trajectories composed of individual video frames ($15\text{--}24$ frames @ 60/80 FPS). 

Every detected behavioral event is evaluated across 5 taxonomic tiers:

```
                          ANOMALY ATLAS MULTI-SCALE TAXONOMY
                                          │
    ┌───────────────────┬─────────────────┼───────────────────┬───────────────────┐
    ▼                   ▼                 ▼                   ▼                   ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│    TIER 0    │ │    TIER 1    │ │    TIER 2    │ │    TIER 3    │ │    TIER 4    │
│ Sub-Syllable │ │ Intra-Chain  │ │ Inter-Chain  │ │  Contextual  │ │    Neural    │
│ Micro-Motion │ │ (Within)     │ │ (Between)    │ │   & Mixed    │ │ (Photometry) │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

---

### Tier 0: Sub-Syllable Micro-Anomalies (Syllable as Motif, Frames as Pieces)

> [!IMPORTANT]
> **The Granularity Paradox of KPMS ($z_t$ vs. $x_t$ vs. Keypoints):**  
> Keypoint-MoSeq assigns a single discrete integer label $z_t \in \{0, \dots, 99\}$ to a $200\text{--}300$ ms behavior. A Parkinsonian mouse executing Syllable 5 (e.g. forward right turn) will still be classified as Syllable 5 as long as its gross body posture fits that state. However, **early neurodegeneration notoriously produces sub-clinical micro-deviations** within an action before changing which action is selected.
>
> Fortunately, Keypoint-MoSeq explicitly separates discrete states $z_t$ from **continuous pose latents $x_t \in \mathbb{R}^{10}$** and reconstructed keypoints. By zooming into the frame-by-frame trajectory ($15\text{--}24$ frames), we capture sub-syllable pathology.

| Sub-Syllable Anomaly | Operational Metric | What It Reveals in Parkinson's (PFF) |
| :--- | :--- | :--- |
| **Hypokinetic Paw Excursion** | Peak-to-trough displacement $\|Y_{\text{paw}}(t) - Y_{\text{paw}}(0)\|$ across syllable frames. | **Hypometria:** Preserved overall movement selection, but smaller physical paw reach/stride amplitude. |
| **Asymmetric Velocity Envelope** | Time-to-peak speed $t_{\text{peak}} / T_{\text{duration}}$, acceleration rise, deceleration skew. | **Kinematic Envelope Distortion:** Slower acceleration rise and prolonged braking phase hidden by average speed. |
| **Mid-Syllable Micro-Pause** | Zero-velocity plateau ($v < 2$ px/s) lasting 2–4 frames mid-syllable. | **Micro-Interruption:** Transient motor arrest/hesitation too brief to trigger a separate discrete KPMS state. |
| **Bilateral Paw Phase Asymmetry** | Temporal lag between left and right forepaw/hindpaw strike peaks ($\Delta t_{\text{paw}}$). | **Sub-Clinical Limp / Ataxia:** Interlimb coordination breakdown with little or no change in gross centroid displacement. |
| **Trajectory Dispersion / Entropy** | Cross-instance variance $\text{Tr}(\text{Cov}(x_t))$ of continuous latents $x_t$ around syllable mean trajectory $\bar{x}(t)$. | **Loss of Motor Stereotypy:** Deterioration of fluid, repeatable movement trajectories into high-variance motor jitter. |
| **Abnormal Boundary Approach** | Trajectory curvature and angular acceleration in the final 3 frames before syllable exit. | **Junction Entry Deficit:** Animal approaches the subsequent action from a distorted posture, destabilizing the next transition. |

#### Two Complementary Analytical Strategies:
1. **Aligned Syllable Trajectory Profiling:**
   - Group all instances of a specific syllable (e.g., Syllable 5, $N \approx 5,000$ occurrences).
   - Align instances at syllable onset ($t=0$).
   - **Crucial Rule:** Keep **duration** and **movement amplitude** as independent measurements. Do *not* automatically time-normalize trajectories, as stretching time conceals genuine bradykinesia and micro-pauses!
2. **Frame-Level `seqNMF` (Sub-Syllable Variant Discovery):**
   - Feed frame-resolved, non-negative kinematic features (paw displacement magnitudes, interlimb joint angles, speed components, ground contact) into a high-resolution `seqNMF` model ($L = 10\text{--}18$ frames).
   - *Discovery Goal:* Reveal whether a single nominal KPMS syllable contains **two reproducible physical variants**—e.g., a fluid "canonical variant" prevalent in Monomers vs. an "interrupted/hypokinetic variant" that progressively dominates in PFF mice.
   - *Sampling Reality:* At 60 FPS, a 250 ms syllable spans ~15 frames (at 80 FPS, ~20 frames). This is sufficient for cubic spline fitting and profile extraction, provided robust confidence filtering is applied to prevent DLC jitter from being misclassified as micro-pauses.

---

### Tier 1: Intra-Chain Anomalies (Inside the Motif)
These metrics assess the structural integrity, execution tempo, and biomechanical coordination of a single motif instance from start to finish.

| Anomaly Type | Formal Definition | Operational Metric | Parkinson's Hypothesis |
| :--- | :--- | :--- | :--- |
| **Premature Abort** | Motif initiates but terminates before reaching terminal lag $L$. | $\text{Reached Lag} < L - 1$; transition into pause/freeze. | Inability of striatal circuits to maintain ongoing motor chunk. |
| **Phase Dropout / Omission** | Intermediate canonical syllable skipped (e.g. $A \to C$ instead of $A \to B \to C$). | Absence of template peak at lag $l$ with successful transition to lag $l+1$. | Striatal signal dropouts during multi-step sequence chaining. |
| **Return Loop / Restart** | Sequence regresses to an earlier phase (e.g. $A \to B \to A \to B$). | Negative index transition within motif bounds. | Motor indecision, hesitation, or failure to clear sub-goal. |
| **Intruder Insertion** | Non-template syllable inserted between canonical phases. | Syllable $S_{\text{intruder}}$ observed with duration $\ge 1$ frame within motif. | Decreased action boundary suppression in basal ganglia. |
| **Phase Repetition (Stutter)** | Same intermediate syllable repeated multiple times ($B \to B \to B$). | Syllable dwell time at phase $l$ exceeds $3\times$ median template duration. | Action release delay / akinesia within sequence. |
| **Junction Latency / Micro-Pause** | Deceleration and hesitation at the boundary between phases. | Centroid speed nadir and duration in $[-100\text{ ms}, +100\text{ ms}]$ join window. | Impaired start/stop boundary signaling in nigrostriatal circuits. |
| **Sequence Effect (Intra-Bout Decay)** | Progressive reduction in speed, amplitude, or stride length across the motif. | Negative slope $\frac{d(\text{speed})}{dt}$ or $\frac{d(\text{stride})}{dt}$ from start to end of chain. | Classic human Parkinsonian micrographia/hypometria analog. |
| **Ventral Paw Desynchronization** | Loss of bilateral gait symmetry or paw slippage during motif. | Phase difference $\Delta \phi(\text{LF}, \text{RF})$ and paw speed variance from BotCam. | Sub-clinical ataxia, limb dragging, or gait instability. |

---

### Tier 2: Inter-Chain Anomalies (Between Motifs)
These metrics evaluate how distinct motifs are chained, repeated, or switched across time.

| Anomaly Type | Formal Definition | Operational Metric | Parkinson's Hypothesis |
| :--- | :--- | :--- | :--- |
| **Perseveration (Looping Runs)** | Unbroken back-to-back repetitions of the same motif ($A \to A \to A$). | Run length $N_{\text{run}}$ of consecutive factor activations in $H_k(t)$. | Pathological motor stereotypy, compulsive repetition, or dyskinesia. |
| **Canonical Transition Loss** | Normal sequence transitions disappear ($P(\text{Chain B} \mid \text{Chain A}) \to 0$). | Off-diagonal probability collapse in chain-to-chain transition matrix. | Syntax fragmentation; breakdown of higher-order grammar. |
| **Anomalous Switching** | Emergence of bizarre or rare chain-to-chain transitions. | Probability surge in historically near-zero transition pairs. | Loss of basal ganglia filtering; random action switching. |
| **Burst Restarts** | Rapid abort-and-restart cycles within a brief temporal window ($<3\text{ s}$). | Frequency of identical chain onsets separated by $<1.5\text{ s}$ gaps. | Repeated failed attempts to launch sustained motor program. |
| **Inter-Chain Hesitation Gaps** | Prolonged pauses between completed action chains. | Inactive dwell time (speed $<5$ px/s) between motif offset and next onset. | General initiation impairment (akinesia / freezing). |
| **Repertoire Entropy Collapse** | Animal relies on a tiny subset of chains, abandoning rich repertoire. | Chain participation ratio $N_{\text{eff}} = \frac{1}{\sum p_k^2}$ and Shannon entropy $H(\text{Chain})$. | Progressive behavioral narrowing with disease progression. |

---

### Tier 3: Contextual & Mixed Anomalies (State, Environment & Progression)
The meaning of an action sequence depends heavily on the animal's global behavioral state, physical arena location, and disease stage.

```
                      CONTEXTUAL MODULATION LAYERS
                      
┌────────────────────────────────────────────────────────────────────────┐
│ Longitudinal Axis:   3w PI (Early) ──► 8w PI ──► 16w PI ──► 24w PI     │
│ Pharmacology Axis:   Baseline Saline  ◄──────────────►  L-DOPA Rescue  │
│ Macro-State Axis:    Sustained Run   ◄──────────────►  Quiet Inactive  │
│ Spatial Zone Axis:   Center Open     ◄──────────────►  Corner Thigmo   │
└────────────────────────────────────────────────────────────────────────┘
```

* **Predecessor History Dependence:** Does Chain A execute normally when preceded by Locomotion ($X \to A$), but collapse when preceded by Grooming ($Y \to A$)?
* **Macro-Regime State Coupling (`shMoSeq`):** Does the motif fail specifically during active exploratory states, while remaining intact during self-directed maintenance?
* **Spatial Geography:** Are chain aborts and micro-pauses concentrated in the open arena center (elevated threat/anxiety) versus along the protective perimeter walls?
* **Within-Session Fatigue:** Does chain completion rate or stride length decay systematically across the 40-minute recording session (fatigue curve)?
* **Longitudinal Divergence:** At what exact wave (`3w`, `8w`, or `16w`) does each specific anomaly reach statistical significance, and does L-DOPA normalize it?

---

### Tier 4: Neural & Dopamine Coupling Anomalies
Every session in the Karen cohort incorporates time-locked striatal **dLight1.3b** fiber photometry ($\Delta F/F$, RANSAC z-score).

```
DOPAMINE COUPLING AT CHAIN JUNCTIONS

Normal Traversal:      [Phase 1] ───(DA Burst)───► [Phase 2] ───(DA Burst)───► [Phase 3]
                                         ▲
                                         │ Green-Light Continuation Signal
                                         
Premature Abort:       [Phase 1] ───(DA Dip)────► [ABORT / FREEZE]
                                         ▲
                                         │ Dopamine Failure / Clearance Failure
```

1. **Continuation vs. Abort Divergence:**
   * Align dLight traces to the onset of the *same* intermediate syllable (e.g. accelerating sprint) when the chain is successfully completed vs. when it abruptly terminates.
   * *Hypothesis:* Transient striatal dopamine surges provide a physiological "green light" to execute subsequent phases; dopamine terminal loss causes premature aborts.
2. **Phase Boundary Start/Stop Signals:**
   * Measure phasic dopamine transients at chain initiation ($t=0$) and termination ($t=T$).
3. **L-DOPA Pharmacodynamic Uncoupling:**
   * Test whether L-DOPA restores the discrete, phasic event-related dopamine modulation, or merely produces an elevated tonic baseline that obscures sub-second action signaling while driving hyper-stereotypy.

---

## 3. Data Architecture: The Master Occurrence Table

To make this analysis computable and searchable, every detected chain instance across all 110 sessions will be populated into a **Master Occurrence Table** (`chain_occurrences_master.parquet`):

```python
# Conceptual Schema for Master Occurrence Table (1 row = 1 detected chain instance)
schema = {
    # 1. Identification & Metadata
    "occurrence_id": str,          # e.g., "OCC_2026-03-05T11_01_01_Arena1_C1_0042"
    "session_id": str,             # Unique session key
    "animal_id": str,              # Mouse ID (for repeated-measures grouping)
    "genotype": str,               # "PFF" vs. "Monomer"
    "condition": str,              # "3w_PI", "8w_PI", "16w_PI", "24w_PI", "L_dopa"
    "arena_chamber": str,          # "Arena1", "Arena2", "Arena3"
    
    # 2. Motif Architecture
    "chain_id": int,               # seqNMF motif ID (e.g., Chain 1 to 10)
    "start_frame": int,            # Video frame index @ 60/80 Hz
    "end_frame": int,              # Video frame index
    "duration_ms": float,          # Wall-clock duration of instance
    "seqnmf_h_peak": float,        # Peak activation amplitude in H_k(t)
    "reconstruction_error": float, # Reconstruction residual against template W_k
    
    # 3. Tier 1: Intra-Chain Metrics
    "completion_status": str,      # "COMPLETE", "ABORTED", "VARIANT", "LOOPED"
    "max_phase_reached": int,      # Index of furthest lag traversed
    "phase_sequence": list[int],   # Realized syllable trajectory, e.g. [14, 4, 3]
    "omitted_phases": list[int],   # Missing canonical phases
    "inserted_syllables": list[int],# Unexpected intruder syllables
    "return_loop_detected": bool,  # True if trajectory moved backward
    "join_latency_mean_ms": float, # Mean pause duration at phase transitions
    "velocity_start_px_s": float,  # Centroid speed at phase 0
    "velocity_end_px_s": float,    # Centroid speed at final phase
    "velocity_slope": float,       # Intra-bout acceleration/deceleration slope
    "paw_cadence_hz": float,       # Ventral paw strike frequency from BotCam
    "paw_duty_factor": float,      # Stance phase fraction from BotCam
    
    # 4. Tier 2: Inter-Chain Metrics
    "preceding_chain_id": int,     # ID of prior chain (or -1 if isolated)
    "preceding_gap_ms": float,     # Inactive gap duration before start
    "following_chain_id": int,     # ID of next chain
    "is_immediate_repeat": bool,   # True if preceding_chain_id == chain_id (A -> A)
    "repetition_run_length": int,  # Current position in A -> A -> A run
    
    # 5. Tier 3: Contextual & Spatial
    "arena_zone": str,             # "CENTER", "WALL", "CORNER"
    "distance_to_wall_px": float,  # Centroid distance to nearest perimeter
    "session_elapsed_pct": float,  # 0.0 to 100.0% through recording
    "macro_state_id": int,         # Inferred shMoSeq macro-state
    
    # 6. Tier 4: Photometry Integration
    "dlight_baseline_z": float,    # Mean z-scored dLight in [-500ms, 0ms] window
    "dlight_peak_onset_z": float,  # Peak transient in [0ms, +300ms] window
    "dlight_join_transient_z": list[float], # DA amplitude at each phase boundary
    "dlight_terminal_z": float,    # DA amplitude at completion / abort frame
    
    # 7. Media & Validation Links
    "video_clip_path": str,        # Relative path to rendered 3-panel MP4
    "ethologist_validation": str,  # "CONFIRMED_DEFICIT", "VARIANT", "ARTIFACT", "UNREVIEWED"
    "ethologist_notes": str        # Expert qualitative commentary
}
```

---

## 4. The Ethologist-in-the-Loop Human-Machine Workflow

To ensure that machine learning findings correspond to real mammalian biology, we integrate an active human-in-the-loop review pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Compute Anomaly Outlier Scores across all occurrences          │
│         - Statistical Z-scores on join latency, velocity decay, etc.   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Render 3-Panel Synchronized Exemplar Clips (100% Automated)    │
│         Panel A: TopCam (Dorsal Spine & Heading)                       │
│         Panel B: BotCam (Ventral Paws, Stride, Glass Viewport)         │
│         Panel C: 2D Body Centroid Trajectory + Syllable Phase Timeline │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Generate Ranked Comparative Review Galleries                   │
│         Show 10 Top Canonical Examples vs. 10 Top Anomalous Examples   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Expert Ethologist Web/CSV Triage Interface                     │
│         Labels: [True Deficit] [Biological Variant] [Tracking Noise]   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Zoom-In Quantitative Hypothesis Testing                        │
│         Fit repeated-measures LMMs & survival curves on validated data │
└────────────────────────────────────────────────────────────────────────┘
```

### Video Clip Standards:
* **Duration:** Chain duration + $1.0\text{ s}$ pre-onset and post-offset padding.
* **Layout:** $1920 \times 1080$ side-by-side or stacked layout:
  * Left: Overhead TopCam with dorsal skeleton overlay.
  * Center: Zoomed BotCam with paw color-coding (Left Fore: Cyan, Right Fore: Blue, Left Hind: Orange, Right Hind: Red).
  * Right: Dynamic phase trajectory graph, instantaneous velocity, and aligned dLight signal trace with a moving playhead needle.

---

## 5. From Anomaly Atlas to Targeted Hypotheses: "Zooming In"

Once the anomaly atlas has cataloged the entire cohort, statistical "zooming in" focuses on four high-priority, biologically definitive research questions:

### Project A: The Locomotor Sequence Effect (Stride Decay & Early Compensation)
* **Atlas Filter:** Extract all occurrences of Locomotor Chains (e.g. Chain 8 & Chain 5).
* **Targeted Analysis:**
  * Fit a Linear Mixed Model: $\text{Speed Decay Slope} \sim \text{Genotype} \times \text{Condition} + (1 \mid \text{Animal})$.
  * Contrast stride length vs. paw cadence at `3w_PI` vs. `16w_PI`: does cadence increase in early disease as a compensatory mechanism to rescue speed?

### Project B: Transition Hesitation & Akinesia at Branching Junctions
* **Atlas Filter:** Extract phase transition latencies across all chains.
* **Targeted Analysis:**
  * Categorize transitions into **Low Entropy (Deterministic)** vs. **High Entropy (Branching)**.
  * Test whether PFF-associated hesitation latency is selectively exacerbated at high-entropy decision junctions, representing impaired action selection.

### Project C: Striatal Dopamine as an Action Continuation Gate
* **Atlas Filter:** Match identical syllable tokens occurring in Completed vs. Aborted chains.
* **Targeted Analysis:**
  * Run Receiver Operating Characteristic (ROC) analysis: does the pre-onset dLight amplitude predict sequence completion?
  * Assess whether the Area Under the Curve (AUC) drops significantly across longitudinal PFF progression and is restored by L-DOPA.

### Project D: Syntactic Grooming Chain Integrity
* **Atlas Filter:** Extract Grooming Chains (Cephalocaudal phases: Paws $\to$ Face $\to$ Head $\to$ Body).
* **Targeted Analysis:**
  * Construct a semi-Markov phase survival model: calculate hazard rates for phase dropouts and abnormal return loops ($B \to A$).
  * Compare empirical sequence entropy against duration-preserving Markov null models.

---

## 6. Implementation Roadmap & Milestones

1. **Milestone 1 (Immediate):**
   * Finalize the pending 23 BotCam inference videos (completing the 141-video production queue).
   * Run cohort fusion ([`fuse_multiview_cohort.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/fuse_multiview_cohort.py)) to generate full 23-keypoint dataset.
2. **Milestone 2:**
   * Fit full 23-keypoint Keypoint-MoSeq model across all sessions.
   * Extract recurring motifs using `seqNMF` ($K=10\text{--}15$ factors, $L=3\text{--}5$ lag bouts).
3. **Milestone 3:**
   * Populate the Master Occurrence Table with Tier 1–4 metrics across all sessions.
   * Generate initial 3-panel video review galleries for ethologist validation.
4. **Milestone 4:**
   * Execute targeted LMMs and publish definitive manuscript figures connecting multi-camera biomechanics, behavioral syntax, and striatal dopamine dynamics.
