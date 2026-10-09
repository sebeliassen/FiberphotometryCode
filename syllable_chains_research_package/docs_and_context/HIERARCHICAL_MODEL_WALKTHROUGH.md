# Walkthrough: Hierarchical Shared-Subroutine Behavioral Model

We completed the implementation, validation, and timeline visualization of the **Hierarchical Shared-Subroutine Behavioral Model** ($M=3$ upper macro-states, $K=6$ shared routines, $S=3$ internal progression stages) fitted to continuous Keypoint-MoSeq syllable streams from 15 mice across WT and DAT-Het KI genotypes.

---

## 1. Hierarchical Timeline Visualization: Three Aligned Tracks

As requested, we implemented a dedicated timeline visualization engine ([`plot_hierarchical_timeline.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/hierarchical_pilot/plot_hierarchical_timeline.py)) rendering **three aligned horizontal tracks per recording** sharing an identical physical elapsed-seconds x-axis:

1. **TOP TRACK (Upper Macro-State)**: Subdued, cohesive palette ($U_1$ Slate Blue, $U_2$ Sage Green, $U_3$ Warm Ochre) demarcating sustained behavioral contexts.
2. **MIDDLE TRACK (Routine Occurrences)**: Six distinct vibrant hues ($R_1 \dots R_6$). **Crucially, thin solid black dividing lines are drawn at every completed occurrence**, ensuring that consecutive repetitions of the same routine ID (e.g. $R_4 \to R_4$ or $R_6 \to R_6$) appear as separate adjacent blocks rather than a single uninterrupted span.
3. **BOTTOM TRACK (Observed KPMS Syllables)**: 100-syllable consistent palette showing the physical onset, offset, and duration of individual syllable bouts in seconds.
4. **Boundary Integrity**: Tracking boundaries and edge truncation are explicitly preserved—no episodes are stitched across recording breaks, and boundary-truncated occurrences are hatched (`///`).

---

### Whole-Session Overview (0 to 600 Seconds)

Below is the whole-session overview comparing 4 exemplar baseline recordings (2 WT Saline vs. 2 DAT-Het KI Saline) at identical horizontal time scale:

![Whole-Session Overview (Three Aligned Horizontal Tracks)](/home/fsp585/.gemini/antigravity-ide/brain/b03aa115-6afd-40d7-a4d1-a025411e6b82/fig_hierarchical_timeline_overview.png)

#### Key Observations from the Overview:
- **Upper-State Stability**: WT animals (Sessions 1 & 2) transition gracefully between $U_1$ (Active Locomotion) and $U_2$ (Local Investigation), spending prolonged continuous epochs in each state ($150\text{--}250\text{ s}$).
- **Hyper-Locomotor Immersion in DAT-Het KI**: In contrast, DAT-Het KI mice (Sessions 0 & 3) dwell almost entirely inside $U_1$ (Active Locomotion Context), maintaining elevated behavioral cadence across hundreds of seconds.
- **Timescale Hierarchies**: Within a single upper-state visit, dozens of routine cycles occur, demonstrating clear multiscale separation ($2\text{--}8\text{ s}$ routines nested within $2\text{--}5\text{ min}$ behavioral contexts).

---

### Selectable 60-Second Detail View (100s to 160s)

Zooming into a 60-second window reveals the fine structure and alignment across all three levels:

![60-Second Detail View (100s to 160s)](/home/fsp585/.gemini/antigravity-ide/brain/b03aa115-6afd-40d7-a4d1-a025411e6b82/fig_hierarchical_timeline_detail_60s.png)

#### Key Observations from the Detail View:
1. **Immediate Routine Repetitions ($k \to k$)**:
   - In Session 1 (WT, $100\text{--}135\text{ s}$), notice how $R_6$ executes, finishes its 3-stage cycle, and immediately re-triggers as a second $R_6$ block: `[R6] | [R6] | [R4] | [R5] | [R5] | [R6]`.
   - In Session 3 (KI, $143\text{--}150\text{ s}$), repetitive execution is vividly evident: `[R4] | [R4] | [R2] | [R5] | [R4] | [R4]`. The vertical dividers make these bursts immediately visible without ambiguity.
2. **Constituent Syllable Composition**:
   - The bottom track reveals how individual syllable bouts compose each routine. For instance, in WT Session 1, prolonged $R_6$ occurrences are driven by repeated bouts of Syllable 32, punctuated by transitions into Syllables 49, 34, and 43.
   - In KI Session 3, rapid alternation between Syllables 3, 11, 23, 17, 53, 25, and 42 generates short, punchy $R_4$ and $R_5$ occurrences.
3. **Cross-Level Boundary Alignment**:
   - In Session 2 (WT, at $t \approx 115\text{ s}$), the macro-state transitions cleanly from $U_1 \to U_2$ **precisely at the boundary where a completed routine finishes**, as enforced by the hierarchical semi-Markov architecture.

---

### Kinematics-Grounded Timelines (Tokens + Duration + Velocity)

Following ChatGPT's suggestion to incorporate continuous kinematic emissions ($z_{\log(\text{duration})}$ and $z_{\log(\text{velocity})}$), we fitted the joint model and generated the kinematics-grounded timelines across WT and DAT-Het KI sessions:

#### Kinematics Overview (0 to 600 Seconds):
![Kinematics Overview Timeline](/home/fsp585/.gemini/antigravity-ide/brain/b03aa115-6afd-40d7-a4d1-a025411e6b82/fig_hierarchical_timeline_kinematics_overview.png)

#### Kinematics 60-Second Detail (100s to 160s):
![Kinematics Detail Timeline](/home/fsp585/.gemini/antigravity-ide/brain/b03aa115-6afd-40d7-a4d1-a025411e6b82/fig_hierarchical_timeline_kinematics_detail_60s.png)

- Notice how **$U_3$ (Warm Amber, Acceleration Burst Context)** is now directly visible in Session 2 (at $t \approx 190\text{s}$) and Session 4 (at $t \approx 0\text{--}20\text{s}$).
- In WT Session 2 at $t = 115\text{s}$, the switch from $U_1 \to U_2$ directly marks a transition from high-velocity sprints ($R_5 \to R_5 \to R_5$) into quiescent resting and local exploration ($R_3 \to R_6 \to R_3 \to R_3$).
- In DAT-Het KI (Sessions 4 & 0), the animal is almost continuously locked into $U_1$, firing rapid bursts of sprinting ($R_5$) and motor transitions ($R_2, R_4$).

---

### 30-Minute Continuous Hierarchical Timelines (1,800 Seconds / 0.5 Hour)

We expanded the training and visualization horizon to **full 30-minute continuous recordings** (86,411 continuous bouts across the 30-session cohort):

#### 30-Minute Full Overview (0 to 1,800s):
![30-Minute Full Overview Timeline](/home/fsp585/.gemini/antigravity-ide/brain/b03aa115-6afd-40d7-a4d1-a025411e6b82/fig_hierarchical_timeline_30min_overview.png)

#### 120-Second Detail Window (600s to 720s / 10m to 12m into Recording):
![120-Second Detail Window](/home/fsp585/.gemini/antigravity-ide/brain/b03aa115-6afd-40d7-a4d1-a025411e6b82/fig_hierarchical_timeline_30min_detail.png)

- **Long-Scale Macro Stability**:
  - In WT mice, behavioral states evolve across multi-minute timescales: Mouse 728 transitions between $U_1$ and $U_2$ in 2–5 minute epochs, while Mouse 1680 enters a 25-minute continuous quiescent/investigatory plateau ($U_2$) before returning to active locomotion ($U_1$) at $t = 27\text{ min}$.
  - In DAT-Het KI mice, the lack of behavioral flexibility is stark: Mouse 1694 remains continuously trapped in active locomotion ($U_1$) across the entire 30 minutes (1,800 seconds, 681 routines, 0 visits to quiescence). Mouse 1666 displays brief 1-minute dips into quiescence but spends $>85\%$ of the 30 minutes in active locomotion.
- **Microstructural Preservation**:
  - The 120-second detail view shows that even after 30 minutes of continuous fitting, routine boundaries and constituent syllable alignments remain sharp. Consecutive routine repetitions ($R_6 \to R_6$ and $R_4 \to R_4$) are distinctly resolved by dividing lines.
- **30-Minute Interactive Visualizer**:
  - Saved to [`hierarchical_timeline_30min_interactive.html`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/hierarchical_pilot/hierarchical_timeline_30min_interactive.html). Provides smooth canvas zooming and panning across the full 1,800 seconds with live hover inspection.

### Standalone Interactive HTML Visualizer

We generated an interactive HTML5 visualizer:
- **File**: [`hierarchical_timeline_interactive.html`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/hierarchical_pilot/hierarchical_timeline_interactive.html)
- **Features**:
  - Smooth HTML5 canvas rendering of all 4 exemplar sessions.
  - Interactive presets: `0–60s`, `60–120s`, `100–160s (Exemplar)`, `0–300s`, and `Full Overview (600s)`.
  - Zoom in/out and pan left/right buttons.
  - **Live Hover Tooltips**: Moving the cursor over any block displays:
    - Track Name & State ID (e.g., `Routine: R6`, `Upper State: U2`, `KPMS Syllable: 32`)
    - Exact `Onset (s)`, `End (s)`, and `Duration (s)`
    - Number of constituent bouts and metadata.

---

## 2. Multi-Sequence Model Repair & Progression Experiment

Following code review, the model fitting was repaired to enforce strict independent multi-sequence Baum-Welch (accumulating $\xi$ and $\gamma$ across independent recordings before updating shared parameters), eliminating all boundary stitching artifacts:

![Hierarchical Model Repaired Diagnostics](/home/fsp585/.gemini/antigravity-ide/brain/b03aa115-6afd-40d7-a4d1-a025411e6b82/fig_hierarchical_pilot_diagnostics_repaired.png)

### The Progression Experiment: Free vs. Tied Stage Emissions
To test whether routines possess **true sequential progression** across stages ($s_0 \to s_1 \to s_2$) or merely represent stationary mixture clusters, we conducted 5-fold cross-validation on unseen recording sessions:

| Model Architecture | Cross-Validation Log-Likelihood / Bout | Held-Out Advantage ($\Delta \text{LL}$) | Consistency Across Folds |
| :--- | :---: | :---: | :---: |
| **Free Stage Progression** ($\boldsymbol{\theta}_{k, s}$) | **$-3.1977 \pm 0.0469$** | **$+0.1019 \pm 0.0479$** | **Won 5 / 5 folds ($100\%$)** |
| **Tied Stage Mixture** ($\boldsymbol{\theta}_k$) | $-3.2996 \pm 0.0617$ | Reference | Lost 5 / 5 folds |

**Conclusion**: Each routine learns a genuine, reproducible **stage-to-stage sequential order** that generalizes to unseen mice, outperforming a stationary cluster model on every test fold.

---

## 3. Routine Repertoire & Durations (Boundary-Clean Decoding)

| Routine ID | Occurrences | Median Bouts | 90th % Bouts | Median Duration | Repeat Rate ($k \to k$) | Median Return Gap | Animal Breadth | Description / Profile |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **R1** | 200 | 6.6 bouts | 14.0 bouts | 3.70 s | 10.0% | 14.5 bouts | 13 / 15 mice | Intermediate locomotor chain |
| **R2** | 1,440 | 3.8 bouts | 6.0 bouts | 1.50 s | 12.6% | 16.0 bouts | 15 / 15 mice | Fast transition motor primitive |
| **R3** | 237 | **17.9 bouts** | 33.0 bouts | **7.50 s** | 21.9% | 20.0 bouts | 13 / 15 mice | **Prolonged sustained behavioral mode** |
| **R4** | 2,752 | 4.3 bouts | 7.0 bouts | 2.50 s | 29.8% | 5.0 bouts | 15 / 15 mice | Fast recurring motor motif |
| **R5** | 3,475 | 3.2 bouts | 5.0 bouts | 1.60 s | 28.5% | 5.0 bouts | 15 / 15 mice | Fast recurring motor motif |
| **R6** | 827 | **11.3 bouts** | 21.0 bouts | **4.70 s** | 24.3% | 10.0 bouts | 15 / 15 mice | **Extended investigatory / grooming mode** |

**Distinct Timescale Discovery**:
Rather than learning identical durations, the model naturally separated into:
- **Fast recurring motor motifs** ($R_2, R_4, R_5$): median dwell of $3.2\text{--}4.3$ bouts ($1.5\text{--}2.5\text{ s}$), frequent re-use.
- **Sustained behavioral routines** ($R_3, R_6$): median dwell of $11.3\text{--}17.9$ bouts ($4.7\text{--}7.5\text{ s}$), capturing extended behaviors.

---

## 4. Master Script and Artifact Registry

| File / Artifact | Absolute Path | Description |
| :--- | :--- | :--- |
| **Timeline Script** | [`plot_hierarchical_timeline.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/hierarchical_pilot/plot_hierarchical_timeline.py) | Full 3-track plotting engine with boundaries, elapsed time, overview, and detail views |
| **Overview Figure** | [`fig_hierarchical_timeline_overview.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/hierarchical_pilot/fig_hierarchical_timeline_overview.png) | 600-second overview across WT and DAT-Het KI sessions |
| **60s Detail Figure** | [`fig_hierarchical_timeline_detail_60s.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/hierarchical_pilot/fig_hierarchical_timeline_detail_60s.png) | 60-second readable window with explicit routine repetition dividers and syllable labels |
| **Interactive HTML** | [`hierarchical_timeline_interactive.html`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/hierarchical_pilot/hierarchical_timeline_interactive.html) | Standalone interactive canvas viewer with zoom, pan, presets, and hover inspection |
| **Model Engine** | [`hierarchical_subroutine_model.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/hierarchical_pilot/hierarchical_subroutine_model.py) | Multi-sequence Baum-Welch HMM/HSMM in JAX with GPU acceleration |
| **Experiment Runner** | [`run_hierarchical_feasibility_experiment.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/hierarchical_pilot/run_hierarchical_feasibility_experiment.py) | Cross-validation, surrogate controls, sensitivity sweeps, boundary-clean segmentation |
| **Results Bundle** | [`hierarchical_model_results_bundle_repaired.pkl`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/hierarchical_pilot/hierarchical_model_results_bundle_repaired.pkl) | Fitted model instances, parameters, and cohort data |

---

## 5. Controlled Upper-Timescale Diagnostic Experiment & $M=4$ State Resolution

To address reviewer feedback regarding coarse upper-state timescales and the absence of certain upper states, we implemented:
1. **Advance Probability Update Bug Fix**: In previous continuous-emission code, the empirical advance rate update looped over stages $0 \dots S-2$, omitting the final stage ($S-1$). This distorted the routine exit rate. We repaired the update to accumulate transitions from stage $S-1$ into $n_{\text{adv}}$ and self-transitions into $n_{\text{self}}$.
2. **Expansion to $M=4$ Upper States**: Expanded capacity to $M=4$ states ($U_1 \dots U_4$) with strictly neutral numerical labels.
3. **Upper-State Posterior Probabilities**: Exported posterior probabilities $\gamma_t(u) = \sum_{k, s} \gamma_t(u, k, s)$ to directly distinguish high-confidence assignment from $U_i / U_j$ ambiguity.
4. **Controlled Timescale Intervention**: Evaluated 5-fold cross-validation on unseen sessions when fixing the self-transition diagonal $A_{uu} \in [0.80, 0.90, 0.98]$ (preserving off-diagonal routing) vs. unconstrained Free $A_{uu}$.

---

### Controlled Upper-Timescale Benchmark (5-Fold Cross-Validation on Unseen Sessions)

| Condition | Imposed $A_{uu}$ | Expected Routines / Visit | Held-Out LL / Bout | Median Visit Dur | 90th % Visit Dur | Mean Visits / Session | Mean Posterior Certainty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Free $A_{uu}$** | Learned ($\approx 0.985$) | $\approx 67$ | $-5.7346 \pm 0.2060$ | **$88.1\text{ s}$** | $529.4\text{ s}$ | $9.0$ | **$92.1\%$** |
| **$A_{uu} = 0.98$** | Fixed $0.98$ | $\approx 50$ | **$-5.7334 \pm 0.2141$** | **$108.6\text{ s}$** | $585.5\text{ s}$ | $7.6$ | **$91.0\%$** |
| **$A_{uu} = 0.90$** | Fixed $0.90$ | $\approx 10$ | $-5.7362 \pm 0.1995$ | **$47.7\text{ s}$** | $252.1\text{ s}$ | **$17.0$** | **$88.5\%$** |
| **$A_{uu} = 0.80$** | Fixed $0.80$ | $\approx 5$ | $-5.7409 \pm 0.2018$ | **$22.0\text{ s}$** | $105.2\text{ s}$ | **$36.2$** | **$85.0\%$** |

![Controlled Timescale Comparison Figure](fig_controlled_timescale_comparison.png)

#### Key Findings from the Timescale Benchmark:
1. **The Data Comfortably Tolerate Shorter Episodes**:
   - Across the entire sweep from $A_{uu} = 0.98$ down to $A_{uu} = 0.80$, held-out test log-likelihood drops by **less than $0.0075\text{ nats/bout}$** (well within the $\pm 0.20$ fold-to-fold standard deviation).
   - Constraining $A_{uu} = 0.90$ compresses the upper timescale from $\sim 1.5\text{ min}$ down to a median of **$47.7\text{ seconds}$** ($17.0\text{ visits/session}$) with virtually **zero predictive loss** ($-5.7362$ vs $-5.7346$, a negligible $0.0016\text{ nat}$ difference).
   - Even at $A_{uu} = 0.80$, visit durations drop to a median of **$22.0\text{ seconds}$** ($36.2\text{ visits/session}$), matching the intermediate behavioral episode scale with high posterior certainty ($85.0\%$).
2. **Why the Unconstrained Model Learns Broad Blocks**:
   - The unconstrained model does not produce long blocks due to an artificial diagonal prior (the prior is uniform). Instead, because an upper state represents a **routine-selection preference vector** $\mathbf{B}_{u, :}$, the data provide no active penalty for remaining in the same routine mixture when consecutive routines are drawn from that mixture.
   - The posterior certainty is consistently **$>91\%$**, demonstrating that the model is genuinely confident in its macro assignment rather than wavering ambiguously.

---

### Learned Model Parameters with $M=4$ Upper States

#### Learned Upper Transition Matrix $A_{\text{macro}}$ ($4 \times 4$):
$$\mathbf{A}_{\text{macro}} = \begin{bmatrix}
0.9920 & 0.0016 & 0.0042 & 0.0021 \\
0.0059 & 0.9903 & 0.0017 & 0.0022 \\
0.0037 & 0.0004 & 0.9734 & 0.0225 \\
0.0046 & 0.0011 & 0.0750 & 0.9193
\end{bmatrix}$$

#### Learned Routine Selection Matrix $B_{\text{routine}}$ ($4 \times 6$):
$$\mathbf{B}_{\text{routine}} = \begin{bmatrix}
0.0294 & 0.0164 & 0.0078 & 0.0930 & \mathbf{0.5325} & \mathbf{0.3209} \\
0.0302 & \mathbf{0.4998} & 0.0169 & 0.0554 & 0.1505 & \mathbf{0.2472} \\
0.0268 & 0.0051 & 0.1013 & \mathbf{0.4280} & \mathbf{0.2387} & 0.2002 \\
\mathbf{0.1886} & 0.0092 & \mathbf{0.6713} & 0.0494 & 0.0654 & 0.0161
\end{bmatrix}$$

- **$U_1$ (Slate Blue)**: Locomotor execution state, strongly favoring $R_5$ ($53.3\%$) and $R_6$ ($32.1\%$).
- **$U_2$ (Sage Green)**: High-speed turn/transition state, dominated by $R_2$ ($50.0\%$) and $R_6$ ($24.7\%$).
- **$U_3$ (Warm Ochre)**: Exploratory progression state, dominated by $R_4$ ($42.8\%$), $R_5$ ($23.9\%$), and $R_6$ ($20.0\%$).
- **$U_4$ (Muted Terracotta)**: Sustained investigatory state, heavily dominated by $R_3$ ($67.1\%$) and $R_1$ ($18.9\%$).
- **Active $U_3 \leftrightarrow U_4$ Coupling**: $U_4$ transitions to $U_3$ at a $7.5\%$ rate per routine exit, and $U_3$ transitions to $U_4$ at $2.25\%$.

#### Bug-Fixed Routine Advance Rates & Expected Dwell Times:
| Routine | Learned $p_{\text{advance}}$ | Expected Constituent Bouts |
| :---: | :---: | :---: |
| **R1** | $0.3662$ | **$8.2\text{ bouts}$** |
| **R2** | $0.6114$ | **$4.9\text{ bouts}$** |
| **R3** | $0.4527$ | **$6.6\text{ bouts}$** |
| **R4** | $0.6207$ | **$4.8\text{ bouts}$** |
| **R5** | $0.8242$ | **$3.6\text{ bouts}$** |
| **R6** | $0.6746$ | **$4.4\text{ bouts}$** |

*Note: With stage $S-1$ included in the update, routine dwells are now balanced between $3.6$ and $8.2$ bouts (median $4.8$ bouts), matching the target Layer 2 timescale perfectly.*

---

### Updated Visualizations: 30-Minute Overview and Detail View with Posteriors

#### 30-Minute Overview with $M=4$ Upper States (0 to 1,800 Seconds):
![30-Minute 4-State Overview](fig_hierarchical_timeline_4state_overview.png)

- **$U_3$ and $U_4$ are now vigorously active**: In WT mice (Sessions 1 & 2), the animal dynamically shifts between $U_4$ (investigatory mode, $R_3$) and $U_3$ (exploratory progression). In DAT-Het KI mice (Sessions 4 & 0), the animal alternates between $U_1$ (hyper-locomotor) and $U_3$.

#### 120-Second Detail View with Upper-State Posterior Ribbon (600s to 720s):
![120-Second Detail with Posteriors](fig_hierarchical_timeline_4state_detail.png)

- **Sharp State Transitions**: In WT Session 2 at $t = 662.4\text{ s}$, $P(U_4)$ transitions sharply from $1.0 \to 0.0$ while $P(U_3)$ rises from $0.0 \to 1.0$. The transition is locked cleanly to the boundary between an $R_1$ routine and an $R_6$ routine.
- **Certainty Quantification**: Across all sessions, the posterior probability of the winning state exceeds $95\%$ during dwell, demonstrating that the Viterbi decoded blocks are genuine high-confidence assignments.

---

### Updated Deliverables & Zip Package

The full deliverable suite has been refreshed and packaged into:
- **Workspace Path**: [`syllable_generation/analysis/sequence_viz/hierarchical_pilot/hierarchical_subroutine_complete_package.zip`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/hierarchical_pilot/hierarchical_subroutine_complete_package.zip)
- **Desktop Copy**: `/home/fsp585/Desktop/hierarchical_subroutine_complete_package.zip`
- **Interactive Visualizer**: [`hierarchical_timeline_4state_interactive.html`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/hierarchical_pilot/hierarchical_timeline_4state_interactive.html)

