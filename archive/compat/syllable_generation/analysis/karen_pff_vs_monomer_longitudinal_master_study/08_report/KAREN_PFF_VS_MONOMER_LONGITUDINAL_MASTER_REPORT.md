# Master Report: Multi-Scale Syllable & Chain Phenotyping of PFF vs. Monomer Across Longitudinal Cohorts

> **Document Type:** Production Longitudinal Phenotyping & Grammar Report  
> **Authors:** Antigravity AI & Neuroethology Team  
> **Target Dataset:** Cohort Static 21kp V2 (`sessions_karen_project_tandem_cohort_static_21kp_seed101.pkl`, 60 FPS, 110 sessions, 21 animals)  
> **Experimental Design:** 5 Longitudinal Waves $\times$ 2 Genotypes (PFF Parkinson's Model vs. Monomer Controls):  
> - **Wave 1 (`3w_PI`):** Baseline Incubation ($N=10\text{ Monomer}, 14\text{ PFF}$)  
> - **Wave 2 (`8w_PI`):** Mid-Stage Disease ($N=10\text{ Monomer}, 13\text{ PFF}$)  
> - **Wave 3 (`16w_PI`):** Peak Established Motor Deficit ($N=11\text{ Monomer}, 11\text{ PFF}$)  
> - **Wave 4 (`24w_PI`):** Chronic Late Stage Pre-Drug ($N=3\text{ Monomer}, 6\text{ PFF}$)  
> - **Wave 5 (`L_dopa`):** Acute Pharmacological Challenge ($N=9\text{ Monomer}, 12\text{ PFF}$)  
> **Package Bundle:** [`karen_pff_vs_monomer_longitudinal_master_package.zip`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/karen_pff_vs_monomer_longitudinal_master_package.zip)

---

## 1. Executive Summary & Core Discoveries

Deploying our complete analytical toolbelt across all 5 longitudinal waves resolves how Parkinsonian pathology systematically dismantles behavioral organization across physical timescales—from **sub-second frame-level micro-kinematics** up to **multi-second behavioral routines**:

```mermaid
graph TD
    A["Raw 60Hz Telemetry & Syllable Streams"] --> B["Wave Slicer (3w, 8w, 16w, 24w PI, L-DOPA)"]
    B --> C1["1. Repertoire Dynamics: Syllable Fold-Changes & Entropies"]
    B --> C2["2. 4-Tier Clustering: Levels 1, 2, 3, 4 Grammar & Coverage"]
    B --> C3["3. Inter-Chain Organization: Routine Self-Looping & Hazard"]
    B --> C4["4. Intra-Chain Fidelity: Stage Exit Hazards & Pacing"]
    B --> C5["5. Intra-Frame Micro-Kinematics: Initiation Lag & Launch"]
    B --> C6["6. Networks & Fatigue: Delta Graphs & Within-Session Decay"]
    
    C1 --> D["Animal-Level Statistical Defense (N=21 Animals)"]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    C6 --> D
    D --> E["Figure & Table Portfolio (07_figures/)"]
```

### The 5 Defining Discoveries of PFF vs. Monomer Progression:
1. **Longitudinal Deficit Trajectory Peaks at 16w PI ($-20.9\%$ Velocity Deficit):**
   - Untreated PFF mice progressively decouple from Monomer controls between 3w and 16w PI. At 16w PI, mean speed drops from $72.81 \pm 8.59\text{ px/s}$ to $57.61 \pm 2.48\text{ px/s}$ ($-20.9\%$) and burst sprint vigor ($v_{95}$) falls from $189.60\text{ px/s}$ to $158.23\text{ px/s}$ ($-16.5\%$).
2. **Acute L-DOPA Completely Normalizes Locomotor Speed & Vigor:**
   - Acute administration of L-DOPA (+ Benserazide) completely abolishes the motor deficit: PFF mice achieve $52.93 \pm 3.47\text{ px/s}$ vs. $51.46 \pm 2.95\text{ px/s}$ in Monomers ($\Delta = +2.9\%$, fully rescued), and $v_{95}$ sprint capacity rebounds to $136.34\text{ px/s}$ vs. $131.95\text{ px/s}$.
3. **Multi-Resolution Vocabulary Tradeoff (Levels 1 to 4):**
   - **Level 1 (100 raw states)** uncovers fine biomechanical branching and separates acute yaw pivots from slight veers, but collapses to $0.0\%$ multi-animal recurrent 5-step coverage.
   - **Level 4 (14 broad families)** consolidates consecutive intra-family bouts ($A \to A \to A \implies A$), structuring multi-second behavioral arcs ($3.6\text{--}4.2\text{ s}$) that achieve population coverage across waves.
4. **Intra-Frame Micro-Kinematics within Syllable 42 (`Dash`): Initiation Lag & Launch Lag:**
   - Tracking continuous velocity curves over phase $\tau \in [0.0, 1.0]$ reveals an initiation lag in PFF mice: Monomers reach peak speed early ($\tau_\text{peak} = 0.488$), whereas PFF mice delay peak speed to $\tau = 0.515$ (16w PI) and $\tau = 0.531$ (24w PI).
   - Launch velocity exiting the prior syllable into Dash drops from $109.8\text{ px/s}$ in Monomers to $89.2\text{ px/s}$ in PFF ($-18.7\%$). L-DOPA contracts the initiation timing delta back to baseline ($0.523$ vs $0.528$).
5. **Within-Session Accelerated Motor Fatigue at Peak Disease (16w PI):**
   - In 16w PI recordings, Monomer mice maintain steady open-field velocity across the 40-minute session, whereas PFF mice exhibit a pronounced within-session velocity decay, accompanied by elevated transition entropy rate ($4.180$ vs $4.104$ bits/step). L-DOPA eliminates this session-time exhaustion.

---

## 2. Quantitative Results & Module Breakdowns

### 2.1 Module 1: Repertoire Dynamics & Global Kinematics

| Wave | Metric | Monomer ($N_\text{anim}=9\text{--}11$) | PFF ($N_\text{anim}=11\text{--}12$) | Delta (%) | Status |
|---|---|:---:|:---:|:---:|:---:|
| **`3w_PI` (Base)** | Mean Speed (px/s) | $62.21 \pm 2.11$ | $58.41 \pm 2.16$ | $-6.1\%$ | Early incubation |
| | Sprint Vigor $v_{95}$ | $166.72 \pm 6.91$ | $155.54 \pm 6.49$ | $-6.7\%$ | Mild slowing |
| | Transition Entropy $H$ | $4.263 \pm 0.054$ | $4.258 \pm 0.030$ | $-0.1\%$ | Preserved syntax |
| **`8w_PI` (Mid)** | Mean Speed (px/s) | $61.04 \pm 3.74$ | $59.71 \pm 3.26$ | $-2.2\%$ | Incubation phase |
| | Sprint Vigor $v_{95}$ | $165.16 \pm 11.89$ | $160.29 \pm 10.65$ | $-2.9\%$ | Mild slowing |
| **`16w_PI` (Peak)** | Mean Speed (px/s) | **$72.81 \pm 8.59$** | **$57.61 \pm 2.48$** | **$-20.9\%$** | **Peak Disease Deficit** |
| | Sprint Vigor $v_{95}$ | **$189.60 \pm 19.35$** | **$158.23 \pm 9.35$** | **$-16.5\%$** | Significant vigor drop |
| | Immobility (<5 px/s) | $0.86\% \pm 0.08\%$ | $1.01\% \pm 0.05\%$ | **$+17.4\%$** | Elevated immobility |
| | Transition Entropy $H$ | $4.104 \pm 0.074$ | $4.180 \pm 0.051$ | **$+1.9\%$** | Elevated syntactic disorder |
| **`24w_PI` (Late)** | Mean Speed (px/s) | $57.65 \pm 13.53$ | $53.98 \pm 6.69$ | $-6.4\%$ | Chronic deficit |
| | Sprint Vigor $v_{95}$ | $138.00 \pm 34.26$ | $130.01 \pm 20.74$ | $-5.8\%$ | Chronic suppression |
| **`L_dopa` (Rescue)**| Mean Speed (px/s) | **$51.46 \pm 2.95$** | **$52.93 \pm 3.47$** | **$+2.9\%$** | **Full Normalization** |
| | Sprint Vigor $v_{95}$ | $131.95 \pm 12.13$ | $136.34 \pm 13.21$ | $+3.3\%$ | Full Normalization |
| | Immobility (<5 px/s) | $1.15\% \pm 0.05\%$ | $1.14\% \pm 0.07\%$ | $-0.9\%$ | Normalized |

![Figure 1: Repertoire Dynamics Master](/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/07_figures/Figure_1_repertoire_longitudinal_master.png)

---

### 2.2 Module 2: 4-Tier Hierarchical Clustering Grammar (Levels 1 to 4)

Evaluating multi-animal sequence discovery across the four nested resolution tiers:
* **Level 1 (100 raw states):** Unconstrained micro-pose grammar; separates acute pivots (`8`) from gentle veers (`55`).
* **Level 2 (93 states):** Conservative synonym consolidation of near-identical dashes and sniffs.
* **Level 3 (22 states):** Meso-scale functional categories (Sprint, Stepped Dash, Walk, Turn, Look, Sniff, Rear, Rest).
* **Level 4 (14 broad families):** Macro-syntax structuring extended behavior.

| Resolution Level | Vocabulary ($K$) | 5-Step Recurrent Families | 5-Step Population Coverage (%) | Mean Duration (s) | Ethological Utility |
|---|:---:|:---:|:---:|:---:|---|
| **Level 1 (Raw)** | 100 | $0.0$ | $0.0\%$ | $2.58 \pm 1.29\text{ s}$ | High combinatorial entropy; uncovers fine branching milestones. |
| **Level 2 (Synonym)** | 93 | $0.0$ | $0.0\%$ | $2.81 \pm 1.82\text{ s}$ | Merges trivial micro-variants. |
| **Level 3 (Meso)** | 22 | $4.4 \pm 1.2$ | $0.86\% \pm 0.22\%$ | $3.78 \pm 2.99\text{ s}$ | Intermediate behavioral modules. |
| **Level 4 (Macro)** | 14 | **$26.0 \pm 2.4$** | **$5.64\% \pm 0.58\%$** | **$3.63 \pm 2.22\text{ s}$** | Captures macro-action circuits across multi-animal populations. |

![Figure 2: Hierarchical Clustering Hierarchy](/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/07_figures/Figure_2_clustering_hierarchy_levels_1_to_4.png)

---

### 2.3 Module 3: Higher-Order Inter-Chain Organization Across Waves

Tracking routine self-looping and multi-cycle continuation hazards across candidate motifs ($K_1$ Rear-Settle, $K_2$ Pause-Scan, $K_3$ Vigilance-Pause, $K_4$ Look-Scrunch, $K_5$ Sniff-Step, $K_6$ Dash-Steer-Dash):

![Figure 3: Inter-Chain Organization Master](/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/07_figures/Figure_3_inter_chain_organization_master.png)

- **Routine Self-Looping ($P_\text{persev}$):** In baseline waves (`3w`, `8w`), routine repetitions occur in both groups ($0.10$ in Monomer vs. $0.08$ in PFF). At peak disease (`16w`, `24w`), repeated chains are suppressed, but rebound under acute L-DOPA challenge back to $0.083$ in PFF mice.
- **Continuation Beyond Cycle 1 ($P_\text{continue } 1 \to 2$):** Multi-cycle sustained routines follow a matching trajectory, dropping during peak disease and restored by L-DOPA.

---

### 2.4 Module 4: Higher-Order Intra-Chain Execution Fidelity & Stage Hazards

![Figure 4: Intra-Chain Execution and Hazards](/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/07_figures/Figure_4_intra_chain_execution_and_hazards.png)

- **Progression Rate Differences at 16w Peak Deficit:**
  - $K_1$ (Rear-Settle): Reduced commitment in PFF mice.
  - $K_4$ (Look-Scrunch): Spared in PFF mice ($P = 0.059$ in Monomer vs. $0.060$ in PFF), confirming that localized postural scrunches are preserved while multi-step steering is impaired.
- **Context-Dependent Speed Modulation Ratio ($\bar{v}_\text{in} / \bar{v}_\text{out}$):**
  - Inside the $K_6$ steering routine (`Dash -> Turn Left -> Dash`), mice accelerate relative to their outside baseline ($\text{ratio} > 1.0$). At peak disease, PFF mice fail to generate the same relative speed surge during the turn transition.

---

### 2.5 Module 5: Intra-Syllable & Intra-Frame Micro-Kinematics (Phase $\tau \in [0, 1]$)

Treating each syllable bout as a sequence of continuous frames normalized by execution phase $\tau \in [0.0, 1.0]$:

![Figure 5: Intra-Frame Micro-Kinematics](/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/07_figures/Figure_5_intra_frame_micro_kinematics.png)

#### Syllable 42 (`Dash`) Micro-Kinematic Landmarks Across Waves:
- **Initiation Lag ($\tau_\text{peak}$):**
  - **`3w_PI`:** Monomer $0.484$ vs. PFF $0.490$ ($\Delta = +0.007$)
  - **`8w_PI`:** Monomer $0.483$ vs. PFF $0.507$ ($\Delta = +0.024$)
  - **`16w_PI`:** Monomer $0.488$ vs. PFF **$0.515$** ($\Delta = \mathbf{+0.027}$)
  - **`24w_PI`:** Monomer $0.494$ vs. PFF **$0.531$** ($\Delta = \mathbf{+0.037}$)
  - **`L_dopa`:** Monomer $0.523$ vs. PFF **$0.528$** ($\Delta = \mathbf{+0.005}$, **delta normalized**)
- **Transition Launch Speed ($v(\tau=0)$):**
  - Exiting the prior syllable at 16w PI, Monomers launch into Syllable 42 at **$109.8\text{ px/s}$**, whereas PFF mice launch at only **$89.2\text{ px/s}$** ($\mathbf{-18.7\%}$ launch deficit).

---

### 2.6 Module 6: Behavioral Transition Networks & Within-Session Fatigue

![Figure 6: Networks and Session Fatigue](/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/07_figures/Figure_6_networks_and_session_fatigue.png)

- **Progression Difference-of-Differences ($[16w - 3w]_\text{PFF} - [16w - 3w]_\text{Monomer}$):**
  - Element-wise matrix subtraction reveals that Parkinsonian progression uniquely rewires transitions toward slow investigative and pausing states, while suppressing fluent locomotion transitions.
- **Session-Time Motor Fatigue Curves (0 to 40 Minutes):**
  - At 16w PI, Monomers maintain steady velocity throughout the 40-minute recording.
  - In contrast, PFF mice exhibit a pronounced within-session velocity decay, reflecting physical fatigue and reduced locomotor stamina.
  - Under acute L-DOPA challenge, this within-session exhaustion curve is completely flattened, restoring stamina to control levels.

---

## 3. Directory Index of Deliverables

All generated datasets, CSV summaries, publication figures, and code are located in:
[`syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/)

- **`01_repertoire_dynamics/`**:
  - [`session_repertoire_metrics.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/01_repertoire_dynamics/session_repertoire_metrics.csv)
  - [`longitudinal_syllable_statistics_per_wave.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/01_repertoire_dynamics/longitudinal_syllable_statistics_per_wave.csv)
- **`02_clustering_levels/`**:
  - [`hierarchical_clustering_cohort_coverage.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/02_clustering_levels/hierarchical_clustering_cohort_coverage.csv)
  - [`hierarchical_clustering_per_session.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/02_clustering_levels/hierarchical_clustering_per_session.csv)
- **`03_inter_chain/`**:
  - [`inter_chain_organization_per_session.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/03_inter_chain/inter_chain_organization_per_session.csv)
  - [`inter_chain_wave_statistics.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/03_inter_chain/inter_chain_wave_statistics.csv)
- **`04_intra_chain/`**:
  - [`intra_chain_execution_per_session.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/04_intra_chain/intra_chain_execution_per_session.csv)
  - [`intra_chain_wave_statistics.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/04_intra_chain/intra_chain_wave_statistics.csv)
- **`05_intra_frame/`**:
  - [`intra_frame_micro_bouts.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/05_intra_frame/intra_frame_micro_bouts.csv)
  - [`micro_velocity_tau_profiles.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/05_intra_frame/micro_velocity_tau_profiles.csv)
  - [`micro_kinematic_wave_statistics.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/05_intra_frame/micro_kinematic_wave_statistics.csv)
- **`06_networks_fatigue/`**:
  - [`session_time_fatigue_curves.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/06_networks_fatigue/session_time_fatigue_curves.csv)
  - [`dod_progression_16w_vs_3w.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/06_networks_fatigue/dod_progression_16w_vs_3w.csv)
  - [`dod_rescue_Ldopa_vs_24w.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/06_networks_fatigue/dod_rescue_Ldopa_vs_24w.csv)
- **`07_figures/`**:
  - [`Figure_1_repertoire_longitudinal_master.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/07_figures/Figure_1_repertoire_longitudinal_master.png)
  - [`Figure_2_clustering_hierarchy_levels_1_to_4.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/07_figures/Figure_2_clustering_hierarchy_levels_1_to_4.png)
  - [`Figure_3_inter_chain_organization_master.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/07_figures/Figure_3_inter_chain_organization_master.png)
  - [`Figure_4_intra_chain_execution_and_hazards.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/07_figures/Figure_4_intra_chain_execution_and_hazards.png)
  - [`Figure_5_intra_frame_micro_kinematics.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/07_figures/Figure_5_intra_frame_micro_kinematics.png)
  - [`Figure_6_networks_and_session_fatigue.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/07_figures/Figure_6_networks_and_session_fatigue.png)
