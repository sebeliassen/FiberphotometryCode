# Karen Project: Multimodal Behavioral Phenotyping, Striatal Photometry & Literature Request Guide

**Document Path:** `docs/KAREN_PROJECT_SYSTEM_OVERVIEW_AND_LITERATURE_REQUEST.md`  
**Purpose:** Comprehensive system overview and self-contained research dossier for AI research tools (e.g., NotebookLM, Claude Projects, ChatGPT) and collaborators.  
**File Size:** < 50 KB (fully text-optimized, well sub-10MB limit).  
**Subject:** $\alpha$-Synuclein Pre-Formed Fibril (PFF) Parkinson's Disease Mouse Model, Longitudinal Progression, L-DOPA Rescue, Multi-View Pose Estimation (TopCam + BotCam), Keypoint-MoSeq Syllable Modeling, and Multi-Fiber Striatal Dopamine Photometry.

---

## 1. Executive Scientific Overview & Mission

### 1.1 The Biological System
This research investigates the neuroethological and kinematic progression of **Parkinson's Disease (PD)** using the mouse **$\alpha$-Synuclein Pre-Formed Fibril (PFF)** model compared to **Monomer** control-injected animals. 
* **Pathology:** Intracerebral injection of $\alpha$-synuclein PFFs triggers progressive aggregation, Lewy body-like inclusions, retrograde transmission along nigrostriatal pathways, and progressive loss of dopaminergic neurons in the substantia nigra pars compacta (SNc) projecting to the striatum.
* **Control:** Monomer animals receive non-fibrillar $\alpha$-synuclein, serving as vehicle/surgical controls.
* **Therapeutic Challenge:** Cohorts are tested with **L-DOPA (+ Benserazide)** to evaluate pharmacodynamic motor rescue, behavioral normalization, and potential dyskinesia phenotypes.

### 1.2 The Core Scientific Question
1. **Sub-second Ethology:** Does nigrostriatal dopamine depletion manifest as global slowing (bradykinesia), or does it selectively disrupt specific kinematic syllables, transition syntax (action sequence grammar), and micro-scale motor vigor?
2. **Multimodal Gait & Paw Mechanics:** By combining dorsal (overhead) and ventral (through-glass) high-speed video tracking, what specific paw kinematic abnormalities (stride length, stance/swing duty cycle, limp, cadence, coordination) characterize early vs. late PFF pathology?
3. **Dopamine-Movement Coupling:** How do sub-second striatal dopamine release transients (measured via fluorescent dLight biosensors) correlate with, predict, or modulate discrete behavioral syllables and action transitions in health versus disease?
4. **Longitudinal Biomarkers:** Can unsupervised machine learning detect subtle, pre-symptomatic motor syntax changes before macroscopic locomotor decline becomes evident?

---

## 2. Experimental Design, Longitudinal Cohorts & Conditions

### 2.1 Longitudinal Waves
Mice are tracked across 5 longitudinal experimental waves post-injection (PI):

| Experimental Wave | Timepoint Post-Injection | Biological Milestone | Primary Purpose |
| :--- | :--- | :--- | :--- |
| **`3w_PI`** | **3 Weeks PI** | Early / Pre-symptomatic | Baseline prior to substantial nigrostriatal loss |
| **`8w_PI`** | **8 Weeks PI** | Mid / Emerging pathology | Early phenotype detection & subtle syntax changes |
| **`16w_PI`** | **16 Weeks PI** | Peak pathology | Established motor deficits & dopamine terminal loss |
| **`24w_PI`** | **24 Weeks PI** | Late / Chronic degeneration | Maximum behavioral impairment and compensation |
| **`L_dopa`** | **Drug Challenge** | Pharmacological rescue | L-DOPA + Benserazide vs. Saline response |

### 2.2 Animals, Arenas & Sessions
* **Total Synchronized Sessions:** **110 master sessions** (~23.86 million frames, >82 hours of behavioral recording).
* **Statistical Cohort:** **99 sessions** fully balanced across genotypes (**56 PFF vs. 43 Monomer**) across 15+ individual mice.
* **Multi-Chamber Open Field Hardware:**
  * 4 partitioned quadrants recorded simultaneously in one rig.
  * **Arena 1 (`top_left`)**, **Arena 2 (`bottom_left`)**, and **Arena 3 (`bottom_right`)** are active mouse chambers.
  * **Arena 4 (`top_right`)** is intentionally unpopulated (the fiber photometry console accommodates 3 optical patch cords).

---

## 3. Multimodal Data Streams & Technical Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │           4-Chamber Recording Rig            │
                       └──────────────────────┬───────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       ┌────────────────────────┐                          ┌────────────────────────┐
       │     TopCam (Dorsal)    │                          │     BotCam (Ventral)   │
       │   972x972 @ 60/80 FPS  │                          │   972x972 @ 60.0 FPS   │
       └───────────┬────────────┘                          └───────────┬────────────┘
                   │                                                   │
                   ▼                                                   ▼
       ┌────────────────────────┐                          ┌────────────────────────┐
       │   DLC Top (17 kpts)    │                          │   DLC Bot (6 kpts)     │
       │ Spine, Head, Tailbase  │                          │ Paws, Snout, Tailbase  │
       └───────────┬────────────┘                          └───────────┬────────────┘
                   │                                                   │
                   └─────────────────────┬─────────────────────────────┘
                                         │
                                         ▼
                   ┌──────────────────────────────────────────┐
                   │ Canonical Egocentric Multiview Fusion    │
                   │ Unified 23-Keypoint Pose: [Y_top; Y_bot] │
                   └─────────────────────┬────────────────────┘
                                         │
                                         ▼
                   ┌──────────────────────────────────────────┐
                   │    Keypoint-MoSeq (AR-HMM / SLDS)        │
                   │    PCA Latent Space (D=10), 100 States   │
                   └─────────────────────┬────────────────────┘
                                         │
          ┌──────────────────────────────┴──────────────────────────────┐
          ▼                                                             ▼
┌─────────────────────────────────┐                   ┌─────────────────────────────────┐
│ Unsupervised Behavioral Syllables│                   │ Synchronized Striatal Photometry│
│ - Discrete states z_t (0 to 99) │ ◄── [Time-Lock] ─►│ - dLight1.3b (470nm signal)      │
│ - Latents x_t, speed, heading   │                   │ - Isosbestic control (415nm)    │
│ - Transition bigrams & syntax   │                   │ - RANSAC dF/F & Wallace z-score │
└─────────────────────────────────┘                   └─────────────────────────────────┘
```

### 3.1 High-Speed Synchronized Video
* **Dorsal Camera (TopCam):** Overhead viewpoint capturing arena bounding coordinates, torso contour, and dorsal spine curvature ($972 \times 972$ viewport).
* **Ventral Camera (BotCam):** Underneath viewpoint recorded through optical glass floor ($972 \times 972$, 60.0 FPS) capturing paw contact, stride, and ventral snout kinematics.
* **Hardware Timestamp Sync:** Master hardware clock logs TTL pulses across cameras and photometry. Jitter is $\le 1$ frame ($<12.5$ ms) with **0.000 ms microsecond offset** over 50+ minute continuous sessions.

### 3.2 DeepLabCut (DLC) Tracking & 23-Keypoint Pose
1. **Dorsal View (17 Bodyparts):**
   `nose`, `left_ear_tip`, `right_ear_tip`, `neck`, `mid_back`, `mouse_center`, `mid_backend`, `mid_backend2`, `mid_backend3`, `tail_base`, `left_shoulder`, `left_midside`, `left_hip`, `right_shoulder`, `right_midside`, `right_hip`, `head_midpoint`.
2. **Ventral View (6 Bodyparts, HRNet-W32 model):**
   `nose_tip`, `tail_base`, `left_front_paw`, `right_front_paw`, `left_hind_paw`, `right_hind_paw`.
3. **Multiview Fusion Mathematics:**
   * Homologous midpoints: $c_{\text{top}} = \frac{1}{2}(\text{nose}_{\text{top}} + \text{tail}_{\text{top}})$ and $c_{\text{bot}} = \frac{1}{2}(\text{nose}_{\text{bot}} + \text{tail}_{\text{bot}})$.
   * Ventral pose rotated into canonical dorsal heading frame:
     $$Y^{\text{bot\_ego}}_t = R(-\theta_{\text{bot}}) \left(Y^{\text{bot}}_t - c_{\text{bot}}\right) \times s$$
   * Virtual translocation into TopCam coordinates:
     $$Y^{\text{virtual}}_t = c_{\text{top}} + R(+\theta_{\text{top}}) Y^{\text{bot\_ego}}_t$$
   * Concatenated into a unified **23-keypoint representation** that preserves paw excursions relative to the anatomical body center.

### 3.3 Multi-Fiber Striatal Photometry (dLight1.3b)
* **Sensor:** Genetically encoded dopamine sensor (AAV9-hSyn-dLight1.3b) targeting striatal subdivisions (Dorsal Striatum / Ventral Striatum).
* **Dual-Wavelength Modulation:**
  * **470 nm:** Dopamine-dependent fluorescence excitation.
  * **415 nm:** Isosbestic point (dopamine-insensitive) used to capture movement artifacts and photobleaching.
* **Processed Channels in Master Timelines:**
  * Raw signals: `signal_0` through `signal_3`.
  * `dff`: $\Delta F / F_0$ calculated via baseline normalization.
  * `zscore_ransac`: RANSAC robust regression against the 415 nm isosbestic reference.
  * `zscore_wallace`: Adaptive baseline filtering.
  * `prominence`: Peak transient detection.

### 3.4 Unsupervised Behavioral Segmentation (Keypoint-MoSeq)
* **Model Class:** Autoregressive Hidden Markov Model with Switching Linear Dynamical Systems (AR-HMM / SLDS).
* **Dimensionality:** PCA reduces the 23-keypoint coordinates to **$D = 10$ principal components**.
  * Empirical variance analysis confirms: PC01–PC02 (78.9% variance) govern dorsal spine posture; PC03–PC10 (18.9% variance) are 91–97% dominated by ventral paw locomotion and gait kinematics.
* **Repertoire:** Segmented into **100 discrete behavioral syllables** ($z_t \in \{0, \dots, 99\}$).
* **Duration:** Median syllable duration of **$250.0$ ms** (20 frames @ 80Hz / 15 frames @ 60Hz), matching sub-second mammalian motor primitives.

---

## 4. Statistically Validated Phenotypes (PFF vs. Monomer)

The dataset has been rigorously benchmarked using linear mixed-effects models and **10,000 label-shuffling permutations**:

| Behavioral Dimension | Metric | Monomer (Control) | PFF (Parkinson's) | Effect Size & Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Global Locomotion** | Mean Speed | $74.8 \pm 38.1$ px/s | $61.2 \pm 15.0$ px/s | **$-18.2\%$ slower in PFF** ($d = -0.50$, $p = 0.031$, $p_{\text{perm}} = 0.011$) |
| **Peak Vigor** | 95th Percentile Speed ($v_{95}$) | $232.5 \pm 155.9$ px/s | $175.2 \pm 80.5$ px/s | **$-24.7\%$ sprint reduction** ($d = -0.48$, $p = 0.032$, $p_{\text{perm}} = 0.015$) |
| **Bout Dynamics** | Active Bout Duration | $2.97$ s | $2.61$ s | **$-12.0\%$ shorter continuous bouts** ($p = 0.029$, $p_{\text{perm}} = 0.029$) |
| **Immobility** | Stationary Fraction (<5 px/s) | $0.94\%$ | $1.05\%$ | **$+11.1\%$ more freezing/resting** ($p = 0.040$, $p_{\text{perm}} = 0.041$) |
| **Kinematic Stability** | Normalized Movement Jerk | $2366.9 \pm 503.2$ | $2627.8 \pm 628.7$ | **$+11.0\%$ higher jerkiness/ataxia** ($d = +0.45$, $p = 0.034$, $p_{\text{perm}} = 0.026$) |
| **Syntax Grammar** | Transition Entropy Rate | $2.560$ bits/step | $2.624$ bits/step | **$+2.5\%$ disordered syntax** ($p = 0.029$, $p_{\text{perm}} = 0.034$) |
| **Micro-Kinematics** | Syllable 43 Speed | $111.7$ px/s | $96.5$ px/s | **$-14.5\%$ velocity attenuation** ($p = 0.015$, $p_{\text{perm}} = 0.011$) |
| **Micro-Kinematics** | Syllable 9 Speed | $63.3$ px/s | $59.7$ px/s | **$-5.2\%$ velocity attenuation** ($p = 0.018$, $p_{\text{perm}} = 0.019$) |
| **Syntax Transitions** | Bigram $14 \to 65$ | $0.038\%$ | $0.097\%$ | **$+153.7\%$ altered transition** ($p = 0.0002$, $p_{\text{perm}} = 0.008$) |
| **Endurance Fatigue** | Within-Session Speed Drop | $+1.57\%$ (stable) | $-11.33\%$ (fatigue) | PFF mice exhibit progressive motor fatigue across 40 min |

---

## 5. Directory Index, Key Scripts & Data Assets

### 5.1 Primary Data Stores
* `syllable_generation/data/sessions_karen_project_w_sylls.pkl`: Master pickled database containing 110 fully synchronized `SessionRecord` objects (13.68 GB on disk). Each record includes:
  * `s.session_id`: Unique identifier (`{timestamp}_{chamber}`)
  * `s.metadata`: `genotype` (`PFF` / `Monomer`), `condition` (`3w_PI`, `8w_PI`, `16w_PI`, `24w_PI`, `L_dopa`), `animal_id`
  * `s.all_data_df`: Frame-by-frame synchronized master DataFrame containing `velocity_px_s`, `syllable`, `heading`, `angular_velocity`, and aligned photometry channels
  * `s.video_refs_df`: Metadata and filepaths for both TopCam (`role: "primary"`) and BotCam (`role: "ventral"`)
  * `s.tracking`: Dict holding DLC dataframes (`topcam`, `botcam`, `fused_multiview`)
* `syllable_generation/syllabizers/keypoint_moseq_syll/karen_project_2/final_karen_model/results.h5`: Extracted KPMS results file (1.06 GB) containing latent state trajectories $x_t$ and discrete syllable sequences $z_t$.

### 5.2 Key Pipelines & Analytical Scripts
* **Multiview Fusion:**
  * `syllable_generation/multiview_top_bot_experiments/fuse_multiview_cohort.py`: High-performance egocentric 23-keypoint fusion script.
  * `syllable_generation/multiview_top_bot_experiments/DUAL_VIDEO_PIPELINE_ARCHITECTURE.md`: Complete mathematical specification of multi-camera alignment, heading initialization, and PCA variance properties.
* **Keypoint-MoSeq Training:**
  * `syllable_generation/multiview_top_bot_experiments/train_full_multiview_model.py`: Production KPMS training script on fused 23-keypoint coordinates.
  * `syllable_generation/syllabizers/keypoint_moseq_syll/run_historical_baseline_reproduction.py`: Standalone benchmark runner leveraging vectorized AR dynamics and parallel banded Cholesky GMRF solver.
* **Phenotyping & Kinematic Analysis:**
  * `syllable_generation/analysis/ulrik_kinematics/experiments/karen_phenotyping/run_karen_5tier_phenotyping_suite.py`: Comprehensive 5-tier statistical pipeline (Macro locomotion, Syllable usage, Micro-kinematics, Syntax matrices, and Longitudinal LMMs).
  * `syllable_generation/analysis/ulrik_kinematics/experiments/karen_phenotyping/run_karen_longitudinal_lmms.py`: Linear mixed-effects repeated measures models with FDR adjustment.
  * `syllable_generation/analysis/ulrik_kinematics/experiments/karen_phenotyping/run_karen_permutation_tests.py`: 10,000 label-permutation verification engine.

---

## 6. Targeted Literature Request & Research Prompt

> **Instructions for the Research Tool / AI Agent:**  
> Use the technical context above to retrieve, synthesize, and recommend scientific literature addressing the specific downstream research challenges outlined below.

### Primary Downstream Research Challenges:
1. **Dopamine Modulation of Sub-Second Behavioral Syllables:**
   * How does rapid phasic dopamine (dLight / GRAB-DA) in the dorsal vs. ventral striatum correlate with discrete action initiation, syllable vigor, and syllable transition probability?
   * How does chronic progressive dopamine loss in $\alpha$-synuclein PFF models alter striatal terminal release dynamics during spontaneous open-field behavior?
2. **Computational Ethology & Behavioral Syntax in Movement Disorders:**
   * What published studies utilize unsupervised state-space models (Keypoint-MoSeq, MoSeq, B-SOiD, VAME) to quantify motor syntax, entropy rate, and sequence degradation in Parkinson's disease, ataxia, or basal ganglia models?
   * How do researchers distinguish between "classical gross bradykinesia" and "micro-scale motor slowing within discrete syllables"?
3. **Multi-Camera (Dorsal + Ventral) Pose Estimation & Biomechanics:**
   * What methodologies exist for fusing multi-view keypoint tracking (overhead torso/spine + underside paw contact) in freely moving rodents?
   * How do paw-level gait parameters (duty factor, stride length, interlimb coordination, swing velocity) correlate with disease severity in pre-formed fibril (PFF) or 6-OHDA models?
4. **L-DOPA Pharmacodynamics & Behavioral Normalization:**
   * How does systemic L-DOPA (+ benserazide) treatment acutely alter behavioral syllable distributions, velocity scaling, and transition syntax?
   * How are L-DOPA-induced dyskinesias (LID) or stereotypic motor fragments manifested in continuous state-space models?
5. **Pre-Symptomatic Biomarkers in Longitudinal Neurodegeneration:**
   * What longitudinal behavioral metrics (kinematic jerk, transition entropy, bout duration) show statistical divergence earliest (e.g., at 3 or 8 weeks post-injection) before overt motor symptoms emerge?

### Key Benchmark Papers to Cross-Reference:
* **Keypoint-MoSeq Foundation:** Weinreb et al., *Nature Methods* (2024), "Keypoint-MoSeq: parsing behavior using unsupervised modeling of 3D keypoint trajectories."
* **Striatal Dopamine & MoSeq:** Markowitz et al., *Cell* (2018), "The Striatum Organizes Spontaneous Behavior Through Sub-second Dopamine Dynamics."
* **PFF Parkinson's Pathophysiology:** Luk et al., *Science* (2012) / Volpicelli-Daley et al., *Nature Protocols* (2014) on $\alpha$-synuclein transmission and behavioral onset timelines.
* **Multi-View Rodent Biomechanics:** Karashchuk et al., *eLife* (2021) on Anipose / 3D tracking, and recent through-glass ventral gait analysis literature.
