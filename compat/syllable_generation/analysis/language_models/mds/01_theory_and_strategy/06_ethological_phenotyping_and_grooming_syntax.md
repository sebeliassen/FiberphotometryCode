# 06: Ethological Phenotyping, Behavioral Syntax, and Naturalistic Open-Field Organization

---

## Executive Summary & Integration with Modules 01–05

This document establishes the **biological ground truth and ethological foundation** of our computational behavioral language model. While Modules **01 through 05** formulate the mathematical, architectural, neural, and statistical machinery:

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    COMPUTATIONAL PIPELINE (01-05)                                 |
+───────────────────────────+───────────────────────────+───────────────────────────────────────────+
| Module 01                 | Module 02                 | Module 03                                 |
| Generative Syllabization  | Discrete Grammars & PSTs  | Information Theory & LAIS                 |
| (Sticky AR-HMM, 400ms)    | (Bout-Level Embeddings)   | (Predictability & Memory)                 |
+───────────────────────────+───────────────────────────+───────────────────────────────────────────+
| Module 04                 | Module 05                                                             |
| Photometry & rSLDS        | Geometric Contrastive Testing                                         |
| (DA / ACh Neural States)  | (Sinkhorn-Wasserstein & MMD Phenotyping)                              |
+───────────────────────────+───────────────────────────────────────────────────────────────────────+
                                              ▲
                                              │ Grounded & Validated By
                                              ▼
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                 ETHOLOGICAL PHENOTYPING DOMAIN (06)                               |
+───────────────────────────+───────────────────────────+───────────────────────────────────────────+
| Macro-Scale Dynamics      | Meso-Scale Grammar        | Micro-Scale Kinematics & Rigidity         |
| • Habituation Kinetics    | • Berridge 4-Phase Chains | • Velocity Profile Compression            |
| • Spatial Home Bases      | • aDLS Transition Syntax  | • Heavy-Tailed Dwell Power Laws           |
| • Roundtrip Constraints   | • Aberrant Matrix Skips   | • Subclinical PFF / DAT-KI Biomarkers     |
+───────────────────────────+───────────────────────────+───────────────────────────────────────────+
```

Module **06** provides the **domain-level biological assays and phenotypic benchmarks** (Berridge, Kalueff, Golani, Eilam, Datta, Weinreb) required to translate mathematical representations into validated neurological discoveries.

---

## 1. Natural Temporal Structure of Spontaneous Open-Field Behavior

### 1.1 Habituation Kinetics & Repertoire Shift Dynamics
Spontaneous rodent behavior across a 30- to 60-minute open-field session is fundamentally non-stationary. It follows a structured, non-linear kinetic trajectory driven by the interplay between neophobia, spatial curiosity, and eventual spatial habituation:

1. **Exploration & Spatial Mapping Phase ($t = 0 - 10\text{ min}$):**
   * Maximal kinetic energy expenditure; elevated root-mean-square (RMS) velocity.
   * High frequency of rearing postures (both wall-supported rears and unsupported center rears).
   * Systematic perimeter mapping to establish an internal spatial representation of boundaries.
2. **Maintenance & Localized Exploitation Phase ($t = 10 - 25\text{ min}$):**
   * Logarithmic decay of locomotor velocity and rearing rate.
   * Time budget transitions toward localized exploratory sniffs, lingering, and self-grooming.
3. **Habituated Home-Base Resting Phase ($t = 25 - 60\text{ min}$):**
   * Repertoire shifts toward extended bouts of stationary rest, crouching, and low-amplitude head sweeps.

$$\text{Kinetic Decay Profile: } \bar{v}(t) \approx v_0 \cdot \exp\left(-\frac{t}{\tau_{\text{hab}}}\right) + v_{\infty}$$

* **Neurological Marker:** A flattening of $\tau_{\text{hab}}$ (failure to habituate) reflects hyper-dopaminergic perseveration (e.g., DAT-KI), while accelerated kinetic collapse into immobility reflects early motor fatigue or basal ganglia degeneration (e.g., alpha-synuclein PFF).

---

### 1.2 Exploration, Exploitation, and Spatial Risk Assessment
Rodent navigation resolves an evolutionary conflict between **exploratory foraging drive** and **predator-avoidance anxiety**:
* **Thigmotaxis:** Movement hugging perimeter walls dominates early exposure as a risk-minimization strategy.
* **Peep-and-Hide Sequences:** The rodent extends its anterior body outward from the perimeter while anchoring its hindquarters near the wall, systematically increasing extension distance across successive ventures.
* **Center Ventures:** Brief, ballistic excursions characterized by high acceleration, transitioning rapidly into focused investigation or direct retreat back to the perimeter.

---

### 1.3 Spatial Home-Base Mechanics & Golani/Eilam Roundtrip Constraints
Spatial locomotion is organized around an ethologically established **"home base"** (Golani et al., 1993; Eilam & Golani, 1989), defined as the spatial coordinate possessing:
1. Maximal cumulative occupancy time.
2. Lowest average velocity.
3. Highest concentration of non-locomotor maintenance behaviors (grooming, crouching, resting).

```
[ Spatial Arena Perimeter ]
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │   [Home Base Node] ◄══════════════════════════════════┐      │
  │          │                                            │      │
  │          │ Outward Leg (Low velocity, sensory probing)│      │
  │          ▼                                            │      │
  │     [Stop 1] ──► [Stop 2] ──► [Stop 3] ──► [Stop N] ──┘      │
  │                               (Homeward: Ballistic Return)  │
  └─────────────────────────────────────────────────────────────┘
```

* **The Roundtrip Stop Constraint Rule:** In healthy rodents, exploratory excursions from the home base maintain a strict biological budget of **$8 \text{ to } 10 \text{ stops per roundtrip}$**.
* **Asymmetric Kinematics:**
  * *Outward Leg:* Low average velocity, frequent sensory stops, variable heading angles.
  * *Homeward Leg:* High velocity, direct, ballistic trajectory directly returning to the home base.
* **Compensatory Compression in Motor Disease:** When rodents experience subclinical nigrostriatal dopamine loss or motor fatigue, they compress roundtrips down to **$1 \text{ to } 4 \text{ stops per trip}$** to preserve exploratory coverage while minimizing biomechanical expenditure.

---

## 2. Behavioral Syntax Micro-Structure & Quantitative Ethology

### 2.1 The Canonical Cephalocaudal Syntactic Grooming Chain
Self-grooming occupies $30\text{–}50\%$ of rodent waking hours. Approximately $10\text{–}15\%$ of grooming is organized into strict, rule-governed **syntactic grooming chains** (Berridge & Fentress, 1986; Kalueff et al., 2004).

```
   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
   │   Phase 1   │       │   Phase 2   │       │   Phase 3   │       │   Phase 4   │
──►│  Elliptical │──────►│  Unilateral │──────►│  Bilateral  │──────►│    Body     │──► [Complete]
   │  Snout (1s) │       │   Face (0.2s│       │   Ears (2s) │       │ Licking (2s)│
   └─────────────┘       └─────────────┘       └─────────────┘       └─────────────┘
          │                     │                     │                     │
          ▼ [Abort]             ▼ [Abort]             ▼ [Abort]             ▼
     (Premature Termination / Aberrant Return to Open-Field Locomotion)
```

| Phase | Anatomical Target | Typical Duration | Micro-Kinematic Description | Neural Substrate |
| :--- | :--- | :---: | :--- | :--- |
| **Phase 1: Elliptical Strokes** | Nose, mystacial vibrissae, snout | $0.8 - 1.2\text{ s}$ | 5–9 rapid, small-amplitude bilateral elliptical strokes | Striatal D1 dopamine receptor activation |
| **Phase 2: Unilateral Strokes** | Facial region below eyes | $\approx 0.25\text{ s}$ | Rapid asymmetrical unilateral strokes ascending one side of face | Anterior dorsolateral striatum (aDLS) |
| **Phase 3: Bilateral Strokes** | Forehead, over pinnae (ears) | $2.0 - 3.0\text{ s}$ | Symmetrical, high-amplitude bilateral sweeps passing above ears | Cerebellar & basal ganglia coordinate loops |
| **Phase 4: Body Licking** | Flank, ventral abdomen, tail | $1.0 - 3.0\text{ s}$ | Lateral postural spinal flexion with rapid contact licking | Brainstem premotor circuits & striatal SPNs |

---

### 2.2 Basal Ganglia Control & Aberrant Syntax Breakdown
The serial execution of syntactic chains relies on dopamine-tuned striatal projection neurons in the **anterior dorsolateral striatum (aDLS)**:
* **D1 Receptor Activation:** Promotes chain initiation and progressive syntactic completion.
* **D2 Receptor Activation:** Modulates transition switching thresholds; excessive D2 activation induces syntactic fragmentation.
* **Basal Ganglia Lesions / Alpha-Synuclein Pathology:** Animals initiate grooming movements normally, but suffer a selective collapse in **chain completion rate** (dropping from $\approx 85\%$ in controls to $\approx 50\%$ in disease).

#### Quantitative Grooming Metrics (Kalueff et al., 2007)
1. **Transition Matrix Anomalies:**
   $$\mathbf{T}_{\text{groom}} = \begin{pmatrix} 
   T_{0 \to 1} & T_{0 \to 2} & \cdots & T_{0 \to 4} \\
   T_{1 \to 0} & T_{1 \to 2} & \cdots & T_{1 \to 4} \\
   \vdots & \vdots & \ddots & \vdots \\
   T_{4 \to 0} & T_{4 \to 1} & \cdots & T_{4 \to 4}
   \end{pmatrix}$$
   * *Correct Progressive Transitions:* $0 \to 1 \to 2 \to 3 \to 4 \to 0$
   * *Reversed Transitions (Aberrant):* $3 \to 2$, $4 \to 1$
   * *Skipped Transitions (Aberrant):* $1 \to 4$, $0 \to 3$
2. **Bout Interruption Rate:** Percentage of initiated chains terminating prior to Phase 4.
3. **Rostral-to-Caudal Ratio:** Ratio of time spent in facial grooming (Phases 1–3) versus body grooming (Phase 4).

---

## 3. Mathematical Formalisms of Repertoire Entropy and Sequence Structure

### 3.1 Zero-Order Repertoire Usage Entropy ($H_0$)
Quantifies the overall breadth and balance of the behavioral syllable dictionary $\mathcal{S} = \{s_1, \dots, s_K\}$:

$$H_0 = -\sum_{i=1}^{K} p(s_i) \log_2 p(s_i)$$

* **Interpretation:** A sharp drop in $H_0$ indicates repertoire restriction and dominance by a few hyper-frequent stereotypies.

---

### 3.2 First-Order Transition Entropy ($H_1$)
Quantifies syntax predictability and probabilistic transition diversity:

$$H_1 = -\sum_{i=1}^{K} p(s_i) \sum_{j=1}^{K} P(s_j \mid s_i) \log_2 P(s_j \mid s_i)$$

* **Interpretation:** Pathological motor rigidity manifests as a reduction in $H_1$, indicating that actions follow deterministic, inflexible transition routes.

---

### 3.3 Heavy-Tailed Dwell Scaling Exponents ($\alpha$)
Natural behavioral bout durations follow scale-free power-law distributions:

$$P(t) \sim t^{-\alpha}, \quad \text{for } t \ge t_{\min}$$

* **Interpretation:** In Parkinsonian rigidity, the power-law tail truncates into an exponential decay $P(t) \sim t^{-\alpha} e^{-\lambda t}$, signaling a loss of flexible multi-scale temporal organization.

---

### 3.4 Local Active Information Storage (LAIS)
Measures the point-in-time predictability of the current state $X_t$ given its history $\mathbf{X}_{t-1}^{(k)}$:

$$a_X(t, k) = \log_2 \frac{p(x_t \mid \mathbf{x}_{t-1}^{(k)})}{p(x_t)}$$

* **Interpretation:** Time series of $a_X(t, k)$ generates the **"Behavioral Rigidity Curve"**, pinpointing exact frames of stereotyped perseveration versus flexible switching.

---

## 4. Genotype Phenotypes: Alpha-Synuclein PFF vs. DAT Mutant Models

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    GENOTYPE PHENOTYPE MATRIX                                      |
+───────────────────────+───────────────────────────────────────+───────────────────────────────────+
| Behavioral Scale      | Alpha-Synuclein PFF (Parkinsonian)    | DAT-KI / DAT-KO (Hyper-Dopamine)  |
+───────────────────────+───────────────────────────────────────+───────────────────────────────────+
| Syllable Kinematics   | • Compressed low-velocity modes       | • Exaggerated peak accelerations  |
| (100–500 ms)          | • Loss of unsupported center rears    | • Elevated movement vigor         |
|                       | • Initiation micro-hesitations        | • Hyper-extended limb excursions  |
+───────────────────────+───────────────────────────────────────+───────────────────────────────────+
| Action Syntax         | • Syntactic grooming chain aborts     | • Cyclic, repetitive syntax loops |
| (1–3 seconds)         | • Elevated aberrant transition skips  | • Reduced transition entropy (H1) |
|                       | • Fragmented staccato locomotor steps | • Persistent wall-rearing chains  |
+───────────────────────+───────────────────────────────────────+───────────────────────────────────+
| Macro-Strategy        | • Shortened home-base trips (1-4 stop)| • Flattened habituation kinetics  |
| (5–60+ seconds)       | • Wall-supported posture compensation | • Suppressed transition to rest   |
|                       | • Premature resting state collapse    | • Extended exploratory dwell time |
+───────────────────────+───────────────────────────────────────+───────────────────────────────────+
```

---

## 5. Methodological Benchmark: Why Keypoint-MoSeq is Essential

| Metric / Dimension | Keypoint-MoSeq (Our Choice) | B-SOiD | VAME | MotionMapper |
| :--- | :---: | :---: | :---: | :---: |
| **Generative Model** | **Sticky AR-HMM (Dynamical Systems)** | UMAP + HDBSCAN | VAE + RNN Latents | Wavelet + t-SNE |
| **Median Bout Duration** | **$\approx 400\text{ ms}$ (Kinematic Scale)** | $33 - 100\text{ ms}$ (Sub-kinematic) | $33 - 100\text{ ms}$ (Fragmented) | $33 - 100\text{ ms}$ (Noise-driven) |
| **Boundary Alignment** | **Physiological Accelerometry** | Weak (Mid-acceleration) | Moderate (Smoothed) | Weak (High-freq jitter) |
| **Grooming / Rear Parsing** | **Distinct Asymmetric Modules** | Flickers 5–10 sub-switches | Blends postural states | Blends postural states |
| **Noise Separation** | **Keypoint Error vs. True Pose** | Requires rigid filtering | Sensitive to jitter | Sensitive to tracking noise |

---

## 6. How Module 06 Integrates with Modules 01–05

| Analysis Layer | Implementation in Modules 01–05 | Biological Grounding in Module 06 |
| :--- | :--- | :--- |
| **Syllable Segmentation** | [01_prior_art](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/mds/01_theory_and_strategy/01_prior_art_and_competitive_landscape.md) & [methodological_guide](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/mds/01_theory_and_strategy/methodological_guide_bout_vs_frame_evaluation.md) | Resolves $\approx 400\text{ ms}$ kinematic boundaries matching physiological changepoints. |
| **Syntactic Grammars** | [02_sequence_embeddings](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/mds/01_theory_and_strategy/02_advanced_sequence_embeddings_and_architectures.md) | Quantifies Berridge 4-phase cephalocaudal grooming chains and aberrant transition matrices. |
| **Rigidity & Predictability** | [03_information_theory](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/mds/01_theory_and_strategy/03_information_theoretic_metrics_and_memory.md) | Computes LAIS curves $a_X(t, k)$ to benchmark behavioral flexibility vs. motor perseveration. |
| **Neuromodulatory Coupling**| [04_photometry_coupling](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/mds/01_theory_and_strategy/04_photometry_coupling_and_recurrent_switching.md) | Directly maps dLight1 / ACh3.0 transients to striatal D1/D2 action chunk concatenation. |
| **Cohort Phenotyping** | [05_contrastive_testing](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/mds/01_theory_and_strategy/05_geometric_contrastive_testing_and_phenotyping.md) | Maps Sinkhorn displacement vectors $\mathbf{v}(\mathbf{z})$ to compensatory home-base trip compression. |

---

## References & Foundational Literature

1. **Berridge, K. C., & Fentress, J. C. (1986).** Contextual control of trigeminal sensorimotor function: Grooming syntax and basal ganglia mechanisms. *Journal of Neuroscience*, 6(11), 3259-3269.
2. **Kalueff, A. V., et al. (2004).** Grooming analysis algorithm for neurobehavioural stress research. *Kalueff Lab Publications*.
3. **Kalueff, A. V., et al. (2007).** Analyzing grooming microstructure in neurobehavioral experiments. *Nature Protocols*, 2(10), 2538-2544.
4. **Golani, I., Benjamini, Y., & Eilam, D. (1993).** Stopping behavior: The basic architectural unit of locomotor behavior in mice. *PNAS*, 90(18), 8576-8580.
5. **Eilam, D., & Golani, I. (1989).** Home base behavior of wild and laboratory mice in the open-field. *Behavioral Brain Research*, 34(1-2), 19-33.
6. **Kafkafi, N., et al. (2003).** Open-field behavior withstands drastic changes in arena size. *Behavioral Brain Research*, 142(1-2), 113-121.
7. **Weinreb, C., et al. (2026).** Spontaneous behavior is a succession of self-directed tasks. *Neuron* / *bioRxiv*.
8. **Datta, S. R., et al. (2019/2023).** Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics. *Nature Methods*.
9. **Lizier, J. T., Prokopenko, M., & Zomaya, A. Y. (2012).** Local active information storage as a tool to understand distributed neural information processing. *Frontiers in Neuroinformatics*, 6, 1-15.
