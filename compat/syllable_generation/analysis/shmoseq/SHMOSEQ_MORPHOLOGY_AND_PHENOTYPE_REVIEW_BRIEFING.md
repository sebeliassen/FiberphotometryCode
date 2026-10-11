# Briefing Dossier: shMoSeq Macro-State Idiosyncrasy, Morphological Confounding, and Genotype Phenotyping

**Date**: September 2026  
**Target Audience**: External Reviewing Language Model / Machine Learning Ethologist  
**Project**: Fiber Photometry & Behavioral Sequencing (Keypoint-MoSeq & shMoSeq)  
**Authors**: Antigravity Pair-Programming Agent & Experimentalist  

---

## 1. Executive Summary & The Central Paradox

We are analyzing open-field behavioral recordings from an amphetamine pharmacogenomics experiment in mice carrying a dopamine transporter mutation (**DAT-Het** vs. **Wild-Type [WT]**).

### The Paradox:
1. **At the low-level syllable stage (Keypoint-MoSeq, 100 syllables)**:
   - The genotype difference in baseline spontaneous behavior (Saline) is **statistically significant** under an exact combinatorial animal-level permutation test ($N=15$ mice, 6,435 partitions):
     - **Total Variation Distance**: $0.3540$, exact **$p = 0.0356$** ($<$ 0.05).
     - **$L_2$ Euclidean Distance**: $0.1078$, exact **$p = 0.0468$** ($<$ 0.05).
     - Multiple individual syllables show robust divergence ($p < 0.02$): Syllable 20 ($p=0.0042$), Syllable 26 ($p=0.0089$), Syllable 27 ($p=0.0183$), Syllable 16 ($p=0.0197$), Syllable 32 ($p=0.0210$).

2. **At the high-level macro-state stage (shMoSeq HHMM)**:
   - When syllables are clustered into higher-order behavioral states using state-of-the-art **shMoSeq** (Hierarchical Hidden Markov Model from the Sandeep Datta Lab; Weinreb et al., *Neuron* 2026):
     - **8-State Model ($K=8$)**: Total Variation exact **$p = 0.754$** (Null median $0.4135$); Drug response $\Delta L_2$ exact **$p = 0.586$**; **0 of 8 states** survive Benjamini-Hochberg FDR ($q < 0.05$).
     - **12-State Model ($K=12$)**: Total Variation exact **$p = 0.289$**; Drug response $\Delta L_2$ exact **$p = 0.761$**; **0 of 12 states** survive FDR ($q < 0.05$).
   - **The higher-order model completely erased the statistical significance that was demonstrably present in the raw syllables.**

---

## 2. Experimental Design & Computational Pipeline

### A. Animals and Cohorts
- **Subjects**: $N=15$ mice (10 Wild-Type, 5 DAT-Het knock-in).
- **Sessions**: Each mouse was recorded twice in an open-field arena across consecutive days:
  - Day 1: **Saline** (Baseline spontaneous exploration, 60 min).
  - Day 2: **Amphetamine** (Hyperlocomotion / stereotypy challenge, 60 min).
- **Total Dataset**: 30 paired sessions $\times$ 142,800 frames per session ($40.01$ Hz camera, $\sim 4.28$ million frames total).
- **Physical Alignment**: All sequences strictly aligned to start at the physical arena return frame (`detected_return_frame`) following intraperitoneal injection, eliminating human handling artifacts.

### B. Tracking & KPMS Architecture
- **Keypoints**: 17 anatomical keypoints tracked via DeepLabCut HRNet-W32.
- **Keypoint Placement**: Tracing the full animal perimeter and axial spine:
  - Axial: `nose`, `head_midpoint`, `neck`, `mid_back`, `mouse_center`, `mid_backend`, `mid_backend2`, `mid_backend3`, `tail_base`.
  - Lateral Perimeter: `left_ear_tip`, `right_ear_tip`, `left_shoulder`, `right_shoulder`, `left_midside`, `right_midside`, `left_hip`, `right_hip`.
- **Keypoint-MoSeq (KPMS)**: Egocentrically centered and rotation-aligned, PCA down to 10 latent dimensions, modeled with an autoregressive hidden Markov model (AR-HMM, 100 discrete syllables, lag 3).

### C. shMoSeq Architecture
- **Hierarchical Hidden Markov Model (HHMM)**: Discrete syllable sequence $S_t \in \{0, \dots, 99\}$ emitted by macro-state $Z_t \in \{0, \dots, K-1\}$.
- **Prior Configuration**:
  - `trans_kappa: 5.0e5` (Sticky transition prior)
  - `trans_beta: 1.0`
  - `emission_base_sigma: 1.0`, `emission_biases_sigma: 1.0`
  - `emission_gd_iters: 100`, `emission_gd_lr: 5e-3`
- **Fitting**: Condition-blind Gibbs sampling on NVIDIA RTX 4090 (JAX GPU backend), 50 iterations, monotonic log-joint convergence.

---

## 3. Empirical Discovery 1: Morphological Confounding in Pose Tracking

### The Physical Intuition
Because keypoints are distributed along the animal perimeter and axial skeleton, **raw pose coordinates are sensitive to the static body dimensions (morphology) of each individual mouse**.

### Data-Driven Validation:
We measured physical body dimensions across all 30 sessions using high-confidence frames:
- **Spine Length (Neck to Tail-Base)**: Ranges from **37.85 px to 53.40 px** (**41.1% variation** across mice).
- **Body Width (Left Midside to Right Midside)**: Ranges from **23.25 px to 29.91 px** (**28.6% variation** across mice).
- **Aspect Ratio**: Ranges from **1.60 to 2.04**.

### Intra-Animal Stability Across Separate Days (Saline vs. Amphetamine):
When comparing Day 1 (Saline) vs. Day 2 (Amphetamine) for the *exact same animal*:
- **Spine Length Correlation**: $\mathbf{r = +0.8781}$
- **Body Width Correlation**: $\mathbf{r = +0.9254}$
- **Aspect Ratio Correlation**: $\mathbf{r = +0.7995}$

Each mouse exhibits an invariant physical morphological fingerprint.

### Direct Coupling to Keypoint-MoSeq Latent Dimensions:
Keypoint-MoSeq aligns heading and center-of-mass, but **does not perform isometric scale normalization**. When we correlated the mean latent PCA coordinates with static physical measurements across animals:
- **Latent $\mathbf{PC_0}$ with Spine Length**: $\mathbf{r = +0.9280}$ ($p < 10^{-6}$)
- **Latent $\mathbf{PC_4}$ with Body Width**: $\mathbf{r = +0.8731}$ ($p < 10^{-5}$)

**Conclusion**: Latent $PC_0$ and $PC_4$ do not primarily capture dynamic movement—they capture whether the mouse is physically long or wide. Consequently, discrete syllables inherit animal-specific morphological offsets.

---

## 4. Empirical Discovery 2: The shMoSeq Hyper-Sticky ($\kappa$) Collapse

Despite morphological leakage, the 100-syllable distribution retains significant genotype information ($p = 0.0356$) because movements vary across genotypes. 

Why did shMoSeq destroy this significance?

### The Mechanism of State Entrapment:
1. shMoSeq uses a sticky transition prior $\kappa = 500,000$.
2. This creates an overwhelming mathematical penalty against switching macro-states:
   $$P(Z_{t+1} = k \mid Z_t = k) \gg P(Z_{t+1} \neq k \mid Z_t = k)$$
3. In practice, 18 of the 30 sessions feature a single macro-state occupying $\ge 80\%$ of the entire 60-minute session (episode lengths up to 30–58 minutes).
4. Animals become trapped in idiosyncratic states:
   - WT Mouse 2192: 85% of Saline session in State $Z_2$.
   - WT Mouse 2191: 78% of Saline session in State $Z_3$.
   - DAT-Het Mouse 2174: 82% of Saline session in State $Z_4$.
   - DAT-Het Mouse 2182: 75% of Saline session in State $Z_9$.
5. When each mouse is transformed into a nearly one-hot state vector, **within-genotype variance explodes**.
6. Under the exact combinatorial animal permutation test ($\binom{15}{8} = 6,435$ partitions), random animal shuffles frequently generate large group differences simply because individual mice are so extreme. The true genotype difference is rendered indistinguishable from random animal draws ($p = 0.754$ for $K=8$, $p = 0.289$ for $K=12$).

---

## 5. Comparative Statistics Summary

| Feature / Model | Raw KPMS Syllables (100-dim) | shMoSeq $K=8$ Model | shMoSeq $K=12$ Model |
| :--- | :---: | :---: | :---: |
| **Multivariate Saline Distance (TV)** | $0.3540$ | $0.3297$ | $0.5951$ |
| **Exact Permutation $p$-value (TV)** | **$p = 0.03559$ (SIGNIFICANT)** | $p = 0.75416$ (Not Sig) | $p = 0.28873$ (Not Sig) |
| **Multivariate Saline Distance ($L_2$)**| $0.1078$ | $0.3022$ | $0.4439$ |
| **Exact Permutation $p$-value ($L_2$)**| **$p = 0.04678$ (SIGNIFICANT)** | $p = 0.77778$ (Not Sig) | $p = 0.23605$ (Not Sig) |
| **Drug Response Delta ($\Delta L_2$)** | — | $0.3115$ ($p = 0.586$) | $0.2341$ ($p = 0.761$) |
| **Nominally Sig Units ($p < 0.05$)** | **7 Syllables** (Top $p = 0.0042$) | 1 State ($Z_0$: $p = 0.028$) | 1 State ($Z_2$: $p = 0.028$) |
| **FDR Significant Units ($q < 0.05$)** | Retains Omnibus Sig | **0 States** ($q = 0.225$) | **0 States** ($q = 0.338$) |

---

## 6. Questions & Strategic Review Directions for the LLM

We request your expert critique on the following core questions:

1. **Pre-Processing & Decoupling Morphology from Movement**:
   - Given that Keypoint-MoSeq $PC_0$ ($r = 0.928$) and $PC_4$ ($r = 0.873$) are driven by mouse spine length and body width, what is the best practice to eliminate morphological leakage before AR-HMM fitting?
   - Should we apply Procrustes alignment (uniform isometric scaling of each frame or per-animal median skeleton scaling) to $Y$?
   - Does Keypoint-MoSeq or the Datta Lab's codebase contain an existing hyperparameter for size normalization?

2. **Transition Stickiness ($\kappa$) & Timescale Tuning**:
   - In Weinreb et al. (*Neuron* 2026), behavioral macro-states were conceptualized as "self-directed tasks" with typical durations of tens of seconds (not 40-minute monolithic locks).
   - Is $\kappa = 500,000$ excessively large for 40 Hz data, inducing artificial state entrapment?
   - What is the mathematical and empirical guidance for scaling $\kappa$ relative to frame rate and session length?

3. **Hierarchical Random Effects in shMoSeq**:
   - Currently, shMoSeq assumes all sessions share a single emission matrix $\beta_k$. 
   - Should animal identity be modeled as a hierarchical random effect (e.g. animal-specific emission offset $\beta_{k, \text{animal}} \sim \mathcal{N}(\beta_k, \sigma^2_{\text{animal}})$) so that individual syllable preferences do not force animals into separate macro-states?

4. **Phenotyping Strategy**:
   - Given that raw syllables already provide statistically valid genotype discrimination ($p = 0.0356$), what is the valid scientific role of shMoSeq in this study? Is it strictly an exploratory ethological decomposition tool rather than a classification/phenotyping tool?
