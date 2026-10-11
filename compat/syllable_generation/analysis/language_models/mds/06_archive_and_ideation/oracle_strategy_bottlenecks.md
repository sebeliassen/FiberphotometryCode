# 07: Oracle Strategy Bottlenecks, Forward Models, and the Predictive Frontier

---

## Executive Summary

This document formalizes the **Oracle Strategy Bottleneck & Predictive Frontier** framework. 

When evaluating behavioral language models, we face a fundamental question:
> *"When a model fails to predict the exact next syllable $s_{t+1}$, how much of that error is due to **missing high-level cognitive context** (intent/strategy) versus **irreducible stochastic motor noise** (the biomechanical butterfly effect)?"*

To answer this, we establish a theoretical **Oracle Macro-Bottleneck Ceiling** and define the full spectrum of **"In-Between" Forward Models** bridging past history, biological neuromodulation, and future ground-truth intent.

```
                                  THE PREDICTIVE FRONTIER SPECTRUM
                                  
  Tier 0            Tier 1            Tier 2            Tier 3            Tier 4            Tier 5
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ 1st-Order   │   │ Causal Past │   │ Photometry  │   │ Imagined    │   │ Horizon     │   │ Full Macro  │
│ Markov      │   │ Strategy    │   │ Intent      │   │ Future      │   │ Sweep       │   │ Oracle (120s│
│ P(s'|s)     │   │ P(s'|s,c_p) │   │ P(s'|s,DA)  │   │ P(s'|s,\hat{c)│ │ P(s'|s,c*_H)│ │ P(s'|s,c*)  │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼                 ▼                 ▼
  [Baseline: 0b]   [Inertia/Habit]   [Neural Drive]   [Forward Model]   [Multi-Scale]     [THE CEILING]
```

---

## 1. The 6-Tier Predictive Spectrum

### Tier 0: Memoryless Kinematic Baseline
* **Conditioning:** $P(s_{t+1} \mid s_t)$
* **Information Content:** $0\text{ bits}$ of macro-context.
* **Role:** Standard first-order transition matrix baseline; assumes memoryless Markovian dynamics.

---

### Tier 1: Causal Past-Inferred Strategy Model (Our Current Core Model)
* **Conditioning:** $P(s_{t+1} \mid s_t, c_{\text{past}})$, where $c_{\text{past}} \in \{1, \dots, K\}$ is clustered from the past window $[t-120\text{s}, t]$.
* **Information Content:** $\le \log_2(K)\text{ bits}$ of historical context.
* **Role:** Measures how much knowledge of *past behavioral inertia and recent action budgets* sharpens sub-second next-action prediction.

---

### Tier 2: Photometry-Coupled Biological Intent *(The Neuro In-Between)*
* **Conditioning:** $P(s_{t+1} \mid s_t, \text{DA}_t, \text{ACh}_t)$, where $\text{DA}_t$ (dLight1) and $\text{ACh}_t$ (ACh3.0) are real-time photometric signals.
* **Information Content:** Continuous neuromodulatory state.
* **Role:** Dopamine and acetylcholine transients often lead behavioral transitions by $100\text{–}500\text{ ms}$, acting as an endogenous biological "lookahead" signal before overt physical kinematics change.

---

### Tier 3: The "Imagined Future" / Forward-Model Planner *(The Cognitive In-Between)*
* **Mechanism:** A two-stage neural architecture:
  1. **Stage 1 (Macro-Planner):** A sequence model forecasts the *expected future strategy token* from past trajectory:
     $$\hat{c}_{\text{future}} = \arg\max_k P(c_{\text{future}} = k \mid \mathbf{h}_{t-120\text{s}:t})$$
  2. **Stage 2 (Micro-Controller):** Condition the sub-second transition on the *imagined future goal*:
     $$P(s_{t+1} \mid s_t, \hat{c}_{\text{future}})$$
* **Role:** Directly models **top-down cognitive forward planning** (the brain's internal model of future intent).

---

### Tier 4: The Variable Lookahead Horizon Sweep *(The Timescale In-Between)*
* **Conditioning:** $P(s_{t+1} \mid s_t, c^*_{H})$, where $c^*_{H} \in \{1, \dots, K\}$ is the Oracle token extracted over lookahead horizon $H$:
  * $H = 3\text{ seconds}$ (Micro-motif Oracle: e.g. "grooming chain initiation")
  * $H = 15\text{ seconds}$ (Meso-task Oracle: e.g. "perimeter boundary run")
  * $H = 60\text{ seconds}$ (Sub-macro Oracle)
  * $H = 120\text{ seconds}$ (Full macro Strategy Oracle)
* **Role:** Maps the **Timescale Gain Curve**, answering: *At what future horizon does knowing the macro-goal provide the greatest predictive boost?*

---

### Tier 5: The Full Macro-Oracle Ceiling *(The Absolute Upper Bound)*
* **Conditioning:** $P(s_{t+1} \mid s_t, c^*_{\text{future}})$, where $c^*_{\text{future}} \in \{1, \dots, K\}$ ($K = 8 \text{ or } 16$) is derived directly from the **ground-truth future 120-second window** $[t, t+120\text{s}]$.
* **Information Content:** $\le \log_2(K)\text{ bits}$ of ground-truth future macro-state.
* **Role:** **The theoretical maximum predictive accuracy** that *any* $K$-state discrete bottleneck could ever provide to a sub-second controller.

---

## 2. Mathematical Formalism & Key Diagnostic Metrics

### 2.1 Information Bottleneck & Syllable Cross-Entropy
Let $\mathcal{L}(\mathcal{M})$ be the cross-entropy loss (or perplexity) of model $\mathcal{M}$ over test sequence $S = (s_1, \dots, s_T)$:

$$\mathcal{L}(\mathcal{M}) = -\frac{1}{T} \sum_{t=1}^{T} \log_2 P_{\mathcal{M}}(s_{t+1} \mid \text{Context}_t)$$

We compute:
* $\mathcal{L}_0 = \mathcal{L}(\text{Tier 0: 1st-Order Markov})$
* $\mathcal{L}_{\text{causal}} = \mathcal{L}(\text{Tier 1: Causal Past})$
* $\mathcal{L}_{\text{imagined}} = \mathcal{L}(\text{Tier 3: Forward Model})$
* $\mathcal{L}^*_K = \mathcal{L}(\text{Tier 5: Oracle with } K \text{ states})$

---

### 2.2 Model Strategy Efficiency ($\eta_{\text{strat}}$)
Quantifies what percentage of the theoretically available macro-information our causal model successfully extracts from the past:

$$\eta_{\text{strat}} = \frac{\mathcal{L}_0 - \mathcal{L}_{\text{causal}}}{\mathcal{L}_0 - \mathcal{L}^*_K} \times 100\%$$

* **$\eta \to 100\%$:** The past completely determines the future macro-strategy; the animal exhibits strong behavioral continuity.
* **$\eta \ll 50\%$:** The animal frequently executes abrupt, unprompted strategy shifts that cannot be anticipated from past history alone.

---

### 2.3 Irreducible Biological Motor Noise ($\sigma^2_{\text{motor}}$)
The remaining cross-entropy under the full Oracle model defines the baseline biomechanical noise:

$$\mathcal{H}_{\text{motor}} = \mathcal{L}^*_{K=16}$$

Any error remaining when the model *knows the true future macro-budget* reflects genuine, stochastic micro-kinematic variation.

---

### 2.4 Granularity Capacity Ratio ($K=8$ vs $K=16$)
Quantifies whether $K=8$ discrete strategies is an information bottleneck bottlenecking predictability:

$$\Delta \mathcal{I}_{\text{granularity}} = \mathcal{L}^*_{K=8} - \mathcal{L}^*_{K=16}$$

If $\Delta \mathcal{I} \approx 0$, $K=8$ macro-strategies fully saturates the macro-state space.

---

## 3. Monday Implementation Checklist

When returning to the code on Monday, execute the following evaluation workflow:

- [ ] **Step 1: Extract Future Bag-of-Words (BoW):**
  * Compute rolling 120-second normalized syllable histograms $\mathbf{p}_{\text{future}}^{(120\text{s})}(t) \in \Delta^K$.
- [ ] **Step 2: Fit Oracle Strategy Tokenizer:**
  * Run K-Means / GMM with $K \in \{8, 16\}$ on all pooled future BoW vectors to assign discrete oracle token $c^*(t) \in \{1, \dots, K\}$.
- [ ] **Step 3: Train / Evaluate Conditional Transition Matrices:**
  * Fit $K$ conditional transition matrices $\mathbf{T}^{(k)}_{i, j} = P(s_{t+1} = j \mid s_t = i, c^* = k)$.
  * Evaluate test log-likelihood across Tier 0 (Markov), Tier 1 (Causal Past), Tier 3 (Imagined), and Tier 5 (Oracle).
- [ ] **Step 4: Generate Diagnostic Figures:**
  1. *Perplexity Waterfall Plot:* Bar chart comparing perplexity from Tier 0 $\to$ Tier 1 $\to$ Tier 3 $\to$ Tier 5.
  2. *Horizon Gain Curve:* Predictability gain as a function of lookahead horizon $H \in [1\text{s}, 120\text{s}]$.
  3. *Genotype Efficiency Comparison:* Contrast $\eta_{\text{strat}}$ between Monomer (healthy controls) vs. Alpha-Synuclein PFF (Parkinsonian mice).

---

## 4. Key Takeaways to Remember

1. **The Oracle is a Diagnostic Ruler, Not a Deployable Model:** It tells you the physical limit of what a $K$-state macro-summary can do.
2. **The "Imagined Future" (Tier 3) is the True Machine Learning Target:** It tests if the agent can predict its own next 2-minute goal to improve its next 400ms step.
3. **Photometry is Nature's Oracle:** $\text{dLight1}$ and $\text{ACh3.0}$ provide internal forward-looking state evidence before physical pose reflects the transition.
