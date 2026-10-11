# Information-Theoretic Metrics of Behavioral Complexity and Memory Depth

## Executive Summary

Standard single-step transition probabilities and confusion matrices fail to capture the multi-scale temporal complexity of animal behavior. This document details the mathematical and algorithmic formulations of **information-theoretic metrics** designed to quantify long-range behavioral memory, non-Markovian structure, and instantaneous predictability from discrete action bout sequences.

We formulate:
1. **Predictive Information & Excess Entropy** (Bialek, Nemenman, Tishby) to measure global memory capacity.
2. **Active Information Storage (AIS) & Local AIS (LAIS)** (Lizier et al.) to track moment-to-moment behavioral rigidity vs. exploratory switching.
3. **Renewal Entropy Rate** ($h_{\text{renewal}}$) to integrate discrete syllable syntax with continuous physical dwell times.
4. **Multi-Horizon Cross-Entropy & Perplexity** across extended prediction horizons ($h \in [1, 120]$ bouts).

---

## 1. Excess Entropy & Predictive Information

```
                 Past Behavioral History (X_past)        Future Trajectory (X_future)
             [ ... s_{m-3}, s_{m-2}, s_{m-1} ]  ───►  [ s_m, s_{m+1}, s_{m+2}, ... ]
                                     \                    /
                                      \                  /
                             Mutual Information: I(X_past ; X_future)
                                      = PREDICTIVE INFORMATION
```

### Mathematical Formulation (Bialek et al., 2001)
Predictive Information (also known as Excess Entropy or Effective Measure Complexity) quantifies the total mutual information shared between the semi-infinite past history and the future trajectory of a behavioral sequence $\mathbf{X} = (s_1, s_2, \dots, s_T)$:

$$I_{\text{pred}}(T) = I(\mathbf{X}_{\text{past}}; \mathbf{X}_{\text{future}}) = \lim_{T \to \infty} \left[ \sum_{t=1}^T H(X_t \mid X_{t-1}, \dots, X_1) - T \cdot h_\mu \right]$$

where $h_\mu$ is the asymptotic entropy rate of the behavioral process:
$$h_\mu = \lim_{L \to \infty} \frac{H(X_1, \dots, X_L)}{L} = \lim_{L \to \infty} H(X_L \mid X_{L-1}, \dots, X_1)$$

### Scaling Regimes: Markovian vs. Complex Biological Behavior
The scaling behavior of $I_{\text{pred}}(T)$ as a function of observation window $T$ diagnoses the underlying computational depth of the animal:

```
Predictive Information I_pred(T)
   ▲
   │               Complex Biological Behavior: I_pred(T) ~ T^gamma (Power-Law Memory)
   │               *  *  *  *  *  *  *  *
   │            *
   │          *
   │        *
   │       *
   │      *────────────────────────────────── 1st-Order Markov Chain: I_pred(T) -> Constant
   │     *
   └────┴────────────────────────────────────► Observation Window T
```

1. **Memoryless / 1st-Order Markov Process:**
   * $I_{\text{pred}}(T)$ rapidly saturates to a small finite constant proportional to the state space:
     $$I_{\text{pred}}(T) \to \text{const} = H(X) - h_\mu$$
   * Future choices are constrained only by the single preceding state ($h \le 1$).
2. **Deterministic Periodic Systems:**
   * $I_{\text{pred}}(T) \propto \log(T)$.
3. **Complex Biological Systems (Non-Markovian Ethology):**
   * $I_{\text{pred}}(T)$ exhibits sub-extensive, power-law divergence:
     $$I_{\text{pred}}(T) \propto T^\gamma \quad (0 < \gamma < 1)$$
   * This proves that past behavioral choices continually constrain future actions across extended horizons of 30 to 60+ seconds ($T = 60\text{–}120$ bouts).

---

## 2. Active Information Storage (AIS) & Local AIS (LAIS)

While Predictive Information provides a global metric across an entire recording, **Active Information Storage (AIS)** measures the amount of information currently in use by an animal to determine its next action.

```
                  Past History Embedding: X_{t-1}^{(k)} = [s_{t-k}, ..., s_{t-1}]
                                          │
                                          ▼
                               Current State: s_t
                                          │
                  AIS: A_X(k) = I( X_{t-1}^{(k)} ; s_t )  [Global Average]
                  LAIS: a_X(t, k) = log2 [ P(s_t | History) / P(s_t) ]  [Local Instantaneous]
```

### A. Global Active Information Storage (Lizier et al., 2012)
AIS quantifies the mutual information between the past history state $\mathbf{X}_{t-1}^{(k)} = (s_{t-1}, s_{t-2}, \dots, s_{t-k})$ and the next state $s_t$:

$$A_X(k) = I(\mathbf{X}_{t-1}^{(k)}; X_t) = H(X_t) - H(X_t \mid \mathbf{X}_{t-1}^{(k)}) = \sum_{x_t, \mathbf{x}_{t-1}^{(k)}} P(x_t, \mathbf{x}_{t-1}^{(k)}) \log_2 \frac{P(x_t \mid \mathbf{x}_{t-1}^{(k)})}{P(x_t)}$$

* **Theoretical Bounds:** $0 \le A_X(k) \le H(X_t) \le \log_2 K$.
* **Interpretation:** Measures the fraction of the animal's next action that is deterministic and self-generated from internal memory vs. driven by stochastic noise or external sensory perturbations.

### B. Local Active Information Storage (LAIS): Instantaneous Predictability
By evaluating the log-probability ratio point-by-point in time, **Local AIS** produces an instantaneous time-series signal $a_X(t, k)$ aligned with the behavioral stream:

$$a_X(t, k) = \log_2 \frac{P(x_t \mid \mathbf{x}_{t-1}^{(k)})}{P(x_t)}$$

```
LAIS a_X(t, k) (bits)
   ▲
+4 │    Stereotyped Syntax (Grooming / Fast Locomotion): High Predictability [a_X(t) >> 0]
   │    ╭──────╮
 0 ┼────┴──────┴───────────┬───────────────┬────────────────► Session Time (t)
   │                       ╰───────────────╯
-3 │       Stochastic Exploration / Strategic Switching: High Surprise [a_X(t) < 0]
```

* **When $a_X(t, k) > 0$ (High Predictability / Rigidity):** The observed syllable $x_t$ was heavily expected given history $\mathbf{x}_{t-1}^{(k)}$ ($P(x_t \mid \text{past}) > P(x_t)$). Indicates stereotyped motor subroutines (e.g., continuous locomotion strides, grooming chains).
* **When $a_X(t, k) < 0$ (Surprise / Behavioral Switching):** The observed syllable $x_t$ was unexpected given history ($P(x_t \mid \text{past}) < P(x_t)$). Pinpoints the exact moment an animal **aborts a motor program or executes a strategic switch**.

---

## 3. History Embedding Optimization & Bias Correction

To compute AIS accurately on discrete biological sequences without overfitting, we must select the optimal history length $k$ and correct for finite-sample estimation bias.

### A. Non-Uniform History Selection (Ragwitz Criteria / Forward Selection)
Instead of arbitrary uniform lag embedding $\mathbf{x}_{t-1}^{(k)}$, we use iterative forward selection maximizing conditional mutual information:

$$s_{t-l}^* = \arg\max_{s_{t-l}} I(s_{t-l}; s_t \mid \mathbf{x}_{\text{selected}})$$

Terminating when $I(s_{t-l}^*; s_t \mid \mathbf{x}_{\text{selected}})$ fails to exceed an empirical null threshold determined by surrogate permutation testing ($\alpha = 0.01$).

### B. Finite-Sample Bias Correction (Grassberger / Miller-Madow)
Naive plug-in entropy estimators overestimate mutual information on finite sequences ($T \approx 10^4$). We apply the Miller-Madow and Grassberger analytical bias corrections:

$$\hat{H}_{\text{MM}}(X) = \hat{H}_{\text{plugin}}(X) + \frac{|\mathcal{X}_{\text{active}}| - 1}{2T}$$

$$A_X^{\text{corrected}}(k) = \hat{A}_X(k) - \mathbb{E}_{\pi}[\hat{A}_X^{\text{shuffled}}(k)]$$

where $\mathbb{E}_{\pi}$ is the mean AIS obtained by shuffling session syllable sequences while preserving unigram marginal frequencies.

---

## 4. Renewal Entropy Rate: Integrating Syntax with Dwell Times

Because behavioral syllables vary continuously in physical duration $d_m$ (from 100 ms to 1,500 ms), sequence entropy must be normalized by real physical time.

```
Action Stream: [ (Syllable 14, 250ms), (Syllable 4, 800ms), (Syllable 18, 400ms) ]
                    │                        │                    │
                    ▼                        ▼                    ▼
             Entropy: H(s_1)          Entropy: H(s_2)      Entropy: H(s_3)
             ─────────────────────────────────────────────────────────────
                             Mean Duration: E[D_m] = 483 ms
```

### Mathematical Formulation
The **Renewal Entropy Rate** ($h_{\text{renewal}}$) integrates the conditional entropy of discrete syllable transitions with the expected dwell time $\mathbb{E}[D_m]$:

$$h_{\text{renewal}} = \frac{H(s_m \mid s_{m-1}, s_{m-2}, \dots)}{\mathbb{E}[D_m]} \quad \left[\frac{\text{bits}}{\text{second}}\right]$$

where $\mathbb{E}[D_m] = \frac{1}{M} \sum_{m=1}^M d_m$ is the mean physical duration per bout (in seconds).

### Universal Estimation via Context Tree Weighting (CTW)
Context Tree Weighting computes an exact theoretical upper bound on $H(s_m \mid \text{History})$:
$$\hat{H}_{\text{CTW}}(\mathbf{S}) = -\frac{1}{M} \log_2 P_{\text{CTW}}(s_1, \dots, s_M)$$
$$\hat{h}_{\text{renewal}} = \frac{\hat{H}_{\text{CTW}}(\mathbf{S})}{\mathbb{E}[D_m]} \quad \left[\frac{\text{bits}}{\text{second}}\right]$$

---

## 5. Multi-Horizon Cross-Entropy & Perplexity Forecasting

To quantify the preservation of non-Markovian memory across physical time, we evaluate predictive loss across a horizon sweep $h \in [1, 120]$ discrete bouts.

```
Current Bout: s_m ───► Macro-Strategy: S_m ───► Forecast at Horizon h: P(s_{m+h} | s_m, S_m)
```

### Mathematical Formulations

1. **Horizon Cross-Entropy Loss ($\mathcal{H}(h)$):**
   $$\mathcal{H}_{\text{Strategy}}(h) = -\frac{1}{N} \sum_{i=1}^N \ln P(s_{i+h} \mid s_i, \mathcal{S}_i)$$
   $$\mathcal{H}_{\text{Markov}}(h) = -\frac{1}{N} \sum_{i=1}^N \ln P(s_{i+h} \mid s_i)$$
2. **Strategy Forecasting Gain ($\Delta \mathcal{H}(h)$):**
   $$\Delta \mathcal{H}(h) = \mathcal{H}_{\text{Markov}}(h) - \mathcal{H}_{\text{Strategy}}(h) = \mathcal{I}(s_{m+h}; \mathcal{S}_m \mid s_m)$$
3. **Preserved Predictive Ceiling Fraction ($\% \text{ Ceiling}$):**
   $$\% \text{ Ceiling Preserved}(h) = \frac{\mathcal{H}_{\text{Uniform}} - \mathcal{H}_{\text{Strategy}}(h)}{\mathcal{H}_{\text{Uniform}} - \mathcal{H}_{\text{Strategy}}(h=1)} \times 100\%$$

---

## 6. Comprehensive Information-Theoretic Summary Table

| Metric | Mathematical Formula | Units | Temporal Scope | Primary Ethological / Biological Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **1st-Order Markov Perplexity** | $2^{\mathcal{H}(h=1)}$ where $\mathcal{H} = -\sum P(s_i, s_j) \log_2 P(s_j \mid s_i)$ | Effective states | Single bout ($h=1$) | Immediate physical transition randomness |
| **Predictive Information ($I_{\text{pred}}$)** | $\lim_{T \to \infty} [\sum_{t=1}^T H(X_t \mid X_{<t}) - T \cdot h_\mu]$ | bits | Global ($T \to \infty$) | Total historical memory capacity required for optimal long-term forecasting |
| **Active Information Storage (AIS)** | $I(\mathbf{X}_{t-1}^{(k)}; X_t) = H(X_t) - H(X_t \mid \mathbf{X}_{t-1}^{(k)})$ | bits | Past window $k$ bouts | Fraction of current choice dictated by internal history vs. external stochasticity |
| **Local AIS (LAIS)** | $a_X(t, k) = \log_2 \frac{P(x_t \mid \mathbf{x}_{t-1}^{(k)})}{P(x_t)}$ | bits | Instantaneous bout $t$ | Real-time index of motor stereotypy ($>0$) vs. cognitive strategic switching ($<0$) |
| **Renewal Entropy Rate ($h_{\text{renewal}}$)** | $\frac{H(s_m \mid s_{<m})}{\mathbb{E}[D_m]}$ | bits / second | Continuous physical time | True information generation density per unit of physical time |
| **Strategy Mutual Information ($\mathcal{I}_h$)** | $\mathcal{H}_{\text{Markov}}(h) - \mathcal{H}_{\text{Strategy}}(h)$ | nats or bits | Multi-horizon ($h \in [1, 120]$) | Information about future action $s_{m+h}$ stored in macro-strategy state $\mathcal{S}_m$ |

---

## 7. Python Implementation Blueprint

To deploy these metrics on our Keypoint-MoSeq dataset:
* **Toolbox:** `idtxl` (Information Dynamics Toolkit xl) or custom Numba-accelerated discrete estimators.
* **Algorithm:** Compute LAIS $a_X(t, k=5)$ across all 110 sessions to generate a continuous 40-minute "Behavioral Flexibility / Rigidity Curve" contrasting Parkinsonian PFF vs. Monomer cohorts.
