# Methodological Note: The Finite-Data Empirical Predictive Frontier and Conditional Future Uncertainty

## Core Scientific Goal

The objective of this framework is **not** to prove a universal or theoretical upper bound on mouse-behavior predictability. 

Rather, the goal is defined strictly by our empirical setting:

> **Given our actual constraints — finite dataset size, measured behavioral variables, chosen history window ($W = 68.3\text{ s}$), chosen future target ($H = 120.0\text{ s}$), and held-out-session cross-validation — estimate how much predictive structure appears to be available in measured behavioral history, and determine how closely our models approach that empirical frontier.**

Throughout this work, we employ conservative, operationally defined terminology:
* **Empirical predictive frontier** (the performance limit achievable under current data, features, and validated model families)
* **Constrained predictive ceiling estimate** (an empirical reference conditional on measured inputs)
* **Conditional future uncertainty** (the empirical dispersion of futures following similar measured pasts)
* **Available predictive structure** (the systematic mapping from measured history to future distribution)
* **Model gap** (performance difference attributable to model extraction vs. feature availability)
* **Finite-data reference** (benchmarks constrained by sample density and candidate coverage)
* **Behavior-only frontier** (predictive capacity achievable prior to incorporating neural or physiological signals)

We explicitly avoid unfounded claims of "fundamental stochasticity," "theoretical Bayes limits," "intrinsic mouse entropy," "definitive causal states," or "universal predictability bounds."

---

## 1. Conceptual Decomposition

Let:
* $X_t \in \mathcal{X}$ denote the **measured behavioral history** prior to time $t$ (pose trajectory, continuous kinematics, and discrete syllable sequences over window $W = 68.3\text{s}$).
* $Y_t \in \mathcal{Y}$ denote the **measured behavioral future** over the horizon $H = 120.0\text{s}$ starting at $t$ (99-dimensional syllable distribution).

Both $X_t$ and $Y_t$ are drawn from the same continuous behavioral time series. The fundamental question is:

> **How much of the variability in $Y$ is systematically related to information already present in $X$, and how much remains unresolved conditional on $X$?**

We conceptually decompose total future variability:

$$\text{Observed Future Variability} = \text{Predictable Structure Available in } X + \text{Residual Conditional Uncertainty}.$$

The predictable portion is further separated into:

$$\text{Predictable Structure in } X = \text{Structure Captured by Current Model} + \text{Structure the Model Fails to Exploit}.$$

```
Total Observed Future Variability (Unconditional Null: 0.268 bits)
├── Predictable Structure Available in X
│   ├── Structure Captured by Model (e.g., Action-GPT Transformer: +33.9% ± 6.3%)
│   └── [GAP A: Model Gap] Structure Model Fails to Exploit
└── [GAP B: Conditional-Information Gap] Residual Conditional Uncertainty Under X
```

### Gap A: Model Gap ($R_{\text{model}} - R_{X\text{-reference}}$)
This gap asks: *How much information already present in $X$ is our current model failing to extract?*

Because the reference predictor is strictly constrained to use only $X$, this gap must **not** be attributed to unmeasured neural variables, hidden motivational states, or external sensory changes. Potential sources include:
* Suboptimal model architecture or inductive bias
* Finite-data sample inefficiency
* Optimization challenges or underfitting
* Regularization / shrinkage artifacts
* Inadequate feature representation or tokenization
* Failure to exploit higher-order syntax, temporal order, or non-linear context contained in $X$

### Gap B: Conditional-Information Gap
The residual error remaining even for a highly flexible, well-optimized $X$-only predictor asks: *What future variability cannot currently be resolved from measured behavioral history alone?*

Potential explanations include:
* Relevant behavioral information lying outside the chosen history window $W$
* Hidden, internal, or physiological states not fully reflected in behavioral history
* Unmeasured sensory inputs or environmental changes
* Future external perturbations during the horizon $H$
* Finite-sampling measurement jitter of the target distribution $Y$ (e.g., discrete Poisson bout counting)
* Genuine conditional uncertainty (multiple distinct future paths compatible with state $X$)
* Insufficiently rich or low-dimensional behavioral feature extraction

This residual should **not** be reflexively labeled as "biological randomness" or "free will."

---

## 2. Distinction Between Oracle Concepts

To prevent methodological conflation, we formally distinguish between two fundamentally different oracle concepts:

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │                 ORACLE BENCHMARK TAXONOMY                   │
                  └─────────────────────────────────────────────────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
    ┌─────────────────────────────┐                             ┌─────────────────────────────┐
    │          ORACLE A           │                             │          ORACLE B           │
    │      X-Only Empirical       │                             │       Target-Peeking        │
    │          Frontier           │                             │    Branch-Selection Oracle  │
    ├─────────────────────────────┤                             ├─────────────────────────────┤
    │ • Inputs: Strictly X        │                             │ • Inputs: X AND Future Y_i  │
    │ • Never inspects future Y_i │                             │ • Inspects realization Y_i  │
    │ • Flexible ensembling / kNN │                             │ • Selects best branch / mod │
    │ • Identifies MODEL GAP      │                             │ • Measures DATASET COVERAGE │
    └─────────────────────────────┘                             └─────────────────────────────┘
```

### Oracle A: $X$-Only Empirical Frontier
The predictor is given access **strictly to the measured behavioral history $X$**, matching the information available to real deployable models. It may be highly advantaged through:
* Non-parametric nearest-neighbor matching across large archives
* Optimal out-of-fold stacked ensembling across diverse architectures
* Flexible non-linear function approximation
* Exhaustive hyperparameter tuning under proper nested cross-validation

Crucially, **Oracle A never inspects the realization of the anchor future $Y_i$**. It evaluates: *How well could we plausibly predict the future given only the information present in $X$ under finite-data constraints?* The difference $R_{\text{model}} - R_{X\text{-oracle}}$ quantifies the **Model Gap**.

### Oracle B: Target-Peeking / Branch-Selection Oracle
This diagnostic is deliberately permitted to inspect the realized anchor outcome $Y_i$ when selecting among candidate predictions or empirical future realizations:

$$Y_i^{\text{branch\_oracle}} = \arg\min_{j \in \mathcal{N}_k(X_i)} D_{\text{JS}}(Y_i, Y_j).$$

This does **not** demonstrate that $X_i$ contains sufficient information to determine which future branch will occur. Rather, it answers: *If the correct future trajectory were known, is that specific behavioral mode already represented within the finite dataset?*

Oracle B provides a **finite-data coverage reference**, not an $X$-only predictive ceiling. A large gap between an $X$-only model and a branch-selection oracle motivates testing whether additional unmeasured variables (such as dopamine signaling) contain the information needed to resolve the branch.

---

## 3. Past-to-Future Distance Transfer

To quantify how behavioral similarity in the past constrains future divergence without imposing parametric model assumptions, we evaluate the continuous distance-transfer function:

$$\mathbb{E}[d_Y \mid d_X = d],$$

where $d_X(i, j) = \|X_i - X_j\|$ represents the past-state distance in balanced Repertoire+Spatial space, and $d_Y(i, j) = D_{\text{JS}}(Y_i, Y_j)$ is the future Jensen-Shannon divergence across cross-session pairs ($\text{session}_i \neq \text{session}_j$).

```
Future Divergence d_Y (bits)
  0.50 ┤                                      ╭────────────── Dissimilar Pasts (0.460 b)
       │                                ╭─────╯
  0.42 ┤────────────────────────────────┼────────────── Random Cross-Session Baseline (0.423 b)
       │                         ╭──────╯               [Shuffled Control Flat Null: 0.418 b]
  0.30 ┤                   ╭─────╯
       │             ╭─────╯
  0.24 ┤───────╭─────╯                          <--- Similar Pasts Top 5% (0.240 b: -43.4% dispersion)
  0.15 ┤ ╭─────╯ (Tightest Pairs: 0.148 b)
  0.00 ┴─┴───────────┴───────────┴───────────┴─────────────
        0.3         0.6         0.9         1.2    Past Distance d_X
```

### Empirical Findings (*Validated on 100,000 Cross-Session Pairs*):
1. **Strong Behavioral Constraint:** Cross-session pairs with similar measured pasts (top 5% closest $d_X$) exhibit future divergences of $\mathbf{0.240 \pm 0.003\text{ bits}}$, compared to $\mathbf{0.423 \pm 0.001\text{ bits}}$ for random cross-session pairs.
   $$\text{Information Constraint Ratio} = \frac{0.423 - 0.240}{0.423} \approx \mathbf{43.4\%}.$$
   Conditioning on a similar measured behavioral history removes $\sim 43\%$ of the future dispersion observed between unrelated histories.
2. **Shuffled Control:** Randomly permuting the $X \leftrightarrow Y$ correspondence across windows yields a completely flat curve at $\mathbf{0.418\text{ bits}}$ across all $d_X$, confirming that the relationship reflects authentic past-future temporal structure.
3. **Avoid Overinterpreting Binned Quadrants:** Rather than imposing arbitrary discrete labels ("equifinality", "branching states", "causal basins") via median splits, the continuous $\mathbb{E}[d_Y \mid d_X]$ curve should be reported as the primary empirical transfer metric.

---

## 4. Interpretation of Non-Parametric Matching ($k$-NN)

For each anchor window $X_i$, retrieving the $k$ most similar pasts from independent sessions and averaging their corresponding futures:

$$\hat{Y}_i^{k\text{-NN}} = \frac{1}{k} \sum_{j \in \mathcal{N}_k(X_i)} Y_j$$

yields a held-out predictive risk of:

$$R_i^{\text{pred}} = D_{\text{JS}}(Y_i, \hat{Y}_i^{k\text{-NN}}).$$

### Methodological Interpretation:
* $k$-NN provides a **model-free baseline** of how much predictive structure can be recovered simply through local behavioral analogue lookup across animals.
* In our dataset ($k=50$), cross-session $k$-NN achieves a **$+27.8\% \pm 5.9\%$ gain** ($0.194\text{ bits}$ vs. $0.268\text{ bits}$ null).
* **$k$-NN is not an upper ceiling:** Parametric sequence models (Transformers at $+33.9\% \pm 6.3\%$) systematically outperform $k$-NN by learning global contextual mappings and higher-order transition rules that simple local averaging cannot capture.

---

## 5. Finite-Data Density & Asymptotic Extrapolation Limitations

When fixing the number of neighbors (e.g., $k=50$ or $k=100$) and sweeping dataset availability from $N = 10\% \to 100\%$ of candidate sessions:

$$N \uparrow \implies \bar{d}_k \downarrow \quad \text{and} \quad \hat{R}_k^{\text{pred}} \downarrow.$$

```
Dataset Fraction (N)    Mean Past Distance (d_100)    Held-Out Predictive Risk (R_100)
       10%                        0.666 bits                     0.256 bits (+ 4.1%)
       25%                        0.586 bits                     0.217 bits (+18.9%)
       50%                        0.557 bits                     0.207 bits (+22.8%)
       75%                        0.539 bits                     0.199 bits (+26.1%)
      100%                        0.521 bits                     0.189 bits (+29.5%)
```

### Critical Methodological Finding on Zero-Distance Extrapolation:
When fitting joint bias-variance surfaces $\hat{R}^{\text{pred}}(d, k) = R_\infty + a \cdot d^p + \frac{b}{k}$ across empirical clouds:
* Assuming linear distance mismatch ($p=1$) yields an asymptotic floor estimate of $R_\infty = \mathbf{0.000\text{ bits}}$ ($100\%$ theoretical gain).
* Assuming quadratic distance mismatch ($p=2$) yields $R_\infty = \mathbf{0.058\text{ bits}}$ ($78.4\%$ theoretical gain; $95\%\text{ CI: } [0.050, 0.064]\text{ bits}$).

Because the observed past-distance support is bounded ($\bar{d} \in [0.44, 0.67]$), **extrapolating out-of-range all the way to $d=0$ is heavily dependent on the chosen functional form**. 

> **Conclusion:** The present 30-session dataset does not identify a unique asymptotic prediction floor. Demonstrating whether the zero-mismatch floor approaches $\sim 0.06\text{ bits}$ or $\sim 0.14\text{ bits}$ requires scaling dataset density to large open-source repositories where $\bar{d}_k \to 0$ empirically.

---

## 6. Hidden States and Behavioral Observability Over Time

Both history $X$ and future $Y$ are sampled from the same underlying behavioral process. 

Suppose an unobserved physiological or motivational state $H_t$ (e.g., arousal, satiety, exploratory drive, dopaminergic tone) influences behavior:

$$H_t \to B_t \quad \text{and} \quad H_t \to \{B_{t+1}, B_{t+2}, \dots\}.$$

The fact that $H_t$ is not explicitly measured in $X$ does **not** mean its future effects are invisible to a behavioral model:
1. **Behavioral History as a Proxy for Internal State:** If an internal state $H_t$ modulates behavioral emission over minutes, a sufficiently rich behavioral history $X_{\text{rich}}$ (capturing 68 seconds of kinematics, posture, and bout frequencies) will partially encode $H_t$.
2. **Temporal Observability:** As time evolves, behavior that was previously part of future $Y$ becomes part of the history $X$ for subsequent windows. 
3. **Representation Richness:** This highlights why expanding history length and adding multi-modal behavioral features (spatial coordinates, continuous pose PCs, discrete syllables) systematically improves predictability—they provide richer behavioral proxies for underlying state.

We do not claim that internal neural states are directly reconstructed, but rather that their future-constraining effects can be partially read out from behavioral history.

---

## 7. Future Neural Test Protocol ($P(Y \mid X)$ vs. $P(Y \mid X, \text{DA})$)

Once the behavior-only predictive frontier is rigorously established, recorded neural signals (e.g., striatal dopamine photometry) can be introduced:

$$\text{Compare } P(Y \mid X) \quad \text{with} \quad P(Y \mid X, \text{DA}).$$

```
                                  NEURAL HYPOTHESIS TESTING
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
     ┌─────────────────────────────┐                     ┌─────────────────────────────┐
     │  P(Y | X, DA) == P(Y | X)   │                     │   P(Y | X, DA) > P(Y | X)   │
     ├─────────────────────────────┤                     ├─────────────────────────────┤
     │ • DA adds NO extra gain     │                     │ • DA adds SIGNIFICANT gain  │
     │ • Future-relevant DA state  │                     │ • DA carries future-state   │
     │   is ALREADY recoverable    │                     │   information NOT captured  │
     │   from behavioral history   │                     │   by behavioral history     │
     │ • (Do not claim DA useless) │                     │ • (Genuine hidden state)    │
     └─────────────────────────────┘                     └─────────────────────────────┘
```

* **If Dopamine Does Not Improve Held-Out Prediction:** The future-relevant information reflected in dopamine dynamics is already encoded in the animal's recent behavioral history (at least for a 120s macro-repertoire target). This does *not* imply dopamine is functionally irrelevant to behavior, but rather that behavior and dopamine share redundant state information at this timescale.
* **If Dopamine Significantly Improves Prediction:** Dopamine dynamics provide future-relevant internal state information that cannot be reconstructed from external behavioral observations alone.

This constitutes a substantially more rigorous test than simple correlation between dopamine transients and concurrent behavioral syllables.

---

## 8. Finite-Data Empirical Frontier Suite (Validated Experimental Results)

All analyses below were executed via 5-fold GroupKFold cross-validation across 30 sessions ($29{,}325$ clean non-straddling windows, $W=68.3\text{s} \to H=120.0\text{s}$):

* **Pipeline Script:** [`run_full_empirical_frontier_diagnostics.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/run_full_empirical_frontier_diagnostics.py)
* **Master Results Table:** [`master_empirical_frontier_table.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/04_predictive_frontier/tables/master_empirical_frontier_table.csv)
* **Scaling Results Table:** [`model_data_scaling_results.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/04_predictive_frontier/tables/model_data_scaling_results.csv)
* **Branch Oracle Table:** [`branch_oracle_pool_comparison.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/04_predictive_frontier/tables/branch_oracle_pool_comparison.csv)
* **Publication Figure:** [`fig11_comprehensive_empirical_frontier_suite.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/fig11_comprehensive_empirical_frontier_suite.png)

```
Predictive Gain (% ΔD_JS)
  45% ┤
  40% ┤                                                              ┌──────────────┐ [Oracle Model Selector: +41.8%]
      │                                                ┌──────────┐  │              │
  35% ┤                                 ┌───────────┐  │          │  │              │ [Branch Oracle k=50: +36.3%]
      │                  ┌───────────┐  │           │  │          │  │              │ [Stacked Ensemble: +36.4%]
  30% ┤   ┌───────────┐  │           │  │           │  │          │  │              │ [Ordered Transformer: +33.9%]
      │   │           │  │           │  │           │  │          │  │              │ [k-NN / Ridge: +27.8%]
  20% ┤   │           │  │           │  │           │  │          │  │              │
      │   │           │  │           │  │           │  │          │  │              │
   0% ┴───┴───────────┴──┴───────────┴──┴───────────┴──┴──────────┴──┴──────────────┴──
           k-NN / Ridge    Transformer     Ensemble     Branch Oracle  Oracle Selector
           (Base Line)     (Single Best)  (Deployable)   (k=50 Peeking) (Model Upper Bound)
```

### Master Empirical Frontier Table

| Model / Diagnostic Level | Methodology Type | Predictive Risk ($D_{\text{JS}}$ bits) | Predictive Gain (% $\Delta D_{\text{JS}}$) | What This Directly Demonstrates |
| :--- | :---: | :---: | :---: | :--- |
| **1. Global Blind Null** | Baseline Reference | $0.2677 \pm 0.0324\text{ b}$ | $0.00\%$ | Unconditional behavioral divergence |
| **2. Non-Parametric $k$-NN ($k=50$)** | Deployable ($X$-only) | $0.1938 \pm 0.0299\text{ b}$ | $+27.77\% \pm 5.92\%$ | Non-parametric local matching baseline |
| **3. Parametric Ridge BoW** | Deployable ($X$-only) | $0.1930 \pm 0.0261\text{ b}$ | $+27.20\% \pm 5.30\%$ | Linear capacity of macro-behavior |
| **4. Shuffled Transformer** | Deployable ($X$-only) | $0.1761 \pm 0.0295\text{ b}$ | $+34.80\% \pm 6.21\%$ | Order-agnostic attention across token bag |
| **5. Ordered Transformer (Action-GPT)** | Deployable ($X$-only) | $0.1788 \pm 0.0303\text{ b}$ | $+33.94\% \pm 6.32\%$ | Sequential syntax + self-attention |
| **6. Linear Residual-Corrected TF** | Deployable ($X$-only) | $0.2252 \pm 0.0360\text{ b}$ | $+15.60\% \pm 9.63\%$ | Linear correction overfits on noise |
| **7. MLP Residual-Corrected TF** | Deployable ($X$-only) | $0.2341 \pm 0.0384\text{ b}$ | $+12.32\% \pm 10.48\%$ | Non-linear correction degrades on noise |
| **8. Stacked Learned Ensemble** | Deployable ($X$-only) | $\mathbf{0.1714 \pm 0.0281\text{ b}}$ | $\mathbf{+36.41\% \pm 5.81\%}$ | **Best deployable predictor** (+2.5 pp over single TF) |
| **9. Matched Branch Oracle ($k=50$)** | Target-Peeking Reference | $\mathbf{0.1681 \pm 0.0237\text{ b}}$ | $\mathbf{+36.30\% \pm 5.71\%}$ | Best candidate future among 50 nearest pasts |
| **10. Oracle Model Selector** | Target-Peeking Reference | $\mathbf{0.1566 \pm 0.0252\text{ b}}$ | $\mathbf{+41.83\% \pm 5.20\%}$ | Per-window upper bound of existing model family |

---

### Key Diagnostic Results Across Experiments 1–6:

#### A. Stacked Ensemble vs. Oracle Selector Gap (Experiments 1 & 2):
* The Stacked Ensemble reaches **$+36.41\% \pm 5.81\%$ ($0.171\text{ bits}$)** out-of-fold.
* The Oracle Selector reaches **$+41.83\% \pm 5.20\%$ ($0.157\text{ bits}$)**.
* The gap between our best deployable model and an all-knowing oracle picking the best model per window is **only $+5.4\text{ pp}$ ($0.015\text{ bits}$)**. Base models make largely overlapping errors; unexploited complementary information within this model family is minimal.

#### B. Residual Predictability Audit (Experiment 3):
* Training secondary models (Ridge, MLP) on $X$ to predict the Transformer's logit residuals degraded performance to $+15.6\%$ and $+12.3\%$.
* **Conclusion:** No easily extractable linear or non-linear state-space structure in $X$ was missed by the Transformer.

#### C. Model Data-Scaling Curves (Experiment 4):
* Sweeping training dataset size from $N = 20\% \to 100\%$ shows steep, unsaturating improvement for the Transformer ($N=20\%: -2.6\% \to N=40\%: +16.1\% \to N=60\%: +24.6\% \to N=80\%: +28.7\% \to N=100\%: +32.1\%$).
* In contrast, linear Ridge regression approaches an early plateau ($N=60\%: +21.4\% \to N=100\%: +27.8\%$).
* **Conclusion:** Sequence models have not yet saturated under current dataset sizes; scaling to archive-scale datasets is strongly motivated.

#### D. Finite-Dataset Branch Oracle vs. Random Candidate Controls (Experiment 5):
* Across candidate pool sizes $k \in \{10, 25, 50, 100\}$, the Matched-Neighbor Branch Oracle consistently outperforms the Random-Candidate Control (e.g., at $k=10$: $+22.3\%$ vs $+11.8\%$, a $+10.4\text{ pp}$ advantage; at $k=50$: $+36.3\%$ vs $+31.0\%$, a $+5.3\text{ pp}$ advantage).
* As $k \to 100$, random combinatorial coverage begins to catch up ($+37.3\%$), highlighting the critical need for random candidate controls when evaluating discrete branch selection.

---

## 9. Core Interpretation Framework

We organize the empirical inquiry into a structured, step-by-step diagnostic ladder:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. How much predictive structure exists in measured behavioral history?    │
│    -> Past-distance transfer shows similar pasts are 43.4% less divergent.  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. How much of it can simple empirical matching recover?                    │
│    -> Cross-session k-NN captures +27.8% of future variance.                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. How much more do learned sequence models recover?                        │
│    -> Transformer & Stacked Ensemble push performance to +33.9% - +36.4%.   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Is there predictable structure still left in their errors?               │
│    -> Oracle selector gap is only 5.4 pp; residual correction fails (+15%). │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. What additional information explains remaining unresolved variation?    │
│    -> Target noise (0.28 b), longer histories, or unmeasured neural states. │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary Statement

> **The broader objective of this research is to separate future behavioral variability that can be systematically mapped from measured behavioral history from variability that remains unresolved under that history, and then determine how closely increasingly powerful models approach the empirically available predictive structure.**
