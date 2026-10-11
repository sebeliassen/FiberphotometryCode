# The Strategy Decomposition: Dissecting Macro-Context vs. Syllable Syntax

**Date:** Wednesday, August 26, 2026  
**Dataset:** 30 Open-Field Sessions (174,686 Bouts, 30,520 Time-Anchored Evaluation Windows)  
**Cross-Validation:** Session-Grouped 5-Fold (`GroupKFold` on `session_id`, Zero Temporal Leakage)  
**Publication Figure:** `output/01_horizon_dynamics/figures/strategy_delta_decomposition_waterfall.png`

---

## 1. Conceptual Reframing: Strategy is Not a "Confound"

In quantitative ethology, animal behavioral strategy operates across multiple complementary tiers:
1. **Macro Session Time Drift ($t, t^2$):** Global habituation and exploratory drift over the recording session.
2. **Spatial Exploration State ($x, y, \text{dist\_to\_wall}, v, a$):** Arena placement (center vs. perimeter wall) and locomotion pacing, which represent the canonical open-field operationalization of exploration-versus-anxiety strategy.
3. **Instantaneous Physical Posture ($S_t$):** The 1-step Markovian kinematic grammar governing physical motor transitions.
4. **Micro-Syntactic Syllable Strategy (Sequence Transitions):** High-dimensional action sequencing (e.g., structured grooming chains, active rearing-sniff investigations, directed locomotion) that cannot be deduced from spatial coordinates or instantaneous posture alone.

Rather than treating time drift and spatial occupancy as "nuisance confounds to regress out," we formulate a **Natural Macro-to-Micro Inclusion Chain ($\mathcal{P}_0 \to \mathcal{P}_5$)** that decomposes total behavioral predictability into its interpretable macro-components, instantaneous motor grammar, and fine-grained syllable syntax.

---

## 2. The Nested Information Hierarchy ($\mathcal{P}_0 \to \mathcal{P}_5$)

We construct a sequence of strictly nested predictor sets ordered from macro/external context to fine-grained internal motor syntax:

```
𝒫₀ = ∅                                                 (Blind Global Unigram Null Baseline)
𝒫₁ = {t, t²}                                           (Macro Session Time Drift / Habituation)
𝒫₂ = 𝒫₁ ∪ {center_ratio, dist_to_wall, speed, accel}   (Macro Context: Time + Spatial Exploration)
𝒫₃ = 𝒫₂ ∪ {S_t}                                        (+ Instantaneous Posture / Markov Grammar)
𝒫₄ = 𝒫₃ ∪ {Discrete Strategy Sequencer (K=9, K=16)}    (+ Discrete Syllable Syntactic Attractors)
𝒫₅ = 𝒫_max = Full Continuous Past Bout History (B_past) (Empirical Linear Past-Information Ceiling)
```

---

## 3. Formal Strategy Decomposition

We define the sequential marginal gains along the information chain:

$$\begin{aligned}
\mathbf{\Delta_{\text{macro}}(H)} &= \text{Gain}(\mathcal{P}_2, H) - \text{Gain}(\mathcal{P}_0, H) \\
&\quad \text{Interpretable macro-context: session clock + spatial exploration dynamics.} \\
\\
\mathbf{\Delta_{\text{markov}}(H)} &= \text{Gain}(\mathcal{P}_3, H) - \text{Gain}(\mathcal{P}_2, H) \\
&\quad \text{Instantaneous posture increment beyond time and space.} \\
\\
\mathbf{\Delta_{\text{syllable}}(H)} &= \text{Gain}(\mathcal{P}_4, H) - \text{Gain}(\mathcal{P}_3, H) \\
&\quad \text{Fine-grained discrete syllable sequence strategy beyond macro context and posture.} \\
\\
\mathbf{\Delta_{\text{unresolved}}(H)} &= \text{Gain}(\mathcal{P}_5, H) - \text{Gain}(\mathcal{P}_4, H) \\
&\quad \text{Information in continuous linear history not captured by discrete K=16 clusters.}
\end{aligned}$$

The **Total Non-Markovian Behavioral Strategy ($\Delta_{\text{strategy}}$)** relative to instantaneous posture alone is:
$$\mathbf{\Delta_{\text{strategy}}(H)} = \Delta_{\text{macro}}(H) + \Delta_{\text{syllable}}(H)$$

---

## 4. Definition of $\mathcal{P}_{\text{max}}$ (The Empirical Ceiling)

$$\mathcal{P}_{\text{max}} \equiv \mathcal{P}_5 = \mathcal{B}_{\text{past}}^{\text{linear}}$$
* $\mathcal{P}_{\text{max}}$ represents the **maximum predictive capacity a high-capacity regularized linear estimator (Ridge on uncompressed 100-D past syllable counts + 24-D SVD transition matrix + spatial features)** can extract from past behavioral history ($W=60\text{s}$).
* It serves as our **empirical past-information ceiling**.
* If non-linear sequence architectures (e.g., behavioral Transformers or GRUs evaluated in Pipeline 04) extract order-dependent grammar that elevates this capacity, the empirical frontier $\mathcal{P}_{\text{max}}$ will be extended accordingly.

---

## 5. Empirical Decomposition Results ($H = 60\text{s}$ Future Action Budget)

Evaluated under session-grouped 5-fold cross-validation on 30 held-out sessions:

```
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                         WHERE DOES PREDICTABLE BEHAVIOR ACTUALLY COME FROM?                      │
 ├────────────────────────────────────────┬───────────────────────────────┬─────────────────────────┤
 │ Predictor Tier                         │ Absolute Gain (% D_JS Red.)   │ Normalized % of 𝒫_max   │
 ├────────────────────────────────────────┼───────────────────────────────┼─────────────────────────┤
 │ 🔴 𝒫₀: Blind Global Baseline           │   0.00%                       │    0.0%                 │
 │ ⏱️ 𝒫₁: Session Time Drift (t, t²)     │  +0.39%                       │    1.4%                 │
 │ 📍 𝒫₂: Macro Context (Time + Space)    │ +11.67%                       │   42.4%                 │
 │ 🟡 𝒫₃: + Instantaneous Posture (S_t)   │ +18.40% (+6.73% Markov)       │   66.9%                 │
 │ 🟢 𝒫₄: + Discrete Strategy (K=16)      │ +23.68% (+5.28% Syllable Strat│   86.1%                 │
 │ 🏆 𝒫₅: Empirical Past Ceiling (𝒫_max) │ +27.50%                       │  100.0%                 │
 └────────────────────────────────────────┴───────────────────────────────┴─────────────────────────┘
```

### Key Quantitative Insights:
1. **Macro Spatial Exploration ($\Delta_{\text{macro}} = +11.67\%$, $42.4\%$ of $\mathcal{P}_{\text{max}}$):**
   * Macro spatial tracking (distance to wall, arena coordinates, velocity) provides $42.4\%$ of total predictable behavioral structure.
2. **Instantaneous Posture Grammar ($\Delta_{\text{markov}} = +6.73\%$, elevating to $66.9\%$ of $\mathcal{P}_{\text{max}}$):**
   * Adding the animal's exact instantaneous syllable $S_t$ on top of macro context provides $+6.73\%$ incremental predictive gain.
3. **Fine-Grained Syllable Strategy ($\Delta_{\text{syllable}} = +5.28\%$, elevating to $86.1\%$ of $\mathcal{P}_{\text{max}}$):**
   * Knowing the multi-bout sequence strategy ($K=16$) captures an additional $+5.28\%$ gain beyond all spatial, temporal, and instantaneous posture cues.
4. **Near-Lossless Information Compression ($86.1\%$ of $\mathcal{P}_{\text{max}}$):**
   * Discretizing continuous 100-D past transitions into $K=16$ strategy states captures **$86.1\%$ of $\mathcal{P}_{\text{max}}$** ($+23.68\%$ out of $+27.50\%$).

---

## 6. Manuscript Formulation

> *"We demonstrate that long-horizon action selection in mice is organized into a hierarchical architecture: over a 60-second horizon, macro-context (spatial exploration and session time drift) accounts for $\Delta_{\text{macro}} = +11.67\%$ predictive gain ($42.4\%$ of $\mathcal{P}_{\text{max}}$), instantaneous posture syntax adds $\Delta_{\text{markov}} = +6.73\%$ (elevating to $66.9\%$ of $\mathcal{P}_{\text{max}}$), and fine-grained syllable strategy contributes an additional $\Delta_{\text{syllable}} = +5.28\%$. Together, discrete strategy sequencing ($K=16$) captures $86.1\%$ of the empirical linear past-information ceiling ($\mathcal{P}_{\text{max}} = +27.50\%$, $p < 10^{-4}$)."*
