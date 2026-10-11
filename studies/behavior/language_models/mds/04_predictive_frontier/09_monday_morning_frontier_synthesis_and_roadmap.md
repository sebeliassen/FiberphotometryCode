# Monday Morning Briefing: Predictive Frontier Synthesis & Uncertainty Roadmap

---

## Predictive Frontier — What Are We Actually Trying to Do?

We have mouse open-field behavior segmented into MoSeq syllables. For each moment, we define:

$$
X = \text{recent behavioral history}
$$

(currently around $68.3\text{ s}$), and

$$
Y = \text{syllable repertoire over the next } 120.0\text{ s}.
$$

The original $P_n$ ladder asked:

> **What information in the past improves prediction of the future?**

That gave us: space $\to$ current action $\to$ recent repertoire $\to$ richer context $\to$ Transformers $\to$ ensembles.

The **predictive frontier** question is one level deeper:

> **Given the behavioral information in $X$, how much of the future is actually mappable, how much of that mapping do our models currently recover, and why does the remaining error remain?**

We are **not** trying to determine some universal "mouse behavior is $43\%$ predictable" number.

The useful conceptual decomposition is:

$$
\text{Current Prediction Error}
$$

contains some combination of:

$$
\boxed{\text{Model / Representation Limitations}}
$$

$$
+
$$

$$
\boxed{\text{Finite-Data Limitations}}
$$

$$
+
$$

$$
\boxed{\text{Future Variation Unresolved by Measured } X}
$$

$$
+
$$

$$
\boxed{\text{Finite-Window Variability of the 120-s Target}}.
$$

These pieces do not necessarily separate additively in real behavioral data, but they define the conceptual dimensions of our empirical program.

---

## What Have We Learned So Far?

### 1. There Really Is a Mapping ($X \to Y$)
When two behavioral pasts are similar, their futures are substantially more similar than random pairs — roughly **$43\%$ less divergent** in the similar-past analysis ($D_Y^{\text{similar}} \approx 0.240\text{ bits}$ vs $D_Y^{\text{random}} \approx 0.423\text{ bits}$). So measured behavioral history $X$ genuinely constrains future repertoire $Y$.

### 2. Richer Mappings Extract Progressively More Structure
How much of that relationship can different mappings exploit?
- **Non-Parametric $k$-NN ($k=50$):** $\approx +26.6\%$ to $+27.8\%$ over the global null ($0.194 - 0.197\text{ bits}$).
- **Parametric Ridge BoW:** $\approx +27.2\%$ to $+30.1\%$ ($0.187 - 0.193\text{ bits}$).
- **Sequence & Contextual Transformers:** $\approx +31.9\%$ to $+34.8\%$ ($0.176 - 0.182\text{ bits}$).
- **Stacked Ensembles & Soft MoE:** $\approx +34.8\%$ to $+36.4\%$ ($0.171 - 0.175\text{ bits}$).

So a lot of future structure is accessible without exotic models, but richer contextual mappings do extract meaningful additional information.

### 3. Model Complementarity & $X$-Conditioned Gating
The **Target-Peeking Oracle Selector** (which sees the true future and retrospectively chooses the best base model per window) achieves:

$$
R_{\text{oracle}} \approx 0.157\text{ bits} \quad (+41.3\% \text{ to } +41.8\% \text{ gain}).
$$

This exposes that base models possess complementary advantages:
- Transformers win the majority ($\sim 66\%$) of windows.
- Ridge BoW wins a meaningful minority ($\sim 18\%$) with the **highest decisive margin** ($\Delta_i = 0.026\text{ bits}$ vs $0.013\text{ bits}$ for sequence models).
- $k$-NN wins $\sim 16\%$ of windows during dense motif recurrences.

Crucially, the **$X$-Conditioned Soft Mixture-of-Experts (MoE)** beats every individual base model:

> **The behavioral past contains real information about which predictive strategy should be trusted more for a particular behavioral situation.**

The mapping is heterogeneous across behavioral regimes rather than one homogeneous rule.

### 4. We Look More Data-Limited Than Architecture-Limited
When evaluating scaling curves ($20\% \to 100\%$ of sessions), performance keeps steadily climbing with sample volume without showing sharp asymptotic saturation. 

> **We currently look more data-limited than clearly architecture-limited.**

This makes pretraining on the larger institute archive scientifically essential.

### 5. Extrapolated Asymptotic Floors Are Fragile
Attempting to fit parametric curves through finite-neighborhood points ($d_k \to 0$) to declare a definitive metaphysical "Bayes limit" was extremely sensitive to mathematical curve choices (yielding anywhere from $0.08$ to $0.18\text{ bits}$). 

> **The current 30-session dataset does not identify a unique behavior-only prediction floor.** We stop forcing a metaphysical number out of it and build a concrete empirical frontier.

---

## Where We Drifted Off-Course & How We Recover

### The Drift
We originally set out to study:

> **The uncertainty of the future itself given similar pasts.**

Gradually, $k$-NN, ensembles, oracle selectors, and gating became another extension of:

$$
\text{Build Better Predictor} \longrightarrow \text{Measure Performance}.
$$

Useful work — but still mostly along the predictive modeling axis ($P_n$).

### The Recovery: Empirical Natural Analogues (Twins)
Meteorology’s "natural analogue" idea is:
> *Find two states that are extremely similar now and observe how their futures diverge.*

We will analyze **matched history twins followed forward in time**, rather than using their futures merely to train another predictor.

---

## Actionable Next Steps (Roadmap)

```
                            THE TWO COMPLEMENTARY LADDERS
┌──────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
│      THE P-LADDER (Predictive Mapping)           │      THE U-LADDER (Conditional Uncertainty)      │
│  "How well can we predict future repertoire Y?"  │  "How uncertain is Y given similar histories X?" │
├──────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│ Space (P0)                                       │ Natural Analogue Divergence U(ε, Δ)              │
│   ↓                                              │   ↓                                              │
│ Current Action (P1)                              │ History Horizon Sweep (W = 10s → 120s)           │
│   ↓                                              │   ↓                                              │
│ Recent Repertoire BoW (P2)                       │ Future Horizon Sweep (H = 5s → 120s)             │
│   ↓                                              │   ↓                                              │
│ Contextual Transformers (P5/P6)                  │ Finite-Window Resampling Baseline                │
│   ↓                                              │   ↓                                              │
│ Soft MoE Gated Ensemble (P_gate)                 │ Large-Archive Density Scaling R(N)               │
└──────────────────────────────────────────────────┴──────────────────────────────────────────────────┘
                                       │
                                       ▼
                     THE NEURAL STEP: P(Y | X) vs P(Y | X, DA)
           Does striatal dopamine resolve future variation that behavior cannot?
```

### 1. Build an Uncertainty / Analogue-Divergence Analysis (No Predictor)
**Priority #1.** For pairs of behavioral windows with past distance $d_X$, measure how their futures diverge across varying future horizons $\Delta$:

$$
U(\epsilon, \Delta) = \mathbb{E}[d_Y(\Delta) \mid d_X < \epsilon].
$$

Systematically vary three parameters:
1. **Similarity threshold:** How close the pasts are ($\epsilon$).
2. **History conditioning window:** How much past we condition on ($W \in \{10\text{ s}, 30\text{ s}, 68\text{ s}, 120\text{ s}\}$).
3. **Future forecast horizon:** How far forward we look ($\Delta \in \{5\text{ s}, 15\text{ s}, 30\text{ s}, 60\text{ s}, 120\text{ s}\}$).

*Question:* If two mice have indistinguishable histories over $68\text{ s}$, do their next $5\text{ s}$ stay identical? What about $20\text{ s}$? $60\text{ s}$? $120\text{ s}$?

### 2. Test Whether Longer Histories Collapse Uncertainty
Test whether expanding history window $W$ reduces future dispersion among matched pasts:

$$
W \uparrow \quad \Longrightarrow \quad \text{Dispersion } D_Y(\text{matched}) \downarrow ?
$$

- If increasing $W$ continuously reduces future dispersion, apparent "randomness" was partly unconditioned history.
- If dispersion plateaus despite large $N$ and dense matching, we identify the empirical boundary of behavior-only observability.

### 3. Estimate Finite-Window Variability of Target $Y$
Our target $Y$ is a single finite $120\text{ s}$ realization. Even two samples drawn from the identical behavioral state generator will not yield identical 99-syllable empirical histograms.
- Compute a **block/bout-aware bootstrap resampling baseline** (preserving continuous syllable dwell structure).
- Quantify how much Jensen-Shannon divergence exists strictly from finite-sample Poisson/multinomial draw jitter.

### 4. Direct Apples-to-Apples Gating Decomposition
Finalize the clean single-split evaluation:

$$
\text{Best Single (Ordered TF)} \longrightarrow \text{Fixed Stacked Ensemble} \longrightarrow \text{Soft MoE Gate} \longrightarrow \text{Target-Peeking Oracle}.
$$

Verify exact per-session error bars and feature correlations ($w_m(X)$ vs entropy, dwell, and transition dynamics).

### 5. Push Scaling Curves on the Full Institute Archive
Since performance curves are still climbing, pretrain on the full archive ($10^2 - 10^3$ sessions) to test whether empirical frontiers bend toward true saturation.

---

## One-Sentence Reset for Monday

> **We have shown that recent behavior strongly constrains future repertoire and that increasingly flexible models recover progressively more of that structure, but the models remain data-limited and we do not yet know the behavior-only limit; the next step is therefore to complement the prediction ladder with a model-free analysis of how future uncertainty changes as behavioral histories become more similar, more informative, and more densely sampled.**
