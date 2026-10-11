# Document 07: Neural Photometry Testing Protocol ($P(Y \mid X)$ vs. $P(Y \mid X, \text{DA})$)

## 1. Grounding Neural Hypotheses in the Empirical Behavioral Frontier

With the behavior-only predictive frontier rigorously mapped ($+36.41\% \pm 5.81\%$ via Stacked Ensemble), we can perform a definitive statistical test on the informational content of recorded striatal dopamine (DA) photometry signals:

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

---

## 2. Formal Statistical Testing Regimes

### Regime 1: $P(Y \mid X, \text{DA}) = P(Y \mid X)$ (Redundant State Information)
* **Interpretation:** The future-predictive information encoded in striatal dopamine fluctuations is already fully expressed in the animal's preceding 68s behavioral history.
* **Caution:** This does *not* imply dopamine has no causal role in generating behavior. Rather, it demonstrates that behavior and dopamine share redundant state information at the 1–2 minute timescale.

### Regime 2: $P(Y \mid X, \text{DA}) > P(Y \mid X)$ (Independent Hidden State Information)
* **Interpretation:** Striatal dopamine provides future-relevant latent state information (e.g., internal motivation, reward expectation, latent arousal) that cannot be reconstructed from external behavioral observations alone.
* **Significance:** This provides rigorous, out-of-fold proof of an internal neural driver of future macro-behavioral switching.

---

## 3. Evaluation Standards
1. **Held-Out Session Evaluation:** Evaluate exclusively on held-out sessions using GroupKFold to avoid session-specific overfitting.
2. **Nested Architecture:** The base behavioral model must match the Stacked Ensemble ($+36.4\%$) to prevent attributing model underfitting to neural signals.
3. **Session-Level Uncertainty:** Report confidence intervals across session folds rather than window-level pseudo-replication.
