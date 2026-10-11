# Document 05: Oracle Taxonomy, Model Ensembling & Residual Diagnostics

## 1. Formal Oracle Taxonomy

To ensure clear methodological communication, we formally separate two distinct oracle concepts:

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

* **Oracle A ($X$-Only Stacked Ensemble):** Strictly constrained to use only measured history $X$. Achieves $\mathbf{+36.41\% \pm 5.81\%}$ ($0.171\text{ bits}$).
* **Oracle B (Target-Peeking Branch Selector):** Allowed to inspect the realized outcome $Y_i$ to choose the best candidate branch ($+36.30\%$ at $k=50$, $+40.36\%$ at $k=100$) or best model ($+41.83\%$).

---

## 2. The Stacked Ensemble vs. Oracle Selector Benchmark

Evaluating the combination of diverse model families ($k$-NN, Ridge BoW, and Transformer) on held-out sessions:

| Model Architecture | Out-of-Fold Risk ($D_{\text{JS}}$ bits) | Predictive Gain (% $\Delta D_{\text{JS}}$) | Advantage over Baseline |
| :--- | :---: | :---: | :---: |
| **Global Blind Null** | $0.2677 \pm 0.0324\text{ b}$ | $0.00\%$ | — |
| **$k$-NN ($k=50$)** | $0.1938 \pm 0.0299\text{ b}$ | $+27.77\% \pm 5.92\%$ | $+27.8\text{ pp}$ |
| **Ridge BoW ($\mathcal{P}_5$)** | $0.1930 \pm 0.0261\text{ b}$ | $+27.20\% \pm 5.30\%$ | $+27.2\text{ pp}$ |
| **Ordered Transformer** | $0.1788 \pm 0.0303\text{ b}$ | $+33.94\% \pm 6.32\%$ | $+33.9\text{ pp}$ |
| **Shuffled Transformer** | $0.1761 \pm 0.0295\text{ b}$ | $+34.80\% \pm 6.21\%$ | $+34.8\text{ pp}$ |
| **Stacked Learned Ensemble** | $\mathbf{0.1714 \pm 0.0281\text{ b}}$ | $\mathbf{+36.41\% \pm 5.81\%}$ | $\mathbf{+36.4\text{ pp}}$ |
| **Oracle Model Selector** | $\mathbf{0.1566 \pm 0.0252\text{ b}}$ | $\mathbf{+41.83\% \pm 5.20\%}$ | $\mathbf{+41.8\text{ pp}}$ |

### Takeaway:
* The Stacked Ensemble improves by $+2.5\text{ pp}$ over the single best Transformer model.
* The gap to the Oracle Selector is only **$+5.4\text{ pp}$ ($0.015\text{ bits}$)**. Base models make largely overlapping errors; complementary predictive capacity within this model family is largely exhausted.

---

## 3. Residual Predictability Audit

We tested whether secondary models trained on $X$ could predict the out-of-fold logit residuals of the Transformer:
* **Linear Residual Correction:** $+15.60\% \pm 9.63\%$ ($0.225\text{ bits}$)
* **MLP Residual Correction:** $+12.32\% \pm 10.48\%$ ($0.234\text{ bits}$)

Both models degraded held-out performance relative to the raw Transformer ($+33.94\%$), confirming that **no easily extractable linear or non-linear state-space structure in $X$ was missed**.

---

## 4. Branch Oracle vs. Random Candidate Controls

| Candidate Pool Size ($k$) | Matched-Neighbor Branch Oracle | Random-Candidate Control | Matched Advantage ($\Delta$) |
| :---: | :---: | :---: | :---: |
| **$k = 10$** | $+22.27\% \pm 7.00\%$ | $+11.85\% \pm 4.98\%$ | $\mathbf{+10.42\text{ pp}}$ |
| **$k = 25$** | $+31.34\% \pm 6.16\%$ | $+22.73\% \pm 4.88\%$ | $\mathbf{+8.61\text{ pp}}$ |
| **$k = 50$** | $+36.30\% \pm 5.71\%$ | $+30.96\% \pm 4.55\%$ | $\mathbf{+5.34\text{ pp}}$ |
| **$k = 100$** | $+40.36\% \pm 5.32\%$ | $+37.27\% \pm 4.42\%$ | $\mathbf{+3.09\text{ pp}}$ |

Matched pasts provide a substantial predictive advantage over random candidates at realistic pool sizes ($k=10-50$). At $k=100$, random combinatorial coverage begins to catch up, emphasizing the necessity of random controls.

---

## 5. Code Provenance
* **Script:** [`pipelines/04_predictive_frontier/run_full_empirical_frontier_diagnostics.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/04_predictive_frontier/run_full_empirical_frontier_diagnostics.py)
* **Tables:** 
  - [`output/04_predictive_frontier/tables/master_empirical_frontier_table.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/04_predictive_frontier/tables/master_empirical_frontier_table.csv)
  - [`output/04_predictive_frontier/tables/branch_oracle_pool_comparison.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/04_predictive_frontier/tables/branch_oracle_pool_comparison.csv)
