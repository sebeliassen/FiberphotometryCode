# Document 04: Density Scaling & Asymptotic Bias-Variance Modeling

## 1. Fixed-$k$ Density Scaling Sweep ($N = 10\% \to 100\%$)

In non-parametric prediction via local matching:

$$\hat{Y}_i^{k\text{-NN}} = \frac{1}{k} \sum_{j \in \mathcal{N}_k(X_i)} Y_j,$$

fixing $k$ while increasing candidate data density $N$ produces two simultaneous effects:
1. The matching past distance shrinks: $N \uparrow \implies \bar{d}_k \downarrow$.
2. The out-of-fold predictive error drops: $N \uparrow \implies R_k^{\text{pred}} \downarrow$.

```
Dataset Fraction (N)    Mean Past Distance (d_100)    Held-Out Predictive Risk (R_100)    Predictive Gain (%)
       10%                        0.666 bits                     0.256 bits                     + 4.1%
       25%                        0.586 bits                     0.217 bits                     +18.9%
       50%                        0.557 bits                     0.207 bits                     +22.8%
       75%                        0.539 bits                     0.199 bits                     +26.1%
      100%                        0.521 bits                     0.189 bits                     +29.5%
```

Increasing data density consistently improves local non-parametric prediction.

---

## 2. Joint Bias-Variance Surface Modeling

To understand the interaction between sample variance ($\propto 1/k$) and mismatch bias ($\propto d^p$), we sampled a **203-point empirical grid** across candidate subsets $N \in [20\% \dots 100\%]$ and neighborhood sizes $k \in [20 \dots 200]$.

We fitted the parametric surface:

$$\hat{R}^{\text{pred}}(d, k) = R_\infty + a \cdot d^p + \frac{b}{k}.$$

```
Empirical Risk Cloud (203 points)
Risk R (bits)
  0.26 ┤        *  *  (k=20, N=20%)
  0.22 ┤      *   *   *
  0.19 ┤    *   *   *   *  (k=100, N=100%: 0.189 b)
  0.15 ┤  ----------------------------------- Assumed Asymptote Floor:
       │  Linear Mismatch (p=1):      R_∞ = 0.000 bits (100% theoretical gain)
  0.06 ┤  Quadratic Mismatch (p=2):   R_∞ = 0.058 bits ( 78% theoretical gain)
  0.00 ┴────────┴────────┴────────┴──────────
               0.45     0.55     0.65         Past Distance d_k
```

### Asymptotic Extrapolation Sensitivity Across 2,000 Bootstrap Folds

| Assumed Mismatch Exponent | Estimated Floor ($R_\infty$) | $95\%$ Bootstrap Confidence Interval | Implied Theoretical Gain |
| :---: | :---: | :---: | :---: |
| **Linear Mismatch ($p=1$)** | $\mathbf{0.000\text{ bits}}$ | $[0.000, 0.000]\text{ bits}$ | $100.0\%$ |
| **Super-Linear ($p=1.5$)** | $\mathbf{0.021\text{ bits}}$ | $[0.012, 0.031]\text{ bits}$ | $92.2\%$ |
| **Quadratic Mismatch ($p=2$)** | $\mathbf{0.058\text{ bits}}$ | $[0.050, 0.064]\text{ bits}$ | $78.4\%$ |

---

## 3. Methodological Warning: Extrapolation Limits of Finite Data

Because our observed past distances are bounded within $\bar{d} \in [0.44, 0.67]$, **extrapolating out-of-range all the way to $d=0$ is heavily sensitive to the chosen functional form**.

* In our 30-session dataset, finite candidate density means the nearest matching past is still at distance $d \approx 0.52$.
* We cannot mathematically prove whether the true infinite-data zero-mismatch floor is $0.06\text{ bits}$, $0.10\text{ bits}$, or $0.14\text{ bits}$ without scaling to massive open-source repositories where $\bar{d}_k \to 0$ empirically.

> **Rule:** Do not report extrapolated $R_\infty$ values as proven physical limits. Report the empirical frontier achieved under current data density.

---

## 4. Code Provenance
* **Scripts:** 
  - [`pipelines/04_predictive_frontier/run_density_scaling_sweep.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/04_predictive_frontier/run_density_scaling_sweep.py)
  - [`pipelines/04_predictive_frontier/run_joint_bias_variance_surface.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/04_predictive_frontier/run_joint_bias_variance_surface.py)
* **Figures:**
  - [`output/presentation/figures/fig7_density_scaling_asymptote.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/fig7_density_scaling_asymptote.png)
  - [`output/presentation/figures/fig8_joint_bias_variance_floor.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/fig8_joint_bias_variance_floor.png)
