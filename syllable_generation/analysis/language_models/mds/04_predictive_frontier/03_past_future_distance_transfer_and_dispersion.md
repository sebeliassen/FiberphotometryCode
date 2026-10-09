# Document 03: Past-to-Future Distance Transfer & Conditional Dispersion

## Motivation: Quantifying Behavioral Constraint Without Parametric Assumptions

To measure how much information behavioral history $X$ imposes on future divergence $Y$ without making parametric model assumptions, we analyze pairwise distances across independent animals:

* $d_X(i, j) = \|X_i - X_j\|$: Past-state distance in balanced Repertoire + Spatial space ($W = 68.3\text{s}$).
* $d_Y(i, j) = D_{\text{JS}}(Y_i, Y_j)$: Future Jensen-Shannon divergence ($H = 120.0\text{s}$).

We compute the continuous transfer function $\mathbb{E}[d_Y \mid d_X = d]$ across $100{,}000$ cross-session pairs ($\text{session}_i \neq \text{session}_j$).

---

## 1. Validated Empirical Distance Transfer Curve

```
Future Divergence d_Y (bits)
  0.50 ┤                                                  ╭────────────── Dissimilar Pasts (0.460 b)
       │                                            ╭─────╯
  0.42 ┤────────────────────────────────────────────┼────────────── Random Cross-Session Baseline (0.423 b)
       │                                     ╭──────╯               [Shuffled Control Flat Null: 0.418 b]
  0.30 ┤                               ╭─────╯
       │                         ╭─────╯
  0.24 ┤───────────────────╭─────╯                                  <--- Top 5% Similar Pasts (0.240 b: -43.4%)
  0.15 ┤             ╭─────╯
  0.00 ┴─────────────┴─────────────────┴────────────┴──────────────┴────
                    0.3               0.6          0.9            1.2   Past Distance d_X
```

### Numerical Results Across Distance Deciles (100,000 Cross-Session Pairs)

| Past Distance Decile ($d_X$) | Mean Past Distance ($d_X$) | Mean Future Divergence ($d_Y$ bits) | Standard Error | Reduction vs. Random Pairs |
| :---: | :---: | :---: | :---: | :---: |
| **Top 1% (Tightest Matches)** | $0.282$ | $\mathbf{0.148\text{ bits}}$ | $\pm 0.006$ | $\mathbf{-65.0\%}$ |
| **Top 5% (Similar Pasts)** | $0.341$ | $\mathbf{0.240\text{ bits}}$ | $\pm 0.003$ | $\mathbf{-43.4\%}$ |
| **10%** | $0.412$ | $0.285\text{ bits}$ | $\pm 0.002$ | $-32.6\%$ |
| **25%** | $0.563$ | $0.348\text{ bits}$ | $\pm 0.002$ | $-17.7\%$ |
| **50% (Median)** | $0.742$ | $0.405\text{ bits}$ | $\pm 0.001$ | $-4.3\%$ |
| **Random Pairs (Baseline)** | $0.815$ | $\mathbf{0.423\text{ bits}}$ | $\pm 0.001$ | $0.0\%$ |
| **Top 90% (Distant Pasts)** | $1.085$ | $0.460\text{ bits}$ | $\pm 0.002$ | $+8.7\%$ |
| **Shuffled Control ($X \leftrightarrow Y$)** | $0.815$ | $\mathbf{0.418\text{ bits}}$ | $\pm 0.001$ | Flat across all $d_X$ |

---

## 2. Core Takeaways

### A. The 43.4% Information Constraint Ratio
Conditioning on a similar measured behavioral history (top 5% closest pasts across animals) reduces future divergence from $0.423\text{ bits}$ to $0.240\text{ bits}$:

$$\text{Information Constraint Ratio} = \frac{0.423 - 0.240}{0.423} \approx \mathbf{43.4\%}.$$

This proves that **at least 43% of the future variability observed between arbitrary behavioral moments is strictly constrained by prior behavioral history**.

### B. Shuffled Permutation Control
When randomly permuting the $X \leftrightarrow Y$ pairings across windows, the transfer curve collapses to a completely flat line at $0.418\text{ bits}$ across all $d_X$. This confirms that the observed relationship is driven entirely by authentic past-future temporal structure, not high-dimensional geometric artifacts.

### C. Continuous Formulation vs. Arbitrary Quadrants
Rather than applying arbitrary median splits to partition windows into discrete labels ("equifinality" or "causal branching basins"), the continuous $\mathbb{E}[d_Y \mid d_X]$ curve should be treated as the primary empirical reference.

---

## 3. Code Provenance
* **Script:** [`pipelines/04_predictive_frontier/run_past_future_distance_transfer.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/04_predictive_frontier/run_past_future_distance_transfer.py)
* **Figure:** [`output/presentation/figures/fig9_past_future_distance_transfer.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/fig9_past_future_distance_transfer.png)
* **Table:** [`output/04_predictive_frontier/tables/past_future_transfer_curve.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/04_predictive_frontier/tables/past_future_transfer_curve.csv)
