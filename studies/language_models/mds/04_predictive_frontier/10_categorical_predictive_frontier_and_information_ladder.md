# Categorical Predictive Frontier & Progressive Information Ladder ($X_1 \to X_2 \to X_3$)

## 1. Executive Summary

This report documents the discretization of future behavioral horizons ($H = 120\text{ s}$) into higher-order macro-strategies ($K \in \{4, 8, 16\}$), the evaluation of empirical Bayes ceilings, and the resolution of the **State Persistence Floor vs. Local Analogue Ceiling** paradox via a three-tier Progressive Information Ladder.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PROGRESSIVE INFORMATION CONDITIONING LADDER                             │
├────────────────────────────────┬──────────────────────────────────────┬────────────────────────────────┤
│ X1: Repertoire Alone           │ X2: Repertoire + Current Macro-State │ X3: Full Behavioral History    │
│ • Unconstrained across states  │ • Stratified: S(X_j) == S(X_i)       │ • Repertoire + State + Spatial │
│ • Apparent Oracle: 47.5%       │ • Branch Oracle: 61.7%               │   + Instantaneous Pose         │
│ • Effective Futures: 2.77      │ • Pop. Oracle: 84.9% (Surpasses 83.2)│ • Branch Oracle: 59.2%         │
│ • Flawed (State Unconditioned) │ • Effective Futures: 2.03 (Branch)   │ • Pop. Oracle: 83.7%           │
└────────────────────────────────┴──────────────────────────────────────┴────────────────────────────────┘
```

---

## 2. Intrinsic Macro-Strategy Validation ($K \in \{4, 8, 16\}$)

Across disjoint cross-session splits and unsupervised clustering:
* **$K = 4$ emerged as the clear headline macro-scale:** Cross-Session Adjusted Rand Index (ARI) = **$0.8549$**, Silhouette = **$0.1517$**.
* **$K = 8$ and $K = 16$ act as resolution controls:** ARI = $0.5686$ ($K=8$) and $0.3889$ ($K=16$).

### Class Distribution (Neutral Labeling for $K = 4$):
* **Cluster 0:** $11,271$ windows ($38.4\%$)
* **Cluster 1:** $755$ windows ($2.6\%$)
* **Cluster 2:** $14,186$ windows ($48.4\%$)
* **Cluster 3:** $3,113$ windows ($10.6\%$)

---

## 3. Resolving the Logical Ceiling Conflict (Persistence Floor)

### The Diagnostic Paradox:
* **Autoregressive State Persistence Floor:** $P(Y_{\text{future}} = S(X)) = \mathbf{83.20\%}$.
* A trivial 1-line rule ("predict current state continues") achieves $83.2\%$. Therefore, $A^*_{\text{Bayes}}(\text{full } X) \ge 83.2\%$.
* **Why did unconstrained repertoire matching ($X_1$) show $\sim 47.5\%$?**
  * $d_X(\text{repertoire})$ compared only continuous 68.3s syllable compositions.
  * Two windows $X_A$ and $X_B$ could have identical syllable mixtures while mouse $A$ was in Cluster 0 (persisting in 0) and mouse $B$ was in Cluster 2 (persisting in 2).
  * Collapsing them into the same analogue pool manufactured artificial "future entropy" from **unconditioned current-state identity**.

### The Stratified Resolution ($X_2$ and $X_3$):
* When analogues are conditioned to share the same current macro-state ($S(X_j) = S(X_i)$):
  * Across the general population ($\epsilon \ge 0.40$), oracle accuracy reaches **$83.64\% - 84.86\%$**, fully restoring and surpassing the $83.2\%$ persistence floor.
  * In the dense $8.8\%$ branch points ($\epsilon \le 0.25$, where windows have $>1,250$ cross-session twins), oracle accuracy is **$61.7\%$** and effective futures $N_{\text{eff}} \approx \mathbf{2.03}$.
  * These dense regimes represent **active behavioral choice points** where animals in the same state naturally transition into alternative strategies.

---

## 4. Master Empirical Conditioning Table ($K = 4$)

| Conditioning Level | Radius $\epsilon$ | Dataset Coverage | Density ($n_i$ twins) | Mean $d_X$ | Empirical Oracle $A^*(Y_{\text{state}})$ | Conditional Entropy $H(Y \mid X)$ | Effective Futures ($N_{\text{eff}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unconditional Global Null** | — | $100\%$ | — | — | **$48.38\%$** | $1.516\text{ bits}$ | **$2.86$** |
| **State Persistence Floor** | — | $100\%$ | — | — | **$83.20\%$** | — | — |
| \midrule | | | | | | | |
| **$X_1$: Repertoire Alone** | $0.10$ | $8.9\%$ | $2,399.0$ | $0.0003$ | **$47.53\%$** | $1.461\text{ bits}$ | **$2.77$** |
| *(Unconstrained)* | $0.25$ | $8.9\%$ | $2,388.0$ | $0.0014$ | **$47.77\%$** | $1.455\text{ bits}$ | **$2.76$** |
| | $0.50$ | $63.6\%$ | $589.9$ | $0.4070$ | **$78.58\%$** | $0.654\text{ bits}$ | **$1.66$** |
| \midrule | | | | | | | |
| **$X_2$: Repertoire + State $S(X)$** | $0.10$ | $8.8\%$ | $1,257.2$ | $0.0003$ | **$61.66\%$** | $0.993\text{ bits}$ | **$2.03$** |
| *(Stratified Current State)* | $0.25$ | $8.9\%$ | $1,251.5$ | $0.0015$ | **$61.83\%$** | $0.989\text{ bits}$ | **$2.03$** |
| | **$0.40$** | **$30.2\%$** | **$390.9$** | **$0.2703$** | **$83.64\%$** | **$0.465\text{ bits}$** | **$1.46$** |
| | **$0.50$** | **$62.1\%$** | **$405.6$** | **$0.4055$** | **$84.86\%$** | **$0.506\text{ bits}$** | **$1.47$** |
| \midrule | | | | | | | |
| **$X_3$: Full History + State $S(X)$** | $0.10$ | $7.8\%$ | $1,304.1$ | $0.0241$ | **$59.18\%$** | $1.037\text{ bits}$ | **$2.07$** |
| *(Repertoire + Pose + Space + State)* | $0.25$ | $7.8\%$ | $1,304.9$ | $0.0241$ | **$59.19\%$** | $1.037\text{ bits}$ | **$2.07$** |
| | **$0.40$** | **$22.2\%$** | **$461.4$** | **$0.2515$** | **$79.71\%$** | **$0.544\text{ bits}$** | **$1.54$** |
| | **$0.50$** | **$52.3\%$** | **$215.8$** | **$0.3976$** | **$83.71\%$** | **$0.510\text{ bits}$** | **$1.48$** |

---

## 5. Artifacts & Code Reference

* Pipeline Script: [run_stratified_categorical_frontier.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/04_predictive_frontier/run_stratified_categorical_frontier.py)
* Data Table: [stratified_categorical_frontier_ladder.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/04_predictive_frontier/tables/stratified_categorical_frontier_ladder.csv)
* Multi-K Sweep Table: [categorical_predictive_frontier_multi_k_results.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/04_predictive_frontier/tables/categorical_predictive_frontier_multi_k_results.csv)
* Master Figure: `syllable_generation/analysis/language_models/output/presentation/figures/fig16_stratified_categorical_predictive_frontier.png`
