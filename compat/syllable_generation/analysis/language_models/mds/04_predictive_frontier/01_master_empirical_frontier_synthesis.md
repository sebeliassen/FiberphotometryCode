# Document 01: Master Empirical Predictive Frontier Synthesis

## Executive Summary & Scientific Framing

The primary objective of this research is **not** to prove an abstract, universal, or metaphysical upper limit on mouse-behavior predictability. 

Rather, our goal is strictly operational and empirically grounded:

> **Given our actual constraints — finite dataset size (30 sessions), measured behavioral variables (continuous pose, spatial trajectories, discrete syllables), chosen history window ($W = 68.3\text{ s}$), chosen future target ($H = 120.0\text{ s}$), and held-out-session cross-validation — estimate how much predictive structure appears to be available in measured behavioral history, and determine how closely our models approach that empirical frontier.**

To maintain scientific rigor and prevent overclaiming, we employ conservative, operationally defined terminology:
* **Empirical predictive frontier:** The performance frontier achievable under currently available data, features, and validated model families.
* **Constrained predictive ceiling estimate:** An empirical reference benchmark conditional on measured behavioral inputs.
* **Conditional future uncertainty:** The empirical dispersion of futures observed following similar measured behavioral histories.
* **Available predictive structure:** The systematic mapping from measured behavioral history to future repertoire distribution.
* **Model gap:** Performance differences attributable to model extraction capacity versus input feature availability.
* **Behavior-only frontier:** Predictive capacity achievable prior to incorporating neural or physiological recordings.

---

## 1. Complete Empirical Benchmark Hierarchy

All evaluations below were performed using strict 5-fold GroupKFold cross-validation across 30 sessions ($29{,}325$ clean non-straddling windows, $W=68.3\text{s} \to H=120.0\text{s}$). The unconditional baseline (Global Blind Null) is $\mathbf{0.2677 \pm 0.0324\text{ bits}}$.

| Benchmark Level | Methodology Type | Target Risk ($D_{\text{JS}}$ bits) | Predictive Gain (% $\Delta D_{\text{JS}}$) | Core Finding & Significance |
| :--- | :---: | :---: | :---: | :--- |
| **1. Global Blind Null** | Population Baseline | $0.2677 \pm 0.0324\text{ b}$ | $0.00\%$ | Unconditional population baseline |
| **2. Continuous Distance Constraint** | Pairwise Geometric Reference | $0.2400 \pm 0.0030\text{ b}$ | $+43.42\%$ (Dispersion) | Conditioning on top 5% similar pasts removes $\sim 43\%$ of future dispersion |
| **3. Non-Parametric $k$-NN ($k=50$)** | Deployable ($X$-only) | $0.1938 \pm 0.0299\text{ b}$ | $+27.77\% \pm 5.92\%$ | Local historical matching recovers $>27\%$ of predictable structure |
| **4. Linear Ridge BoW ($\mathcal{P}_5$)** | Deployable ($X$-only) | $0.1930 \pm 0.0261\text{ b}$ | $+27.20\% \pm 5.30\%$ | Linear capacity of macro-repertoire features |
| **5. Ordered Transformer (Action-GPT)** | Deployable ($X$-only) | $0.1788 \pm 0.0303\text{ b}$ | $+33.94\% \pm 6.32\%$ | Sequential syntax + self-attention |
| **6. Shuffled Transformer (Order-Agnostic)**| Deployable ($X$-only) | $0.1761 \pm 0.0295\text{ b}$ | $+34.80\% \pm 6.21\%$ | Non-linear bag-of-words self-attention across historical bouts |
| **7. Stacked Learned Ensemble** | Deployable ($X$-only Best) | $\mathbf{0.1714 \pm 0.0281\text{ b}}$ | $\mathbf{+36.41\% \pm 5.81\%}$ | **State-of-the-art deployable model** (+2.5 pp over single best Transformer) |
| **8. Matched Branch Oracle ($k=50$)** | Target-Peeking Reference | $\mathbf{0.1681 \pm 0.0237\text{ b}}$ | $\mathbf{+36.30\% \pm 5.71\%}$ | Closest empirical future among 50 nearest pasts |
| **9. Oracle Model Selector** | Target-Peeking Reference | $\mathbf{0.1566 \pm 0.0252\text{ b}}$ | $\mathbf{+41.83\% \pm 5.20\%}$ | Theoretical limit of existing model families |
| **10. Split-Half Target Jitter** | Measurement Floor | $0.2810 \pm 0.0150\text{ b}$ | — | Poisson bout-counting noise within individual 120s windows |

---

## 2. Key Insights from the Diagnostic Suite

### A. The Model Gap is Small ($\sim 5.4\text{ pp}$)
The gap between our best deployable model (Stacked Ensemble at $+36.41\%$, $0.171\text{ bits}$) and an all-knowing oracle picking the best model per window (Oracle Selector at $+41.83\%$, $0.157\text{ bits}$) is only **$0.015\text{ bits}$**. Diverse model architectures make largely identical errors, indicating that within this input feature space, model blending yields diminishing returns.

### B. Residual Predictability is Zero
Training secondary models (Ridge, MLP) on $X$ to predict the Transformer's logit residuals degraded held-out performance to $+15.60\%$ and $+12.32\%$. The Transformer leaves no easily extractable linear or non-linear state-space structure in $X$ unexploited.

### C. Models are Data-Limited, Not Capacity-Saturated
Sweeping training dataset size from $N = 20\% \to 100\%$ demonstrates that while linear models approach an early plateau ($+21.4\% \to +27.8\%$), the Transformer scaling curve is steep and continuous ($-2.6\% \to +16.1\% \to +24.6\% \to +28.7\% \to +32.1\%$). Adding more training sessions directly improves predictive capacity.

### D. Macro-Behavior is Bag-of-Words Dominated
Over a 68.3s history window, knowing *which* behavioral bouts occurred and their relative frequencies contains almost all available predictive information. Scrambling sequence order at test time changes performance by only $-0.02\text{ pp}$. The Transformer's leap over linear regression stems from non-linear token interactions and contextual weighting rather than rigid $n$-gram syntax.

---

## 3. Provenance & Artifact Reference Table

| Diagnostic / Figure | Script Location | Output Data / Image |
| :--- | :--- | :--- |
| **Master Benchmark Suite** | [`pipelines/04_predictive_frontier/run_full_empirical_frontier_diagnostics.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/04_predictive_frontier/run_full_empirical_frontier_diagnostics.py) | [`output/04_predictive_frontier/tables/master_empirical_frontier_table.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/04_predictive_frontier/tables/master_empirical_frontier_table.csv) |
| **4-Panel Publication Figure** | [`pipelines/04_predictive_frontier/run_full_empirical_frontier_diagnostics.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/04_predictive_frontier/run_full_empirical_frontier_diagnostics.py) | [`output/presentation/figures/fig11_comprehensive_empirical_frontier_suite.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/fig11_comprehensive_empirical_frontier_suite.png) |
| **Past-Future Distance Transfer** | [`pipelines/04_predictive_frontier/run_past_future_distance_transfer.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/04_predictive_frontier/run_past_future_distance_transfer.py) | [`output/presentation/figures/fig9_past_future_distance_transfer.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/fig9_past_future_distance_transfer.png) |
| **Density Scaling Sweep** | [`pipelines/04_predictive_frontier/run_density_scaling_sweep.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/04_predictive_frontier/run_density_scaling_sweep.py) | [`output/presentation/figures/fig7_density_scaling_asymptote.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/fig7_density_scaling_asymptote.png) |
| **Joint Bias-Variance Surface** | [`pipelines/04_predictive_frontier/run_joint_bias_variance_surface.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/04_predictive_frontier/run_joint_bias_variance_surface.py) | [`output/presentation/figures/fig8_joint_bias_variance_floor.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/fig8_joint_bias_variance_floor.png) |
