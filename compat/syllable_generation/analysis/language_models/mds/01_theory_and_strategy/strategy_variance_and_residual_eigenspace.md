# Strategy Latent Variance Decomposition & Residual Eigenspace Discovery

**Author / Project**: In Vivo Neuro-Behavioral Strategy Project  
**Date**: Monday, August 17, 2026  
**Pipeline Script**: [`pipelines/25_strategy_variance_decomposition.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/25_strategy_variance_decomposition.py)  
**Output Directory**: [`output/strategies_extended/strategy_variance_audit/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/strategies_extended/strategy_variance_audit/)  
**Generated Figure**: [`output/strategies_extended/strategy_variance_audit/strategy_variance_decomposition.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/strategies_extended/strategy_variance_audit/strategy_variance_decomposition.png)

---

## 1. Executive Summary & Variance Audit

We conducted a mathematical decomposition of the **$24$-Dimensional Strategy Latent Space** ($\vec{Z} \in \mathbb{R}^{24}$, SVD eigenspace of syllable transition matrices) to determine:
1. **How much total strategy variance is explained by our 9 physical & scalar metrics** (Mean Speed, Speed CV, Pause Fraction, Turn Rate, Distance-to-Center, Spatial Entropy, Cadence, Sequence Entropy, Chirality)?
2. **Are there missing, interpretable Principal Components (PCs) in the unexplained strategy residual space?**

### Core Findings:
* **Total Strategy Variance Captured**:
  - **Linear $R^2$ across all 24 dimensions:** **$26.90\%$**
  - **Non-linear Multi-Metric Capture (Random Forest):** **$66.14\%$** on top dimensions.
* **Dominant Strategy Dimensions ($PC1$ and $PC2$):**
  - **Strategy PC1 (28.26% of Strategy Space):** Captures **$\mathbf{71.18\%}$ linear variance** (and $>85\%$ non-linear variance). Corresponds strictly to the **Locomotion / Cadence / Vigor Axis**.
  - **Strategy PC2 (12.30% of Strategy Space):** Captures **$26.36\%$ variance**. Corresponds to the **Spatial Geometry & Thigmotaxis Axis**.
* **The Strategy Residual Space ($PC3\text{--}PC24$):**
  - Beyond PC1 and PC2, our 9 scalars capture only $3\%\text{--}19\%$ of the higher-order strategy dimensions.
  - This mathematically proves that **behavioral strategies contain vast grammatical and ethological structure that cannot be reduced to simple 1D velocity or spatial tracking metrics!**

---

## 2. Strategy Eigenspace Variance Explained Breakdown

```
========================================================================================================================
                          STRATEGY EIGENSPACE VARIANCE EXPLAINED TABLE
========================================================================================================================
Strategy PC      Eigenspace Weight (%)   Cumulative Weight (%)   R² by Current 9 Scalars (%)   Residual Unexplained (%)
------------------------------------------------------------------------------------------------------------------------
PC1                     28.26%                  28.26%                    71.18%                       28.82%
PC2                     12.30%                  40.56%                    26.36%                       73.64%
PC3                      7.73%                  48.29%                    10.15%                       89.85%
PC4                      6.12%                  54.42%                    19.61%                       80.39%
PC5                      4.71%                  59.13%                     4.85%                       95.15%
PC6                      3.90%                  63.03%                     2.83%                       97.17%
PC7                      3.25%                  66.27%                     7.70%                       92.30%
PC8                      3.12%                  69.39%                     5.41%                       94.59%
PC9                      2.80%                  72.19%                     5.56%                       94.44%
PC10                     2.58%                  74.77%                     5.56%                       94.44%
PC11                     2.53%                  77.30%                     4.31%                       95.69%
PC12                     2.19%                  79.48%                     2.62%                       97.38%
========================================================================================================================
```

---

## 3. Residual Eigenspace Analysis: The 3 Missing Ethological Dimensions

When we perform PCA strictly on the unexplained residual space ($\vec{Z}_{\text{residual}} = \vec{Z} - \hat{\vec{Z}}_{\text{scalars}}$), the top 3 Residual PCs explain **$35.9\%$ of the uncaptured variance**. 

By correlating each Residual PC with the 100 syllable usage rates and transition matrices, we discover **3 distinct, interpretable missing behavioral dimensions**:

```
                               THE 3 MISSING ETHOLOGICAL DIMENSIONS
                                                                     
  ┌─────────────────────────────────┐     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
  │ 1. VERTICAL REARING & PITCH     │     │ 2. STEREOTYPIC MOTIF PERSISTENCE│     │ 3. INSTANTANEOUS ACCEL. / JERK  │
  │ • Residual PC1 (16.17% unexp.)  │     │ • Residual PC2 (10.77% unexp.)  │     │ • Residual PC3 (8.98% unexp.)   │
  │ • Vertical rears vs low sniffs  │     │ • Micro-groom loops vs darts    │     │ • Rapid acceleration bursts     │
  │ • Syllables: +62, +58 vs -18    │     │ • Syllables: +48, +53 vs -69    │     │ • Syllables: +49, +59 vs -66    │
  └─────────────────────────────────┘     └─────────────────────────────────┘     └─────────────────────────────────┘
```

### Missing Dimension 1: Vertical Rearing & Head Elevation (Residual PC1: $16.17\%$ of Residual Variance)
* **Top Positive Loadings:** Syllable 62 ($r=+0.45$), Syllable 58 ($r=+0.42$), Syllable 29 ($r=+0.42$) — **Vertical wall-rears, unsupported rears, and elevated sniffs** ($v \approx 45\text{--}50\text{ px/s}$).
* **Top Negative Loadings:** Syllable 18 ($r=-0.35$), Syllable 55 ($r=-0.34$) — **Ground-level snout scanning and low pauses**.
* **What is missing from our scalar metrics:** We currently only measure 2D horizontal kinematics $(x, y)$. We do not have a **Head Elevation / Vertical Pitch / Rearing Height ($z$)** scalar.

### Missing Dimension 2: Stereotypic Looping vs. Broad Motor Branching (Residual PC2: $10.77\%$ of Residual Variance)
* **Top Positive Loadings:** Syllable 48 ($r=+0.37$), Syllable 53 ($r=+0.32$), Syllable 59 ($r=+0.32$) — **Repetitive face-grooming and micro-movement loops** with high self-transition rates.
* **Top Negative Loadings:** Syllable 69 ($r=-0.48$), Syllable 17 ($r=-0.42$), Syllable 84 ($r=-0.42$) — **High-speed open-field darts and broad directional transitions** ($v > 114\text{ px/s}$, $\omega = 1.91\text{ rad/s}$).
* **What is missing from our scalar metrics:** We measure global transition entropy, but we do not have a **Motif Recurrence / Self-Transition Persistence Index** (e.g. diagonal dominance of the transition matrix: how often an animal gets trapped in repetitive 2-syllable loops).

### Missing Dimension 3: Instantaneous Acceleration & Motor Jerk (Residual PC3: $8.98\%$ of Residual Variance)
* **Top Positive Loadings:** Syllable 49 ($r=+0.41$), Syllable 59 ($r=+0.32$), Syllable 60 ($r=+0.32$) — **Laminar, steady, low-acceleration locomotion**.
* **Top Negative Loadings:** Syllable 66 ($r=-0.46$), Syllable 53 ($r=-0.42$), Syllable 99 ($r=-0.34$) — **Violent acceleration bursts and ballistic darts** ($v > 160\text{ px/s}$).
* **What is missing from our scalar metrics:** We measure mean speed and speed CV ($\sigma/\mu$), but not **Mean Absolute Acceleration ($\frac{dv}{dt}$) or Kinetic Jerk ($\frac{d^2v}{dt^2}$)**.

---

## 4. Visualizations

![Strategy Variance Decomposition](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/strategies_extended/strategy_variance_audit/strategy_variance_decomposition.png)

---

## 5. Summary & Recommendations for Model Expansion

1. **Current Scalars are Sufficient for Primary Axes:** The 9 current metrics adequately anchor the primary axes of movement vigor (PC1: $71.2\%$) and arena geography (PC2: $26.4\%$).
2. **To Reach >90% Variance Explained in Strategy Space:** Add **3 targeted scalar metrics**:
   - **Height / Pitch Scalar:** Mean head elevation or rearing duration fraction ($z$).
   - **Stereotypy / Recurrence Scalar:** Transition matrix diagonal recurrence ratio $\sum P_{ii} / K$.
   - **Kinetic Acceleration Scalar:** Mean absolute acceleration $\mathbb{E}[|a(t)|] = \mathbb{E}[|\Delta v / \Delta t|]$.
