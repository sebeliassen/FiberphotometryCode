# Sliding Window Optimization Metrics & Biexponential Memory Scaling

**Date:** Wednesday, August 26, 2026  
**Dataset:** 30 Open-Field Sessions (174,686 Bouts)  
**Cross-Validation:** Session-Grouped 5-Fold (`GroupKFold` on `session_id`, Zero Temporal Leakage)  
**Publication Figures:**  
* `output/01_horizon_dynamics/figures/past_memory_capacity_curve.png` (3-Panel Biexponential Memory Curve, Accumulation vs. Dilution Rates, & Retention)  
* `output/01_horizon_dynamics/figures/past_window_vs_future_horizon_grid.png` (2D $W \times H$ Grid)  
**CSV Data:** `output/01_horizon_dynamics/tables/past_memory_capacity_results.csv`

---

## 1. The Core Scientific Discovery: $W^* \approx 68\text{s}$ (Superseded by Within-Cohort Adversarial Audit)

> [!CAUTION]
> **CRITICAL RETRACTION & AUDIT UPDATE (September 2026):**
> **Do not cite $W^* = 68.3\text{s}$ as an "intrinsic physical constant of behavior."**
> 
> A rigorous adversarial audit revealed that the $68.3\text{s}$ peak was a **mathematical artifact of pooling all 4 cohorts into a single global dataset**:
> 1. **The Group-Identity Confound:** In a pooled model, expanding the history window $W \to 68\text{s}$ provides sufficient syllable co-occurrence evidence for the model to infer *which cohort* the animal belongs to (e.g. recognizing DAT-KI stereotypic darting/grooming). The model then exploits the **$+20.5\%$ static group prior** (see `06_nmf_retention_investigation_and_adversarial_audit.md`), artificially inflating predictive gain up to $\sim 68\text{s}$.
> 2. **Within-Cohort Reality:** When evaluated **strictly within-cohort** under Leave-One-Animal-Out cross-validation against each cohort's *own* marginal prior:
>    * **No cohort peaks at $68\text{s}$**.
>    * Every single cohort reaches its true memory peak at **$W^* = 15\text{s}\text{--}20\text{s}$** (WT Saline peaks at $15\text{--}20\text{s}$, $+9.8\%$ total gain / $+2.3\%$ above $S_t$).
>    * By $W = 68.3\text{s}$, past context in WT is already diluting, and DAT-KI mice experience a **catastrophic collapse ($-13.6\%$ below $S_t$)** due to severe overfitting to idiosyncratic past bouts.
> 
> *Refer to `08_deconstruction_of_the_68s_window_and_true_memory_timescales.md` for the complete mathematical deconstruction.*

---

### Historical Context: The Original Pooled Biexponential Fit

---

## 2. Biexponential Memory Kinetics (Bateman Function)

A simple monotonic exponential saturation ($A - B e^{-W/\tau}$) is physically inadequate because behavioral memory is governed by **two opposing forces**:
1. **Fast Information Accumulation ($\tau_{\text{rise}}$):** Accumulating syllable unigram/bigram statistics and spatial moments increases the signal-to-noise ratio of the active behavioral strategy.
2. **Slow Non-Stationary State Dilution ($\tau_{\text{decay}}$):** Because open-field mice transition between motivational states (e.g. exploration $\to$ grooming $\to$ resting) every $1\text{--}2$ minutes, an overly long window pools across multiple distinct strategies, diluting the active state with obsolete historical noise.

We model this biophysical process with a **Biexponential Rise-and-Decay Function (Bateman Kinetics)**:

$$\mathcal{P}_{\text{max}}(W) = \mathcal{P}_0 + A \cdot \left( e^{-W / \tau_{\text{decay}}} - e^{-W / \tau_{\text{rise}}} \right)$$

![Biexponential Memory Capacity Curve](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/01_horizon_dynamics/figures/past_memory_capacity_curve.png)

### Fitted Model Parameters ($R^2 = 0.9707$):
* **Baseline ($\mathcal{P}_0$):** $+19.00\%$
* **Dynamic Amplitude ($A$):** $+12.38\%$
* **Fast Sequence Accumulation Timescale ($\tau_{\text{rise}}$):** $\mathbf{17.2\text{ seconds}}$
* **Slow Non-Stationary State Drift ($\tau_{\text{decay}}$):** $\mathbf{841.9\text{ seconds}}$ ($\approx 14.0\text{ minutes}$)

### Exact Analytical Peak Memory Window ($W^*$):
Taking the derivative $\frac{d\mathcal{P}_{\text{max}}}{dW} = 0$:

$$W^* = \frac{\tau_{\text{rise}} \cdot \tau_{\text{decay}}}{\tau_{\text{decay}} - \tau_{\text{rise}}} \ln\left( \frac{\tau_{\text{decay}}}{\tau_{\text{rise}}} \right) = \mathbf{68.3\text{ seconds}} \quad (\approx 1.14\text{ minutes})$$

* **Peak Reachable Predictive Capacity:** $\mathbf{+30.18\%}$ gain.

---

## 3. Practical Window Selection: Should We Use $60\text{s}\text{--}68\text{s}$ or $120\text{s}$?

A critical engineering question is whether to strictly adopt $W = 60\text{s}\text{--}68\text{s}$ or whether $W = 120\text{s}$ (our initial benchmark choice) is acceptable.

### Quantitative Comparison:
* **$W = 68.3\text{s}$ (Analytical Optimum):** $\mathcal{P}_{\text{max}} = \mathbf{+30.18\%}$ ($100.0\%$ peak retention).
* **$W = 60.0\text{s}$ (Production Standard):** $\mathcal{P}_{\text{max}} = \mathbf{+29.77\%}$ ($98.6\%$ peak retention, $-0.41\text{ pp}$).
* **$W = 120.0\text{s}$ (Extended Standard):** $\mathcal{P}_{\text{max}} = \mathbf{+29.99\%}$ ($99.4\%$ peak retention, $-0.19\text{ pp}$).

### The Three Trade-Off Dimensions:

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                   TRADE-OFF MATRIX: 60s vs. 120s SLIDING WINDOWS                         │
├───────────────────────┬────────────────────────────┬─────────────────────────────────────┤
│ Evaluation Dimension  │ 60s - 68s Window           │ 120s Window                         │
├───────────────────────┼────────────────────────────┼─────────────────────────────────────┤
│ ⚡ Temporal Agility   │ High (±30s buffer).        │ Moderate (±60s buffer).             │
│                       │ Rapidly detects strategy   │ Smeared across boundaries during    │
│                       │ state switches.            │ sudden behavioral transitions.      │
├───────────────────────┼────────────────────────────┼─────────────────────────────────────┤
│ 🎯 Predictive Capacity│ 98.6% - 100% of peak       │ 99.4% of peak (virtually identical, │
│                       │ reachable ceiling.         │ only -0.19 pp difference).          │
├───────────────────────┼────────────────────────────┼─────────────────────────────────────┤
│ 📊 Clustering Density │ ~150 - 180 bouts.          │ ~300 - 360 bouts.                   │
│                       │ Sparsity in rare syllables.│ Denser unigram bag-of-words.        │
├───────────────────────┼────────────────────────────┼─────────────────────────────────────┤
│ 🧠 Photometry Sync    │ High resolution for        │ Smooth baseline drift;              │
│                       │ phasic dopamine tone.      │ averages out transient bursts.      │
└───────────────────────┴────────────────────────────┴─────────────────────────────────────┘
```

### Recommendation:
1. **For Behavioral Forecasting & Fiber Photometry Modeling ($W = 60\text{s}$):**
   * **Recommended:** $W = 60\text{s}$ is the ideal operational standard. It captures $98.6\%$ of maximal information while providing $2\times$ faster temporal agility to synchronize with sub-minute dopamine tone fluctuations.
2. **For Unsupervised Strategy Phenotyping ($W = 120\text{s}$):**
   * **Completely Fine:** If one uses $W = 120\text{s}$ for macroscopic clustering (e.g. UMAP / HDBSCAN topic modeling), it is well within the plateau ($99.4\%$ retention). It provides slightly denser bag-of-words histograms that help cluster rare syllables without sacrificing predictive power.
3. **Beyond Rigid Rectangular Windows (Future Work):**
   * Rather than a hard rectangular window, an **exponentially decaying memory kernel** ($e^{-t / \tau_{\text{mem}}}$ with $\tau_{\text{mem}} \approx 60\text{s}$) or **recurrent models (GRU / Attention-decay Transformers)** naturally weigh recent history heavily while fading ancient history, eliminating boundary smearing entirely.

---

## 4. Dense Past Memory Capacity Table ($H = 120\text{s}$ Action Budget)

| Past Window ($W$) | Macro Context ($\mathcal{P}_2$) | Strategy $K=16$ ($\mathcal{P}_4$) | $\mathcal{P}_{\text{max}}$ Ceiling | % of Peak ($W^*$) | $\mathcal{P}_4$ Retention (% of $\mathcal{P}_{\text{max}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2.0 s** | +7.86% | +17.17% | **+19.01%** | 63.0% | 90.3% |
| **4.0 s** | +9.63% | +19.00% | **+21.91%** | 72.6% | 86.7% |
| **6.0 s** | +9.80% | +19.13% | **+22.80%** | 75.5% | 83.9% |
| **8.0 s** | +10.96% | +20.63% | **+24.59%** | 81.5% | 83.9% |
| **10.0 s** | +10.51% | +20.79% | **+24.65%** | 81.7% | 84.3% |
| **15.0 s** | +11.12% | +21.99% | **+26.18%** | 86.7% | 84.0% |
| **20.0 s** | +11.50% | +23.37% | **+27.15%** | 90.0% | 86.1% |
| **25.0 s** | +11.76% | +23.90% | **+27.83%** | 92.2% | 85.9% |
| **30.0 s** | +11.96% | +24.37% | **+28.33%** | 93.9% | 86.0% |
| **40.0 s** | +12.25% | +25.01% | **+29.04%** | 96.2% | 86.1% |
| **50.0 s** | +12.48% | +25.35% | **+29.49%** | 97.7% | 86.0% |
| **60.0 s (Std)** | +12.67% | **+25.64%** | **+29.77%** | **98.6%** | **86.2%** |
| **68.3 s (W\*)** | — | — | **+30.18% (Model Peak)**| **100.0%** | — |
| **75.0 s** | +12.84% | +25.75% | **+29.98%** | 99.3% | 85.9% |
| **90.0 s** | +12.96% | **+25.76%** | **+30.05%** | 99.6% | 85.7% |
| **105.0 s** | +13.05% | +26.41% | **+30.01%** | 99.4% | 88.0% |
| **120.0 s** | +13.12% | +26.06% | **+29.99%** | 99.4% | 86.9% |
| **140.0 s** | +13.18% | +25.51% | **+29.88%** | 99.0% | 85.4% |
| **160.0 s** | +13.21% | +25.40% | **+29.74%** | 98.5% | 85.4% |
| **180.0 s** | +13.21% | +25.11% | **+29.50%** | 97.7% | 85.1% |
| **200.0 s** | +13.21% | +25.53% | **+29.20%** | 96.8% | 87.4% |
| **220.0 s** | +13.20% | +25.03% | **+28.86%** | 95.6% | 86.7% |
| **240.0 s** | +13.18% | +24.72% | **+28.51%** | 94.5% | 86.7% |
| **260.0 s** | +13.15% | +24.00% | **+28.14%** | 93.2% | 85.3% |
| **280.0 s** | +13.11% | +23.57% | **+27.79%** | 92.1% | 84.8% |
| **300.0 s** | +13.07% | +23.63% | **+27.47%** | 91.0% | 86.0% |
| **320.0 s** | +13.02% | +23.46% | **+27.17%** | 90.0% | 86.3% |
| **340.0 s** | +12.97% | +23.28% | **+26.89%** | 89.1% | 86.6% |
| **360.0 s** | +12.92% | +23.14% | **+26.60%** | 88.1% | 87.0% |
