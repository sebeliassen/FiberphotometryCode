# Executive Summary: Definitive Model Selection, Paradigm Shift, and Roadmap

**Date:** September 3, 2026  
**Status:** Major Model-Selection Milestone Complete  
**Document Number:** `09` (Presentation Preparation & Manuscript Integration)  
**Primary Assets Produced Today:**
- Pipeline Scripts:
  - [`run_definitive_coordinate_system_suite.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/presentation/run_definitive_coordinate_system_suite.py)
  - [`run_latent_regime_sufficiency.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/presentation/run_latent_regime_sufficiency.py)
  - [`run_entropy_null_and_shared_coupling.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/presentation/run_entropy_null_and_shared_coupling.py)
  - [`run_definitive_story_discrimination_suite.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/presentation/run_definitive_story_discrimination_suite.py)
- Publication Figures:
  - [`fig_definitive_coordinate_system_suite.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/cohort_comparisons/fig_definitive_coordinate_system_suite.png)
  - [`fig_latent_regime_sufficiency.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/cohort_comparisons/fig_latent_regime_sufficiency.png)
  - [`fig_entropy_null_and_shared_coupling.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/cohort_comparisons/fig_entropy_null_and_shared_coupling.png)
  - [`fig_definitive_story_discrimination_suite.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/cohort_comparisons/fig_definitive_story_discrimination_suite.png)
- Verified LOAO Database: 15 unique mice (8 WT, 7 KI) tested twice (Saline vs. Amphetamine) across 60,718 strictly matched observation anchors.

---

## 1. Where We Started vs. Where We Are

### The Old Hypothesis (Now Retired)
> *"DAT-KI is more Markovian, has less memory, and its predictability is merely a byproduct of lower repertoire entropy."*

### The Definitive Model (The Two-Scale Architecture)
$$\boxed{
\begin{aligned}
\textbf{Fast Scale (Local Transition):} \quad & P(S_{t+1} \mid S_t) \quad \text{\textbf{Both WT and KI are strictly 1st-order Markovian.}} \\
\textbf{Slow Scale (Macro-Regime):} \quad & Q_t \to P(Y_{t:t+120}) \quad \text{\textbf{DAT-KI is locked into persistent multi-minute regimes.}}
\end{aligned}
}$$

Today was a **model-selection day**: rather than collecting endless metrics, we implemented adversarial controls that systematically eliminated flawed narratives and isolated the true physical phenomenon.

---

## 2. What Hypotheses Were Formally Eliminated Today

| Old Hypothesis / Phrase | Empirical Verdict Today | What Replaced It |
| :--- | :--- | :--- |
| **"KI is more Markovian than WT"** | **FALSIFIED.** At the local single-bout scale ($S_{t+1}$), both WT and KI have nearly identical 1st-order memory depth ($+2.34$ vs $+2.41$ bits/bout). Higher orders ($m \ge 2$) fail in both. | Local transitions are 1st-order Markovian in **both** genotypes. |
| **"WT has deep sequential grammar"** | **FALSIFIED.** WT out-of-sample log loss worsens from $3.517$ bits (order 1) to $4.125$ bits (order 4). | WT does not possess deep local syntax; local syntax is shallow across genotypes. |
| **"History gain is just static entropy"** | **FALSIFIED.** Under circular temporal shift, each mouse's vocabulary, counts, and marginal entropy $H(S)$ are 100% preserved, but KI history gain collapses from $+6.71\%$ to $-2.04\%$ ($p = 0.0156$). | Temporal alignment to the future is required; static entropy alone cannot produce history gain. |
| **"History is driven by sequential syntax"** | **FALSIFIED.** Scrambling sequence order inside the 20-bout window produced identical gain ($+6.40\%$ chronological vs $+6.84\%$ shuffled, $p = 0.688$). | The signal is **Regime Composition** (bag of events), not syntactic sequence. |
| **"Four discrete attractor states"** | **RETIRED.** PCA achieves virtually identical predictive gain to NMF ($+4.72\%$ vs $+5.12\%$). | **A low-dimensional predictive subspace / manifold** that captures $76.3\%$ of history information. |
| **"Shared NMF proves KI uses fewer components"** | **CORRECTED.** In a shared $K=12$ basis, WT uses $N_{\text{eff}} = 9.31 \pm 0.36$ and KI uses $N_{\text{eff}} = 8.99 \pm 0.46$. | KI does **not** globally collapse component count; it **selectively reweights specific modes** (suppressing C3/C10, enriching C4/C9). |

---

## 3. The Core Empirical Findings

### A. The Two Temporal Scales (Fast Syntax vs. Slow Macro-Regimes)
1. **At the Single-Bout Scale ($S_{t+1}$):**
   * Predicting the immediate next action from recent history ($S_t, S_{t-1}, \dots$) shows an enormous boost from Order 0 to Order 1 ($+2.34$ bits/bout in WT, $+2.41$ bits/bout in KI).
   * Beyond Order 1, higher-order sequence context hurts held-out generalization due to transition matrix sparsity.
   * **Both genotypes execute behavior through simple 1st-order local transitions.**
2. **At the Multi-Minute Scale ($120\text{s}$ Future Repertoire):**
   * WT Saline retains essentially **zero predictive information** from past event composition ($\Delta_H = -0.06\%$). WT transitions fluidly across behavioral regimes.
   * DAT-KI Saline retains **massive predictive information** from past event composition ($\Delta_H = +6.71\%$).

### B. The Timescale of the Macro-Regime (Local Drift Null)
* When pairing a 20-bout history window with a target from the **same local 3–6 minute neighborhood** ($|\delta| \in [180\text{s}, 360\text{s}]$), KI Saline **retains $+5.91\%$ of its predictive gain**!
* It only drops to the null baseline ($+2.27\%$) when shifted 15–20 minutes away.
* **Interpretation:** DAT-KI is locked into **slowly drifting macroscopic regimes that persist across several minutes**. Within a 3–6 minute span, the animal remains trapped in the same regime, which is why local surrogates retain predictive power.

### C. Low-Dimensional Latent Sufficiency
* In DAT-KI, the 100-dimensional event-composition history is **$76.3\%$ captured by just $K=4$ latent dimensions** ($+5.12\%$ out of $+6.71\%$).
* The residual history information plunges to $+1.47\%$ at $K=4$ and becomes negative beyond $K=8$.
* This compression is identical under linear PCA ($+4.72\%$), proving it is an intrinsic geometric property of the behavioral subspace.

### D. Coordinate-System Dissociation (Trend)
* In WT Saline, context is organized primarily by **physical dwell time**: a 15-second physical clock window provides $+1.08\%$ gain over bout-based windows.
* In DAT-KI, context is organized by **discrete event counts**: physical duration weighting actively degrades predictive performance ($+6.71\% \to +4.01\%$).
* Genotype interaction contrast: $\Delta_{\text{WT} - \text{KI}} = +2.15\%$ ($p = 0.0954$).

---

## 4. The Unified Scientific Narrative for the Paper / Presentation

```
Level 1: Static Repertoire Constraint
   └── DAT-KI exhibits lower marginal syllable entropy and restricted overall repertoire.

Level 2: Fast vs. Slow Temporal Decoupling
   ├── Fast Scale: Both WT and KI execute behavior via simple 1st-order Markov transitions.
   │   WT has no deeper sequential syntax than KI.
   └── Slow Scale: WT rapidly mixes across macro-states, destroying event-history context.
       DAT-KI becomes locked into persistent multi-minute macro-regimes.

Level 3: Latent State Representation
   └── The multi-minute macro-regime in DAT-KI is:
       ├── Invariant to sequence order inside the window (composition, not grammar).
       ├── Degradable by dwell time (event-based clock).
       └── 76% compressible into a compact, 4-dimensional latent subspace.
```

---

## 5. Status of the Database & Verification
* **Subject IDs:** Verified from raw metadata `animal_id`.
* **Sample Size:** 15 unique mice in total:
  * WT ($N=8$): `728, 1680, 740, 1775, 1773, 1765, 1767, 1759`
  * KI ($N=7$): `1694, 1664, 1666, 1658, 1771, 1781, 1749`
* Each animal has exactly one Saline session and one Amphetamine session.
* Within-cohort cross-validation is **100% strictly Leave-One-Animal-Out (LOAO)**.

---

## 6. Clear Action Plan for Tomorrow

1. **Slide / Figure Harmonization:**
   * Integrate the definitive 4-panel figure ([`fig_definitive_story_discrimination_suite.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/cohort_comparisons/fig_definitive_story_discrimination_suite.png)) and the coordinate-system suite figure into the main slide deck outline in [`01_professor_pitch_narrative_and_slides.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/mds/05_presentation_prep/01_professor_pitch_narrative_and_slides.md).
2. **Explicit State-Space Modeling (HMM / HSMM):**
   * Now that we know behavior is governed by **1st-order local transitions + slow multi-minute macro-regimes**, a Hidden Markov Model (HMM) or Hidden Semi-Markov Model (HSMM) with $K=4\text{--}6$ latent states is mathematically justified (not just exploratory model shopping).
   * Fit an HMM to extract explicit state dwell times, transition matrices between macro-regimes, and state persistence curves to formalize the multi-minute regime locking in DAT-KI.
3. **Manuscript Section Drafting:**
   * Write the "Two Scales of Behavioral Organization" results section in [`04_master_research_synthesis_and_manuscript_draft.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/mds/05_presentation_prep/04_master_research_synthesis_and_manuscript_draft.md), using the exact statistical tables generated today.
