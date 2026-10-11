# LM Science Hyperparameter Sweeps & Model Baseline Decision (Tuesday, August 4, 2026)

This update records the complete empirical evaluation of LM-inspired hyperparameter extensions (Axis 4 & Final LM Science Axes) and establishes the final **Behavioral Strategy Model Baseline**.

---

## 1. What We Tried (XYZ)

We benchmarked 4 advanced language modeling and feature engineering extensions against our baseline transition model across 32 total experimental configurations:

1. **Postural Kernel Smoothing ($\sigma \in [0.25, 1.00]$)**: Diffusing transition probabilities between posturally similar 10D pose centroids.
2. **Higher-Order $N$-Gram Syntax ($N \in [2, 3]$)**: Extending transition horizons beyond single steps.
3. **TF-IDF / IDF Stopword Downweighting**: Applying Inverse Document Frequency ($\text{IDF}$) to downweight ubiquitous baseline transitions.
4. **Kinetic Arousal Window Pooling**: Weighting window frames by movement velocity/acceleration ($\text{Arousal}_t = v_t + |a_t|$).

---

## 2. Why It Didn't Work (ABC)

| Extension Tested | Performance Impact | Root Cause Analysis (Why It Failed) |
| :--- | :---: | :--- |
| **Postural Kernel Smoothing ($\sigma > 0$)** | Acc dropped 71.67% $\to$ **40.0%–55.0%** | **Over-Smoothing**: Blurs the sharp, specific transition probabilities (e.g. rearing $\to$ perimeter run) that distinguish `WT` from `KI` mice. |
| **Higher $N$-Gram Syntax ($N \ge 3$)** | Acc dropped 66.67% $\to$ **55.0%** ($K=5$) | **Exceeds Memory Horizon**: Animal behavioral strategies operate primarily on 1-to-2 step transition horizons. $N \ge 3$ introduces noise & cluster collapse. |
| **TF-IDF / IDF Downweighting** | Acc collapsed 66.67% $\to$ **26.67%** ($K=3$) | **Behavioral Stopwords Are Informative**: Unlike NLP where "the"/"is" are meaningless, baseline behavioral transitions (static breathing, wall sniffing) define essential baseline rhythm and cadence. Downweighting them distorts UMAP density. |
| **Kinetic Arousal Window Pooling** | Acc dropped 71.67% $\to$ **56.67%** | **Suppresses Quiet States**: Over-emphasizes fast running and suppresses quiet grooming/thigmotaxis, which are just as vital for mouse phenotyping. |

---

## 3. Final Model Decision (LMN)

By **Occam's Razor**, the simplest model is the mathematical champion. We lock down the unweighted 1-step transition probability matrix ($P(S_t \mid S_{t-1})$) without artificial smoothing, IDF scaling, or arousal masking.

### 🏆 Locked Production Baselines:
1. **The 9-Vibe Model (`HDBSCAN mcs=300, N=1, σ=0.00`)**:
   * **Accuracy**: **66.67%** (vs 36.67% raw syllable baseline).
   * **Entropy Balance ($H_{\text{norm}}$)**: **0.944** ($K_{\text{eff}} = 7.97$).
   * **Use Case**: Primary model for cohort analytics, Vibe Passports, and fiberphotometry signal alignment.
2. **The 20-Vibe Model (`HDBSCAN mcs=150, N=1, σ=0.00`)**:
   * **Accuracy**: **71.67%** (All-time project high).
   * **Entropy Balance ($H_{\text{norm}}$)**: **0.924** ($K_{\text{eff}} = 15.94$).
   * **Use Case**: Maximum decodability classification benchmark.
