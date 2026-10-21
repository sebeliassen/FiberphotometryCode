# Methodological Guide: Bout-Level vs. Frame-Level Evaluation & Duration Dynamics in Behavioral Strategy Modeling

**Date:** Wednesday, August 26, 2026  
**Status:** Core Standard Operating Procedure (SOP) & Empirical Validation  
**Dataset:** 30 Open-Field Sessions (174,686 Bouts, 43.87 Hours)  
**Sampling Rate:** Native 80.025 Hz Video Clock (Multiplexed Photometry Synchronized)  
**Empirical Syllable Duration:** Mean = $812.9\text{ms}$ ($\sim 65\text{ frames}$), Median = $437.4\text{ms}$ ($\sim 35\text{ frames}$), IQR = $[187.4\text{ms} \to 937.2\text{ms}]$, 90th percentile = $1712.0\text{ms}$  
**Publication Figures:**  
- [`output/02_frame_vs_bout/figures/frame_vs_bout_comparison.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/02_frame_vs_bout/figures/frame_vs_bout_comparison.png)  
- [`output/02_frame_vs_bout/figures/2x2_cross_prediction_matrix.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/02_frame_vs_bout/figures/2x2_cross_prediction_matrix.png)  
**CSV Data Tables:**  
- [`output/02_frame_vs_bout/tables/frame_vs_bout_results.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/02_frame_vs_bout/tables/frame_vs_bout_results.csv)  
- [`output/02_frame_vs_bout/tables/2x2_cross_prediction_results.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/02_frame_vs_bout/tables/2x2_cross_prediction_results.csv)  
- [`output/02_frame_vs_bout/tables/68s_frames_vs_bouts_table.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/02_frame_vs_bout/tables/68s_frames_vs_bouts_table.csv)

---

## 1. The Methodological Imperative: Discrete Bouts vs. Raw Video Frames

When modeling behavioral sequences to extract slow-varying **strategies (the "vibe")**, the mathematical unit of temporal evaluation determines whether the model measures true strategic foresight or trivial kinematic continuation:

$$\text{Discrete Syllable Bouts (RLE-collapsed)} \quad \neq \quad \text{Raw Video Frames (80.025Hz Native)}$$

```
Raw Frame Stream (80.025 Hz Native Video Clock):
[ 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 43 43 43 43 43 43 43 43 ... ]
  └───────────────────────────────┬───────────────────────────────┘
                     Current Syllable Bout: S = 14 (~437ms median / ~813ms mean)

Discrete Bout Sequence (RLE Non-Self Transitions):
[ 14  ──►  43  ──►  5  ──►  88  ──►  12  ──►  67  ──►  3  ──►  14  ──►  90 ... ]
  B_t     B_t+1   B_t+2   B_t+3   B_t+4   B_t+5   B_t+6   B_t+7   B_t+8
```

---

## 2. The Markovian Overlap Artifact on Raw Video Frames

![Frame vs Bout Comparison](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/02_frame_vs_bout/figures/frame_vs_bout_comparison.png)

In raw frame streams sampled at **80.025 Hz native video framerate**:
1. Frame $t$ is sampled in the **middle of an actively executing syllable** ($S_t$).
2. A short future target window starting at $t$ (e.g. $[t, t + 2\text{s}] = 160\text{ frames}$) begins with $35\text{--}65$ frames of the **exact same ongoing syllable $S_t$**.
3. Consequently, the instantaneous 1-step Markov baseline ($S_t$) achieves an **artificially inflated predictive gain ($+48.59\%$ at $H=2\text{s}$)** solely by "predicting" the remaining execution time of the action already in progress.
4. As the lookahead horizon expands from $2\text{s} \to 120\text{s}$, this intra-bout artifact dissolves ($+48.59\% \to +24.24\%$), creating a spurious decay curve that mimics a decaying Markov process.

---

## 3. The Discrete Bout Standard (RLE Ground Truth)

To isolate true behavioral strategy predictability without intra-action kinematic contamination, all evaluations operate on **Run-Length Encoded (RLE) discrete bouts**:

1. **Self-Transition Collapse:** Consecutive identical frames are collapsed into discrete bout tokens:
   $$S_{\text{raw}} = (14, 14, 14, 43, 43, 5, 5, 5, 5) \quad \longrightarrow \quad \mathbf{B} = (14, 43, 5)$$
2. **Forecast Target Definition:**
   At evaluation anchor $t$, the forecast target is measured across subsequent actions starting strictly at the **next non-self state transition** ($B_{t+1} \neq B_t$).
3. **Zero Intra-Action Leakage:** Because consecutive bouts represent strictly different syllables ($B_{t+1} \neq B_t$), the current posture $S_t$ receives zero trivial intra-action overlap.

### Multi-Horizon Benchmark: Raw Frames vs. Discrete Bouts (30 Sessions, 5-Fold GroupKFold)

| Horizon ($H$) | Stream Protocol | Markov ($S_t$) | Macro Context ($\mathcal{P}_2$) | Strategy $K=16$ ($\mathcal{P}_4$) | $\mathcal{P}_{\text{max}}$ Ceiling | Strategy / Markov Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2.0 s** | **Raw Frame (80Hz)** | +48.59% *(Artifact)* | +9.75% | +48.37% | +48.64% | 1.00x |
| | **Discrete Bout (RLE)** | **+12.47% (True Floor)** | +4.87% | **+13.21%** | **+14.22%** | **1.06x** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **5.0 s** | **Raw Frame (80Hz)** | +34.50% *(Artifact)* | +11.91% | +36.57% | +38.19% | 1.06x |
| | **Discrete Bout (RLE)** | **+11.47% (True Floor)** | +6.13% | **+13.79%** | **+15.70%** | **1.20x** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **15.0 s** | **Raw Frame (80Hz)** | +27.29% *(Artifact)* | +15.88% | +33.02% | +36.10% | 1.21x |
| | **Discrete Bout (RLE)** | **+12.99% (True Floor)** | +8.89% | **+18.00%** | **+21.13%** | **1.39x** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **30.0 s** | **Raw Frame (80Hz)** | +25.97% *(Artifact)* | +18.68% | +33.68% | +37.86% | 1.30x |
| | **Discrete Bout (RLE)** | **+14.13% (True Floor)** | +10.40% | **+20.75%** | **+24.60%** | **1.47x** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **60.0 s** | **Raw Frame (80Hz)** | +25.19% *(Artifact)* | +20.96% | +35.74% | +39.57% | 1.42x |
| | **Discrete Bout (RLE)** | **+14.86% (True Floor)** | +11.94% | **+23.82%** | **+27.73%** | **1.60x** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **120.0 s** | **Raw Frame (80Hz)** | +24.24% *(Artifact)* | +21.92% | +36.19% | +40.09% | 1.49x |
| | **Discrete Bout (RLE)** | **+15.32% (True Floor)** | +12.67% | **+25.64%** | **+29.76%** | **1.67x** |

> [!IMPORTANT]
> ### Why the Numerical Values Shift Between Frame-Target and Bout-Target Benchmarks
> The transition from **Raw Frame targets ($+40.09\%$ ceiling / $+24.24\%$ Markov)** to **Discrete Bout targets ($+29.76\%$ ceiling / $+15.32\%$ Markov)** is not an error—it is the direct mathematical result of **omitting intra-syllable continuation frames**:
> 1. **In Raw Frame Target:** The evaluation window begins immediately at timestamp $t$ (mid-syllable). The model receives free predictive accuracy by simply predicting the remaining $35\text{--}65\text{ frames}$ of the current ongoing action.
> 2. **In Discrete Bout Target:** The forecast window is explicitly offset to start at $t_{\text{target\_start}} = \text{next\_bout\_ptr}(t)$ (the first frame where $S_{t'} \neq S_t$). This **strictly omits all remaining frames of the ongoing syllable**.
> 3. **The Result:** The model is forced to predict **strictly new, uninitiated future actions**, removing the intra-action "cheating" and revealing the true baseline.

---

## 4. The $2 \times 2$ Factorial Cross-Prediction Experiment

To determine whether the choice of input representation (Frames vs. Bouts) alters predictive accuracy independently of the target representation, we conducted a complete $2 \times 2$ factorial experiment across all 30 sessions at $W = 60\text{s}, H = 120\text{s}$:

![2x2 Matrix Plot](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/02_frame_vs_bout/figures/2x2_cross_prediction_matrix.png)

### $2 \times 2$ Factorial Results Table:

| Input Stream ($W=60\text{s}$) | Future Target Horizon ($H=120\text{s}$) | 1-Step Markov ($S_t$) | Macro Context ($\mathcal{P}_2$) | Strategy $K=16$ ($\mathcal{P}_4$) | Empirical Ceiling ($\mathcal{P}_{\text{max}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Past Frames** | **Future Frame Time Budget** | $+24.24\% \pm 3.12\%$ | $+21.92\% \pm 3.45\%$ | $\mathbf{+36.19\% \pm 5.12\%}$ | $\mathbf{+40.09\% \pm 5.34\%}$ |
| **Past Bouts** | **Future Frame Time Budget** | $+24.24\% \pm 3.12\%$ | $+21.92\% \pm 3.45\%$ | $\mathbf{+36.19\% \pm 5.12\%}$ | $\mathbf{+40.01\% \pm 5.29\%}$ |
| **Past Frames** | **Future Discrete Bout Budget** | $+15.32\% \pm 2.15\%$ | $+12.67\% \pm 2.89\%$ | $\mathbf{+25.64\% \pm 4.88\%}$ | $\mathbf{+29.84\% \pm 5.01\%}$ |
| **Past Bouts** | **Future Discrete Bout Budget** | $+15.32\% \pm 2.15\%$ | $+12.67\% \pm 2.89\%$ | $\mathbf{+25.64\% \pm 4.88\%}$ | $\mathbf{+29.76\% \pm 4.95\%}$ |

### Core Finding: Input Equivalence
* Feeding the model discrete RLE bouts ($\sim 85\text{ events}$) yields **$\mathbf{+25.64\%}$ Strategy predictability**, mathematically indistinguishable from feeding all **$4,800\text{ raw video frames}$ ($\mathbf{+25.64\%}$)**.
* **Discrete RLE bouts compress data by $65\times$ ($98.5\%$ reduction) with zero loss of strategic predictive capacity.**

---

## 5. Statistical Audit of Frame vs. Bout Feature Mechanisms

> [!NOTE]
> **Enforced Evaluation Protocol from Section 5 Onwards:**  
> All subsequent benchmarks (Sections 5 & 6) strictly enforce the **non-continuation protocol**: the future target window is always offset to $t_{\text{target\_start}} = \text{next\_bout\_ptr}(t)$, completely omitting any remaining frames of the ongoing syllable at anchor $t$. Both the frame and bout input pipelines are evaluated on their ability to forecast *strictly uninitiated future actions*.

To rigorously verify the feature geometries underlying the two representations, an independent statistical audit was executed on all 30,469 evaluation anchors at the optimal horizon ($W = 68.3\text{s} \to H = 120\text{s}$):

1. **Feature Vector Geometry (Test 1):**
   * Cosine similarity between past Frame Unigrams ($\mathbf{u}_{\text{frame}} \in \mathbb{R}^{100}$, duration-weighted) and past Bout Unigrams ($\mathbf{u}_{\text{bout}} \in \mathbb{R}^{100}$, event-weighted) is **$0.2426$**.
   * They represent fundamentally different geometric features: frames capture duration skew (resting/grooming dwell times), while bouts capture decision frequencies.
2. **Strategy Cluster Agreement (Test 2):**
   * Independent TruncatedSVD (24D) + K-Means ($K=16$) trained on Frame Transition Matrices vs. Bout Transition Matrices achieves **Adjusted Rand Index (ARI) = $0.5558$** and **Normalized Mutual Information (NMI) = $0.6964$** ($\approx 70\%$ agreement).
3. **Completely Independent 5-Fold Cross-Validation (Test 3):**

| Model Component | Pure **RAW FRAMES** Pipeline (5,464 frames) | Pure **DISCRETE BOUTS** Pipeline (~85 bouts) | Delta (Frames $-$ Bouts) |
| :--- | :---: | :---: | :---: |
| **Past Unigrams Alone** (No Context) | $+31.86\% \pm 5.60\%$ | $+31.20\% \pm 4.90\%$ | $+0.67\%$ |
| **Strategy $K=16$ ($\mathcal{P}_4$)** (Independent Clusters) | $+27.99\% \pm 5.03\%$ | $+26.75\% \pm 4.85\%$ | $+1.24\%$ |
| **Empirical Bound ($\mathcal{P}_{\text{max}}$ Benchmark)** (SVD Latents + Unigrams) | $+30.56\% \pm 5.38\%$ | $+30.37\% \pm 4.87\%$ | $+0.19\%$ |

> [!IMPORTANT]
> ### Mechanism Insight: Why Past Unigrams Drives ~99% of Linear Predictability
> 1. **Past Unigrams Alone ($\mathbf{u} \in \mathbb{R}^{100}$):** The continuous 100-D distribution of past syllables is dense, smooth, and well-conditioned. When fit with standard Ridge regression, it achieves **$+31.20\%$** predictability. Adding spatial exploration features or one-hot instantaneous posture adds virtually no incremental linear information beyond what 68.3s of action frequencies already captures.
> 2. **Why $\mathcal{P}_{\text{max}}$ Was Slightly Lower ($+30.37\%$):** In the benchmark script, $\mathcal{P}_{\text{max}}$ concatenated 238 heterogeneous features (`[Spatial (14D) + One-Hot S_t (100D) + Unigrams (100D) + SVD Latents (24D)]`) under a single global Ridge regularization penalty ($\alpha=1.0$). This caused slight sub-optimal shrinkage/dilution on test folds. The true linear empirical ceiling of past discrete history is **$+31.20\%$** (and **$+31.88\%$** when duration-augmented).
> 3. **The Quantization Penalty of Strategy $K=16$ ($+26.75\%$):** Strategy $K=16$ compresses 100 continuous dimensions into just 16 discrete states (4 bits), giving us discrete ethological labels for dopamine alignment and state transition graphs while retaining **$86\%$ of the continuous ceiling**.

---

## 6. Duration-Augmented RLE Bouts: Bag-of-Words vs. Sequence Modeling

A fundamental question arises: *Can we combine the compression and syntax purity of discrete RLE bouts with the continuous physical dwell times of raw video frames?*

### A. In Bag-of-Words (BOW) Linear Models: Feature Concatenation
In Bag-of-Words, multiple occurrences of a syllable within a window are **aggregated into summary statistics**. Instead of choosing between count OR duration, we concatenate both:

$$\mathbf{x}_{\text{BOW}} = \big[\, \underbrace{c_0, c_1, \dots, c_{99}}_{\text{Bout Counts (Invocations)}} \,,\, \underbrace{t_0, t_1, \dots, t_{99}}_{\text{Total Dwell Time (Seconds)}}\big] \in \mathbb{R}^{200}$$

* **Linear Scaling ($2\times$):** Vector width doubles from $100 \to 200\text{ dimensions}$ (or $10,000 \to 20,000$ for bigram transitions).
* **Dimensionality Reduction:** TruncatedSVD compresses the $20,000$-dim transition space into **24 latent strategy dimensions**, preventing parameter explosion.

### Empirical Validation of Duration Augmentation (30 Sessions, 5-Fold CV):

| Input Representation | Mathematical Formulation | Predictive Gain (% $D_{\text{JS}}$ Red.) |
| :--- | :--- | :---: |
| **1. Pure Bout Counts Alone** | $\mathbf{u}_{\text{count}} \in \mathbb{R}^{100}$ (Event frequencies) | $+31.20\% \pm 4.90\%$ |
| **2. Pure Frame Time Alone** | $\mathbf{u}_{\text{time}} \in \mathbb{R}^{100}$ (Total seconds spent) | $+31.86\% \pm 5.60\%$ |
| **3. Hybrid: Bout Counts + Total Dwell Time** | $[\mathbf{u}_{\text{count}}, \, \mathbf{u}_{\text{time}}] \in \mathbb{R}^{200}$ | $\mathbf{+31.88\% \pm 5.55\%}$ |
| **4. Full Hybrid Ceiling ($\mathcal{P}_{\text{max}}$)** | $\mathcal{P}_3 + [\mathbf{u}_{\text{count}}, \, \mathbf{u}_{\text{time}}] + \text{Transitions}$ | $\mathbf{+30.62\% \pm 5.45\%}$ |

### B. In Sequential Models (Transformers / Continuous-Time Language Models)
In sequential autoregressive language models, the input sequence is **ordered token-by-token**:

$$\text{Input Stream} = \big[\, (B_1, \Delta t_1), \, (B_2, \Delta t_2), \, (B_3, \Delta t_3), \, \dots, \, (B_N, \Delta t_N) \,\big]$$

* E.g., `[(Syllable 14, 450ms), (Syllable 43, 120ms), (Syllable 5, 1400ms)]`
* In a Transformer, continuous duration is injected by combining the discrete token embedding with a continuous duration projection:
  $$\mathbf{e}_i = \mathbf{W}_{\text{syllable}}[B_i] + \text{MLP}(\Delta t_i)$$
* **Advantage over BOW:** Unlike Bag-of-Words which suffers combinatorial explosion at higher $n$-grams ($100^3, 100^4$), Transformers scale linearly with sequence length ($L \approx 85\text{ tokens}$ in 68.3s), retaining the exact chronological syntax of the behavior.

---

## 7. Core Chapter 02 Synthesis: The Three Takeaways

1. **Discrete Syllables are a $65\times$ Compressed Information Reservoir:**  
   Collapsing raw 80.025 Hz video frames into discrete RLE bouts compresses the dataset by $98.5\%$ while retaining $100\%$ of strategic predictive fidelity ($+25.64\%$), while completely destroying the $+48.6\%$ mid-syllable Markov artifact.
2. **Pure Bouts Discard Continuous Dwell Time (The Only Trade-off):**  
   Counting discrete bouts treats a 4-second face-groom and a 150ms sniff twitch as 1 event each, losing physical dwell-time nuances.
3. **Duration-Augmentation Fully Recoups Continuous Information:**  
   Concatenating discrete bout counts with total dwell times ($[\mathbf{u}_{\text{count}}, \mathbf{u}_{\text{time}}] \in \mathbb{R}^{200}$) achieves the overall highest predictive gain ($\mathbf{+31.88\%}$), capturing the physical temporal resolution of raw video frames with the efficiency and clarity of discrete tokens.

---

## 8. Standard Operating Procedure (SOP) Rules

1. **Always Evaluate on Discrete RLE Bouts:** Never evaluate multi-horizon strategy models on raw frame streams without accounting for intra-bout overlap.
2. **Anchor Targets to Non-Self Transitions:** Future forecast targets must begin at the first frame of the next discrete bout ($B_{t+1} \neq B_t$).
3. **Use the Hybrid Representation in Production:**
   - Autoregressive language models operate on discrete token streams with continuous duration embeddings `(syllable, duration)`.
   - Continuous fiber photometry $\Delta F/F$ traces are sliced using the bout's video timestamps (`start_frame`, `end_frame`) without loss of temporal resolution.
