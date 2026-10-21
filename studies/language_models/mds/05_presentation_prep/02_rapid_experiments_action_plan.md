# Rapid 1–2 Day Experiments: Action Plan & Figure Generation

---

## Executive Objective

With only **1 week until the presentation**, we do not need to train new complex deep learning architectures or debug third-party packages (like `state-moseq`). 

Instead, we focus our remaining **1–2 experimental days** on generating **three visual, tangible, and mathematically defensible figures** using our existing dataset of 30 open-field sessions ($T = 174,686\text{ bouts}$, 30,520 evaluation windows):

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE 3 RAPID EXPERIMENTS FOR THE SLIDES                        │
├───────────────────────┬─────────────────────────────────────┬───────────────────────────────────┤
│ Experiment            │ Methodology & Analysis              │ Target Presentation Figure        │
├───────────────────────┼─────────────────────────────────────┼───────────────────────────────────┤
│ 1. NMF Topic Cards    │ Fit Non-Negative Matrix             │ Slide 7: Multi-panel Topic Cards  │
│    & Exemplars        │ Factorization (K=4, 8) on window-   │ with enriched syllables, arena    │
│                       │ by-syllable counts; extract clips   │ heatmaps, and session trajectories│
├───────────────────────┼─────────────────────────────────────┼───────────────────────────────────┤
│ 2. Window Timescale   │ Sweep history window duration       │ Slide 5: Memory Capacity Curve    │
│    Memory Curve       │ W in [5s, 10s, 20s, 40s, 68s, 120s] │ proving why 68.3s is the optimal  │
│                       │ using simple linear Ridge           │ macro-integration scale           │
├───────────────────────┼─────────────────────────────────────┼───────────────────────────────────┤
│ 3. Matched Repertoire │ Mine pairs with ||u_A - u_B|| ~ 0;  │ Slide 8: Syntax Disambiguation    │
│    Syntax Test        │ correlate syntax distance d_order   │ Scatter plot proving order drives │
│                       │ with future divergence D_JS         │ divergent behavioral trajectories │
└───────────────────────┴─────────────────────────────────────┴───────────────────────────────────┘
```

---

## 1. Experiment 1: NMF Macro-Topic Extraction & Real Video Exemplars

### Scientific Goal:
Transform abstract syllable counts into a tangible, human-interpretable **Behavioral Topic Card** that the professor can immediately understand.

### Protocol:
1. **Input Matrix:** Load the existing document-term count matrix $\mathbf{X} \in \mathbb{R}^{N \times K}$ ($N = 30,520\text{ windows}$, $K = 71\text{ syllables}$).
2. **Decomposition:** Fit Non-Negative Matrix Factorization (NMF) with $K_{\text{topics}} \in \{4, 8\}$:
   $$\mathbf{X} \approx \mathbf{W} \mathbf{H}, \quad \mathbf{W} \in \mathbb{R}_{+}^{N \times K_{\text{topics}}}, \quad \mathbf{H} \in \mathbb{R}_{+}^{K_{\text{topics}} \times K}$$
3. **Topic Interpretation:**
   * For each topic $k$, extract the top 5 enriched syllables from row $\mathbf{H}_k$.
   * Plot the time-course activation $\mathbf{W}_{\text{session}, k}(t)$ over a 30-minute recording session (showing early exploration vs. late habituated resting).
   * Find the top 3 windows with highest activation $W_{i, k}$ and retrieve their raw video timestamps and spatial arena coordinates $(x,y)$.

### Target Slide Figure:
* **Panel A:** Heatmap of topic matrix $\mathbf{H}$ (e.g. *Topic 1: Wall Rearing & Running*, *Topic 2: Center Exploration*, *Topic 3: Micro-Grooming*, *Topic 4: Stationary Rest*).
* **Panel B:** Session trajectory $\theta_k(t)$ showing smooth transition between exploratory topics and maintenance topics across 30 minutes.
* **Panel C:** Spatial arena heatmaps showing where each topic occurs (perimeter vs. center).

---

## 2. Experiment 2: The Empirical Window Timescale Sweep ($W = 5\text{s} \to 120\text{s}$)

### Scientific Goal:
Provide empirical justification for why macroscopic modeling (~68 seconds) is necessary, answering the question: *"Why not just use a 5-second window?"*

### Protocol:
1. **Window Sweep:** Compute Bag-of-Words count vectors for evaluation windows with varying past duration:
   $$W \in \{5\text{s}, 10\text{s}, 20\text{s}, 40\text{s}, 68.3\text{s}, 90\text{s}, 120\text{s}\}$$
2. **Forecasting Objective:** Predict future syllable budget over fixed lookahead horizon $H = 120\text{s}$ using 5-fold cross-validated linear Ridge regression.
3. **Metric:** Plot Predictive Gain ($\% D_{\text{JS}}$ reduction relative to marginal unigram baseline) as a function of $W$.

### Expected Result:
* Predictive gain climbs sharply from $W = 5\text{s}$ ($\approx +12\%$) through $W = 20\text{s}$ ($\approx +24\%$), reaching a distinct knee/plateau at $W \approx 68.3\text{s}$ ($\approx +31.88\%$).
* **Takeaway:** Demonstrates that behavioral predictability accumulates over tens of seconds, establishing an empirical integration timescale for mouse macroscopic intent.

---

## 3. Experiment 3: The "Same Repertoire, Different Syntax" Statistical Test

### Scientific Goal:
Mathematically prove that chronological sequence order carries distinct strategic information beyond static syllable counts.

### Protocol:
1. **Pair Mining:** Search all pairs of evaluation windows $(W_A, W_B)$ across the dataset satisfying:
   $$\|\mathbf{u}_{\text{count}}(A) - \mathbf{u}_{\text{count}}(B)\|_2 \le \epsilon \quad (\text{top 1\% most similar count profiles})$$
2. **Syntax Distance ($d_{\text{order}}$):** Measure sequence order difference between $W_A$ and $W_B$ using:
   * Levenshtein edit distance on syllable token strings, OR
   * Euclidean distance in contextual Transformer embedding space $\|\mathbf{z}_{\text{ctx}}(A) - \mathbf{z}_{\text{ctx}}(B)\|_2$.
3. **Future Divergence ($D_{\text{JS}}$):** Measure the Jensen-Shannon divergence between their ground-truth future syllable distributions over $H = 120\text{s}$:
   $$D_{\text{JS}}\left( P_{\text{future}}(A) \,\|\, P_{\text{future}}(B) \right)$$
4. **Statistical Correlation:** Plot a scatter plot of $d_{\text{order}}(A, B)$ vs. $D_{\text{JS}}(\text{future}_A, \text{future}_B)$ and compute Spearman's rank correlation $r_s$.

### Expected Result:
* Significant positive correlation ($r_s > 0.25, p < 10^{-6}$): As sequence syntax diverges, future behavior diverges, even when current action counts are identical.
* **Takeaway:** Directly justifies why language models and sequential architectures are needed over simple Bag-of-Words.

---

## 4. Script Execution Plan

A single unified Python script (`run_rapid_presentation_figures.py`) can execute all three experiments and render publication-quality presentation figures in `< 30 minutes` of compute time:

```bash
python syllable_generation/analysis/language_models/pipelines/presentation/run_rapid_presentation_figures.py
```
