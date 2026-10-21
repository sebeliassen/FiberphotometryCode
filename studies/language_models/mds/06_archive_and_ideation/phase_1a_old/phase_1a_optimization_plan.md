# Optimization & Execution Plan: Phase 1.a Bayesian Hyperparameter Sweep

This document outlines the implementation and execution plan to programmatically search, evaluate, and optimize Phase 1.a sequence segmentation hyperparameters using Bayesian Optimization. It details the custom Gaussian Process regression framework, the log-transformed search space boundaries, the acquisition function, and caching optimizations.

---

## 1. The Weighted Composite Metric

To evaluate any segmentation candidate, we define a composite objective score $S$ that balances biological divergence ($M_1$) and information compression ($M_3$):

$$S = w_1 \cdot M_1 + w_3 \cdot M_3$$

Where:
* **$M_1$ (Class Separation Score)**: Normalized decodability. Let $A$ be the 5-fold cross-validated accuracy of a Random Forest Classifier trained to decode Genotype (`WT` vs `KI`) or Drug (`Saline` vs `AMPH`) from sentence frequencies. Let $C$ be the chance accuracy ($0.5$ for balanced binary classification).
  
  $$M_1 = \max\left(0, \frac{A - C}{1 - C}\right)$$

* **$M_3$ (Sequence Compression Score)**: BPE sequence length reduction ratio. Let $L_{raw}$ be the length of the collapsed RLE transition sequence and $L_{bpe}$ be the length of the sequence after applying Byte-Pair Encoding.
  
  $$M_3 = 1 - \frac{L_{bpe}}{L_{raw}}$$

* **$w_1, w_3$ (Weights)**: User-defined floats (e.g., $w_1 = 1.2$, $w_3 = 0.8$) to adjust prioritization.

---

## 2. Optimization Search Strategy (Bayesian Optimization)

Instead of a coarse grid search, we implement log-scale Bayesian Optimization using standard `scikit-learn` and `scipy` components.

### Bounds & Spaces
We expand the search space bounds to cover $1/10000$ to $10000 \times$ of the grid-search baseline optimal parameters, constrained by physical limits:

| Parameter | Baseline Optimal | Lower Bound | Upper Bound (Physically Constrained) | Search Space (Log-Scale) |
| :--- | :--- | :--- | :--- | :--- |
| **Method 1: Percentile** | `0.15` | `0.000015` | `0.99` (Capped to avoid empty sents) | $\log_{10}(p) \in [-4.82, -0.004]$ |
| **Method 2: idle_vel** | `50.0` px/s | `0.005` px/s | `300.0` px/s (Capped at max behavior speed) | $\log_{10}(v) \in [-2.30, 2.47]$ |
| **Method 2: min_duration** | `20` frames | `1` frame (Capped at positive integer) | `10000` frames (Capped at session limit) | $\log_{10}(d) \in [0.00, 4.00]$ |

### Custom GP + Expected Improvement Loop
1. **Surrogate Model**: `GaussianProcessRegressor` from `sklearn.gaussian_process` fitted with a Matern kernel (`nu=2.5`).
2. **Acquisition Function**: Expected Improvement (EI):
   
   $$EI(x) = (\mu(x) - f(x^+) - \xi)\Phi(Z) + \sigma(x)\phi(Z)$$
   
   where $\Phi$ and $\phi$ are the standard normal CDF and PDF, $f(x^+)$ is the best observed score, and $Z = \frac{\mu(x) - f(x^+) - \xi}{\sigma(x)}$. We use trade-off parameter $\xi = 0.01$.
3. **Candidate Proposal**: Generate 2,000 random coordinates within the log-scale bounds, compute EI for each, and evaluate the candidate maximizing EI.
4. **Budget**: 5 initial random points + 25 guided iterations (total of 30 evaluations per method).

---

## 3. Performance & Caching Optimizations

To evaluate 30 candidates efficiently:
1. **Parquet-Based Data Cache**: Pre-compute collapsed RLE sequences and frame-by-frame velocities once and store them in PyArrow `.parquet` format.
2. **NumPy Vectorized Segmentations**: Slicing active segments is vectorized in NumPy to run in milliseconds.
3. **Fixed-merge BPE Vocabulary**: Train BPE with a vocabulary size defined relative to the session's unique token count: `vocab_size = len(token_list) + 50`. This ensures exactly 50 merges are completed, fixing the bug where compression scores ($M_3$) returned 0.000 due to a negative number of merges.

---

## 4. Execution Plan (Phased Steps)

### Step 1: Implementation of the Bayesian Optimization Engine
* Modify `lib/preprocessing/optimization.py` to:
  * Implement expected improvement and the sequential sampling loop in log space.
  * Adjust BPE vocab training size to guarantee 50 merges are evaluated.

### Step 2: Verification of the BPE Metric Fix
* Run a single evaluation candidate to verify that Method 2 now returns non-zero values for `compression` ($M_3$).

### Step 3: Run Optimization & Trajectory Plotting
* Modify `run_lm_analysis.py` to invoke `bayesian_optimize_m1` and `bayesian_optimize_m2`.
* Modify `lib/visualization/metrics_viz.py` to:
  * Plot Method 1 scores on a semi-log scale.
  * Plot Method 2 evaluations as a 2D log-log scatter plot colored by score, drawing a trajectory line showing the optimization path.
* Run `--run-optimization` and output the optimal parameters and plots.
