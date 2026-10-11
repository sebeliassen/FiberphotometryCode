# Agent Handoff: Language Models & Hyperparameter Optimization (Phase 1.a)

This document provides a complete transition plan for a new chatbot to pick up the development, optimization, and analytical verification of the **Behavioral Language Models** pipeline.

---

## 1. Project Overview & Objective

### What We Are Doing
We are modeling mouse open-field behavior as a "language" by treating behavioral syllables (discrete postural states extracted via Keypoint MoSeq) as words. The raw syllable stream is long, highly repetitive, and lacks sentence boundaries. 

We segment the continuous syllable stream into **discrete sentence units** using two alternative rules:
1. **Method 1 (Slow Syllable Slicing)**: Identify slow-transitioning syllables (long-duration postural pauses) as natural boundaries and use them to split the sequences.
2. **Method 2 (Kinematic Idleness Slicing)**: Segment sentences using epochs of active locomotion. An "active bout" begins when the animal's centroid velocity rises above an active threshold and ends when it remains below an idle threshold for a minimum duration.

### Why We Are Doing This
Standard syllable sequences are noisy. Defining sentence boundaries lets us apply NLP tools:
* **Perplexity & Discrete LMs**: Analyze behavioral grammar/syntax.
* **Embeddings (Behavior2Vec)**: Group similar syllable states using context.
* **DTW Alignment**: Compare spatial/kinematic properties of behavioral sentences.

### How We Evaluate Success (The Joint Optimization)
Based on initial validation sweeps where Method 1 (Slow Syllable Slicing) significantly outperformed Method 2 (Kinematic Slicing), **Method 2 has been retired**.

From now on, we exclusively use Method 1. We optimize its single parameter (percentile $P$) using a joint composite score metric:
$$S = 1.0 \cdot M_1 + 1.0 \cdot M_3$$
where:
* **$M_1$ (Genotype Decodability)**: Normalized 5-fold cross-validated Random Forest accuracy (chance is subtracted out) at classifying WT vs. KI mice based on their sentence-level syllable probability distributions.
* **$M_3$ (BPE Sequence Compression)**: Byte-Pair Encoding (BPE) compression ratio of the segmented sentences using exactly 50 merges. Higher compression indicates recurring structural motifs (words/grammar).

---

## 2. Technical Stack & Environment

* **Conda Environment**: `kpms_gpu_cuml` (containing scikit-learn, scipy, pandas, pyarrow, and W&B).
* **Location**: `/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/`
* **Main Orchestrator**: `run_lm_analysis.py`
* **W&B Integration**: Project is synced live to the user's dashboard at `sebeliassen/language_models_in_vivo`.

### Module Structure
```
analysis/language_models/
├── run_lm_analysis.py         # Main entry point and orchestration script
├── lib/
│   ├── preprocessing/
│   │   ├── rle.py             # Run-Length Encoding transition collapse
│   │   ├── segmentation.py    # Syllable/Kinematic sentence segmenters
│   │   ├── tokenization.py    # Byte-Pair Encoding (BPE) tokenizer
│   │   └── optimization.py    # Advanced log-scale Bayesian Optimization
│   ├── models/
│   │   ├── embeddings.py      # Behavior2Vec (Word2Vec) embeddings
│   │   ├── discrete.py        # Discrete N-gram language models
│   │   └── continuous.py      # Continuous postural PCA forecasting
│   ├── evaluation/
│   │   ├── metrics.py         # Perplexity & Levenshtein distance
│   │   └── dtw.py             # Multi-dimensional Dynamic Time Warping
│   └── visualization/
│       ├── embedding_viz.py   # Embedding cosine heatmaps
│       └── metrics_viz.py     # Perplexity boxplots & sweep plots
└── mds/
    └── phase_1a/
        ├── phase_1a_implementation_plan.md
        ├── phase_1a_optimization_plan.md
        └── agent_handoff.md   # This file
```

---

## 3. Current Task Status

The original 2x2 hyperparameter optimization sweeps are complete (Method 1 achieved 93.3% decodability accuracy, leading to the retirement of Method 2). 

We are currently running a joint optimization sweep for **Method 1 (Slow Syllable Slicing)** with weights $w_1 = 1.0, w_3 = 1.0$ on the baseline pre-injection phase of the **AMPH** dataset.

* **Background Task ID**: `8676ca3f-3621-40bc-980e-30e4a47c52e6/task-949`
* **Log File**: [task-949.log](file:///home/fsp585/.gemini/antigravity-ide/brain/8676ca3f-3621-40bc-980e-30e4a47c52e6/.system_generated/tasks/task-949.log)
* **Dataset Config**: Slices the AMPH dataset to the pre-injection baseline phase (`amph_phase="pre"`, drug-free baseline state) comparing **WT** ($N=16$) and **KI** ($N=14$).
* **W&B Live Run**:
  * `method1-composite-amph` (Active/Running)

---

## 4. How the Optimization Works (Search Logic)

To guarantee that we find the global optimum with high probability and do not get stuck in flat or low-score local minima, the optimization in `lib/preprocessing/optimization.py` uses the following advanced global search features:

1. **Latin Hypercube Sampling (LHS)**: Sets up the initial 50 evaluations using a space-filling LHS design to ensure uniform coverage across the parameter space.
2. **GP Multi-Start Fitting**: Re-optimizes the GP Matern kernel lengthscales 20 times (`n_restarts_optimizer=20`) to avoid poor lengthscale configurations.
3. **Acquisition Portfolio Hedging (EI + UCB)**: Alternates every iteration between **Expected Improvement (EI)** (to exploit promising peaks) and **Upper Confidence Bound (UCB)** with $\beta = 3.0$ (to search high-uncertainty spaces), preventing early convergence trap.
4. **L-BFGS-B Polishing**: Queries a dense grid of 5,000 points, then runs local optimization (`scipy.optimize.minimize`) from the peak to find the exact mathematical peak of the acquisition function.

---

## 5. Next Steps for Continuing

When you pick up this project:

1. **Monitor the Runs**:
   * Inspect the log file `task-805.log` or watch live on W&B until all 4 runs of the 2x2 matrix finish.
2. **Extract the 4 Optimal Parameters**:
   * Inspect the tables printed at the end of the log or W&B run summaries to locate the optimal hyperparameter values for each configuration:
     * **Method 1 (Decodability)**: Optimal percentile $p^*$.
     * **Method 1 (Compression)**: Optimal percentile $p^*$.
     * **Method 2 (Decodability)**: Optimal velocity $v^*$ and duration $d^*$.
     * **Method 2 (Compression)**: Optimal velocity $v^*$ and duration $d^*$.
3. **Analyze & Compare**:
   * Compare how the optimal segmentation rules differ when maximizing WT vs KI decodability vs sequence compression.
4. **Next Phase (Discrete Language Modeling)**:
   * Choose the optimal segmentation settings (e.g. Method 2 optimized for decodability) to segment all baseline session sequences.
   * Train N-gram language models (unigram, bigram, trigram with Laplace/Kneser-Ney smoothing) on WT and KI sentences separately.
   * Evaluate cross-genotype perplexity to measure grammatical syntax differences between genotypes (WT-trained model predicting KI, and vice-versa).
