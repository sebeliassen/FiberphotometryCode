# Walkthrough: Language Models Structure & Skeletons Setup

This walkthrough documents the creation and directory organization of the modular Behavioral Language Models pipeline under `analysis/language_models`.

---

## 1. Directory Structure Created

The directory layout has been set up with modular subfolders under `lib/` to structure preprocessing, models, metrics, and visualization separately:

```
analysis/language_models/
├── run_lm_analysis.py         # Orchestrator and CLI entry point
├── lib/
│   ├── __init__.py
│   ├── preprocessing/         # Data tokenization & segmentation
│   │   ├── __init__.py
│   │   ├── rle.py             # Run-Length Encoding transition collapse
│   │   ├── segmentation.py    # Grouping sequences into sentences
│   │   └── tokenization.py    # Byte-Pair Encoding (BPE)
│   ├── models/                # Sequence model classes
│   │   ├── __init__.py
│   │   ├── embeddings.py      # Behavior2Vec (Word2Vec)
│   │   ├── discrete.py        # Discrete N-gram language models
│   │   └── continuous.py      # Continuous autoregressive pose predictions (10D PCA)
│   ├── evaluation/            # Validation & distance metrics
│   │   ├── __init__.py
│   │   ├── metrics.py         # Perplexity & Levenshtein distance
│   │   └── dtw.py             # Multi-dimensional Dynamic Time Warping
│   └── visualization/         # Visual summaries
│       ├── __init__.py
│       ├── embedding_viz.py   # UMAP & cosine heatmaps
│       └── metrics_viz.py     # Perplexity & DTW box/heat plots
└── output/                    # Destination for analytical artifacts
```

---

## 2. Key Modules & Implementations

### CLI Entrypoint
* **[run_lm_analysis.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/run_lm_analysis.py)**: Sets up standard argument parsing (e.g. `--dataset`, `--verify`, `--run-embeddings`, `--run-perplexity`, `--run-dtw`), imports the modular functions, and sets up session loads from the custom `loader.py`.

### Preprocessing
* **[rle.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/preprocessing/rle.py)**: Implement `collapse_transitions()` to clean syllable run-lengths before sequence modeling.
* **[segmentation.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/preprocessing/segmentation.py)**: Implement `segment_into_sentences()` mapping syllable streams to discrete sentence units.
* **[tokenization.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/preprocessing/tokenization.py)**: Implement `train_bpe_tokenizer()` applying Byte-Pair Encoding (BPE) to learn structural behavioral vocabularies.

### Models
* **[embeddings.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/models/embeddings.py)**: Implement `Behavior2Vec` support mapping syllables to continuous vectors, allowing injection of postural PCA centroids.
* **[discrete.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/models/discrete.py)**: Implement `DiscreteLanguageModel` (N-grams with Laplace/Kneser-Ney smoothing).
* **[continuous.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/models/continuous.py)**: Implement `ContinuousPoseModel` forecasting upcoming 10D postural components.

### Evaluation & Visualizations
* **[metrics.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/evaluation/metrics.py)**: Compute transition perplexities and edit distances.
* **[dtw.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/evaluation/dtw.py)**: Aligns continuous postural coordinates of behavioral sentences using Dynamic Time Warping.
* **[embedding_viz.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/visualization/embedding_viz.py)** & **[metrics_viz.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/visualization/metrics_viz.py)**: Visualize output distributions.

---

## 3. Verification & Phase 1.a Execution Results

We executed the orchestrator script with the `--verify` parameter:
```bash
python run_lm_analysis.py --verify
```

### Outputs & Control Over Resolution
* **Method 1 (Slow Syllable Segmentation)**: Selected bottom 10% slowest syllables (`[98, 85, 5, 2, 31, 81, 73, 55, 46, 13]`) resulting in **8 sentences** of average length **424.25**. Bottom 25% slowest syllables resulted in **31 sentences** of average length **107.32**.
* **Method 2 (Kinematic Idleness Slicing)**: Under hyperactive AMPH state conditions, continuous idle blocks are naturally rare. Cutoff at `idle_vel=15.0` resulted in **1 sentence** of average length **3404.00**, and `idle_vel=30.0` resulted in **2 sentences** of average length **1701.50**.

---

## 4. Hyperparameter Optimization Sweep Results

We executed the grid sweep optimizations:
```bash
conda run -n kpms_gpu_cuml python run_lm_analysis.py --run-optimization --dataset amph
```

### Method 1 Sweep (Percentiles)
Using weights $w_1 = 1.2$ (Genotype decodability) and $w_3 = 0.8$ (BPE compression):
* **Optimal percentile**: **0.150**
  * **Composite Score $S$**: **0.919**
  * **Genotype Decodability $M_1$**: **0.667** (accuracy above chance)
  * **BPE Compression $M_3$**: **0.149**

### Method 2 Sweep (Velocity & Duration)
* **Optimal parameters**: **`idle_vel = 50.0 px/s`**, **`min_duration = 20 frames`**
  * **Composite Score $S$**: **0.880**
  * **Genotype Decodability $M_1$**: **0.733**
  * **BPE Compression $M_3$**: **0.000** (due to high activity, few boundaries are generated per session, meaning BPE cannot find recurring pairs).

* **Plots Saved**:
  - `output/method1_optimization_sweep.png`
  - `output/method2_optimization_sweep.png`

* **Status**: **PASS** (Optimization completed successfully).
