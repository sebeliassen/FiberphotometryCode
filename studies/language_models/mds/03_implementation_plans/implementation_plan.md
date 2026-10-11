# Implementation Plan: Behavioral Language Models Analysis Framework

This plan details the directory layout, module definitions, and workflow steps to establish a behavioral language processing pipeline in the newly created `/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models` directory. It adapts concepts from natural language processing and speech recognition to discrete MoSeq syllables and continuous postural latent states.

---

## Proposed Directory Structure

We will organize the folder with subdirectories under `lib/` to keep code modular and prevent files from becoming too large:

```
analysis/language_models/
├── README.md                 # Framework documentation and math formulas
├── run_lm_analysis.py         # Orchestrator CLI entrypoint (similar to run_analysis.py)
├── lib/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── rle.py            # RLE collapsing (collapsing consecutive syllables)
│   │   ├── segmentation.py   # Bout segmentation (sentence boundaries)
│   │   └── tokenization.py   # BPE tokenization (learning behavioral words)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embeddings.py     # Behavior2Vec (Word2Vec)
│   │   ├── discrete.py       # N-gram & LSTM discrete language models
│   │   └── continuous.py     # Autoregressive PCA models (Continuous GPT)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py        # Perplexity & Levenshtein distance metrics
│   │   └── dtw.py            # Continuous dynamic time warping alignment
│   └── visualization/
│       ├── __init__.py
│       ├── embedding_viz.py  # Cosine heatmaps and UMAPs of embeddings
│       └── metrics_viz.py    # Perplexity and DTW comparison plots
└── output/                   # Directory to save plots, stats, and cached models
```

---

## Proposed Changes

### 1. Preprocessing and Tokenization (`lib/preprocessing/`)
This subfolder handles syllable stream cleanup and segmentation:
* **`rle.py`**:
  * `collapse_transitions(session)`: Collapses run-length frames of identical syllables/states.
* **`segmentation.py`**:
  * `segment_into_sentences(session, method="rest_based", threshold_s=1.0)`: Binds transitions into behavioral sentences using pauses/rests or fixed window durations.
* **`tokenization.py`**:
  * `train_bpe_tokenizer(sequences, vocab_size=100)`: Merges the most frequent adjacent syllable transitions to learn structural chunks.

### 2. Modeling Framework (`lib/models/`)
Contains behavioral models:
* **`embeddings.py`**:
  * `Behavior2Vec`: Maps syllables to continuous vectors using transition context. Supports postural constraints (initializing centroids via PCA values).
* **`discrete.py`**:
  * `DiscreteLanguageModel`: N-gram/LSTM models predicting the next syllable $S_{t+1}$.
* **`continuous.py`**:
  * `ContinuousPoseModel`: Autoregressive forecasting model predicting continuous postural variables $X(t+1) \in \mathbb{R}^{10}$.

### 3. Evaluation Metrics (`lib/evaluation/`)
Implements statistical comparisons:
* **`metrics.py`**:
  * `calculate_perplexity(model, sequences)`: Sequence transition surprise.
  * `calculate_edit_distance(seq_a, seq_b)`: Levenshtein distance on string sentences.
* **`dtw.py`**:
  * `calculate_dtw_distance(traj_a, traj_b)`: Multidimensional DTW on continuous PCA trajectories of bouts.

### 4. Visualization Pipeline (`lib/visualization/`)
Generates figures:
* **`embedding_viz.py`**: Cosine similarity grids and UMAP embeddings.
* **`metrics_viz.py`**: Perplexity comparisons across cohorts and DTW alignment heatmaps.

### 5. CLI Entry Point (`run_lm_analysis.py`)
Exposes arguments for execution:
* `python run_lm_analysis.py --verify`
* `python run_lm_analysis.py --run-embeddings`
* `python run_lm_analysis.py --run-perplexity`
* `python run_lm_analysis.py --run-dtw`


---

## Verification Plan

### Automated Verification
* Write a unit test script checking that:
  - Sequence shuffling increases perplexity significantly.
  - BPE tokenization successfully reduces average sequence length by chunking.
  - DTW returns zero distance for identical PCA trajectories.

### Manual Verification
* Run the CLI arguments and verify that output plots and statistical tables are written correctly to `output/`.
