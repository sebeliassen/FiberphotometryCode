# Behavioral Strategy Modeling Framework (`language_models`)

This directory houses the **Macro-Behavioral Strategy Modeling Pipeline** for discovering interpretable behavioral strategies ("vibes") from MoSeq syllables and aligning them with fiber photometry dopamine ($\Delta F/F$) dynamics.

---

## 📁 Architecture & Pipeline Roadmap

```
syllable_generation/analysis/language_models/
├── pipelines/
│   ├── 01_horizon_dynamics/             # Chapter 01: Multi-Horizon Forecasts & Past Memory Sweep (W* = 68.3s)
│   │   ├── run_multi_horizon_benchmark.py
│   │   └── run_sliding_window_sweep.py
│   ├── 02_frame_vs_bout/                # Chapter 02: Frame vs. Bout Representation & Duration Augmentation
│   │   ├── run_frame_vs_bout_benchmark.py
│   │   ├── test_2x2_cross_prediction.py
│   │   ├── audit_frame_vs_bout_mechanisms.py
│   │   └── test_duration_augmented_rle.py
│   ├── 03_syntax_and_hierarchy/         # Chapter 03: Syllable Transition Sparsity & Grammar Hierarchy
│   └── 04_photometry_alignment/         # Chapter 04: Photometry & Dopamine Strategy Alignment
├── output/                              # Exported CSV tables, model checkpoints, and publication figures
└── mds/01_theory_and_strategy/          # Methodological guides, theory, and mathematical proofs
    ├── sliding_window_optimization.md                     # Chapter 01 Theory & W* = 68.3s derivation
    ├── methodological_guide_bout_vs_frame_evaluation.md   # Chapter 02 Methodological SOP & Audit
    └── behavioral_strategy_modeling.md                    # Core Macro-Strategy Reference Manual
```

---

## 🚀 Key Results Summary

### Chapter 01: Horizon & Memory Dynamics
* **Optimal Memory Horizon:** $W^* = \mathbf{68.3\text{ seconds}}$ ($\approx 1.14\text{ minutes}$), derived by fitting a biexponential Bateman rise-and-decay function ($R^2 = 0.9707$).
* **Strategy Scalability:** Strategy $\mathcal{P}_4$ scales with horizon length ($H$), capturing **$86.1\%$ of maximal empirical predictability** at $H=120\text{s}$.

### Chapter 02: Frame-Level vs. Bout-Level Evaluation
* **Markovian Overlap Artifact:** Proved that on 80.025 Hz raw frames, 1-step Markov baseline is artificially inflated to $+48.59\%$ by intra-syllable continuation.
* **Discrete RLE Bouts:** Eliminates the overlap artifact; Markov drops to its physical floor ($+12.47\% \to +15.32\%$), allowing Strategy to dominate at $1.67\times$ over Markov.
* **Duration-Augmented RLE:** Concatenating discrete bout counts with total dwell times $[\mathbf{u}_{\text{count}}, \mathbf{u}_{\text{time}}] \in \mathbb{R}^{200}$ achieves the overall highest predictive gain ($\mathbf{+31.88\%}$).

---

## 🐍 Python Environment
Always use the exact project binary:
```bash
/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python <script_path.py>
```
