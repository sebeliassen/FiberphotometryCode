# Pipeline Directory: 04_predictive_frontier

This folder contains all the self-contained executable pipelines, benchmarking scripts, and diagnostic tools for the **Finite-Data Empirical Predictive Frontier** and **Conditional Future Uncertainty** analyses.

---

## Script Inventory & Execution Guide

All scripts are configured to run in the GPU environment:
```bash
/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python3
```

| Script Name | Purpose & Analysis Description | Primary Output Data / Figures |
| :--- | :--- | :--- |
| **`run_full_empirical_frontier_diagnostics.py`** | **Master Pipeline:** Runs Experiments 1–6 (Stacking, Oracle Selector, Residuals, Data Scaling, Branch Oracle, Future Geometry) with 5-fold CV across all 30 sessions. | `master_empirical_frontier_table.csv`, `fig11_comprehensive_empirical_frontier_suite.png` |
| **`run_empirical_frontier_suite.py`** | Integrated fast benchmark suite comparing base models, stacked ensembles, and oracle bounds. | `empirical_frontier_suite_results.csv` |
| **`run_past_future_distance_transfer.py`** | Evaluates continuous distance transfer $\mathbb{E}[d_Y \mid d_X]$ across 100,000 cross-session pairs and flat shuffled control. | `past_future_transfer_curve.csv`, `fig9_past_future_distance_transfer.png` |
| **`run_density_scaling_sweep.py`** | Fixed-$k$ density scaling ($N=10\% \to 100\%$) demonstrating $\bar{d}_k \downarrow$ and risk improvement. | `density_scaling_sweep_results.csv`, `fig7_density_scaling_asymptote.png` |
| **`run_joint_bias_variance_surface.py`** | Evaluates 203 empirical points on $(d, k)$ grid and fits bias-variance surfaces with bootstrap CI for $R_\infty$. | `joint_surface_asymptotic_estimates.csv`, `fig8_joint_bias_variance_floor.png` |
| **`run_conditional_future_dispersion.py`** | Neighborhood dispersion analysis measuring future fanning out from similar pasts across mice. | `conditional_future_dispersion_results.csv` |
| **`test_shuffled_vs_ordered_reconciliation.py`** | Head-to-head 5-fold CV reconciling Order-Agnostic Bag Transformer vs Ordered Transformer vs Test-time scrambling. | Summary terminal log output |
| **`test_complete_pn_ladder_audit.py`** | Full $\mathcal{P}_0 \to \mathcal{P}_{6\text{b}}$ Information Ladder audit on clean discrete bout onsets. | Detailed $\mathcal{P}_n$ gain metrics |
| **`test_counts_vs_dwell.py`** | Proves syllable count frequency is redundant with total duration dwell ($\mathcal{P}_4 \text{ vs. } \mathcal{P}_{4.5}$). | Comparative divergence delta |
| **`test_continuous_pose_vs_syllable.py`** | Compares continuous 30-D pose PCs vs discrete syllable tokens ($\mathcal{P}_{2.5} \text{ vs. } \mathcal{P}_3$). | Modality comparison table |
| **`test_exclude_syll0_pn_ladder.py`** | Audits the effect of removing the Syllable 0 inter-bout gap artifact from rasterized data. | Corrected baseline null |
| **`test_audit_transformer_clean_bouts.py`** | Validates Transformer sequence tokenization directly from raw Keypoint-MoSeq bout dictionaries. | Clean token tensors |
| **`run_sequence_benchmark_suite.py`** | Comprehensive sequence model benchmark (BoW, Action2Vec, VOMM, HSMM, Mamba-1, Transformer). | `sequence_benchmark_results.csv` |
| **`run_predictor_hierarchy_benchmark.py`** | Hierarchy benchmark across 6 prediction tiers. | `predictor_hierarchy_results.csv` |
| **`test_discrete_strategy_bottleneck.py`** | Tests discrete vs continuous strategy representation bottlenecks. | `discrete_bottleneck_comparison.csv` |
