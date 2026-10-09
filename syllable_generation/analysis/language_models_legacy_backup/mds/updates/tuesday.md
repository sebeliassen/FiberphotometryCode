# Behavioral Strategy Modeling Pipeline - Status Update (Tuesday, June 24 2026)

This document records the changes made on Tuesday 24 June 2026, continuing from the Monday checkpoint.

---

## 1. Problem Identified: Cluster Imbalance in Rich Token Vocabulary

When inspecting the sub-state PCA diagnostic plots generated last week (e.g. `syllable_53_substates.png`, `syllable_41_substates.png`), it was discovered that K-Means was producing **severely imbalanced cluster sizes** across the majority of syllables.

Concretely, on the baseline (StandardScaler only):
- **60 out of 95 syllables** had a smallest cluster containing < 5% of bouts.
- In extreme cases (e.g. Syllable 53), the third substate consisted of a **single outlier bout** visually isolated at >15 standard deviations from the main cloud.
- The max/min cluster size ratio averaged **117×** across all syllables.

This meant that roughly one-third of the generated rich tokens were **ghost tokens** — they existed in the vocabulary but appeared so rarely in any session's token stream that LDA and downstream topic models could not learn any meaningful co-occurrence patterns from them.

### Root Cause: Heavy-Tailed Feature Distributions

K-Means minimizes Euclidean distance to centroids, implicitly assuming roughly Gaussian feature distributions. The kinematic features are anything but:

| Feature | Raw Skewness | Kurtosis |
|---|---|---|
| Duration (frames) | **42.9** | 4544 |
| Mean Angular Velocity | **18.2** | 567 |
| Mean Velocity | 1.95 | 4.6 |
| Mean Acceleration | 1.44 | 5.5 |
| Mean Dist. from Center | −0.40 | 1.0 |

After `StandardScaler`, these extreme tails persist in standardised units. K-Means would plant two centroids on the main bulk of normal bouts, and the third centroid would get **dragged to chase the sparse tail of extreme outlier bouts**, leaving it with 1–30 members out of thousands. The PCA plots showed this clearly as a yellow singleton floating 15+ units away from the two main blobs.

---

## 2. Fix Applied: log1p Pre-Transform

A `log1p` transform is now applied to the four skewed features **before** `StandardScaler` in `KinematicSyllableClustered.fit()` and `.transform_session_to_bouts()`:

```python
_LOG1P_COLS = [0, 1, 2, 3]  # duration_frames, mean_velocity, mean_angular_velocity, mean_acceleration
# mean_dist_from_center (col 4) is left unchanged — already near-Gaussian (skew = −0.40)

X[:, _LOG1P_COLS] = np.log1p(X[:, _LOG1P_COLS])
# then StandardScaler as before
```

`log1p(x) = log(1 + x)` handles zeros safely and compresses the long right tail of rate-based physical quantities. After the transform:

| Feature | Skewness after log1p |
|---|---|
| Duration (frames) | **−0.05** (near-perfect Gaussian) |
| Mean Velocity | 0.48 |
| Mean Angular Velocity | 2.68 (still right-skewed; Acceleration's log is bimodal) |
| Mean Acceleration | −1.15 |

K-Means now sees distributions that are much closer to what it was designed for.

---

## 3. Empirical Results (95 syllables, 79,604 bouts total, AMPH-pre + IFDN)

| Metric | Baseline | log1p | Direction |
|---|---|---|---|
| Smallest cluster fraction (mean) | 0.046 | **0.116** | ↑ better |
| Max/min cluster ratio (mean) | 117× | **8.6×** | ↓ better |
| Normalised entropy (mean) | 0.646 | **0.811** | ↑ better |
| Gini coefficient (mean) | 0.405 | **0.307** | ↓ better |
| Severely imbalanced (<5% min) | 60/95 | **18/95** | ↓ better |

All improvements are statistically significant (Wilcoxon signed-rank, p < 0.001). Silhouette score drops slightly (0.317 → 0.272, p = 0.0002) — this is expected and healthy: the baseline's inflated silhouette was driven by the artificially isolated tiny cluster, not by genuinely better-separated structure.

### Spotlight Syllable Examples

- **Syllable 4:** C2 went from 29 bouts (0.7%) → 1137 bouts (30%). Three genuinely populated regions now visible in PCA.
- **Syllable 41:** C2 went from 27 bouts (1.4%) → 65 bouts (3.2%), and C1 from 379 → 832. The two large clusters are now much more balanced.
- **Syllable 53:** C2 went from 1 single outlier → 251 bouts (6.8%), forming a proper distinct upper-right cluster (high velocity/angular velocity bouts).

---

## 4. Files Changed

- **[clustering.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/fusion/clustering.py):** Added `_LOG1P_COLS` module-level constant and the log1p transform in both `fit()` and `transform_session_to_bouts()`. This is the only change needed — the rest of the pipeline (segmentation, LDA, strategy mapping) is unaffected.
- **All 95 `syllable_X_substates.png` plots** regenerated in `output/rich_tokens/`.
- Diagnostic scripts added:
  - `run_clustering_diagnostics.py` — full statistical + visual comparison of baseline vs log1p.
  - `run_replot_substates.py` — regenerates the per-syllable substate PCA plots.

---

## 5. Next Steps

- Proceed with Phase 3 (Biological Go/No-Go Gate):
  - Dopamine PETH alignment at strategy transition boundaries.
  - Pharmacological entropy collapse check (AMPH vs Saline strategy distributions).
- The improved vocabulary (fewer ghost tokens, more balanced substates) should make LDA topic coherence stronger and make DA validation more sensitive.
