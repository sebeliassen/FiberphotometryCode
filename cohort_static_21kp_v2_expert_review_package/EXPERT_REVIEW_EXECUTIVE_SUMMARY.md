# Multi-View Static 21kp V2 Pipeline: Expert Review & Verification Dossier

**Project:** Dual-Camera Dorsal/Ventral Behavioral Keypoint-MoSeq (KPMS) & Fiber Photometry Alignment  
**Production Champion Model:** Full Cohort Static 21kp V2, Seed 101 (50 AR + 300 SLDS = 350 Iterations)  
**Target Audience:** Machine Learning Engineers, Computer Vision Scientists, Neuroethologists  
**Date:** October 2026  

---

## 1. Executive Summary & Production Status

This package provides external engineering and scientific reviewers with the complete technical dossier, mathematical formalization, core production codebase, visual assets, and verification test suite for the **Cohort Static 21kp V2 (Seed 101)** pipeline.

The pipeline combines dual high-speed video streams (TopCam dorsal view at ~60 Hz and BotCam ventral view through a transparent arena floor at ~60 Hz) with 21 synchronized keypoints (17 dorsal keypoints + 4 ventral paws) across 1,363 cohort recording sessions.

### Key Milestones Achieved
1. **Production Model Training**: Full 350-iteration SLDS convergence (50 AR warm-up + 300 SLDS iterations) on Seed 101, modeling 100 behavioral syllables across 8.9 GB of cohort keypoint coordinates.
2. **Rigorous 3-Tier Prototype Selection**: Solved the sample-size sparsity limitation of upstream KPMS density sampling ($N \le 50$) by establishing a mathematically sound 3-tier prototype selection engine.
3. **Native 60 FPS Dual-View Grid Movies**: 100% native real-time playback (60 FPS) with standard KPMS display duration (1.0 s prior, 2.0 s post = 180 frames @ 60 FPS) and strict tracking dropout exclusion.
4. **Hardened Visualization Formats**:
   - **Vector PDFs**: Embedded CID TrueType Type 42 fonts with Identity-H Unicode mappings for flawless cross-platform rendering in VS Code / macOS remote preview.
   - **Animated Trajectory GIFs**: Native PIL integer millisecond frame timings (100 ms per step, 400 ms terminal hold, zero loop desynchronization).
   - **High-Res PNGs**: 300 DPI publication figures.
5. **Interactive Ethologist Workstation**: Self-contained browser-based HTML gallery with live video scrubbing, synchronized trajectory views, client-side localStorage persistence, and JSON/CSV label export.

---

## 2. Mathematical Formalization & Architecture

### A. 21-Keypoint Static Fusion
* **Dorsal Stream (TopCam)**: 17 keypoints tracking head, snout, ears, spine, and tail base.
* **Ventral Stream (BotCam)**: 4 high-confidence ventral paws (front-left, front-right, back-left, back-right).
* **Static Arena Registration**: Instead of noisy dynamic affine fitting, each arena floor (Arena 1–4) uses calibrated, invariant geometric transformation matrices, guaranteeing coordinate stability without spatial drift.

### B. The 3-Tier Syllable Prototype Selection Architecture
Upstream Keypoint-MoSeq uses 4D PCA density ratio sampling, but breaks or fails on low-frequency syllables ($N \le 50$). We formulated and implemented a robust 3-tier prototype selector:

$$\text{Manifest: } \texttt{unified\_syllable\_selection\_manifest.json}$$

1. **Tier 1 ($N > 50$ bouts, 96 syllables)**: Built-in KPMS Density Ratio Sampling
   - Projects all bouts into 4D PCA trajectory space.
   - Computes local vs. global density ratio:
     $$\text{density\_ratio}(i) = \frac{\rho_{\text{local}}(i)}{\rho_{\text{global}}(i)} = \frac{\frac{1}{\bar{d}_{\text{local}}(i) + \epsilon}}{\frac{1}{\bar{d}_{\text{global}}(i) + \epsilon}}$$
   - The modal exemplar maximizing this ratio is placed in Tile 1, followed by its $G - 1$ nearest neighbors ordered by distance in 4D PCA space.
2. **Tier 2 ($12 < N \le 50$ bouts, 1 syllable: Syllable 74)**: Compact Neighborhood Selection
   - For every bout $i \in \{0 \dots N-1\}$, forms a candidate ensemble $S_i$ consisting of $i$ and its $G-1$ nearest neighbors.
   - Evaluates the mean pairwise trajectory distance:
     $$\text{Score}(S_i) = \frac{1}{\binom{G}{2}} \sum_{j < k \in S_i} \|y_j - y_k\|_2$$
   - Selects the candidate set $S^*$ with minimum internal dispersion.
   - Computes the medoid $m^* \in S^*$, places $m^*$ in Tile 1, and orders the remaining $G-1$ bouts by distance to $m^*$.
3. **Tier 3 ($N \le 12$ bouts, 2 syllables: Syllables 44 & 47)**: Complete Bout Enumeration
   - Displays all $N$ available bouts without omission.
   - Computes the cohort medoid, places it in Tile 1, orders remaining bouts by distance, and renders unused ($G - N$) slots as clean blank placeholders.

---

## 3. Visualization Pipeline & Display Quality Assurance

### A. Native 60 FPS Dual-View Grid Movies
* **Timing Parameters**: $\text{pre} = 60$ frames (1.0 s), $\text{post} = 120$ frames (2.0 s) $\rightarrow$ 180 frames total (3.0 s @ 60 FPS).
* **Visual Elements**:
  - TopCam and BotCam crops rendered side-by-side in tandem tiles.
  - White onset indicator dot active strictly during bout execution ($0 \le t - \text{pre} < \text{duration}$).
  - Clean frame tracking: Bouts with dorsal or ventral tracking dropouts during the 3-second display window are automatically substituted with clean bouts for that syllable.
  - **Tile 1 Exemplar Badge**: Tile 1 is prominently labeled with a green `EXEMPLAR` badge (Tier 1) or `MEDOID` badge (Tier 2/3). All 77 active movies pass 100% automated badge verification.

### B. Trajectory Vector PDFs & Animated GIFs
* **PDF Font Encoding**: Standard Matplotlib Type 3 bitmap fonts caused rendering failures in remote IDEs. We enforce Type 42 TrueType CID fonts (`matplotlib.rcParams["pdf.fonttype"] = 42`) with `Identity-H` Unicode mapping (`uni: yes`), ensuring clean rendering on macOS/Linux/Windows.
* **GIF Animation Timing**: PIL `save_gif` enforces integer millisecond duration parameters (`duration=100`, terminal hold `400` ms, `loop=0`). All 152 trajectory GIFs (side-by-side and superimposed) are verified animated (`is_animated: True`, `n_frames: 15`).

---

## 4. Package Directory Layout

```
cohort_static_21kp_v2_expert_review_package/
├── EXPERT_REVIEW_EXECUTIVE_SUMMARY.md       <-- You are here
├── run_all_tests.py                         <-- Self-contained regression test runner
├── docs/                                    <-- Complete technical documentation (22 files)
│   ├── SEED101_STATIC21KP_V2_SYLLABLE_CATALOG_FOR_LABELING.md
│   ├── PILOT_17KP_VS_21KP_FULL_COMPARISON_REPORT.md
│   ├── MULTI_VIEW_SYNCHRONIZATION_AND_VFI_ARCHITECTURE.md
│   ├── cohort_multiview_synchronization_deep_dive.md
│   ├── TANDEM_KPMS_MULTIVIEW_ALIGNMENT_AND_EMPIRICAL_BENCHMARK.md
│   ├── REFACTORING_AND_SWE_ARCHITECTURE_PLAN.md
│   └── ...
├── core/                                    <-- Core pipeline modules & synchronizers
│   ├── camera_correspondence.py
│   ├── static_fusion_engine.py
│   ├── cache_manifest.py
│   ├── eval_artifacts.py
│   ├── schemas.py
│   └── ttl_cohort_offsets.json
├── pipeline/
│   ├── 01_data_and_fusion/                  <-- Data alignment & fusion scripts
│   ├── 02_training_runners/                 <-- Production SLDS training runners
│   ├── 03_evaluation_and_eml/               <-- Likelihood evaluation & cross-validation
│   └── 04_reports_and_figures/              <-- Visualization rendering & catalog scripts
│       ├── render_all_cohort_static_21kp_v2_seed101_grid_movies.py
│       ├── render_all_cohort_static_21kp_v2_seed101_trajectories.py
│       ├── select_cohort_static_21kp_v2_syllable_manifest.py
│       └── build_cohort_static_21kp_v2_seed101_labeling_gallery.py
├── tests/                                   <-- Automated test suite (6 test files)
└── results_and_figures/
    ├── config.yml                           <-- Cohort V2 production model configuration
    ├── unified_syllable_selection_manifest.json  <-- 3-Tier prototype manifest (1.5 MB)
    ├── syllable_labeling_gallery_cohort_static_21kp_v2_seed101.html  <-- Labeling gallery (370 KB)
    ├── SEED101_STATIC21KP_V2_SYLLABLE_CATALOG_FOR_LABELING.md
    └── visualizations/
        ├── grid_movies_density/             <-- All 77 active dual-view grid movies (48 MB)
        └── sample_trajectories/             <-- Representative GIFs, PDFs, PNGs across Tiers 1-3
```

---

## 5. Verification Checklist & Instructions

### Running Regression Tests
Run the standalone test suite on CPU in ~1.5 seconds:
```bash
python run_all_tests.py
```

### Inspecting Grid Movies & Trajectories
1. Open `results_and_figures/syllable_labeling_gallery_cohort_static_21kp_v2_seed101.html` directly in any web browser.
2. Review `results_and_figures/visualizations/grid_movies_density/` for full-length 60 FPS tandem videos.
3. Review `results_and_figures/visualizations/sample_trajectories/` to verify animated GIFs and vector PDFs.
