# Scientific Audit Addendum: Anchor Confidence Gating, Continuous Synchronization & Trajectory Dynamics

**Date:** October 1, 2026  
**Cohort:** 118 Sessions Multi-Camera Tandem Cohort (Champion Seed 101)  
**Package:** `tandem_kpms_researcher_audit_and_code_bundle.zip`  
**Regression Test Suite:** `tests/test_multiview_sync_and_renderer_regression.py` (100% Pass)  

---

## 1. Executive Summary & Root Causes

Following exhaustive inspection of the saved Seed 101 training arrays (`data["Y"]`, `data["conf"]`, `data["mask"]`), Video Frame Indexes (VFI), and hardware master clock registrations across all 118 cohort sessions (18,394,117 retained model frames), five critical architectural insights have been established:

1. **Anchor Confidence Bleed Through Egocentric Transformation:**  
   In the legacy fusion engine, egocentric centering and rotation matrices were computed directly from raw BotCam `nose_tip` and `tail_base` tracking without confidence filtering ( < 0.8$) or anatomical body-length gating (5 \le L_{	ext{bot}} \le 350	ext{ px}$). When the BotCam nose tracking snapped to an arena corner screw ($,  = 0.49$) for 22 consecutive frames in January, the transformation coordinate frame rotated and translated by hundreds of pixels. Crucially, **the paws retained their original high DLC confidence ( pprox 0.98$)**, projecting physically impossible paw coordinates into KPMS training arrays with high model certainty. While extreme outliers ($> 6	imes$ medoid distance) were flagged by KPMS preprocessing, moderate anchor distortions (5	ext{--}50	ext{ px}$) entered training directly.

2. **The 78.35% Timing Desynchronization in Training Inputs:**  
   The initial audit noted only "131 dropped frames (0.001%)" because it counted only literal missing frame timestamps. However, because training data fusion used a **single static integer offset per session** (`bot_idx = top_idx - offset`), any early hardware dropped frame or Basler crystal drift accumulated into a permanent post-gap phase shift:
   * Across the cohort's 18,394,117 retained training frames, **14,411,067 frames (78.35%)** were trained with a $\ge 1himBHsframe offset.
   * **3,725,444 frames (20.25%)** had a **$> 5himBHsframe offset ($> 83	ext{--}283	ext{ ms}$ lag)**.
   * In 25 out of 113 sessions, the maximum frame error exceeded 5 frames (reaching up to 17 frames / 283 ms). In the December session (`2025-12-03T16_38_48`), a 6-frame drop at frame 49,419 caused **141,404 subsequent frames (86.23% of that session)** to be trained with a 100 ms phase lag between dorsal spine curvature and ventral paw strike timing.

3. **Syllable 45 Is an Empirical Phantom State (Not Biology):**  
   Audit of all 74,072 frames of Syllable 45 in Champion Seed 101 results revealed that **100.0% of Syllable 45 frames** occur exclusively in session `2026-03-15T09_51_36` after BotCam acquisition aborted at frame 73,244 while TopCam ran to 200,922. Over 127,000 frames were padded with NaNs (imputed by KPMS with `conf = 0.001`). The model allocated a discrete state (state 45) to describe the state of having missing ventral observations. Syllable 45 must not be interpreted biologically.

4. **Relationship to Trajectory Visualizations:**  
   `render_seed101_trajectories.py` plots $\hat{Y} = \mathbf{C}_d x_t + v_t$, the model's learned generative reconstruction from its 10-dimensional latent state space, rather than raw camera frames. The 10D bottleneck acts as a low-rank filter that produces smooth visual plots, but locomotive gaits in the trajectory plots reflect paw kinematics that were artificially phase-lagged by up to 250 ms during training.

5. **Syllable 30 Random Row 1, Col 2 Empty Crop Resolved:**  
   In `syllable_30_tandem_grid_random.mp4`, DLC ventral tracking collapsed ( = 1.37	ext{ px}$,  = 0.491$) for $> 10$ frames. The legacy renderer fell back to the arena center $, cropping empty glass floor, and placed an unverified centroid dot in the center. The renderer now strictly validates pose verification (`is_v_onset == False` $	o$ dark tile annotated `VENTRAL / TRACKING LOST`) and suppresses all centroid dots when tracking is unverified.

---

## 2. Quantitative Evidence & Slices

### A. January Anchor Glitch Slice (`slice_january_screw_confidence_gating.csv`)
Comparison of Left Front Paw projection during corner-screw tracking failure:

| Video Frame | TopCam Nose $ | Old Fused LF Paw $ | Old Conf | New Fused LF Paw $ | New Conf | Status / Resolution |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **94,727** | $ | $ | bash.666$ | $ | $\mathbf{0.000}$ | Corrupted anchor rejected |
| **94,728** | $ | $ | $\mathbf{0.687}$ | $ | $\mathbf{0.000}$ | Corner screw artifact masked |
| **94,729** | $ | $ | bash.734$ | $ | $\mathbf{0.000}$ | Corner screw artifact masked |
| **94,730** | $ | $ | bash.728$ | $ | $\mathbf{0.490}$ | Mouse body recovered, anchor-gated |
| **94,731** | $ | $ | bash.824$ | $ | $\mathbf{0.488}$ | Anatomical pose restored ( pprox 301$) |

### B. March Terminal Abort Truncation (`slice_march_syllable45_truncation_summary.csv`)
Elimination of unfilmed NaN padding in session `2026-03-15T09_51_36_bottom_left`:

* **Original Fused Frames:** 00,922$ frames
* **Original Missing/NaN Ventral Frames:** 27,689$ frames (3.55\%$)
* **Continuous Calibrated Fused Frames:** 3,249$ frames (truncated at BotCam acquisition cutoff)
* **Eliminated Phantom Frames:** **27,673$ unfilmed frames completely removed** from training input.

---

## 3. Package Structure

```
tandem_kpms_researcher_audit_and_code_bundle/
├── README.md                                                     (Complete execution & reproduction guide)
├── MULTIVIEW_SYNCHRONIZATION_AND_VISUALIZATION_AUDIT_REPORT.md  (Original forensic report)
├── CONFIDENCE_GATING_AND_CONTINUOUS_FUSION_ADDENDUM.md          (This scientific addendum)
├── core/
│   ├── fusion_engine.py          (Continuous rate-aware Basler mapping + anchor confidence gating)
│   ├── camera_correspondence.py  (Basler crystal regression & RANSAC anchor outlier rejection)
│   ├── ttl_sync.py               (FP3002 master clock hardware parser)
│   ├── schemas.py                (Anatomical constants, keypoint maps, and Chamber-Arena tables)
│   ├── pair_matching.py          (Tandem session matcher)
│   ├── io_utils.py               (Atomic safe I/O)
│   ├── logging_utils.py          (Structured audit logger)
│   └── evaluation_metrics.py     (Residual calculations)
├── scripts/
│   ├── render_all_seed101_grid_movies.py (Renderer with frozen manifests & tracking-lost protection)
│   ├── render_seed101_trajectories.py    (Trajectory renderer with density sampling)
│   ├── run_pilot_continuous_calibrated_retrain.py (Benchmark pilot training runner)
│   ├── deep_trace_and_export.py          (Independent per-frame forensic decoder)
│   ├── audit_cohort_continuous_sync.py   (Cohort continuous synchronization auditor)
│   └── audit_cohort_basler_calibrated.py (Basler crystal clock calibrator)
├── tests/
│   └── test_multiview_sync_and_renderer_regression.py (Automated regression suite: 100% Pass)
├── frozen_manifests/
│   ├── frozen_bouts_seed101_syll00_density.json
│   ├── frozen_bouts_seed101_syll30_random.json
│   └── frozen_bouts_seed101_syll45_density.json
├── audit_csvs/
│   ├── slice_january_screw_confidence_gating.csv
│   ├── slice_march_syllable45_truncation_summary.csv
│   ├── cohort_continuous_sync_audit_results.csv
│   ├── cohort_basler_calibrated_audit_results.csv
│   ├── cohort_alignment_audit.csv
│   └── phase2_training_inputs_audit.csv
└── diagnostic_proofs/
    ├── test_syll30_random_frame25_fixed.png (Proof of empty crop & dot suppression fix)
    ├── debug_syll30_r1_c2_fixed.png        (Zoomed tile proof: VENTRAL TRACKING LOST)
    ├── test_syll45_frame25_fixed.png       (Proof of VENTRAL ACQ ENDED annotation)
    ├── test_syll00_frame25.png             (Verified density render)
    ├── fresh_panel_7_r2_c4.png             (Verified panel 7 crop)
    └── fresh_panel_10_r3_c3.png            (Verified panel 10 sync)
```

---

## 4. Verification One-Liners for Researchers

* **Run the Regression Test Suite (~3 seconds):**
  ```bash
  conda run -n kpms_gpu_cuml python tests/test_multiview_sync_and_renderer_regression.py
  ```
* **Run Benchmark Pilot Fusion (10 sessions):**
  ```bash
  conda run -n kpms_gpu_cuml python scripts/run_pilot_continuous_calibrated_retrain.py --fuse-only
  ```
* **Render Syllable 30 (Verified with Tracking Lost and Dot Suppression):**
  ```bash
  conda run -n kpms_gpu_cuml python scripts/render_all_seed101_grid_movies.py --mode random --overwrite --syllables 30
  ```
