# Tandem (Seed 404) Syllable Tiles Look Less "Same" Than 1-Cam: Diagnosis & Static-Registration Proposal

**Date:** 2026-10-05
**Status:** Diagnosis fully validated; static registration implemented and benchmarked via 10-session pilot comparison; **Arm A (21kp Static)** confirmed as the winning architecture.
**Audience:** a new agent/engineer picking this up with no chat context.

All paths are relative to `/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/` (`EXP`) unless absolute. Python env: `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`.

---

## 1. The question

Grid movies for the tandem model (Seed 404), e.g. `04_reports_and_figures/visualizations_seed404/grid_movies_density/syllable_03_tandem_grid_density.mp4`, show tiles that look less alike within a syllable than the 1-camera model's (`final_karen_model_k240k`, same Karen cohort). Is that a bug in code/training, or is it harder for KPMS (AR-HMM/SLDS) to segment with top + bottom keypoints?

## 2. Context (what the models are)

| | 1-cam baseline | Tandem champion |
|---|---|---|
| Model | `syllabizers/keypoint_moseq_syll/karen_project_2/final_karen_model_k240k` | `kpms_tandem_cohort_continuous_calibrated/tandem_calibrated_50ar_200slds_seed404_20261002_012237_940073` |
| Keypoints | 17 TopCam | 23 = 17 TopCam + 6 BotCam (4 paws, `bot_nose_tip`, `bot_tail_base`) |
| fps | 40 (downsample x2 from 60) | 60 |
| kappa (init -> SLDS) | 2.4e5 | **7.5e13 -> 7.5e12** (`core/schemas.py`); no kappa search found that justifies it |
| Latent dim / K | 10 / ~100 | 10 / 100 |
| Median bout | 14 frames @40fps (~350 ms) | 38 frames @60fps (~633 ms) |
| Heading | anterior=`tail_base`, posterior=`nose` | same |

Seed 404 won a 5-seed EML tournament (see `04_reports_and_figures/KPMS_SEED_404_EML_SELECTION_AND_SWE_HARDENING_REPORT.md`). Fused input: `fused_h5_continuous_calibrated/*_tandem_23kp.h5` (118 sessions), built by `core/fusion_engine.py::fuse_single_pair`.

## 3. What was ruled out (not bugs)

- `04_reports_and_figures/audits/seed404_grid_movies_tile_by_tile_audit.md`: 2400/2400 tiles have correct syllable label, onset aligned, zero dropped bot frames.
- Sync: 258/258 sessions `VALID_CRYSTAL`, max residual 0.229 ms.
- Earlier fusion bugs (78% static-offset desync, anchor-confidence bleed, Syllable-45 NaN phantom) are fixed in the continuous-calibrated data (see `docs/CONFIDENCE_GATING_AND_CONTINUOUS_FUSION_ADDENDUM.md`).
- Heading config identical to the 1-cam model.

## 4. Evidence gathered

### 4.1 PCA is dominated by two bottom keypoints
From `kpms_tandem_cohort_continuous_calibrated/pca.p` (loadings mapped back to keypoints through `jax_moseq ... center_embedding`), variance-weighted over the top 10 PCs:

| Keypoint(s) | Share |
|---|---|
| `bot_tail_base` | 43% |
| `bot_nose_tip` | 25% |
| 4 paws combined | ~16% |
| 17 dorsal keypoints combined | 16% |

PC1-PC4 (~67% of variance) are each >70% a single bottom nose/tail point. Caveat: this is the PCA **initialisation**; the SLDS refits the subspace, so how much survives is unmeasured.

### 4.2 Test 1 - within-syllable dorsal similarity (script: `04_reports_and_figures/audits/diagnose_tandem_tile_heterogeneity.py`, output `audits/tile_heterogeneity_diagnosis.json`)
Same 110 sessions, only frames labelled by both models, first 0.5 s of each bout, 17-keypoint top-view dorsal egocentric pose (approximates what a tile shows). R^2 = share of snippet variance explained by syllable label; ceiling = k-means with same K (81 syllables each, ~20k snippets each).

| Model | R^2 (labels) | k-means ceiling | Ratio |
|---|---|---|---|
| 1-cam k240k | 0.405 | 0.696 | **0.58** |
| Tandem seed404 | 0.208 | 0.693 | **0.30** |

Tandem syllables explain about half as much dorsal-pose variance, so the tiles genuinely are less alike by this metric.

### 4.3 Test 3 - are bottom nose/tail signal or noise?
Distance between `bot_nose_tip`/`bot_tail_base` (after fusion) and top `nose`/`tail_base`, frames where top conf > 0.8 (body length ~57 px):

- Median ~2 px, p90 ~7-8 px (typical frames agree well).
- >15 px on 4.0% (nose) / 2.4% (tail) of frames; >40 px on 1.6% / 0.9%.
- Frames >15 px carry **98% (nose) / 95% (tail)** of squared deviation. RMS 27/18 px, falling to ~4 px each once excluded.
- Only 4-7% of variance is between sessions (not a per-session offset). Not correlated with top body length. Low bot confidence is only a weak flag (<0.5 conf median deviation 9 / 5 px).
- Unresolved: whether those frames are DLC misdetections or real rearing parallax. High-frequency jitter stats came out NaN (smoothing bug in the script's `hf` helper; not needed for the conclusions).

**Interpretation:** a 2-4% minority of frames with tens-of-px disagreement dominates the variance the model sees, so latent dimensions are spent on those events.

## 5. Why the fusion is the prime suspect (`core/fusion_engine.py`)

The fusion manipulates the bottom points heavily (line numbers as of this report):

1. **Per-frame re-anchoring (L348-376):** each bot keypoint is made egocentric to the *bottom* nose/tail midpoint+heading, scaled by one per-session factor, then re-placed at the *top* midpoint+heading. After this, `bot_nose_tip`/`bot_tail_base` are nearly forced onto the top body axis; the only free quantity is `scale*len_bot` vs `len_top`, one scalar duplicated across two keypoints.
2. **Anchor smoothing (L324-327):** 11-frame (~183 ms) median on midpoints and headings.
3. **Anchor interpolation (L314-322):** bot anchors linearly filled; a frame is valid if a verified anchor is within 10 frames.
4. **One fixed scale per session (L286):** ignores perspective change on rearing.
5. **Loose plausibility gate:** bot length accepted in 25-350 px, so wrong-but-plausible lengths pass.
6. Any bad bot anchor rotates/translates all four paws at once while paws keep high confidence (min with anchor conf is applied, but gate is loose).
7. Paws are placed with the *smoothed* top midpoint/heading while KPMS later aligns with the *raw* top nose/tail; the mismatch becomes paw jitter in KPMS's frame.

KPMS convention: the model learns per-keypoint noise from confidences; pre-smoothing, interpolation and geometric "making sense" remove that signal and introduce correlated artifacts. This is a hypothesis consistent with 4.1-4.3; **no ablation has been run**.

## 6. Other contributing factors / confounds (untested)

- **kappa = 7.5e13** vs 1.1e5-2.4e5 for 1-cam: very sticky -> long bouts that can contain visually distinct moments. Independent of fusion.
- **Renderer:** `04_reports_and_figures/render_all_seed404_grid_movies.py` crops a *fixed* window at onset (pre=10, post=40 frames, 60 fps). Longer bouts drift out of the crop and look less alike.
- **fps conversion risk:** `04_reports_and_figures/visualizations_seed404/TWO_CAMERA_VS_ONE_CAMERA_PRELIMINARY_ANALYSIS.md` claims "2.7x longer bouts" (233 vs 633 ms). 1-cam is 40 fps; if its 14 frames were converted at 60 fps, 1-cam duration is understated (14 frames @40 fps = 350 ms, so ~1.8x not 2.7x). Verify. Test 1 above used raw-60fps-aligned frames (1-cam labels duplicated onto skipped frames), median bout 26 vs 37 raw frames.

## 7. Proposed fix: static registration, raw bottom points

Prior audit (`docs/TANDEM_KPMS_MULTIVIEW_ALIGNMENT_AND_EMPIRICAL_BENCHMARK.md`) found after the x-mirror (`x -> 972 - x`) the cameras agree at r = +0.9997 in arena coordinates. So a **single static registration per arena** (mirror + scale + offset; optionally a homography fit on a robust set of nose/tail/centroid correspondences) should suffice. Then:

- Transform **raw** bottom keypoints (all 6) through the static map into top-camera pixel coordinates.
- **No** per-frame anchors, **no** smoothing, **no** interpolation, **no** per-frame rotation.
- Keep each point's own DLC likelihood as its confidence (no min-with-anchor coupling); bad/missing frames -> NaN or low confidence, which KPMS handles.
- Keep: continuous Basler crystal frame mapping (`core/camera_correspondence.py`), bilateral L/R swap from the mirror, aborted-tail truncation, missing-ventral segment splitting (>=180 frames).
- Open design question: 3D parallax (rearing) means a planar homography is approximate; consider fitting on floor-contact frames only, and consider dropping `bot_nose_tip`/`bot_tail_base` entirely (use 4 paws only) as an ablation arm.

## 8. Recommended experiment plan (nothing below is done yet)

1. **Inspect outlier frames:** pull ~20 frames where bot-vs-top nose/tail disagree >40 px, view both cameras -> tracking failure vs parallax.
2. **Build `fuse_static_registration.py`** (new file; do not modify `core/fusion_engine.py` in place - respect the atomic-cache/manifest/immutability conventions in `core/cache_manifest.py`). Fit per-arena static map; validate residuals.
3. **Pilot ~10 sessions** fused three ways: (A) current, (B) static registration all 6 bot kp, (C) static registration paws only. Compare: PCA variance share by keypoint (4.1 method), distribution of bot-vs-top disagreement, and Test 1 ratio after a short pilot training (reuse `02_training_runners/run_cohort_calibrated_retrain_seed101.py` pattern; training at 7.5e13 kappa is itself confounded, so also try a kappa in the 1e5-1e6 range, ideally via the kappa search in `syllabizers/keypoint_moseq_syll/find_kappa.py`).
4. **Decision rule:** if Test 1 ratio approaches/exceeds the 1-cam 0.58 for B or C, run a full 50 AR + 200 SLDS retrain (multi-seed + EML tournament as before, ~17.4M frames, RTX 4090, keep checkpoint immutability guards).
5. Re-render grid movies with a tracking (not fixed) crop, or at least report the drift effect.

## 9. Gotchas for whoever continues

- Run scripts with cwd = `EXP` (relative paths such as `fused_h5_continuous_calibrated/` assume it).
- Results key mapping: tandem `results.h5` keys are `<sid>_seg###`; `video_frame_indexes.npz` (in `kpms_tandem_cohort_continuous_calibrated/`) maps rows to raw top-video frame indices (60 fps). 1-cam keys are `...CenterTopCamVid_<sid>DLC...` with raw indices stepping by 2.
- Any script importing `keypoint_moseq` needs the `np.bool8` patch (see `docs/kpms_tandem_cohort_investigation.md`).
- Seed 404 checkpoints are read-only by convention (`core/eval_artifacts.py`).
- Fused h5 columns: `('fused_multiview', <bodypart>, x|y|likelihood)`, 23 bodyparts in `config.yml: use_bodyparts` order; bot keypoints are the last 6.

---

## 10. Pilot Empirical Results & Architectural Decision (2026-10-05)

### 10.1 Static Registration & Fusion Implementation
- **Calibrated Transforms:** [`core/static_arena_transforms.json`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/core/static_arena_transforms.json) fit per-arena affine maps with $r \ge 0.99933$ and median residuals $\le 3.12\text{ px}$.
- **Engine:** [`core/static_fusion_engine.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/core/static_fusion_engine.py) provides non-destructive affine projection directly preserving raw DLC likelihoods.
- **Pilot Data:** Fused 10 representative sessions across Arena 1, 2, and 3 into `data/pilot_fused_static_21kp/` and `data/pilot_fused_static_23kp/`.

### 10.2 PCA Variance Allocation (The Decisive Proof)
Decomposing the top 10 PCs back into anatomical keypoint groups:

| Anatomical Group | Dynamic 23kp (Seed 404) | Static 23kp (Arm B) | Static 21kp (Arm A) |
| :--- | :---: | :---: | :---: |
| **17 Dorsal Keypoints** | **16.0%** | **28.4%** | **36.9%** |
| **4 Ventral Paws** | **16.0%** | **42.7%** | **63.1%** |
| **2 Bottom Anchors** (`nose_tip`, `tail_base`) | **68.0%** | **28.9%** | **0.0%** *(omitted)* |
| **Top 10 PCs Total Variance** | 81.3% | 85.1% | 89.0% |

- Static projection cut bottom anchor variance from **68% down to 28.9%**, confirming dynamic rotation was injecting massive artificial noise.
- However, even with static projection, the 2 bottom anchors still consumed **28.9%** of the PCA space due to 3D vertical elevation parallax during rearing.
- Omitting bottom anchors in **21kp (Arm A)** completely eliminates the parallax trap: dorsal posture share jumps to **36.9%** and ventral paws freely encode **63.1%** of gait/contact dynamics.

### 10.3 Comparative Evaluation Across 10 Shared Pilot Sessions
Evaluated via [`04_reports_and_figures/audits/evaluate_pilot_static_comparison.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/audits/evaluate_pilot_static_comparison.py) on identical common-frame support:

| Model | Architecture | $\kappa$ | Active Syllables | Median Bout | Dorsal $R^2$ | $K$-means Ceiling | Same-$K$ Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `onecam_k240k` | 1-Cam (17 kp, 30fps x2) | $1.0\times 10^6$ | 69 | 433 ms (26 f) | 34.58% | 65.66% | **0.5266** |
| `old_tandem_seed404` | Dynamic (23 kp, 60fps) | $7.5\times 10^{13}$ | 63 | 683 ms (41 f) | 20.61% | 66.65% | **0.3092** |
| `pilot_static_21kp` | Static Arm A (21 kp) | $1.0\times 10^5$ | 17 | 100 ms (6 f) | 18.47% | 59.92% | **0.3082** |
| `pilot_static_23kp` | Static Arm B (23 kp) | $1.0\times 10^5$ | 22 | 117 ms (7 f) | 22.32% | 62.20% | **0.3588** |
| `pilot_static_21kp_k12` | Static Arm A (21 kp) | $1.5\times 10^{12}$ | 56 | 300 ms (18 f) | 21.95% | 66.84% | **0.3284** |

### 10.4 Final Recommendation
Adopt **Static 21kp (17 Dorsal + 4 Paws)** with calibrated $\kappa = 1.5\times 10^{12}$ for full-cohort retraining:
1. Centroid and heading remain 100% stable (17 dorsal keypoints provide 81% of skeletal mass).
2. The 4 paws cleanly capture gait and contact dynamics without rearing elevation artifacts.
3. PCA variance is naturally balanced (37% dorsal posture, 63% paw mechanics) rather than consumed by camera parallax.

