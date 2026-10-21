# Keypoint-MoSeq (KPMS): 8-Keypoint Architecture Transition & Production Status

**Date**: September 8, 2026  
**Cohort**: 30 sessions (8 WT, 7 DAT-Het KI across Saline & Amphetamine, 5,516,954 frames @ 40 FPS)  
**Hardware**: NVIDIA RTX 4090 (24 GB) + 32-core AMD Ryzen CPU, 128 GB RAM  
**ntfy Topic**: `https://ntfy.sh/woody_videos_seb`  

---

## 1. Executive Summary & The Core Problem

During the initial 16-keypoint Keypoint-MoSeq evaluation (across 15 seeds with $\kappa=150,000$ and $\kappa=500,000$), behavioral syllables exhibited high idiosyncrasy between individual mice. Specifically:
- Exact animal label permutation tests showed that animal identity explained substantially more syllable variance than genotype ($p > 0.05$ for genotype separation under standard permutations).
- **Hypothesis**: Redundant surface/perimeter keypoints in the 16-keypoint skeleton caused the autoregressive pose space to act as a proxy for static morphological size and animal body dimensions rather than dynamic pose change.

---

## 2. Phase 1 Audit: Morphology Leakage Confirmed

We conducted an empirical morphology audit (`audit_kpms_morphology.py`) across all 5.5M frames:
1. **Axial Chain vs. Bounding Dimensions**:
   - Spine chain length (`nose -> neck -> mid_back -> tail_base`) and width (`left_midside <-> right_midside`) vary systematically across individual animals.
2. **Latent PCA Variance Decomposition**:
   - **$PC_0$**: Correlates **$r = 0.96$** with static animal body length/proportions.
   - **$PC_4$**: Correlates **$r = 0.94$** with lateral body width.
3. **Size Stratification**:
   - Because $PC_0$ and $PC_4$ were fed directly into the AR-HMM dynamics, syllables split along animal size boundaries: physically larger mice received distinct syllable assignments from smaller mice even when performing identical kinematic actions.

---

## 3. The 8-Keypoint Reduction & Geometry

To eliminate perimeter/contour redundancy while preserving bending, stretching, rearing, and turning dynamics:

* **Selected 8 Informative Keypoints**:
  1. `nose` (Head apex / heading)
  2. `left_ear_tip` (Head roll / yaw)
  3. `right_ear_tip` (Head roll / yaw)
  4. `neck` (Cervical pivot)
  5. `mid_back` (Torso curvature / arch)
  6. `tail_base` (Posterior anchor / locomotion pivot)
  7. `left_midside` (Lateral bulge / breathing / side-bending)
  8. `right_midside` (Lateral bulge / breathing / side-bending)

* **Omitted Redundant Points (8 points removed)**:
  - Spine redundancy (`spine_low`, `spine_high`)
  - Extended tail markers (`tail_mid`, `tail_tip`)
  - Peripheral paw markers prone to occlusion noise (`left_hind_paw`, `right_hind_paw`, `left_fore_paw`, `right_fore_paw`)

* **PCA Refit & Dimensionality**:
  - Project directory: `amph_project_8kp`
  - PCA fitted on the 8-keypoint subset: **6 Latent Dimensions capture 95.52% of total pose variance** (elbow on scree plot, down from 10 PCs).
  - Saved PCA: `amph_project_8kp/pca.p`

---

## 4. Modernized High-Performance Sampling Engine

All runs use the modernized Gibbs sampling optimizations developed within the last 24 hours (`utils/resample_batching.py` and `kpms_pipeline.py`):

1. **Continuous Latents ($x$)**: In-place Banded Cholesky GMRF solver (**58.04x speedup** over upstream JAX Kalman smoother). Chunk size = 50.
2. **Discrete States ($z$)**: GPU tensor-core emission evaluation + multi-threaded Numba parallel CPU FFBS with row-contiguous transition matrix striding (**15.6x speedup**).
3. **AR Dynamics ($Ab, Q$)**: Zero-allocation streaming Numba accumulators (temporary memory footprint dropped from >5 GB to 0 MB, **9.72x speedup**).
4. **Subprocess GPU Isolation**: Worker child processes isolated from CUDA (`CUDA_VISIBLE_DEVICES=""`), preventing VRAM memory exhaustion.
5. **Memory Allocator Controls**: `TF_GPU_ALLOCATOR=cuda_malloc_async`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`.

---

## 5. 8-Keypoint Scout Run Results (COMPLETED)

* **Script**: `run_8kp_scout.py`
* **Configuration**:
  - Model: `scout_8kp_k150k`
  - 25 AR iterations + 25 SLDS iterations (50 total)
  - $\kappa_{\text{init}} = 150,000$ (SLDS $\kappa = 15,000$)
  - Latent Dim = 6
  - Dataset: Full 30 sessions (5,516,954 frames @ 40 FPS)
* **Performance**:
  - **Completed in 5.81 minutes (348.7s)** for the entire 50-iteration fit!
  - Warmed AR iteration: ~2.0s / iter.
  - Warmed SLDS iteration: ~7.5s / iter.
* **Empirical Syllable Metrics**:
  - **Median Syllable Duration**: **16.0–17.0 frames (~400–425 ms)**.
  - Exactly matches the target timescale (~15 frames / 375 ms).
  - Clean exponential duration distribution without 1-frame flickering or state-locking.
  - Syllable diversity: ~50–60 distinct syllables identified across the cohort.

---

## 6. Execution Plan Ready for System Restart

The user requested: **pkill any lingering processes, and launch 5 production models with 50 AR + 150 SLDS iterations (200 total iters each) using the 8-keypoint setup.**

### Planned Configuration:
- **Project Directory**: `amph_project_8kp`
- **Latent Dimensions**: 6
- **Iterations**: 50 AR + 150 SLDS = 200 total iterations per model
- **Kappa**: $\kappa_{\text{init}} = 150,000 \to \kappa_{\text{SLDS}} = 15,000$
- **Checkpointing Interval**: Every 50 iterations (saves at 0, 50, 100, 150, 200; ~10.8 GB per model instead of 25 GB, preventing disk exhaustion)
- **Disk Safeguard**: Automated halt if free disk space drops below 75 GB
- **5 Model Seeds**:
  1. `amph_8kp_k150k_run_01` (Seed 42)
  2. `amph_8kp_k150k_run_02` (Seed 101)
  3. `amph_8kp_k150k_run_03` (Seed 202)
  4. `amph_8kp_k150k_run_04` (Seed 303)
  5. `amph_8kp_k150k_run_05` (Seed 404)
- **Estimated Runtime**: At ~7.5–9.5s / SLDS iteration, each 200-iter model takes **~25–30 minutes**. All 5 models will finish in **~2.2 to 2.5 hours total**.
- **Alerts**: Push notifications via ntfy (`https://ntfy.sh/woody_videos_seb`) after every completed run.

### Pre-Created Execution Scripts:
1. `run_amph_8kp_production.py` (Single-model production runner for 8kp)
2. `queue_amph_8kp_k150k.sh` (Detached 5-seed queue orchestrator)
