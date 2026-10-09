# Keypoint-MoSeq Performance Discrepancy & Bug Report

**Date:** September 8, 2026  
**Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM), 32-core Host CPU, 128 GB System RAM  
**Environment:** Linux, CUDA 12.4, JAX with GPU acceleration, Numba with OpenMP threading  
**Dataset:** Jonathan AMPH Dataset (30 sessions, 562 segmented sequences, 5,516,954 frames)  

---

## 1. Executive Summary

We observe an order-of-magnitude performance variance in Keypoint-MoSeq Gibbs sampling iterations across different execution contexts on the exact same dataset and hardware:

| Scenario / Context | Iteration Speed (AR Phase) | Iteration Speed (SLDS Phase) | Notes |
| :--- | :--- | :--- | :--- |
| **Acceptance / Benchmark Test** (`profile_fit_model_gap.py` / `test_amph_production_acceptance.py`) | **~7.8 – 8.1 s / iter** | **~15.8 – 20.0 s / iter** | Full 5.5M frame dataset loaded from checkpoint |
| **Standard 6-Iter Benchmark** (`run_amph_benchmark.py`) | **~7.8 – 12.3 s / iter** | **~23.0 – 37.7 s / iter** | Includes initial JIT compilation & checkpoint saving |
| **Live Production Runs** (`queue_amph_k150k_runs.sh` -> `run_amph_production.py`) | **~69.8 – 72.6 s / iter** | *Halted before SLDS* | **5x - 9x regression** occurring right from Iteration 1 |

---

## 2. Evidence & Logs

### Log Extract A: The Regression in Production (`logs/amph_prod_k150k_run_01.log`)
```text
[Batch Init] Done. State 'x' shape: (562, 10030, 10)
[phase 1] Current iter (0) < AR target (50). Running AR fit in chunks of 10...

  0%|                                            | 0/11 [00:00<?, ?it/s]
  9%|███▎                                | 1/11 [01:12<12:06, 72.61s/it]
 18%|██████▌                             | 2/11 [02:24<10:48, 72.11s/it]
```
*Every single AR iteration is taking ~72 seconds.*

### Log Extract B: The Fast Isolated Profile on the Exact Same Data (`diagnostics/profile_fit_model_gap.py`)
```text
=== PROFILING FIT_MODEL GAP ===
Loading checkpoint...

>>> Test 1: Direct keypoint_slds.resample_model(ar_only=True) (2 iters)...
  Direct Iteration 1: 12.39 s
  Direct Iteration 2: 7.85 s

>>> Test 2: kpms.fit_model(ar_only=True, num_iters=2, save_every_n_iters=None, generate_progress_plots=False)...
100%|█████████████████████████████████████| 3/3 [00:24<00:00,  8.13s/it]
  fit_model (2 iters total): 24.78 s -> 8.13 s/it
```

---

## 3. Detailed Component Profiling (What Works When Fast)

From our granular profiling runs ([MEASURED_GRANULAR_PROFILE_FULL_DATASET.md](MEASURED_GRANULAR_PROFILE_FULL_DATASET.md)), the fast steady-state breakdown per Gibbs sweep is:

- **GPU Discrete Emissions ($z$):** 0.75 s
- **FFBS ($z$):** ~1.0 s
- **AR Streaming Dynamics ($Ab, Q$):** ~2.4 s
- **Markov Transitions ($\pi, \beta$):** ~0.05 s
- **Continuous Latents (Banded Cholesky $x$):** ~4.05 s (SLDS only)
- **Fused Rigid Transform:** ~3.48 s (SLDS only)
- **Centroid Location ($v$):** ~1.39 s (SLDS only)
- **Heading ($h$):** ~0.69 s (SLDS only)
- **Noise Scales ($s$):** ~0.75 s (SLDS only)

Total expected iteration time: **~5.5 – 8.1 s (AR)** and **~15.8 – 20.0 s (SLDS)**.

---

## 4. The Core Issue: Why Does Production Run at 72s / Iter?

1. **Host-to-Device / Device-to-Host Thrashing Across 5.5M Frames:**
   - In `run_amph_production.py`, the model is initialized via `init_model` and `batched_initialize_model`.
   - In production, when `fit_model` or `_wrapped_resample` runs, arrays in `data` and `states` may reside in host memory or trigger implicit CPU copies before being dispatched to JAX/CUDA kernels.
   - For 562 sequences of shape `(10030, 10)`, copying states back and forth between Host and Device over PCIe adds **~60 seconds of serialization / transfer penalty per step**.

2. **Monkeypatch Discrepancies:**
   - Upstream JAX stock kernels for `arhmm.resample_discrete_stateseqs` execute purely on GPU VRAM with zero PCIe transfer when tensors are DeviceArrays.
   - When custom Numba or PyTorch CPU fallbacks intercept `arhmm.resample_discrete_stateseqs`, converting 562 sequences to NumPy, executing CPU FFBS, and converting back to JAX DeviceArray causes massive bus congestion (~65s).

3. **Cold JIT / Memory Allocation Churn:**
   - Repeated memory allocations in `kpms.fit_model` vs pre-allocated buffers.

---

## 5. Contents of This Package

- `BUG_REPORT_KPMS_PERFORMANCE.md`: This comprehensive overview.
- `MEASURED_GRANULAR_PROFILE_FULL_DATASET.md`: Component-by-component timing log on 5.5M frames.
- `OPTIMIZATION_EXPLANATION.md`: Architecture and design of the custom kernels.
- `kpms_pipeline.py`: Production pipeline harness.
- `run_amph_production.py`: Script executing the production runs.
- `run_amph_benchmark.py`: Benchmark script used for 2 AR + 2 SLDS verification.
- `queue_amph_k150k_runs.sh`: Production queue runner for 10 consecutive models.
- `utils/resample_batching.py`: High-performance monkeypatches, Banded Cholesky solver, state cache.
- `utils/batch_init.py`: Batched model state initializer with GPU array placement.
- `diagnostics/`: Complete suite of 6 parity and semantic validation tests.

---

## 6. Root Cause Resolution & Verification (September 8, 2026)

All leading causes identified by code inspection were confirmed, corrected, and validated:

1. **Restored Kernel Bindings in `apply_optimizations()`**:
   - `arhmm_gibbs.resample_discrete_stateseqs = fast_resample_discrete_stateseqs`
   - `arhmm_gibbs.resample_ar_params = vectorized_resample_ar_params`
   - `arhmm_model.resample_discrete_stateseqs = fast_resample_discrete_stateseqs`
   - `arhmm_model.resample_ar_params = vectorized_resample_ar_params`

2. **Eliminated Upstream Serialization**:
   - Replaced `MIXED_MAP_ITERS = 64` with `MIXED_MAP_ITERS = 1` in `run_amph_production.py` and `run_amph_benchmark.py`.

3. **Resolved CPU Device Commitment on Fresh Initialization**:
   - In `utils/batch_init.py`, updated `batched_init_model_by_video` so concatenated states (`x, v, h, s, z`) and `noise_prior` are explicitly placed on `jax.devices()[0]` (`cuda:0`).

4. **Added NumPy bool8 Compatibility**:
   - Added `np.bool8 = np.bool_` in `resample_batching.py`, ensuring `kpms.fitting.check_for_nans` monkeypatch installs cleanly without throwing `AttributeError`.

5. **Fixed Checkpoint Save Arguments**:
   - Updated `kpms.save_hdf5` in `kpms_pipeline.py` to supply both `exist_ok=True, overwrite=True`.

6. **Empirical Verification Results (5.5M Frames)**:
   - **AR Iteration 0 (Cold)**: 9.02 s
   - **AR Iteration 1 (Warm)**: 5.70 s
   - **AR Iteration 2 (Warm)**: **4.60 s** (down from **72.6 s/iter**, a **16x speedup**)
   - **SLDS Iteration 4 (Warm)**: **17.10 s**
   - **All 6/6 Parity & Semantic Test Suites**: 100% PASSED (Kernel Parity, Double-Buffer FFBS, NaN Checker, Resume Continuity, CPU Cache Validity, Continuous Sampler Equivalence).
