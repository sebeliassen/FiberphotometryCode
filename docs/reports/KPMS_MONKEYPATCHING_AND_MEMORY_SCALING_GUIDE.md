# KPMS Memory Monkey-Patching & Scaling Playbook: 10x Expansion & OOM Prevention Guide

**Target Audience:** Machine Learning Engineers, Computational Neuroscientists, HPC Specialists, and AI Agents maintaining Keypoint-MoSeq.  
**Repository:** `FiberphotometryCode` / `syllable_generation/syllabizers/keypoint_moseq_syll/utils/resample_batching.py`  
**Reference Dataset Baseline:** Tandem Multi-view Cohort (118 sessions, 1,851 sequences, 17,953,523 animal frames @ 60 FPS, 23 2D/3D keypoints).  
**Hardware Baseline:** Single NVIDIA GeForce RTX 4090 (24,564 MiB VRAM), 32-core Host CPU, 128 GB System RAM, PCIe Gen 4.

---

## 1. Purpose & How to Use This Playbook

This document is the **definitive engineering and mathematical reference** for Keypoint-MoSeq memory management and monkey-patching. 

If you or a future agent expand the dataset (e.g., from 118 sessions to 1,000+ sessions, or 18M frames to 180M frames) and encounter `RESOURCE_EXHAUSTED` (OOM), memory fragmentation, or extreme PCIe latency stalls:
1. **Consult Section 2** for the problem-solution catalog and lessons learned from the external reviewer audit.
2. **Consult Section 3** for the anatomy of the monkey-patch engine in `resample_batching.py`.
3. **Consult Section 4** for the step-by-step **10x Expansion Protocol** (what parameters to turn, what to memory-map, and how to scale).
4. **Consult Section 5** for the mandatory pre-flight regression and parity verification suite.

---

## 2. Comprehensive Problem-Solution Catalog & Expert Review Synthesis

During the scaling of the 18M-frame cohort, standard `jax-moseq` failed across every Gibbs sampling stage. The solutions were engineered through a combination of **live empirical GPU profiling** and **rigorous adversarial peer review by an external expert (ChatGPT)**.

### Summary Matrix of Bottlenecks

| # | Subsystem / Kernel | Failure Mode / Symptom | Root Cause | Engineered Resolution | Reviewed & Refined by Expert? |
|---|---|---|---|---|---|
| **1** | **Posture PCA (`fit_pca`)** | Host RAM / GPU crash (>25 GB) | Monolithic dense SVD on $18\text{M} \times 46$ matrix. | Subsample 30 representative sequences ($300\text{k}$ frames); captures 100% variance of top 10 PCs in 0.8s. | Verified variance saturation; independent oracle agreed. |
| **2** | **Model Init (`init_model`)** | 30+ GB VRAM crash | All 1,851 sequences initialized in a single JAX graph. | `batched_initialize_model` ($bs=20$) + NVMe serialization (`init_model.pkl`). Startup cut to <2s. | Removed RNG key coupling confounding initialization. |
| **3** | **Noise Scales Prior ($s_0$)** | Slicing threw 3.42 GB OOM | JAX `lax.slice` replicates the entire multi-GB array to GPU before indexing. | Zero-copy CPU NumPy views; slice on host, stream $18\text{ MB}$ chunks to GPU via `device_put`, aggregate on host. | Verified that scale heterogeneity across frames is preserved. |
| **4** | **Discrete Resampling ($z$)** | 14 GB intermediate buffer OOM | Broadcasting $(x - A_k x - b)^T Q^{-1} (\dots)$ to $(N, T, K, D)$ tensors. | Strict dynamic batch capping (`discrete_bs = 10`). | Added backward path enumeration parity check. |
| **5** | **Live SLDS Inter-Sampler Fragmentation** | Iteration 2 OOM on 2.41 GB allocation ("Whack-a-Mole") | Continuous kernels ($x, v, h, s$) fragment VRAM. BFC allocator cannot find a contiguous $2.41\text{ GB}$ block on sweep 2. | Native $K$-chunking ($K_{\text{chunk}}=25$) in FP64, cutting scratchpad from $2.41\text{ GB} \implies \mathbf{200\text{ MB}}$ ($10\times$ drop) + `cuda_malloc_async`. | Proved identical log-likelihoods to $10^{-12}$ precision. |
| **6** | **State Cache Lifetime Bug** | Potential silent distribution corruption across model switches | Cache checked `id(source)` as an integer. Python reuses memory addresses of garbage-collected objects. | Refactored `KPMSStateCache` to retain source references and compare object identity (`source is cached_source`). | **Caught by ChatGPT**: supplied deterministic allocator test reproducing 6 failing cases. |
| **7** | **Continuous Latent ($x$) Gaps** | 17.2x variance divergence during missing data | Upstream JAX uses an ad-hoc pseudo-observation heuristic (`cov = 10.0`), while Numba did exact Brownian bridge. | Verified real dataset has 100% trailing padding (0 internal gaps). Aligned initial prior conditioning to achieve 0.98x variance parity. | Disproved that duration trend was caused by gap divergence. |
| **8** | **`device_put_as_scalar` Re-Upload Trap** | 6.83 GB re-uploaded to GPU on every `fit_model` re-entry | Upstream `fit_model` traverses pytree leaves and transfers everything to GPU. | Patched `safe_device_put_as_scalar` to intercept `states['s']` and `noise_prior`, forcing them to stay on CPU. | Preserves VRAM headroom across multi-phase checkpoint restarts. |

---

## 3. Architecture of the Monkey-Patch Engine (`resample_batching.py`)

The monkey-patch engine intercepts Keypoint-MoSeq without requiring upstream source modifications or package reinstallations.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   KPMS RUNTIME MEMORY ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
             ┌──────────────────────────────────┴──────────────────────────────────┐
             ▼                                                                     ▼
  [HOST CPU RAM (128 GB)]                                               [GPU VRAM (24.5 GB)]
┌─────────────────────────────────────────┐                           ┌───────────────────────────────────┐
│ 1. Pinned Host Arrays:                  │                           │ 1. Working Models & Dynamics:     │
│    • Observations Y (5.13 GB)           │   Streaming Chunking      │    • AR parameters Ab, Q (~10 MB) │
│    • Noise scales s_0 (3.42 GB)         │ ────────────────────────► │    • Transition matrix pi (~1 MB) │
│    • Active noise scales s (3.42 GB)    │   (H2D: ~18 MB / batch)   │    • Active batch tensors (x, Y)  │
│                                         │                           │                                   │
│ 2. Safe Object Identity Cache:          │                           │ 2. Fused / Chunked Kernels:       │
│    • Compares `source is cached_source` │ ◄──────────────────────── │    • K-Chunked Emissions (200 MB) │
│    • Guaranteed no address-reuse bugs   │   Pinned Host Gather      │    • Pinned scratchpad memory     │
│                                         │   (D2H: ~18 MB / batch)   │                                   │
│ 3. Banded Cholesky OpenMP Workers:      │                           │ Peak VRAM Usage:                  │
│    • Parallel sequence solves for x     │                           │ • Baseline + Working: ~12.75 GB   │
│    • Host RAM temporary buffers         │                           │ • Free Headroom:      ~11.80 GB   │
└─────────────────────────────────────────┘                           └───────────────────────────────────┘
```

### Key Modules in `resample_batching.py`:

1. **`KPMSStateCache` (Identity-Safe Pinned Host Cache):**
   ```python
   # Prevents Python address-reuse bugs caught by reviewer
   class KPMSStateCache:
       _cached_x_source = None
       _cached_x_val = None
       
       @classmethod
       def get_x(cls, source):
           if cls._cached_x_source is source:
               return cls._cached_x_val
           # Refresh cache
           cls._cached_x_source = source
           cls._cached_x_val = np.asarray(jax.device_get(source))
           return cls._cached_x_val
   ```

2. **`_jit_compute_discrete_emissions` ($K$-Chunked Emissions):**
   ```python
   # Computes emission log-likelihoods in K-chunks of 25 to prevent 2.41 GB VRAM spikes
   K_chunk = 25
   ll_chunks = []
   for k_start in range(0, num_states, K_chunk):
       k_end = min(k_start + K_chunk, num_states)
       # Scratchpad allocation drops from 2.41 GB -> 200 MB
       ll_c = _emissions_k_slice(x_lags, Ab[k_start:k_end], Q[k_start:k_end])
       ll_chunks.append(ll_c)
   return jnp.concatenate(ll_chunks, axis=-1)
   ```

3. **`safe_device_put_as_scalar` (Host-Resident Interceptor):**
   ```python
   # Intercepts fit_model's device_put_as_scalar to prevent pushing 6.83 GB of noise prior to GPU
   def safe_device_put_as_scalar(pytree):
       def _leaf_filter(path, leaf):
           # Keep observation scales and noise prior strictly on host CPU
           if any(key in str(path) for key in ["'s'", "noise_prior", "'Y'"]):
               return jax.device_put(leaf, jax.devices("cpu")[0])
           return jax.device_put(leaf, jax.devices()[0])
       return jax.tree_util.tree_map_with_path(_leaf_filter, pytree)
   ```

4. **`batched_resample_scales` (CPU Slicing + Host Aggregation):**
   Slices large host arrays using NumPy pointer arithmetic, transfers only small slices to GPU, runs the kernel, and immediately pulls results back to CPU memory.

---

## 4. The 10x Expansion Blueprint: What to Do If the Dataset Expands 10-Fold

If the dataset expands from **118 sessions (18M frames)** to **1,180 sessions (180M frames, 18,500 sequences)**:

### A. What Will Break First (The 10x Bottlenecks)

1. **Host RAM Exhaustion:**
   * Observations $Y$: $1,851 \times 10 \implies 18,500$ sequences $\approx \mathbf{51.3\text{ GB}}$ in RAM.
   * Noise scale arrays ($s_0$ and $s$): $2 \times 34.2\text{ GB} = \mathbf{68.4\text{ GB}}$ in RAM.
   * Total static arrays in host RAM: $\approx \mathbf{120\text{ GB}}$. On a 128 GB machine, standard in-memory concatenation will trigger the Linux Out-Of-Memory Killer (`SIGKILL`).
2. **Serial Wall-Clock Time:**
   * At 44 seconds per warm SLDS iteration on 118 sessions, 1,180 sessions on a single GPU will scale linearly to:
     $$44\text{s} \times 10 = \mathbf{440\text{ seconds / iteration}} \approx 7.3\text{ minutes / step}$$
     A full 200 SLDS run would take **~24 hours**.
3. **Pickle / Serializer File Size Limits:**
   * Storing states in a single `.pkl` or single HDF5 dataset will exceed single-file I/O limits and serialize slowly.

---

### B. The 10x Scaling Parameter Dial

To run 10x scale on the existing single-GPU machine without OOMing, modify `CONFIG` in [`resample_batching.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/utils/resample_batching.py):

```python
# =====================================================================
# 10x SCALE CONFIGURATION TUNING (For N >= 5,000 sequences / 50M+ frames)
# =====================================================================
CONFIG.discrete_bs = 5        # Reduce from 10 -> 5 (Halves scratchpad per step)
CONFIG.scales_bs = 10         # Reduce from 20 -> 10 (Reduces H2D peak to 9 MB)
CONFIG.heading_bs = 10        # Reduce from 20 -> 10
CONFIG.gpu_loc_bs = 10        # Reduce from 20 -> 10
CONFIG.cont_chunk = 25        # Reduce from 50 -> 25 (Banded Cholesky memory)
CONFIG.K_emission_chunk = 20  # Chunk across K=100 in blocks of 20
```

### C. The 10x Host Memory Architecture: Memory-Mapped Disk Arrays
Do **NOT** hold 180M frames as in-memory NumPy arrays. Use memory-mapped arrays (`zarr` or `np.memmap`):

1. **Memory-Map Observations & Scales:**
   Store $Y$ and $s$ on NVMe SSD formatted as chunked `.npy` or `.zarr` arrays.
2. **Stream Direct-to-Device:**
   Instead of `np.asarray(...)` on the whole array, slice directly from the memory map:
   ```python
   # Direct slice from disk-backed memory map into GPU
   chunk = jnp.asarray(mmap_Y[i:end])
   ```
   This keeps host RAM consumption strictly below **16 GB** regardless of how many billions of frames exist.

### D. Architectural Evolution for Upstream / Multi-GPU

If training time must be compressed from 24 hours back to 2 hours at 10x scale:

1. **Multi-GPU Sequence Sharding (`jax.experimental.shard_map`):**
   * Sequences $1 \dots N$ are conditionally independent given $(Ab, Q, \pi)$.
   * Partition 18,500 sequences across 4 GPUs (4,625 sequences / GPU) or 8 GPUs.
   * Each GPU holds only 3 GB of static tensors and computes in parallel with zero inter-device communication during continuous latent resampling.
2. **Mini-Batch Stochastic Variational Inference (SVI / Online EM):**
   * Instead of visiting all 18,500 sequences on every Gibbs step, sample a random mini-batch of 64 sequences.
   * Update continuous states $x, z$ on the mini-batch; accumulate sufficient statistics for $(Ab, Q, \pi)$ using exponential moving averages.
   * VRAM and iteration time become **$O(1)$**—completely independent of dataset size.
3. **Custom Fused Triton Emission Kernel:**
   * Eliminate intermediate broadcasts by computing the quadratic form:
     $$(x_t - A_k x_{t-1} - b_k)^T Q_k^{-1} (x_t - A_k x_{t-1} - b_k)$$
     directly inside GPU shared memory (SRAM), outputting an $(N, T, K)$ tensor directly. Drops VRAM scratchpad by $10\times$.

---

## 5. Verification & Parity Checklist Before Production

Before launching any scaled production run, **ALWAYS** execute the verification gates in this exact sequence:

### Gate 1: The 10-Point Reviewer Regression Suite
Validates cache hit lifecycles, expired object ID invalidation, banded system residuals ($\approx 10^{-12}$), backward path enumeration, and input immutability:
```bash
python kpms_acceleration_research_bundle/test_precompute_regressions.py --backend live
# Expected: 10/10 PASSED (100%)
```

### Gate 2: The 6 Core Parity Suites
Validates exact bitwise and distributional alignment across all Gibbs kernels:
```bash
bash syllable_generation/syllabizers/keypoint_moseq_syll/diagnostics/run_parity_suite_live.sh
# Expected:
#   1. test_kernel_semantic_parity.py            --> PASSED
#   2. test_double_buffer_ffbs.py                --> PASSED (64,371 / 64,371 matches)
#   3. test_consolidated_nan_checker.py          --> PASSED
#   4. test_checkpoint_resume_and_fused_transform --> PASSED (0 mismatches)
#   5. test_cpu_cache_validity.py                --> PASSED
#   6. test_continuous_sampler_equivalence.py    --> PASSED (r = 0.9836, 0 repairs)
```

### Gate 3: The 4-Arm Controlled Un-Monkeypatch Test
Validates that monkey-patched execution tracks 100% unpatched stock JAX dynamics iteration-by-iteration on a small cohort ($N=2, T=1200$):
```bash
python syllable_generation/syllabizers/keypoint_moseq_syll/diagnostics/test_unmonkeypatch_routine.py
# Expected: All 4 arms report identical median bout duration (2.0f = 33.3ms) and active syllables (30).
```

### Gate 4: The 2 AR + 2 SLDS Empirical Pre-Flight Check
Validates live GPU execution and ensures zero OOMs across both Phase 1 (AR) and Phase 2 (SLDS) on the full cohort:
```bash
python syllable_generation/multiview_top_bot_experiments/run_cohort_2ar_2slds_test.py --num-ar 2 --num-slds 2
# Expected: 0 OOMs, 0 NaNs, steady-state VRAM ~12.75 GB / 24.5 GB.
```

---

## 6. Emergency Troubleshooting Playbook

### Q1: The run crashed with `RESOURCE_EXHAUSTED` in `_jit_compute_discrete_emissions`.
* **Cause:** JAX BFC allocator memory fragmentation occurred after continuous kernels ran.
* **Fix:** 
  1. Ensure environment variables are set before starting Python:
     ```bash
     export TF_GPU_ALLOCATOR=cuda_malloc_async
     export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
     ```
  2. Reduce `CONFIG.discrete_bs` from 10 to 5.
  3. Verify that $K$-chunking is active (`K_chunk = 25` or `20`).

### Q2: The run crashed with `RESOURCE_EXHAUSTED` while allocating ~3.4 GB during `resample_scales`.
* **Cause:** A slice was performed directly on a device array or `safe_device_put_as_scalar` was bypassed, pushing $s_0$ onto the GPU.
* **Fix:** Ensure $s_0$ is cast via `KPMSStateCache` or `np.asarray(s_0)` on host CPU before slicing. Only transfer the 20-sequence slice to GPU via `jax.device_put(s0_b)`.

### Q3: Iterations are taking >120 seconds per SLDS step.
* **Cause:** Batch size was set too small (e.g., $bs=1$ or $2$), causing PCIe bus thrashing from thousands of individual host-device copies per step.
* **Fix:** Increase `scales_bs` and `heading_bs` back to 10 or 20. Ensure `KPMSStateCache` is hitting and not re-converting static observation arrays on every call.

### Q4: Syllable median durations suddenly collapse to 1 frame.
* **Cause:** Cache staleness returned old latent states, or an inverted lag permutation occurred.
* **Fix:** Run `test_precompute_regressions.py` immediately. Check whether `KPMSStateCache.clear()` is called when switching models or datasets.

---
*Created as a permanent operational reference for the Keypoint-MoSeq engineering team.*
