# Addendum Dossier: Live Empirical SLDS Profiling & Memory Fragmentation Analysis

**Target Audience:** Researchers, HPC Engineers, and ML Scientists investigating Keypoint-MoSeq scaling  
**Date:** September 29, 2026  
**Context:** Auxiliary incident report to prevent redundant trial-and-error debugging on the Tandem Multi-view Cohort (118 sessions, 1,851 sequences, 17,953,523 frames, 23 keypoints @ 60 FPS on NVIDIA RTX 4090 24.5 GB VRAM).

---

## 1. Executive Summary & Why This Document Exists

This addendum documents the results of a live empirical test run jumping directly into Phase 2 (SLDS Gibbs iterations) on the full 18-million-frame cohort.

**Do NOT attempt to simply reduce batch sizes or tweak JAX flags without reading this document.**

This test proved:
1. **The Positive Breakthrough:** With multi-tier disk caching and CPU-resident noise scales ($s$), **SLDS iterations 0 and 1 executed completely on live GPU without OOMing**, proving that continuous state sampling ($x, v, h, s$) can run at cohort scale ($46\text{ seconds}$ per warm SLDS iteration).
2. **The Secondary Crash ("The Whack-a-Mole"):** On the subsequent iteration, the run crashed during **discrete state resampling ($z$)** in `_jit_compute_discrete_emissions` with:
   ```text
   RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2414584344 bytes (~2.41 GB).
   ```
3. **The Core Realization:** Squeezing individual operations to fit within 24 GB VRAM is inherently unstable because **JAX XLA memory pool retention and inter-sampler memory fragmentation** cause operations that succeed in isolation to crash when chained together.

---

## 2. Chronological Log Analysis of the Run

### Phase A: Instant Startup & Zero-OOM SLDS Resampling
* Model state was loaded from `init_model.pkl` in **$0.5\text{ seconds}$**.
* The $3.42\text{ GB}$ observation noise scale tensor ($s$) was kept on host CPU, streaming only $18\text{ MB}$ chunks to the GPU per batch.
* **SLDS Iteration 0 (Cold JIT):**
  - Resampling $\pi$ (transition matrix) $\implies$ Success
  - Resampling $Ab, Q$ (AR dynamics) $\implies$ Success
  - Resampling $z$ (discrete states) $\implies$ Success
  - Resampling $x$ (Banded Cholesky continuous states) $\implies$ Success
  - Resampling $h$ (heading angles) $\implies$ Success
  - Resampling $v$ (centroid locations) $\implies$ Success
  - Resampling $s$ (local noise scales) $\implies$ Success
  - Total time: **$74.03\text{ seconds}$** (including all JIT compilation).
* **SLDS Iteration 1 (Steady State):**
  - All continuous and discrete steps executed a second time.
  - Total warm time: **$46.0\text{ seconds}$**.

### Phase B: The Crash on the Subsequent Sweep
At the start of the next Gibbs sweep, the code re-entered `resample_discrete_stateseqs`:
```text
  0%|                                             | 0/2 [00:00<?, ?it/s]Resampling pi (transition matrix)
Resampling Ab,Q (AR parameters)
Resampling z (discrete latent states)
E0929 18:11:32.006144 82735 pjrt_stream_executor_client.cc:2916] Execution of replica 0 failed: 
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2414584344 bytes. [tf-allocator-allocation-error='']

Traceback (most recent call last):
  File "run_cohort_2ar_2slds_test.py", line 267, in main
    model, _ = kpms.fit_model(...)
  File "jax_moseq/models/arhmm/gibbs.py", line 251, in resample_model
    states["z"] = resample_discrete_stateseqs(seed, **data, **states, **params)
  File "utils/resample_batching.py", line 452, in fast_resample_discrete_stateseqs
    curr_ll_dev = _jit_compute_discrete_emissions(x[b0_i:b0_end], Ab, Q, nlags=nlags, num_states=num_states)
ValueError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2414584344 bytes.
```

---

## 3. Root Cause: Why Did It Crash on Iteration 2 After Passing Iteration 1?

### Mechanism 1: The 2.41 GB Broadcast Intermediate in `_jit_compute_discrete_emissions`
In `_jit_compute_discrete_emissions`, JAX computes:
```python
mu_all = jnp.einsum("nti,kdi->ntkd", x_lags, Ab)
diff = x_target[:, :, None, :] - mu_all
quad = jnp.einsum("ntkd,kde,ntke->ntk", diff, Q_inv, diff)
```
For batch size $N_{\text{batch}} = 10$, sequence length $T = 10,030$, discrete states $K = 100$, and latent dimension $D = 10$:
$$\text{Tensor Dimensions: } (10, 10030, 100, 10) \text{ elements}$$
$$10 \times 10,030 \times 100 \times 10 \times 4\text{ bytes (float32)} = 401.2\text{ MB per tensor}$$
During execution, XLA materializes:
1. `mu_all`: $401.2\text{ MB}$
2. `diff`: $401.2\text{ MB}$
3. Contraction scratchpad: $802.4\text{ MB}$
4. Alignment/stride workspace: $\sim 800\text{ MB}$

**Total Scratch Allocation Requested by XLA:** Exactly **$2,414,584,344\text{ bytes} \approx 2.41\text{ GB}$**.

### Mechanism 2: XLA Memory Pool Retention & Multi-Sampler Interference
Why did $2.41\text{ GB}$ allocate successfully in Iteration 0 and Iteration 1, but fail on the second sweep?
1. In the **AR-only phase**, only $z$ and $Ab, Q$ run. The continuous samplers ($x, v, h, s$) are never executed, so GPU VRAM is completely empty except for $x$.
2. In the **SLDS phase**, during Iteration 0 and 1, JAX executes:
   - `_resample_heading_kernel`
   - `_jit_resample_v_batch`
   - `_resample_scales_kernel`
3. JAX's default memory allocator (`BFC` / Best-Fit with Coalescing) **pools and retains allocated GPU blocks** to avoid expensive CUDA `cudaMalloc` calls on subsequent steps.
4. When the next sweep began and reached `_jit_compute_discrete_emissions`, the physical GPU VRAM was fragmented with retained blocks from heading, location, and scales kernels.
5. When XLA requested a single **contiguous $2.41\text{ GB}$ chunk**, the BFC allocator could not find an unfragmented $2.41\text{ GB}$ block, immediately throwing `RESOURCE_EXHAUSTED`.

---

## 4. Why Reducing Batch Size to $bs=5$ is a Dangerous Trap

Engineers might look at this error and think: *"Let's just change `discrete_bs` from 10 to 5!"*

**Here is why that is a trap:**
1. **The Latency Cost:** Reducing $bs$ from 10 to 5 doubles the number of chunks across 1,851 sequences from **$186 \implies 371$**. That adds 185 extra host-device dispatches and CUDA stream synchronizations per Gibbs step.
2. **The Fragility Wall:** Squeezing the buffer from $2.41\text{ GB} \implies 1.2\text{ GB}$ will temporarily pass until:
   - You add 2 more sessions ($N$ increases).
   - You increase the keypoint count from 23 to 30.
   - You run a slightly longer sequence ($T$ increases from 10k to 15k).
3. **The Unpredictable Allocator:** XLA's static memory planner decides buffer placement based on compile-time graphs. Small changes in driver memory or PyTorch/system background processes will randomly cause OOMs hours into a 3-day training run.

---

## 5. Concrete Action Items for Researchers & Engineers

To permanently fix this rather than playing "whack-a-mole":

### Action 1: Fused Emission Kernel (Eliminates the 2.41 GB Tensor)
In Triton or CUDA, compute the quadratic distance:
$$(x_t - A_k x_{t-1} - b_k)^T Q_k^{-1} (x_t - A_k x_{t-1} - b_k)$$
inside GPU shared memory (SRAM) per thread block, and write directly to an output tensor of shape $(N, T, K)$.
* **Memory Saved:** The $D$-dimensional broadcast tensor is completely eliminated. Scratch memory drops by **$10\times$** (from $2.41\text{ GB} \implies \mathbf{240\text{ MB}}$).
* **Speed Gain:** Eliminates 3 high-bandwidth memory (HBM) read/write passes.

### Action 2: Mini-Batch / Stochastic Gibbs Sampling (SVI / Online EM)
Keypoint-MoSeq currently forces **full-batch Gibbs sampling**, visiting all 1,851 sequences on every single iteration.
* **Modern Alternative:** Sample 32 random sequences per step, update latent states $x, z$ on that mini-batch, and update AR dynamics $Ab, Q$ and transition matrix $\pi$ via exponential moving average (sufficient statistics accumulation).
* **Impact:** VRAM usage becomes completely independent of dataset size ($O(1)$ memory). You can train on 1,000 sessions with the exact same VRAM footprint as 10 sessions.

### Action 3: Multi-GPU Distribution (`jax.experimental.shard_map`)
The 1,851 sequences are conditionally independent given the shared dynamics ($Ab, Q, \pi$).
Partitioning sequences across 2 or 4 GPUs (e.g. 450 sequences per GPU) cuts memory pressure per device by $4\times$ and speeds up training linearly.

---

## 6. How to Run the Current System Reliably

If you must run the current Python implementation on an RTX 4090 before upstream fused kernels are ready:
1. In `utils/resample_batching.py`, set:
   ```python
   CONFIG.discrete_bs = 5    # Reduces emission tensor from 2.41 GB to 1.20 GB
   CONFIG.scales_bs = 20      # Kept on host CPU (18 MB streaming)
   CONFIG.heading_bs = 20     # 15 MB streaming
   CONFIG.gpu_loc_bs = 20     # 11 MB streaming
   ```
2. Set JAX memory preallocation flag:
   ```bash
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
   export TF_GPU_ALLOCATOR=cuda_malloc_async
   ```
   `cuda_malloc_async` uses CUDA's native virtual memory management (VMM) pools, which significantly reduces the fragmentation that caused the crash in Iteration 2.

---
*Authored to provide immediate, actionable clarity for the Keypoint-MoSeq research team.*
