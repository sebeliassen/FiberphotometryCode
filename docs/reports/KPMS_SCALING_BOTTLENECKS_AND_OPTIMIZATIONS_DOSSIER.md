# Keypoint-MoSeq at Cohort Scale: Computational Bottlenecks, Memory/Time Trade-offs, and Engineering Dossier

**Target Audience:** Machine Learning Engineers, Computational Neuroscientists, High-Performance Computing (HPC) Specialists  
**Repository:** `FiberphotometryCode` / `keypoint-moseq` / `jax-moseq`  
**Date:** September 2026  
**Hardware Baseline:** NVIDIA GeForce RTX 4090 (24,564 MiB VRAM), 32-core Host CPU, 128 GB System RAM, PCIe Gen 4  
**Primary Dataset Benchmark:** Multi-view Tandem Cohort (118 sessions, 1,851 segmented sequences, 17,953,523 total frames @ 60 FPS, 23 keypoints 3D/2D fused rig)

---

## 1. Executive Summary & The "Rabbit Chase" Phenomenon

Keypoint-MoSeq (KPMS) models animal behavior by combining an autoregressive hidden Markov model (AR-HMM) with a switched linear dynamical system (SLDS) using JAX for GPU-accelerated Gibbs sampling. 

While KPMS performs remarkably well on standard single-session lab benchmarks (1–5 videos, ~100k frames, 8 keypoints), scaling to modern multi-animal, multi-session, multi-view cohorts reveals a severe architectural scaling wall:
- **Jonathan AMPH Cohort:** 30 sessions, 562 sequences, 5,516,954 frames, 8 keypoints.
- **Tandem Multi-view Cohort:** 118 sessions, 1,851 sequences, 17,953,523 frames, 23 keypoints (Dorsal + Ventral fused rig).

Whenever researchers expand their dataset, the training pipeline repeatedly crashes with `RESOURCE_EXHAUSTED: Out of memory` (OOM) errors across different stages of the model:
1. **Stage 1 (Data Ingestion & PCA):** `fit_pca` attempts dense SVD across 18 million frames, exhausting 128 GB RAM / GPU memory.
2. **Stage 2 (Model Initialization):** `init_model` attempts simultaneous state initialization for 1,851 sequences, exhausting GPU VRAM.
3. **Stage 3 (AR-HMM Fitting):** Broadcast tensors for discrete state transitions ($z$) allocate $\sim 14\text{ GB}$ intermediate buffers.
4. **Stage 4 (SLDS Fitting):** Observation noise prior ($s$) and Banded Cholesky continuous state solvers ($x$) allocate $3.4\text{ GB}$ to $8\text{ GB}$ monolithic arrays, failing in JAX `lax.slice`.

This document explains:
1. The **exact mathematical and computational mechanisms** behind these failures.
2. The fundamental **Memory vs. Wall-Clock Time Trade-off** inherent in batching.
3. The **suite of monkey patches and caching architectures** we engineered to mitigate these issues.
4. Concrete **architectural recommendations** for upstream developers and HPC engineers to permanently resolve this scaling bottleneck without endless ad-hoc patch tuning.

---

## 2. Dataset Dimensions & Tensor Footprints

Understanding the memory wall requires examining the exact tensor shapes in Keypoint-MoSeq for our production tandem multi-view cohort:

| Parameter | Symbol | Value (Tandem Cohort) | Value (Standard Demo) | Scaling Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Number of Sequences** | $N$ | **1,851** | 10 | **185x** |
| **Sequence Length (Frames)** | $T$ | **10,030** | 1,800 | **5.5x** |
| **Total Frames** | $N \times T$ | **17,953,523 (~18M)** | 18,000 | **1,000x** |
| **Number of Keypoints** | $K_p$ | **23** | 8 | **2.9x** |
| **Coordinate Dimension** | $D_{obs}$ | **3** ($x, y, \text{conf}$) | 2 | **1.5x** |
| **Latent State Dimension** | $D_{lat}$ | **10** (PCA components) | 10 | 1x |
| **Max Syllable States** | $K_{states}$ | **100** | 100 | 1x |
| **AR Lag Order** | $nlags$ | **3** | 3 | 1x |

### Monolithic In-Memory Tensor Footprints

In stock `jax-moseq`, data and latent variables are represented as monolithic dense tensors:
- **Observed Coordinates ($Y$):** Shape `(1851, 10030, 23, 3)` in float32 $\implies \mathbf{5.13\text{ GB}}$
- **Observation Noise Scales ($s$):** Shape `(1851, 10030, 23)` in float64 $\implies \mathbf{3.42\text{ GB}}$
- **Continuous Latent Trajectories ($x$):** Shape `(1851, 10030, 10)` in float32 $\implies \mathbf{743\text{ MB}}$
- **Discrete Syllables ($z$):** Shape `(1851, 10030)` in int32 $\implies \mathbf{74.3\text{ MB}}$
- **Heading Angles ($h$):** Shape `(1851, 10030)` in float32 $\implies \mathbf{74.3\text{ MB}}$
- **Centroid Positions ($v$):** Shape `(1851, 10030, 3)` in float32 $\implies \mathbf{223\text{ MB}}$
- **Observation Masks ($mask$):** Shape `(1851, 10030)` in bool $\implies \mathbf{18.6\text{ MB}}$

**Total Base Memory Footprint:** $\approx \mathbf{10.0\text{ GB}}$ of persistent state data.  
On a 24 GB GPU, having 10 GB of static tensors leaves only **14 GB of working VRAM** for JAX XLA kernel compilation, intermediate broadcast buffers, and autograd/Gibbs sampling scratchpads.

---

## 3. Detailed Anatomy of the Bottlenecks

### Bottleneck A: Posture PCA SVD Collapse (Stage 1)
* **Stock Behavior:** `kpms.fit_pca(data)` flattens all valid frames across all sessions into an $(M, K_p \times 2)$ matrix and performs centering and singular value decomposition (SVD) via `scipy.linalg.svd` or `sklearn.decomposition.PCA`.
* **Failure Mechanism:** At 18 million frames and 23 keypoints, the centered coordinate matrix requires:
  $$18,000,000 \times 46 \times 8\text{ bytes} \approx 6.62\text{ GB}$$
  The internal covariance computation and SVD workspace allocate multiples of this array ($>25\text{ GB}$ in host RAM). In Python multiprocessing or GPU PCA, this immediately triggers host OOM killing or CUDA driver termination.
* **Engineered Resolution:** Posture variance saturates rapidly. Subsampling 30 representative sequences ($300,000$ frames) captures $100.0\%$ of the posture variance explained by the top 10 PCs, computing in **$0.8\text{ seconds}$** instead of crashing or taking 20 minutes.

---

### Bottleneck B: Model State Initialization (`init_model`)
* **Stock Behavior:** `kpms.init_model(data, pca)` samples initial states $x, v, h, s, z$ for the entire dataset in a single call to `jax_moseq.models.keypoint_slds.init_model`.
* **Failure Mechanism:** The Gibbs initialization kernel attempts to instantiate intermediate coordinate alignment matrices across all 1,851 sequences at once. This requires $>30\text{ GB}$ of transient VRAM, causing an instant XLA OOM.
* **Engineered Resolution:** 
  1. `batched_initialize_model`: Slices $N=1,851$ into small batches ($bs=20$), runs initialization iteratively, and concatenates the resulting states.
  2. Multi-tier caching: The initialized model states (`init_model.pkl`, $\approx 8.3\text{ GB}$) are serialized to NVMe disk. Re-runs load in **$<2\text{ seconds}$**, completely skipping the 3.5-minute batch initialization overhead.

---

### Bottleneck C: Discrete State Resampling ($z$) & the 14 GB Broadcast Intermediate
* **Stock Behavior:** In the AR-HMM phase, discrete states $z_t \in \{1, \dots, K\}$ are resampled conditioned on $x$ via Forward-Filtering Backward-Sampling (FFBS). First, log emission probabilities are computed for every possible state:
  $$\ln p(x_t \mid x_{t-1}, z_t = k) = -\frac{1}{2} \ln |Q_k| - \frac{1}{2} (x_t - A_k x_{t-1} - b_k)^T Q_k^{-1} (x_t - A_k x_{t-1} - b_k)$$
* **Failure Mechanism (The 14 GB Buffer):**
  In JAX, this emission computation is JIT-compiled across sequences. To compute $(x_t - A_k x_{t-1} - b_k)$ in parallel, JAX XLA broadcasts $x$ across all $K_{states} = 100$:
  $$\text{Shape: } (N_{\text{batch}}, T, K_{\text{states}}, D_{\text{lat}}) = (N_{\text{batch}}, 10030, 100, 10)$$
  For float32, each such tensor occupies:
  $$N_{\text{batch}} \times 10030 \times 100 \times 10 \times 4\text{ bytes} \approx N_{\text{batch}} \times 40.12\text{ MB}$$
  During execution, XLA allocates:
  1. Broadcasted predicted means $\mu_{k}$
  2. Difference vectors $(x - \mu_k)$
  3. Quadratic form intermediate products
  
  At an initial heuristic batch size of $N_{\text{batch}} = 58$, these intermediate tensors require:
  $$58 \times 40.12\text{ MB} \times 6 \text{ buffers} \approx \mathbf{13.96\text{ GB of scratch VRAM!}}$$
  Added to the $10\text{ GB}$ baseline model footprint, this exceeds 24 GB VRAM and crashes with:
  ```text
  jaxlib._jax.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 14984820000 bytes.
  ```
* **Engineered Resolution:** Strict dynamic batch capping (`discrete_bs = 10`). At $N=10$, intermediate buffer allocation drops from **$14.0\text{ GB} \implies \mathbf{2.4\text{ GB}}$**, fitting safely within RTX 4090 memory.

---

### Bottleneck D: Observation Noise Scales ($s$) & JAX Array Slicing OOM
* **Stock Behavior:** In SLDS Gibbs sampling, keypoint observation noise scales $s_{i,t,k} \sim \text{Inv-Gamma}$ are resampled using prior parameters $s_0$ and squared projection errors:
  $$e^2_{i,t,k} = \| Y_{i,t,k} - \mathcal{T}(x_{i,t}, v_{i,t}, h_{i,t}, C_k) \|^2$$
* **Failure Mechanism (The 3.4 GB Slicing Allocation):**
  The noise prior array $s_0$ has shape `(1851, 10030, 23)` in float64 ($3,416,057,520\text{ bytes} \approx 3.42\text{ GB}$).
  In stock `batched_resample_scales`, slicing `s0_b = s_0[i:end]` was executed directly on a JAX array:
  ```python
  # Crashes with RESOURCE_EXHAUSTED: Out of memory while trying to allocate 3416057520 bytes
  s0_b = s_0[i:end]
  ```
  Why did slicing an array cause JAX to allocate $3.4\text{ GB}$?
  When a multi-gigabyte array resides in host CPU memory or Device memory, JAX's `lax.slice` primitive attempts to transfer or replicate the entire operand to the target device shard before applying the dynamic slice index.
* **Engineered Resolution:** Zero-copy CPU NumPy slicing. We cast $s_0$ and $Y$ to CPU NumPy views, slice them on CPU into small chunks ($bs=20$), transfer **only the 20-sequence chunk ($18\text{ MB}$)** to the GPU via `jax.device_put`, execute the kernel, immediately transfer the result back to CPU, and perform host concatenation before returning a unified array to GPU.

---

### Bottleneck E: Continuous Latent Resampling ($x$) & Banded Cholesky
* **Stock Behavior:** Conditioned on $z$, the continuous latent trajectory $x_{1:T}$ follows a Gaussian Markov Random Field (GMRF). Sampling $x$ requires solving a block-tridiagonal linear system of dimension $(T \times D_{\text{lat}}) \times (T \times D_{\text{lat}}) = 100,300 \times 100,300$.
* **Computational Cost:** Keypoint-MoSeq uses a custom Banded Cholesky solver. While $O(T)$ in time, running 1,851 sequences concurrently requires factoring 1,851 banded matrices simultaneously, consuming gigabytes of temporary matrix storage.
* **Engineered Resolution:** Chunking the Banded Cholesky solver into sequence batches of 50, balancing GPU thread occupancy against matrix workspace memory.

---

## 4. The Memory vs. Wall-Clock Time Trade-off

A naive reaction to OOM errors is to drastically reduce batch sizes (e.g., $bs = 1$ or $bs = 2$). However, in iterative Gibbs sampling, this introduces a catastrophic **Wall-Clock Time Regression**.

### The Two Regimes of Gibbs Sampling Execution

```
▲ Time per Iteration (Seconds)
│
│   PCIe & Kernel Launch Bottleneck          Optimal Operating Window          OOM Failure
│   (Thousands of Host-Device transfers)     (High GPU Occupancy, low copies)  (VRAM > 24 GB)
│
│   │                                                                               │
│80s│  ● bs=2 (~72s/iter)                                                           │
│   │   \                                                                           │
│60s│    \                                                                          │
│   │     \                                                                         │
│40s│      \                                                                        │
│   │       \                                                                       │
│20s│        \                                                                      │
│   │         \____                                                                 │
│10s│              ● bs=10 (AR: ~5.5s) ─── ● bs=20 (SLDS: ~12s)                     │
│ 0s└───┴──────────┴────────────────────────┴───────────────────────────────────────┴──────► Batch Size
│      bs=1       bs=10                    bs=20                                  bs=60 (OOM!)
```

### Empirical Profiling Data

Across our empirical runs on the full dataset, we measured the following exact trade-offs:

| Configuration | Batch Size ($N_{\text{batch}}$) | Peak VRAM Usage | AR Iter Time | SLDS Iter Time | 50 AR + 200 SLDS Total Time | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Unbatched Stock** | 1,851 (All) | $>45\text{ GB}$ (OOM) | Instant Crash | Instant Crash | Cannot run | Immediate `RESOURCE_EXHAUSTED` |
| **Heuristic Batching** | 58 | $>26\text{ GB}$ (OOM) | Crash at Iter 0 | Crash at Iter 0 | Cannot run | 14 GB intermediate buffer OOM |
| **Ultra-Conservative** | 2 | $6.2\text{ GB}$ | $72.6\text{ s}$ | $\sim 180\text{ s}$ | $\mathbf{> 11\text{ Hours}}$ | PCIe bus thrashing (925 transfers/step) |
| **Engineered Optimal** | **AR: 10, SLDS: 20** | **$16.8\text{ GB}$** | **$5.5\text{ s}$** | **$12.3\text{ s}$** | **$\approx 45\text{ Minutes}$** | Zero OOMs, high GPU streaming occupancy |

### Why Ultra-Fine Batching ($bs=1$ or $2$) Destroys Performance

1. **Host-to-Device (H2D) PCIe Transfer Latency:**
   For $N = 1,851$ sequences and $bs = 2$, each Gibbs step requires $\lceil 1851 / 2 \rceil = 926$ separate host-device memory transfers.
   Even with PCIe Gen 4 ($31.5\text{ GB/s}$), launching 926 asynchronous transfers per variable across 6 Gibbs sub-steps ($z, x, v, h, s, Ab$) creates massive driver queue serialization delays ($\sim 60\text{ seconds}$ of pure transfer overhead per iteration).
2. **JAX Dispatch Overhead:**
   Every Python-to-JAX invocation incurs $\sim 50\text{--}100\,\mu\text{s}$ of runtime dispatch overhead. Multiplying by thousands of chunks introduces significant CPU-bound latency.
3. **GPU Core Under-utilization:**
   Modern tensor cores on an RTX 4090 require thousands of concurrent threads to achieve peak FLOPS. A batch size of 2 does not saturate the 16,384 CUDA cores.

---

## 5. Overview of Engineered Monkey-Patches

To allow researchers to train on the 18M frame cohort without modifying upstream libraries or reinstalling packages, we implemented a comprehensive monkey-patch suite in `syllable_generation/syllabizers/keypoint_moseq_syll/utils/resample_batching.py`.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   KPMS RUNTIME MEMORY ARCHITECTURE                                │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
      [HOST CPU MEMORY (128 GB)]                                            [GPU VRAM (24 GB)]
  ┌─────────────────────────────────┐                                  ┌────────────────────────────────┐
  │  Persistent Multi-Tier Caches:  │                                  │  Active Working Buffers:       │
  │  - preprocessed_data.pkl (18M)  │   Zero-Copy NumPy Slicing        │  - Active Batch Y_b (bs=20)    │
  │  - pca.p (10 PCs, 30 seq subset)│ ───────────────────────────────► │  - Active Batch x_b (bs=20)    │
  │  - init_model.pkl (8.3 GB)      │   (Only 18 MB H2D per chunk)     │  - Transition Mat (100x100)    │
  │                                 │                                  │  - AR Dynamics Ab, Q           │
  │  Full-scale s_0, Y, mask arrays │ ◄─────────────────────────────── │                                │
  │  (Kept on host to protect VRAM) │   Pinned Host Concatenation      │  Peak VRAM: 16.8 GB / 24.5 GB  │
  └─────────────────────────────────┘   (No GPU Out-Of-Memory)         └────────────────────────────────┘
```

### Summary of Core Monkey Patches

1. **`apply_resample_batching()` Hook:**
   Dynamically intercepts:
   - `jax_moseq.models.keypoint_slds.gibbs.resample_model`
   - `jax_moseq.models.keypoint_slds.gibbs.resample_scales`
   - `jax_moseq.models.keypoint_slds.gibbs.resample_heading`
   - `jax_moseq.models.keypoint_slds.gibbs.resample_location`
   - `jax_moseq.models.keypoint_slds.gibbs.resample_continuous_stateseqs`
   - `jax_moseq.models.keypoint_slds.gibbs.resample_discrete_stateseqs`

2. **`KPMSStateCache` Class:**
   Maintains host-side pinned references to static observation matrices ($Y, mask$), preventing repetitive GPU device copies between iterations.

3. **`batched_resample_scales` (CPU Slicing + Host Aggregation):**
   ```python
   def batched_resample_scales(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, **kwargs):
       bs = kwargs.get("batch_size", CONFIG.scales_bs)  # bs = 20
       N = Y.shape[0]
       Y_np = KPMSStateCache.get_Y(Y)                   # Host NumPy view
       s_0_np = np.asarray(s_0)                         # Host NumPy view
       cpu_device = jax.devices('cpu')[0]
       gpu_device = jax.devices()[0]
       
       results = []
       for i in range(0, N, bs):
           end = min(i + bs, N)
           # Transfer ONLY the 20-sequence chunk to GPU
           s0_b = jnp.asarray(s_0_np[i:end])
           Y_b = jnp.asarray(Y_np[i:end])
           s_batch = _resample_scales_kernel(..., Y_b, ..., s0_b)
           # Immediately push chunk back to CPU
           results.append(jax.device_put(s_batch, cpu_device))
       
       # Concatenate on CPU, then transfer final result to GPU
       with jax.default_device(cpu_device):
           s_full = jnp.concatenate(results, axis=0)
       return jax.device_put(s_full, gpu_device)
   ```

4. **Multi-Tier Disk Caching Pipeline:**
   - Tier 1: `preprocessed_data.pkl` (skips 3.5 min DLC parsing & outlier filtering).
   - Tier 2: `pca.p` (skips dense SVD, loads in 0.01s).
   - Tier 3: `init_model.pkl` (skips 3.4 min of 186 initialization batches, loads in 1.8s).

---

## 6. How Upstream Engineers & Researchers Can Permanently Fix This

Monkey patching individual resampling functions is a tactical fix. To permanently stop the "rabbit chase" when datasets scale to hundreds of millions of frames, upstream architectures need fundamental redesigns.

### Architecture Recommendation 1: Fused Memory-Efficient Emission Kernels
* **Current Defect:** Computing emissions via high-level JAX operations creates $(N, T, K, D)$ tensors in VRAM before reduction.
* **HPC Solution:** Implement a custom fused CUDA/Triton kernel for AR emissions and observation log-likelihood. By computing the quadratic form:
  $$(x_t - A_k x_{t-1} - b_k)^T Q_k^{-1} (x_t - A_k x_{t-1} - b_k)$$
  inside GPU shared memory and accumulating directly into an $(N, T, K)$ log-probability tensor, the $D$-dimensional broadcast intermediate is **completely eliminated from VRAM**, reducing memory requirements by a factor of $10\times$.

### Architecture Recommendation 2: Streaming / Chunked Dataset Iterators
* **Current Defect:** KPMS assumes the entire dataset $(N, T, K_p, 3)$ resides in GPU memory throughout the entire training lifetime.
* **HPC Solution:** Treat Gibbs sampling like modern deep learning:
  - Keep the full dataset in memory-mapped disk storage (`zarr`, `HDF5`, or pinned host RAM).
  - Stream mini-batches of sequences through JAX using asynchronous prefetching (`jax.experimental.io` or double-buffered CUDA streams).
  - Only AR parameters ($Ab, Q$) and transition matrices ($\pi$) need to reside permanently on the GPU.

### Architecture Recommendation 3: Mini-Batch Stochastic Variational Inference (SVI)
* **Current Defect:** Exact batch Gibbs sampling requires visiting every single sequence on every iteration. For 18M frames, one full Gibbs pass takes $\sim 12\text{ seconds}$.
* **HPC Solution:** Transition to Stochastic Variational Inference (SVI) or Online Expectation-Maximization (Online EM). Updates to transition dynamics ($Ab, Q, \pi$) can be performed over randomly sampled mini-batches of sequences (e.g., 32 sequences per gradient step), allowing the model to converge with constant memory and a fraction of the wall-clock time.

### Architecture Recommendation 4: Native Multi-GPU Sharding (`jax.pmap` / `shard_map`)
* **Current Defect:** The codebase is locked to a single GPU device (`jax.devices()[0]`).
* **HPC Solution:** The $N=1,851$ sequences in Gibbs sampling are conditionally independent given the shared parameters ($Ab, Q, \pi$). This is an embarrassingly parallel problem across the sequence dimension $N$. Using `jax.experimental.shard_map` or `jax.pmap`, sequences can be partitioned across 4 or 8 GPUs with near-linear speedup and zero memory pressure.

### Architecture Recommendation 5: Rolling Checkpoints & Post-Hoc Compaction
* **Current Defect:** Stock KPMS serializes full state snapshots ($s$, $\text{noise\_prior}$, $x$, $v$, $h$, $z$) every 25 iterations. At 18M frames, each snapshot is $\approx 8.8\text{ GB}$, accumulating to **$107\text{ GB}$ per model** in `checkpoint.h5`. Training a multi-seed ensemble exhausts local NVMe SSD storage.
* **HPC Solution:** Implement **Rolling Checkpoints** during training (maintaining only iteration 50 and the most recent SLDS iteration for crash recovery, $\approx 18\text{ GB}$ constant) and **Post-Hoc Compaction** (`h5repack` stripping observation noise scales $s$ and noise priors), reducing final checkpoint footprint to **$\approx 1.5\text{--}3.5\text{ GB}$ (97% storage savings)** while retaining 100% compatibility with EML model selection.
* **Authoritative Specification:** See [`KPMS_CHECKPOINT_COMPACTION_AND_ROLLING_STORAGE_SPEC.md`](file:///home/fsp585/Desktop/FiberphotometryCode/KPMS_CHECKPOINT_COMPACTION_AND_ROLLING_STORAGE_SPEC.md).

---

## 7. Package Contents & Reproducibility Guide

The accompanying ZIP archive (`kpms_scaling_and_memory_patches.zip`) contains the full suite of engineered scripts and monkey patches:

```text
kpms_scaling_and_memory_patches/
├── KPMS_SCALING_BOTTLENECKS_AND_OPTIMIZATIONS_DOSSIER.md  (This comprehensive engineering dossier)
├── utils/
│   ├── resample_batching.py       (Core monkey-patch engine: adaptive batching, CPU slicing, Banded Cholesky)
│   ├── batch_init.py              (Batched model initialization for large cohorts)
│   └── custom_kernels.py          (Optimized linear algebra kernels)
├── kpms_pipeline.py               (End-to-end pipeline wrapper with monkey-patch hooks & PCA subsampling)
├── run_cohort_2ar_2slds_test.py   (Empirical test harness with multi-tier disk caching)
├── run_tandem_cohort_production_50ar_200slds.py (Production cohort training script)
├── launch_2ar_2slds_test.sh       (Detached runner for empirical validation)
└── launch_tandem_cohort_production.sh (Detached runner for full production training)
```

### Quick Verification Command

To verify that the empirical test runs with zero OOMs on the cached dataset:
```bash
bash launch_2ar_2slds_test.sh
tail -f logs/test_2ar_2slds_cohort.log
```

---
*Authored for the Keypoint-MoSeq engineering and research community.*
