# Keypoint-MoSeq at cohort scale: engineering and research dossier — revision 2

**Prepared:** 29 September 2026  
**Audience:** engineers, Keypoint-MoSeq/JAX contributors, computational researchers  
**Purpose:** make the existing speed and memory work reviewable, establish what remains uncertain, and replace repeated OOM-driven patching with a predictable scaling workflow.

## September 29 auxiliary update: what changed and what to test first

**Status:** source audit and incident interpretation; no access to the original machine or GPU run. This review updates the earlier dossier and takes precedence where the new source changes behavior. The supplied incident report is preserved unedited in `auxiliary_bundle/KPMS_SCALING_ADDENDUM_LIVE_CRASH_ANALYSIS.md`.

### The short answer

The earlier dossier covered full-array residency, intermediate peaks, misleading timings and the need for multi-iteration validation. The new archive supplies a real implementation change—CPU output for the large-batch scales sampler—and a reported failure on a later discrete-emission call. It makes the investigation more specific.

**Do not yet label the cause “proven BFC fragmentation.” Test the repeated `fit_model` entry first.** The bundled reference calls a utility that transfers the model tree to the device. That can undo the new host-residency policy for `s` whenever fitting is entered again. The reported failing line is exactly the runner’s second SLDS `fit_model` call. This is a concrete, source-supported hypothesis; its contribution to the actual OOM remains to be measured.

### 1. Exact archive comparison

The auxiliary archive has 101 files: the prior 100 paths plus one new report. **Four existing files changed; none were removed.** Both revisions are preserved in this handoff. `AUXILIARY_DIFF.patch` is an exact unified comparison, not a proposed fix.

| Changed file, relative to `auxiliary_bundle/` | Actual change |
|---|---|
| `core_patches/resample_batching.py` | Large-batch scales returns a NumPy array on the host. Heading and location explicitly slice host `s` and transfer each slice. |
| `production_and_test_runners/run_cohort_2ar_2slds_test.py` | Cache restoration keeps `s` on the host; adds `--skip-ar` and `--num-slds`; changes phase labels and defaults AR time to 5.5 seconds when skipped. |
| `production_and_test_runners/run_tandem_cohort_production_50ar_200slds.py` | Cache restoration keeps `s` on the host. |
| `production_and_test_runners/launch_2ar_2slds_test.sh` | Forwards command-line arguments to the Python runner. |

The emission implementation, batch planner, initialization routine and allocator settings did not change. In particular, the auxiliary code still defaults/caps `discrete_bs` at 10; it does not implement the report’s recommendation to use 5. The host-scales change does not cover the `N <= bs` fast path or initialization’s final full-device placement. Nor does it establish residency through upstream wrappers or every downstream consumer.

### 2. What the new run supports

The addendum reports successful complete SLDS work followed by an OOM at `_jit_compute_discrete_emissions`, requesting **2,414,584,344 bytes (2.415 GB; 2.249 GiB)**. It reports 74.03 seconds cold and 46 seconds warm. No separate raw run log, profiler trace, allocator dump or installed-version export was added; the traceback is embedded in the prose report.

This is useful evidence of progress and a later failure. It is not yet a demonstrated stable throughput, a proven allocator diagnosis, or proof of the exact tensor composition of the allocation. The continuous solver in this patch remains a **CPU Numba** implementation; a successful hybrid SLDS sweep should not be described as every sampler running on the GPU.

The new 46-second figure replaces the earlier unsupported 12.3-second tandem figure as the **new reported timing to investigate**, not as a validated planning constant. Resolve the iteration/timing issues below before extrapolating either number.

### 3. First hypothesis: re-entering fitting reuploads host states

Source trail:

1. The new scales sampler returns host NumPy `s` at the end of a large-batch sweep.
2. The test runner invokes `kpms.fit_model` once at line 250 and again at line 267.
3. The unchanged bundled `diagnostics_and_benchmarks/fitting_reference.py` applies `device_put_as_scalar(model)` before its sampling loop.
4. A local reference implementation in `diagnostics_and_benchmarks/profile_kpms_breakdown.py` transfers the tree via `jax.device_put` before scalar conversion. Published upstream utility and fitting sources corroborate this pattern [U1, U2].

**Inference, conditional on matching installed source and a GPU default device:** a host NumPy `s` can return to the GPU at the start of each fit call, before the next `z` computation. A NumPy `noise_prior` in the model is another candidate. An already device-committed CPU array may behave differently; inspect actual types and placement rather than assuming all host representations are equivalent. Old model references and cache-held sources may add simultaneous live arrays.

This does not establish why the first call fits and the second fails. It identifies a specific placement boundary to instrument. At the stated FP64 shape, moving a full `s` adds approximately 3.416 GB of device payload, easily large enough to matter to a 2.415 GB scratch request.

**Decisive experiment:** from the same initial checkpoint and RNG state, compare one continuous call against split calls that perform the same counted sweeps. First do this on small data with checkpointing disabled; then use the failing shape if budget permits. Log leaf type, dtype, bytes and device before fit entry, immediately after its conversion, after each sampler, on return, and on re-entry. Specifically track `s`, `noise_prior`, `x`, `z` and cache-held source arrays. Count H2D bytes where available. A single call is a diagnostic comparison, not a complete architectural fix.

If uploads are confirmed, implement selective placement at the wrapper boundary and verify every use of host `s`/prior. Simply moving `s` to the CPU immediately before calling a wrapper that uploads it again cannot enforce the intended policy. Preserve scalar conversion, seeds, error checks and sampling order when adapting the wrapper.

### 4. The test harness has measurement and control defects

These findings come from the uploaded code. Confirm that the live files match these bytes.

| Finding | Consequence | Required correction |
|---|---|---|
| `args.num_slds` appears in the banner, but sampling remains two fixed `fit_model` calls | `--num-slds 20` does not request 20 counted sweeps | Make the argument control real execution; reject nonpositive counts; test counts on a tiny fixture. |
| Bundled `fit_model` uses an inclusive iteration range and resamples at every index | `start_iter=0, num_iters=1` executes two sweeps in that reference; the next `(1,2)` call also executes two | Inspect installed semantics and count sampler entries independently of labels. Keep displayed index, completed-sweep count and call ID distinct. |
| Failure trace names runner line 267 | This is the second fit call; its “warm” elapsed-time print is after return | The claimed warm print and failure cannot both originate from completion of that same call. Obtain raw logs; 46 s may instead be an internal/progress timing or another run. |
| Both fit calls pass `save_every_n_iters=0` | In the bundled reference this still creates/updates checkpoints and saves at a call’s final iteration | Use the installed API’s actual no-save option for a kernel timing experiment. Here the bundled reference and public docs specify `None`. Measure saving separately. |
| `t_ar_cold` is only assigned when AR runs, but is printed unconditionally | A successful `--skip-ar` run can later raise `UnboundLocalError` in its summary | Report AR metrics as not measured when skipped; do not fabricate a measured AR time from 5.5. |
| Cache restoration prints “0.5s” as literal text | This is not a timer result | Time and synchronize the operation; label disk read, CPU copying and device transfer separately. |
| Python sets memory fraction to `0.75` unconditionally before importing JAX | Exporting `0.85` in the shell does not change this runner’s value | Make configuration explicit and log the effective settings before backend initialization. |

The inclusive loop and checkpoint behavior are visible in the bundled reference, with a matching public implementation [U2]. This remains subject to installed-version verification. Do not silently halve the reported timings: compilation, checkpointing, labels and the raw timeline must first be reconciled.

### 5. Why the fragmentation explanation is not established

An allocation failure provides an allocation size and a failing call. It does not, by itself, distinguish live-array pressure, overlapping work, duplicate state copies, growing executable caches, another GPU process, or allocator fragmentation.

The supplied runners already request `TF_GPU_ALLOCATOR=cuda_malloc_async` before importing JAX. JAX documents this as an experimental alternative to BFC with different allocation/fragmentation behavior [J1]. Verify which allocator the installed runtime actually uses; neither the environment string nor `tf-allocator-allocation-error` proves it. Recommending the same flag again is not a new experiment. Raising a shell memory-fraction value is also ineffective while the script overwrites it.

The report’s exact scratch decomposition is unsupported. With `B=10, T=10030, L=3, K=100, D=10`, one `(B,T-L,K,D)` array is **401,080,000 bytes in FP32** or **802,160,000 bytes in FP64**. Three such FP64 arrays total **2,406,480,000 bytes**, close to the requested size; this illustrates ambiguity, not proof of three arrays. Record actual operand/intermediate dtypes and compiler allocation data. Do not invent an alignment workspace to make a byte count match.

JAX recommends XProf for device-memory analysis; synchronized device-memory snapshots can identify retained arrays/executables, although JIT internals are opaque in that view [J2]. Pair these with compiled memory/buffer analysis when the installed backend exposes it [J3], and with a device-usage timeline. A memory snapshot alone does not prove a fragmented free-block layout.

### 6. A short experiment sequence for this incident

| Order | Experiment | What it resolves |
|---|---|---|
| A | Verify installed fitting/utility source; fix iteration count, skipped-AR summary and no-save controls on tiny fixtures | Whether labels correspond to actual work and the harness is capable of a soak test. |
| B | Observe residency before/after fit-entry conversion and compare one-call versus split-call execution | Whether the host-state optimization is undone at re-entry. |
| C | Track synchronized live arrays/executables, GPU totals and host RSS across at least five real completed sweeps, with a resource/time stop | Whether memory rises, stabilizes, or spikes at a particular boundary. This is an investigation gate, not production certification. |
| D | If needed, compare serial emission batches to the existing overlapped pipeline at fixed `B=10`, then separately try `B=5` | Whether simultaneously live emission batches or per-batch scratch explains the remaining peak; quantify the actual runtime trade-off. |
| E | Only if evidence still points to allocation policy: fresh-process allocator A/B with identical shapes, code, counted sweeps and residency | Whether allocator choice changes failure after controlling the more direct causes. |
| F | If emissions still dominate: prototype tiled/algebraically reorganized emissions and compare likelihoods, peak memory and whole-iteration time | Whether a custom kernel is justified; start with the smallest implementation that answers the question. |

Keep batch settings, dtype, checkpoint and RNG fixed when comparing execution boundaries or allocators. The existing planner can overwrite a manually edited default batch size; set and log controlled overrides after planning, or change planner configuration explicitly in the experimental copy. Record rejected hypotheses as well as successful changes.

Reducing a batch from 10 to 5 is a legitimate controlled diagnostic/mitigation. It changes the number of sequence batches from 186 to 371, but that is not a measured transfer or synchronization count. Its effect on time and headroom must be measured. It does not establish scalable architecture, and adding two sessions does not deterministically imply the next failure.

A fused kernel may remove large intermediates, but neither an exact 10x reduction nor a 240 MB peak follows from this trace. Arbitrary minibatch parameter updates using an exponential moving average are not a drop-in exact Gibbs sampler; SVI and online EM are different inference approaches. Multi-GPU execution also requires measuring shared updates, reductions and replication. Keep those as separate options after the local failure is understood.

### 7. Immediate Antigravity instruction

Use the updated `ANTIGRAVITY_PROMPT.md`. Its first task is now to test fit-entry placement and reconcile iteration/timing semantics, before adding another memory flag or writing a custom kernel. Request the full raw log, exact run command, effective config, source hashes and installed fitting/utility source for the reported incident.

Do not claim stable 46 s/sweep or a solved cohort until correctly counted repeated sweeps, host-residency checks and save/reload/export all meet the agreed budget. The original dossier’s long-term objective—bounded memory through the whole lifecycle—still applies.

### Primary references checked on September 29, 2026

- **U1:** [JAX-MoSeq utility source](https://jax-moseq.readthedocs.io/en/stable/_modules/jax_moseq/utils/utils.html) — model-tree device conversion utility. Corroborates the local reference; not an export of the user’s installation.
- **U2:** [Keypoint-MoSeq fitting source](https://keypoint-moseq.readthedocs.io/en/latest/_modules/keypoint_moseq/fitting.html) — fit-entry conversion, loop and save behavior. The bundled `fitting_reference.py` provides the local evidence discussed above.
- **J1:** [JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html) — allocator choices and caveats.
- **J2:** [JAX device memory profiling](https://docs.jax.dev/en/latest/device_memory_profiling.html) — live-buffer/executable profiling and its limits.
- **J3:** [JAX ahead-of-time compilation](https://docs.jax.dev/en/latest/aot.html#debug-information-and-analyses-when-available) — backend-dependent compiled analyses.

---

## 1. The request and the current conclusion

The research pipeline repeatedly encounters a new bottleneck when the dataset grows. Fixing PCA exposes initialization; fixing initialization exposes emissions; smaller batches avoid one allocation but can increase transfers and dispatch overhead; improving the kernels exposes caching, checkpointing, or another full-array allocation. The goal is reliable training as the cohort expands, with scientifically defensible outputs and a measurable resource budget.

**The supplied patches already address substantial speed and memory problems. They do not yet make the full pipeline memory-bounded.** Several stages still concatenate full outputs, retain host/device copies, or place all model states on the GPU. The strongest engineering direction is therefore an explicit array-residency plan and bounded execution for every stage, backed by reproducible performance and scientific validation. A universal batch size or another isolated kernel patch cannot provide that guarantee.

This conclusion comes from static inspection of the supplied implementation. Runtime bottleneck rankings still require measurement on the actual machine.

### Evidence and scope

- **Source-observed:** present in the uploaded Python/shell files and inspectable without the original environment.
- **Reported:** stated in the supplied documents or transcript; not independently reproduced here.
- **Calculated:** arithmetic using explicitly stated shapes and dtypes; not measured peak memory.
- **Proposed:** fixes or experiments for Antigravity/the engineering team; not applied to the preserved code.

The auxiliary source snapshot contains 84 Python files and 2 shell launchers. Both it and the previous snapshot are preserved in the ZIP, including experimental and older implementations. No claim is made that this captures every modification in the live repository or installed packages. The user’s CLI, GPU environment, data, and running jobs were unavailable. Only source inventory, syntax, hashes, and archive integrity were checked here. The historical documents may describe different code revisions.

## 2. Workload and corrected dimensions

The target application combines top/bottom camera keypoints into a fused pose representation, trains AR-HMM and SLDS phases, and later aligns behavioral syllables with kinematics and photometry. Sequence boundaries, missing-frame handling, and frame-index mappings are therefore part of scientific correctness.

| Dataset/context | Reported scope | Interpretation |
|---|---|---|
| Jonathan AMPH | 30 sessions; 562 sequences; 5,516,954 active frames; 8 keypoints | Source of the detailed September 7 timing report; not the full tandem cohort. |
| Earlier Karen scaling report | 1,004 sequences; about 10.07 million frames | Separate scaling milestone; its timings are not directly comparable to AMPH or tandem. |
| Tandem full cohort | 118 available sessions; 24,111,516 frames before the reported filtering stages | Investigative notes describe 6 empty recordings/chambers and 112 active entries; confirm the dataset manifest’s exact definition of “session.” |
| Tandem formatted training data | 1,851 sequences; sequence length 10,030; 23 fused **2D** keypoints; 17,953,523 active frames | Reported tensor shape `(1851, 10030, 23, 2)`; confidence is supplied separately. |

The supplied `pure_numba_resample_continuous_stateseqs` explicitly asserts `d == 2`. The historical top-level dossier’s `(x, y, conf)` coordinate dimension of 3 is inconsistent with this implementation and the cohort investigation. Fusing two camera views here does not establish a reconstructed 3D pose model.

For `N=1851, T=10030`, the padded sequence grid has **18,565,530 slots**, not 17,953,523. The difference is 612,007 slots (3.30% of the padded grid), assuming the reported active count applies to this exact formatting. The active-frame count is a mask sum; allocation sizes follow the padded shapes.

The notes describe an RTX 4090 with 24,564 MiB VRAM and a host with roughly 128 GB RAM and 32 cores. Exact CPU topology, usable RAM, competing processes, package versions, CUDA/driver versions, and allocator behavior need a fresh environment report.

### Calculated array payloads

These numbers use decimal GB and binary GiB separately. FP64 is the planning case because the cohort runner calls `convert_data_precision(data, np.float64)` and core CPU kernels use float64. **Log actual JAX and NumPy dtypes at runtime**; the supplied files do not establish the effective x64 configuration in every environment.

| Array | Shape | Dtype used for calculation | GB | GiB |
|---|---|---|---:|---:|
| Observations `Y` | `(1851,10030,23,2)` | float64 | 6.832 | 6.363 |
| Each of `s`, full `noise_prior`, full confidence | `(1851,10030,23)` | float64 | 3.416 | 3.181 |
| Latents `x` | `(1851,10030,10)` | float64 | 1.485 | 1.383 |
| Location `v` | `(1851,10030,2)` | float64 | 0.297 | 0.277 |
| Heading `h` | `(1851,10030)` | float64 | 0.149 | 0.138 |
| Discrete states `z` | `(1851,10027)` for 3 lags | int32 | 0.074 | 0.069 |
| Mask | `(1851,10030)` | bool | 0.019 | 0.017 |
| Full emission table | `(1851,10027,100)` | float64 | 14.848 | 13.828 |
| One emission batch, `B=10` | `(10,10027,100)` | float64 | 0.080 | 0.075 |
| One broadcast emission intermediate, `B=10` | `(10,10027,100,10)` | float64 | 0.802 | 0.747 |
| Banded precision workspace, `Bx=50` | `(50,40,100300)` | float64 | 1.605 | 1.495 |

The five full state arrays alone total approximately **5.421 GB** in this dtype scenario. This excludes observations, priors, confidence, caches, parameters, old outputs, concatenation copies, workspaces, compiler/allocator overhead, and other processes. Some live on the host, some on the device, and some on both. Summing all payloads into one “GPU footprint” would be misleading without residency evidence.

At `B=20`, an FP64 observation chunk is approximately **73.8 MB** and a scale/prior chunk **36.9 MB**. The historical “18 MB per chunk” statement does not describe these shapes and dtypes.

References: `original_bundle/docs/kpms_tandem_cohort_investigation.md`; `original_bundle/production_and_test_runners/run_cohort_2ar_2slds_test.py`; `original_bundle/core_patches/resample_batching.py`.

## 3. Complete map of the speed and memory work

The latest supplied implementation is `auxiliary_bundle/core_patches/resample_batching.py`; the previous version is preserved under `original_bundle/`. Its real activation function is **`apply_optimizations()`**, with `restore_original_kernels()` and the `kpms_optimizations()` context manager. The historical `apply_resample_batching()` name is not an entry point in the supplied file.

| Component / symbol | Speed mechanism | Memory mechanism and remaining limit |
|---|---|---|
| `BatchConfig`, `precompute_optimal_batch_config` | Avoid repeated manual configuration; estimate batch sizes before work | Formula-based scratch estimates with caps: discrete 10; scales/heading/location 20; continuous 50. A heuristic, not a complete peak-memory model. |
| `KPMSStateCache` | Reuses CPU copies of `x`, `z`, `mask`, `Y` | Avoids repeat conversions; retains source references and host copies. Does not explicitly allocate page-locked/pinned host memory. |
| `_jit_compute_discrete_emissions` | JIT GPU emission calculations | Processes sequence batches; still expresses `(B,T-L,K,D)` intermediates. Actual materialization/fusion requires compiler or profiler evidence. |
| `fast_resample_discrete_stateseqs` | GPU emissions plus parallel CPU Numba FFBS; prelaunches the next emission batch and requests async host copy | Two host emission slots plus filter workspace bounded by batch size; retains full `z` output and transfers it to the device. Actual overlap is unmeasured here. |
| `_zero_alloc_accumulate_ar_stats`, `vectorized_resample_ar_params` | Numba sufficient-statistics accumulation; batched JAX MNIW posterior solves | Avoids full lagged-design/one-hot arrays; thread-local stats still allocate. “Zero allocation” and whole-pipeline O(1) memory would be overstatements. |
| `pure_numba_resample_continuous_stateseqs` | CPU Numba banded precision assembly, Cholesky and Gaussian sampling | Chunked band matrix/workspaces; allocates full `x_out` and returns full device `x`. Supports 2D only. |
| `_numba_fused_rigid_transform` | Fuses subtraction and 2D rotation into a preallocated buffer | Reduces intermediate arrays and memory passes in the continuous step. |
| `batched_resample_scales` | JIT scale kernel | NumPy host slicing of full `Y` and array-valued prior; CPU output collection/concatenation; the auxiliary large-batch path returns host NumPy output. The small-batch path and wrapper placement still need review. |
| `batched_resample_heading`, `batched_gpu_resample_location` | Batched JIT calls to captured upstream kernels | Host observation slicing and CPU output aggregation; full result is returned to the device. |
| `fast_consolidated_check_for_nans` | Reduces per-leaf host synchronization for GPU NaN checks | Detailed host inspection occurs on failure. Uses NaN checks, not a general finite-value check. |
| `batch_init.py::batched_init_model_by_video` | Initializes shared parameters on a small subset and states in batches | Offloads states to CPU, then concatenates and transfers **all** states to GPU; full prior is computed on CPU. |
| `kpms_pipeline.py`, cohort runners | PCA subset and disk caches avoid repeated startup work | PCA sampled before preprocessing; caches full preprocessed data and initialization. Loading/deepcopy/checkpoint peaks remain. |
| `trimming.py` | Parallel per-recording cleaning; worker setup targets CPU | Reduces retained frames; spawn-based workers and copied arrays can increase aggregate host RAM. |
| `custom_kernels.py` | Older scales-only JIT/batching patch | Fixed batch 50; retains GPU result list, passes full `s_0`, and uses a different seed shape. Historical alternative, not an additional layer to enable. |

The dispatcher replaces discrete/AR samplers in `jax_moseq.models.arhmm.gibbs` and the `arhmm` namespace; scales, heading, location, and optionally continuous sampling in `keypoint_slds.gibbs` and `keypoint_slds`; and the NaN checker in `keypoint_moseq.fitting`. It does **not** replace `resample_model` in the supplied implementation. Actual dispatch through installed upstream aliases and cached JIT functions still needs verification.

`BatchConfig.banded_solver` is declared with `numba`/`lapack` comments, but the supplied production implementation does not read it to select a backend. LAPACK, PCG, associative scan, mixed precision, whole-loop JIT, and other frontier scripts remain in `diagnostics_and_benchmarks/` for review; their existence does not mean they are enabled or accepted.

## 4. What the historical results do and do not establish

| Evidence | What is reported | What can be concluded from this handoff |
|---|---|---|
| `docs/MEASURED_GRANULAR_PROFILE_FULL_DATASET.md` | **AMPH**, 562 sequences: intermediate AR 5.57 s / SLDS 15.85 s; later AR 4.23 s / SLDS 11.65 s; about 8.97 GB VRAM | Useful historical hypotheses about where time went; no raw run artifact or current-revision replication is included. |
| Original top-level dossier and pasted transcript | **Tandem** summary: AR 5.5 s / SLDS 12.3 s; peak VRAM 16.8 GB; very small batches at 72.6 s AR | Treat as unverified claims requiring a cohort-matched run. Do not substitute AMPH profiling as evidence for these figures. |
| `docs/REAL_DATASET_SCALING_AND_REALIZATIONS.md` | Earlier 1,004-sequence case: 79.77 s warmed SLDS | Different workload/revision; retain as historical context, not a current regression target. |
| `docs/BUG_REPORT_AND_PARITY_RESOLUTION.md` | Early broad parity claims; later explicit distinction between direct AR and upstream augmented-state model | Read the later qualification. Universal posterior or bitwise equivalence is not established. |
| `run_cohort_2ar_2slds_test.py` | Harness intended to measure cold and warm AR/SLDS | Executable intent, not proof the exact supplied revision completed. Its “loaded ... in 0.5s” cache message is hard-coded. |

The headline 45-minute forecast is arithmetic: `50×5.5 + 200×12.3 = 2735 s = 45.6 min`. It is not a measured production completion time and excludes startup, compile, checkpoint, export, and later-iteration variation. One warm iteration does not establish a stable distribution of runtimes.

### Corrections to retain when discussing the problem

1. **PCA failure:** the detailed investigation identifies full-array JAX interpolation/alignment before PCA frame selection. Do not attribute the whole event solely to dense SVD on all frames or to exhaustion of 128 GB host RAM without a trace.
2. **Slicing failure:** host-side NumPy slicing can avoid uploading a full operand before slicing. It does not prove that every JAX slice copies its whole input. Confirm the array’s device/sharding, source dtype, and compiled allocation at the failing call.
3. **14 GB allocation:** multiple tensor shapes/dtypes can produce similar sizes; the full FP64 `(N,T-L,K)` emissions above already approach 14.85 GB. Match a traceback to HLO/buffer metadata instead of assuming six particular broadcast buffers are live.
4. **“Pinned” / “zero-copy”:** the cache uses `np.asarray` and normal NumPy arrays. A NumPy slice may be a view; device transfer and dtype conversion are separate costs. The code does not explicitly establish pinned host storage.
5. **Top-10 PCA variance:** the runner prints `np.cumsum(pca.explained_variance_ratio_)[-1]` under a “Top 10” label. That sums every returned component. A true top-10 calculation must select the first ten; neither result proves held-out posture coverage.
6. **“Exact parity”:** changed random generators do not generally produce identical draws. Statistical agreement on selected summaries is not proof of identical conditional distributions; see the scientific checks below.
7. **Future architectures:** fused kernels reduce specific intermediates, not every allocation by a fixed factor. Multi-GPU speedup is workload-dependent. SVI/online EM changes inference and cannot be promised to converge faster or preserve Gibbs results.

## 5. Why the next scale increase can still fail

### Persistent arrays and transient peaks are separate problems

For a batch of sequences, reducing `B` can bound intermediate work. It does not bound `N`-sized model states. In the supplied source:

- Batched initialization collects all chunks, concatenates them, and transfers full `x/v/h/s/z` to the GPU.
- Scales, heading, and location collect CPU result chunks and concatenate them. In the auxiliary revision, the large-batch scales path retains the result on the host; heading/location still return full GPU arrays. Fit-entry conversion may reupload host model leaves; see the auxiliary review.
- Continuous sampling allocates `x_out` for every sequence and makes a device copy at return.
- The state cache may keep both host copies and source arrays alive.
- Cache loading retains `cached_model`, creates `copy.deepcopy(cached_model)`, and then restores device states. Export/checkpoint code may create further copies.
- Preprocessing uses up to 16 workers, materialized task/result lists, and per-recording copies. Process RSS summed across workers must be measured separately from the parent.

The smallest safe batch can therefore still OOM. Conversely, tiny batches increase the number of launches, conversions, transfers, and synchronization points. “926 batches” is not the same as a measured count of 926 PCIe transfers, and total transfer overhead should be profiled rather than inferred from batch count alone.

### The planner is a starting heuristic

`precompute_optimal_batch_config` estimates several scratch terms using FP64 and hard caps. The cohort runner calls it before initialization without passing a model, so `D=10`, `K=100`, and `L=3` are assumed. It does not establish a complete accounting of persistent states, outputs, old outputs, asynchronous in-flight work, host caches, or checkpoints. It returns at least batch size one even if one sequence cannot fit the estimated budget. Smaller caps are not a capacity guarantee.

### Scaling variables matter differently

- Increasing `N` increases retained state and full-sweep work, even with fixed batch workspaces.
- Increasing `T` increases per-sequence emissions and smoother workspaces; sequence batching alone cannot solve an oversized single sequence.
- Increasing `K` increases emission storage and can increase the dense FFBS transition work approximately quadratically in `K`.
- Increasing latent dimension `D` or lag count `L` increases band width. With scalar bandwidth `W=(L+1)D-1`, band storage is proportional to `B×T×D×(W+1)`; banded factorization work also depends strongly on `W`.
- Increasing keypoints raises observation, confidence, scale, alignment and preprocessing costs. Increasing workers can amplify memory before model fitting begins.

The “rabbit chase” is a predictable result of optimizing one allocation without maintaining a stage-by-stage model of these costs.

## 6. Specific engineering issues to review first

These are source findings or targeted hypotheses, not applied fixes.

| Priority | Finding | Proposed action / acceptance evidence |
|---|---|---|
| P0 | Repeated `apply_optimizations()` overwrites `_ORIGINAL_FUNCTIONS['check_for_nans']` with the already-patched function | Capture the original only once; define nested-context behavior; verify apply/apply/restore and exception unwinding restore every original callable. Test in a fresh process. |
| P0 | `set_batch_sizes` casts every recognized override to `int`; string backend values fail and unrelated config fields are not validated | Validate only positive integer size fields there; use explicit typed setters for bool/backend options; reject unknown keys rather than silently ignoring them. |
| P0 | Preflight does not model full lifecycle memory or reject an impossible one-sequence case | Record actual shapes/dtypes/residency after initialization, account for simultaneous live buffers, and fail early with the limiting stage and shape. |
| P0 | File-existence caches have no visible dataset/config/code fingerprint checks in the cohort runner | Add manifest-based validation for frame order, masks, keypoints, preprocessing, PCA, dimensions, parameters, precision, versions, code and seed; preserve old caches during experiments. |
| P1 | Full-state concatenation/device returns and deepcopies scale with `N` | Profile liveness, then eliminate copies or implement bounded host-backed state/output stores, including initialization, checkpoints and extraction. |
| P1 | `N <= bs` fast paths bypass host-slicing paths | Cover one-sequence/one-batch/remainder cases and host-backed inputs explicitly; fast and batched paths need equivalent placement and shape contracts. |
| P1 | Shared parameters are initialized from the first five sequences; PCA uses evenly spaced sequence indices | Define and record representative sampling by animal/session/condition; compare stability and held-out reconstruction, especially rare postures. |
| P1 | Older tests may target earlier interfaces | Example: `test_optimized_kernels_standalone.py` passes `pi` to `_numba_ffbs_single_sequence`, while the current function expects `pi_T`. Update the reference orientation before treating a mismatch as a sampler defect. |
| P1 | Upstream fallback is selected by `use_banded_continuous=False` | The dispatcher restores upstream continuous sampling without wrapping it in the CPU chunking logic. A fallback can reintroduce full-cohort memory pressure; validate on small data first. |
| P1 | `banded_solver` field is not wired to production dispatch | Implement and test explicit selection if needed, or label it inactive; do not claim a LAPACK experiment used production dispatch merely because the field was set. |

A further source-review target is `vectorized_resample_ar_params` inferring `num_states` from `K_0.shape[0]` or a default before expanding nested hyperparameters. Confirm the installed upstream call signature and meanings of these shapes; require an explicit validated state count rather than assuming `K_0` encodes it.

## 7. Scientific checks that must accompany speed work

### Decide what posterior the continuous sampler should target

The direct Numba sampler assembles a banded AR-trajectory precision matrix. It includes an initial precision of 0.1 for the first lag frames, masked observation/dynamics heuristics of 0.1 precision, a diagonal `jitter` default of `1e-3`, and pivot repair to `1e-4` when a pivot is at most `1e-8`. These are mathematically relevant choices.

The historical parity report later states that upstream uses an augmented lag-state formulation with auxiliary lag noise and reports an installed prior-conditioning issue. Its diagnostic script constructs alternate Kalman samplers. **That does not establish the behavior of the currently installed upstream version, or show that an upstream fix is part of this bundle.** Export and inspect the actual installed source.

Define two distinct comparisons: (a) a backend implementing the *same specified conditional distribution*, and (b) a deliberately different model/inference variant. Record which is intended. Do not silently change the reference’s prior/noise/jitter simply to improve a parity number.

For small systems, build the intended dense precision/covariance and compare posterior means, covariances and solve residuals to the banded construction. Then use multiple random seeds for sampling tests with predeclared tolerances and Monte Carlo uncertainty. Record pivot repairs and inspect every nonzero count. Check fully valid sequences, leading/trailing padding, internal gaps, short sequences near the lag length, low confidence and extreme noise scales. Cross-generator equality of sampled paths is not the criterion.

### Masking and time continuity need cohort-specific review

`filter_with_masks` selects `coordinates[name][keep[name]]`: nonconsecutive retained original frames can become adjacent in the compressed input. A frame-index map preserves alignment for later lookup, but does not by itself preserve AR dynamics across the removed time interval. Audit discontinuities in `video_frame_indexes`, confirm whether formatting splits contiguous runs, and decide explicitly how temporal gaps should be represented.

“No internal missing frames” in a formatted AMPH mask cannot prove there were no removed recording gaps or that tandem data have the same missingness. Trailing padding can also influence valid boundary states through smoothing; excluding it from downstream summary tables is not a proof of zero modeling effect.

### Downstream acceptance

In addition to kernel tests, compare state usage, dwell-time distributions, transitions, held-out fit/reconstruction measures appropriate to the chosen model, and repeated-seed variability. Align arbitrary state labels before comparing syllables. Check that top/bottom synchronization, timestamps, frame indices, and photometry association are unchanged. Define a scientific effect-size tolerance before selecting the faster backend.

An SLDS-only run from an initialized model can be a useful isolated crash/performance test **if the model is valid and the installed API supports it**. It is not evidence that production AR warmup is unnecessary. Prefer a fixed, hashed post-AR checkpoint for representative SLDS benchmarking; verify iteration semantics, metadata arguments and hyperparameter updates against the installed API. The transcript’s abbreviated example should not be copied as a verified invocation.

## 8. A finite experiment plan with decision gates

Run each experiment in a new process with a named configuration and a fixed input/checkpoint. Keep original production outputs separate. Establish the warmed full-iteration baseline before optimizing individual kernels; compare per-kernel traces in separate profiling runs because forced synchronization can change overlap.

| Gate | Experiment | Required result / decision |
|---|---|---|
| 0 — Freeze and complete | Export repository revision/dirty diff, all active patch sources and installed upstream sources; record environment and dataset/config fingerprints | Know exactly what code ran. Resolve missing files and conflicting claims before extending the patch stack. |
| 1 — Correctness and lifecycle | Tiny synthetic fixture plus a few real sequences; patch/restore, masks, seeds, conditional moments, cache invalidation, checkpoint resume | Fix correctness failures before timing comparisons. Agree on the continuous model target. |
| 2 — Identify the limit | Stage markers from load/PCA/init through AR, SLDS, save, reload and extraction; record shape/dtype/device bytes and process-tree/GPU memory | Identify the highest peak and actual dominant warm costs. Distinguish allocator reservation, live bytes and observed device usage. |
| 3 — Controlled batch sweep | Sweep **one** family at a time: discrete `[2,5,10]`; pose/scales `[5,10,20]`; continuous `[5,10,25,50]`, subject to preflight | At least 5 warmed iterations for surviving candidates; record median, range, compile cost, peak memory and transfer evidence. Choose a feasible setting with explicit headroom, not just the fastest isolated call. |
| 4 — Test boundedness | Vary `N` at fixed `T,P,D,K,L`, then vary `T` independently; use synthetic shape fixtures for allocation stress and real subsets for fit checks | Plot peak memory against `N` and `T`; show which slopes remain. Synthetic duplication is only an allocation test, not new scientific evidence. |
| 5 — Prototype the measured fix | Start with the largest remaining full-array peak or measured bottleneck; compare an isolated implementation against the frozen baseline | Retain only changes that pass correctness and improve the relevant budget/end-to-end metric. Record rejected experiments and reasons. |
| 6 — Production readiness | After prior gates: a predeclared soak (e.g. 20 warm SLDS iterations), checkpoint/save/reload equivalence and downstream export, including a deliberately changed cache input | No unexplained memory growth, bounded save/load peaks, invalid cache rejected, agreed scientific checks passed. Then consider 50 AR + 200 SLDS. |

All counts and candidate batch sizes above are **proposed starting budgets**, not known optimal settings. Set a runtime and memory stop condition before each experiment. Stop a sweep after it answers its question; avoid repeated full-cohort retries and changing multiple knobs at once.

The benchmark record should include: run ID, code/source hashes, environment, dataset/checkpoint hash, phase, actual iteration count, all batch settings, random seed, shape/dtype/device ledger, cold compile time, synchronized warm wall times, main/process-tree RSS, device peak and measurement method, checkpoint/extraction times, numerical diagnostics, and outcome/error trace. Store full logs plus a compact JSON/CSV summary. Capture actual function identities/signatures after patching.

## 9. Architecture that can end the recurring chase

### First priority: a bounded execution contract

Write an ownership/lifetime table for every large array. Specify which data and latent state live in host-backed storage, which chunks are active on the device, how outputs are written without full concatenation, and the maximum number of in-flight buffers. Include ingestion, PCA, initialization, each Gibbs update, checkpointing and result extraction.

Implement a preflight model of the form:

`peak device ≈ persistent device arrays + largest simultaneous batch workspaces + old/new outputs + transfer overlap + measured runtime reserve`.

Build a separate host/process-tree model. Use actual dtypes and dimensions and calibrate against measured peaks. Return “does not fit” when even the minimum unit is oversized. A batch cap is not a replacement for this check.

For a fixed maximum sequence length and model dimensions, aim for device working memory that is approximately independent of total sequence count. Host or disk state storage and full-sweep work still grow with the dataset. Very long sequences require a separate temporal algorithm with correct boundary/message handling; arbitrary splitting into independent time chunks changes the model.

### Preserve inference semantics when streaming

Sequence-local conditional updates can be processed in chunks while shared parameters are held at the correct values for that Gibbs step. Global updates must aggregate the required full-data sufficient statistics before drawing shared parameters. Follow the actual Gibbs dependency order; do not train independent models per batch or update global parameters on a changing partial dataset and call it equivalent Gibbs sampling.

Initialization and output stores must follow the same bounded design. Validate deterministic sequence ordering, random-key strategy, restart behavior and cache invalidation. Test a host-backed prototype before adding complicated overlap or a custom kernel.

### Optimize only the measured remaining cost

If emission intermediates or transfer volume dominate, evaluate state/sequence tiling, algebraic contraction changes or a fused kernel against the exact likelihood reference. If CPU FFBS dominates, profile `K`-dependent transition work. If the banded solver dominates, compare controlled Numba/LAPACK/thread settings at identical model semantics. If checkpointing dominates, address serialization and duplicate arrays first.

Multi-GPU sequence distribution is a later option with explicit reduction and synchronization costs. SVI/online EM is a separate research project with different convergence and scientific validation requirements. Neither is needed as the first response to an unmeasured OOM.

### Definition of done

The team has a documented supported range of `N,T,P,D,K,L`; the preflight predicts the limiting stage within a stated empirical tolerance; fixed-shape sequence growth does not require manual per-kernel patching; every stage respects the memory budget including save/resume; performance scales predictably; and scientific acceptance checks pass against the declared model target. Dataset increases then become a measured capacity decision.

## 10. Reproducibility gaps and handoff map

The ZIP preserves all **100 files** from the first archive and all **101 files** from the auxiliary archive. Each snapshot has 84 `.py` and 2 `.sh` files; the auxiliary snapshot adds the incident report. `CODE_INDEX.md` indexes the latest snapshot; `BASELINE_CODE_INDEX.md` indexes the previous one. `SOURCE_MANIFEST.json` records hashes and provenance, and `AUXILIARY_DIFF.patch` records the four changed files. The two original text attachments are also preserved.

| Bundled location | Intended location in the original repository |
|---|---|
| `original_bundle/core_patches/kpms_pipeline.py` | `syllable_generation/syllabizers/keypoint_moseq_syll/kpms_pipeline.py` |
| Other `original_bundle/core_patches/*.py` | `syllable_generation/syllabizers/keypoint_moseq_syll/utils/` |
| `original_bundle/diagnostics_and_benchmarks/*.py` | `syllable_generation/syllabizers/keypoint_moseq_syll/diagnostics/` |
| Tandem/cohort/multiview runners and both launchers | `syllable_generation/multiview_top_bot_experiments/` |
| AMPH runners and `find_kappa.py` | `syllable_generation/syllabizers/keypoint_moseq_syll/` |
| `original_bundle/multiview_and_fusion/*.py` | `syllable_generation/multiview_top_bot_experiments/` |

This flattened review layout is not directly importable as the original project. Do not invent missing imports or replace them with dummy implementations for an end-to-end claim.

Items to request/export from the actual machine:

- Exact repository commit and working-tree diff; installed source and versions of Keypoint-MoSeq, jax-moseq, JAX/jaxlib, Dynamax, Numba, NumPy, SciPy, TensorFlow Probability and numerical runtimes; Python/driver/CUDA details; effective x64 and thread/allocator settings. Other scripts also use pandas, matplotlib, h5py, OpenCV, imageio, Pillow, PyYAML, psutil and threadpoolctl. This is an observed dependency list, not a tested lockfile.
- Missing local modules referenced in the supplied files, including `syllable_generation.datamodel`, `syllable_generation.analysis.replications.datta_utils`, and `utils.optimal_flip_list` for optimized flipping. Resolve any dynamic imports through a full-repository dependency audit.
- The referenced `nvidia_idle_watchdog.py`, absent from this archive; `diagnostics/run_parity_suite_live.sh`, cited in the parity report but not supplied. `diagnostics/cpu_ram_watchdog.py` is included and is a different script.
- Exact project config, bodypart order, coordinate transform/units, PCA and subset definition, noise estimator, hyperparameters, segmentation/mask policy, source-frame mappings, dataset manifest, representative fixture and hashed checkpoint.
- Raw full-cohort timings, memory traces, stack traces, compiled allocation evidence where available, and parity outputs tied to code revisions. The cache files and model/data artifacts referenced by the scripts are not included. The new incident report contains a traceback excerpt, but no separate complete run log or memory trace.

The original launchers contain machine-specific paths and background-job behavior; several Python runners send notifications. Use a separate experiment entry point with notifications disabled and a new output directory. No launchers or notification code were executed in preparing this handoff.

## 11. Ready-to-paste Antigravity prompt

This revision starts with the new fit-entry/residency and measurement investigation. It is also supplied as `ANTIGRAVITY_PROMPT.md`.

```text
# Antigravity: investigate and stabilize KPMS cohort scaling

You have access to my actual project and GPU environment. The accompanying handoff was prepared from uploaded files only; its author did not run my project. Please complete the engineering investigation below, using measured evidence and a finite experiment plan. The goal is to stop finding a new memory or speed problem every time the cohort grows.

## Objective and boundaries

Audit all currently active speed and memory patches, establish a reproducible baseline, and propose/implement the smallest changes that produce predictable memory use without silently changing the scientific model. Preserve existing production data, models, caches and results. Work in a separate branch/worktree or clearly isolated experiment area. Do not launch the full 50 AR + 200 SLDS production run during this investigation, kill unrelated jobs, or send notifications. Do not install newer dependencies into the production environment to make a test pass.

Start with `KPMS_ENGINEERING_DOSSIER.md`, especially its source findings, scientific qualifications, and missing-file list. Treat historical “zero OOM,” “optimal,” and “100% parity” claims as hypotheses. Use the actual installed source and actual array metadata as the authority for this machine.

## Priority update: investigate the second fit call before kernel work

The September 29 auxiliary archive changes four files and adds an incident report. Read `AUXILIARY_REVIEW.md` before the original plan. Keep both supplied revisions and their hashes; the archive diff is evidence, not a proposed patch.

First inspect the installed `keypoint_moseq.fitting.fit_model` and `jax_moseq.utils.device_put_as_scalar`. The bundled fitting reference converts the model tree to the device on every entry; the local utility reference and public upstream source support that behavior. The new scales patch returns NumPy `s` to the host, but the runner's second fit call may upload it again. Inspect NumPy `noise_prior` too. Treat this as a concrete hypothesis, not a proven explanation of the live crash.

Instrument before/after fit-entry placement, after every sampler, on return and on re-entry. Record types, shapes, dtypes, devices and bytes for `s`, `noise_prior`, `x`, `z`, and cache-held sources. From the same checkpoint and RNG state, compare one uninterrupted call with split calls performing the same independently counted sweeps, first on a small fixture. Preserve the actual Gibbs order and checks. If host arrays are reuploaded, fix selective placement in an isolated wrapper and validate every consumer; merely moving `s` before the wrapper is insufficient.

Repair the measurement harness first:
- Make `--num-slds` control the actual sweep count; currently it only changes the banner.
- Check inclusive loop semantics in the installed version. The bundled reference executes two sweeps for `(start_iter=0, num_iters=1)`, so the runner's labels may misattribute elapsed time.
- The incident names runner line 267, the second fit call. Its warm-time print is after return. Reconcile the 74.03/46-second claims against a complete raw log and actual sampler-entry timestamps; do not assume a measured stable 46 s/sweep or simply divide a number by two.
- Use the installed no-save setting for computation-only benchmarks. In the bundled reference `save_every_n_iters=0` still writes checkpoints, while `None` disables them. Use a separate output location for save/resume tests.
- Fix undefined `t_ar_cold` in the skipped-AR summary. Mark skipped AR timings as unmeasured, including the hard-coded 5.5-second fallback.
- Replace the fixed 0.5-second cache message with real measurements.
- Log the effective allocator/batch configuration. The runner already requests `cuda_malloc_async` and overwrites the memory fraction to 0.75; the planner can overwrite a default batch edit. Do not repeat the same flag as a new experiment.

After placement/counting are trustworthy, record synchronized live-memory snapshots and a device-usage trace across at least five counted sweeps within a declared time/memory limit. Then, if needed, compare overlapped versus serial emission dispatch at fixed batch size, and separately test a smaller batch. Change allocator only in a fresh, controlled process after checking installed support and removing configuration overrides. Retain allocator fragmentation as one hypothesis alongside live-state pressure, temporary copies, overlapping work and executable growth.

Request the raw incident log, exact command, source hashes and installed source. Capture compiled memory/buffer information if supported to identify the 2,414,584,344-byte request; neither a float32 decomposition nor an exact 10x fusion benefit is established. Do not alter inference to EMA minibatches to avoid diagnosing the current pipeline.

## 1. Freeze the baseline and finish the source export

1. Record repository commit, dirty diff, Python executable/version, package versions, installed source paths and hashes, GPU/driver/CUDA details, CPU topology, available host/device memory, competing workloads, x64 setting, Numba/BLAS thread settings and allocator environment.
2. Enumerate every active monkey patch and all imported aliases at the actual fitting call sites, including changes inside installed packages. Capture original and replacement module/file/signature, and identify which experimental code is actually enabled. Do not assume the uploaded ZIP contains everything from the live repo.
3. Export all required local source files omitted from the original bundle. In particular, inspect `syllable_generation.datamodel`, `syllable_generation.analysis.replications.datta_utils`, `utils.optimal_flip_list`, the NVIDIA watchdog, and the reported parity-suite launcher. Record optional versus required dependencies. Package the actual supported import layout with a lock/environment export and hashes.
4. Create a dataset/config/checkpoint manifest: `Y/conf/mask/x/z/v/h/s/noise_prior` shapes, dtypes, bytes and devices; active and padded frame counts; bodypart order; frame mappings; preprocessing/segmentation/PCA definitions; hyperparameters and random seeds. Confirm fused observations are 2D, not `(x,y,confidence)` coordinates.

Deliver this baseline record before claiming any reproduced timing or scientific equivalence. Keep source, reported history, measurement and hypothesis clearly separated.

## 2. Resolve correctness and lifecycle issues on small fixtures

Inspect and test these specific source findings against the live code:

- Repeated `apply_optimizations()` can replace the saved original NaN checker with the patched checker. Make installation/restoration idempotent and define nested context behavior. Test apply twice, restore, exceptions, model switching and all exported aliases in a fresh process.
- `set_batch_sizes` casts all recognized values to integers and lacks strict field/range validation. Validate positive sizes explicitly; do not silently accept typos or conflate backend/boolean settings with sizes.
- `banded_solver` is declared but apparently unused by the supplied production dispatcher. Confirm it, and either wire and validate a backend selector or mark/remove the misleading option.
- Confirm the AR-parameter wrapper’s installed calling convention and how `num_states` is determined; inspect its `K_0.shape[0]` fallback and nested hyperparameter handling.
- Correct the “top 10 PCs” metric if it sums every component, and measure cache-load time rather than printing the fixed “0.5s” message.
- Audit stale tests before interpreting them. The supplied `test_optimized_kernels_standalone.py` appears to pass `pi` where current FFBS expects contiguous `pi_T`.
- Test array placement and noise-prior shapes on both `N <= batch_size` and chunked paths, including partial final batches.

Choose and write down the continuous sampler’s intended conditional distribution. Compare the banded precision implementation to an independently constructed dense reference on tiny problems. Inspect initial prior conditioning, auxiliary lag-copy noise, masking, jitter and pivot repairs in the actual upstream version. A modified diagnostic reference is not automatically the upstream production reference.

Use posterior means/covariances and numerical residuals, then multi-seed sampling checks with stated Monte Carlo uncertainty; do not require identical random draws across different generators. Include internal gaps, leading/trailing padding, short sequences and extreme scales. Surface nonzero pivot repairs. Keep model changes separate from implementation-equivalent optimizations.

Audit `video_frame_indexes` for discontinuities after mask compression. Determine whether retained frames across a deleted interval become adjacent in the AR model, and whether formatting splits contiguous runs. Verify downstream timestamp/photometry alignment. Do not infer temporal correctness from a formatted mask with no internal gaps.

## 3. Measure the complete lifecycle

Create an isolated benchmark entry point with no production writes or notifications. Use a fixed, hashed fixture/checkpoint and a fresh process per configuration. Confirm `fit_model` iteration semantics and required metadata against the installed version.

Instrument load, preprocessing, PCA preprocessing/fit, initialization, AR update, each SLDS component, checkpoint save, reload and extraction. Record actual large-array residency and lifetimes, process-tree RSS including workers, observed GPU memory and the measurement method. Separate reserved memory from live payload and total observed usage. Capture the full traceback and allocation metadata for a failure.

Time compilation separately. Synchronize relevant outputs for end-to-end timing; use separate intrusive kernel-profiling runs so synchronization does not erase the overlap being measured. For feasible configurations, collect at least five warm iterations and report median/range plus peak memory and numerical diagnostics. Do not extrapolate the 562-sequence AMPH report into a measured 1,851-sequence tandem result.

An SLDS-only initialization test may isolate a crash, but use a representative fixed post-AR checkpoint for production-relevant SLDS measurements. Do not remove AR warmup from the scientific workflow based on a speed test.

## 4. Run a limited, question-driven experiment matrix

Set wall-time and RAM/VRAM stop conditions before each test. Start at a small safe size and expand only after the preceding gate passes. Do not rerun a known failing configuration without a changed hypothesis and preserved failure record.

- **Batch cost:** vary one family at a time. Candidate starting values: discrete 2/5/10; scales/heading/location 5/10/20; continuous chunks 5/10/25/50, subject to preflight. Measure both full-iteration effects and memory. Record actual transfer counts/bytes or trace evidence instead of inferring them from batch count.
- **Persistent-memory slope:** vary sequence count at fixed per-sequence shape and separately vary sequence length. Include initialization, full-output returns and checkpoint peaks. Synthetic shapes are useful for allocation tests; distinguish them from independent behavioral recordings.
- **Copy/lifetime test:** measure retention of `cached_model`, deepcopy, CPU cache/source references, list-of-chunk outputs, concatenation and final full-device returns. Test a preallocated host output or host-backed store against the baseline.
- **PCA/preprocessing:** isolate full-cohort alignment from PCA fitting; compare the current subset with a reproducible stratified subset on held-out recordings/rare postures. Measure process-tree memory as worker count changes.
- **Kernel optimization:** only investigate fused/tiled emissions, transfer overlap, controlled LAPACK/Numba threading or another solver once profiling identifies that cost as material. Preserve an independent mathematical reference.

For every change provide the hypothesis, minimal diff, fixture, configuration, result, correctness impact, and keep/reject decision. Maintain one experiment ledger so failed ideas are not rediscovered later.

## 5. Implement the durable design where the evidence supports it

The central target is a memory budget for the whole lifecycle. Produce an array ownership/lifetime table and a preflight estimate using actual shapes/dtypes and measured overhead. Include persistent arrays, simultaneous workspaces, old/new output coexistence, transfer overlap, host copies and checkpoints. Return an actionable “cannot fit” result when one sequence is too large; reducing every batch to one is not a solution to an impossible budget.

Prototype bounded host-backed observations/state and reusable device chunks if full-state residency is the remaining limit. Write outputs incrementally instead of building full lists and concatenating; include initialization and checkpoint/export paths. Preserve the exact Gibbs dependency order: sequence-local updates conditioned on appropriate shared parameters, followed by full-data sufficient-statistic reductions for global updates. Do not update shared parameters on arbitrary partial datasets or treat temporal chunks as independent without specifying a different inference method.

Use fixed model/sequence dimensions to test whether GPU peak memory stays within an agreed envelope as total sequence count grows. Host/disk capacity and full-sweep time may still grow with data. Treat exceptionally long sequences as a separate boundary/message-passing problem.

Only propose multi-GPU or SVI/online EM after this baseline is established. State communication costs or changed posterior/convergence assumptions explicitly; do not promise constant total memory, linear speedup or faster convergence without measurements.

## 6. Required handoff and stopping point

Return:

1. Updated ZIP of **all relevant live Python patches and dependencies**, clearly separating active, legacy and experimental code, plus hashes and exact environment information.
2. A concise results dossier with reproducible commands, raw logs/traces, machine-readable benchmark summaries and unresolved limitations.
3. A patch registry and documented enable/disable/restore behavior; specific diffs for accepted fixes.
4. A measured capacity model for the currently supported `N,T,keypoints,D,K,lags`, including startup/save/resume; a proposed regression matrix for future cohort growth.
5. Scientific comparisons against the declared reference, checkpoint/resume validation and cache-invalidation tests.
6. A recommendation with the limiting stage, supported dataset envelope and evidence required before a full production run.

Stop after the agreed investigation/soak gates, or earlier if a correctness issue or missing dependency blocks meaningful comparison. Report the concrete blocker and next experiment instead of expanding into another open-ended optimization chase. Do not label the full cohort solved until an end-to-end run including save/reload/export has demonstrated the agreed resource and scientific criteria.
```
