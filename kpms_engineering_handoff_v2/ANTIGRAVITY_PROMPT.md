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
