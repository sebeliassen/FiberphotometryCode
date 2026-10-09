# September 29 auxiliary update: what changed and what to test first

**Status:** source audit and incident interpretation; no access to the original machine or GPU run. This review updates the earlier dossier and takes precedence where the new source changes behavior. The supplied incident report is preserved unedited in `auxiliary_bundle/KPMS_SCALING_ADDENDUM_LIVE_CRASH_ANALYSIS.md`.

## The short answer

The earlier dossier covered full-array residency, intermediate peaks, misleading timings and the need for multi-iteration validation. The new archive supplies a real implementation change—CPU output for the large-batch scales sampler—and a reported failure on a later discrete-emission call. It makes the investigation more specific.

**Do not yet label the cause “proven BFC fragmentation.” Test the repeated `fit_model` entry first.** The bundled reference calls a utility that transfers the model tree to the device. That can undo the new host-residency policy for `s` whenever fitting is entered again. The reported failing line is exactly the runner’s second SLDS `fit_model` call. This is a concrete, source-supported hypothesis; its contribution to the actual OOM remains to be measured.

## 1. Exact archive comparison

The auxiliary archive has 101 files: the prior 100 paths plus one new report. **Four existing files changed; none were removed.** Both revisions are preserved in this handoff. `AUXILIARY_DIFF.patch` is an exact unified comparison, not a proposed fix.

| Changed file, relative to `auxiliary_bundle/` | Actual change |
|---|---|
| `core_patches/resample_batching.py` | Large-batch scales returns a NumPy array on the host. Heading and location explicitly slice host `s` and transfer each slice. |
| `production_and_test_runners/run_cohort_2ar_2slds_test.py` | Cache restoration keeps `s` on the host; adds `--skip-ar` and `--num-slds`; changes phase labels and defaults AR time to 5.5 seconds when skipped. |
| `production_and_test_runners/run_tandem_cohort_production_50ar_200slds.py` | Cache restoration keeps `s` on the host. |
| `production_and_test_runners/launch_2ar_2slds_test.sh` | Forwards command-line arguments to the Python runner. |

The emission implementation, batch planner, initialization routine and allocator settings did not change. In particular, the auxiliary code still defaults/caps `discrete_bs` at 10; it does not implement the report’s recommendation to use 5. The host-scales change does not cover the `N <= bs` fast path or initialization’s final full-device placement. Nor does it establish residency through upstream wrappers or every downstream consumer.

## 2. What the new run supports

The addendum reports successful complete SLDS work followed by an OOM at `_jit_compute_discrete_emissions`, requesting **2,414,584,344 bytes (2.415 GB; 2.249 GiB)**. It reports 74.03 seconds cold and 46 seconds warm. No separate raw run log, profiler trace, allocator dump or installed-version export was added; the traceback is embedded in the prose report.

This is useful evidence of progress and a later failure. It is not yet a demonstrated stable throughput, a proven allocator diagnosis, or proof of the exact tensor composition of the allocation. The continuous solver in this patch remains a **CPU Numba** implementation; a successful hybrid SLDS sweep should not be described as every sampler running on the GPU.

The new 46-second figure replaces the earlier unsupported 12.3-second tandem figure as the **new reported timing to investigate**, not as a validated planning constant. Resolve the iteration/timing issues below before extrapolating either number.

## 3. First hypothesis: re-entering fitting reuploads host states

Source trail:

1. The new scales sampler returns host NumPy `s` at the end of a large-batch sweep.
2. The test runner invokes `kpms.fit_model` once at line 250 and again at line 267.
3. The unchanged bundled `diagnostics_and_benchmarks/fitting_reference.py` applies `device_put_as_scalar(model)` before its sampling loop.
4. A local reference implementation in `diagnostics_and_benchmarks/profile_kpms_breakdown.py` transfers the tree via `jax.device_put` before scalar conversion. Published upstream utility and fitting sources corroborate this pattern [U1, U2].

**Inference, conditional on matching installed source and a GPU default device:** a host NumPy `s` can return to the GPU at the start of each fit call, before the next `z` computation. A NumPy `noise_prior` in the model is another candidate. An already device-committed CPU array may behave differently; inspect actual types and placement rather than assuming all host representations are equivalent. Old model references and cache-held sources may add simultaneous live arrays.

This does not establish why the first call fits and the second fails. It identifies a specific placement boundary to instrument. At the stated FP64 shape, moving a full `s` adds approximately 3.416 GB of device payload, easily large enough to matter to a 2.415 GB scratch request.

**Decisive experiment:** from the same initial checkpoint and RNG state, compare one continuous call against split calls that perform the same counted sweeps. First do this on small data with checkpointing disabled; then use the failing shape if budget permits. Log leaf type, dtype, bytes and device before fit entry, immediately after its conversion, after each sampler, on return, and on re-entry. Specifically track `s`, `noise_prior`, `x`, `z` and cache-held source arrays. Count H2D bytes where available. A single call is a diagnostic comparison, not a complete architectural fix.

If uploads are confirmed, implement selective placement at the wrapper boundary and verify every use of host `s`/prior. Simply moving `s` to the CPU immediately before calling a wrapper that uploads it again cannot enforce the intended policy. Preserve scalar conversion, seeds, error checks and sampling order when adapting the wrapper.

## 4. The test harness has measurement and control defects

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

## 5. Why the fragmentation explanation is not established

An allocation failure provides an allocation size and a failing call. It does not, by itself, distinguish live-array pressure, overlapping work, duplicate state copies, growing executable caches, another GPU process, or allocator fragmentation.

The supplied runners already request `TF_GPU_ALLOCATOR=cuda_malloc_async` before importing JAX. JAX documents this as an experimental alternative to BFC with different allocation/fragmentation behavior [J1]. Verify which allocator the installed runtime actually uses; neither the environment string nor `tf-allocator-allocation-error` proves it. Recommending the same flag again is not a new experiment. Raising a shell memory-fraction value is also ineffective while the script overwrites it.

The report’s exact scratch decomposition is unsupported. With `B=10, T=10030, L=3, K=100, D=10`, one `(B,T-L,K,D)` array is **401,080,000 bytes in FP32** or **802,160,000 bytes in FP64**. Three such FP64 arrays total **2,406,480,000 bytes**, close to the requested size; this illustrates ambiguity, not proof of three arrays. Record actual operand/intermediate dtypes and compiler allocation data. Do not invent an alignment workspace to make a byte count match.

JAX recommends XProf for device-memory analysis; synchronized device-memory snapshots can identify retained arrays/executables, although JIT internals are opaque in that view [J2]. Pair these with compiled memory/buffer analysis when the installed backend exposes it [J3], and with a device-usage timeline. A memory snapshot alone does not prove a fragmented free-block layout.

## 6. A short experiment sequence for this incident

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

## 7. Immediate Antigravity instruction

Use the updated `ANTIGRAVITY_PROMPT.md`. Its first task is now to test fit-entry placement and reconcile iteration/timing semantics, before adding another memory flag or writing a custom kernel. Request the full raw log, exact run command, effective config, source hashes and installed fitting/utility source for the reported incident.

Do not claim stable 46 s/sweep or a solved cohort until correctly counted repeated sweeps, host-residency checks and save/reload/export all meet the agreed budget. The original dossier’s long-term objective—bounded memory through the whole lifecycle—still applies.

## Primary references checked on September 29, 2026

- **U1:** [JAX-MoSeq utility source](https://jax-moseq.readthedocs.io/en/stable/_modules/jax_moseq/utils/utils.html) — model-tree device conversion utility. Corroborates the local reference; not an export of the user’s installation.
- **U2:** [Keypoint-MoSeq fitting source](https://keypoint-moseq.readthedocs.io/en/latest/_modules/keypoint_moseq/fitting.html) — fit-entry conversion, loop and save behavior. The bundled `fitting_reference.py` provides the local evidence discussed above.
- **J1:** [JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html) — allocator choices and caveats.
- **J2:** [JAX device memory profiling](https://docs.jax.dev/en/latest/device_memory_profiling.html) — live-buffer/executable profiling and its limits.
- **J3:** [JAX ahead-of-time compilation](https://docs.jax.dev/en/latest/aot.html#debug-information-and-analyses-when-available) — backend-dependent compiled analyses.
