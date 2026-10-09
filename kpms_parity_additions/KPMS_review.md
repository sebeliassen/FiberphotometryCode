# KPMS precomputation and parity review

Reviewed 29 September 2026. Target: the attached `resample_batching.py`, SHA256
`634185e1a797f06a26ab99a3a787b08a155d6e0301f44dac93d9a8bc17ddbb72`.

**The supplied evidence does not establish that the latest version is equivalent to
your stable 27× version. I found a reproducible cache correctness defect, but have
not established that it causes your duration trend.** The archive contains one
implementation, reports, and tests; it does not contain the historical revision,
your production checkpoints, or the actual installed JAX-MoSeq sources.

## What the new checks found

| Check | Result here | Meaning |
|---|---|---|
| Expired source object ID reused by another object | Six failing subcases in one test | Existing host cache can return stale x, z, mask, or Y |
| Candidate source-identity cache fix | All 10 test methods pass | Fix addresses the reproduced lifecycle defect in source mode |
| Independent dense Gaussian construction | Pass | Supplied AR-block construction, assembly and sample agree with the model encoded in this implementation on small fixtures |
| Chunk sizes 1, 2, 3, 4, 7, 8 | Pass | Full continuous wrapper handles multiple chunks and a partial final chunk on these fixtures |
| Change one dependency, compare warm and cleared cache | Pass | Ab, Q, z, Cd, sigmasq, s, h, v, Y and mask changes are reflected in the exercised calls |
| Exact banded noise-map covariance | Pass | Gaussian solve orientation and full covariance match a dense inverse |
| HMM posterior path enumeration | Pass | Tiny asymmetric HMM with masks matches exact conditional backward draws |
| AR sufficient statistics | Pass | Chronological lags and target-frame mask match an independent calculation |
| Batch-size planner | Pass | Explicit budget changes alter batching without altering tested continuous draws or input data |
| Deliberately injected defects | All five detected | Tests reject frozen dynamics blocks, reversed lags, dropped local scales, removed sample noise, and transposed transitions |
| Real JAX/Numba execution and upstream conditional parity | **Not run here** | Requires your installed environment; executable tests included |
| Full training against historical 27× revision | **Not run here** | Requires that revision, identical data/checkpoint, and your environment |

Local execution used the actual selected function bodies extracted with Python AST,
with Numba decorators removed and explicitly substituted NumPy/device plumbing.
This is useful numerical and logic evidence, not evidence about compiled Numba,
GPU execution, real JAX object allocation, or full Gibbs-chain equivalence.

The dense oracle deliberately matches the supplied initial prior, masked-frame
rules, observation floor, and *precision* jitter. Agreement with it does not prove
those conventions match the installed upstream Kalman sampler. The separate live
conditional test addresses that distinction.

## 1. Reproduced cache defect, with a causality limitation

In `KPMSStateCache` (`resample_batching.py`, beginning at line 181), cache hits use
an integer `id(source_array)`, while the cache retains the CPU value but no
reference to the source object. Python guarantees uniqueness of an object ID only
while the object is alive. Once it dies, another object may receive that ID:

1. Cache a CPU copy of source A.
2. A is destroyed; the cached integer remains.
3. Source B receives that integer ID.
4. `get_x(B)` can return A's values. The same pattern exists for z, mask and Y.

The new test uses a deterministic allocator model that only reuses an ID after
the original object has died. It never gives two live objects the same ID. It
reproduces six failing cases: four getters plus `update_x` and `update_z` producer
paths. It does **not** claim an actual JAX array allocation collision was observed.

The candidate patch retains each source object and compares with `is`. It also
clears those references in `clear()`. It leaves the integer fields in place for
compatibility. This keeps at most one source per cached field alive until it is
replaced or cleared; check memory behavior on the real workload. A weak-reference
implementation can avoid prolonging source lifetime if supported by all relevant
array types.

**This is not yet the explanation for the observed median decrease.** Normal
fully patched execution refreshes x and z at production time, and the current
model/data often keeps the source objects alive. Those facts can prevent the
failing lifecycle in an ordinary training run. Model switching, temporary inputs,
alternate kernels, and repeated replacements before cache lookup deserve testing.
Instrument cache hits against forced fresh copies on the affected run before
attributing the behavioral change to this defect.

Python reference: https://docs.python.org/3/faq/programming.html#why-does-the-result-of-id-appear-to-be-not-unique

## 2. What “precomputation” actually means in this file

There are three different mechanisms:

| Mechanism | Dependencies | Supplied behavior |
|---|---|---|
| `precompute_optimal_batch_config` | Data/model dimensions and memory budgets | Chooses batch/chunk sizes; does not compute posterior terms |
| `K_blocks`, `g_priors` in continuous sampler | Current Ab and Q, lag ordering | Rebuilt on every sampler call, then reused across times/sequences within that call |
| `KPMSStateCache` | Object identity for x, z, mask and Y | Persists CPU copies across calls; lifecycle defect above |

I did not find cross-epoch reuse of old Ab/Q blocks in the attached continuous
sampler. If Gemini has newer code than this archive, the actual diff is necessary.
Our injected frozen-block mutant fails the new dependency test.

The dossier's statement that observation precision is globally constant is too
broad. With local noise scales, even if rotation cancels,

`J_obs(t) = C.T @ diag(1 / (sigmasq * s[t])) @ C`

still depends on time and the current s. The supplied code correctly uses s inside
the observation loop (line 763). Precomputing unweighted per-keypoint basis
matrices is compatible with reapplying the current weights. Freezing the complete
weighted matrix across time or after s/sigmasq/Cd changes is not. The new fixtures
have heterogeneous scales and reject removal of those weights.

Batch changes also change the PRNG assignment in the batched s/h/v functions and
JAX z fallback: they use batch starts and array shapes to generate draws. This
can change a same-seed training trajectory without changing its target law.
Initialization batching changes random-key progression too. Start comparisons
from a saved common checkpoint to remove that confound. For deterministic
terms, cache on/off, and the current streamed NumPy x/FFBS noise paths, require
numerical equality with coupled randomness. Do not require identical long-chain
labels across distinct upstream and optimized random-number implementations.

## 3. Why the present “all passed” claim is insufficient

* `test_continuous_sampler_equivalence.py` computes variance ratios and counts
  pivot repairs but its final assertions only check masked-sample NaNs, mean
  correlation > 0.95, and variance in a gap being larger than elsewhere. It does
  not assert variance equality, full covariance, finite values everywhere, or
  zero repairs. Its displayed target differs from its actual assertion.
* `test_kernel_semantic_parity.py` uses posterior-mean correlation > 0.85 for its
  continuous comparison. It does assert zero repairs, which is useful, but high
  correlation is not sufficient distributional equivalence. Its s/h comparison
  uses four sequences, below default batch sizes. Its function named
  `test_scales_and_location_parity` does not actually test location.
* The Numba FFBS distribution check correlates aggregate state occupancy counts.
  Occupancy does not determine transitions or bout duration: two sequences can
  spend equal time in every state but switch at very different rates.
* The double-buffer test compares two wrappers using the same emissions and
  FFBS kernel. It checks scheduling/buffer reuse, not an independent posterior.
* The existing cache test keeps the relevant old model arrays alive while
  changing models. It never exercises expired-source ID reuse.
* The four-arm script has no cross-arm equivalence assertions. It prints results
  for 15 synthetic SLDS iterations; that does not cover the observed production
  drift over 100–250 iterations. Its duration helper does not accept a mask.
* The “unpatched FFBS” arm still invokes the optimized z wrapper and optimized
  emissions, using its JAX fallback. It is not a restoration of `_orig_z`.
* `CONFIG.banded_solver` is defined but never consulted in this file. Setting it
  to `lapack` does not select a different solver. Other dispatcher settings also
  need actual callable identities/call counts checked when building ablations.

The new live conditional test reconstructs the optimized sampler's exact mean
and covariance using zero/basis noise, then compares 4,096 independent upstream
draws after whitening. It asserts means, variances, and cross-coordinate/time
covariances, with all-valid, trailing-mask, internal-mask, and two jitter cases.
It reports source hashes and versions. This is a stronger small-fixture gate,
not a substitute for the historical-revision comparison or real-data checks.

The supplied sampler adds `jitter` to the **precision** diagonal (line 806).
That changes both the conditional mean and covariance. Check the installed
upstream implementation's actual jitter semantics; equal numerical argument
values do not establish equivalent regularization. A discrepancy here could
predate the latest optimization, which is why the stable revision matters.

## 4. Duration and speed evidence need a controlled comparison

The CSV does show a falling median together with increasing mean and upper
percentiles and decreasing bout count. That combination is mathematically
possible. It does not establish a log-normal distribution, discovery of real
transition motifs, correct posterior sampling, or convergence. Aggregate active
syllable count is not a sufficient safeguard either. Mean bout duration and bout
count are largely redundant if they cover a fixed number of frames.

The dossier compares different workloads: about 5.53 million AMPH frames versus
416,032 tandem frames. A lower wall time there does not demonstrate an additional
8× algorithmic speedup. Its stated throughput arithmetic is also inconsistent:
416,032 / 100,000 × 200 is about 832 seconds, not 256; 416,032 / 320,000 × 50 is
about 65 seconds, not 24. These inconsistencies weaken the written explanation;
they are not evidence that the current kernels are wrong.

Benchmark the stable and current revisions on the same checkpoint, data arrays,
active and padded frame counts, dimensions, iteration range, precision, hardware,
and hyperparameters. Synchronize all returned device work. Report compilation,
warmed iteration time and complete fitting time separately, and count only
iterations actually executed after resume.

For durations, derive the lag offset from the arrays/model rather than assuming
three. Exclude padding, split runs at masked gaps and sequence boundaries, and
handle duplicated segment overlap using metadata. Verify that summed included
bout lengths equal the included valid-frame count under the chosen censoring
convention. Compare frames first, then convert using the actual fps.

## Recommended decision

Keep the historical 27× version as the reference while running the enclosed
checks. First isolate frozen-condition differences on real checkpoints; then
run matched training comparisons over the interval where the duration changes.
Do not adjust kappa or relax tolerances to make an optimization appear equivalent.
Read `GEMINI_HANDOFF.md` for the commands and experimental sequence.

The uploaded implementation was not replaced. The patch is a separately supplied
candidate, tested locally in source mode only. No production training, GPU test,
or installed-upstream comparison was performed here.
