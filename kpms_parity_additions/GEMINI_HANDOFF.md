# Handoff for Gemini / Antigravity

Investigate the change from the user's known-stable approximately 27× engine to
the current engine. Do not declare the duration decrease normal from aggregate
medians, means, state counts, or the existing six-test summary. Do not modify
kappa, preprocessing, segmentation, data selection, or test thresholds to hide a
difference. Keep production checkpoints unchanged; use a diagnostic checkout and
separate output paths. Do not execute the production runner just to test imports:
it performs unrelated actions including notifications.

The attached review and tests were written from one supplied source snapshot.
The historical source and real JAX/Numba environment were unavailable to the
reviewer. Source-mode tests are not live GPU or upstream-parity evidence.

## First: run the new tests against your actual current file

Extract `kpms_parity_additions.zip` into the KPMS project directory, leaving its
`kpms_parity_additions/` directory intact. Activate the existing KPMS environment.
Run from the directory containing `utils/resample_batching.py`:

```bash
set -o pipefail
mkdir -p diagnostics/precompute_review_logs
python kpms_parity_additions/test_precompute_regressions.py \
  --rb utils/resample_batching.py --backend live \
  2>&1 | tee diagnostics/precompute_review_logs/current_regressions.txt
python kpms_parity_additions/test_live_conditional_parity.py \
  --rb utils/resample_batching.py --draws 4096 \
  2>&1 | tee diagnostics/precompute_review_logs/current_conditionals.txt
```

These are standalone unittest runners: pytest is not required. They return a
nonzero exit status for failures. The first command is expected to fail the
expired-ID cache test on the exact supplied snapshot. The second command may
find model-convention differences; inspect failures rather than loosening gates.
If your source interface differs, adapt test plumbing while retaining independent
oracles and the scientific assertions. Report any such adaptation.

To reproduce the attachment-only run without JAX/Numba:

```bash
python kpms_parity_additions/test_precompute_regressions.py \
  --rb utils/resample_batching.py --backend source
```

The optional `cache_source_identity_candidate.patch` is specifically for the
attached file layout. In an isolated diagnostic checkout, inspect it and run
`git apply --check kpms_parity_additions/cache_source_identity_candidate.patch`
before applying it. Apply only that cache change, rerun both live commands, and
record device-memory effects. It is not an established fix for the user's duration
trend. A changed source hash is expected if your implementation is newer.

## Identify the actual regression interval

1. Locate the precise commit/backup corresponding to the stable 27× run. Have the
   user identify the known-good run if logs cannot tie it to a revision; do not
   assume a commit date alone proves stability.
2. Record current and historical commits, source SHA256s, imported module paths,
   installed package versions, dtype/JAX x64, hardware, thread counts, effective
   CONFIG, dimensions, actual kappa schedule, fps, segmentation settings and
   resume iteration. Obtain a diff of the sampler, initialization and pipeline.
3. Check whether new code actually persists Ab/Q blocks, emissions, lag matrices,
   sufficient statistics, rotated observations, or observation precision across
   calls. The supplied file rebuilds K_blocks and g_priors on every x-sampler call.

## Frozen-condition checks: highest priority

Use the same immutable real checkpoint for both revisions, first at the AR-to-SLDS
transition and then at a later checkpoint in the affected interval. Select valid
segments with distinct states, starts/ends, varying s and masks. Deep-copy state
and parameter trees for each branch; validate that input snapshots are unchanged.

* Compare AR sufficient statistics, emission log-likelihoods, transition inputs,
  un-factorized precision blocks and information vectors at tight FP64 tolerances.
  Stop at the first differing substep and save its inputs and discrepancy metrics.
* Instrument **every cache hit** on a small run to compare the cached value with
  a fresh host conversion of the actual argument. Fail immediately on a mismatch.
  Test model A → B → A, same-shape inputs, different dimensions, checkpoints,
  temporary views, multiple replacements and checkpoint loading with warm caches.
* Bypass each cache getter for the whole diagnostic run, not merely `clear()` at
  the start of each epoch. Do not change numerical dtypes when bypassing it.
* For persistent mathematical caches, perturb one dependency at a time. AR blocks
  depend on Ab/Q; emissions on x/Ab/Q; statistics on x/z/mask; transformed Y on
  Y/v/h; observation precision on Cd/sigmasq/s/mask. A changed h must change the
  information vector where appropriate even if precision is heading-invariant.
* For continuous samples, test zero-noise means and basis-noise covariance on
  tiny fixtures, then upstream frozen-condition distributions. Resolve initial
  priors, masked-observation/dynamics rules and jitter semantics using the
  **installed** upstream source, not a recalled implementation.
* Test explicit lag counts 1–4, nonzero affine biases, non-diagonal Q, heterogeneous
  local scales, sequence starts/ends, partial chunks and noncontiguous slices.

Run batch sizes 1, 2, 3, 20, 42, 50, 60 and N where practical, including N just
above/below the boundary and three batches with a short final batch. Use small T
for boundary tests. For the streamed NumPy x and FFBS implementations, same input
and seed should preserve results apart from justified floating-point tolerance.
For s/h/v and the JAX z fallback, current keys depend on batching: either provide
coupled per-global-sequence random variates, or compare their conditional laws
and deterministic inputs. A same-seed mismatch alone is not a law mismatch.

## Matched training experiment

After resolving deterministic failures, start these arms from the identical saved
model/data checkpoint, in fresh processes or isolated imports:

| Arm | Purpose |
|---|---|
| Historical 27× revision | Primary regression reference |
| Current revision | Reproduce the reported trend |
| Current with every host cache bypassed | Isolate cached CPU data |
| Current using the historical batch/chunk configuration | Isolate scheduling and RNG assignment |
| Current with new mathematical precomputations recomputed directly | Isolate newly introduced algebra/cache dependencies, if present in the actual diff |

Verify actual called function identities and call counts. Merely setting
`banded_solver='lapack'` does nothing in the supplied file. Merely changing
dispatcher settings after applying patches may not change installed callables.

Use the same data, noise prior, initial states/parameters, kappa schedule, fps and
preprocessing. Do not rerun initialization separately for each arm. Cover the
same duration-trend window (for example iterations 50–250 if that is the real
affected interval). Start with a representative subset; confirm any finding on
the affected data. Use at least five seed replicates initially; extend only if
uncertainty prevents a conclusion. Seeds/chains are replicates, not independent
frames. Predefine scientifically meaningful equivalence margins using variation
between historical runs. Failure to reject a difference is not equivalence.

Report masked, boundary-aware bout distributions: median, mean, p10/p25/p75/p90,
fractions of 1–2 and 3–5 frame bouts, transitions per valid adjacent pair,
occupancy and per-state self-transitions, duration distributions by sequence,
latent-motion increments, and available consistently defined likelihood metrics.
Use label-invariant metrics or align states before interpreting per-state changes.
Separate boundary-censored runs and handle segmentation overlap consistently.

Benchmark only matched executed work; synchronize all device outputs, separate
warmup/compilation, record AR and SLDS time separately, and exclude previously
completed resumed iterations from the denominator.

## Return to the user

Return raw test output (including failures), source/version hashes, the old/current
diff, first divergent frozen substep, and matched duration/throughput tables.
Classify each conclusion as a reproduced defect, a possible cause requiring
confirmation, or a passed test with limited coverage. Do not call correlation of
posterior means or matching medians proof of zero statistical deviation.
