# Broad behavioural search handoff — 2026-10-13

**Superseded later on 13 October:** integration and validation were completed and
the separate nohup search was launched. Start with `docs/tasks/CURRENT.md` and
`outputs/broad_search_20261013/README.md` for the running checkpoint. The content
below preserves the earlier unfinished state and should not be used to restart
implementation or launch a duplicate search.

## Resume here

**The broad randomized search has NOT started.** The user authorized implementation
and launch, then requested this handoff because the chat was nearing its token
limit. Finish integration and validation before launching a separate nohup search.
No further permission is needed for the already authorized scope. Do not restart
the old stopped Karen overnight service.

Repository: `/home/fsp585/Desktop/FiberphotometryCode`; baseline `36473e4`, branch
`codex/modular-refactor`. There is extensive unrelated uncommitted work, including
untracked shared modules and folder maps. Preserve it; do not reset, clean, stage
everything or infer ownership from the diff alone. Implementation described below
is unfinished and uncommitted.

Start with this handoff, `docs/tasks/CURRENT.md` and
`docs/architecture/UNIT_CONTRACT.md`. Read the relevant `src/fpcode/AGENTS.md`,
`studies/AGENTS.md`, `ops/AGENTS.md` before editing those owners. Planning documents:

- `docs/architecture/BROAD_BEHAVIOURAL_SEARCH_PLAN.md`
- `docs/architecture/BROAD_SEARCH_EXECUTION_PLAN.md`
- `outputs/broad_search_preflight_20261013/report.md`

The older dated checkpoints describe their own stage, not current launch readiness.
In particular, older references to video review before the broad sweep are
superseded by the user's **quantitative first, qualitative shortlist later** choice.

## User decisions that must survive the handoff

- Primary aim: useful episodes/chains that distinguish separate behaviours while
  grouping similar executions despite minor variation. Search representations and
  detector settings; do not optimize for cohort, genotype, drug or DA differences.
- Reuse frozen KPMS outputs. No new KPMS runs. Preserve complete source bout
  boundaries; whole-bout execution subtypes are distinct from temporal subdivision.
- Include Karen and Jonathan as separate cohorts and model-scoped vocabularies.
  Never compare numerical syllable IDs across models or transfer a crosswalk blindly.
- Include actual feature-input contrasts in the broad search space, rather than
  requiring a separate feature/HSMM optimization campaign. Descriptive sidecars
  must be distinguished from features actually consumed by a detector.
- Include expanded HSMM conditions, but report the attention, initialization,
  training budget, learned/fixed parameters and limitations given to each method.
  A single seeded HSMM target is not a learned multi-routine catalogue.
- Quantify fragmentation, support, consistency, redundancy and similar diagnostics
  separately. There is no agreed universal quality score or method winner.
- Quantitative search comes before expensive matched video review. Later review
  should cover diverse shortlisted outcomes, not just a top scalar score.
- Both Jonathan KI and WT are behavioural evidence. The user's interpretation of
  KI as ADHD / late-stage Parkinson mice is recorded separately from historical
  DAT-Het terminology. DA is optional **WT-only downstream corroboration**, never
  selection ground truth. Reuse existing processed signals and signal metadata.
- Previous data exposure is a limitation (“this might not be perfect”), not a
  declaration that the dataset is unusable. Keep prior exploration transparent.
- Randomized execution must survive the terminal/chat via nohup, with logs,
  durable checkpoints, resource caps and stop/resume controls.

## Verified inputs and empirical products already available

### Jonathan AMPH Run12

Correct source: `data/sessions/jonathan_amph_sessions_run12_w_sylls.pkl`, about
7.53 GB; model `amph_prod_k150k_run_12`. The user's groom IDs 58, 60, 61, 63, 65,
66 belong to `compat/syllable_generation/analysis/sequence_viz/run12_human_labels.csv`
(54 rows), **not IFDN** and not automatically the v2 pickle.

All 5,500,994 saved sparse labels/native VFI rows matched all 30 session tables.
There are 15 paired animals (8 WT, 7 KI), each with Saline and Amphetamine.
Keep original drug labels distinct from pre/post phase. Historical Ulrik loaders
that overwrite drug by phase must not be reused here.

Verified supported projection contains 10,993,910 native frames and 229,579 source
bouts. It excludes 574,168 dense labels filled beyond real model support. Sparse
model VFI usually advances by two: an interior partner frame can be included only
when the next VFI is exactly `f + 2`; final supported frame is `[f, f+1)` and larger
gaps are barriers. Preserve these rules when joining feature exports.

Artifacts under `outputs/broad_search_preflight_20261013/amph_run12/`:

- `manifest.json`, `annotation_identity_manifest.json`
- `native_verified_manifest.json` (very large; inspect selected fields, do not
  print the whole file), `native_input_manifest.json`
- `native_inputs/*.npz`: native frames, labels, source bout bounds and basic
  velocity/angular velocity/centroid channels
- `clock_header_evidence.json`, `annotations.md`, `da_documentation.md`,
  `wt_signal_inventory.csv`

Clocks are distinct: model 40 Hz, encoded cropped video 60 FPS, acquisition about
80 Hz. Historical acquisition rate 80.025 Hz is declared; bounded timestamp drift
is documented, not settled for every recording. Do not infer a native conversion
from the AVI header. WT signal inventory has 16 recordings / 8 animals, raw,
`_dff`, `_zscore`, `_zscore_datta`, original fibre/region/hemisphere metadata.
Newer Wallace/prominence variants were not present. No DA reprocessing was done.

New feature export already completed during this implementation turn:

- `outputs/broad_search_preparation_20261013/jonathan_feature_exports.json`
- `outputs/broad_search_preparation_20261013/jonathan_feature_exports/*.npz`

`studies/session_model_exports/inspect_behavior_inputs.py` was extended to export
heading and `latent_state_0` through `latent_state_9`. All 30 recordings exported
under a 24 GiB address-space cap, 2 numerical threads and 180-second watchdog;
elapsed about 27.30 s, peak RSS 11,242.20 MiB. Original pickle/model untouched.
**Do not unpickle it again just to prepare features.** Join these sparse exports
to the existing verified native support using native VFI; exact frame or supported
interior partner only, with label equality assertions before attaching latents.
Dense heading requires independently verified native indexing. Do not bridge gaps.
An older bout cache includes a B2V fit on all animals; it is unsuitable as a
training-only fitted encoder.

### Karen reviewed inputs

`outputs/karen_reviewed_analysis_view_20261012/reviewed_boundaries/` contains
`sessions.csv` (99 recordings, 21 mice), `manifest.json`, `frame_cache/*.npz` and
`chains/animal_splits.json` (12 train, 4 validation, 5 test animals).

Caches include frames, labels, centroid, heading, 10-dimensional latent features,
FPS, exported velocity, exported static syllable and `analysis_excluded`. Use the
verified review masks and native 60 Hz clock. Model:
`model_21kp_static_50ar_300slds_seed101`. Annotation source:
`outputs/karen_behavior_atlas_20261011/syllable_annotations.csv` (100 rows).

### Completed bounded categorical pilot

`outputs/broad_search_method_pilot_20261013/` contains a completed pilot, not a
random search: four train and two held animals per cohort, 33 recordings total,
2,400 training events per cohort; held 1,050 Karen / 600 Jonathan events. Eight
conditions per cohort: exact, four seqNMF conditions (K4, L3/L6, seeds101/202),
configured single-target HSMM and two reestimated variants. 1,633 raw / 1,262
selected calls; native bounds, complete source support, splits and caps passed.
No method winner, phenotype or DA comparison was claimed.

## Implementation completed or partially completed this turn

### Shared HSMM and inference

`src/fpcode/behavior/episodes/staged.py` now supports optional second background
state, phase skips, entry / early-exit / reentry choices, weighted diagonal
Gaussian feature emissions, and optional duration reestimation. Existing defaults
remain categorical with frozen durations. Hard EM can learn categorical emissions,
transitions, Gaussian means/variances and duration PMFs; initial distribution and
topology remain fixed. Duration updates exclude the first and right-censored last
segment of each block; independent hand-count tests cover this policy.

`src/fpcode/statistics/sequence_inference.py` batches HSMM forward/Viterbi duration
calculations and caches incoming transition calculations. Float64 semantics and
tie preferences were preserved on a bounded comparison against the old code:

`outputs/broad_search_preparation_20261013/hsmm_benchmark.json`

- 24 random cases, structural zeros/right censoring; all MAP paths identical.
- Maximum likelihood difference `7.105427357601002e-15`, tolerance `1e-11`, seed41.
- Synthetic T300/Q7/D32, 2 threads: reference 2.0774 s, vectorized 0.02463 s,
  about 84.34× locally. This is bounded synthetic CPU evidence, not universal
  scientific equivalence or a GPU claim.
- Reference code is saved in
  `outputs/broad_search_preparation_20261013/sequence_inference_reference.py`.

Remaining review: update stale categorical-only fit docstrings; account explicitly
for feature matrices and duration workspaces in memory estimates; validate unusual
feature-weight inputs and expanded topology cases. Keep duration units event-count,
not seconds, unless a separate explicit physical-time implementation is added.

### Source-bout features and codebooks

New `src/fpcode/behavior/analysis/bout_features.py`:
`summarize_source_bouts(...)` provides vectorized, gap-aware source-bout summaries.
Features include duration, speed, net speed, path length, centroid, optional signed
and absolute heading change, latent means and dispersion. Derivatives do not cross
gaps or source bout edges; insufficient support stays NaN. Units are seconds,
pixels / pixels per second, radians, and model-specific latent coordinates.
Centroid coordinates are image proxies, not automatically wall distance.
Current memory estimate is coarse; review its scaling with latent dimensions.

`WholeBoutCodebook` added to
`src/fpcode/behavior/primitives/segmentation.py`: training-only robust median/IQR
scaling, feature-only, categorical+feature joint, and within-parent subtypes.
Joint categorical evidence uses one-hot categories, never arithmetic syllable IDs.
Missing feature rows / unseen within-parent codes remain unsupported. Parent
subtype identities are explicit. Remaining: validate/freeze reconstructed codebook
state, bound assignment batching and memory, and test serialization identities.

### Dependency cache

`cached_npz_component(...)` added to `src/fpcode/common/cache_manifest.py`:
canonical JSON identity hash, per-key kernel flock, atomic data-only NPZ publication,
embedded identity check and uncompressed-size cap. No object arrays / pickle.
Remaining: tests for concurrent writers, killed writers, corruption, mismatched
identity and caps; validate cap inputs; bounded disk eviction / quota policy.

### Broad recipe helpers — integration unfinished

`src/fpcode/workflows/sequence_search.py` has new helpers near lines20–501:

- `feature_bundle_columns`, codebook serialization helpers
- `_representation_inputs`: KPMS, feature-only, within-KPMS, joint and parallel
  routes; training-only transforms, explicit descriptive versus direct features,
  scoped annotations, representation caching
- `_record_stream`, `_blocks_of_record`, `_encode_record`: native bounds/gaps,
  categorical and actual nonnegative mixed-feature seqNMF inputs
- `_propose_motifs`, `_window_profile`, `_exact_rows`
- `_fit_recipe_model`: cached seqNMF, motif/feature-gate routes, expanded HSMM fits
- `_predict_recipe`: seqNMF / HSMM / exact / tolerant / finite grammar predictions

These helpers have parsed and coexist with passing existing tests, but have NOT
been exercised as an integrated broad evaluator. The old `evaluate_configuration`
still implements the narrower exact/seqNMF search. Do not launch that as the new
broad search. There is no public broad `evaluate_recipe` orchestration yet, and
`studies/sequence_viz/run_broad_search.py` does not exist.

Expected prepared record schema:
`feature_names`, `values[N,F]`, `parent_ids[N]`, `split` (train/held), `subject`,
`recording`, `cohort`, `clock`, `fps`, `start_frames`, `stop_frames`,
`source_vocabulary`. Helpers add labels, catalogue, model features and auxiliaries.
Identity must contain actual source/mask/clock/split/window provenance plus
`minimum_support`, `vocabulary`, `annotation_mapping`.

Current recipe fields include method, length, factors, prevalence, seed, feature
weight, seqNMF penalty/iterations/activation quantile, HSMM phase/duration limits,
entry/exit/skip/reentry/background/duration-learning/initialization/pseudocount/
variance-floor/target-count, representation route/resolution/bundle/clusters/
creator-seed/parent-weight/feature-role, tolerant edit budget, feature-gate multiplier
and grammar optional/repetition choices. Make conditional applicability explicit;
unsupported knobs must fail rather than silently do nothing.

Known integration risks / next fixes:

1. Verify finite grammar constructor and parser search-limit parameter names against
   their real contracts; the new branch is not covered by old tests.
2. Fix HSMM zero-proposal handling: currently it can fall into an unsupported-method
   error rather than returning a valid empty catalogue.
3. Restrict the first parallel view to execution features and keep spatial evidence
   in its own view. Current bundle selection can put spatial channels in both.
   Independent parallel one-hots are currently approximated as Gaussian auxiliary
   channels in HSMM; disclose or improve this, do not call it coupled multi-view fit.
4. Test empty held records, missing/all-NaN features, variable-length feature gates,
   JSON numeric finiteness and serialization round trips.
5. Ensure post-fit gate/edit/grammar choices get distinct prediction/job identities
   while reusing only the fitted component they genuinely share. Descriptive
   feature changes should not invalidate non-consuming fits.
6. Add shared evaluation, first-order/shuffle controls, physical-quality diagnostics,
   native overlap seed stability, artifacts and reporting. No phenotype/DA selector.

## Validation evidence and its limits

Latest focused command (before this documentation-only handoff):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python -B -m pytest \
  tests/unit/test_behavior_staged_hsmm.py \
  tests/unit/test_bout_feature_routes.py \
  tests/unit/test_behavior_method_contrast_runner.py \
  tests/unit/test_sequence_search.py -q
```

Result: **45 passed in 0.77 s**. New tests independently check Gaussian density,
feature-sensitive detections, persistent background, interior duration counts,
gap/heading units, training-only normalization, missing rows and categorical-ID
renaming / parent subtypes. Six changed modules parsed; `git diff --check` passed.
These tests do not cover the new broad workflow helpers or operational integration.

Earlier checkpoint: 730 CPU tests, architecture200modules, lint with no new debt
(3,079 accepted historical diagnostics), 146 folder maps and installed-wheel checks
passed. Those results **predate the new HSMM/features/helpers**. Run appropriate
full shared checks once against the final candidate; do not treat the earlier pass
as final validation. Do not regenerate lint debt to conceal new diagnostics.

## Remaining work, in practical order

1. Fix the known helper issues above; add a small complete synthetic broad evaluator
   covering each representation and detector route with train/held identity and
   native-gap assertions. Add meaningful cache and feature-HSMM fit checks.
2. Prepare verified whole-bout feature NPZs once per recording from existing exports;
   assert Jonathan joins against the verified native labels and preserve Karen
   masks. Freeze units, missingness, model identity and source fingerprints.
3. Implement the shared recipe evaluator and study policy / conditional recipe
   generator. Quantitative metrics should remain a vector: support, coverage,
   fragmentation, within-/between-type agreement, physical execution consistency,
   repeat/background structure, complexity and seed reproducibility. Flag metrics
   whose features were consumed by the model. Use native overlap matching, not
   numerical catalogue IDs, across seeds. Controls are exploratory diagnostics,
   not automatically calibrated p-values.
4. Implement a finite randomized scheduler in the existing study owner, reusing
   `src/fpcode/common/trial_runner.py` where suitable. Persist immutable configs and
   RUNNING/terminal states with SQLite single-parent writes, exclusive run lock,
   worker watchdogs, actual thread settings, STOP and explicit resume semantics.
   Empty successful results differ from incompatible recipes, numeric failure,
   timeout, memory limits and user stops. Resume only interrupted jobs; reuse
   successful artifacts and dependency caches.
5. Test bounded real-data routes for both cohorts, including feature-created
   vocabularies and expanded HSMM, plus scheduler timeout/stop/resume/duplicate
   writer behavior. Verify help/import/dry-run do not fit, initialize accelerators
   or write empirical products. Benchmark representative jobs before fixing caps.
6. Run relevant full CPU / architecture / lint checks; regenerate affected folder
   maps, check package/navigation owners when affected. Record exact limits and
   timing/resource evidence in the run manifest and CURRENT.
7. Freeze the runtime and prepared input/recipe manifest; launch the new separate
   nohup run, verify real worker progress and a durable completed checkpoint, then
   report log/status/stop paths. Preserve the old stopped service and its outputs.

## Proposed first tranche and operational design — not yet finalized

These are implementation proposals, not a ready configuration or a user-specified
budget. Finalize using bounded timing and actual host CPU/RAM/disk availability.

- Karen: explore 12 existing train + 4 validation animals with three animal-level
  folds; retain the 5 reserved test animals outside recipe selection. Jonathan:
  three animal-level folds over 15 mice, balanced as feasible by KI/WT. All repeated
  recordings from an animal stay in its fold. Frozen pre-existing KPMS training
  exposure must be disclosed separately.
- Shared bounded windows / event budgets per animal, with fixed matched sampling
  across comparable conditions. Record achieved support rather than pretending
  every animal supplied the same valid contiguous length.
- Balanced KPMS descriptive/direct, feature-only, within-KPMS, joint and parallel
  cells, with exact/tolerant/seqNMF/HSMM/finite-grammar families and conditional
  hyperparameters. Advanced recursive grammars, coupled multi-view discovery and
  physical-duration-conditioned transitions remain prospective unless implemented.
- Example finite tranche: 120 base recipes per cohort × 3 folds × 2 seeds = 1,440
  jobs for two cohorts. Adjust only after representative bounded timing.
- Example resources: 4 workers × 2 numerical threads, 300–600 s per-job watchdog,
  2 GiB per-worker RSS budget, outer MemoryHigh8GiB / MemoryMax12GiB, pause below
  8 GiB free RAM, checkpoint/stop below 80 GiB free disk, cache quota about16GiB.
  These thresholds need actual host checks; leave numerical-library headroom.
- Record NumPy/BLAS settings before numeric imports; avoid nested oversubscription.
  Cache source/transform/representation/model components by real dependency,
  coordinate concurrent writers, and bound hot RAM / disk. Avoid repeatedly
  hashing or reloading the large source pickle per trial.
- Existing `ops/monitoring/launch_behavior_search.py` is hardcoded to the old Karen
  runner. Extend/parameterize it before reuse; do not invoke its unchanged plan.
  A separate service could be named `broad-behaviour-search-20261013.service` and
  wrap `/usr/bin/nohup ...python -B -m studies.sequence_viz.run_broad_search`.
  Source edits after launch must not change the running frozen implementation.
- Log, atomic status, SQLite, service launch metadata / PID and STOP instructions
  should live under the explicitly selected new output directory. An existing PID
  alone is insufficient evidence that the search is working.

## Execution environment notes

Use `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python` without upgrading the
shared environment. Set `PYTHONDONTWRITEBYTECODE=1`; control BLAS/OpenMP threads
before numerical imports. `FP_DATA_ROOT` is a mirror root containing `data/`, not
the data directory itself.

In this chat ordinary sandboxed shell commands failed with a bwrap loopback
`RTM_NEWADDR Operation not permitted` error. Necessary read/check/export commands
succeeded using `sandbox_permissions: require_escalated` with scoped justifications
and automatic approval review. No rejection occurred. `apply_patch` worked normally.
Do not confuse this environment problem with a scientific or user-approval blocker.

At handoff no broad workers were launched; export, benchmark and test processes
were complete. No KPMS fits, source-data changes, signal reprocessing, environment
upgrade, commit/push or external notification occurred. This handoff ends the turn;
it does not install a timer or automatically resume when a token meter changes.
