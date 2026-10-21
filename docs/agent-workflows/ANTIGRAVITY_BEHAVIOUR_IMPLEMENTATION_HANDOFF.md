# Antigravity handoff: first implementation of behavioural abstraction modules

Date: 2026-10-12. Prepared for an Antigravity first implementation pass and later
Codex review. This is an implementation brief, not evidence that the planned
models already work on Karen. Read the actual checkout before applying it.

## Latest checkpoint: bounded first-pass verification complete (2026-10-12)

**The interrupted continuation checklist below is completed for the bounded
first-pass scope.** Preserve the existing module structure and unrelated dirty
work. Older audits and continuation instructions below are historical; do not
restart them or treat their failing counts as the current status. See
[CURRENT.md](../tasks/CURRENT.md) for the current task state and
[the verification report](../../outputs/behavior_verified_first_pass_20261012/verification_report.md)
for exact evidence and synthetic CSV products.

The final pass adds independent HSMM segmentation enumeration (marginal likelihood
versus MAP, terminal policies and structural zeros), context policies and
recording/vocabulary invariance, source/phase/posterior validation, recursively
immutable diagnostic ownership, primitive memory/clock/schema checks, shared-DAG
depth and shared-child composition tests, profile bounds and genuine lower-evidence
identity with context enabled/disabled. Physical timestamps require native frames
or declared clock origins plus FPS; the subsequent Gemini FPS-only implicit
rounded mapping is superseded. Historical diagnostic probes were preserved;
their implicit-mapping assumptions are not current API expectations.

Validation: **120 focused tests**, **642 full CPU tests** (six expected historical
warnings), architecture **198 modules / zero violations / zero cycles**, lint
**3,079 accepted diagnostics / no new debt**, **146 synchronized folder maps**, and
isolated wheel installation/CLI/doctor checks passed. Final broad gates ran once
after the source/test changes. NumPy/SciPy OpenBLAS pools were verified at four
threads with the lab Python 3.10.19 interpreter, `-B` and
`PYTHONDONTWRITEBYTECODE=1`. No empirical Karen fit/load, environment upgrade,
commit/push or notification occurred.

Composition is an **optional adjacent component** to inter/intra-chain
measurements: both consume the same frozen episode stream. Shared child episodes,
choices, optional parts and bounded loops can form assemblies; their intervals
can optionally feed the existing generic organisation/execution consumers.
This incorporates the supplied report's shallow shared-routine idea without
requiring another stage or refitting lower evidence. Automatic induction,
recursive grammars and context-conditioned routine controllers remain future
method choices, not completed functionality.

Next work is a deliberately scoped empirical pilot or an explicitly selected
future method. Synthetic evidence establishes the bounded library path, not
scientific utility or universal equivalence on recordings.

## Historical continuation checkpoint: Codex remediation interrupted (2026-10-12)

**At this dated checkpoint, implementation was unfinished and validation was not green.** The
user asked Codex to inspect Antigravity's claimed remediation and fix remaining
gaps. They subsequently asked to stop and leave a handoff. All edits below are
already in the shared working tree; preserve them and unrelated dirty work.
Do not restart the implementation or treat older completion claims as current.

### Changes already made in the existing owners

- `contracts/behavior_results.py`: optional native-frame/exposure associations;
  normalized/raw coordinate/schema/support checks; completion evidence kind and
  censoring consistency; derivation parent/grammar/trace fields; associated source
  UnitStreams with scoped-reference resolution; phase containment/order checks;
  posterior clock/model/support checks. The UnitStream constructor is unchanged.
  Diagnostics are copied into a read-only outer mapping, not yet deeply frozen.
- `observations/representations.py`: `native_feature_coordinates` rejects guessed
  row coordinates; physical-time series require explicit native mapping.
  `extract_associated_trajectory_features` records recording/role/unit/options
  provenance. Physical-time pose derivatives require FPS or a declared gap policy.
  Interval summaries now always interpret bounds as native coordinates.
- `analysis/execution.py`: explicit clock/recording/native mapping, frame exposure,
  weighted feature integrals/means, per-channel gap-aware geometry, unknown/censored
  completion denominators, and implemented `measure_onset_profiles` /
  `measure_phase_profiles`. Profiles average supported native frame cells without
  interpolating gaps; normalized phase shape retains separate physical pacing.
- `statistics/finite_composition.py`: memoized finite parser with log path scores,
  branch traces, structural zero weights, per-matched-length pruning disclosure,
  nullable/empty-rule rejection, and node/depth/repeat/window/work/memory/match caps.
  Retaining different lengths prevents local pruning from erasing the only length
  compatible with a following sequence component. Hard budget failures are explicit.
- Both grammar wrappers: use the same parser and shared block adapter; enforce
  candidate/rule bounds; retain explicit derivation parents/source streams; emit
  grammar log weights as model scores. Syntactic acceptance no longer claims motor
  completion or an observed physical exit, even inside a recording.
- `statistics/sequence_inference.py`: retained Antigravity's marginal HSMM forward
  recursion; added normalized model/shape/resource validation, stable survival
  accumulation, and prefix sums plus forbidden-emission counts for segment scores
  instead of repeatedly summing duration slices. No structural-zero floors.
- `episodes/staged.py`: ordered topology/PMF validation, consistent shared block
  handling/scoring, selected-path local log weights, source streams and diagnostics.
  Boundary entry is potentially left-censored. Removed fabricated completion 0/1
  probabilities. Immediate reentry remains separate. One-stage immediate reentry
  requires explicit disabling/rejection rather than pretending it is identifiable.
- `context/models.py`: configured inference/scoring share eligible blocks;
  normalized parameters and budgets are checked; unknown types require the explicit
  `error` or `uninformative` policy. Event posterior/physical projection stays
  separate from event-coordinate run diagnostics. Empty streams retain schema.
- `primitives/segmentation.py`: fitted-array/schema/unit validation, training-only
  scaling, explicit FPS/native mapping, bounded eligibility/fit/prediction/RLE
  output. BLAS squared-distance batches replace batch×K×features broadcasting.
  Batch size shrinks to the available cap; no entire held-out feature copy is made.
- `behavior/unit_adapters.py`: new `occurrence_blocks` reuses inclusive order,
  preserves touching validity boundaries and optional exclusion barriers. This
  helper is used by staged/context/grammar methods; existing adapters are retained.
- `workflows/behavior_layers.py`: motion fixture now has adjacent repeated routines;
  primitive fitting uses speed/angular velocity rather than absolute position.
  seqNMF X uses actual supplied bout codes, with frozen demonstration activations.
  Assembly grammar requires two episodes rather than accepting one renamed episode.
  Associated feature provenance and onset/phase profile outputs are returned.

No new source directory or competing implementation was introduced. The modified
source files are the seven planned modules, the existing rich-result contract,
two statistics helpers, the unit adapter and the existing synthetic workflow.
Tests were extended in their existing files, not a new test/report framework.

### Exact executed checkpoint and immediate fixes

One synthetic workflow run passed with counts:
`primitives=7, coarse=4, staged=3, grammar=2, exact=2, seqNMF=2, assembly=1, context=7`.
The assembly has **two source children**; phase profiles contain 80 rows. This
demonstrates the bounded path, not empirical convergence or scientific validity.

The final focused batch returned **108 passed, 2 failed in 1.11 seconds**:

1. `tests/unit/test_behavior_finite_grammar.py:38` compares probability floats
   with exact equality. Log-space reconstruction returns `.7500000000000001`
   rather than `.75`. Use an independently justified tight tolerance, retaining
   the .25/.75 expected masses, summed mass 1 and best-path expectations.
2. `tests/unit/test_behavior_layer_interfaces.py:219` still requires every staged /
   grammar result to be `completed`. Its staged routine starts at the recording
   boundary, so entry is unobserved and the new result is correctly censored.
   Its grammar result is syntactic acceptance with unknown physical completion.
   Assert these separate evidence meanings and retain common-consumer/count checks.
   Do not restore the misleading completion labels to satisfy this test.

The staged reentry unit test was already updated: first boundary invocation
censored, second observed reentry completed, both reach the final phase, and no
fabricated probabilities. The execution fixture now explicitly marks its censored
exit unobserved. New tests cover native bounds with small origins, missing profile
bins, exposure weighting, touching profile boundaries, grammar length-preserving
pruning, zero weights, invalid nullable rules and explicit search budgets.

Focused command (executed with four OMP/OpenBLAS/MKL/NumExpr threads, `-B`,
`PYTHONDONTWRITEBYTECODE=1` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`):

```bash
/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/unit/test_behavior_result_contracts.py \
  tests/unit/test_behavior_representations_execution.py \
  tests/unit/test_behavior_primitive_segmentation.py \
  tests/unit/test_behavior_staged_hsmm.py \
  tests/unit/test_behavior_finite_grammar.py \
  tests/unit/test_behavior_context_models.py \
  tests/unit/test_behavior_workflow.py \
  tests/unit/test_behavior_layer_interfaces.py \
  tests/integration/test_behavior_layer_imports.py \
  tests/regression/test_behavior_layer_moves.py \
  tests/architecture/test_behavior_layers.py
```

### Historical required continuation work (superseded by verified checkpoint above)

1. Resolve the two assertions above and run affected tests. Numerical tolerances
   and corrected observation semantics are deliberate, not permission to weaken
   identity/import/architecture checks or regenerate historical numerical goldens.
2. Add independent HSMM segmentation enumeration for several tiny sequences,
   duration support {1,2}, both terminal survival/exact endings, likelihood versus
   MAP and forbidden emissions. The old HSMM test does not enumerate all paths.
   Check the new prefix-score calculation and resource failures against these cases.
3. Add substantive tests for normalized/invalid model arrays, rich child resolution,
   derivation parents/duplicates, posterior clocks/model/support and normalized
   feature associations. Some validators are newly implemented but not covered
   adequately by the current suite. Diagnostics are only shallowly immutable.
4. Strengthen context tests: both gap policies, independent validity blocks / two
   recordings, vocabulary permutations, explicit unknown policy, impossible input,
   empty input, score versus inference diagnostics, and unchanged frozen evidence.
5. Test VQ budgets on fit and prediction, one-channel SignalSeries, named/unit
   schemas, constant/empty clusters and seconds/ms/ns mappings. Seconds conversion
   now deliberately requires a supplied native mapping and explicit FPS; do not
   reinstate silent rounding or 30-FPS defaults to satisfy the old diagnostic probe.
6. Review parser bounds carefully. Its memoized nullable/depth validation can reuse
   a node validated at a shallower depth; verify depth caps for shared DAG subtrees
   (store subtree height if needed). Global allocation accounting is conservative
   cumulative work, not an empirical peak-RSS measurement. Add repeat/optional
   ambiguity and resource tests; preserve trace/log-weight semantics in consumers.
7. Check remaining coordinate corner cases: FeatureAssociation native mapping
   validation currently uses `np.diff`, which can wrap on unsigned decreasing
   inputs; compare neighbouring values directly. Profile bins use floating native
   bounds, so either handle or explicitly reject coordinates beyond exact float
   integer representation. Validate profile output-budget argument types.
8. Strengthen the workflow test beyond checking dictionary keys: nonempty genuinely
   multi-child assembly, consistent lower seqNMF input, masks/profiles, and equal
   lower arrays/catalogues/refs with context on/off. Update README/API guidance and
   the current completion checklist to the final verified bounded scope.
9. Run strict lint on changed files and fix introduced diagnostics. Then regenerate
   maps and run one final batch of CPU, architecture, lint, maps and installed-wheel
   gates per the commands below. **No broad gate has run against this Codex edit
   set.** Antigravity's earlier 626-test / 20-probe claims predate these edits.
10. Save final synthetic evidence at one explicit new output root and update
    CURRENT.md. Preserve older outputs. The old audit probes intentionally contain
    assumptions that now require explicit mapping/evidence; adapt their expectations
    with stated semantics and keep their original checkpoint evidence intact.

No empirical cohort fit, pickle load, environment upgrade, commit/push or
notification was performed by Codex. Stop requested by the user is honored; this
checkpoint intentionally leaves the remaining implementation/verification visible.

## Historical completion audit — 2026-10-12

**All seven modules now contain real implementations; none remains an empty
placeholder. The required first-pass scope is nevertheless incomplete.** Preserve
this work and complete the missing operations below. The items are implementation
and correctness requirements; formatting, broad refactoring, advanced training
and empirical model selection can wait.

This audit supersedes the first-pass report's blanket completion claim. A marker
of `IMPLEMENTATION_STATUS = "implemented"` means a real operation exists, not that
every required operation or scientific invariant has been verified. The original
brief below remains the intended scope.

### What exists and what is still missing

| Existing owner | Real first-pass functionality found | Remaining implementation work |
| --- | --- | --- |
| `observations/representations.py` | Pose-derived centroid, physical speed, circular heading/yaw, frame-gap derivative masks, trajectory summaries | Coordinate-aware interval summaries, supported path steps across gaps, explicit feature association/provenance and seconds-coordinate support policy |
| `primitives/segmentation.py` | Training-only scaling, frozen k-means centroids, named feature assignment, unsupported labels and frame-coordinate RLE | Physical-unit schema validation, explicit native mapping for seconds inputs, removal of implicit 30-FPS/native-row assumptions, validated fitted arrays and enforceable memory budgets |
| `episodes/staged.py` | Configured duration-aware HSMM Viterbi and HMM decoding, phase records, immediate same-routine reentry, terminal survival | Genuine marginal HSMM likelihood/scoring, impossible-path handling, declared validity-block resets, honest entry/completion evidence and non-fabricated scores |
| `episodes/grammar.py` | Finite terminal/sequence/choice/optional/repeat matching through the shared parser, scoped child references | Reported pruning/search limits, valid-rule checks, stable derivation identities, explicit parent-to-derivation association, validity-block handling and separate syntactic acceptance |
| `assemblies/grammar.py` | The same parser composes episodes into assemblies; child type reuse works | The same parser/boundary/evidence issues as episode grammar; a nonempty end-to-end assembly demonstration |
| `context/models.py` | Genuine configured event-HMM forward/backward posteriors and MAP assignments; physical gaps remain unfilled; both unmatched policies exist | Normalized parameter validation, explicit unknown-type policy, inspectable scoring/diagnostics and stronger independent sequence/vocabulary/frozen-input checks |
| `analysis/execution.py` | Native interval durations, simple feature summaries, occurrence aggregation and phase-duration tables | Native-coordinate/recording/clock association, gap-aware per-channel geometry, censored completion denominators, and declared onset/phase profiles |

### Priority A: complete the scientific core before calling the pass finished

1. **Implement genuine HSMM marginal scoring.**
   `statistics/sequence_inference.py:298` currently places `best_score` into both
   `HSMMResult.log_likelihood` and `best_score`. This is Viterbi decoding, not a
   likelihood summed over possible segmentations. For two states, initial
   probabilities [.5,.5], alternating transitions, duration exactly one, two
   observations with emission probability one: the two legal paths each have mass
   .5, so marginal log likelihood is 0 while MAP score is `log(.5)`. Current output
   reports `log(.5)` for both. Add a log-sum-exp segment forward calculation with
   the same terminal survival/boundary semantics, keep the MAP trace separate,
   and expose real staged scoring. HSMM posterior inference is optional unless
   explicitly claimed; posterior arrays must never be filled with MAP indicators
   and labelled probabilities. `StagedEpisodeModel` currently has `decode_stream`,
   not the reported `predict`/`score` methods. Keep one deliberate API rather than
   adding redundant wrappers merely to reproduce inaccurate prose.
2. **Finish native-time feature consumption.**
   `analysis/execution.py:111` slices features with occurrence frame numbers as
   row indices. A feature series at native frames [100,101,102] with speed 20 and
   occurrence [100,103) returns zero supported samples/NaN rather than three/20.
   `compute_interval_trajectory_metrics` has the same row-coordinate assumption.
   Use actual TimeAxis coordinates and explicit mappings; validate recording,
   clock and conversion evidence, including externally supplied FeatureAssociation.
   The current execution consumer accepts features from another recording/clock.
   Preserve half-open bounds, native peak latency and feature-specific exposure;
   irregular physical sampling requires explicit integration weights or rejection.
3. **Preserve missing steps and declared inference barriers.**
   Trajectory positions at frames [0,1,5,6], x=[0,1,100,101], y=0 produce path length
   101; only the two observed contiguous steps support path length 2. Both position
   channels and actual native adjacency must govern supported steps. Staged and both
   grammar wrappers currently split only on a positive native gap. Two touching
   validity spans [0,5),[5,10) are still separate blocks: parsing A then B across
   them currently attempts an invalid parent and raises a UnitStream containment
   error. Split before inference/parsing so this yields valid independent results,
   not a cross-boundary match or late construction error. Keep exclusions as
   barriers through the existing eligible-view mechanisms.
4. **Handle impossible models before emitting detections.**
   A staged input consisting only of B, where initialization permits background X
   or entry A, is impossible. HSMM currently returns an ordinary empty detection
   result without diagnostics; HMM mode treats the `-1` impossible-path sentinel
   as a phase index and fabricates an episode. Return a distinct supported
   impossible-model outcome or informative rejection. Validate state/catalogue
   shapes, finite/-inf log parameters, normalized initial/emission/transition
   probabilities, normalized nonempty duration PMFs and declared topology. Never
   index phase names with an invalid state or turn impossible evidence into absence
   of behaviour. Unknown held-out types need an explicit caller policy; context
   currently supplies uniform emissions silently.
5. **Finish completion semantics and their denominators.**
   Both grammar wrappers mark every syntactic match `completed`, observed entry
   and exit true, even when the match ends at the recording boundary. Represent
   rule acceptance separately from physical/model termination; preserve unknown
   exits and censoring. Staged output currently assigns constant occurrence score
   1 and a completion `probability` of 0/1 derived solely from a MAP label. Omit
   unsupported probabilities or calculate a defined probability; deterministic
   path evidence is a different field. Starting in phase one at a recording
   boundary alone does not establish observed entry; honor the declared conditional
   start assumption and potential left censoring. Execution aggregation currently
   reports completion rate 0 for one censored invocation and no evaluable exits.
   Expose completed/deviated/unknown/censored counts and an explicit denominator;
   a wholly censored evaluable rate is undefined, while an all-call lower bound,
   if retained, needs a different name.

### Priority B: complete reusable contracts, parser bounds and layer swapping

6. **Finish grammar search/evidence, using the existing shared parser.**
   A choice matching A through weights .25/.75 with `max_derivations=1` currently
   returns only .25 with `truncated=False`. Implement meaningful search/pruning
   disclosure, preserve derivations and distinguish retained/summed mass from the
   best derivation. The current implementation is recursive enumeration, not the
   reported memoized/chart engine. Add bounded nodes/depth/repeats/window/work queue
   and retained matches, with a global work or memory cap; local output slicing
   does not bound intermediate branching. Validate nullable/empty roots and
   unsupported nullable repeats rather than silently discarding their probability
   mass. Two separate AB blocks currently both emit `AB_pos0_d0`; give derivations
   stable scoped identities and associate each explicitly with its parent occurrence.
   Resolve alternative derivations into an explicit exclusive occurrence view
   before downstream assembly/context consumption where necessary. Do not duplicate
   this fix in separate numerical parsers or weaken dependency fences.
7. **Complete rich-result validation.**
   BehaviorResult accepts a phase outside its parent's interval with a foreign
   recording/source reference and advertises the `phases` capability. Validate
   parent containment, source scope, phase order/duplicates and declared support;
   validate derivation-parent links and posterior model/catalogue/native support.
   StatePosteriorRecord currently lacks clock association, and normalized/raw
   features are checked only for equal length. Equal length is insufficient.
   Use explicit source associations to validate external children, and expose
   capability coverage/unknown evidence honestly. Preserve valid zero-event
   results and the existing UnitStream constructor. Do not replace actual
   validation with feature flags or a detector-name whitelist.
8. **Finish primitive schema/clock/resource handling.**
   A model trained with speed in px/s accepts an identically named mm/s feature
   without calibration. Validate units and fitted-array/schema dimensions.
   `segment_stream` replaces seconds coordinates with consecutive row numbers:
   samples at 10,10.1,11 seconds and 10 FPS become one [0,3) bout, losing the
   physical origin and gap. Require an explicit native association/mapping, or
   reject unsupported conversion; no default rate or silently invented native
   frame sequence. Keep training-only transforms and working frame-label/RLE
   support. Add validated K/sample/iteration/batch bounds and a byte budget covering
   distances, input copies and intermediate work. Existing sample/batch parameters
   are useful but do not bound `batch×K×features` allocations by RAM.
9. **Complete the promised generic profiles and nonempty integration.**
   `measure_phase_pacing` is a duration table; declared onset/phase feature profiles
   are still absent. Implement their bounded, mask-aware core in the existing
   execution owner, preserving physical pacing separately. The synthetic workflow
   emits primitives=4, coarse primitives=3, staged=2, grammar=1, exact=1, seqNMF=1,
   assemblies=0, contexts=4. Keep the working paths and add a known nonempty assembly
   example; an empty table alone does not verify higher-level execution. SeqNMF's
   current synthetic X uses an artificial cycling vocabulary rather than actual
   supplied bout tokens: use a consistent frozen lower sequence when claiming a
   detector-only comparison. Verify context off/on leaves lower arrays/catalogues/
   refs unchanged, exercise both context gap policies, multiple recordings,
   vocabulary permutations, different shapes/counts, zero events and coarse views.
   Configured context inference is already real; EM remains optional. Correct the
   report's nonexistent `EventContextHMM.predict/score`, count-vector/bin input,
   extra representation-channel and `fit_predict` claims to the actual API.

### Verification and structure for the continuation

- **Keep the existing owners and seven implementations.** Fix these operations in
  place, reuse the five permitted helper owners, and use the existing tests folders.
  Do not create alternate models, a replacement abstraction tree or copied study
  implementations. Preserve historical facades and unrelated dirty work.
- Audit interpreter: `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`, Python
  3.10.19, with `PYTHONDONTWRITEBYTECODE=1`, `-B`, and OMP/OpenBLAS/MKL/NumExpr set
  to four threads before numerical imports. Read-only doctor `--require cpu` passed
  with no backend initialized. Dirty baseline remains `3ec30fb` on
  `codex/modular-refactor`.
- **104 existing focused tests passed in 1.05s:** the seven new module/contract/
  workflow test files plus layer interfaces, fresh-process imports, relocation
  regressions and layer architecture tests. These checks demonstrate working
  bounded paths; they do not cover the reproduced gaps above. The first-pass
  report's 626-test/full-gate claims were not rerun for this documentation audit.
- Twenty targeted diagnostic probes record suspected gaps and scope markers.
  They deliberately target missing cases; their results are not an estimate of
  failure frequency. Saved evidence and the runnable reproducer are under
  `outputs/behavior_first_pass_review_20261012/{audit_probes.py,audit_results.json,
  validation.json}`. Turn the mathematical/boundary cases into substantive tests
  in their existing owners and rerun affected tests after each completed slice.
  Scoring API and memory-budget probes are interface inventories; verify numerical
  results and actual allocations too, rather than satisfying a parameter name.
- Reproduce the diagnostic observations from the repository:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    /home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python -B \
    outputs/behavior_first_pass_review_20261012/audit_probes.py
  ```

- Keep this checklist/current state truthful. The original implementation report
  is dated evidence; a new continuation report must distinguish implemented
  operations, corrected core gaps and deferred optional features. A passing
  import marker, empty assembly demo or renamed MAP score does not complete scope.
  Run broad gates once against the finished implementation candidate as below.
  No Karen refit, pickle load or empirical execution is part of this audit.

## 0. Copyable starting instruction

> Complete the bounded first-pass scope and current completion checklist in this
> handoff, preserving the existing real implementations and
> repository structure, scientific units, masks, identities and old public APIs.
> Finish shared result validation and missing numerical/consumer operations in
> their existing owners. Use existing helpers and small generic numerical
> mechanisms. Add independent mathematical and interchangeability tests, plus an
> end-to-end synthetic workflow. Do not replace real algorithms with placeholder
> outputs or claim capabilities not represented by validated data. Keep code in
> the named owners, avoid clutter, and leave an exact implementation/review report.
> Verify at milestones; run the required broad gates once on the final candidate.
> Default execution is synthetic and CPU-only. Cohort runs are a separate explicit
> runtime scope. Follow this document's nontrivialities and acceptance criteria.

The goal is a useful, reviewable implementation pass across all seven modules.
A bounded real algorithm is sufficient. Unrestricted grammar induction, a new
neural hierarchy, full automatic discovery from frames to minutes, and proof of
PFF–monomer differences are outside this first pass. If a requested capability
cannot be implemented correctly, record the precise limitation and preserve a
truthful status; do not quietly downgrade its meaning to make a checklist green.

## 1. Read these first, then inspect only relevant source

Repository root on the lab machine:
`/home/fsp585/Desktop/FiberphotometryCode`.

1. [Current task state](../tasks/CURRENT.md).
2. [Root instructions](../../AGENTS.md) and [capability contract](../architecture/UNIT_CONTRACT.md).
3. [Library guidance](../../src/fpcode/AGENTS.md) and
   [behaviour guidance](../../src/fpcode/behavior/AGENTS.md).
4. [Implemented module inventory](../../src/fpcode/behavior/README.md).
5. [Accepted abstraction plan](../architecture/BEHAVIOURAL_ABSTRACTION_PLAN.md).
6. [Validation selection](validation.md).
7. If modifying study or operations files, read their applicable `AGENTS.md` too.

Use `rg` and the generated folder maps for targeted discovery. Do not recursively
dump all data, source, historical reports or terminal environments into context.
Treat dated reports as checkpoints; inspect source for current behaviour.

### Baseline to preserve

- At handoff: dirty `codex/modular-refactor`, committed SHA
  `3ec30fb46607f1d35b231d7773a47168cf4546a0`. The working tree contains substantial
  legitimate uncommitted work. Re-read status; a clean reset loses that work.
- The structure checkpoint passed 598 CPU tests, architecture at 193 canonical
  modules with zero violations/cycles, strict lint with 3,079 accepted historical
  diagnostics and no new debt, 146 synchronized navigation directories, and an
  installed-wheel check outside the checkout.
- Evidence: `outputs/behavior_layer_structure_20261012/{validation.json,cpu.log,
  structure.log,package.log,doctor.json,relocation_manifest.json}`.
- Eleven existing numerical modules were relocated whole. Their numerical
  function/class AST hashes matched the originals. Old import paths preserve
  callable identity, including private helpers.
- [Move/API manifest](../../registry/BEHAVIOR_LAYER_MOVES_20261012.json) is dated
  relocation evidence. Do not overwrite it or regenerate fixtures to excuse a
  changed result. New development has its own provenance and review evidence.
- Earlier Karen atlas/report products remain valid records of their checkpoints.
  Do not overwrite them to make a new method appear previously validated.

Do not reset, clean, delete unrelated files, upgrade the shared environment,
change lint debt, commit/push or send notifications as incidental work.

## 2. Scientific intent and the meaning of independence

The framework separates four choices:

| Choice | Meaning | Example |
| --- | --- | --- |
| Abstraction | Kind of unit | Primitive, episode, assembly |
| Resolution | Granularity inside that abstraction | Raw syllables versus L2/L3 classifications |
| Method | How units/relationships are inferred | Exact chains, seqNMF, staged episode model |
| Analysis | Measurements applied to those units | Usage, pacing, completion, repeats, switching |

Observations/fine movement → primitives → episodes → optional assemblies is a
useful unit ladder. Context is an optional parallel track that can influence
several levels. It need not be the top of an ordered composition tree.

Inter/intra analysis is reusable across unit levels. A stage repeating inside an
episode, a completed episode invoked again, and A→B→A circulation among episodes
are different quantities. A loop model and a loop statistic are also different:
the model defines possible composition; the statistic describes observed calls.

Independence means replacing one method while keeping its lower evidence,
resolution, split, clock and applicable analysis definitions fixed. It does not
mean equal catalogue IDs, equal occurrence counts or unchanged biological results.
Some methods use additional continuous evidence; declare that change explicitly
instead of presenting the comparison as a detector-only swap.

Select definitions on training/validation support, stability, interpretable
execution and held-out predictions. Do not select them because their PFF–monomer
contrast has the smallest p value. More occurrences do not increase independent
animal N. Use objective movement descriptions rather than inferring intentions,
neural control modules, or motivation from a kinematic state.

## 3. Keep the structure: explicit anti-clutter rules

The current owners are already appropriate:

```text
src/fpcode/
  contracts/                  Shared records, units, identities, validated payloads
  behavior/
    observations/             Physical features and trajectory representations
    primitives/               Short-unit segmentation, bouts and resolution views
    episodes/                 Exact, seqNMF and other episode detectors
    assemblies/               Larger compositions of episode results
    context/                  Optional persistent context models
    analysis/                 Shared within/between-unit measurements
    unit_adapters.py          Explicit representation conversions
    compatibility.py          Consumer requirements and incompatibility reports
  statistics/                 Generic numerical inference and statistics
  workflows/                  Explicit composition of capabilities
studies/<project>/            Cohort policy, hypotheses, selected empirical runs
tests/{unit,integration,regression,architecture}/
outputs/<new-run>/            Generated products and executed verification evidence
```

**Structure preservation is a deliverable, not a cosmetic preference.**

- Implement each of the seven planned files in place. Do not create competing
  `new_*`, `v2_*`, `final_*`, `fixed_*`, `experimental_*`, `misc`, `scratch`,
  `utils` or cohort-only copies of reusable algorithms.
- Do not create a new top-level package, an additional numbered layer tree,
  a general plugin manager, service container or reflective auto-discovery system.
- Use a small number of coherent classes/functions in each owner. One class does
  not need a separate directory, CLI, config directory or registry.
- Keep fitted parameters next to the relevant method; shared payloads belong in
  contracts; generic numerical machinery can be reused from statistics.
- Keep old facades as identity reexports. New canonical imports use actual owners.
  Do not put implementations back in old `behavior/sequences/*` facades.
- Temporary development files go in `/tmp`; generated outputs have one new explicit
  output root. Do not leave copied models, notebooks, logs or one-off shell drivers
  throughout source directories.
- Update existing README/status/tests. Avoid a second overlapping architecture
  document that disagrees with this one or the accepted plan.
- Final report lists every added file with its owner and reason. Also list any
  removed/moved file and compatibility impact. Unjustified additions fail review.

### Bounded additional files that may genuinely be needed

Prefer existing records and mechanisms. If necessary, these additions have clear
owners; they are a limit on architectural expansion, not a requirement to create
all of them:

| Potential file | Justification |
| --- | --- |
| `contracts/behavior_results.py` | A common result envelope and validated optional stage/completion/reference/posterior payloads; reuse SignalSeries for features |
| `contracts/composition.py` | Immutable finite grammar specification and parse records shared by primitive/episode composition; data definitions only |
| `statistics/sequence_inference.py` | Generic array/topology HMM/HSMM inference reused by staged and context methods |
| `statistics/finite_composition.py` | Generic bounded symbolic composition/parser reused by the two grammar wrappers |
| `workflows/behavior_layers.py` | One explicit synthetic/method-comparison composition workflow; no implicit fitting on import |

Do not duplicate a parser in both grammar modules or HMM numerics in staged and
context just to satisfy current fences. Shared numerical helpers operate on
generic arrays/symbols/contracts and never import behavioural detectors. Contracts
must not become a hidden home for fitting, parsing or dynamic-programming engines.
This layout works with current top-level dependency directions; no broadening of
the dependency rules should be necessary.

Adding a different structural owner requires a concrete reason and targeted
boundary tests. Avoid changing the plan merely because an import is inconvenient.

## 4. Existing interface: preserve it and extend honestly

Current definitions are in `contracts/behavior_units.py`:

```python
FrameSpan(start: int, stop: int)  # native half-open [start, stop)
UnitType(type_id: str, description: str = "")
UnitOccurrence(occurrence_id: str, type_id: str, interval: FrameSpan,
               score: float | None = None, score_kind: str | None = None)
UnitStream(session: SessionKey, level: str, model_id: str, clock: str, fps: float,
           valid_intervals: tuple[FrameSpan, ...], catalog: tuple[UnitType, ...],
           occurrences: tuple[UnitOccurrence, ...])
```

- Levels currently accepted: `primitive`, `episode`, `assembly`, `context`.
  Raw observations use existing `PoseSeries`, `TimeAxis`, `SignalSeries` records.
- Scores are finite with explicit `model_score`, `posterior` or `probability` kind;
  probability kinds are bounded [0,1]. That field does not magically calibrate
  a detector score or provide a state-posterior matrix.
- Valid spans are ordered/disjoint. Touching spans remain separate declared
  boundaries. Every occurrence fits completely within one valid span.
- Occurrences may overlap, have partial coverage or be empty. Preserve input order
  in the immutable result; chronological views are explicit.
- Full `SessionKey`/model identity matters. Catalogue numbers are not cross-fit IDs.
- `fps` is an explicit constant conversion on one clock. It does not synchronize
  clocks or define variable-rate data without a mapping.

Keep `UnitStream` as the interoperable interval projection where possible. Add a
small result envelope around it for validated richer payloads. Do not break its
existing constructor or add mandatory detector-specific fields to every method.

### Suggested optional payload responsibilities

Names may differ; invariants must not:

1. **Feature association.** Reuse `SignalSeries` for `(samples, channels)` values,
   named units and per-channel validity. Add only the missing recording,
   coordinate-frame and provenance association. Preserve raw physical features
   separately from training-normalized values.
2. **Scoped occurrence reference.** Recording key + source model identity +
   occurrence ID. Bare `call_3` is not globally unique.
3. **Phase/constituent record.** Parent reference, phase/child identity, order,
   native interval or source occurrence reference, and evidence kind.
4. **Completion/boundary evidence.** Per occurrence: observed entry/exit, final
   phase reached, completed/deviated/unknown semantics, left/right censoring and
   any actual posterior with its defined meaning.
5. **Composition alternatives.** Rule/grammar identity, derivation identity,
   source child references, score semantics and search truncation status.
6. **Posterior association.** Explicit observation coordinate, sample/native
   mapping, state catalogue, probabilities, masks and model/recording identity.

Validate foreign references, duplicate IDs, parent containment, same-recording/
clock association, supported coverage conventions and array shapes. Some stages
tile a parent, others leave gaps or represent alternatives; declare that policy.
Validate probabilities only on supported rows. Arrays use owned read-only copies
through existing `frozen_array` infrastructure.

Keep method-native W/H templates, fit diagnostics and parameters alongside this
common view. Do not embed fitted detector instances into shared records.

`compatibility.py` must derive rich capability availability from validated payloads.
Adding `"supports_completion": True` or whitelisting the word `"stages"` without
data is not an implementation. Plain UnitStream should still reject analyses
needing genuine completion/progression. Distinguish available schema from an
individual recording with zero observations or wholly unknown completion.

## 5. Required first-pass scope for each module

These are deliberate bounded targets. Implement the real operation, document its
limits, and add independent tests before increasing sophistication.

### 5.1 `observations/representations.py`

Implement label-independent physical feature construction from explicit,
confidence-masked pose/trajectory inputs. Return associated SignalSeries-style
features with named units, native sample coordinates, per-feature support and
provenance. A useful core includes selected-landmark/centroid displacement and
speed, planar heading/yaw when supported, and interval trajectory summaries.
Body-relative features can be optional rather than an excuse to invent 3D motion.

- Role/landmark selections, thresholds and filtering choices come from the caller.
  Do not hardcode Karen chambers, syllable IDs, FPS or bodypart lists into algorithms.
- PoseSeries/TimeAxis may carry frame coordinates without a physical sampling
  rate. Physical derivatives then require caller-supplied FPS or an explicit
  frame-to-time mapping. Reject missing conversion evidence instead of borrowing
  a cohort FPS; seconds-based inputs use their declared physical coordinates.
- PoseSeries.valid marks finite supported observations; confidence acceptance is
  an additional explicit policy, potentially different by feature/landmark.
- Derivatives and smoothing reset across frame gaps and invalid-feature support.
  Do not connect two supported points across an unobserved motion trajectory.
- Preserve raw features. Temporal smoothing affects information bandwidth and must
  be recorded separately from native acquisition rate.
- Reuse `observations/kinematics.py` helpers when their semantics fit. Do not
  import `analysis/phenotyping.py` to obtain `frame_kinematics`: that implementation
  conditions validity on labels and the observation→analysis dependency is forbidden.
- Behaviour code cannot simply import `tracking/pose_measurements.py` under the
  current top-level boundary. A workflow can compose tracking outputs explicitly;
  any extraction of genuinely generic geometry needs its own unchanged-old-API
  parity evidence. Basic new observation-only calculations are often simpler.

Minimum evidence: analytical translation/stationarity, heading wrap, nonzero
frame origin, missing frames, low-confidence channel, masks and physical units.

### 5.2 `primitives/segmentation.py`

Implement one clearly named **vector-quantized primitive baseline** using a
bounded frozen-centroid clustering model on caller-selected continuous features.
It is an adjacent baseline, not KPMS or a proven equivalent behavioural ontology.

- Fit only supplied training rows/sequences, with explicit K, seed, iteration cap,
  sample cap and memory budget. Fit normalization on training evidence alone.
- Use a small NumPy/SciPy implementation or an existing lower generic helper.
  Do not import `analysis/variants.py` just to steal clustering code, and do not
  add a new mandatory ML package or upgrade the lab environment.
- Record feature names/order, units, location/scale and selected catalogue identity.
  Held-out assignment verifies the schema; named reordering can be an explicit
  adapter, never a silent positional assumption.
- Unsupported rows receive -1; do not drop them before RLE. Use the existing
  `from_frame_labels` adapter to produce a primitive UnitStream.
- Specify finite handling for constant features, insufficient training support,
  duplicate/empty clusters and deterministic ties. Use bounded distance batches,
  not an N×K×F broadcast across an entire cohort without memory accounting.
- Classification/resolution remains separate. Do not bake L2/L3 merges or human
  theme labels into the fitter.

Minimum evidence: separable synthetic features, train-only transformations,
unchanged prototypes under held-out outliers, deterministic configuration,
schema mismatch handling, and gaps breaking adjacent equal-label runs.

### 5.3 `episodes/staged.py`

Implement a small configured categorical staged episode model with background,
ordered phase topology and explicit entry/progression/exit rules. The first pass
requires genuine finite explicit-duration HSMM score/decode operations and a
useful rich result. A configured parameter model is sufficient; parameter learning
is optional. Do not expose a dummy training API.

Keep modes separately named:

| Mode | First-pass scope and semantics |
| --- | --- |
| Explicit-duration HSMM | Required: finite duration PMFs, segment-based likelihood and duration-aware traceback |
| Stage HMM | Optional staged comparison: geometric phase dwell in the explicitly declared observation coordinate; generic HMM inference is still needed for context |

If HSMM implementation remains unfinished, report that missing required scope
explicitly. An HMM baseline does not satisfy the HSMM requirement by renaming it.

The preferred first duration coordinate is **observed primitive/event counts**.
An explicit-duration event-count model is a real HSMM. Report physical durations
by native intervals, while stating that seconds are not its duration likelihood.
A physical-time model requires a declared grid or a different duration treatment;
simply exporting seconds does not change the fitted clock.

Required details:

- MVP input is one primitive UnitStream presented through a chronological,
  exclusive view. Reject overlapping inputs until an explicit upstream selection
  is supplied. Preserve source references when ordering; do not change source IDs.
- Reset at every declared validity boundary and excluded-event barrier. For this
  bounded staged MVP, unmatched native gaps also split inference: a parent routine
  must not silently span unobserved or skipped evidence. A future bridging model
  requires a separately declared observation/support policy.
- Parameters include catalogue mapping, initial distribution, legal transition
  mask and categorical emissions; duration PMFs are explicit in HSMM mode.
- Every recording/validity block has separate inference. Never concatenate sessions
  and learn transitions across artificial joins.
- Preserve structural zeros as `-inf` in log space. Use stable log-sum-exp for
  likelihood/posteriors and a separate MAP/Viterbi traceback.
- Support variable phase durations. Permit partial observed invocations with honest
  boundary evidence rather than dropping every non-textbook occurrence.
- Split final-phase→same-routine-entry into two invocations. Equal routine labels
  alone do not identify those boundaries. Preserve transition/segment traceback.
- At recording/missing boundaries, final-phase occupancy without an observed exit
  remains censored. Final phase reached and invocation completed are separate facts.
- For the bounded MVP, reject ambiguous one-stage same-routine reentry semantics
  or return explicitly unknown boundaries. Do not fabricate them from a self-loop.
- Catalogue/episode parameters are shared across supplied recordings. A separate
  context model may later influence recruitment; no macro-context fit is required
  to make this episode method usable.
- A seqNMF initializer is optional and workflow-supplied through an explicit
  neutral template/config record. The staged method must work without seqNMF.

For HSMM right censoring, terminal occupancy uses survival mass under the declared
boundary model. With `P(D=1)=.4, P(D=2)=.6`, one observed step followed by recording
termination has `P(D>=1)=1`; two observed steps have `P(D>=2)=.6`. Treating each
recording end as an exact duration biases results shorter. An explicit terminal
completion observation is different and may use an exact-duration term.

Full left-truncated likelihood is harder. A first pass can state a conditional
start assumption and conservatively mark boundary invocations as potentially
left-censored. Do not call that a fully implemented stationary left-censoring
model, and do not fit uncensored duration distributions from those invocations.

Parameter learning is optional only where its absence is explicit. If implemented,
use real independent-sequence sufficient statistics/EM and convergence diagnostics;
do not simulate an apparent fit or silently reuse fixed defaults. All training
transformations and priors need caller-defined scope.

Minimum evidence: enumerate tiny legal paths/segmentations, verify likelihood and
MAP trace, exact-zero forbidden transitions, immediate reentry, censored ending,
validity reset and physical reporting with variable bout durations.

### 5.4 `episodes/grammar.py`

Implement detection from a **caller-specified finite grammar**, not unrestricted
unsupervised grammar induction. A useful specification has:

- Terminal/type match or explicitly interchangeable terminal set.
- Sequence.
- Weighted choice.
- Optional component.
- Bounded repetition with minimum/maximum and declared count weights.

Shared immutable rule data and parse records belong in contracts. Shared bounded
symbolic inference belongs in a lower generic numerical owner. This module supplies
primitive-input validation, eligible sequence views, physical projection and
episode results; it should not duplicate the generic parser.

For the MVP require a chronological exclusive primitive stream. Reject overlapping
inputs until an explicit upstream view is supplied. Match within declared eligible
blocks; missing, excluded or disallowed unmatched intervals are barriers.

Return episode candidates, scoped child references, derivations, actual scores,
and truncation information. Preserve multiple derivations when supported. If two
branches accept the same token with weights .25/.75, an exhaustive sequence
likelihood sums to 1; the best derivation has .75. Those are different outputs.

Reject recursive rules, unbounded loops, non-progressing cycles, nullable repeated
children if unsupported, and an empty-match root. Bound grammar size, repeat count,
window length and retained candidates. If a budget truncates search, say so; do
not label a retained-candidate normalization as an exhaustive global posterior.

Syntactic acceptance is an observed match to a declared rule. It is not evidence
of intended task success or observed motor termination. Rich completion payloads
must distinguish those meanings.

Minimum evidence: hand-enumerated derivations, sequence/choice/optional/repeat
boundaries, impossible/empty rules, gap barriers and retained ambiguity.

### 5.5 `assemblies/grammar.py`

Reuse the same finite composition machinery on **episode** results. This module
validates operand level, preserves episode definitions/references and emits an
assembly stream and rich parse information. It cannot import an episode detector.

Initially require one associated exclusive chronological input stream. Multiple
overlapping libraries do not automatically form one meaningful token sequence;
sorting their calls is not a sufficient association/selection policy.

Allow a child type to be reused in several assembly definitions. The catalogue of
definitions may be a DAG while an individual invocation has specific child refs.
Keep overlapping alternative assemblies where represented; do not silently force
one parent or reinterpret every run/return window as a discovered assembly.

Minimum evidence: the same numerical parser used at both levels, shared children,
correct native containment/reference identities, unchanged child catalogue and
explicit overlapping/ambiguous parent handling.

### 5.6 `context/models.py`

Implement one clearly named **categorical event-context HMM** with configured
parameters, real inference and bounded optional EM. Its observations are supplied
unit events; persistence is initially measured in event steps. It models changing
recruitment distributions, not an ordered routine definition and not a shMoSeq
reimplementation or context HSMM.

- Use generic HMM inference shared with staged HMM mode; do not import staged.py.
- Consume common lower results, and record the actual operand level/catalogue.
- MVP inputs are a single chronological exclusive UnitStream at `primitive`,
  `episode` or `assembly` level. Reject unresolved overlaps and mixed libraries;
  an explicit upstream view can supply an eligible sequence without changing IDs.
- Keep lower units frozen; do not silently revise episode definitions while fitting
  context. Group metadata does not set emission probabilities.
- Record the observation coordinate. Event slots with timestamps are still event
  observations. A future fixed-frame/bin mode needs explicit binning and support.
- Do not infer a complete continuous context track across sparse unmatched time.
  In event mode, project posterior assignments onto the observed event intervals,
  split physical intervals across unobserved/unmatched gaps, and retain the
  event-coordinate state-run description separately. Only a declared model of
  that intervening time can support a continuous physical-time projection.
- Declare unmatched-time policy: `event_steps` can connect observed events inside
  one validity block without modelling intervening seconds; `break` splits at
  those gaps. Missing validity and excluded-event barriers always reset inference.
  Test both policies. Neither policy authorizes filling the physical gap.
- Fit/decode independent validity blocks; aggregate sufficient statistics without
  cross-session transition counts. Handle unknown held-out types explicitly.
- Initial state/emission/transition probabilities and any sticky prior are explicit.
  A persistence prior is not evidence of a natural additional behavioural level.

Minimum evidence: tiny HMM enumeration, event-coordinate semantics, separate
sequences, native gaps, vocabulary permutation, EM diagnostics if fit exists,
and unchanged lower input/catalogue after context is added.

### 5.7 `analysis/execution.py`

Implement generic interval-based execution measurements from common unit results
and associated continuous features. Support primitive, episode and assembly inputs
through the same applicable consumer.

Useful first outputs include physical duration, feature-supported count/exposure,
sum/mean/SD/peak, native peak latency, displacement/path summaries when original
positions support them, and declared onset/phase profiles. Add phase pacing and
completion summaries only from genuine rich payloads.

- Match recording, clock and actual native coordinates; equal length is insufficient.
- Preserve feature-specific validity and denominators. Unknown/no-support conditional
  measures remain NaN; counts can be zero with positive declared exposure.
- Means over variable-duration bouts need time weighting for a physical-time mean.
  Distinguish a mean of occurrences from a pooled time/feature summary.
- Phase-normalized shape and physical pacing are separate outputs. Do not interpolate
  across unsupported gaps simply to fill every phase bin.
- Entry/exit support and censoring accompany boundary-dependent metrics. Calling a
  routine incomplete requires an operational deviation/termination definition;
  loss of observation is unknown, not behavioural failure.
- Import shared payloads and observation helpers, never episode/assembly/context
  detector classes. Keep the existing organisation measurements reusable.

Minimum evidence: hand-calculated interval arithmetic, per-channel support,
different unit counts/levels, duration stretching, unsupported capability rejection,
and distinction between observed completion and censored termination.

## 6. Nontrivialities: do not approximate away these details

### 6.1 Native time, rows and different clocks

Native frame numbers, array positions, original bout rows, merged bout rows,
seqNMF padded positions and physical seconds are different coordinates. Never
slice an array with native bounds until a verified mapping says they are row
positions. Nonzero origins and dropped frames must appear in tests.

Use half-open bounds everywhere. A 60-FPS span `[100,160)` is one second.
Two intervals touching at 160 have zero intervening native gap. Do not add an
extra frame or treat the exclusive stop as an observed sample.

A clock string match alone does not establish row alignment. Require actual
coordinates/associations. Do not infer FPS from length, maximum frame, a remembered
40/80-Hz convention or the apparent duration of a historical cohort.

Current native adapters reject fractional bounds; explicit conversions may accept
exactly representable coordinates, but rounding/truncation must never hide an
unknown mapping. UnitStream permits arbitrary Python integers; existing organisation
analysis has an explicit int64 limit that must remain honest.

### 6.2 RLE and resolution change the modelled clock

Raw `A A A A B B A A` frames become bouts `A B A` with lengths `4,2,2`.
Coarser classification can merge neighbouring bouts again. Preserve raw evidence
and derive views rather than overwriting source labels.

A lag of three bouts has no fixed number of milliseconds. Refit or explicitly
adapt token-based models when that view changes; native features should remain
reusable. A grammar asking for adjacent `A A` may be unobservable after RLE even
if the mouse performed repeated micromovements within the A bout.

For Karen, 14/19 named primary exact anchors begin and end with the same syllable.
RLE/template structure can prevent strict adjacent copies. Zero strict self-calls
for those templates is not proof of no repeated underlying behaviour.

### 6.3 Validity, unmatched time and background

Observed support, assigned primitive support, episode coverage and supported
feature exposure are distinct. Missing observations are not model background;
unmatched valid time is not automatically rest; excluded detections are not absent
events. Touching declared validity blocks are still distinct boundaries.

Keep all native calls until an explicit analysis-view policy. Resolve overlaps
on the inclusive library before exclusions. Removing B from A B A must not invent
A→A. Large gap tolerances still cannot cross missing boundaries.

Empty detections retain observation support and catalogue. Positive exposure with
zero calls gives zero rate; no successor opportunities gives undefined transition
probability. Do not replace that NaN with zero to make plots or tests convenient.

### 6.4 Motion support and physical interpretation

- One invalid paw need not invalidate all centroid motion: support is per feature.
- A backward derivative at frame f uses f−1; its first sample after a gap is
  unsupported. Filtering must not leak future/previous segments across gaps.
- Heading crosses ±π: use circular differences. Endpoint wrapped heading change
  differs from accumulated rotation during multiple turns.
- Projected 2D body-relative paw motion includes translation/rotation/scale-frame
  choices. It is not paw contact, body height, stride length or calibrated mm.
- A pixel threshold such as body length >1 is not a generic millimetre threshold.
- Path length is a sum of displacement magnitudes; net displacement is an endpoint
  difference. Summing bout net displacements does not give episode net displacement.
- Record smoothing windows and units. Native 60-Hz samples with 150-ms smoothing
  do not retain an unfiltered 60-Hz movement bandwidth.

### 6.5 Numerical inference and forbidden transitions

Use float64 for the small reference inference path and stable log-sum-exp. Zero
probability is `-inf`, not `log(1e-12)` for a forbidden arc. Positive numerical
floors belong only to an explicitly declared learned-emission regularization.

Keep forward likelihood, MAP path score and posterior probabilities separate.
Validate finite supported posteriors and row normalization. Detect impossible
sequences and report them; all-`-inf` rows must not normalize into NaNs and then
be accepted as a successful model.

Duration PMFs normalize on declared support. HSMM segment recursion must account
for initial/boundary semantics, terminal survival and legal segment transitions.
HMM self-loop durations, multi-stage sums of geometric durations, log-duration
emissions and arbitrary explicit HSMM duration distributions are different models.

### 6.6 Completion, continuation and same-routine restart

Do not decode invocations merely by RLE of routine ID. A final-phase→first-phase
transition can restart the same ID without changing its label. Preserve boundary
arcs/segments before making the interval projection.

Reaching the final stage at the last observed sample supplies final-stage evidence,
not an observed exit. Boundary censoring, actual deviation, syntactic rule
acceptance and observed model completion need separately named evidence semantics.
A stage ending at a recording boundary is not necessarily a biological end.

One-stage self-dwell versus reentry may be observationally unidentifiable. Even
an HSMM can prefer latent segmentations because of priors, not observable restart
evidence. Use the bounded model's scope/rejection rules and expose uncertainty.

### 6.7 Identity, model output and ambiguous composition

`from_frame_labels` creates `bout_i`; native table adaptation creates `call_i`
unless an explicit ID column is supplied. Those IDs depend on original ordering.
Preserve them before sorting/filtering or making child references. Never regenerate
the IDs from filtered rows and silently point to different occurrences.

Match IDs with recording and source model identity. Catalogue-code crosswalks
are local; integer factor 3 is not the same routine across fits/resolutions.
Native interval overlap and constituent/kinematic correspondence can establish
candidate associations, including many-to-many ones.

Full seqNMF templates have weighted features/lags and permit variation. Dominant
tokens are a stricter description. Do not redesign the existing detector as exact
symbolic matching or use loss of dominant-label agreement as proof of motif loss.

Best-parse scores, activation scores and log-likelihoods are model scores unless
their probability interpretation is actually implemented. Uniform score kind
does not make scores comparable across separate fitters or different-length units.
Overlapping parses may share child occurrences; avoid inflated union coverage and
describe whether inference counts alternatives or selected observed occurrences.

### 6.8 Training, covariance and independent evidence

Fit scalers, PCA, centroids, catalogues and hyperparameters on training evidence.
Do not use held-out extremes or outcomes to improve their appearance. Animal-level
splits are necessary for animal generalization; overlapping windows and paired
recordings do not become independent by shuffling rows.

If mixed continuous/categorical emissions are added later, feature scaling and
likelihood weighting are explicit. Many correlated Gaussian features can dominate
tokens. Diagonal Gaussian independence is an approximation. Missing dimensions
can be marginalized under a declared model; NaN→0 fabricates a measured mean.

More training samples can help fit a model, but many calls from one mouse do not
increase independent group N. Repeated weeks and many adjacent method choices are
correlated evidence, not replication.

## 7. Old prototypes: audit references, not ready-made canonical implementations

Relevant source exists under:

- `studies/sequence_viz/probabilistic_episode_pilot/`
- `studies/sequence_viz/export_chatgpt/hierarchical_subroutine_model.py`
- `studies/sequence_viz/hierarchical_pilot/`

Read source without importing it merely to inspect capabilities. Audited concerns
include hardcoded study themes/FPS/paths, stale namespaces, native-frame-as-row
slicing, geometric bout-count phases described too broadly as HSMM durations,
model initialization of JAX/GPU at import, and forbidden transitions made positive
by numerical floors. Older decoding can call a terminal final phase completed
without an observed exit. Historical comparison prose overstates seqNMF rigidity
and contains inconsistent counts.

Do not copy these claims or code wholesale. Extract only a mechanism whose
assumptions you can state and verify independently, remove cohort/import side
effects, preserve the old study source, and show bounded equivalence where reuse
is claimed. Increased detection coverage alone does not establish valid routines.

## 8. Order of work: complete small vertical slices

1. **Inventory and baseline.** Record checkout, interpreter, existing passes,
   owned files and intended bounded APIs. Read status-specific import tests.
2. **Common result sidecars.** Add only needed typed records, validators and
   payload-backed capability checks; keep current UnitStream construction working.
3. **Features and execution arithmetic.** Implement physical features and generic
   within-unit summaries; use independent simple expectations.
4. **Primitive alternative.** Implement frozen-centroid fit/assignment and adapt
   its output through existing gap-aware common interfaces.
5. **Generic HMM/HSMM numerics.** Verify enumerated small cases before building
   staged/context wrappers. Implement duration mode truthfully.
6. **Staged episodes.** Add native projection, phase records and invocation boundary
   evidence. Test immediate reentry and censored endpoints.
7. **Generic finite parser, then both grammar wrappers.** Implement shared
   mechanism once and test operand-level independence/reference preservation.
8. **Context wrapper (optional at runtime).** Reuse numerical HMM code, keep lower units fixed
   and avoid filling unobserved physical gaps in event mode.
9. **Synthetic workflow.** One explicit workflow composes these slices, swaps
   adjacent episode choices, and runs the same applicable analyses. Include
   context off/on and fine/coarse resolution views.
10. **Final gates and review report.** Regenerate maps once, run required checks on
    the finished candidate, record limitations and update current state.

Do not spend the whole pass implementing a large first model while leaving every
other module an empty API. Conversely, do not sacrifice correctness to make all
seven say implemented. Deliver the complete bounded core and explicitly describe
any missing advanced feature.

Status should be truthful: `implemented` for documented bounded functionality,
`planned` for untouched placeholders; partial limitations remain explicit in the
module/README/review report. Do not assign implemented status based on importability.

## 9. Verification: tests that are independent of the implementation

Retain existing tests. Add small synthetic or enumerated cases with independently
known results. A model generating data and then scoring it with its own formulas
is insufficient as the only oracle.

### Required numerical cases

| Area | Independent expectation |
| --- | --- |
| Speed | Position advances 2 px per frame at 10 FPS: supported speed 20 px/s; stationary point gives 0 |
| Support | Nonzero frame origin, dropped frame and low-confidence channel: unavailable derivatives at the appropriate boundaries, no smoothing across them |
| Heading | Crossing +π to −π by a small physical turn yields small circular yaw, not a nearly 2π jump |
| Features | Supported centroid survives an invalid paw when only that paw-dependent feature is unavailable |
| Geometry | Rigid translation/rotation preserves declared body-relative ratios; distinguish path length and endpoint displacement |
| Primitive fit | Held-out outlier and changed group metadata do not alter training normalization or centroids |
| HMM | Enumerate all legal paths for two states and three observations; compare likelihood, marginals and MAP path |
| Structural zero | An impossible transition has exactly zero path/posterior mass |
| HSMM | Enumerate segmentations for at most five observations and duration support {1,2}; compare likelihood and duration traceback |
| Survival | PMF {.4,.6}: terminal one-step survival 1, terminal two-step survival .6 under the stated fresh-start boundary model |
| Invocation | Final phase immediately reenters the same episode's first phase: two occurrences, not one long ID run |
| Censoring | Stop during a final phase: final-stage evidence but no fabricated observed completion |
| Coordinate | Stretch native durations while preserving bout tokens: event-count likelihood unchanged, reported seconds changed |
| Grammar ambiguity | Identical terminal in choice branches weighted .25/.75: two derivations, summed sequence mass 1, best derivation .75 |
| Grammar limits | Repeat at min/max accepted; max+1 rejected; empty/nonprogressing/recursive unsupported rules rejected explicitly |
| Shared parser | Equivalent symbolic pattern parsed at primitive and episode levels uses the same engine; output levels and child scopes differ correctly |
| Shared children | Two assembly definitions can reference one episode type without inventing duplicated child models or unique-parent ownership |
| Independent records | Two sessions' sufficient statistics include no fictitious cross-session transition |
| Vocabulary | Permuting IDs and emission columns consistently leaves likelihood and physical intervals invariant |
| Execution | Hand-computed means, supported counts, physical exposure, stage durations and native peak latency for a few intervals |

For duration/stretch tests, normalized shape may stay similar while native pacing
changes; do not assert every profile invariant under resampling. Test valid masks
and support counts alongside values.

### Required interfaces and workflow cases

- Exact, actual seqNMF activation, new staged and grammar results enter the same
  applicable unit organisation/execution consumer without detector-class branches.
- Different event counts/catalogue sizes are accepted; no requirement of identical
  output matrices or cross-fit IDs. Include zero-event and partial-coverage cases.
- Plain UnitStream rejects completion/stages; validated rich results support the
  requested endpoint; malformed/foreign-record payloads fail before analysis.
- Missing/excluded calls preserve barriers. A B A with B excluded cannot become
  A→A; neither stage inference nor parsing crosses a missing validity block.
- Sorting/filtering preserves original scoped IDs and child references.
- Clock/FPS/schema mismatch has an explicit adapter or an informative rejection.
- Fine/coarse resolution changes do not mutate raw observations. Cached native
  features remain reusable; dependent token-based fits must not be silently reused.
- Context off leaves lower fit/results unchanged; context on consumes the same
  fixed units. No implicit joint fit or change to the lower catalogue.
- Original facade identities and independent existing numerical fixtures remain
  substantive. Any intentional old numerical change needs its own justification.

### Import-safety test migration is required

At the structural handoff, `tests/integration/test_behavior_layer_imports.py`
required all seven modules to be `planned` and to have no public callables. The
first implementation pass updated those assertions to real API names and
`implemented` status; the fresh-process import check passed in the current audit.

Retain truthful module/API assertions and documented bounded limitations.
Keep planned-only checks if a future module is genuinely still a placeholder.
**Do not delete the subprocess test or equate implemented status with verified
completion of every requirement.** Preserve
the fresh-process guarantees: no JAX/Torch/KPMS/study import, no implicit fit,
argument parsing, or artifact writes; imports work from a separate temporary CWD.
NumPy/SciPy numerical imports are acceptable CPU dependencies.

### Architectural tests remain binding

Current fences in `ops/repo/dependencies.py` mean:

- Observations cannot import primitive/higher model implementations or analysis.
- Higher layers cannot import `primitives.segmentation` internals.
- Assemblies/context/shared analysis cannot import episode detector implementations.
- Shared analyses cannot import assembly/context fitters.
- Foundation compatibility/adapters have narrow dependencies; the unit adapter's
  specific primitive-bouts exception is not permission to import other fitters.
- Old facades cannot be used by new canonical consumers to escape those rules.

Use common records and generic numerical helpers rather than weakening fences.
Add targeted tests if a genuinely justified boundary refinement is necessary.

## 10. Run checks efficiently and record their limits

Select an interpreter with existing required dependencies. On this lab machine:

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
AG_HANDOFF_PYTHON=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
"$AG_HANDOFF_PYTHON" -B -m fpcode.cli.doctor --json --require cpu
```

Use an equivalent environment elsewhere; do not upgrade the shared one incidentally.
Set threads before startup and inspect actual pools for substantive parallel work.
Do not combine many worker processes with uncontrolled 32-thread BLAS pools.

During each completed numerical slice, run affected independent tests. During the
interface slice, useful existing checks are:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$AG_HANDOFF_PYTHON" -B -m pytest -q -p no:cacheprovider \
  tests/unit/test_behavior_unit_contracts.py \
  tests/unit/test_behavior_layer_interfaces.py \
  tests/integration/test_behavior_layer_imports.py \
  tests/regression/test_behavior_layer_moves.py \
  tests/architecture/test_behavior_layers.py
```

New module-specific tests belong alongside those. Do not rerun every Make target
after every edit. Once the candidate is complete, run the required batch:

```bash
make folder-maps PYTHON="$AG_HANDOFF_PYTHON"
make test-cpu PYTHON="$AG_HANDOFF_PYTHON"
make architecture PYTHON="$AG_HANDOFF_PYTHON"
make lint PYTHON="$AG_HANDOFF_PYTHON"
make folder-maps-check PYTHON="$AG_HANDOFF_PYTHON"
make package-check PYTHON="$AG_HANDOFF_PYTHON"
```

Repeat a passed gate only for relevant changes, a failure or an unresolved concern.
Do not rewrite `registry/LINT_BASELINE.json` to hide new diagnostics. Do not update
golden expectations merely because an algorithm produced something different.
Installed-wheel smoke checks should exercise the new contracts, real APIs and
common consumer from outside the source checkout, with no backend initialization.

### Resources and complexity

Reuse invariant arrays, emissions, eligibility masks and frozen fits. Declare caps
for training samples, K, phase/state count, duration support, grammar node/repeat/
parse counts and batch memory. Avoid unbounded enumeration of ordinary recordings.

An N×K×F broadcast, all-state-pair trace over T, or HSMM duration table can exceed
RAM despite a small final result. Budget temporaries and workers, fail clearly
when a minimum operation exceeds budget, and stream/chunk where appropriate.
For long traces, avoid retaining a T×Q×Q tensor unless a bounded toy scope justifies it.

Start with a dependable CPU reference. GPU acceleration is optional only after
representative timing, available VRAM and precision/transfer costs justify it.
Do not add CUDA/JAX/Torch imports to solve small categorical matrix calculations.
Record dtype, seed, tolerance and representative timing; equal seeds do not imply
bitwise equality across backends or batching policies.

## 11. Data and empirical scope

Default handoff execution implements code and validates bounded synthetic inputs.
Preparing an explicit empirical workflow is useful; automatically fitting all
Karen sessions or repeatedly launching KPMS is not part of default verification.

If the user separately requests a Karen pilot, reuse existing verified products:

- `outputs/karen_behavior_atlas_20261011/{sessions.csv,frame_cache,measurements,chains}`
- `outputs/karen_clustering_levels_20261012`: explicit maps, anchors and fitted calls
- `outputs/karen_general_chain_organisation_20261012`: existing unit organisation
  metrics, support, sensitivity and comparison evidence

Use a new selected output root. No pickle reload is needed. Preserve exact input
fingerprints, original mouse split, annotations and exclusions. Read the study
guidance before making policy choices. A cached file path or historical claim is
not enough to infer its coordinate conventions.

Karen context for interpreting any later pilot: 99 recordings, 21 independent mice
(12 PFF/9 monomer); these are experimental groups, not a genetic contrast. Earlier
weeks versus a separate L-DOPA wave confound drug, age and wave. Closest 24-week
paired baseline has only nine mice. Primary named exact anchors cover median
4.06% of modelled time, with sparse loop opportunities; wider detector coverage
needs qualitative/held-out validation. Current seqNMF higher-order advantages are
limited/unstable. A new fitter must not be advertised as guaranteeing significance.

Promising non-significant effects remain worth reporting with original units,
uncertainty, mouse support and sensitivity. Do not select a method solely to
increase N-like occurrence counts or improve a group p value.

## 12. Deliverables for Codex review

Write a compact implementation report in the same selected generated output root,
for example `outputs/behavior_first_pass_<run-date>/implementation_report.md`.
Record exact output location in `docs/tasks/CURRENT.md` so the next reviewer finds it.
Do not create a parallel handoff/report hierarchy for each module.

The report must include:

1. **Changed-file inventory:** added/modified/moved files, owner and rationale;
   explicit confirmation of structure preservation and any justified exception.
2. **Module status table:** actual bounded algorithm, public entry points, supported
   inputs/outputs, duration/observation coordinate, fitting availability and limits.
3. **Contract changes:** sidecar schemas, invariants, capability derivation,
   cross-model identity and old-constructor/import compatibility.
4. **Numerical evidence:** independent toy expectations, enumeration comparisons,
   tolerances, impossible-case handling and posterior normalization where claimed.
5. **Integration evidence:** exact/seqNMF/staged/grammar swaps, resolution changes,
   context off/on, shape differences and unsupported capability outcomes.
6. **Nontrivialities addressed:** clocks, support, durations, censoring, reentry,
   ambiguity, score interpretation, held-out transforms, dependency and import safety.
7. **Executed checks:** full commands, selected interpreter, baseline/dirty tree,
   exit outcomes and log locations. A synthetic pass is not cohort validation.
8. **Resource evidence:** configured caps/threads, representative timings and any
   large temporaries or unimplemented long-recording scaling path.
9. **Remaining limitations:** exact missing features, identifiable versus assumed
   boundaries, unimplemented fitting/modes, and review questions. No vague “all done”.
10. **Reproduction:** one runnable small synthetic workflow with fixed inputs and
    seeds; output tables/units/reference maps sufficient to inspect the result.

Update `behavior/README.md`, truthful module status/import tests, folder maps and
`docs/tasks/CURRENT.md`. Keep unrelated dirty work and historical evidence intact.

## 13. Final acceptance and reviewer red flags

An acceptable first pass has real bounded operations in the seven owners, common
results usable by applicable consumers, explicit richer capabilities, independent
numerical evidence, clean dependency directions and a reproducible synthetic
workflow. Any unavailable advanced mode is rejected or explicitly omitted.

Codex should specifically look for:

- A rewritten numerical policy hidden inside a relocation or wrapper.
- Silent coordinate/FPS conversions and gaps stitched by filtering.
- Duration modes labelled seconds/HSMM without the matching likelihood.
- Forbidden transitions made possible by epsilons.
- Final-phase occupancy automatically labelled completed, or same-ID restarts merged.
- Grammar best-path scores called confidence, lost derivations or unreported pruning.
- Capability flags without validated payloads; fabricated stages/probabilities.
- Context filling sparse unobserved physical gaps or silently refitting lower units.
- New directories/wrappers/copies created to bypass existing owners/fences.
- Deleted/softened import, identity, architecture or golden tests to get a pass.
- Held-out/outcome leakage, pseudo-animal N or evidence claims based only on coverage.
- Shared-environment upgrades or automatic empirical workloads during setup/help.

The deliverable is a disciplined first implementation that can be reviewed and
improved. Biological validation, richer grammars, full temporal hierarchy recovery
and empirical superiority remain separate questions.
