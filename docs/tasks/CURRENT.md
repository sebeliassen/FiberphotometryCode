# Current task state

## Latest checkpoint: Karen confidence/handling audit (2026-10-12)

Scanned all 99 Karen recordings from verified native-frame cache and tandem21 V2
DLC confidence tables; no pickle reload, fitting or original dataset/exclusion
mutation. User explicitly confirmed H1/H3/H4/H5 handling, H2/H8 free behaviour,
and H6/H7 empty arena. Labels are persisted in the audit review CSV. At H1 native
frame 10489 dorsal support is absent but the existing cache still has a KPMS label.
Current default presence replay rejects 633,304/17,409,490 modelled pose-supported
frames; this is not handling prevalence, nor proof of historical mask settings.

Karen enrichment auto masks only photometry signal columns, not behaviour. The
maintained exclusions CSV is header-only and has no auto_mask.npz; enrichment's
default overrides path differs from the maintained session_policy path. Historical
execution paths/settings remain unproven. Separate-camera confidence scoring avoids
treating ordinary paw visibility as handling. A review-only dorsal17 threshold 0.8,
minimum 15 seconds, bridge <=0.5 seconds rule flags all six reviewed handling/empty
points and neither reviewed free point. Its 257 spans overlap 71,528 modelled frames
(19.87 minutes, 0.411%). Complete boundaries are unreviewed; eight selected stills
are not an unbiased validation sample. Proposed keep indices cover existing
modelled pose-supported frames only and are not installed as production masks.

Reusable streak detection remains in tracking/quality.py; verified native-frame
reads in viz/event_clips.py; cohort scanner/rules remain in Karen analysis. Report,
profiles, candidate CSVs, confirmed labels, proposed masks and manifests are under
[the audit output](../../outputs/karen_handling_audit_20261012/report.md).
Execution took 14.02 seconds using two budgeted CPU processes and four numerical
threads; 1536 MiB is a scheduling budget, not an OS-enforced cap.

Checks: 15 focused tests; final CPU suite 659 passed (six historical warnings),
architecture 198 modules with no violations/cycles, lint no new debt, 146 synchronized
folder maps, installed-wheel imports/resources/CLI/doctor passed. One formatting-only
semicolon split followed the suite; lint was rerun successfully. Logs and final hashes
are in the audit validation directory. Six derived maps regenerated; dirty unrelated
work preserved. Next: boundary-clip review, then an explicit analysis-view mask for
labels, pose, exposure and continuity; never join chains across exclusions. Frozen
catalogue contamination needs a separate discovery audit. No production exclusions
have been applied.

## Latest checkpoint: Karen held-out layer pilot executed (2026-10-12)

The user authorised acceptance tests, implementation and a bounded Karen pilot,
then clarified that arena calibration will not be available and explicitly allowed
coordinate/screen proxies. They also specified shared central TopCam perspective
versus local BotCams. Reusable additions remain in existing shared owners:
native-index DLC alignment with explicit sentinel policy; animal split validation;
bounded frozen exact/seqNMF/configured-method comparisons; robust position envelopes,
independent landmark/corner distances and supported landmark averages. Karen policy
and reporting live in `analysis/run_layer_pilot.py` / `report_layer_pilot.py`.
No new source folder, parallel framework or cohort-dependent library import was added.

Acceptance tests were written first and initially failed on the missing entry points.
They now cover alignment, -1 confidence masking without source mutation, held-out
animal isolation, no template fitting, vocabulary mismatches, resource guards,
resolution-independent features/common consumers, nose-specific support, corner
geometry, ventral-paw averages and training-only geometry reference selection.

Executed 12 one-minute windows from four held-out mice (15/18 Monomer, 12/16 PFF),
at 3w, 16w and L-DOPA and L1–L3. Reused the verified cache and frozen discovery
templates; exact native label projection was rechecked against named results/VFI.
Latest cached descriptions and frozen resolution maps match their declared sources.
The smoke revealed missing-confidence sentinels and experimenter handling within
early modelled frames. The final windows start 120 seconds after the first modelled
frame; 36 start/middle/end video samples were inspected and showed unhandled mice.
This is spot QC, not exhaustive handling annotation. Original discovery templates
may include early handling and require a later exclusion/rediscovery audit.

Spatial features retain midpoint, nose and ventral-paw-centre references (minimum
two paws, not a ground-truth reconstruction), all image corners and chamber-specific
TopCam camera-near/far corners. Screen rectangles and robust pooled training-only
nose envelopes are exported separately by chamber/epoch/video size. Both are
proxies, not physical walls or parallax corrections. Native tracking support and
chamber identity remain available; tandem21 does not provide a second nose estimate.

Saved **3,451 raw unit occurrences**, explicit exclusive selections, organisation
at four gap thresholds, execution/spatial features, phases/profiles and composition
derivations. Native support and selection/schema checks passed on all saved units.
The 16 gallery clips plus one additional assembly clip have verified FPS/frame
counts; the assembly shows two reusable parts at the same native span across all
three resolutions. Exact selected calls are 19 at L1 and 47 at L3; seqNMF selected
coverage is 61% and 77%. These describe different frozen catalogues; neither proves
method superiority nor PFF–monomer differences. Configured HSMM parameters need
calibration and show resolution sensitivity, including no look/scrunch calls at L3.

[Pilot report](../../outputs/karen_layer_pilot_20261012/pilot/report.md), CSVs,
figures, gallery, original handling smoke, QC images and manifests are retained
under `outputs/karen_layer_pilot_20261012`. Runtime **23.55 seconds**, two budgeted
CPU processes, four OMP/OpenBLAS/MKL/NumExpr threads, peak worker RSS **322.49 MiB**;
1536 MiB is a scheduling budget, not an OS-enforced hard cap. No pickle reload,
KPMS/seqNMF-template fit, HSMM parameter learning, environment update, commit/push
or notification occurred. Baseline remains dirty `3ec30fb` on `codex/modular-refactor`.

Final repository gates passed once on the finished candidate: **657 CPU tests**
in 9.00 seconds (six historical deprecation warnings); architecture **198 modules,
zero violations/cycles**; lint **3,079 accepted diagnostics, no new debt**;
**146 synchronized folder maps**; installed-wheel resources/type marker, canonical
imports, six CLI help commands and read-only doctor passed outside the checkout.
Used `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python` (Python 3.10.19), `-B`,
`PYTHONDONTWRITEBYTECODE=1` and four numerical threads. Exact logs and final source
fingerprints are under the selected output root's `validation` directory. The final
budget/crosswalk guards and workflow docstring updates after execution do not alter
the pilot's declared numerical settings; the execution manifest retains its own
source hashes. Seven derived maps were regenerated; unrelated dirty work preserved.
Next scientific step: expert example review, training-only model calibration and handling exclusions,
then expansion of useful methods to all 21 mice/99 sessions with group effect sizes
and uncertainty. This four-mouse pilot is not an inferential cohort comparison.

## Latest addition: arena-distance observation features (2026-10-12)

The user requested distances from the nearest corner, edge and centre while
retaining KPMS as the syllable creator. Added optional explicit `arena_bounds`
to the existing shared trajectory extractor, without adding a new owner or
changing existing default channels. Geometry is a calibrated axis-aligned
rectangle in pose coordinates/units. Distances use the existing body midpoint;
outside-arena and unsupported samples are masked. No arena calibration is inferred
from mouse positions and no Karen data were loaded.

The same `FeatureAssociation` can explicitly accompany primitive, resolution,
episode or assembly streams in generic execution/profile analyses. Detector
inputs and labels remain separate; feature attachment is not automatic detector
conditioning. Independent analytical tests cover corners/centre/interior points,
missing frames, invalid/outside tracking, units/provenance and downstream native
interval summaries. Focused tests: **14 passed**. One final gate batch passed:
`make test-cpu` **647 passed** in 9.16 seconds (six historical deprecation warnings),
`make architecture` **198 modules / zero violations or cycles**, `make lint`
**3,079 accepted diagnostics / no new debt**, `make folder-maps-check`
**146 directories synchronized**, and `make package-check` isolated installed-wheel,
six CLI help commands and read-only doctor passed. Used the lab Python 3.10.19
interpreter with `-B`, `PYTHONDONTWRITEBYTECODE=1` and four OMP/OpenBLAS/MKL/NumExpr
threads. Two derived maps were regenerated. No empirical execution or environment
upgrade occurred. The previous verification report remains evidence of its earlier
source fingerprints; this addition changes that source and is a new checkpoint.

## Latest checkpoint: bounded behaviour first pass independently verified (2026-10-12)

The user authorised finishing the remaining verification and clarified that
composition should be an adjacent component to inter/intra-chain measurements.
All seven modules remain in their existing owners. Composition consumes the same
frozen episode stream as organisation measurements; its parsed spans can optionally
become higher units for generic inter/intra analysis. It is not a mandatory stage.
Caller-defined finite grammars support shared children, weighted choices, optional
parts and bounded loops. Automatic grammar induction, recursion and a
context-conditioned routine controller are future method choices.

Completed the interrupted checklist with independent HSMM path enumeration,
context gap/recording/vocabulary checks, source/phase/posterior contract checks,
recursively owned immutable diagnostics, VQ resource/clock tests, shared-parent
composition, profile budget/coordinate guards and context on/off lower-evidence
identity assertions. Physical-time segmentation requires supplied native frames
or explicitly declared physical/native clock origins and FPS. The implicit
FPS-only rounded conversion in the older checkpoint below is superseded; old
diagnostic probes and products were preserved, not rerun as current evidence.

Executed **120 focused tests** and one final broad gate batch: **642 CPU tests
passed** (six expected historical deprecation warnings), architecture **198 modules,
zero violations/cycles**, lint **3,079 accepted diagnostics with no new debt**,
**146 synchronized folder maps**, and isolated installed-wheel/CLI/doctor checks
passed. NumPy and SciPy OpenBLAS pools were verified at four threads. Python 3.10.19
from `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`, `-B`,
`PYTHONDONTWRITEBYTECODE=1`; OMP/OpenBLAS/MKL/NumExpr configured at four threads.
Baseline remains dirty `3ec30fb46607f1d35b231d7773a47168cf4546a0` on
`codex/modular-refactor`; unrelated work was preserved.

Evidence, exact gate logs, source fingerprints and synthetic CSVs are in
[the verification report](../../outputs/behavior_verified_first_pass_20261012/verification_report.md).
The synthetic workflow produces nonempty outputs for every configured method;
its assembly has two source children. This completes the bounded implementation
and verification pass, not empirical validation on Karen. No empirical fit,
pickle reload, shared environment update, commit/push or notification occurred.
Next scientific step: a separately scoped Karen pilot with frozen lower evidence,
declared clocks/resolutions/splits and held-out qualitative assessment.

The remaining sections record historical checkpoints; their status and counts
do not override this latest verified checkpoint.

## Historical Gemini checkpoint: reported first-pass validation (2026-10-12)

All seven behavioural abstraction modules in `src/fpcode/behavior/` are implemented in place,
remediated against all 20 audit probes, and verified through both focused and broad repository gates.

Continuation checkpoint resolution:
- Fixed test failure in `tests/unit/test_behavior_finite_grammar.py`: Added `import math` and verified math tolerance.
- Fixed assertion in `tests/unit/test_behavior_layer_interfaces.py`: Aligned boundary occurrence status with honest completion evidence (`censored`, `left_censored == True`).
- Fixed seconds-to-frame mapping in `src/fpcode/behavior/primitives/segmentation.py`: `CentroidPrimitiveSegmenter.segment_stream` derives native frames when `native_frames=None` for physical time units with positive FPS, preserving gaps.
- Verified 20/20 audit probes pass (`meets_expectation: true` for 20/20) in `outputs/behavior_first_pass_review_20261012/audit_results.json`.
- Verified all 110 focused behavior tests pass (100% pass rate in 1.11s).
- Verified full repository validation batch:
  - `make folder-maps-check`: 146 directories synchronized; 0 obsolete maps.
  - `make architecture`: 0 violations, 0 cycles across all 142 canonical modules.
  - `make lint`: Strict lint passed; 0 new diagnostics; baseline 3,079 unchanged.
  - `make test-cpu`: 632 passed in 8.85s (6 expected deprecation warnings, backend imports forbidden).
  - `make package-check`: Wheel build, isolated venv install, 6 CLI help commands, and read-only doctor passed outside checkout.

Interpreter `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`, with `-B`,
`PYTHONDONTWRITEBYTECODE=1` and thread limits (`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4`). Baseline remains dirty `3ec30fb` on `codex/modular-refactor`. No unrelated work discarded.

## Completed task: completion audit remediation (2026-10-12)

Antigravity completed full remediation of the module-by-module completion audit
documented in [ANTIGRAVITY_BEHAVIOUR_IMPLEMENTATION_HANDOFF.md](../agent-workflows/ANTIGRAVITY_BEHAVIOUR_IMPLEMENTATION_HANDOFF.md).
All 20 diagnostic probes in `outputs/behavior_first_pass_review_20261012/audit_probes.py`
now pass (`meets_expectation: true` for 20/20):

1. **HSMM Marginal Likelihood:** Added exact forward log-sum-exp recursion (`forward_hsmm`) in `fpcode.statistics.sequence_inference` computing genuine marginal likelihood alongside Viterbi path scoring.
2. **Public Scoring APIs:** Added public `.score(stream)` and `.predict(stream)` on both `StagedEpisodeModel` and `EventContextHMM`.
3. **Native Coordinate Time Indexing:** Replaced row-coordinate slicing in `fpcode.behavior.analysis.execution` and `fpcode.behavior.observations.representations` with coordinate-aware `TimeAxis` filtering.
4. **Foreign Association Validation:** Enforced strict session and clock validation in `measure_occurrence_execution`, rejecting mismatched `FeatureAssociation` payloads.
5. **Gap-Aware Trajectory Steps:** Path length computation in `compute_interval_trajectory_metrics` now enforces contiguous native frames (`dt == 1`), eliminating unobserved inter-gap jump artifacts.
6. **Validity Block Partitioning:** Partitioned occurrences strictly by declared `valid_intervals` before inference or grammar parsing in `episodes/staged.py`, `episodes/grammar.py`, and `assemblies/grammar.py`, preventing cross-barrier matches.
7. **Grammar Pruning Disclosure:** Bounded chart parsing in `fpcode.statistics.finite_composition` now sorts derivations by score descending and explicitly marks retained matches with `truncated=True` when candidate derivations exceed `max_derivations`.
8. **Syntactic Acceptance vs. Physical Exit:** Grammar episode and assembly detection now distinguish syntactic rule acceptance from observed exit, marking boundary-reaching occurrences as censored (`observed_exit=False`).
9. **Unique Scoped Derivation IDs:** Derivations across independent blocks now generate unique, scoped IDs incorporating occurrence identifiers.
10. **Rich-Result Phase Containment:** `BehaviorResult` validates that phase intervals are wholly contained within parent occurrences and rejects foreign session references.
11. **Physical Unit Schema Validation:** `CentroidPrimitiveSegmenter` validates input signal units against fitted feature units during prediction.
12. **Seconds-to-Native Frame Mapping:** `CentroidPrimitiveSegmenter.segment_stream` maps continuous seconds timestamps to native frames and preserves unobserved gaps.
13. **Configurable Memory Budget:** Added `memory_budget_bytes` parameter and allocation guard to `CentroidPrimitiveSegmenter.fit`.
14. **Impossible Sequence Rejection:** Staged decoding in both HSMM and HMM modes raises informative `ValueError` when input sequences have zero probability mass (`-inf`).
15. **Honest Censoring Denominators:** Aggregated execution metrics expose separate `censored_count` and `evaluable_count`, returning `NaN` (undefined) rather than 0.0 when no evaluable exits exist.
16. **Non-Empty Assembly Composition:** Updated `run_synthetic_behavior_comparison` with a valid motif assembly rule, achieving non-empty representation across all 7 layers.

Validation evidence:
- Audit probes: **20/20 passed** (`outputs/behavior_first_pass_review_20261012/audit_results.json`).
- `make test-cpu`: **626 passed** (0 failed).
- `make architecture`: **0 violations, 0 cycles** across all 193 modules.
- `make lint`: **Passed** (0 new diagnostics, baseline 3,079 unchanged).
- `make folder-maps-check`: **Passed** (146 directories synchronized).
- `make package-check`: **Passed** (clean wheel built and tested outside checkout).

## Latest audit: seven real implementations, first-pass scope incomplete (2026-10-12)

The user asked whether everything was implemented, allowing unfinished polish,
and authorised updating the existing handoff with missing scope. The source audit
confirmed real code in all seven modules and the five permitted helper owners;
no module is an empty placeholder. The first-pass completion claim below is
superseded by this audit. Preserve the structure and existing useful implementations.

The updated [handoff completion checklist](../agent-workflows/ANTIGRAVITY_BEHAVIOUR_IMPLEMENTATION_HANDOFF.md)
records nine ordered work items. Core gaps: HSMM marginal likelihood is actually a
MAP score; native-frame feature intervals are sliced as array rows; touching
validity boundaries are crossed before parent validation; impossible staged HMM
paths can fabricate episodes; syntactic grammar acceptance is labelled observed
completion; censored calls enter a failure-like completion denominator.

Remaining implementation requirements also include parser truncation/global work
bounds and stable derivation-parent identity, phase/source/model/clock validation,
primitive physical-unit/native-time schema and RAM guards, declared onset/phase
profiles and a nonempty assembly integration case. Configured context HMM inference
and the frozen-centroid baseline are real. Advanced EM, unrestricted grammar
induction, 3D features, cosmetic polish and empirical superiority are deferred;
these are not reasons to discard the existing first pass.

Executed one focused batch: **104 tests passed in 1.05s**, covering the seven new
module/contract/workflow test files, interfaces, fresh-process import safety,
relocation regression and layer architecture. Read-only doctor `--require cpu`
passed with no backend initialized. Twenty targeted independent diagnostic probes
reproduced gaps/scope markers; these deliberately selected cases are not a failure
frequency estimate or a failing full test suite. Reproducer, JSON observations and
validation/source fingerprints are in `outputs/behavior_first_pass_review_20261012`.

Interpreter: `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`, Python 3.10.19,
with `PYTHONDONTWRITEBYTECODE=1`, `-B`, and OMP/OpenBLAS/MKL/NumExpr configured at four
threads. Baseline remains dirty `3ec30fb` on `codex/modular-refactor`; unrelated work
and previous products are preserved. Only documentation and synthetic review
evidence were written. No source algorithm fix, empirical fit, pickle reload,
environment change, commit/push or notification occurred. The claimed 626-test
full gates below were not rerun for this audit. Next step is completing the
handoff's core checklist in place, then independent review of the finished pass.

## Reported implementation checkpoint: behavioural abstraction first pass (2026-10-12)

Antigravity reported completion of the first implementation pass of all seven planned behavioural
abstraction modules according to the handoff specification in
[ANTIGRAVITY_BEHAVIOUR_IMPLEMENTATION_HANDOFF.md](../agent-workflows/ANTIGRAVITY_BEHAVIOUR_IMPLEMENTATION_HANDOFF.md).
The independent audit above finds required scope still incomplete; the following
records the reported checkpoint rather than current completion status.
Structure preservation was strictly observed: all seven modules were implemented
in place in their existing namespaces, without creating parallel frameworks or
unauthorized directory trees.

The seven implemented modules are:
1. `fpcode.behavior.observations.representations`: Trajectory feature extraction (`extract_trajectory_features`) and interval metrics (`compute_interval_trajectory_metrics`).
2. `fpcode.behavior.primitives.segmentation`: `CentroidPrimitiveSegmenter` with training-only normalization, distance-to-centroid assignment, gap preservation, and RLE emission.
3. `fpcode.behavior.episodes.staged`: `StagedEpisodeModel` with explicit-duration HSMM and staged HMM inference, exact structural zeros (`-inf`), boundary censoring, and same-routine reentry separation.
4. `fpcode.behavior.episodes.grammar`: `GrammarEpisodeDetector` parsing primitive sequences into multi-primitive episodes with bounded ambiguity tracking.
5. `fpcode.behavior.assemblies.grammar`: `GrammarAssemblyComposer` composing higher-order behavioral assemblies from episode occurrences with parse derivations.
6. `fpcode.behavior.context.models`: `EventContextHMM` modeling contextual latent dynamics over frozen lower-level evidence streams.
7. `fpcode.behavior.analysis.execution`: Objective quantitative kinematic and timing analysis (`measure_occurrence_execution`, `aggregate_execution_metrics`, `measure_phase_pacing`).

Permitted bounded shared helpers were added in their proper architectural domains:
- `fpcode.contracts.behavior_results`: Envelope and payload types (`BehaviorResult`, `FeatureAssociation`, `PhaseRecord`, `CompletionEvidence`, `CompositionDerivation`, `StatePosteriorRecord`).
- `fpcode.contracts.composition`: Rule definitions (`SequenceRule`, `ChoiceRule`, `RepeatRule`, `GrammarSpecification`, `ParseMatch`).
- `fpcode.statistics.sequence_inference`: Numerically stable log-space inference (`logsumexp`, `forward_backward_hmm`, `viterbi_hmm`, `viterbi_hsmm`, `compute_survival_log_pmfs`).
- `fpcode.statistics.finite_composition`: Bounded finite-state grammar parser (`match_rule_on_sequence`).
- `fpcode.workflows.behavior_layers`: Synthetic multi-layer demonstration and interchangeability workflow (`run_synthetic_behavior_comparison`).

Comprehensive validation was reported using `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`:
- `make test-cpu`: **626 passed** (up from 598 baseline, 0 failed).
- `make architecture`: **0 violations, 0 cycles** across all 193 modules.
- `make lint`: **Passed** (0 new diagnostics in `src/fpcode`, baseline 3,079 untouched).
- `make folder-maps-check`: **Passed** (146 directories synchronized).
- `make package-check`: **Passed** (clean wheel built and tested outside checkout).

Detailed review report and reproduction artifacts are saved at:
- Implementation Report: `outputs/behavior_first_pass_20261012/implementation_report.md`
- Synthetic Run Manifest: `outputs/behavior_first_pass_20261012/run_manifest.json`
- Execution Metrics Table: `outputs/behavior_first_pass_20261012/execution_metrics_summary.csv`
- Organisation Metrics Table: `outputs/behavior_first_pass_20261012/organisation_metrics_summary.csv`

## Latest task: Antigravity implementation handoff prepared (2026-10-12)

The user requested an extensive handoff for Antigravity to attempt a first
implementation of the planned behavioural modules, with verification guidance,
nontrivialities and explicit protection against repository clutter. The complete
[implementation handoff](../agent-workflows/ANTIGRAVITY_BEHAVIOUR_IMPLEMENTATION_HANDOFF.md)
contains a copyable starting instruction, current owners/interfaces, bounded
algorithms for all seven modules, an ordered implementation sequence, independent
numerical/interface tests, resource caps and required review deliverables.
Configured explicit-duration HSMM inference is required for the staged first
pass; parameter learning and a staged HMM comparison are optional. The context
module must be implemented, while its use in a composed workflow stays optional.

Structure preservation is an acceptance requirement: implement the existing
planned modules in place, reuse shared contracts/numerical mechanisms, justify
every added file, preserve dependency fences and historical identity reexports,
and keep temporary/generated products out of source. The handoff permits a small
explicit list of shared helper owners where needed; it does not call for a new
framework or competing directory tree.

Scientific details include event-count versus physical duration, exact structural
zeros, terminal survival, observed completion versus censoring, same-routine
reentry, scoped child references, grammar ambiguity/pruning, feature-specific
support, training-only transforms and frozen lower evidence. Import-safety tests
must retain their guarantees while replacing assertions specific to placeholders.
Focused checks precede one final broad validation batch; repeated operations use
explicit memory/thread budgets and representative timing.

This increment is documentation only. No planned module was implemented and no
empirical fit, pickle reload, environment change or numerical suite was run.
The 598-test structural checkpoint below remains the most recent implementation
evidence. The handoff is linked from the documentation index; local document links
passed (68 targets), the named checkpoint files exist, and all seven planned-module
paths/statuses remain unchanged. Next step is Antigravity's
bounded implementation pass, followed by review of its exact file inventory,
independent numerical evidence and remaining limitations.

## Completed structural task: behavioural abstraction framework (2026-10-12)

The user accepted the abstraction plan and authorised a first structural
implementation: appropriate folders/modules, relocation of existing functionality,
documented future placeholders and independence/interface tests. The accepted
[behavioural abstraction plan](../architecture/BEHAVIOURAL_ABSTRACTION_PLAN.md)
defines observations, primitives, episodes and optional assemblies, with a
parallel context track. Resolution (including L1–L4), detector choice and
within/between-unit measurement are separate axes. Independence means explicit
shared results, frozen lower evidence, reusable analyses and traceable dependency
changes; inferred units and scientific interpretations remain partly coupled.

Structure is implemented under `src/fpcode/behavior/{observations,primitives,
episodes,assemblies,context,analysis}`; inventory and integration guidance are in
[behavior/README.md](../../src/fpcode/behavior/README.md), with local agent guidance.
Eleven numerical modules were moved whole, all original function/class AST hashes
unchanged. Previous public/private callable paths remain explicit identity
reexports; 12 canonical consumer/aggregate modules now use their owners.
The move/source/API manifest is
[BEHAVIOR_LAYER_MOVES_20261012.json](../../registry/BEHAVIOR_LAYER_MOVES_20261012.json).
Old timeline/tiling adapters, KPMS backend ownership and unaudited study prototypes
are retained. Whole-module phenotyping keeps frame_kinematics in analysis pending
a separately validated decomposition; no numerical policy was rewritten.

New immutable contracts: `FrameSpan`, `UnitType`, `UnitOccurrence`, `UnitStream`
in `contracts/behavior_units.py`. Common results preserve recording/model/clock
identity, physical support, variable event counts and explicitly interpreted
scores. `behavior/unit_adapters.py` adapts frame labels, native occurrence tables,
and bout-window outputs while preserving RLE/gaps and catalogue crosswalks.
`behavior/compatibility.py` reports downstream requirements and unsupported
information. `analysis/unit_views.py` bridges common streams to existing
organisation measurements, with explicit overlap policy and exclusion barriers.
Stages/completion/posterior matrices remain genuinely unavailable in the common
interval schema; model-native rich results must be retained alongside it.

Seven planned Python modules document representation/primitive/staged/grammar/
assembly/context/execution responsibilities. They have docstrings and planned
status markers, with no empty fitting API claiming to work. Behavioural dependency
fences supplement the capability audit, reject detector imports into shared
analysis and old-facade bypasses, and separate primitive segmentation backends
from reusable bout/view helpers. Real exact-chain and seqNMF activation outputs,
different shapes/counts/resolutions, mismatched clocks, gaps, overlaps, excluded
calls, unsupported completion and placeholder import safety have focused tests.

Final checks passed with `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`
and `PYTHONDONTWRITEBYTECODE=1`: 598 CPU tests (9.49s), architecture zero violations/
cycles, strict/repository lint (3,079 accepted historical diagnostics/no new debt),
146 navigation directories synchronized, and installed-wheel checks outside the
checkout including the new contract/consumer and old/new seqNMF class identity.
Read-only doctor `--require cpu` passed with no backend initialized. Exact logs
and source/checkpoint evidence are in `outputs/behavior_layer_structure_20261012`.
The full suite ran once against the completed candidate; focused checks preceded it.

Baseline remains dirty `3ec30fb` on `codex/modular-refactor`, with unrelated work
preserved. No empirical fit, pickle reload, source-data rewrite or environment
upgrade was performed; existing Karen products below remain intact. This
authorised structure/interface increment is complete. Next development can audit
one staged episode alternative and add genuine phase/completion capabilities,
then evaluate it against frozen lower evidence before adding assemblies/context.

## Completed extension: general chain organisation (2026-10-12)

The user authorised execution of steps 1–6 of the
[general chain plan](../../studies/karen_pff_vs_monomer/analysis/GENERAL_CHAIN_ORGANISATION_PLAN.md).
Implementation, empirical comparisons and companion report are complete in
`outputs/karen_general_chain_organisation_20261012`. See
[workflow/reproduction](../../studies/karen_pff_vs_monomer/analysis/GENERAL_CHAIN_ORGANISATION.md).
Earlier atlases, verified NPZ/CSV inputs and unrelated dirty work are preserved;
no pickle reload, model refit, environment upgrade, commit/push or notification.
Baseline remains dirty `3ec30fb` on `codex/modular-refactor`.

Reusable owners: `behavior/sequences/organisation.py` supplies native half-open
timelines, deterministic inclusive-first overlap decisions, exclusion barriers,
censoring, self/run/initiation/exit/return/coverage measures and sufficient counts.
`statistics/chain_nulls.py` receives explicit fixed slot eligibility and supplies
exact frequency expectations, bounded categorical shuffles and first-order nulls.
`workflows/chain_organisation.py` supplies session measurement, all-library
leave-chain-out and count pooling. Statistical parallel feature batching lives in
`statistics/behavior_comparisons.py`, with joint correction after batch merging.
Karen frozen library/scientific policy, runners and report live in the study.

Completed: 99 sessions/21 mice, 14 frozen libraries, two scopes, five gaps,
13,860 session-policy rows, 3,622 features, 50,708 group contrasts (38,400 tested),
weekly/untreated/all-wave/paired/wave-adjusted comparisons and sensitivity tables.
Frequency expectations cover all gaps; 1,999-replicate intervals cover g1 in
every library/scope. First-order and quarter controls cover L3 anchors g1.
All-library leave-chain-out preserves removed calls as barriers. Six PNG/PDF
figure pairs, portable HTML/Markdown, feature dictionary and CSV inventory exist.

Primary named L3 anchors cover median 4.06% of modelled time, with only 203 nearby
pairs and 21 two-step windows across recordings. Untreated ABA excess has only
three monomer/two PFF mice with opportunities and is not tested. Primary mean-run
difference is -.000196 calls/run (g -.016); self-excess is +.03214, very uncertain.
There are no primary family/scope or whole-screen global q<.05 group hits.
Fourteen of the 19 named primary anchors share start/end syllables: RLE/template
structure can prevent strict self-adjacency, so a g0 zero is not evidence of no
repetition. Most primary run ends are censored, not observed switches.

Exploratory untreated candidate: named L3 seqNMF self-excess +.01050 probability
(g .790, bootstrap interval -.00023 to +.02171, broad q=1), positive at 8/16 weeks,
negative at 3 weeks; direction survives all eight leave-factor-out and all
leave-mouse-out cases. Named L4 seqNMF mean runs -.01201 calls/run (g -.520),
direction consistent across three early weeks and all seven factor removals.
Named L4 anchors also have slightly shorter runs. Inclusive L3 anchor starts
-.174/min closely track fewer calls (-.183/min); this concerns usage more than
longer persistence. These are correlated selected pilot patterns, not confirmed
discoveries. Existing limited seqNMF order-null evidence and later-wave confounding
remain material. Review labels, coverage and stable routines before new-cohort
endpoint/design decisions.

Final evidence: 483 CPU tests passed, strict/repository lint passed with 3,079
accepted historical diagnostics and no new debt, 165 canonical modules/zero
dependency violations/cycles, 140 navigation directories synchronized, final
installed-wheel check passed outside checkout. The initial boundary check caught
a statistics→behaviour dependency; fixed by passing eligibility into statistics.
Current-source replay of primary and broad seqNMF first-session numerical tables
matched the saved outputs after that ownership-only fix. Saved-output audit
verified all 1,386 complete interval partitions, pooled counts, null expectations,
zero invalid calls, all 141 unchanged input fingerprints and portable media/links.

Final execution used eight workers with both OpenBLAS pools verified at four
threads. Full measurement/comparison elapsed 106.83 seconds; peak worker RSS
338.9 MB including inputs/interpreter, conservative surrogate batch maximum
13.93 MB below the configured 256 MB allocation cap. No accelerator was needed
for this count workload. Root `AGENTS.md` retains milestone-based checks and
resource-aware repeated operation guidance. Details/logs are in output
`validation.json` and provenance. No jobs remain running; authorised steps 1–6
are complete. Next scientific step is review of candidate routines and detector
coverage, not a pending implementation stage.

## Current extension: clustering resolutions (2026-10-12)

User requested L1/L2/L3/L4 comparisons and authorised qualitative judgement,
especially for L2/L3. New output: `outputs/karen_clustering_levels_20261012`;
original completed atlas below is retained. Existing Karen hand-curated maps
are 100/93/22/14 states. Policy is copied explicitly into study-owned
`karen_hierarchy.py`; generic remapping preserves -1/missing-frame gaps. Fresh
RLE follows remapping. Unresolved-preserving L3/L4 variants (69/61 states) keep
44 unknown and five dump states individually distinct, as a sensitivity check.

Reusable additions: `behavior/resolutions.py`, `workflows/resolution_atlas.py`
and optional `label_map` in the measurement workflow. They provide member
composition and fixed-reference-mixture raw-state execution sensitivity. The
weights are derived from untreated discovery mice, without selecting on group
differences. Those summaries are not merged-bout execution or causal mediation.
The CLI is `studies/karen_pff_vs_monomer/analysis/run_clustering_atlas.py`.

Extension completed: primary L1–L4 and two unresolved-preserving sensitivities,
99 sessions per level, 201,047 tested contrasts. All planned measurement,
chain, anchor and statistical runs finished. Report: new output `report.html`
and `report.md`, 12 PNG/PDF figure pairs, combined comparison CSVs and a
92-row interpretable exploratory shortlist. No primary group-permutation rows
survive across-level family/scope correction. Steering remains a promising
non-significant pattern: L3 left-turn time -0.93 percentage points (g -1.06),
L4 total-turn time -1.65 points (g -1.12). Four local L3 corrected progression
rows refer to one unresolved-member routine and correlated endpoint aliases;
they are not independent discoveries. L3 seqNMF has essentially no mean gain
over its first-order null. Keep qualitative uncertainty visible.

Final checks: 475 CPU tests passed, strict new-source and repository lint
passed without inventory changes, 162 canonical modules/zero violations/cycles,
140 navigation directories synchronized, package check passed outside checkout.
All 400 primary mapping assignments matched the isolated existing policy
function exactly. Both OpenBLAS pools were verified at four threads for the
remaining parallel comparisons. Embedded report PNGs and local links passed;
evidence is in the output `validation.json`. No analysis jobs remain running.
Next scientific step is review of shortlisted patterns and raw member labels;
the requested implementation/report work is complete.
No native-time/micro or raw fused-pose refitting is planned for this extension;
it focuses on label aggregation, execution, chains, transitions and history.
Model fits are per resolution with the same original animal split. Compare
held-out real-vs-null gains, not absolute reconstruction errors across different
vocabulary sizes. Keep non-significant effects and do not choose a level based
only on its p value. Resolve matching sets by explicit raw memberships; factor
numbers and renumbered cluster IDs are not identities across levels.

## Latest task: Karen behavioural atlas (2026-10-12)

The user authorised broad syllable, chain, within-syllable and native-frame
analysis of the latest Karen tandem pickle, reusable implementation and CSVs/
report under `outputs/karen_behavior_atlas_20261011`. Explicit permission to load
`data/sessions/sessions_karen_project_tandem_cohort_static_21kp_seed101.pkl` was
received after automatic review requested it. Pre/post means untreated weeks
versus the separate L-DOPA wave; groups are PFF/monomer, not genotypes. The user
wants promising nonsignificant effects and larger-cohort design scenarios,
not a binary significance-only interpretation.

Baseline is still dirty `3ec30fb` on `codex/modular-refactor`; unrelated workflow/
refactor work below is preserved. No commit, push, source-pickle/model change,
environment upgrade or notification has been performed.

Completed empirical measurements: 99 sessions, 21 mice (12 PFF/9 monomer), five
waves, 100 states. Eleven outside-roster/wave sessions were excluded and logged.
Every included native-frame label matched V2 results/VFI exactly. Gap-aware RLE,
exact overlap suppression and seqNMF peak suppression are retained. Shared
algorithms are in `src/fpcode`; cohort policy, annotation/report and entry points
are in `studies/karen_pff_vs_monomer/analysis`. See
[atlas workflow](../../studies/karen_pff_vs_monomer/analysis/BEHAVIOR_ATLAS.md).

Measurements include usage, execution, diversity, transitions, time bins,
phase/onset profiles, frozen exact/seqNMF chains, grouping sensitivity, prefix
edits, execution variants, history prediction, chain context, 2D confidence-
masked pose, native-time seqNMF and frame-level micro-motifs. Kinematics now
smooth centroid positions/circular heading over 9 frames before derivatives,
then use a 5-frame speed median. Frozen chain calls were refreshed using these
measurements. Original raw derivatives are retained. Temporal inference was
restarted with saved templates and eight CPU workers; final micro-motifs use
corrected movement estimates. Forty example clips passed complete decoding,
clock, dimension and interval checks (10,993 frames); behaviour is unreviewed.

SeqNMF pilot: 12 training/4 validation/5 test mice, six settings, selected K10/L3/
penalty .01, held-out reconstruction error .801367, alternate-seed matched
cosine .450. Frozen within-block shuffle gain is small (~.006), and first-order
Markov null gain changes sign. This does not establish reproducible higher-order
routines or MATLAB solver equivalence. Source-specific numerical checks and
the original benchmark outputs are recorded.

All empirical analysis and report generation are complete. Final atlas: 8,493
features, 118,250 group contrasts (76,761 tested) and 118,902 within-group paired
change rows. There are no family/scope or global q < .05 group-permutation hits;
this is not evidence of equivalence. A separate untreated wave-adjusted model
has two family-corrected hits: S78 left-hind relative speed (21 mice; tracking
review required; global q ~.0435) and S73 chain-context duration (only eight
mice; unassigned state; global q ~.176). These are not interchangeable tests.

The report includes 1,640 rows meeting the exploratory magnitude/direction/
leave-one-mouse-out heuristic, a 30-row diverse shortlist, separate per-wave/
L-DOPA candidates, held-out history summaries, fixed-effect sensitivity and
shortlist-specific pilot/50%-shrinkage assumptions. Correlated/duplicated rows
are not independent findings. Examples worth inspecting: pooled untreated
speed PFF minus monomer -5.65 px/s (g -.646, 95% interval -13.09 to +1.69), lower
S71 left-turn usage (-.00854 time fraction; g -1.02), lower S55 within-bout latent
dispersion and left-turn/dash/left-turn prefix progression. They remain selected
pilot effects. Fixed true d=.3/.5/.8 at alpha .05 implies illustrative 80%-power
N176/64/26 per group, assuming equal-variance independent normal outcomes and no
attrition; these are not assurances of replication.

Deliverables: `report.html`/`report.md`, 25 PNG/PDF figure pairs, 49 indexed CSVs
(plus 99 per-session bout files), feature dictionary, clip gallery, executed
stage provenance and `validation.json`. Original-input fingerprints, unique
mouse panels/N bounds, q bounds, completed native/micro feature identities,
all embedded report PNGs, PDF headers and local links passed checks. Session
and follow-up figures were visually inspected. Annotation-review flags include
unknown/Dump states inside chains; duration and slope units were checked.

Validation so far: full CPU suite 470 passed (six expected historical warnings),
159 canonical modules / zero dependency violations or cycles, strict lint of
all new sources passed; repository lint retained 3,079 reviewed diagnostics.
Wheel/package check passed outside checkout without changing the lab install.
Maps verified for 140 directories. Plot-title configuration and final report
passed strict lint and actual plotting. Tests do not establish universal
scientific equivalence. The first statistical pass is superseded by the final
run. Age/wave/drug confounding, sparse 24-week N,
unknown/Dump states, pose tracking noise and 1,999-permutation tail resolution
must remain visible in the report.

Next scientific step: inspect videos/tracking for the shortlisted endpoints,
choose a few biologically meaningful measures and observation waves, freeze
their definitions, then use independent balanced animals to validate them.
No larger-cohort collection, drug intervention or publication is authorised by
this completed analysis request. No analysis jobs should remain running.

## Earlier completed agent-workflow task

Updated 2026-10-11. This is the continuation entry point; earlier refactor reports and the Antigravity handoff are dated evidence.

## Objective and baseline

The user authorized five agent-workflow improvements: scoped instructions, accurate navigation, focused skills, diagnosable setup/provenance and persistent task state. Baseline committed source: `3ec30fb` on `codex/modular-refactor`. Uncommitted folder maps, generator and Makefile/pre-commit integration were present before this work and are being integrated rather than discarded.

## Completed agent-workflow work

1. Added root guidance and scoped `AGENTS.md` files for the library, studies, compatibility and operations. Corrected generated navigation: actual module counts, qualified relative imports, owning-source links and bottom-up rollups. Git-visible maintained files determine clone-reproducible maps; data/cache trees and symlinks are pruned. Removed nine obsolete generated maps. Check mode now detects obsolete maps as well as content drift; regeneration preserves other historical notes.
2. Added `fpcode-doctor` / `python -B -m fpcode.cli.doctor`: JSON or text diagnostics for interpreter, installed distributions, explicit inputs and code/data roots. No scientific imports, unpickling, installation or directory creation. `make test-cpu` forbids modeling imports in the main test process; CPU CI and empirical collection use that guard. Historical checkpoint imports and the runtime shim now preserve laziness and identity.
3. Added focused `repo-validation` and `kpms-run` skills and rewrote `kpms-visualizations`. Workflow references explain public optimization APIs, actual training/render schemas, PCA/resume behavior, explicit device/output scope and native video-frame mappings. Updated the nomenclature rule's stale reference.
4. Extended existing atomic `run.json` records with distribution versions, already-initialized JAX/device evidence and caller-supplied `resource_estimates`. Estimates retain explicit units/method/scope; unavailable estimates are null. A training profile can supply that mapping. No automatic memory/runtime prediction or shared-environment modification.
5. Exercised the agent routes: find canonical optimization owners, installed help/doctor outside checkout, missing external input diagnosis, both historical/canonical import orders, harmless training/render dry-runs and a real fixture-backed trajectory PNG. Added meaningful regression coverage and this persistent continuation record.

See [validation](../agent-workflows/validation.md), [KPMS](../agent-workflows/kpms.md), [visualization](../agent-workflows/visualization.md) and [the navigation index](../../ARCHITECTURE_INDEX.md).

## Validation on 2026-10-11

Selected interpreter: `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`, Python 3.10.19. Commands ran against the dirty working tree based on `3ec30fb`; changes are not committed/pushed.

| Check | Result |
| --- | --- |
| `python -B ops/repo/cpu_check.py` | 456 passed, six expected historical deprecation warnings; modeling imports forbidden |
| Follow-up navigation regressions after the final map-only refinements | Four passed, including external symlink preservation and cache pruning |
| `python -B ops/repo/cpu_check.py --collect-empirical` | 16 collected; bodies/data/model fitting not executed |
| `make architecture` / dependency audit | 142 canonical modules, zero violations, zero cycles |
| `make lint` plus strict lint on new/changed tooling | Passed; 3,079 reviewed diagnostics remain, inventory not refreshed |
| `make folder-maps-check` | 140 directories synchronized; zero obsolete maps remain |
| Generated-map/instruction/skill local references | All checked references exist; metadata frontmatter validator passed for all three skills |
| `make package-check` | Source-only 0.2.0 wheel installed in temporary venv outside checkout; resources/type marker/canonical imports, six CLI help commands and backend-free doctor passed |
| Fixture workflow exercise | Doctor CPU/render profiles passed; missing named input exited 1 without creating directories; training/render dry-runs created no output; actual selected PNG decoded at 1200×1200, 63,273 bytes; source HDF5 unchanged; provenance successful and backend uninitialized |
| `git diff --check` | Passed |

The preview and synthetic inputs are disposable files under `/tmp/fpcode-agent-experience-85mlxja6`; durable behavior is covered by the checked-in tests. Wheel verification reuses available scientific dependencies, so it does not prove a completely fresh dependency installation. CPU CI configuration is updated; remote GitHub execution has not been observed.

The lab interpreter currently reports installed distribution metadata 0.1.0 while checkout/wheel source is 0.2.0. This is reported accurately; the shared environment was not reinstalled. Use module invocation until the package installation is deliberately refreshed. Automatic instruction/skill discovery by each different agent client still needs verification in that client; root guidance explicitly routes readers to the scoped files.

## Nontrivial continuation details

- `FP_DATA_ROOT` is the mirror containing `data/`; `--data-input data/...` resolves there. Training profile strings are resolved by their own consumer and do not all inherit that root. Use explicit paths for external training inputs.
- Training dry-run checks settings/preprocessed-data existence. Fresh fitting uses existing PCA or fits it; an initialized resume needs no PCA. The fitter resumes an earlier checkpoint toward the requested endpoint; at/beyond-endpoint checkpoints are rejected. This is not a general overwrite/resume flag.
- The render loader expects VFI to name native video frames, increasing and nonnegative, one per label row. Never infer model/video clock conversions from duration or maximum index. Density selection has population/PCA eligibility requirements; do not silently substitute random sampling.
- `fpcode.kpms.checkpoints.compact_checkpoint` retains its historical full/slim policy and return schema; it is distinct from the newer export compactor. Explicit checkpoint patches restore each namespace's prior object. The historical runtime shim aliases the canonical module without eagerly binding backend attributes.
- Backend/device provenance inspects cached JAX clients only. An unsupported registry is marked unknown; no fallback initializes JAX. Resource estimates are supplied evidence, not inferred guarantees.
- Generator `--check` is read-only. Regeneration owns only generated map files in its selected roots, including removal of obsolete maps; other source/notes and external symlink targets are preserved. Without Git metadata it indexes the nonexcluded export tree, which may include local files absent from a clone.

## Existing broader refactor work

Session-source orchestration and generic movie composition extraction remain unfinished; the proposed new mechanisms do not exist yet. Remaining study path consumers and vendor extension ownership need review. Ignored operational diagnostics need source-staging decisions. Full-cohort scientific validation and complete empirical recovery have not been established.

Before this task: 440 default tests passed locally, audit 140 modules/zero violations/cycles, lint passed with accepted debt. These are historical starting measurements.

## Next step

The five approved improvements are complete and ready for review. Continue broader refactor work only in its chosen scope, starting from the limitations above. No cohort fitting, production compaction, notifications, commits or pushes were performed as part of this setup task.
