# General chain organisation: execution plan

Date: 2026-10-12. Status: **steps 1–6 implemented, executed and validated; companion report complete**.

The user authorised steps 1–6. The implementation and reproduction commands are
in [GENERAL_CHAIN_ORGANISATION.md](GENERAL_CHAIN_ORGANISATION.md). Outputs are in
`outputs/karen_general_chain_organisation_20261012`; this document retains the
scientific definitions and initial choices for review.

## Purpose and user intent

Test whether PFF mice generally repeat, persist in, restart or circulate between
recognised behavioural routines differently from monomer mice. Existing reports
mostly measure those properties per chain. They do **not** currently provide a
dedicated overall chain self-transition tendency.

The user wants promising nonsignificant patterns, effect sizes, uncertainty and
larger-cohort follow-up value—not a significance-only screen. Analyse the same
99 recordings/21 mice at individual weeks, pooled untreated/all waves and paired
longitudinal changes. PFF/monomer are experimental groups, not genetic groups.
Pre/post L-DOPA means earlier recordings versus the separate L-DOPA wave;
drug, age and recording wave are confounded. The 24-week baseline has only nine
mice (six PFF, three monomer).

Continue into implementation within the authorised scope when the planning
phase ends; a repeated conceptual permission request is unnecessary.

## Start here: repository and existing results

Read `docs/tasks/CURRENT.md`, `docs/architecture/UNIT_CONTRACT.md`,
`src/fpcode/AGENTS.md` and `studies/AGENTS.md`. Preserve unrelated dirty work.
Reuse shared mechanisms; put new reusable timeline/statistical mechanisms in
`src/fpcode`, and Karen library choices, thresholds and reporting in this study.

Repository root on the lab machine:
`/home/fsp585/Desktop/FiberphotometryCode`.

Existing output roots, relative to that root:

- `outputs/karen_behavior_atlas_20261011`: verified raw atlas, sessions,
  frame cache, raw annotations, original chains and discovery split.
- `outputs/karen_clustering_levels_20261012`: L1–L4 comparison, unresolved-state
  sensitivities, frozen mapped anchors and cross-level reports.

Useful inputs:

- Raw atlas `sessions.csv`, `frame_cache/<session_id>.npz`,
  `chains/animal_splits.json`, `chains/chain_catalog.csv`,
  `chains/chain_occurrences.csv` and `measurements/bouts_by_session/*.csv`.
- Resolution output `cluster_membership.csv` and each level's
  `cluster_annotations.csv`, `mapping.json`, `anchors/anchor_mapping_audit.csv`,
  `anchors/anchor_occurrences.csv`, and, at L2/L3/L4, `chains/chain_catalog.csv`
  and `chains/chain_occurrences.csv`.
- Existing `chain_transitions.csv` and `chain_grouping_sensitivity.csv` are useful
  checks, but are not sufficient by themselves to reconstruct unmatched time,
  all eligibility boundaries or overall null-adjusted tendencies.

L1 uses the original atlas's discovery chains. The primary hierarchies have
100/93/22/14 states. Unresolved-preserving L3/L4 have 69/61 states. Those
sensitivity runs have frozen anchors but **no newly fitted seqNMF models**.
Do not imply otherwise. Factor IDs are local to a fit; use raw membership and
anchor crosswalks for comparisons, never equal factor numbers across levels.

Relevant existing owners:

- `fpcode.behavior.sequences.chains`: exact matching, prefix progression and
  `group_occurrences`.
- `fpcode.workflows.sequence_atlas`: overlap resolution (`_exclusive_events`),
  episode tables and existing order-null audit. Inspect its tie behaviour.
- `fpcode.workflows.anchored_sequences`: frozen motifs mapped/RLE-collapsed across
  levels, including collapsed-pattern audit.
- `fpcode.behavior.resolutions` and `fpcode.workflows.resolution_atlas`:
  state mapping, membership and composition sensitivity.
- `fpcode.statistics.behavior_comparisons`: mouse-level contrasts, paired changes,
  wave adjustment and BH correction.
- `fpcode.statistics.followup_planning` and `fpcode.viz.behavior_atlas`:
  effect stability, prospective sensitivities and reporting.
- Study policy: `karen_hierarchy.py`; workflow notes: `BEHAVIOR_ATLAS.md` and
  `CLUSTERING_LEVELS.md`.

## 1. Freeze the library and the estimand

Primary proposal: **L3, interpretable frozen exact anchors derived from the
original training-mouse motifs**. Choose eligibility from labels/membership and
support, without looking at PFF–monomer differences. Export the selected library
and all exclusions before inference. Confirm that it has enough coverage and
transition opportunities to support an overall measure; if not, report that
limitation rather than expanding it to obtain a stronger effect.

Sensitivity analyses:

- Corresponding anchors at L1/L2/L4 and the unresolved-preserving variants.
- Fresh exact-discovery libraries at primary L1–L4.
- SeqNMF libraries analysed **separately** from exact motifs.
- Full inclusive versus interpretable-member policies, explicitly labelled.

No template refitting is needed for the first pass. Use existing frozen calls,
catalogs and discovery splits. Define all results as tendencies **among detected
routines**, not universal properties of behaviour. Coverage is part of the
result. A library detecting few routines cannot justify broad conclusions.

## 2. Construct a complete, auditable event timeline

For each session and library:

1. Use half-open native-frame intervals `[start_frame, end_frame_exclusive)`.
   Read the recording's explicit FPS; Karen's verified primary clock is 60 Hz.
   Keep missing-frame/invalid-label boundaries from the numerical cache.
2. Resolve overlapping calls into an exclusive chronological stream, using a
   documented deterministic score/tie rule. Retain original overlapping calls,
   selected/rejected identities and rejection reasons. Compare with the existing
   resolver before changing shared behaviour.
3. Represent unmatched valid time, excluded/unresolved detections, missing time
   and recording boundaries explicitly. **Do not filter events and then join
   their former neighbours.** An excluded intervening chain is a barrier.
4. Export each possible successor pair with IDs, frames, gap, validity segment,
   self-transition flag, eligibility for each gap policy and exclusion reason.

Two different notions must remain separate:

- **Strict adjacency:** intervals abut, with no intervening event, unmatched
  interval or missing time. This is the strict immediate self-transition.
- **Gap-tolerant succession:** consecutive eligible calls within an explicitly
  allowed valid unmatched interval, with no intervening different/excluded chain
  or missing time. Call this nearby succession, not strict immediacy.

Use 1 second as the proposed primary gap-tolerant policy, frozen before looking
at effects, and 0/0.5/2/5 seconds as sensitivity. Also export chain-only
next-detection transitions as a secondary diagnostic; they can bridge long
unrecognised periods and must not be mistaken for immediate continuation.

RLE already collapses consecutive identical **syllable frames**. Preserve it and
existing overlap/peak protections. Do not interpret self-transitions removed by
syllable RLE as an unresolved implementation issue. This task concerns repeated
multi-action chain calls.

## 3. Overall measurements

Compute session summaries first, with explicit numerators and denominators.
Zero events/opportunities produce missing probabilities or run means, not zero
repetition. Rates can be zero when modelled exposure is positive.

| Measure | Operational definition |
| --- | --- |
| Overall self-transition probability | Eligible same-ID successor pairs / all eligible successor pairs |
| Chain-specific self-transition probability | Same-ID successors from A / eligible successors from A; diagnostic contribution |
| Mean repetition-run length | Mean number of calls per maximal same-chain run, including singleton runs |
| Longest repetition run | Maximum calls in a run; descriptive, sensitive to recording duration/event count |
| Repeated-call fraction | Calls after the first in their run / all eligible calls |
| Initiations | First calls of runs, including singleton runs; per modelled minute and per chain call |
| Repeated-run time | Union of chain-call durations in runs of length >=2 / modelled time; separately relative to detected-chain time |
| Exit probability after r calls | Observed switches after repetition position r / observed eligible successors at that position |
| Short-loop return | Return to starting chain after 2, 3 or 4 eligible transitions, with intermediate IDs distinct from the start |
| Simple cycle return | As above, additionally requiring distinct intermediate IDs; report separately from inclusive returns |

For loop returns, all steps must satisfy the same eligibility policy within one
valid block. Distinguish A→A persistence from A→B→A circulation. Treat overlapping
return windows as descriptive windows, not independent samples. If counting
distinct cycles, define rotation equivalence and a non-double-counting rule first.

Separate successful exits from loss of observation: recording ends, missing
data, excluded calls and excessive gaps censor continuation. They are not all
behavioural exits. Store run boundary/censor flags and report complete-run
summaries versus censored lower bounds. Longest-run comparisons need event-count
or exposure sensitivity because longer recordings offer more opportunities.

## 4. Frequency and transition baselines

### Frequency-preserving categorical shuffles

Within each session's declared valid reference block, shuffle chain IDs while
preserving their exact counts and the fixed event slots/gap eligibility. Keep
barrier markers fixed. Reference blocks must not be defined from observed
same-chain runs. Distinguish the reference blocks from gap-based transition
eligibility: conditioning on many two-event blocks can remove the repetition
signal by construction. Document the block policy before examining group effects.
This controls the fact that frequent chains naturally generate more
self-transitions and returns. Add within-recording-time strata as a sensitivity
to slow changes in chain usage, where support permits.

Use exact analytical frequency-matched expectations for endpoints where the
chosen block/eligibility policy permits them; validate against a small shuffle
reference. Use bounded batches of simulations for null intervals and endpoints
without an adequate analytic calculation, reusing invariant eligibility arrays.

For overall self-transition probability, run length in **calls**, and return
probabilities, export observed value, null mean, null interval, observed-minus-
null excess and Monte Carlo precision. Use a fixed seed and a declared replicate
budget; choose >=1,999 if computationally practical. Count independent animals
in group inference, not null replicates or transition windows.

This ID shuffle breaks label–duration association. Do not use it to claim a
duration-based significance result. Repeated-run seconds can remain descriptive
unless a separate justified duration-aware surrogate is implemented and tested.

### First-order transition simulations

Use estimated within-session/block transition probabilities to assess longer
run shapes and multi-chain circulation beyond ordinary one-step structure.
Keep starts, block lengths and eligibility policy explicit; document sparse-row
fallbacks and finite-sample fitting uncertainty. Estimate without borrowing group
differences. Simulations preserve transition probabilities **in expectation**,
not exact observed transition counts.

**Do not use a self-transition-preserving first-order null to test the primary
self-transition probability itself.** That probability is built into the null.
These controls address higher-order persistence/circulation. A longer-history
effect after state aggregation may reflect hidden raw-member identity; avoid
automatically interpreting it as a new biological routine.

## 5. Primary endpoints and exploratory extensions

Proposed primary set, at the frozen primary library/gap policy:

1. Overall self-transition **excess over the frequency-matched null**.
2. Mean repetition-run length in calls, with singleton/censor handling reported.
3. Two-step return **excess over the frequency-matched null** (A→B→A).

Exploratory extensions: strict adjacency, other gaps/levels/libraries, longer
loops, exit curves, initiation rates, concentration in dominant chains,
transition entropy, effective number of routines and changes across recording
time. Coverage and recognised/unrecognised exposure are essential diagnostics.

Always run **leave-one-chain-out** checks to determine whether an overall effect
comes from several routines or one dominant chain. Excluding a chain must create
barriers rather than artificial new neighbours. Include leave-one-mouse-out
checks and review dependence on unknown/Dump states and overlap resolution.

## 6. Mouse-level comparison and follow-up interpretation

Produce one value per mouse/wave. For probabilities, primary within-wave values
can pool eligible numerators/denominators across that mouse's recordings; export
equal-session averaging as a sensitivity. For rates, pool counts/exposure.
For run means, retain counts/sums and explain the weighting. Never count repeat
recordings as additional independent mice.

Compare PFF minus monomer at each wave, pooled untreated, all waves and paired
changes (including the existing adjacent/long-range untreated pairs). Use the
wave-adjusted mouse-clustered analysis separately from pooled descriptive means.
Pooling available wave summaries equally per animal is one estimand; report
attendance and consider balanced-attendance sensitivity.

Report effect size, original-unit difference, individual-mouse uncertainty,
sample support, week direction, leave-one-out stability and correction scope.
Separate the three primary endpoints from the exploratory screen; account for
the number of levels/gaps tested. Do not choose the most significant resolution
as though it were prespecified. Preserve promising nonsignificant findings.
Any new-cohort sensitivity is conditional on a plausible effect/endpoint design,
not proof that more N will validate a selected pilot estimate.

## 7. Output and provenance contract

Select a **new explicit output root** at execution time, for example
`outputs/karen_general_chain_organisation_<run-date>`. Preserve both earlier
atlases and all empirical inputs. Reuse the verified numerical cache; no pickle
reload, KPMS fit or environment upgrade is needed for this plan.

Save CSVs for:

- Frozen library/membership policy and coverage audit.
- Full/selected event timelines and pair eligibility/exclusions.
- Run boundaries, censoring, return windows and session-level numerators/sums.
- Null replicate summaries and excess measures.
- Mouse/wave endpoints, group contrasts, paired changes and model results.
- Gap/level/library and leave-one-chain/mouse-out sensitivities.
- A follow-up candidate table, feature definitions/units and file inventory.

Produce a companion HTML/Markdown report with mouse-level weekly plots, observed
versus null comparisons, run-length/exit curves, short-loop summaries and a clear
distinction between strict adjacency and gap-tolerant succession. Record input
fingerprints, exact library choice, source hashes, seeds, all thresholds and
null/weighting rules using existing provenance mechanisms.

User preference: run independent workloads in parallel and use multithreading
for OpenMP/OpenBLAS. The lab has previously exposed 32 CPUs; check availability
at execution. A reasonable starting budget is four threads per process with
bounded parallel processes. Set `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4`
**before numerical imports**, verify actual pools where available, record the
configuration and avoid oversubscription. Retain completed fits rather than
repeating them solely to change threading.

Implementation should vectorize pair/return masks and run-boundary calculations,
cache timeline/eligibility arrays across gap policies and reuse simulated blocks
across compatible endpoints. Set explicit RAM/VRAM caps and chunk sizes, including
temporary-array and worker-copy costs. Compare a representative CPU calculation
with an available GPU path only when simulation throughput is substantial enough
to justify startup/transfers; retain appropriate precision and a CPU path. Do not
use a GPU solely because it is present. Follow root `AGENTS.md` for this policy.

## 8. Validation and completion criteria

Use the existing interpreter when suitable:
`/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`, with
`PYTHONDONTWRITEBYTECODE=1`. Follow the repo-validation skill/current validation
guidance; the latest completed suite had 475 passing CPU tests.

Meaningful small expectations should cover:

- A A A: two self pairs, one three-call run, one initiation; censor flags at ends.
- A B A: no self pair, one eligible two-step return.
- A then missing data then A: no linked pair/run/return.
- A then excluded B then A: no manufactured self-transition after filtering.
- The same sequence just inside/outside the gap threshold, with native units.
- Overlap selection, deterministic ties and no double-counted time.
- No events/one event/zero opportunities: correct missing values and rate zeroes.
- Incomplete runs are not counted as observed switches.
- Frequency-only synthetic data: null-adjusted excess near zero in aggregate.
- A controlled persistence example: excess detected beyond frequency alone.
- A first-order synthetic example: higher-order diagnostic behaves as expected;
  do not assert that its self-transition probability exceeds its own fitted null.
- Repeated sessions do not inflate independent mouse N; excluded-chain removal
  preserves barriers; simulated/shuffled labels never cross validity blocks.

Run focused tests, then the required CPU suite/architecture/lint checks for shared
behaviour changes, map synchronization and installed-package checks when needed.
Batch the broader checks once against the final implementation rather than
repeating them after each edit; repeat only checks affected by later changes.
Do not refresh lint debt or gold fixtures to conceal a new diagnostic. Verify
CSV identities, bounds, exposure conservation, unchanged inputs and report links.
Synthetic checks do not establish universal ethological validity.

Completion means the three primary endpoints and declared sensitivities are
computed, audited and reported with interpretable limitations—not merely a list
of per-chain results. Update `docs/tasks/CURRENT.md` with actual commands, output,
checks, findings and any unresolved work. No runnable general-chain command
exists yet; implement an explicit reusable workflow/study entry point before
documenting its final invocation.
