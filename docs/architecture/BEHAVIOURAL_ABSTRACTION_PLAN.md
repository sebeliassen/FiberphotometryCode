# Behavioural abstraction and independent method choices

Date: 2026-10-12. Status: accepted design; first structural implementation complete.

The user wants a reusable behavioural-analysis framework organised by layers of
abstraction, with adjacent methods at each layer. This plan answers which layers
exist, what belongs to each, and how choices can remain independently selectable.
The user subsequently authorised module structure, mechanical relocation,
documented placeholders and independence/interface tests. The implemented
inventory is in [behavior/README.md](../../src/fpcode/behavior/README.md).
No new cohort fit or changed detector definition is part of this first increment.
The current [capability contract](UNIT_CONTRACT.md) continues to govern ownership.

## 1. Separate four decisions

| Decision | Question | Example |
| --- | --- | --- |
| Abstraction | What kind of behavioural unit do we represent? | Syllable, routine, assembly |
| Resolution | How finely do we distinguish units at that level? | Raw syllables versus L2/L3 category views |
| Method | How do we identify the units or their relationships? | seqNMF versus staged episode model |
| Analysis | What do we measure within or between those units? | Pacing, completion, repetitions, transitions |

Changing resolution within a layer is different from constructing a higher unit.
The existing L1–L4 syllable classifications are resolution choices, not four
successive layers of temporal composition. Inter/intra analysis is a measurement
axis that can be applied at several unit levels.

The target is independently selectable methods and traceable dependencies.
Scientific results will still depend on the observations, vocabulary, boundaries
and assumptions used to define the units.

## 2. Unit layers and a parallel context track

These are proposed conceptual layers, not new package directories or a mandatory
pipeline. Keep names distinct from the existing L1–L4 resolution labels.

| Layer | What is represented | Current groundwork | Adjacent choices |
| --- | --- | --- | --- |
| Observation and fine movement | Native frames, pose geometry, trajectories and continuous movement features | Verified clocks/caches, confidence masks, kinematics, pose and micro-motif measurements | Different feature sets or trajectory representations; optional inferred micromovement units |
| Primitive | Short movement units such as syllables | KPMS labels, gap-aware bouts, annotations, explicit resolution views | Alternative segmenters or primitive representations; different resolutions of a fixed vocabulary |
| Episode | Recurring arrangements of primitives, with possible internal phases | Exact chains, seqNMF factors/calls, progression and variants | Shared staged HMM/HSMM episodes, restricted grammars, trajectory-grounded episode detectors |
| Assembly | Larger arrangements of episodes that have an identifiable structure | Repetition runs, return windows and loop measurements provide candidate structure | Repeated subroutines, constrained episode grammars, larger motifs and process representations |
| Context track | Persistent regimes that influence recruitment, transitions or execution | Recording-time analyses and previous shMoSeq/context prototypes | Context HMM/HSMM, shMoSeq-like dynamics, measured spatial or experimental context |

Assemblies are optional. A transition table or repetition statistic can describe
episodes without defining a new assembly unit. An inferred assembly requires an
explicit definition, occurrence boundaries and evidence of useful structure.

Context can attach to primitives, episodes or assemblies; it need not be an
ordered assembly. A long regime can contain many different sequences. In the
updated working layout it is the highest behavioural description, while retaining
its own time-aligned track and lower-level inputs. An optional supervised
cohort/genotype/condition readout sits above it and can consume context and lower
behavioural descriptions directly.

Method names do not fix their position. A staged HSMM may identify episodes next
to seqNMF, while a context HSMM describes longer regimes. A grammar may compose
primitives into episodes or episodes into assemblies.

Composition can also be an adjacent component in an inter-chain analysis
configuration: transition/run/return measurements and an episode grammar consume
the same frozen episode stream. A caller can inspect the grammar's relationships
and derivations alongside those measurements, or select its parsed intervals as
assembly units for further inter/intra analyses. This does not impose another
mandatory pipeline stage, change primitive resolution or refit lower episodes.
Keep the reusable component in its existing owner and compose it through workflows;
shared measurements continue to consume results rather than detector internals.

The subsequent planning decision combines composition and inter/intra analyses
into unit5's organisation/composition workspace: run all applicable complementary
methods on shared episode evidence. Measurements remain available across levels
rather than forming a separate final abstraction layer. Composition alternatives
keep their own identities; a union of methods does not pool duplicate occurrences.
See section5 of the [broad search plan](BROAD_BEHAVIOURAL_SEARCH_PLAN.md) for the
updated numbering, DOCX relationship, context and optional supervised readout.

The supplied *Hierarchical Models of Sequential Behaviour* report motivates shared
episode definitions, alternatives, optional parts and loops across parent routines.
The current implementation is a shallow caller-specified finite grammar, with
bounded repeats and traceable child reuse. Automatic grammar induction, recursive
subroutine calls and context-controlled recruitment remain separate future method
choices, not capabilities established by inter-chain statistics or this first pass.

## 3. Apply within-unit and between-unit analyses across layers

| Unit being studied | Within-unit examples | Between-unit examples |
| --- | --- | --- |
| Movement trajectory / micromovement | Shape, coordination, speed profile | Ordering, recurrence, coupling of body-part movements |
| Syllable | Pacing, posture variation, constituent micromovements | Usage, switching, transition history |
| Episode | Progression, alternatives, omissions, completion and timing | Recruitment, repeated invocations, switching and returns |
| Assembly | Branch choice, constituent repetition, completion and pacing | Reuse, switching between assemblies, context dependence |

Loops need an explicit scope: repeating a stage inside an episode, invoking the
same completed episode again, or circulating among several episodes. These are
different measurements even if the same model can represent all three.

A frame provides pose geometry; movement identity and timing require neighbouring
observations. Keep frame-level provenance without forcing every frame to be a
separately named behavioural component.

## 4. Shared foundations that permit independent choices

### 4.1 Preserve native observations and physical time

All branches refer to the same recording identity and native frame/time axis.
Retain units, coordinate systems, confidence, missing data, valid intervals and
exposure. Use existing `SessionKey`, `TimeAxis`, `PoseSeries`, `EventInterval`
and `Bout` foundations where their semantics fit.

Calculate reusable continuous features independently of behavioural labels.
Summaries within a syllable or routine then use that unit's intervals. Different
feature sets may have different validity masks; document their support rather
than silently forcing one global complete-case mask.

Higher methods may consume an appropriate lower event view or continuous
observations directly. Skipping an intermediate layer is permitted and recorded.

### 4.2 Make classification a view over a preserved base vocabulary

For the initial broad exploration, preserve the frozen KPMS source bouts. Different
executions may receive whole-bout subtypes or continuous descriptors, without
splitting an individual bout or composing several syllables into a new primitive.
Primitive resolution groups physical types/executions; alternatives sharing an
episode role are handled by episode substitutions/optional parts rather than by
globally merging their primitive identities. The broader independent segmentation
baseline remains a separately declared representation. See sections2.1–2.2 of the
[broad search plan](BROAD_BEHAVIOURAL_SEARCH_PLAN.md) for these clarified choices.

Keep raw labels, original bouts and membership crosswalks. Store coarser labels
and merged bouts as derived views, with their physical intervals and constituent
original bouts. Human descriptions annotate types; they are not the identity key.

Mapping followed by RLE can merge boundaries and alter the bout clock. Methods
must declare whether their lag is in frames, seconds, original bouts or merged
bouts. Preserve native time even when modelling uses a different coordinate.
Changing resolution therefore invalidates dependent token-based fits; it should
not require recomputing independent native kinematics.

### 4.3 Exchange common results while retaining method-specific information

An adjacent detector should expose a small common result:

- The unit-type catalogue, local model/type identity and model provenance.
- Unit occurrences with recording identity and half-open physical intervals.
- Observational support, missing/unmatched/background distinctions and censoring.
- Links to source observations and constituent units where applicable.
- Optional capabilities: stages, membership uncertainty, completion evidence,
  alternative assignments, overlapping occurrences and model-specific scores.

Do not collapse every method into a hard sequence as its only output. Maintain
full seqNMF templates/activations and probabilistic models' uncertainty alongside
an explicit event view for analyses that require discrete occurrences. An
activation score is not automatically a calibrated probability. Unsupported
capabilities stay unavailable; adapters must not fabricate stages or completion.

Overlap resolution, membership exclusions, confidence thresholds and exclusive
stream construction are explicit analysis-view choices. Retain original calls.
Missing/excluded intervals remain barriers where the measurement requires them;
unmatched valid time does not automatically mean inactivity.

### 4.4 Keep shared analyses independent of detector internals

Usage, execution, transition and recurrence analyses consume documented unit
views rather than importing a particular fitter. Replacing an episode detector
should leave reusable measurement implementations unchanged wherever the new
result supports the same measurements.

Every endpoint retains its definition, support and denominator: occurrences per
valid minute, eligible self-successor pairs, completed invocations, observed
exits versus censoring, and constituent timing. Distinguish continuation within
an invocation from a completed invocation followed by a new one. Some methods
cannot supply that distinction; mark the corresponding endpoint unsupported.

Representations with different boundaries can produce different estimands even
when the endpoint has the same name. Explain coverage and definition differences
before interpreting disagreement as a biological contradiction.

### 4.5 Make dependencies explicit and reuse frozen results

Each configuration identifies its evidence, resolution view, detector, optional
assembly/context model and requested analyses. Reuse cached features and frozen
fits whose inputs and semantic settings match. Changing an upper method reuses
the same lower representation; changing a lower representation refreshes its
dependents. Changing a report style or adding an independent metric needs no refit.

Default comparisons use modular fitting with frozen lower units. Joint models
that revise lower boundaries or labels form a separate declared branch, with all
affected layers recorded. They can be useful but provide less isolation between
choices. Never silently replace the baseline lower representation with joint-fit
outputs during an upper-layer comparison.

## 5. Evaluate the choices with controlled comparisons

The [broad search companion](BROAD_BEHAVIOURAL_SEARCH_PLAN.md) describes wider
exploration before settling on methods, including unusual/weak branches, explicit
feature routes and assemblies/context early in the comparison. It is a planning
document; it does not initiate implementation or empirical execution.

1. Freeze a common data scope, physical-time conventions and animal-level split.
2. Compare adjacent episode methods on one fixed primitive representation.
3. Vary primitive resolution with the episode method held fixed.
4. Examine selected method-by-resolution combinations for interactions.
5. Add assemblies or context only when they contribute stable held-out information.

Select methods using held-out prediction on common observable targets, stability,
interpretability, coverage/support and computational cost. Different methods'
internal reconstruction or likelihood objectives are not interchangeable scores.
Pair numerical checks with ordinary sampled occurrence videos, including ambiguous
and unmatched intervals; exemplar videos alone do not establish useful definitions.

Match units across fits using native-time overlap, constituent membership and
physical execution, allowing many-to-many correspondences. A factor/type number
is local to its fitted library. Differences across methods may reveal alternative
definitions rather than a clean one-to-one match.

Hold data splits and comparison scope fixed, and keep representation selection
separate from searching for the largest group effect. Group inference retains
independent animals, paired recordings and repeated-wave structure. Additional
windows, occurrences, methods or resolutions do not increase independent N.

Use a bounded comparison matrix rather than the full Cartesian product. Inspect
interactions explicitly where lower vocabulary or timing affects the higher unit;
changing one factor at a time cannot establish full independence by itself.

## 6. Implementation sequence for a later authorised task

1. **Inventory existing capabilities.** For each current method, record inputs,
   outputs, unit meaning, time coordinate, uncertainty and boundary assumptions.
   Distinguish audited library functions from old study prototypes.
2. **Agree the common unit/result requirements.** Build on existing contracts;
   identify only genuinely missing records and optional capability declarations.
   Avoid a large new framework before concrete method swaps require it.
3. **Adapt current primitive and episode results.** Start with KPMS views, exact
   chains and seqNMF. Preserve existing numerical policies and complete call data.
4. **Route shared analyses through explicit unit views.** Demonstrate that the
   same rates, execution and transition code accepts adjacent detector outputs;
   disclose measurements each method cannot support.
5. **Audit and adapt one probabilistic episode alternative.** Existing
   `studies/sequence_viz/probabilistic_episode_pilot` and shared-subroutine
   prototypes are candidates, not established Karen results. Check actual duration
   and progression semantics before calling a model an explicit-duration HSMM.
6. **Run one bounded pilot comparison.** Freeze inputs, use the same held-out mice
   and evaluate detector/resolution effects before adding a further abstraction.
7. **Extend to optional assemblies/context.** Reuse the accepted unit results;
   require a useful predictive or descriptive contribution from the extra level.

Ownership follows the existing repository structure: reusable contracts,
algorithms and workflows under `src/fpcode`; generic design notes under `docs`;
cohort policy, selected comparisons and scientific reports under `studies`;
explicitly selected generated products under `outputs` or configured storage.
No new layer-numbered package tree or duplicate cohort-only implementation is
required by this plan.

Repeated workloads use the existing resource guidance: cached invariant inputs,
vectorization, bounded batches, RAM budgets and coordinated worker/BLAS threads.
Consider GPU acceleration only after a representative workload demonstrates a
benefit. Validate at meaningful milestones; no empirical work or full numerical
suite is needed for this design document alone.

## 7. Acceptance criteria and limits

The first useful implementation should show that we can:

- Swap an episode method with the lower data/view and analysis definitions fixed.
- Swap a resolution view while reusing independent native features and identify
  exactly which fits, boundaries and summaries need refreshing.
- Run inter/intra analyses at several unit levels without duplicating their logic.
- Preserve original evidence, uncertainty, overlaps, gaps, censoring and units.
- Compare definitions and report limitations when units do not correspond.
- Omit assemblies/context, or add one explicitly, without rewriting the base units.

Independence is strongest for configuration, implementation and cached evidence.
It is partial for inferred unit boundaries and biological interpretation. Coarser
primitives can erase distinctions, and models can disagree about what counts as
one invocation. The practical criterion is whether one choice can be changed
while the others remain explicit and fixed, with its consequences traceable.
