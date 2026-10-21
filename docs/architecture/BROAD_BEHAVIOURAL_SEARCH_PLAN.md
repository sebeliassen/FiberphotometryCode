# Broad exploration of behavioural representations

Scientific follow-up: [Gemini suggestions](GEMINI_SUGGESTIONS_20261013.md) assesses
appropriate additions for interpretable chain families, execution variants,
composition and downstream phenotyping. Recommendations are distinct from the
frozen current run; see [CURRENT.md](../tasks/CURRENT.md) for execution state.

## Additional cohort scope agreed on 2026-10-13

Include Jonathan AMPH alongside Karen PFF/monomer when the broad search is
eventually authorised. The located candidate input is
`data/sessions/jonathan_amph_sessions_v2_w_sylls.pkl`, also referenced by existing
Jonathan language-model and kinematics analyses. Reuse its existing KPMS evidence;
no new KPMS runs are required. Other labelled Jonathan pickle variants exist, so
freeze the selected version and verify model/vocabulary identity before execution.
This is an input-location finding from filenames and source references, not a
fresh inspection of pickle contents or signal quality.

The subsequent verified selection is **Jonathan AMPH Run12**:
`data/sessions/jonathan_amph_sessions_run12_w_sylls.pkl`, paired with
`amph_prod_k150k_run_12` and
`compat/syllable_generation/analysis/sequence_viz/run12_human_labels.csv`.
All 5,500,994 source-model labels/native VFI rows match the saved session syllables
across all 30 recordings. The six supplied grooming IDs belong to this pairing,
not IFDN or an automatically interchangeable v2 vocabulary. Use sparse VFI support
with the verified regular-step projection; exclude the combined table's filled
larger gaps. See `outputs/broad_search_preflight_20261013/annotations.md` and
`native_verified_manifest.json`. The user authorised proceeding with initial
verification and bounded method contrasts; the previous search remains stopped.

Actual stored Run12 signal variants include raw, `_dff`, `_zscore` and
`_zscore_datta`; the builder's newer suffixes must not be assumed present. The
WT-only inventory covers 16 recordings/eight animals, with original physical
fiber, region and hemisphere metadata retained. Existing signal processing and
documentation are reused rather than rebuilt.

User-supplied cohort interpretation: KI mice are the ADHD / late-stage Parkinson
model; WT mice are wildtypes. Retain both genotypes for behavioural exploration,
with saline/amphetamine and pre/post-injection context explicit. Use **WT DA only**
as an optional supporting verification metric; exclude KI DA from this role.
DA association must not become the primary recipe score, a required success
criterion, or evidence by itself that an episode definition is correct. Keep it
outside behavioural model construction and assess it after behavioural choices
are frozen, with animal-held-out evaluation and appropriate exploratory correction.

Before any eventual execution, verify DA channel identity, preprocessing/units,
behaviour–photometry clocks, validity/handling masks, injection timing, animal IDs
and session pairing. Retain missing DA as missing rather than excluding otherwise
eligible behavioural evidence. Drug/time trends and repeated observations need
explicit controls. Compare recipe behaviour within each cohort first; different
KPMS vocabularies require an explicit mapping before cross-cohort routine claims.
Jonathan is a second dataset for method robustness, not automatically an
independent replication of Karen disease effects. This addition authorises no
model fitting or search execution.

Date: 2026-10-13. Status: conceptual plan for discussion and later work.

This document specifies a possible broad search space and the capabilities that
could make it executable. It does not implement methods, prescribe test runs or
launch a search. The previous Karen overnight search remains stopped.

Build on the accepted [behavioural abstraction plan](BEHAVIOURAL_ABSTRACTION_PLAN.md),
the [capability contract](UNIT_CONTRACT.md) and the
[current module inventory](../../src/fpcode/behavior/README.md). This is a companion
to those documents, not a competing architecture or a new package tree.

## 1. Purpose: learn what the choices do

The purpose is to explore how different methods, representations and feature
routes change our description of behaviour. Interesting outcomes include a
stable useful representation, a surprising split into spatial variants, a method
that captures interruptions, a catalogue that collapses into background, or a
method that fails for an understandable reason.

The paper's primary scientific object is **good episodes/chains**. Frame features,
primitive creation and resolution are alternative ways to supply useful evidence
for that object. A lower representation is not necessarily preferable just because
its clusters look tidy; ask what chains it enables. Grammar/composition,
inter/intra measurements, context and cohort readouts are later uses and can also
help diagnose chain quality, without becoming the objective of the search.

The user's concrete target is a catalogue that distinguishes many separate
behaviours while grouping very similar executions into coherent chains. Variable
tempo, minor substitutions, repeated or optional constituents should not necessarily
create unrelated routines. Report constituent diversity and recurring structure,
within-chain similarity, separation between chains, catalogue redundancy and
support across animals. These remain separate quantitative diagnostics, alongside
video review, rather than one score that rewards maximum diversity or coverage.
Fragmentation is an expected outcome across hyperparameters; measure its continuum
instead of excluding every branch that fragments. Homogeneous actions such as
grooming can still form meaningful routines.

The supplied N-gram/seqNMF slides explain the motivation: frequency-ranked exact
strings may split similar routines and emphasise short loops. They are background
material, not proof that every seqNMF fit solves those problems. Exact motifs are
a useful baseline; methods allowing variation must demonstrate useful grouping
in their actual outputs.

Keep the existing labelled KPMS fit as a reusable source of categorical evidence.
New KPMS fitting runs are outside the intended initial scope. Feature-based and
hybrid representations can complement it, including direct feature inputs to
resolution or episode models when those inputs are declared explicitly.

The central questions are:

- What changes when an episode detector is HSMM rather than seqNMF?
- Which effects come from vocabulary resolution, time representation or features?
- When should position accompany a routine, and when should it help define it?
- What do assemblies, shared parts and context add to the same lower units?
- Which interactions matter, and which layer choices remain mostly independent?
- What appears coherent on video, and which numerical improvements are misleading?

PFF/monomer separation, significant contrasts and apparent L-DOPA recovery are
not search objectives. Cohort comparisons can follow later, using explicit
definitions chosen for representation quality. The existing exact/seqNMF sweep
is useful evidence about those branches; it is not a comparison of the entire
framework.

Exploration should not be restricted to branches already known to work. Keep
unusual, unstable, empty and overcomplicated results visible. A failed or weak
branch can identify a structural limitation, an unsuitable representation, or an
insufficient fitting budget.

## 2. Search complete recipes, with independently selectable parts

A recipe describes observations, primitive resolution, temporal representation,
episode method, feature routes, optional assemblies/context and analysis views.
The search samples recipes and compares their outputs. It should also construct
matched recipe pairs differing in one major decision.

Resolution and abstraction remain distinct: L1–L4 are vocabulary views, not four
successive stages of temporal composition. Context is the highest behavioural
description in the working layout, while remaining a time-aligned track that can
consume evidence from multiple lower units. An optional supervised cohort/condition
readout sits above it. Composition and inter/intra measurements belong together
in an organisation/composition workspace.

| Decision area | Candidate alternatives | What the comparison asks |
| --- | --- | --- |
| Frame representation (unit 1) | Movement, posture and spatial channels; selected combinations, transforms and temporal summaries | What information is supplied for each frame? |
| Primitive/bout interpretation (unit 2) | Fixed KPMS bouts; whole-bout execution subtypes; feature-only or hybrid whole-bout clustering; parallel descriptors; separate independent segmentation baseline | What identity or execution description is assigned to each unit? |
| Primitive resolution (unit 3) | Identity; existing KPMS-specific L1–L4; quantitative merges; reference-guided or hybrid grouping | How finely are the created primitive types distinguished, independently of naming them? |
| Time representation | RLE bouts; bouts plus physical durations; fixed physical-time bins; within-bout profiles | Is the model describing order, physical pacing, or both? |
| Episode method | Exact motifs; tolerant/edit-based motifs; seqNMF; staged HMM; HSMM; finite grammar; explicit hybrids | How do detection boundaries, variation and phase structure change? |
| Feature role | Description; episode evidence; transition/duration conditioning; assembly/context evidence | Does a feature explain a unit, define it, or explain its recruitment? |
| Organisation/composition (unit 5) | Run applicable inter/intra measurements together with repeat/return structure, finite grammar and prospective shared-subroutine discovery | What relationships and larger structures describe the same episode evidence? |
| Context (unit 6) | Measured region/movement context; inferred primitive/episode/assembly regimes; optional bypass | What persistent behavioural regimes organise recruitment and execution? |
| Optional top readout (unit 7) | Predict known cohort/genotype or condition from selected behavioural descriptions | What information about the animal/recording is expressed by those descriptions? |
| Event/analysis view | Raw overlaps; declared exclusive selection; different thresholds, gaps and censoring policies | Which conclusions depend on converting model evidence into discrete calls? |

The working numbering is: **1 frame representation → 2 primitive/bout creation →
3 primitive resolution → 4 episodes/chains → 5 organisation/composition →
6 context → 7 optional cohort/condition readout**. Within/between-unit measurement
is a reusable analysis bank across these levels, not a separate highest layer.
This numbering is a conceptual workflow; a measurement need not create a new unit,
and a readout can also consume lower-level evidence directly.

KPMS and its source bout intervals stay fixed in the initial main comparison,
following the user's preference. Features may describe or distinguish different
executions of a whole bout. They do not cut that bout into temporal fragments or
join successive KPMS syllables into a new primitive; multi-syllable composition
belongs to unit4. An independent segmentation baseline has a separate vocabulary
and boundary provenance, rather than silently replacing the main syllables.

### 2.1 Frame information and bout creation are separate decisions

The explicit order is **what information each frame carries → how that information
is interpreted within the chosen primitive intervals → downstream episode and
organisation choices**. Feature-bundle selection belongs to unit1. It should not
be hidden inside a choice labelled "clustering method". The consumer separately
declares whether it uses native trajectories, whole-bout summaries or profiles.

For the main KPMS-anchored routes, keep one interpretation per original bout.
For example, bouts of syllable14 could receive execution subtype v0 or v1, while
each keeps its original start/end and parent syllable14. This partitions a
population of bout executions, not an individual bout's timeline. Changes within
a bout can still be measured as trajectories without becoming new primitives.

A feature-only whole-bout baseline can use the same KPMS intervals as a segmentation
scaffold, but cluster their physical profiles without using KPMS identities as
clustering evidence. This separates feature-derived type creation from changing
boundaries. A within-KPMS subtype model instead fits separately within each parent
syllable. A joint model consumes both categorical identity and physical features.
These are different unit2 mechanisms operating on the same unit1 information.

The existing independent centroid baseline clusters frame observations and uses
gap-aware RLE to construct its own bouts. Keep this as a distinct alternative
primitive representation. Its intervals can differ from KPMS; that is not a
KPMS execution-subtype result. Prospective change-point or jointly temporal
primitive mechanisms remain outside the main fixed-bout exploration and are
deferred choices, not assumed additions to it. Recognising a multi-syllable
routine or its phases remains unit4's responsibility.

Where a separate primitive mechanism has a downstream boundary/persistence policy,
changing that policy can reuse assignments where valid. A parameter intrinsic to
a jointly fitted model may require refitting or reinference. KPMS retains its own
pose/latent input requirements and is not assumed to accept an arbitrary feature
bundle. These dependencies remain explicit without merging unit1 and unit2.

All original feature sidecars may still accompany the output. The subset consumed
by primitive creation is distinct from features available only for later analysis.

**Frozen KPMS route.** The existing fitted KPMS artifact supplies native-aligned
primitive labels and their model-specific annotations. Unit2 imports/adapts those
labels and constructs the declared bout view; it does not refit KPMS on whichever
new feature bundle unit1 happens to select. Movement, posture and arena features
can still accompany these primitives and enter later models or analyses. Changing
a spatial feature bundle therefore need not change the existing KPMS syllables,
their human labels or their cached bouts.

**Feature-created primitive route.** Unit2 consumes the selected representation,
projected onto whole source bouts in the main comparison, to fit/assign new types.
It creates a new vocabulary and model identity; existing KPMS names are optional
reference descriptions, not inherited labels. A hybrid consuming KPMS labels as
evidence remains distinct from a feature-only model using KPMS solely for intervals.
The independent frame-clustering branch additionally changes the segmentation.

The feature stream is available to multiple consumers, not exhausted at unit2.
Record each consumer's active feature subset and projection. A frozen-KPMS recipe
which changes only features unused by every downstream consumer has not produced
a distinct modelling experiment; reuse its unchanged outputs rather than treating
duplicate fits as new evidence.

### 2.1a Feature-clustering and hybrid input routes for chains

Treat frozen KPMS evidence as both a valid primitive stream and, in explicitly
hybrid recipes, a categorical information source. The exploratory objective is to
learn whether adding/reorganising this evidence produces useful episodes, rather
than to require a new complete replacement for KPMS.

| Candidate route | Representation produced | Question for the episode layer |
| --- | --- | --- |
| Frozen KPMS only | Existing labelled primitive bouts | Baseline chains from the established vocabulary |
| Frozen KPMS plus direct features | Same primitives, with selected movement/posture/spatial evidence consumed by a higher model | Does information missing from the discrete label help chain boundaries, phases or variants? |
| Feature-only whole-bout clustering | New types from selected physical summaries/profiles on unchanged KPMS bout intervals; KPMS identity excluded from clustering evidence | Does a feature-derived vocabulary help chains when segmentation is held fixed? |
| Within-KPMS execution subtypes | One subtype per whole bout, fitted within its parent KPMS syllable; retain parent identity and intervals | Does execution variation within one syllable matter for forming chains? |
| Joint KPMS–feature whole-bout clustering | New types from categorical KPMS evidence plus appropriately encoded/weighted physical profiles on unchanged intervals | Does a hybrid grouping retain pose-derived information while distinguishing execution or spatial modes? |
| Parallel/twin whole-bout descriptors | Separate movement/posture and spatial codebooks, or continuous tracks, attached to each unchanged source bout | Does retaining separate views help chains without forcing every combination into one primitive label? |
| Coupled multi-view clustering | Prospective models linking the views rather than fitting them independently | Is agreement/shared structure useful, or does coupling erase complementary information? |
| Independent frame-feature clustering | Existing centroid route creates its own frame assignments and RLE bouts, with a separate source identity | How does changing segmentation as well as identity affect chains? Keep separate from the fixed-bout contrasts |

"Twin" is a provisional interpretation here: two parallel feature-derived views,
with the possible coupled variant distinguished separately. It is not a claim
that a particular twin-clustering algorithm already exists in the library.
Different camera/landmark views could also be declared where they provide actual
evidence; do not assume matching signals that the recordings do not contain.

KPMS IDs are categories, not numerical magnitudes. Joint clustering requires an
explicit categorical representation/distance and block weighting, not treating
syllable17 as numerically close to syllable18. Preserve the frozen KPMS identity
so human annotations remain available as reference descriptions. A new hybrid
type still has its own identity, support and potentially mixed description.

For main parallel views, attach each assignment/profile to the same source bout
and retain the KPMS parent. An episode model can consume both categorical tracks
or their supported continuous projections. A genuinely independent frame track
retains its own intervals and is a separately declared branch. Do not automatically
enumerate every movement×space×KPMS combination
as a new vocabulary; that can create many rarely observed types and extra human
labelling pressure. An explicit joint-code branch is possible, but it is a distinct
representation with its own cardinality and support tradeoff.

Whole-bout execution clustering already has groundwork: the shared
[`analysis/variants.py`](../../src/fpcode/behavior/analysis/variants.py) fits and
assigns numerical variants, and
[`workflows/behavior_extensions.py`](../../src/fpcode/workflows/behavior_extensions.py)
applies it separately within each syllable using duration, speed, turning,
peak-phase and latent-dispersion measurements. This existing analysis does not
yet make variant tokens an interchangeable primitive creator for the broad search.
New whole-bout hybrid/parallel encoders and downstream adapters remain prospective.
Cache profiles, fitted codebooks and bout assignments independently so higher-level
combinations can reuse them.

Earlier work documented the same general idea: local clustering of whole syllable
executions into rich tokens using duration, velocity, angular velocity,
acceleration and centre distance. See the
[`micro-fusion design`](../../studies/language_models/mds/01_theory_and_strategy/behavioral_strategy_modeling.md)
and dated
[`cluster-imbalance notes`](../../studies/language_models/mds/02_daily_updates/tuesday.md).
The latter reports outlier-dominated rare subtypes, directly relevant to the
sparsity concern here. These are historical design/results references, not proof
that a particular subtype scheme improves chains or an instruction to reuse their
cohort implementation unchanged. Keep new reusable functionality in its shared owner.

Direct unit1 → unit3/unit4/other-consumer routes are valid. The numbered units
describe responsibilities, not mandatory information compression at each step.
The recipe records the consumed channels and frozen KPMS source, making these
cross-layer dependencies explicit and preventing circular reuse of a model's own
new output as though it were independent upstream evidence.

### 2.1b Same routine, execution variant, or useful separate routine?

Do not assume that adding a feature warrants a new behaviour label. A head turn
while walking slowly in the centre and a head turn while running near an edge
could be separate routines, variants of one routine, or one routine performed
in different contexts. The exploratory question is what description the extra
information supports and what it costs in fragmentation.

Keep several routes open on the same source bouts and selected feature bundle:

| Route | Where the added information enters | Distinction it permits |
| --- | --- | --- |
| Descriptive passthrough | Measurements after detecting KPMS-based chains | One routine with different executions/context |
| Continuous episode evidence | Unit4 consumes KPMS identities and continuous profiles | Features affect recognition or execution modelling without requiring new primitive IDs |
| Whole-bout subtype tokens | Unit2 gives each execution a parent-preserving subtype; unit3 chooses its granularity | Chains can use categorical execution distinctions |
| Parallel descriptors | KPMS identities and separate execution/location evidence enter unit4 | Different aspects can contribute without a Cartesian token vocabulary |

A parent routine with feature-dependent variants is also a valid unit4 output.
The choice need not be forced into either one undifferentiated chain or many
unrelated chains. Preserve the parent/variant relationship where a detector
actually supplies it; do not fabricate that relationship for every method.

For matched comparisons, record both potential benefit and representational cost:

- Cross-animal recurrence, supported occurrences per type, held-out coverage and
  the number of rare or effectively unused subtype/chain IDs.
- Reproducibility of assignments and chain families across training samples and
  seeds; splitting that depends on isolated outliers is a distinct outcome.
- Many-to-many correspondence between chains on the same native spans and source
  constituents: how many new definitions represent similar source routines?
  Compare constituents and physical profiles as well as temporal overlap, since
  overlap alone does not establish that two routines have the same meaning.
- Better discrimination of recognisable executions, more coherent phases or
  boundaries, and information about future/withheld observations on a declared
  common target. Separating the supplied speed/location feature alone is circular
  evidence of utility.
- Qualitative review of matched ordinary and ambiguous occurrences: does the
  distinction describe a useful routine difference, or mainly rename similar
  behaviour according to position/intensity?

No universal threshold or winner is selected now. A feature can help describe
execution or recruitment while being unhelpful as a categorical primitive split.
Conversely, a reproducible subtype can preserve a distinction that a bout mean
or post-hoc summary misses. Keep both possibilities visible in the later atlas.

### 2.2 Primitive resolution without labelling every new vocabulary

Human annotation cannot be a mandatory prerequisite for every unit1 × unit2
combination. Separate **resolution construction** from **descriptive naming**.
An anonymous vocabulary can be merged, compared and passed to episode methods.
Names can accompany it later without changing its intervals or granularity.

The existing L1–L4 crosswalks belong to the particular KPMS vocabulary. They cannot
be applied directly to a newly fitted feature-clustering vocabulary just because
both use integer IDs.

| Resolution option | How it could work | Human-label requirement |
| --- | --- | --- |
| Identity/pass-through | Retain the primitive creator's vocabulary unchanged | None |
| Quantitative resolution | Merge primitive types using physical movement/posture/execution profiles or supported source-membership relationships; choose cut levels or thresholds | None; names remain optional |
| KPMS-reference-guided resolution | Relate new primitives to frozen KPMS states/categories and optionally group by sufficiently supported reference relationships | Reuse existing annotations; retain ambiguous/unmatched cases |
| Hybrid resolution | Quantitative structure with KPMS relationships as optional guidance or interpretive metadata; preserve supported subtypes within a shared reference category | Sparse human review of selected representatives, not every trial |

These are candidate alternatives. Automatic quantitative/reference-guided map
construction is not implied by the existing supplied-map remapping interface.
Finer execution distinctions use an explicitly fitted whole-bout subtype source,
rather than pretending a merge operation creates new evidence. Unit2 constructs
that source; unit3 controls how much of its distinction is retained. It does not
cut an existing KPMS bout into new temporal parts.

For KPMS reference descriptions, compare both vocabularies on the same supported
native intervals. A useful mapping retains the distribution of overlapping KPMS
states/categories for each new primitive, together with support and variation
across animals. Time overlap provides a common comparison even when model-specific
centroids or latent spaces are not directly comparable. Common feature profiles
can supplement that comparison where they share units and coordinate semantics.

Do not force a nearest named syllable onto every primitive. A new state may cover
several KPMS types, or several whole-bout subtypes may describe different
executions of one KPMS syllable. Mixed descriptions and anonymous IDs
are legitimate outcomes. Unassigned/Dump references are not reliable behavioural
names, and KPMS correspondence is a reference, not ground truth or a universal
quality criterion.

Reference **annotation alone** should not collapse primitives or alter their
bouts. Reference-guided **resolution** is a separate explicit branch which can
change them. In particular, merging all new states sharing a KPMS label could
erase exactly the distinctions the new creator discovered.

Quantitative merging criteria should be explicit: similarity in movement,
posture, execution or spatial profiles need not produce the same resolution.
**Sequence-role equivalence is outside unit3:** physically different primitives
are not merged merely because both precede rearing or occupy the same routine
position. Unit4 handles those alternatives through substitutions, insertions,
omissions, repeats or optional branches in a shared episode definition. A
transition-history diagnostic may describe a resolution, but it does not define
its type grouping in the main plan.

Keep original bout intervals and parent identities available after any type map.
Existing L1–L4 views may remap then RLE, coalescing adjacent source bouts; that is
an explicit coarser event view with a changed bout clock, not a learned routine.
A boundary-preserving mapped view can separate type-resolution effects from
this event-view effect. Neither view authorises arbitrary multi-syllable grouping
as new primitive creation; discovered temporal composition belongs to unit4.

Fit grouping/reference relationships on the training scope and freeze
them for held-out recordings. Balance contributions across animals and record
support so long stationary periods or one recording do not define the mapping.

Cache type profiles, reference-overlap distributions and resolution crosswalks
against the actual primitive-model identity, evidence, criterion and training
scope. Human-readable descriptions can be cached separately; a display-name change
should not invalidate an otherwise unchanged quantitative resolution. Human review
can concentrate on distinct, useful or uncertain representation families after
exploration. The choice among these approaches remains open at this planning stage.

### 2.3 Human labelling across later units

The broad search must not require new human action labels for every vocabulary,
episode catalogue, phase arrangement, grammar or context model.

| Output | Labelling policy |
| --- | --- |
| Existing frozen KPMS syllables | Reuse the current annotations/crosswalks for that exact fitted vocabulary |
| New primitives or quantitative resolution groups | Anonymous IDs, quantitative profiles, representative spans and optional mixed KPMS correspondence |
| New episodes/phases | Local episode/phase IDs, constituent descriptions and physical execution profiles; semantic names optional |
| Grammar/shared-subroutine structures | Describe child identities, order, alternatives and reuse; do not require a human name for every parent rule |
| Inferred context | State IDs with occupancy, recruitment and movement/spatial profiles; behavioural names may follow review |
| Cohort/condition readout | Use the recorded experimental labels, separately from human action annotation |

Annotations attach to model/vocabulary identity. Integer IDs from a new KPMS fit
or another primitive creator cannot inherit names solely because the numbers match.
Reference transfer retains mixed, uncertain and unassigned cases. An anonymous
result is preferable to a confident but unsupported ethological name.

Configured HSMM phases and caller-defined grammars can use a small reusable set
of assumptions/examples. An exploratory learned/discovery branch must be able to
propose phases/rules from training evidence without asking a person to label every
candidate. Automated semantic naming is not a substitute for implementing that
discovery operation.

Human review should concentrate on representative, distinct or ambiguous output
families, after the search has retained their provenance and native spans. It can
refine interpretation without becoming a prerequisite for computing every result.
Existing handling/exclusion review is a different kind of annotation and remains
part of the frozen observational support, not a parameter to vary per recipe.

## 3. Episode families and their structural choices

The largest decisions should be explored before assuming that small numerical
adjustments within one method settle the representation question.

| Family | Choices worth exploring | Current position |
| --- | --- | --- |
| Exact motifs | Motif lengths, catalogue limits, prevalence, frequency versus order/description-based ranking | Existing discovery and detection; ranking can favour short motifs even with a larger length ceiling |
| Tolerant motifs | Substitutions, insertions, omissions, repeated phases, edit costs, allowed timing variation | Existing alignment/progression groundwork; a general tolerant episode detector would need explicit construction |
| seqNMF | Bout versus physical-time input, feature blocks, factor count, template support, competition penalty, initialization, call threshold | Existing fitting/inference; consistent mixed-feature encoding is missing |
| Staged HMM | Phase proposals, categorical/mixed emissions, phase topology, early exits and reentry | Configured categorical inference exists; train-only learning and expanded topologies need work |
| HSMM | Phase count, phase proposal method, duration family/dispersion/support, background, exits, optional phases | Configured categorical inference exists; learned and feature-conditioned alternatives need work |
| Grammar episodes | Primitive alternatives, shared prefixes, optional parts, bounded repeats, rule weights | Finite parser/detector exists; automatic rule proposals/learning are missing |
| Explicit hybrids | seqNMF proposals → HSMM phase recognition; exact/tolerant proposals → HSMM; episode output → grammar assemblies | Must be real compositions with distinct provenance, not aliases for another method |

### Illustrative breadth, not chosen defaults

Possible starting ranges include motif/template support from roughly 2–12 bouts,
small through deliberately excessive catalogues/factor counts, weak through strong
regularization, and HSMM phase counts from a few phases to deliberately elaborate
models. Physical-time supports should have their own second-based ranges.

These ranges are examples for discussion. Not all are accepted by current APIs,
and they should not be encoded as available until the corresponding operation
exists. Wider ranges should include underspecified and overcomplicated cases,
rather than only a neighbourhood around the previous winners.

For HSMM, distinguish three provenance classes:

1. Configured recipes with stated assumptions.
2. Training-calibrated models with learned emissions/durations under a declared
   topology.
3. Models whose phase proposals/topology are also derived from training data.

The current configured HSMM is not a learned competitor to unsupervised seqNMF.
Comparing the configured branch is still useful, but its interpretation differs.
Independent detectors for a few target routines also differ from a competing
multi-routine model that partitions a recording.

Current HSMM durations count primitive observations. On bout input, three
observations do not mean three seconds. Explore duration coordinates explicitly;
do not relabel event durations as physical time. Optional/skipped phases and
arbitrary branching are also real functionality changes, not harmless flags on
the current staged topology.

Alternative physically distinct syllables can fill a common episode role through
declared substitution costs, alternative phase emissions or grammar branches.
This is episode variation in unit4, with the original primitive identities
retained. It does not require unit3 to merge the alternatives globally. An
insertion/omission/repetition also changes an occurrence's constituent path, not
the upstream vocabulary. Record what variation a method can actually recognise.

## 4. Features: availability, representation and influence are separate

Keep the original native feature sidecars available across downstream results.
Each model separately records which subset influenced fitting/detection. A model
change should not silently discard features needed for descriptive analyses.

| Feature role | Example | Consequence |
| --- | --- | --- |
| Descriptive passthrough | Summarise corner distance within syllables, episodes or phases | Unit definitions stay fixed; no detector refit is needed |
| Emission/detection evidence | Use syllables plus speed and edge distance to identify phases | The feature can split or merge behavioural units |
| Transition/duration conditioning | Let position affect routine recruitment or phase persistence | The model needs an explicit conditioned component |
| Assembly/context input | Infer longer organisation from episode identities and spatial/movement context | Lower units can remain frozen while their organisation changes |

An "edge-running routine" and a "running routine usually performed near an edge"
are different descriptions. Both are legitimate exploration outcomes. Greater
spatial separation is not automatically better behavioural modelling.

### Higher-layer models can consume unit1 information directly

At each applicable layer, the model input is **lower-unit evidence plus an
explicitly selected projection of the cached native feature stream**:

- Unit2 can use frame features to create primitives, or import frozen KPMS labels.
- Unit3 can optionally use supported feature profiles to construct quantitative
  resolution groups, independently of whether unit2 consumed those features.
- Unit4 can detect episodes from primitive identities alone, or from identities
  plus motion/posture/space/durations.
- Unit5 can describe fixed units using features, and prospective composition
  models can consume feature evidence where explicitly implemented.
- Unit6 can infer context from lower evidence plus time-aligned feature summaries.
- Unit7 can consume selected behavioural/feature summaries for its stated readout.

Higher models choose appropriate frame, bout, phase or window representations;
the original clocks, units and feature support remain traceable. A bout mean is
not interchangeable with a within-bout trajectory. Native features should not be
irreversibly reduced to the one summary selected by an earlier consumer.

Descriptive features and actual model evidence are different routes. The current
`FeatureAssociation` supports carrying and analysing features; mixed/conditioned
episode and context models still need the proposed consuming adapters/fitting.

### How to investigate whether a feature helps chains

For a later empirical exploration, construct matched comparisons with the same
training animals, native windows, primitive stream, resolution and episode family:

1. Primitive identities only, with the feature carried for description.
2. The same identities plus that feature as actual episode evidence.
3. Leave-one-feature-block-out variants and appropriately declared non-informative
   feature controls, where useful.

With frozen labelled KPMS primitives, this isolates downstream feature use without
requiring new KPMS fits or annotations. Compare it separately with changing the
primitive creator's feature inputs; those are different decisions.

Inspect changes in boundaries, catalogue splitting/merging, phase structure,
reproducibility, ambiguity, representative video coherence and held-out information
on an explicitly common target. A model conditioned on corner distance separating
corner versus centre behaviour is an expected representational consequence; it
does not independently demonstrate better explanation of that same distance.
Future or withheld observations can provide additional evidence where an honest
prediction/validation protocol is defined.

Report observation support alongside these changes. Adding a poorly observed
feature can change the available frames and apparent performance; distinguish
common-support comparisons from each model's full usable scope. Do not replace
missing values with zeros or silently invalidate otherwise supported frozen KPMS
labels because a descriptive sidecar is absent.

Feature utility is conditional on the method, resolution, route and intended
description. A channel can be useful for intra-chain interpretation or context
while adding little to episode detection. The result should state where it helped,
changed the question, or had no observable benefit. This is a planning comparison,
not an instruction to run experiments or code checks now.

### Feature bundles

- **Movement:** speed, angular velocity/turning, displacement and optionally
  acceleration.
- **Posture/body-part relationships:** supported pose geometry, head/body
  orientation and relative limb/body measurements; retain the distinction between
  a pose and its translational speed or arena location.
- **Spatial:** nearest edge, nearest corner, centre distance and optionally
  individual corner identity/distances or coarse spatial regions.
- **Execution profiles:** physical duration, within-bout movement curves and
  phase-normalised profiles, rather than only mean values.
- **Combined bundles:** syllables plus movement; syllables plus space; movement
  plus space; all blocks with deliberately varied relative weights.

Vary bundles and block weights deliberately. Several correlated distance
channels should not receive several times the influence of the syllables merely
because more columns were appended. Circular heading needs an appropriate
representation. seqNMF requires declared nonnegative encodings; standardizing a
signed channel is not enough to make it a valid input.

### Spatial branches grounded in Karen

Candidate choices include body midpoint, nose and supported ventral-paw-centre
references; screen rectangles versus frozen training-derived robust envelopes;
continuous distances versus coarse bins; and nearest-corner summaries versus
camera-near/far corner distinctions.

These are proxies, not calibrated walls. Preserve chamber, camera, epoch,
coordinate units and reference-point provenance. The shared central TopCam and
local BotCams have different geometry. Tandem21 does not supply an independent
second nose estimate; the paw-centre reference is supporting anatomical evidence,
not a reconstructed ground truth.

Geometry and normalization learned from observations must be fitted on the
training scope and frozen for held-out animals. Per-animal rescaling could erase
the very spatial or movement differences the representation is meant to retain.
Missing features must not become zero distances, compressed elapsed time or
bridges across excluded intervals.

Current feature associations support descriptive passthrough. Current episode
and context models do not automatically consume those associations. A spatial
conditioning trial must change the actual model inputs or equations, and record
that change.

## 5. Organisation/composition together, then context and optional readout

### 5.1 Unit 5 uses a union of applicable methods

The default philosophy is **look at everything supported on the same episode
evidence**. Do not introduce an exclusive choice of "classical inter/intra analysis
or grammar". The earlier separation between assembly and an analysis layer
described software responsibilities, not a requirement to choose one or to place
all measurements above context.

The combined workspace can produce:

- Inter-chain self transitions, repetitions, initiations, continuation, switching
  and return/loop measurements.
- Intra-chain execution, pacing, variability, constituent changes and completion
  where genuine evidence is available.
- Explicit repeat/alternation/shared-part composition and its candidate assembly
  intervals, constituent identities and alternative derivations.
- The same inter/intra measurement bank applied to those inferred assemblies.

For A→B→A, ordinary analysis can measure the return tendency and time gaps. Grammar
composition can ask whether it is a recurring larger structure whose A component
is reused elsewhere. Execution analyses can then describe A, B and the parent
occurrence together. These outputs answer complementary questions.

All applicable measurements can be run by default, reusing upstream results.
Grammar formulations, proposed rules, bounded repetition and alternative parses
still have genuine model choices. Explore those as identified composition branches,
alongside the common measurements. A union of methods does not mean concatenating
their overlapping events into one stream and counting the same evidence several
times. Keep original episode counts and each derived assembly view identifiable.

Missing phases/completion remain unavailable capabilities, rather than zero
measurements. Resource limits may bound expensive discovery/rendering, but a weak
result should not by itself eliminate a method from the exploration.

### 5.2 Connection to the supplied grammar/superroutine report

The supplied *Hierarchical Models of Sequential Behaviour* DOCX proposes shared
episode definitions, reusable subroutines, ordered composition, alternatives,
optional parts, repetition/exit structure and context-dependent recruitment.
It also discusses exact reuse/compression proposals such as SEQUITUR/Lexis,
restricted stochastic grammar induction and longer-term recursive program/library
ideas. Those are useful prospective branches for the combined workspace.

The current finite grammar implementation is a meaningful subset: caller-defined
rules, weighted alternatives, optional children, bounded repeats and timed child
references/derivations. It does not automatically discover grammars, implement
arbitrary recursion or condition recruitment on inferred context. The report's
shared-subroutine prototype with predefined stages is also not unrestricted
grammar learning.

Empirical status is limited: the earlier layer pilot exercised configured grammar
and assembly examples; the repaired overnight search and its full-recording
finalist analysis did not search learned superroutine/grammar alternatives. The
broad plan includes both the existing grammar component and the DOCX-inspired
extensions without claiming the latter are already implemented.

### 5.3 Unit 6: highest behavioural context

Context describes persistent regimes that organise recruitment, repetition,
branch choices or execution. It can consume episode/assembly evidence, time-aligned
organisation summaries and passthrough movement/spatial features. It remains
different from a longer ordered routine: a context can contain many structures.

Candidate choices include observed spatial/movement regimes, history baselines
and learned contexts over primitives or episodes. The current event context HMM
is configured and measures persistence in event steps. Learned context models,
feature conditioning or a physical-duration context HSMM need additional work.

A modular branch freezes lower definitions. An explicitly joint branch may revise
them using context. These have different dependencies and cache invalidation
requirements. The analysis bank can describe behaviour within/between inferred
contexts afterward; that does not create another mandatory behavioural layer.

### 5.4 Unit 7: optional cohort/genotype/condition readout

Above context, a supervised readout could predict known animal-group/genotype or
recording-condition labels from behavioural descriptions. For Karen the animal
group is PFF versus monomer, not a separate genetic grouping. A readout can use
context summaries, lower-level usage/execution/organisation or combinations; it
need not discard information by forcing every input through context.

This readout is a downstream inference task, not another discovered movement unit
or proof of a biological label. Distinguish animal-level group predictions from
recording-level condition predictions. Repeated sessions remain grouped by animal;
learned preprocessing/representations and any label-based selection belong inside
the relevant training scope.

Keep readout performance out of the broad representation-quality objective unless
a later task explicitly changes that objective. Otherwise the architecture search
would become a search for cohort discrimination. In Karen, the separate L_dopa
wave confounds drug, age and recording wave, so predicting its label would not by
itself identify a drug effect or recovery.

## 6. Independent choices need explicit interfaces and dependencies

Retain `UnitStream` and `BehaviorResult` as the shared interval/result contracts.
Keep feature associations, phases, completion, derivations and uncertainty as
honest optional capabilities. A downstream analysis requiring phase or completion
evidence must distinguish a missing capability from a measured zero.

The recipe should record:

- Which observations and transforms each model consumed.
- Which features were merely carried forward.
- Which lower outputs were frozen and which were jointly revised.
- The time coordinate, units, supported intervals and event-view policy.
- Configured, fitted or upstream-seeded model provenance.
- Capability requirements for each requested downstream operation.

Where possible, swapping episode methods should reuse observation features,
primitive labels and downstream analysis code. Adding a descriptive feature or
changing assembly composition should reuse the same fitted episode model.
Scientific independence will be partial: vocabulary, boundaries, duration
coordinates and feature support legitimately affect later choices.

## 7. Explore broadly and retain contrasts, rather than only winners

Use a mixture of broad random recipes and paired changes. Roughly half the effort
could cover diverse valid combinations, with the remainder split between matched
comparisons and deeper reruns of distinct outcomes. The allocation is provisional.

Give each method family meaningful coverage. A cheap method, a huge parameter
grid or an early high reconstruction score should not consume the entire search.
Deliberately cover interactions such as:

- Episode method × primitive resolution.
- Episode method × time/duration representation.
- Method × movement/spatial conditioning.
- Resolution × feature bundle.
- Episode method × assembly/context choice.

Keep surprising and low-scoring branches. Revisit representatives of different
output behaviours: stable, fragmented, spatially split, broad, selective,
ambiguous, empty or highly sensitive to pacing. Vary data and optimization budgets
for some representatives so that insufficient fitting is not mistaken for an
intrinsic limitation of the method.

Classify outcomes: unavailable capability, incompatible recipe, resource-limited,
numerical failure, completed with no detections, or completed with a usable result.
These convey different information. Never silently ignore an active knob, fall
back to another detector, or fill absent measurements with fabricated values.

## 8. Produce a representation atlas rather than one universal fitness

Keep a vector of observations:

- Time coverage, catalogue occupancy and occurrence/duration distributions.
- Overlap, ambiguity, fragmentation, unused units and boundary uncertainty.
- Stability across seeds, training animals and reasonable input perturbations.
- Which phases, omissions, repetitions, shared parts or context patterns appear.
- Feature profiles and changes in unit splitting/merging when a feature is routed.
- Model complexity, observation support, runtime and memory use.

Coverage is not itself an objective to maximize: one unit spanning a recording
would cover everything. Stable broad states and precise sparse routines may both
be useful descriptions.

Keep method-specific diagnostics separate: seqNMF reconstruction/order controls,
HSMM likelihood/duration calibration/phase uncertainty, and grammar coverage or
description/compression measures where implemented. Do not combine incomparable
objectives into a supposedly universal quality score.

A common prediction comparison would require an explicitly identical observable
target and prediction protocol, for example future KPMS labels or withheld
movement channels. Retrospective reconstruction is not future prediction.
Predicting a feature already supplied to a detector is also not independent
evidence that the representation explains it.

Numerical summaries should be paired with qualitative comparisons of the same
recording spans. Tempo changes, optional/omitted/repeated phases, shared prefixes,
spatial displacement, genuine cycles and censored endings are useful conceptual
diagnostics for distinguishing method behaviours. This document does not specify
or request their implementation as tests.

For each sampled configuration, retain a concise record of the choices,
capabilities, outcome vector and matched alternatives. Render representative
native-video spans for typical, ambiguous and unusual cases across branches;
include unmatched/background periods. Rendering every trial is unnecessary.
Retain native spans so additional examples can be generated later.

The resulting report should answer "what happens when we change this choice?",
show distinct representation families and explain interesting failures. It should
not be a top-five list dominated by one detector's internal score.

## 9. Functionality that could make this possible

These are potential work packages, not tasks being executed now:

| Capability | Why it matters | Existing owner to extend |
| --- | --- | --- |
| Shared native-feature projection/encoding | Consistent event/time inputs, masks, training-fitted transforms and explicit feature weights | `behavior/observations`, contracts and workflow adapters |
| HSMM/HMM learning and phase initialization | Meaningful configured versus learned method comparisons | `behavior/episodes/staged.py` and generic statistical inference |
| Continuous/mixed emissions and conditioned transitions/durations | Real movement/spatial conditioning rather than decorative sidecars | Episode/context models and their statistical components |
| Mixed-input seqNMF adapter | Correct nonnegative feature encodings and channel-block weighting | Existing seqNMF/workflow owners |
| Motif/grammar/assembly proposal generation | Explore data-derived shared parts and larger structures | Existing episode/assembly owners and composition workflows |
| Learned context alternatives | Compare context inferred from different unit streams/evidence | Existing context owner |
| Recipe/capability orchestration | Compose compatible alternatives, retain provenance and classify unavailable branches | Existing behaviour workflows and contracts where needed |
| Dependency-aware fitting/result reuse | Explore downstream choices without repeatedly fitting invariant upstream models | Existing workflows and generic cache/runner infrastructure |
| Qualitative comparison reports | Inspect multiple methods on identical spans and retain diverse outcomes | Existing visualization and study reporting owners |

Start by making several major branches executable, with simple explicit versions,
rather than spending the entire budget refining seqNMF. A broad configured-method
pilot can coexist with building train-fitted alternatives. Configured, learned
and seeded branches must remain distinguishable throughout.

## 10. Keep structure and reuse available resources

Reusable implementations remain in `src/fpcode`, following the existing owners.
Karen splits, proxy choices and study interpretation remain in
`studies/karen_pff_vs_monomer`. Operational launching/monitoring stays in `ops`.
Inputs/models and generated products remain in their selected ignored storage.
Do not add a second behavioural framework, a new package tree, one folder per
trial family or generic mechanisms hidden inside a cohort runner.

Reuse the persistent workers, parent watchdogs, transactional trial records and
resource safeguards from the existing search. A broader dependency cache could
reuse observations → feature transforms → resolution/time representations → fitted
episodes → assemblies/context/profiles. Cache identities include observations,
masks, clocks, training split, transforms, parameters, source version and seed.
Durable fitted artifacts permit later downstream work without refitting.

### 10.1 Cache unit 1/unit 2 combinations and their reusable components

Random recipes will repeatedly choose the same frame representation and primitive
mechanism while changing resolution, episodes, assemblies or feature descriptions.
That combination should be reusable. Cache its components as well, so changing
the creator reuses frame features and changing a postprocessing policy can reuse
model assignments where valid.

| Cached component | Identity depends on | Work it can avoid |
| --- | --- | --- |
| Native feature channels | Recording/input, clock, extractor settings, support masks and any geometry/calibration reference | Repeated pose loading and feature extraction |
| Selected frame representation | Consumed channel identities, selection, temporal context, transform/weight settings and fitted training references | Rebuilding/scaling the same model inputs |
| Fitted primitive model | Consumed representation, creator, model parameters, training animals/sample selection, seed and fitting budget | Refitting the same feature–creator combination |
| Frame assignments or state evidence | Frozen fitted model, recording representation and inference settings | Reassigning frames for each downstream trial |
| Bout stream | State evidence plus boundary/RLE/persistence policy and native support | Reconstructing identical intervals |
| Whole-bout summaries/profiles and subtype evidence | Frozen source bout identity/intervals, consumed channels and support, summary/profile settings, training-fitted subtype model and assignment policy | Recomputing execution descriptors or reclustering the same whole bouts |
| Higher-level results | Exact lower artifact dependencies and episode/assembly/context/analysis choices | Repeating unaffected downstream operations |

For example, motion features → centroid clustering with 16 states could be fitted
once per training scope/seed. Its assignments and bouts can then feed exact,
seqNMF, HSMM and assembly branches. The same motion representation can also feed
another primitive creator without extracting motion again. A new independent
seed remains a new fit, rather than an apparent replicate served from cache.

For the main frozen-KPMS routes, source bouts are cached independently of any
execution subtype. Reuse the same whole-bout profiles across different cluster
mechanisms/counts, and reuse a fitted subtype model across resolution/episode
choices. A changed subtype identity does not imply new temporal boundaries.
Continuous features passed directly to unit4 need no subtype fit at all.

The pair `(features chosen, creator chosen)` is not a sufficient cache key by
itself. Include the actual parameters, fitted transforms/calibration, training
split, observation support, native coordinate/time identity, source version and
numerical settings that influence the result. Raw geometry or normalization
derived from training observations cannot be shared across folds as though it
were a fixed input.

Key each fitted component by what it actually consumed. Adding a descriptive
corner-distance sidecar should not invalidate a primitive model trained only on
motion, while supplying that distance to the model must create a different input
identity and fit. Do not invalidate the whole pipeline simply because an unrelated
reporting feature was added.

Use bounded RAM caches for hot inputs and durable disk artifacts for expensive
representations, fits and reusable assignments. Frame evidence can be large;
chunking, quotas and explicit eviction should prevent caching every dense payload
indefinitely. Concurrent workers requesting the same component should share one
completed artifact, with coordinated creation and atomic publication. An incomplete
artifact must not be returned as a usable result after a crash.

The scheduler can group trials sharing upstream components to improve reuse while
still preserving balanced exploration across method families. Record cache hits,
misses and avoided fits separately from genuinely new model evaluations. This is
a proposed cache design, not an implemented extension of the previous RAM-only
search cache.

Repeated workloads should use invariant-input reuse, vectorization, batching and
bounded parallelism where they materially help. Keep RAM/disk budgets and avoid
nested thread oversubscription. GPU paths are a possible later option for suitable
numerical workloads; they are not assumed to help interval parsing or small matrix
operations. Record fitting budget and convergence evidence instead of declaring
every method equally trained after the same arbitrary iteration count.

Every comparison report must state the conditions each method was given to become
useful: initialization/proposal source, training animals and sampled observations,
feature encoding, parameter ranges, number of configurations/restarts/seeds,
iteration/time/work budgets, convergence, manual tuning and development attention.
Distinguish configured, seeded and trained parameters, including which parameters
remain fixed. Unequal development effort limits an apparent method comparison;
make that visible rather than describing all methods as equally optimized.

The planning decision still to make is how much breadth to open first: a wide
configured-method and passthrough atlas, learned episode alternatives, actual
feature conditioning, and assembly/context discovery. Those can be developed in
parallel within the established structure. None is authorized for execution by
this planning document alone.
