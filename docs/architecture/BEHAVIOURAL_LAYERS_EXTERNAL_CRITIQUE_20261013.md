# Behavioural episode discovery: brief for an external scientific critique

Date: 13 October 2026. This document is self-contained so it can be pasted into
another chatbot. We want criticism and methodological ideas, including explicit
uncertainty. We are not asking for code or a confident endorsement of our framework.

## The scientific aim

We have mouse behavioural recordings with existing keypoint-MoSeq (KPMS) state
assignments, pose-derived movement/posture features, and some spatial information.
We want useful recurring **episodes / chains**: a catalogue that distinguishes
different behaviours while grouping very similar executions despite modest
substitutions, repetitions, interruptions or changes of pace.

Exact frequency-ranked n-grams are a baseline. They can split one routine into
many near-identical strings and emphasise common short background loops. We are
considering tolerant matching, seqNMF, staged HSMMs and finite grammars as alternative
ways of representing these patterns. None is assumed to solve the problem simply
because it supports variation. Homogeneous grooming routines can also be useful;
“more different constituents” is not automatically better.

The primary aim is representation quality, not selecting a method that maximizes
disease, genotype, drug or dopamine differences. Such comparisons can follow after
we freeze behavioural choices. We intend quantitative exploration first, then
expensive video review of a diverse shortlist. There is no universal agreed quality
score and no comprehensive semantic ground truth.

## Data and constraints

Two datasets are available, each with its own fitted KPMS model and vocabulary:

- Karen: PFF / monomer cohort; 99 recordings from 21 animals in the reviewed input
  view. Existing animal splits have 12 train, 4 validation and 5 reserved test mice.
- Jonathan: paired saline / amphetamine recordings; 30 recordings, 15 animals
  (8 WT, 7 KI). The user describes KI as the ADHD / late-stage Parkinson model;
  historical documentation uses DAT-Het terminology. This interpretation is kept
  separate from measured outcomes. Both groups are used for behavioural exploration.

We reuse the existing KPMS fits; additional KPMS fitting is outside this search.
Identical numerical syllable IDs across datasets have no shared meaning. Some
human annotations exist, but they belong to a specific model and cannot label a
new feature-derived vocabulary automatically. Jonathan groom IDs58/60/61/63/65/66
have been verified as AMPH Run12, not the initially suspected IFDN model.

Dopamine is available only for Jonathan and only WT DA is eligible as optional
later corroboration. Signals already have substantial processing and multiple
suffix variants. DA is not independent semantic truth: motion, drug, time, region,
signal processing and behaviour can all affect associations. Missing DA should
not remove otherwise eligible behavioural evidence.

Recording clocks and model sampling differ. Jonathan has sparse native-frame
model support, a 40 Hz model, 60 FPS encoded video and acquisition around80Hz;
these are distinct coordinates. Verified support projection excludes unsupported
filled gaps. All methods must preserve masks, native frame identity, gaps, repeated
recordings and animal-level splits. This is an actual constraint on the estimand,
not just engineering housekeeping.

We have explored some data previously, including a Karen exact/seqNMF sweep.
This means the evaluation is not perfectly untouched; we will disclose that
limitation. The frozen KPMS fits themselves may also have seen animals later held
out from our chain fitting. Holding out chain models does not undo that exposure.

## Our layer layout

The numbering describes responsibilities, not an obligation to pass through every
stage or discard lower information. Features remain available to later consumers.
Context can be a parallel track. Measurement applies across levels.

| Layer | Responsibility | Options and open choices |
| --- | --- | --- |
| 1. Frame representation | Preserve time-aligned evidence | Movement, pose/posture, latent features, heading, spatial coordinates; normalization, derivatives, temporal summaries and missingness |
| 2. Primitive / bout interpretation | Assign an identity or execution description to a local unit | Frozen KPMS bouts; feature-only whole-bout clusters; subtypes within a KPMS parent; joint categorical+feature clusters; independent parallel execution/spatial views |
| 3. Primitive resolution | Decide which local types count as similar | Original vocabulary; model-specific human annotation merges; training-derived quantitative merges; prospective reference-guided / hybrid grouping |
| 4. Episodes / chains | Detect recurring temporally structured routines | Exact motifs, tolerant/edit motifs, seqNMF, staged HSMM, finite optional/repeat grammars; possible future hybrids |
| 5. Organisation / composition | Describe how the same episodes are organized and reused | Within/between-unit measurements, transitions, repeats, returns, shared parts and assemblies; prospective shared-subroutine or richer grammar discovery |
| 6. Context | Describe persistent regimes and recruitment conditions | Spatial / movement context; inferred primitive/episode/assembly regimes; context-conditioned recruitment or execution; optional bypass |
| 7. Optional external readout | Ask what fixed behavioural descriptions tell us about known labels | Later genotype/cohort/condition predictions or associations; kept outside representation selection |

Layer3 resolution is not a series of four temporal levels: historical L1–L4
annotation vocabularies are alternative views of primitive identities. Naming
units and constructing them are separate choices. Human labels should not be
mandatory for every candidate representation.

In the first search, layer2 changes whole-bout identities while preserving original
KPMS bout boundaries. Temporal subdivision and independently segmented frame
features are separate prospective alternatives. A multi-syllable sequence-role
substitution is a layer4 issue, not a local movement subtype by definition.

## Feature routes we want to compare

1. **Descriptive:** keep the detector categorical; measure movement/posture/space
   afterwards. These features cannot explain an improvement in its detection.
2. **Direct episode input:** categorical events plus actual continuous features
   contribute to detection or emission likelihood.
3. **Feature-only whole-bout vocabulary:** cluster execution summaries regardless
   of the KPMS parent. The timing still comes from the frozen source bouts.
4. **Within-parent subtypes:** distinguish execution variants inside each original
   KPMS category, keeping their parent identity visible.
5. **Joint categorical+feature vocabulary:** an explicit weighted categorical
   representation plus continuous summaries. Numerical KPMS IDs are not distances.
6. **Parallel views:** keep base categories and separate execution / spatial
   codebooks. We want to avoid an automatic Cartesian vocabulary explosion.

Feature bundles include duration, movement, posture/heading/latents, spatial
coordinates and combinations. Transformations and codebooks fit on training
animals only. Pixel position/speed are image-coordinate proxies; latent coordinates
are model-specific. Geometry normalization and the biological meaning of spatial
differences remain unresolved.

## What our first implementations actually do

- Exact and tolerant branches propose motifs from animal-balanced training
  frequency, with some categorical composition diversity. This proposal stage can
  bias what later detectors are capable of finding.
- seqNMF consumes nonnegative categorical channels and, when selected, signed-split
  standardized features / separate parallel view blocks. It uses a fixed factor
  budget, length, competition penalty and seeded initialization. Its errors and
  activations are method-specific, not directly comparable to HSMM likelihoods.
- HSMMs currently have seeded routine phases, categorical and optional diagonal
  Gaussian emissions, event-count duration distributions and configurable entry,
  exit, skip and reentry. Emissions/transitions, Gaussian parameters and optional
  interior duration distributions can be reestimated. Initial distribution and
  topology remain fixed. Multiple targets are fitted independently, not as one
  jointly learned catalogue. Hard EM is bounded and can reach local optima.
- Finite grammars currently allow declared bounded optional/repeated components
  from proposed motifs. They are not unrestricted grammar induction or recursion.
- Continuous profile gates in exact/tolerant/grammar branches are training-calibrated
  execution restrictions, rather than a learned generative feature model.
- Parallel-view codebook evidence can reach detectors, but the present HSMM
  auxiliary-channel approximation is Gaussian; this is not a full coupled
  multi-view categorical model.

These methods have received unequal development attention. We will report the
conditions each was given to become useful, including proposals, tuning budget,
feature access, seeds, initialization, fixed structure and stopping rules. We
cannot interpret an underdeveloped branch losing as proof that its model family
is intrinsically worse.

## The difficult unresolved questions

### 1. What is the estimand for a “good” episode catalogue?

How should we distinguish compactness, interpretable diversity, recurrence and
cross-animal usefulness from trivial compression? A single broad type can have
high support/coverage while conflating actions. Many narrow types can have excellent
within-type similarity while merely memorizing strings. Could a principled
description-length / rate-distortion / predictive formulation help without
replacing biological usefulness with an easy mathematical surrogate?

We want a diagnostic vector rather than arbitrary weights in one scalar score.
What small set of diagnostics exposes the important tradeoffs, and how should a
diverse shortlist be selected when the Pareto frontier itself is very large?

### 2. How do we assess coherence without defining coherence into the input?

If speed/pose/position helped create a type, low variance of those same features
is partly built in. Sequence edit similarity can also reward exact templates by
construction. Which held-out channels, perturbations, predictive tasks or
independent annotations can add evidence without just repeating the objective?
What can genuinely be established before qualitative video review?

### 3. Should variants be separate routines or attributes of one routine?

Turning faster, being near a wall, or using one extra sniff may be execution
variants of the same routine, or meaningful behavioural distinctions. How can
we factor identity, execution and context without deciding the answer by feature
weight? What observable evidence would favour a subtype, a phase substitution,
a context-conditioned routine, or a separate episode?

### 4. Is order in bout space the right temporal model?

Event-index seqNMF/HSMM can handle some variable physical tempo, but an event lasts
different amounts of time and has already been segmented by KPMS. Physical-time
bins introduce their own alignment / stretching choices. Which changes are genuine
routine invariance and which discard the pace we wanted to measure? How can we
isolate representation effects from effects of detector flexibility and duration?

### 5. How much do frozen KPMS boundaries constrain the answer?

A feature-only vocabulary on fixed KPMS bouts is not independent segmentation.
It can never recover an episode constituent whose boundaries are absent or
misplaced in that segmentation. Is this a reasonable pragmatic first experiment,
and what is the cheapest informative falsification of that assumption? What can
animal-held-out chain evaluation say when the underlying KPMS fit was not held out?

### 6. Which null hypothesis is informative for each method?

Frequency-preserving shuffle destroys first-order transitions and the RLE
no-self-transition property. A fitted first-order surrogate addresses a different
hypothesis and does not preserve exact counts or transition-estimation uncertainty.
After annotation merges, adjacent equal labels may be legitimate fixed source
bouts. Mixed features introduce category-feature associations and duration clocks.
What should be preserved to ask whether there is useful higher-order organization?
How should we avoid calling every rejection of a weak null an interesting routine?

### 7. Can two different vocabularies verify the same method claim?

We can compare tradeoffs within each dataset, but the same hyperparameters imply
different effective complexities under different primitive vocabularies and
recording geometry. How do we assess robustness across cohorts without comparing
raw IDs or declaring they have the same biological routines? Is cross-model
alignment useful now, or does it introduce another inference problem prematurely?

### 8. What constitutes a fair method comparison under finite effort?

Equal wall time, equal fits, equal parameter dimension and equal researcher
attention are different budgets. Should we compare common input routes, each
method's best reasonable route, and effort-versus-quality curves separately?
How do we disclose seeded proposals that make some families almost inherit the
baseline? What failed conditions are evidence about implementation rather than
the scientific model family?

### 9. How do we select and validate after a broad adaptive search?

Thousands of recipes, folds and seeds produce dependent exploratory evidence.
Some metrics may be improved by selective thresholding, overlap resolution or
ignoring difficult missing data. What minimal selection/confirmation split should
we preserve? What should be frozen before examining phenotype or WT DA? How do
we report previous exposure and uncertainty without pretending exploration was
preregistered or discarding useful data?

### 10. What should qualitative review actually decide?

Human review is expensive and can be influenced by candidate selection and names.
Which matched clips and failure cases provide the most information per minute?
Could a small blinded paired-review rubric check both “same routine despite
variation” and “different routine that should stay separate”? How much reviewer
agreement is enough to improve decisions without pretending to create definitive
ground truth? How should rare meaningful routines survive frequency-based screening?

### 11. Where do hierarchy and context improve the target, and where do they hide it?

Shared subroutines may support a compositional representation, but apparent shared
parts can also be background transitions or overlap artifacts. Context models can
explain recruitment but might mostly track location, injection phase or individual
animal. When should higher structure be inferred jointly versus analysed after
the catalogue is frozen? What would count as evidence for hierarchical structure
rather than another level of clustering?

### 12. What can optional WT dopamine tell us?

DA may be aligned to behaviour and useful as another modality, but it is only
available in one cohort and has processed variants. What later analysis adds
information while controlling for movement, session time, drug, repeated animals
and signal-processing choices? Which positive or negative result would actually
change confidence in a behavioural representation?

## What we would like from you

Please identify our most consequential hidden assumptions, contradictions or
confounds. Prioritize a few concrete recommendations that fit a finite compute
budget and quantitative-first workflow. Distinguish cheap checks, high-value
experiments and ideas that would require changing the scientific target.

For each recommendation, explain what observation would support or falsify it,
and what the proposed metric or experiment still cannot establish. Discuss
alternatives where the answer depends on how we define behaviour. Tell us which
questions remain genuinely unanswerable from our available evidence.

If you mention methodological literature, provide verifiable primary references
and distinguish established results from analogy or speculation. Do not invent
citations, claim you inspected our data, or assume every listed option is already
implemented. A strong disagreement with our framing is welcome if you explain it.
