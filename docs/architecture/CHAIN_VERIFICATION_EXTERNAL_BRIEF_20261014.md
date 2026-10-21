# Behavioural chains: data, results and questions for external critique

Date: 14 October 2026. This brief is self-contained: the receiving chatbot has
**no access to our code, files, reports or recordings**. Please reason from the
facts below, distinguish assumptions from evidence, and suggest scientific
measurements rather than implementation changes. The random search is stopped.

## The actual downstream task

We want to identify **behavioural patterns across different conditions and
genotypes**, using:

1. **Chains themselves:** recurring episodes, their recruitment and execution.
2. **Intra-chain modifications:** changes in pacing, repetitions, substitutions,
   omissions, supported progression, and execution variants within a common routine.
3. **Inter-chain combinations:** repetitions, alternations, switches, returns,
   shared subroutines and larger assemblies of routines.

The desired catalogue distinguishes different behaviours while grouping very
similar executions. A useful chain family can preserve identity while its execution
or surrounding sequence changes with condition. The same framework should expose
these changes rather than hide them inside one average or create a new unrelated
identity for every minor variant.

Good clustering is a foundation, not the entire downstream objective. We want to
know which chains, modifications and combinations reveal interpretable phenotype
or condition tendencies. Condition-related effects can help prioritise candidates
during exploration; we must distinguish that discovery from subsequent verification.
We do not demand a significant group effect or a dopamine association from every
otherwise useful chain. Homogeneous routines such as grooming are legitimate;
more distinct constituent syllables are not automatically better.

## What data we actually have

| Dataset | Animals and recordings | Conditions and repeated observations | Current analysis scope |
| --- | --- | --- | --- |
| Karen | 21 mice, 99 reviewed recordings; 12 PFF and 9 monomer controls | Untreated post-injection waves at 3, 8, 16 and 24 weeks, plus a separate L-DOPA wave; individual availability varies | Search used 16 exploration mice/73 recordings, with three animal folds. Five reserved test mice were excluded. |
| Jonathan AMPH Run12 | 15 mice, 30 recordings; 8 WT and 7 KI | Each mouse recorded under saline and amphetamine, with recorded injection timing and pre/post-injection epochs | All 15 mice/30 recordings used in three animal folds. WT and KI contribute behavioural evidence. |

Karen PFF/monomer is a treatment/cohort contrast, not a genotype contrast. Its
L-DOPA wave also differs in age and recording wave, so a pure drug interpretation
is not established. Jonathan saline and amphetamine recordings occurred on
different dates in a fixed order; paired drug contrasts can include session/order
effects. Pre/post injection and injected drug are separate fields. The user
describes KI mice as an ADHD/late-stage Parkinson model; historical documentation
uses DAT-Het terminology. This is supplied model context, not a measured stage
assignment for each recording.

**Behavioural evidence available now:**

- Existing, frozen keypoint-MoSeq (KPMS) categorical labels and gap-aware native
  bout intervals. Each cohort has its own fitted model/vocabulary; numeric IDs
  have no shared meaning between models.
- Prepared features for complete original bouts: duration in seconds, mean/net
  speed in pixels/second, path length in pixels, centroid coordinates, signed and
  absolute heading change in radians, ten source-model latent dimensions and
  latent dispersion. These are movement/posture proxies, not calibrated joint
  angles, millimetres, height, force or verified paw contacts.
- Verified frame-level centroid/heading/latent exports underlying those summaries,
  with recorded missingness/support. Source videos and tracking artifacts exist
  for later targeted review. The recent physical assessment used bout summaries,
  not new frame-by-frame pose analysis or video judgements.
- Model-specific human syllable annotations with incomplete coverage. Jonathan's
  grooming IDs58/60/61/63/65/66 were bound to the actual AMPH Run12 model; they are
  not an interchangeable annotation for IFDN or another KPMS vocabulary.
- Animal/session identities, cohort/genotype, recording condition, injection
  timing where available, and reviewed exclusion/support metadata.

Original bouts and support gaps remain intact. Karen's native video clock is60Hz.
Jonathan's model uses40Hz, encoded video playback60FPS, and the declared native
acquisition policy80.025Hz: these are different clocks. Native frame identities
are verified; gaps are barriers and sampling ratios are not guessed from lengths.
Photometry association must use actual timestamps and verify alignment/drift.

**Photometry:** Jonathan's selected sessions already contain processed signals,
with raw, `_dff`, `_zscore` and `_zscore_datta` variants and signal metadata for
physical fibre/colour, hemisphere and region (VS/DMS where present). We have a
WT-only inventory covering8mice/16recordings. Region-specific eligible N depends
on signal coverage and implants. Only **WT DA** is eligible for the proposed
verification; no KI DA or WT-versus-KI DA comparison is planned. There is no DA
modality available for Karen in this analysis. Multiple processing suffixes are
views of one underlying signal, not independent replicates. Existing processing
includes centered rolling transformations, so apparent temporal precedence does
not establish causality. We intend to reuse processing and documented masks,
select trace identity/units from provenance, and avoid choosing the suffix with
the strongest association.

## Current framework and what has actually been executed

The layout preserves frames/features → whole-bout identity → optional primitive
resolution → episodes/chains → organisation/composition → context → downstream
condition/genotype readouts. Measurements can consume lower-level evidence;
intra/inter analysis and composition are complementary.

The search compared exact motifs, tolerant edit motifs, seqNMF, independently
refined staged HSMM targets, and finite optional/repetition grammars. Inputs varied:
categorical KPMS alone, actual categorical-plus-feature detector inputs,
feature-only whole-bout clusters, subtypes within KPMS parents, joint codebooks,
and parallel categorical/execution/spatial views. Whole-bout boundaries stayed
fixed; no new KPMS fitting or independent segmentation was performed.

seqNMF explored K4/8/12, template lengths3/5/8 bouts, regularisation and detection
thresholds, with40/80/120 updates. HSMM explored1–3 independently seeded targets,
phase/background/duration/emission choices and8/16/32/64 hard-EM budgets. It is
target refinement with fixed topology, not unrestricted joint catalogue discovery.
Finite grammar supports declared rules, not arbitrary recursive grammar induction.
Development effort and optimisation conditions differ; compare strong individual
configurations and their supplied conditions, not method-family averages.

The stopped overnight run retained22,338 successful and2,569 empty evaluations,
plus4 interrupted and49 unstarted jobs. The first finite tranche had1,440 completed
evaluations. Final analysis retained6,592 complete cohort/configuration evaluations;
48 candidate configurations received additional physical/fragmentation assessment
(195 fold/seed evaluations,13,093 selected calls). Nine complementary candidates
form a provisional smaller shortlist.

**Crucial scope:** fits used600 training bouts/animal and evaluation windows used
180 held bouts/animal. Current coverage and organisation evidence concern these
sampled windows, not complete sessions. No new group-effect or DA association was
calculated for these shortlisted chains. No qualitative chain review was performed.
General intra/inter measurement machinery exists from earlier work, but has not
yet been applied comprehensively to these new catalogues. Fold-specific chain IDs
are not automatically shared identities; matching or a fixed reference catalogue
is needed before pooling specific chains.

## Verification measures already calculated

| Measure | What it assesses | Important limitation |
| --- | --- | --- |
| Sequence agreement and separation | Original KPMS-string similarity within a type versus other types | Exact matching scores1 by construction; longer/variable strings may score lower without being worse routines. |
| Coverage and support | Selected native-time exposure, calls and animal representation | Sampled-window coverage; high coverage can reward permissive detection. |
| Seed F1 | Types matched across two random starts by native-time overlap, then matched-duration F1 | Stability, not correctness; deterministic fits have no alternate-seed evidence. Two seeds do not establish general stability. |
| Physical separation | Three native-time phases of duration-weighted bout features; same-type versus other-type distances across animals | Coarse profiles; features consumed by construction partly reward the fitting objective. |
| Unused-channel consistency | The same physical comparison on channels not used to construct the detector/codebook | Additional evidence, not fully independent ground truth. |
| Source-conditioned physical separation | Subtract training-only per-KPMS-state median features, then compare execution profiles | Partial composition adjustment, not a full matched sequence/context null. |
| Fragmentation/concentration | Singleton and single-animal types; types with<4 calls or<2 animals; inverse-concentration effective type count | Thresholds are exploratory; rare condition-specific routines may be informative. |
| Repetition/redundancy | Adjacent same-type calls, raw overlap rejection, duration variability, distinct constituents and exact periodicity | Many KPMS states may describe the same action; literal repetition does not exhaust behavioural micro-loops. |
| Shuffle/first-order controls | Frozen detector applied to shuffled category/feature pairs or training-derived first-order labels on fixed event slots | Two draws/job, no detector refit; controls do not preserve every duration/context association and are not p-values. |

Physical scaling is training-only median/IQR, clipped at±8. Comparisons require
two calls per contributing animal and two animals per own type. The reported
separation is `1 − within-type squared distance / other-type squared distance`;
positive values indicate relative distinction, not an absolute quality threshold.
Missing evidence stays missing. Fold/seed summaries refer to one configuration.

Representative results:

| Cohort / candidate | Sequence agreement | Coverage | Seed F1 | Physical separation | Source-conditioned separation | Fragile types |
| --- | --- | --- | --- | --- | --- | --- |
| Karen tolerant2887 | .58 | 9.2% | unavailable | .42 | .32 | 17% |
| Karen seqNMF2967 | .38 | 25.0% | .50 | .16 | .02 | 0% |
| Karen seqNMF3025 | .37 | 53.3% | .29 | .17 | .12 | 3% |
| Karen HSMM1792 | .31 | 16.0% | unavailable | .23 | .20 | 11% |
| Jonathan seqNMF919 | .39 | 26.8% | .74 | .50 | .48 | 8% |
| Jonathan seqNMF2729 | .58 | 27.8% | .45 | .21 | .12 | 15% |
| Jonathan seqNMF2368 | .49 | 45.0% | .57 | .15 | .05 | 4% |
| Jonathan HSMM1792 | .44 | 17.5% | unavailable | .12 | .03 | 0% |

Jonathan919 has four observed types, uses KPMS+duration inputs, and its conditioned
physical effect persists across folds(.46/.49/.48). Unused movement/posture
separation is~.39. First-order event-detection excess is+2.4percentage points, not
a calibrated significance result. Other strong candidates often do not exceed
first-order detection. Karen2887's conditioned effect is dominated by one fold
(.78/.13/.04). HSMM1792 converged8/9 Jonathan targets but only2/9 Karen targets.
A Jonathan tolerant coherence leader scored.69 sequence agreement but had~68%
fragile types and essentially zero physical separation. Thus no single metric is
sufficient, and detected type counts are not verified distinct behavioural themes.

## Downstream measurements proposed, not yet run on this shortlist

| Domain | General across the catalogue | Chain-/relationship-specific |
| --- | --- | --- |
| Recruitment | Initiation rate, occupied time, diversity/concentration | Frequency, duration and exposure-normalised use of A |
| Intra-chain execution | Pacing and variability; variant prevalence | Stage durations, speed/turning profiles, substitutions/repetitions in A |
| Progression | Continuation/interruption where observable | Stage-specific continuation and eligible exit destinations; not fabricated failed attempts from fixed-length calls |
| Inter-chain organisation | Repetition, switching, run lengths, returns | A→A,A→B,A→B→A, gaps and combinations/assemblies |
| Context/shared parts | Context-dependent recruitment/execution | Same constituent inside A versus outside; shared-part and family/variant differences |
| WT DA | Association with initiation/continuation/switching | A onset/stages/termination, variants, pacing and trial-to-trial execution |

Plan: retain complementary catalogues, freeze identities for each declared
evaluation, infer broader supported recording coverage with saved models, and
compute animal/session/condition summaries. Separate catalogue-wide from specific
effects, with occurrence-weighted and equal-type summaries explicitly distinguished.
Karen contrasts include PFF/monomer by untreated wave and progression; Jonathan
includes WT/KI within condition, paired saline/amphetamine contrasts and genotype
differences in that contrast. Preserve opportunity denominators for repetition
and transitions. Mouse is the independent unit; calls, sessions, fibres and
processing variants do not increase independent animal N. Report effect sizes,
uncertainty, multiplicity and leave-one-mouse-out stability, including promising
nonsignificant effects. Prior exploration and upstream KPMS exposure make evaluation
imperfectly untouched; this does not make the recordings unusable.

WT DA questions: event association; execution/variant association within a chain;
and incremental information beyond movement, primitive identities, drug/phase
and recording time. Assess the last using animal-level validation and temporal
controls. A weak reproducible association may be useful; a null result need not
invalidate a chain. No guaranteed DA effect, causal claim or proof of chunking is
assumed.

## Where we most want your ideas

1. **Phenotype-relevant structure:** what measures can reveal condition/genotype
   differences in modifications and combinations that ordinary usage or pairwise
   transitions miss? Separate a useful measurement from a proposal for a new detector.
2. **Incremental value:** how should we establish that a chain or assembly adds
   information beyond its constituents, their frequencies, movement and context,
   without requiring every good chain to be strongly predictive?
3. **Family versus variant:** how can we test whether two executions belong to
   one routine with different pacing/parts, versus genuinely different behaviours?
   How do we preserve condition-specific changes without fragmentation or merging?
4. **Composition:** what verifies that A→B→A or a shared subroutine is meaningful
   organisation, beyond frequency, dwell-time, first-order transitions and context?
5. **Detection bias:** what checks distinguish a real group difference in chain
   recruitment/progression from condition-dependent detector sensitivity, tracking
   quality, unusual movement or different exposure/missingness?
6. **DA corroboration:** with8WT mice and already processed, temporally correlated
   signals, which small set of DA readouts and matched controls is most informative?
   What would count as useful incremental evidence rather than a movement correlate?
7. **Rare routines and discordant evidence:** how should we treat a rare but
   condition-specific chain, or a physically consistent catalogue with low KPMS
   string agreement? What information is needed before discarding either?
8. **Evidence standards:** which suggestions provide genuinely additional
   verification, which repeat the fitting objective, and which are better described
   as phenotype characterisation? What should precede a small matched video review?

Please propose a **prioritised table of at most8–10 additional measures**, with:
the biological/behavioural question; exact quantity/denominator; required data;
general versus intra/inter-specific applicability; baseline/control; independent
unit and validation split; fitting/selection circularity; sparse-data feasibility;
what a positive/negative result would and would not establish; and priority/cost.
Flag any required data we do not currently have. Prefer3–5 actionable ideas with
clear added value over a catalogue of generic metrics. Challenge our assumptions
where necessary, and state what cannot be determined without recordings/code.

Local provenance for the researcher (not required by the receiving chatbot):
[final quantitative report](../../outputs/broad_search_assessment_20261014_final/report.md),
[verified data/DA inventory report](../../outputs/broad_search_preflight_20261013/report.md),
[previous assessed external suggestions](GEMINI_SUGGESTIONS_20261013.md).
