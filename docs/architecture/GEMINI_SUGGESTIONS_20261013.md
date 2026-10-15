# Gemini suggestions for interpretable behavioural chains

Date: 13 October 2026. Status: scientific recommendations and discussion, not an
implementation request or a change to the frozen randomized search.

Gemini received the [external critique brief](BEHAVIOURAL_LAYERS_EXTERNAL_CRITIQUE_20261013.md),
without access to code, recordings or other project documents. Its response is a
source of hypotheses and alternative approaches; statements about what our data
or implementation necessarily does require independent verification. This note
records the suggestions we consider appropriate for the existing
[layer layout](BEHAVIOURAL_ABSTRACTION_PLAN.md) and
[broad search plan](BROAD_BEHAVIOURAL_SEARCH_PLAN.md).

## Target and constraints

The downstream target is interpretable recurring chains that distinguish
different behaviours while grouping similar executions. Their constituents,
variants and combinations should support later condition/phenotyping questions.
Useful behavioural description does not require proof of a central motor program,
a positive next-action prediction score, or a dopamine association.

Keep the numbering: 1 observations, 2 primitive/bout interpretation, 3 primitive
resolution, 4 episodes/chains, 5 organisation/composition, 6 context, 7 optional
external readout. Features remain available to later consumers. Composition and
inter/intra measurements are complementary uses of shared episode evidence.

Continue quantitative exploration before expensive matched qualitative review.
Reuse frozen KPMS and preserve original native clocks, masks and bout identities.
Maintain model-scoped numerical labels and annotation maps. Both Jonathan KI and
WT contribute behavioural evidence; WT-only DA is optional later corroboration.
Do not select behavioural definitions for the strongest phenotype/drug/DA contrast.

## Suggestions worth incorporating

| Suggestion | Layer | Contribution to interpretable chains and downstream readouts | Current position / recommended next step |
| --- | --- | --- | --- |
| Separate chain family, execution variant and context | 4–6 | Retains a common routine while exposing speed, repetitions, substitutions and spatial recruitment as distinct measurements. Allows phenotypes to differ in execution without inventing unrelated routine identities. | Highest conceptual priority. Variant/context groundwork exists; explicit shared-family definitions and mappings across candidate catalogues need further work. Separate families remain an option when evidence supports them. |
| Constituent-conditioned execution coherence | Analysis of 2–4 | Tests whether episode membership explains execution beyond the ordinary properties of its constituent syllables, instead of rewarding a cluster for using its own input features. | Extend current physical diagnostics. Examine unused channels and phase-aligned profiles, conditional on constituent identity where support permits; fit evaluation models on training animals. Flag related channels and remaining circularity. |
| Physical pacing, phase repetitions and interruption | 4 | Makes changes in tempo, omissions, repetition and termination interpretable even when overall chain frequency is unchanged. | Phase/completion and execution machinery exists. Apply it to compatible results, distinguish event-count duration from physical time, and separate evidence of interruption from censored/missing observation. Do not fabricate completion evidence for detectors that lack it. |
| Shared subroutines and alternative constituents | 5 | Supports meaningful combinations and modifications: the same investigation sequence could appear in several larger routines, with condition-dependent branching or reuse. | Strong next-stage addition after credible episodes. Finite caller-defined composition exists; automatic discovery of reusable parts, alternatives or hierarchy is not yet established. Retain child references and evidence for each proposed relation. |
| Broader proposals for flexible detectors | 4 | Avoids limiting discovery to routines with a frequent exact canonical string; supports themes distributed across many variants. | Audit proposal recall before expanding the model menu. Consider grouped near-matches, trajectory-grounded proposals or BASS as alternative discovery routes. Compare proposal strategies separately from detector families. |
| Catalogue redundancy and joint competition | 4–5 | Prevents several definitions from repeatedly claiming the same behaviour, stabilizing interpretable recruitment counts. | Raw overlaps and an explicit exclusive view already exist. First measure collision/redundancy and selection sensitivity. Consider calibrated joint decoding or fitting if those effects are substantial; some shared support is legitimate composition. |
| Context-, duration- and category-aware controls | Validation across 4–6 | Separates sequence organization from ordinary syllable properties, feature/category associations and slow session/context changes. | Extend the simpler current controls where the scientific question warrants it. Controls answer distinct questions; they are not universal pass/fail definitions of meaningful behaviour. Require adequate support in each conditioning stratum. |
| Boundary and missing-support sensitivity | 1–4 | Identifies whether apparent chain fragmentation or rarity depends strongly on segmentation or unavailable observation. | Targeted audit using actual exclusions and gap mechanisms. Inspect boundary-adjacent execution and selected-versus-full exposure. Preserve observed support; boundary relaxation or a missing-data model would be a separately defined experiment. |
| Blinded matched video comparisons | Validation of 4–5 | Grounds descriptive names and checks both grouping of variants and separation of different behaviours. | Essential after a quantitative shortlist. Use different animals, representative and borderline calls, hard negatives and deliberately retained rare candidates. Paired/triad formats are options; inconclusive discrimination is not proof of equivalence. |
| Multidimensional downstream readouts | 7 consuming 4–6 | Separates changes in recruitment, execution, branch choice, repeat/return organization and context rather than reducing phenotyping to counts. | Already consistent with the framework. Freeze behavioural definitions for each evaluation; assess associations/predictions on animal-level held evidence and report multiplicity. WT DA remains an optional separate modality. |

## What a reusable chain description should retain

An illustrative family might be **rear → investigate → descend**. Its records could
retain a family identity, supported constituents/phases, optional additional
investigation, tempo and turn magnitude, wall/centre context, neighboring routines
and uncertainty/censoring. This is an example of a description, not a claim that
this family has been identified in the present datasets.

That permits different downstream questions:

- **Recruitment:** how often and under which opportunities/context does the family occur?
- **Execution:** are phase durations, vigour or turning different?
- **Modification:** are optional phases, repetitions or exits chosen differently?
- **Organisation:** does it repeat, return or enter a different larger assembly?
- **Context:** are those differences explained or modified by position/session time?

The factorization is a hypothesis. A continuous execution change may deserve an
attribute; a stable, distinct variant may deserve a subtype or separate family.
Neither changing the next-action distribution nor preserving it is sufficient by
itself to decide semantic identity. Keep coarse families and finer variants
available when both descriptions are useful, rather than forcing one resolution.

Constituent counts and edit agreement are not counts of distinct semantic actions.
Descriptive family names should follow trajectory/video evidence, with model and
annotation provenance. Similarity across independently fitted vocabularies requires
explicit grounding; never infer it from numerical ID equality.

## Scientific qualifications to Gemini's proposals

### Proposal bias and independently fitted targets

The exact-proposal bottleneck is a substantive concern, especially for rare variable
routines. Our HSMM emissions can move during training, so their seeds are not hard
exact-match constraints, but initialization and limited target budgets still matter.
The current independent HSMM targets and jointly competing seqNMF factors differ
in catalogue fitting as well as model family. Retain this distinction in reports.
A joint wrapper requires a coherent objective and calibrated scores; merely adding
independent scores to a global decoder is insufficient.

[BASS](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009672)
is a relevant alternative because it grows a dictionary from single characters,
updates motif probabilities jointly, and removes shorter explanations as longer
ones emerge. Its permitted variation and background assumptions still need
evaluation for our data. Its success elsewhere is not evidence of success here.

### Segmentation, missingness and pharmacological mixtures

Frozen boundaries constrain all whole-bout routes; they cannot expose an internal
boundary absent from a source bout. A feature-only vocabulary on those bouts is
conditional on KPMS segmentation, but is not intrinsically incoherent. Grouping
several source states into one smooth action can be useful without establishing a
multi-action or neural motor chunk. The
[KPMS paper](https://www.nature.com/articles/s41592-024-02318-2) describes inferred
pose/noise and autoregressive dynamics, rather than a simple local boundary rule.

Pose-dependent missingness is plausible, not established for every excluded
interval. Missing keypoints, unsupported model frames and reviewed handling masks
are different mechanisms. Similar gap-neighbor distributions do not prove missing
data harmless. Joining unobserved intervals could create false sequences.

Label-free pooled fitting can favor common stereotyped regimes. That is a reason
to examine representation coverage across conditions, not to remove every
drug-associated routine. Equal recording/event budgets reduce exposure imbalance
but do not equalize temporal structure. Normalizing away entropy differences could
erase a real behavioural effect. Condition-specific routines can be meaningful;
they should not be selected because they maximize a desired condition contrast.

### Prediction, nulls and model fairness

Next-action prediction and information bottlenecks are useful optional diagnostics,
not mandatory definitions of a meaningful episode. Distinct routines can end in
the same state and have the same next-action distribution. When testing future
prediction, prevent future observations from influencing assignment/boundaries;
whole-recording inference can otherwise leak information into the proposed predictor.

Failing a first-order or triplet-preserving control does not prove a behaviour
meaningless or an upstream segmentation artifact. Exact-count surrogates can have
little freedom in sparse data; uniform sampling is a separate technical requirement.
Conditioned feature permutations also require enough observations and careful
within-animal/session strata. Quantify uncertainty and actual control constraints.

Equal type counts are not equal complexity across model families. Raw likelihood
ratios are not common scores across different inputs/families; use well-defined
held-out improvements against an appropriate common baseline where possible.
Maintain separate common-input and best-engineered comparisons, with actual
proposal strategy, features, budgets, seeds, convergence and researcher effort.

An independent critique of the pilot already found many hard-EM targets reaching
their cap; the completed broad run also shows frequent cap termination (24 of540
fitted targets met the convergence criterion within their short allocated budgets).
Evaluate whether adequate additional fitting changes shortlisted HSMM outcomes
before drawing model-family conclusions. Successful execution is not convergence.
Any further fitting is a future explicitly scoped task, not an alteration of the
current frozen tranche.

### Qualitative grounding, downstream claims and dopamine

Triad/paired review can test perceptual coherence and distinctness without requiring
semantic names. Reused clips, repeated reviewers and animal-specific cues limit a
simple independent-binomial analysis. Failure to distinguish variants may reflect
power/viewpoint limitations. Preserve rare candidates without treating arbitrary
numerical rarity thresholds as established biological requirements.

Existing upstream-model exposure and earlier exploration constrain claims of an
untouched full-pipeline test. They do not establish a known fixed magnitude of
optimism, nor guarantee that relative downstream comparisons are unbiased. State
which fitted components actually saw which animals and what each held-out test
does assess. Keep a small set of defensible representations when they answer
different questions; one universal winner is not required.

WT DA can assess incremental information beyond movement and primitive signals,
using appropriate animal/session validation and temporal controls. Processed
variants of one trace are sensitivity analyses, not independent observations.
[Markowitz et al.](https://www.nature.com/articles/s41586-022-05611-2) provide a more
nuanced account involving choice, reinforcement and future vigour than a simple
instantaneous movement interpretation. A boundary-associated signal does not
prove causality or chunking; a null association does not invalidate a chain.

## Practical sequence after the current tranche

1. Review the diagnostic vector and fitting conditions; retain diverse useful,
   weak, empty and fragmented outcomes rather than a single scalar winner.
2. Examine proposal coverage, redundancy, convergence, condition coverage and
   source/feature support. Use those results to justify specific follow-up work.
3. Add stronger conditional execution/control diagnostics where needed, then
   perform matched qualitative review on the shortlist.
4. Define grounded families and variants, retain supported phase/completion
   evidence, and investigate shared parts/assemblies on frozen lower evidence.
5. Freeze descriptions for a declared downstream evaluation; examine recruitment,
   execution, modifications and organisation with animal-level validation.

No code, runtime, source data, KPMS fit or search-space change was made when adding
this note. The current run and its fitting conditions remain reproducible.
