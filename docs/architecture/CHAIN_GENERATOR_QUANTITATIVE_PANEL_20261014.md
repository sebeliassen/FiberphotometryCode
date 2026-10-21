# Next quantitative panel for chain-generator evaluation

**Execution update, 14 October 2026:** The authorised fixed 23-setting behavioural
round completed 2,038/2,038 jobs successfully, including 1,927 broader held-recording
evaluations and 111 saved-fit reproduction/training-descriptor checks. See the
[compact configuration scorecard](../../outputs/generator_verification_20261014/scorecard.md)
and [completed readout checkpoint](../../outputs/generator_verification_20261014/report.md).
The [supporting WT DA pass](../../outputs/generator_verification_wt_da_20261014/report.md)
also completed, with original 470-nm processed traces and actual camera/photometry
timestamp audits. The following text retains the original proposal: secondary
boundary/forecasting, calibrated arena-zone and detector-input surrogate work
are not implied complete. Baseline transfer currently covers chain and primitive
recruitment frequencies; execution/organisation transfer is not separately tested.
DA matching is observational; no temporal-shuffle or incremental predictive DA
model was run. No model/KPMS refit or random-search restart occurred.

Date: 14 October 2026. The user explicitly prefers further quantitative
evaluation before qualitative review. This updates the earlier recommendation
to move directly to a small video screen. The absolute objective remains finding
condition/genotype behavioural patterns using chains, intra-chain modifications
and inter-chain combinations. This document assesses the new Gemini suggestions
and defines the original bounded next round; see the execution update above.
The random search remains stopped and existing KPMS stays frozen.

The concrete evaluation set has now been selected: **23 cohort-specific
configurations (9 core + 14 diversity challenges; 12 Karen / 11 Jonathan)**,
covering111existing completed fold/seed evaluations. See the
[full hyperparameter list and plan](../../outputs/generator_verification_panel_20261014/report.md)
and [exact frozen-parameter manifest](../../outputs/generator_verification_panel_20261014/panel_manifest.json).
The executed and pending readouts are distinguished in the execution update.

## Evaluation principle

Evaluate a generator as a useful, reproducible representation with multiple
readouts—not as a winner on one statistic. Condition effects, stable habits,
annotation concordance, context association and predictive utility are different
questions. Strong performance on one does not prove semantic identity or require
the others to be positive. Report each best complete configuration and its fold/
animal support; do not compare averages across a method's random settings.

Group information is a legitimate exploration/prioritisation axis given the
user's objective. Preserve total effects as well as effects beyond lower-level
baselines. If this information selects the representation, ensuing significance
claims on the same animals are exploratory rather than independent verification.
Selection/testing dependence is not removed simply by using unsupervised fits
or calling a derived metric external.
[Kriegeskorte et al.](https://www.nature.com/articles/nn.2303)

## Recommended measurements and priority

| Measurement | Concrete comparison | Added value and qualification | Priority |
| --- | --- | --- | --- |
| Group/condition differences in chains themselves | Native-time initiation/occupation, durations and catalogue diversity; general summaries and supported individual chains | Directly addresses the downstream task. Show animal-level effects, uncertainty and exposure, alongside primitive frequencies/movement and matched arbitrary-window summaries. Strong inherited primitive effects can still be useful descriptions. | Core |
| Group differences in intra-chain modifications | Same constituent inside versus outside a declared chain, with matched epoch/context/support; group×membership contrast for pacing, turning and execution; variants/phase proportions where homologous phases exist | Tests whether changes are concentrated in particular routines rather than uniform constituent changes. Observational membership selection and detector-consumed features remain explicit. Fixed-length calls do not establish aborted attempts. | Core |
| Group differences in inter-chain combinations | General repetition/switch/run/return readouts, then supported A→A,A→B,A→B→A and assemblies; compare observed and condition/session-matched nuisance controls | Distinguishes usage from organisation and tests modifications/combinations. Keep physical opportunity denominators, gaps/censoring and context regimes. Controls require adequate draws and documented preservation; current two-draw search controls are insufficient for a new inferential panel. | Core |
| Cross-session baseline transfer | Within-animal versus between-animal similarity of recruitment, execution and organisation on comparable pre-injection epochs; baseline primitive/movement representation as comparator | Useful repeatability/generalisation. Jonathan has two pre-injection session baselines; same-condition independent repeats are not established. Karen3w→8w includes age/wave change. Camera/chamber, morphology, epoch and exposure can contribute to mouse identification. Label it transfer/repeatability, not guaranteed stable-trait ICC. | Core where support permits |
| Partial annotation enrichment/coverage | Chain-specific enrichment for verified coarse labels, coverage of labelled bouts and distribution across types; account for known labels, unresolved labels and opportunity frequencies | Practical additional check. Human labels are partial primitive annotations, not frame-level chain ground truth. Useful mixed-action routines and multiple grooming variants are allowed; no requirement for exactly1–2 pure grooming chains. Extend beyond grooming where annotations support it. | Core, low incremental cost |
| Spatial/context association beyond primitives | Held-animal location/context prediction with chain features versus a comparable baseline of constituents, movement, epoch and source context; use verified arena geometry for wall/corner/centre names | Useful supporting context evidence for routes where raw location was withheld; descriptive input-coherence evidence for spatial-input routes. Data-derived occupancy bins are not automatically arena zones. No spatial association is required of every behaviour. | Supporting, after geometry/input audit |
| WT DA inside/outside and execution associations | Matched same-constituent, same-session/epoch contrasts and incremental execution/context association, evaluated at mouse/region level | Additional modality with up to8WT mice, not independent semantic truth. Requires actual sensor/trace provenance, masks, timestamps and appropriate temporal controls; suffixes are sensitivity views. Null results do not invalidate chains. | Include in the round after readiness checks |
| Boundary entropy profile | Internal/exit uncertainty relative to matched source-state, length, context and detector-on-surrogate or displaced-boundary baselines; report sample support | Useful boundary diagnostic, partly selected by motif/sequence fitting. Exact motifs constrain internal transitions by definition. Some valid routines have predictable exits or internal branching. No universal entropy cliff/pass threshold. | Secondary |
| Post-chain predictive horizon | Held-animal log-loss improvement for1–5 subsequent bouts versus matched final-state/history, movement, epoch/context and arbitrary-window baselines | Supports useful history/context compression; not proof of behavioural or neural hierarchy. Assignment and boundary selection must not use the predicted future. Stage-specific classifiers need sparse-outcome support. A meaningful routine can end with no persistent predictive effect. | Later/targeted unless leakage-free eligibility is established |

## Improve the proposed group ablation

Do not rank generators by `abs(d_real) − abs(d_surrogate)` alone. Separate
standardisations, sign changes and selective extremes make this hard to interpret;
maximising it can reward noisy or rare catalogues. Nor is Markov excess a mandatory
definition of useful behaviour.

Prefer two reports for each same-unit observable Q:

1. **Total phenotype/condition contrast** in Q, with the primitive/movement
   comparator and exposure/support diagnostics.
2. **Contrast in nuisance-adjusted excess**: compute each animal/session's
   `Q_observed − mean(Q_surrogate)` under a declared null, then compare that
   same-unit excess between groups/conditions. Preserve signed effects. Carry
   Monte Carlo and animal-level uncertainty rather than only subtracting absolute
   effect sizes.

Controls should fit the question: nuisance-matched categorical frequency/order,
duration/category associations for duration-sensitive detectors, and slow epoch/
context recruitment where relevant. Apply fixed detectors/codebooks under a
coherent surrogate input intervention. A feature stream and native event-slot
duration cannot be arbitrarily mismatched and called a realistic behaviour.
Quantify what a null preserves and what it destroys; overconditioning can remove
the actual phenotype of interest. Keep total and adjusted questions side by side.

For intra analysis, estimate a group×inside/outside contrast for the same source
constituent, with animal/session grouping and explicit common support. A faster
inside chain alone is not evidence of differential group modulation. Percentage
changes with different baseline means require care. Inferential unit remains mouse,
not bouts, fibres or sessions. Karen untreated progression and separate L-DOPA
wave, and Jonathan true drug/pre-post/genotype fields retain their documented
confounds; paired difference-in-change is a useful additional contrast.

## New source checks that change Gemini's proposed assumptions

### Grooming annotations are more extensive than the original hint

The actual54-row Run12 annotation CSV contains **nine** grooming IDs:
`27,44,52,58,60,61,63,65,66`. The six IDs in the brief were verified hints, not
the complete label set. ID66 has the note "some hands", but this table does
not provide a complete face/head/body/caudal phase ordering. Consequently,
transition asymmetry among these IDs cannot establish cephalocaudal syntax.
Incomplete labels cannot support exhaustive precision/recall of true grooming.

All nine current core configurations use identity primitive resolution, so that
annotation merge is not an input to these particular fits. Some configurations
in the wider search **did** use human annotation merges: anchor concordance is
partly supervised for those routes and must be labelled accordingly. Even for
identity fits, independent human primitive labels are not independent chain
annotations. Use the complete model-bound map and retain its uncertainty.

### Location was used by several shortlisted configurations

Actual consumed-feature metadata establishes:

- Direct spatial-input routes: Karen2967,Karen2729,Jonathan2729—all parallel
  execution/spatial-codebook configurations.
- Raw spatial coordinates not directly supplied: Karen2887,Karen3025,Karen1792,
  Jonathan919,Jonathan2368,Jonathan1792.

For the latter group, some models still consume movement features derived from
centroid displacement; unused absolute location is not complete statistical
independence. Spatial segregation is a conditional/context diagnostic, not
"100% non-circular" ecological validation. Prepared centroids do not include
verified arena centre/boundary geometry; wall/corner/radial interpretations require
that additional calibration. Preserve per-camera/recording transformations.

### Boundary and forecast selection can reuse the test evidence

Exact matching necessarily fixes internal motif transitions, so comparing their
entropy to unconstrained outside transitions can reward the definition itself.
seqNMF calls use finite future-containing windows; HSMM paths are offline
sequence decodes. Whether a boundary exists can depend on later observations.
Before prospective next-bout analysis, either establish that the assignment uses
only evidence up to the forecast origin, or use a prefix-only decoding policy.
Otherwise label it retrospective association rather than zero-circularity
prediction. Held-out mice alone do not remove this mechanistic leakage.

## Practical next round and stopping point

1. **Preserve the nine as a core, and add a diversity challenge panel from existing results.**
   The nine were diagnostic-selected, not a representative search-space sample:
   allidentityresolution, shorttemplates, mostlyseqNMF. Include supported omitted
   routes/families, longer templates and alternative resolutions; retain strong
   configurations within each region, not method-wide averages. Use active
   parameter coverage and actual detection overlap to manage redundancy. See the
   [diversity audit](../../outputs/generator_diversity_20261014/report.md).
   Lock chosen settings and evaluation policy before the new readouts. Retain current
   structural/sensitivity evidence. Define common reference catalogues or supported
   correspondences before pooling specific IDs/stages across fits; overall catalogue
   invariants can be analysed without pretending local IDs match.
2. **Infer broader supported recordings with saved fitted models.** No new KPMS
   or hyperparameter search. Use matched native-time epochs, animal-level splits,
   original masks and fixed overlap/threshold policies. The current180-bout windows
   do not support full-session fingerprinting or robust rare-combination estimates.
   Keep Karen's five reserved animals out of iterative tuning/selection.
3. **Run the core panel together:** general/specific recruitment, intra and inter
   phenotype contrasts; partial annotation enrichment; comparable-baseline transfer.
   Include a matched primitive/movement and arbitrary-window representation where
   appropriate, with similar complexity/opportunities so any structured compression
   is not mistaken for novel biological information.
4. **Add spatial and WT DA where prerequisites/support permit.** Complete trace/
   timestamp/region eligibility audit before DA outcomes; missing data stays missing.
   A sparse optional modality should not halt all behavioural evaluation. Boundary
   diagnostics can be an inexpensive accompanying sensitivity. Post-chain forecasting
   can wait unless its actual detector assignment passes the future-information audit.
5. **Issue a configuration-level scorecard, then narrow for video.** Show effect
   size/uncertainty, fold and mouse influence, constituent/context baselines,
   annotations, input circularity, sample counts and compute conditions separately.
   Do not produce a weighted "truth score" or fill unavailable axes withzero.
   Promising nonsignificant effects and small supported condition-specific motifs
   remain eligible as exploratory candidates.

This is one targeted evaluation round on a core plus diversity challenge panel
of fixed candidates, followed by qualitative
grounding of the promising and discordant results—not a new open-ended metric or
hyperparameter search. The execution framework, full-session inference and new
readouts still need implementation/integration and appropriate validation; this
document does not claim those analyses already ran.

Evidence: [current assessment](../../outputs/broad_search_assessment_20261014_final/report.md),
[recent sensitivity/critique triage](GEMINI_VERIFICATION_TRIAGE_20261014.md),
[actual human labels](../../compat/syllable_generation/analysis/sequence_viz/run12_human_labels.csv),
[new analysis brief](CHAIN_VERIFICATION_EXTERNAL_BRIEF_20261014.md),
[detector/feature workflow](../../src/fpcode/workflows/sequence_search.py).
