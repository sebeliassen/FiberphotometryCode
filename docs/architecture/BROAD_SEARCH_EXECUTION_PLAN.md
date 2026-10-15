# Broad behavioural search: readiness and staged execution

Later checkpoint on 13 October: the separate finite nohup search was implemented,
validated and launched under subsequent user authorization. Its frozen settings
and status belong in [CURRENT.md](../tasks/CURRENT.md) and the
[run notes](../../outputs/broad_search_20261013/README.md). The historical readiness
assessment below records the earlier planning stage. Scientific follow-up ideas
are assessed in [Gemini suggestions](GEMINI_SUGGESTIONS_20261013.md); that note does
not modify the running tranche. Quantitative exploration precedes matched video
review of a diverse shortlist.

Date: 2026-10-13. Status: proposed implementation and execution plan; documentation
only. No implementation, tests, pickle inspection, model fitting or search launch
is authorised by writing this document. The previous search remains stopped.

Companion: [full search-space discussion](BROAD_BEHAVIOURAL_SEARCH_PLAN.md).

Subsequent user instruction "proceed" authorises input verification and the first
implementation/method contrasts with bounded checks. It does not resume the old
service or request an indefinite broad search. The staged plan below still
describes later milestones; current evidence belongs in CURRENT.md and the
preflight report.

## Readiness assessment

Completed checkpoint: input/annotation pairing and initial categorical method
execution are recorded in [the preflight report](../../outputs/broad_search_preflight_20261013/report.md).
Seeded HSMM emissions/transitions and separate chain diagnostics are implemented;
feature-conditioned methods, learned duration/topology/catalogue discovery and
the full broad recipe space remain further work. The eventual random run must
use nohup with a persistent log and saved checkpoints, as clarified by the user.

We are ready to start a scoped implementation and bounded pilot once requested.
We are not yet ready to launch the full broad search. Existing exact/seqNMF
execution, frozen KPMS evidence, native interval contracts and downstream analysis
provide a useful foundation. Learned HSMM alternatives, some actual feature-input
routes, interchangeable whole-bout encoders and a durable dependency cache still
need implementation or verification. Configured methods must not be presented as
equally trained competitors to learned models.

The primary outcome is a useful description of episodes/chains: their boundaries,
constituents, timing, variants and repeatability across animals. Cohort differences,
DA association and classification are later readouts, outside recipe selection.
No new KPMS runs are needed.

Good chains should distinguish many separate behaviours while grouping very
similar executions, including modest timing or constituent variation. Quantify
within-chain consistency, between-chain similarity, constituent diversity,
redundancy and support/fragmentation across hyperparameters. No single diagnostic
proves meaningful behaviour; a uniform grooming routine remains potentially useful.

## 1. Freeze inputs and establish what is available

- Karen: use the reviewed-boundary observation view and verified frozen KPMS
  sources; retain existing provenance and animal splits. Earlier analyses make
  validation less perfectly independent; record that practical limitation while
  continuing to use these observations for exploration.
- Jonathan: candidate `data/sessions/jonathan_amph_sessions_v2_w_sylls.pkl`.
  The user's grooming-label hint instead resolves to AMPH Run12:
  `compat/syllable_generation/analysis/sequence_viz/run12_human_labels.csv`, consumed
  with model `amph_prod_k150k_run_12`, with candidate paired session input
  `data/sessions/jonathan_amph_sessions_run12_w_sylls.pkl`. Verify this pairing
  against saved labels/VFI before selecting it; do not apply these labels to IFDN
  or to v2 by matching numerical IDs.
  Verify the selected variant, KPMS model/vocabulary, annotations, animal IDs,
  session pairing, saline/amphetamine assignment, injection times, native frame
  mappings, feature availability and masks. File existence/source references do
  not establish its contents or quality.
- Retain Jonathan KI and WT behaviour. User interpretation is KI as the
  ADHD / late-stage Parkinson model and WT as wildtypes. Check age/stage metadata
  before assigning those descriptions to individual recordings.
- Verify WT DA channel identity, signal processing/units, alignment, missingness,
  artefacts and usable pre/post-injection support. KI DA is excluded from this
  verification role. Missing DA does not remove eligible behavioural observations.
  Existing processing and verification documentation should be reused. Inspect
  actual `signal_meta` and suffixes rather than defaulting to signal reprocessing.
- Create separate cohort manifests and animal-grouped discovery/development/final
  evaluation scopes. Fit transforms, codebooks and geometry only on training animals.
  Verify counts before deciding how many folds are supportable.

Deliverable: versioned input/capability manifest, exclusions and exposure summaries,
split specification and a clear list of unavailable features. This stage decides
whether every requested comparison is actually supported.

## 2. Make the first broad comparison executable

Use existing shared owners and contracts; extend the current infrastructure rather
than create another framework. Implement meaningful acceptance checks before a
substantial empirical run, including gaps/masks, physical clocks, animal isolation,
honest capability failures and actual consumption of active feature inputs.

Initial episode families: exact motifs as a structural baseline, seqNMF, and a
train-fitted staged HSMM. Retain configured HSMM as a separately identified
diagnostic branch. Add tolerant motifs or other alternatives where their current
implementation is verified; unavailable branches remain explicit backlog items.

Initial representation routes:

1. Frozen KPMS bouts with cohort-specific identity/coarse resolution views.
2. The same bouts with movement/posture features supplied directly to supported
   episode models, alongside a matched descriptive-only control.
3. Whole-bout execution subtypes and feature-only whole-bout types, retaining the
   original KPMS intervals and parent identity where applicable.
4. Spatial features as descriptions versus actual model inputs, after movement
   routes work. Use camera-specific proxies with declared units, not physical walls.

Joint categorical/continuous and parallel/twin routes belong in the broader
exploration, but enter only once their encoding and downstream consumers work.
Independent frame segmentation is a distinct later comparison because it changes
both vocabulary and boundaries. Do not silently narrow the final search to the
first executable branches.

Compare order-only and physical-duration representations where methods support
them. Preserve gaps and censoring. Any feature route unsupported by a method is
reported as incompatible rather than silently ignored.

Cache native features, frozen bouts, whole-bout profiles, training transforms,
codebooks, fits and inference separately. Include exact consumed dependencies,
split, seed, masks, clocks, numerical settings and source versions in identities.
Coordinate concurrent creation and publish completed artifacts atomically.

Deliverable: an executable capability matrix, recipe specification and validated
adapters for several materially different branches.

## 3. Run a bounded, deliberately varied pilot

Select a small animal-grouped sample in both cohorts with representative recordings
and continuous windows. Include censoring, interruptions and varied movement;
avoid assessing only convenient examples. Decide exact sample size and runtime
from the verified manifest and resource benchmark, rather than invent them now.

Run matched recipes on the same spans, multiple seeds where fitting is stochastic,
and enough convergence diagnostics to distinguish underfitting from method failure.
Render identical native-video spans with each detector's boundaries and phases,
plus ambiguous and unmatched/background examples. Compare model output to video
without seeing genotype/group labels where practicable.

Review coverage, fragmentation, overlapping calls, routine support across animals,
tempo sensitivity, phase evidence, ambiguity, stability and runtime/memory. Use
method-specific diagnostics separately; seqNMF reconstruction and HSMM likelihood
are not one universal score. Include order controls appropriate to the observation
representation; first-order null gains alone are not proof of coherent behaviour.

Pilot exit conditions: verified interval support; active inputs truly consumed;
several viable method families; interpretable matched differences; known failure
categories; and measured RAM/disk/runtime adequate for a bounded larger run.
If a family fails, diagnose budget/initialisation/representation before rejecting it.

Deliverable: pilot comparison report, representative clips and a revised recipe
space. This is the point to agree the larger run's concrete scope and resource caps.

## 4. Run a balanced broad exploration

Use a finite, checkpointed first tranche. Suggested allocation is approximately
50% diverse valid recipes, 30% matched contrasts and 20% stability/deeper-fit
checks; revise after pilot costs are known. Allocate budgets across method families
so cheap exact fits or a huge seqNMF grid do not dominate.

Launch the random tranche under **nohup**, recording the PID, exact command,
manifest, log path and checkpoint location. The chat/foreground terminal must
not be required to keep it running. Verify launch/dispatch/checkpoint evidence
and explicit stop/resource guards; do not resume the previously stopped service.

Reports state what each model was given to become useful: proposal/initialisation,
training animals and observations, tuning ranges, restarts, iteration/time budgets,
convergence, manual attention and fixed versus learned parameters. Differences in
development effort are a comparison limitation, not hidden as equal optimisation.

Explore method × resolution, method × time representation, feature bundle ×
consumer route, and subtype granularity × cross-animal support. Include both
cohorts within their own model/vocabulary identities. Reuse caches and frozen
fits for downstream changes. Record empty, unstable, incompatible, numerical and
resource-limited outcomes as different result categories.

Benchmark representative workloads before selecting workers, thread limits,
RAM/cache caps, disk quotas and timeouts. Reuse existing watchdog/checkpoint
safeguards; confirm they cover new fit types. Use GPU only if a bounded benchmark
shows useful speedup with validated numerical tolerance. No indefinite launch is
the default for this new search.

Deliverable: a representation atlas explaining what changes when each choice
changes, including useful failures and diverse outcomes rather than one leaderboard.

## 5. Add organisation, composition and context to retained episode families

Apply the shared inter/intra measurement bank, repeat/return structure, execution
profiles and supported finite grammar/composition to the same frozen episode calls.
These are complementary analyses, not mutually exclusive choices. Keep alternative
assembly parses separate so overlapping evidence is not counted twice.

Compare measured contexts and verified configured/learned context routes. Shared
subroutine discovery and learned grammar remain explicit implementation work;
predefined grammar examples cannot stand in for discovering grammar. Freeze lower
definitions first; any joint model that revises them is a separately declared branch.

Deliverable: comparisons of what organisation/context adds and whether useful
episode distinctions survive downstream analysis. Later layers must not delay the
initial chain-method pilot.

## 6. Freeze a small selection, then assess downstream evidence

Choose a diverse set of representations using behavioural quality and support,
then freeze definitions before group comparisons or DA inspection. Evaluate on
animal-held-out observations; repeated sessions never become independent animals.

Karen readouts can include PFF/monomer and wave changes with the existing
age/wave/L-DOPA confounding stated. Jonathan readouts retain genotype, drug and
pre/post time separately. A second dataset tests method robustness; it does not
automatically replicate Karen disease findings. Different KPMS vocabularies need
explicit mapping before claiming the same chain across datasets.

For WT DA, compare episode-aligned profiles and appropriately held-out associations,
with movement-only/time/drug baselines and control alignments. Account for temporal
autocorrelation, slow drift, repeated animals, plausible lags and multiple explored
endpoints. Keep processing variants identified. Behaviour-related photometry
artefacts and common drug trends can produce associations without validating a
routine. DA is supporting evidence, not a requirement, optimisation target or
causal explanation. Avoid selecting routines or lags for the strongest DA result.

Deliverable: frozen candidate definitions, downstream estimates with uncertainty,
support/missingness and sensitivity summaries, plus limits of prior data exposure.
Prior exploration makes this evaluation less perfectly independent; it does not
invalidate using the datasets for method development and behavioural exploration.

## Open questions and things to consider before starting

| Issue | Current uncertainty or risk | Proposed handling |
| --- | --- | --- |
| What counts as a good chain? | Coverage/stability can reward generic background states; sparse meaningful routines may score poorly | Use several diagnostics plus blinded matched video review; retain distinct granularity families |
| Discovery target | Common routines, rare events and variable routines need different sampling/detectors | Include all as explicit outcome types; do not collapse them into one quality score |
| Vocabulary across cohorts | KPMS IDs and L1–L4 annotations may be model-specific | Fit/evaluate within cohort; verify mappings or compare physical profiles instead of equating IDs |
| Input and signal readiness | Jonathan channels, alignment, masks, session counts and selected variant remain unverified | Complete stage 1 before empirical fitting |
| Feature confounding | Spatial proxies may identify cameras/animals; duration subtypes may merely encode speed | Camera-aware training references, held-out animals and matched simpler-feature baselines |
| Catalogue fragmentation | Extra subtypes can erase shared routines or create rare types | Track parent relationships, support and duplicates; allow execution variants without new identities |
| Model comparability | Configured HSMM versus trained seqNMF is an unequal learning comparison | Separate configured/learned/seeded provenance and check budget/convergence sensitivity |
| Previous exploration | Earlier Karen/Jonathan analyses make validation less perfectly independent | Record this practical limitation and use animal-held-out evaluation; the datasets remain useful |
| Dopamine independence | DA is related to movement and drug/time; processing artefacts can look behavioural | WT only, late downstream evaluation, movement/time baselines and alignment controls |
| Scope and cost | All combinations including learned grammar could make an unbounded project | Stage the work, cache invariants and set a finite first tranche after timing evidence |
| Human annotation burden | New vocabularies cannot inherit all KPMS names | Preserve mixed/unmatched descriptions; label representative useful outputs, not every trial |

No further conceptual choice is required to begin input verification and scoped
implementation when requested. The pilot should resolve most routine choices.
Full-run sample sizes, exact recipes and resource limits remain contingent on
verified inputs and measured costs.
