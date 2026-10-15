# Gemini verification critique: disposition and route to qualitative review

Date: 14 October 2026. Gemini saw only the standalone brief, not source or
recordings. Its response supplies hypotheses and useful measurements, not verified
diagnoses of these data. We read the complete response, checked relevant source,
and ran two bounded sensitivity audits on saved calls. No search was resumed,
model fitted, DA reprocessed, group comparison executed or video rendered.

## Decision

Keep the useful additions, correct unsupported claims, and begin a bounded
qualitative screening stage without waiting for every new quantitative measure.
Our goal is condition/genotype patterns through chains, intra-chain modifications
and inter-chain combinations. Qualitative grounding is additional evidence needed
to decide whether grouping reflects related actions; it need not await proof of
neural chunking, predictive gain or a significant DA effect.

The initial recommendation is to review Jonathan919 alongside2729/2368, and
Karen2887 alongside2967/3025, retaining HSMM1792 as a different structural reference.
These are candidate settings, not identical chain identities across folds.

## What is already addressed, and what remains

| Topic in Gemini's response | Actual status | Disposition / timing |
| --- | --- | --- |
| Fixed600/180-bout budgets and condition-dependent physical exposure | Correctly disclosed sampled-window limitation. Budgets are divided across each animal's recordings; one reproducible contiguous whole-bout window per recording, not a first-N injection block. Uniform event counts do not imply equal elapsed exposure. | Real issue for session-level phenotype/organisation estimates. Use broader inference and fixed native-time epoch/exposure summaries before those claims. Does not invalidate sampled discovery or block review of existing calls. |
| Native clocks, masks and gap barriers | Native KPMS projections and whole-bout endpoints verified; calls already use native frames. Different40Hz model/60FPS playback/~80Hz acquisition clocks are explicitly separated. | Preserve this evidence when extracting clips. DA-specific timestamp coverage/drift still needs checking before DA association; no second model-grid conversion. |
| Physical selection circularity | Consumed features flagged; unused-channel and per-KPMS-state residual diagnostics already calculated. Those are additional diagnostics, not independent truth. | Retain, with the new duration/phase-weighting sensitivities below. |
| Jonathan919 may be primarily a duration grouping | Plausible sensitivity concern, not an established artifact. Already had unused-channel evidence. | Checked now: duration/path exclusion and equal-bout phase weighting reduce but do not remove its physical distinction. Prioritise discordant strings and matched kinematic foils in qualitative review. |
| Karen2887 fold heterogeneity/single-mouse influence | The .78/.13/.04 residual split was already highlighted. It is not proof of non-ergodicity, a single-mouse driver or a specific disease phenotype. | Checked now: thin cross-animal support and sign-sensitive weaker folds. Retain cautiously as a narrow hypothesis/reference, alongside broader alternatives. |
| Fold identity/aggregation | Cross-animal physical scores use equal feature, animal-pair and eligible-type weights. Seeds are summarized within fold, then folds equally. No best-type matching or pooled chain-ID equality. | Structural summaries do not need common IDs. Specific families pooled across fits do need a declared reference/crosswalk, which remains a real prerequisite. |
| Tracking/support differs by condition | Barriers are preserved, but formal condition/epoch-stratified missingness, availability and edge-proximity effects have not been quantified for the shortlist. | Important inclusion before recruitment/attrition group claims. Annotate support/boundary limitations in review manifests; report physical opportunity denominators and unknown censoring. |
| Detector sensitivity under duration/posture shifts | Input mechanisms known, but deliberate frozen-detector perturbation experiments not yet run. | Targeted follow-up for feature-sensitive candidates if a condition effect emerges. Sensitivity is not automatically artifact: duration/kinematics may be the phenotype being measured. Perturbations must preserve a coherent time/feature intervention. |
| Context-conditioned residuals and incremental execution gain | State-only adjustment done; neighbour/context-aware adjustment and held-animal nested prediction not done. | Useful small addition after/alongside review, especially for919. No need to make every chain pass. Sparse transition strata need regularisation and support diagnostics. |
| Soft synonym costs and flank-context similarity | Original KPMS annotations/physical evidence exist; the proposed new scores are not implemented. | Targeted diagnostic for ambiguous families, not a required preliminary pipeline. Meaningful variants can have different recruitment contexts—the intended phenotype. Video can supply the missing semantic evidence. |
| Arena radial/zone covariates | Centroid coordinates available; a verified arena centre/boundary mapping is not part of the prepared feature table. | Define geometry per chamber/recording before treating pixel radius or zones as comparable context; this is an extra input requirement, not automatically available from centroids. |
| Stage pacing and prefix progression | Existing intra-chain machinery and the planned measurement bank already cover these aims. New shortlist not comprehensively analysed. | Add phase-duration/log-ratio views where homologous positive phases exist. Progression needs eligible prefixes/phase evidence; fixed-length successful calls do not reveal failed attempts. |
| Epoch-/duration-aware inter-chain controls | Simple fixed-slot shuffle/first-order controls done; richer context/semimarkov controls not done. | Important before claims that combinations exceed local/context recruitment. Not a prerequisite to viewing examples of the proposed combinations. Audit what surrogates actually preserve and whether sufficient rearrangement remains. |
| WT DA trace/movement/context issues | WT-only policy, existing suffix inventory, region/fibre metadata, masks and processing concerns already documented. No associations run. | Include matched inside/outside constituent contrasts and a small set of execution readouts after families are credible. Trace/provenance/timestamp checks precede DA analysis, not behavioural video review. |
| Jonathan fixed-order session confounding | Already disclosed. Pre/post and injected drug stay separate. | Include paired difference-in-change as a useful contrast, plus component/pre-post/session summaries. It does not remove all order/time confounding or justify banning every direct descriptive comparison. |
| Rare routines, prior exposure and multiplicity | Rare outputs preserved; upstream exposure and exploratory selection disclosed. Mouse-level inference/multiplicity planned. Reserved Karen animals untouched. | Keep informative rare examples with honest support. Do not adopt arbitrary4-mouse/40%-influence/positive-every-fold gates; do not consume reserved animals to tune the shortlist. |
| Matched blinded clips | Planned, not yet rendered/reviewed for the new shortlist. | Next qualitative stage. Include representative, discordant, borderline and matched outside-chain examples across animals/conditions. |

## New bounded checks completed

Evidence: [candidate_sensitivity.json](../../outputs/gemini_triage_20261014/candidate_sensitivity.json)
and its [reproducible script](../../outputs/gemini_triage_20261014/check_saved_candidates.py).
Scope: nine existing evaluations (six Jonathan919 seeds/folds, three Karen2887
folds), saved calls and original prepared features, with fixed fitted models and
training-only normalisation. Leave-one-mouse-out here removes held observations;
it does **not** retrain detectors, estimate a population confidence interval or
establish independent confirmation.

### Jonathan919

| Source-conditioned execution comparison | Fold0 | Fold1 | Fold2 | Same-configuration summary |
| --- | --- | --- | --- | --- |
| Original native-time phase profiles | .464 | .492 | .477 | .477 |
| Exclude duration and path length | .177 | .247 | .225 | .216 |
| Also replace duration-weighted phases with equal bout slots | .076 | .143 | .131 | .116 |

All variants remain positive across folds. Under the strictest version, all30
single-held-mouse omissions across six fits remain evaluable and positive
(range.048–.219). Original-score omission range is.399–.582. This supports retained
physical distinction while showing that the original~.48 score partly reflects
duration and phase allocation. It is not proof of one ethological routine, synonymy
or higher-order organisation. Transition/context adjustment is still outstanding.

### Karen2887

Original conditioned scores are.781/.127/.040. Excluding duration/path gives
.609/.070/−.016. In the strongest fold there is only one own type with enough
repeated cross-animal support. Its calls come from mice10/19/20/4 with6/2/1/11calls;
mouse7 has none. Removing mouse10 or4 makes the cross-animal comparison unavailable.
Removing other mice leaves it unchanged. That indicates dependence on a small
supported comparison, not evidence that one mouse generated a spurious.78 effect.
In the other folds, original-score omission ranges include−.045 and−.153.

We should review this narrow catalogue without describing it as a general robust
cohort routine, and compare its supported types to broader Karen candidates.
Neither a diagnosis of an idiosyncratic stereotypy nor a late-wave PFF interpretation
follows from these numbers.

Baseline scores reproduce the earlier saved assessment. Full/strict results and
missing-comparison cases are retained. Final calculation:1.43s,51.7MiB measured
peakRSS,4GiB address-space cap,one process,two configured numerical threads,float64.

## Where Gemini's proposed conclusions need correction

- **Incremental prediction is not automatically out-of-objective validation.**
  If chain assignment used the target physical feature, a held-animal regression
  can still predict it partly by construction. Predicting next syllables from
  offline chain assignments that already used those syllables leaks future
  evidence. A credible prospective test needs prefix-only assignment; a
  retrospective execution analysis must be labelled accordingly. Gains against
  an additive neighbour model can also reflect nonlinear neighbour interactions,
  not uniquely a longer memory or motor preparation. Non-independent selection
  and evaluation require care even when the resulting statistic looks useful.
  [Kriegeskorte et al.](https://www.nature.com/articles/nn.2303)
- **Context similarity does not establish shared semantic identity.** A common
  family may change context with condition, while different actions may share
  context. Soft latent/transition agreement is a useful sensitivity, but cosine.80,
  JS.15 or edit agreement.65 are not validated cutoffs. Latent origin/scales,
  correlated construction inputs and sparse context estimates matter.
- **Prediction/co-articulation does not prove hierarchical planning.** Hidden
  movement/context, smoothing and offline segmentation can create such associations.
  A negative result does not prove memoryless concatenation or fully explained DA.
  Gemini's disease-specific attrition/entropy predictions are hypotheses, not
  established findings for these cohorts.
- **DiD is a contrast with assumptions, not a repair of fixed-order design.**
  Different within-session habituation/drift or drug/session interactions remain.
  Difference-in-change is worth reporting, but causal interpretation depends on
  appropriate counterfactual trend assumptions; retain the actual component data.
  [Callaway and Sant'Anna](https://arxiv.org/abs/1803.09015)
- **All photometry suffixes are not centered rolling filters.** Current helpers
  compute legacy `_dff` by reference subtraction/division and `_zscore` by global
  mean/SD; the Datta/RANSAC path has centered5s rolling baseline and20s rolling
  z-score plus zero-phase reference filtering. These current helpers are source
  evidence, not complete proof of each stored historical trace's provenance.
  Nonlinear rolling baseline/SD effects cannot be represented as universal W/2
  backward copying or fixed identical transfer functions. Raw-only analysis and
  a W/2 buffer are not universal requirements. Avoid unsupported anticipatory
  claims and check the actual stored processing/mask behaviour.
- **Clock alignment must not be optimised to a behavioural DA association.**
  Behaviour–DA/reference cross-correlations can change for biological or artifact
  reasons. They may be diagnostics, but do not independently identify clock drift.
  Use timestamps/acquisition/frame anchors as the primary evidence.
- **Sensitivity is not proof of an artifact.** Changing duration inputs can
  legitimately change duration-defined variants; changing only timestamps can
  also mechanically alter calls/minute. Record changes in detections separately
  from changes in physical exposure and define coherent perturbations. HSMM
  duration PMFs here concern event counts, not direct physical-second priors.
- **Censoring and evidence thresholds are not universal gates.** A support gap
  is unobserved continuation, not an abort. Survival-style inference needs a
  defined risk set and suitable censoring assumptions. High support/LOMO failures
  can reflect scarcity rather than wrong identity. Inconclusive human discrimination
  can reflect view, task or reviewer limitations; it is not automatic rejection.

## Bounded path into qualitative review

1. **Keep the current candidate set and attach the new qualifications.** No
   universal winner, new hyperparameter search or numerical pass gate is required.
   Select a declared fold/seed catalogue for each review panel; retain provenance
   and do not pool equal numeric type IDs across fits.
2. **Build a small first-pass sample manifest.** Cover every observed type in a
   few complementary catalogues with2–3 examples where support permits. Prefer
   different animals and conditions, retaining discordant strings/tempo and a
   few boundary/low-support cases. Add matched outside-chain foils. Annotate
   genuine gaps and sampled-window boundaries; hide detector/group identity in
   the reviewer-facing display while keeping an answer key.
3. **Validate native video mapping and each extracted interval before scaling.**
   Include pre/post context; model-derived onset/offset may be inaccurate even
   with correct clocks. Existing within-call agreement would not detect systematic
   temporal misplacement. Record playback versus physical time. Begin qualitative
   screening rather than requesting20–30clips for every chain in every catalogue.
4. **Review relatedness, distinctness and boundaries separately.** Ask whether
   members share a behaviour, whether variants preserve family identity, whether
   separate types split similar acts, and whether timing captures a complete
   meaningful episode. This screening is not a powered blind discrimination study.
   Expand review for contested/useful families rather than reject on uncertainty.
5. **Then perform the larger downstream work.** Broader inference with supported
   native-time epoch denominators; condition-specific support/sensitivity audits;
   general/specific intra/inter readouts; stronger context/duration controls where
   combinations are claimed; and WT-only DA with documented processing/alignment.
   Neighbour-conditioned execution gain or synonym/context scores can accompany
   ambiguous cases. Shared-subroutine predictive models and conditional MI are
   later research additions requiring suitable identity, causal/prefix and sample
   support—not requirements before watching the first examples.

This accepts the genuinely important checks while using qualitative evidence to
resolve semantic questions that further scalar diagnostics cannot settle alone.

Local evidence: [final assessment](../../outputs/broad_search_assessment_20261014_final/report.md),
[new external brief](CHAIN_VERIFICATION_EXTERNAL_BRIEF_20261014.md),
[earlier assessed suggestions](GEMINI_SUGGESTIONS_20261013.md),
[actual sampler](../../studies/sequence_viz/broad_search_inputs.py),
[actual detector encoding](../../src/fpcode/workflows/sequence_search.py),
[physical estimator](../../studies/sequence_viz/assess_broad_search.py),
[photometry helpers](../../studies/session_building/jonathan/utils/photometry.py).
