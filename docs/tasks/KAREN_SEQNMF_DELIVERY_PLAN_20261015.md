# Karen seqNMF delivery: two-week proposal

Date: 15 October 2026. **Proposal, not an executed branch migration or empirical run.**

The user must finalise in approximately two weeks. Their intended core is the
working seqNMF workflow presented to the lab on **14 September**, applied to
Karen's data, with small interpretable intra/inter-chain extensions. Preserve the
broader framework/search work and clearly distinguish lab-reviewed methodology,
the new Karen adaptation and exploratory research. “Branching” in this request
primarily means Git branches for preserving the work, not another behavioural
branch-discovery project.

## Concrete reference recovered from the supplied slide

**15 October provenance update:** Further supplied slides and exact MP4 names
identify a composite presentation workflow, not one interchangeable fit. The
original three final movies, raw-state saved model/log/episode CSV, exact branch
counts, coverage table, Level2 theme-usage figure and four-group MI figure survive.
See [reference recovery](../../studies/sequence_viz/baselines/seqnmf_sep14/README.md)
for artifact/source identities and the acceptance procedure. In particular,
the saved branching model uses K10/L3/lambda1.0/80iterations, whereas the unified-
sniff quantitative generator below uses lambda.005/150iterations. The A/C movie
renderer uses exact-string candidates and selected neighbours rather than loading
that seqNMF fit. The exact final B edit is associated with the parent six-panel
rank2 movie by matching mice/HUD and three sampled decoded frames; its editing
command is not recovered. Both located duration PNGs with the expected basename
differ numerically from the supplied quantitative slide. Treat the generator
below as a candidate recipe, not an already reproduced artifact/model.

The supplied Quantitative findings slide shows Level3 chain duration, self-
repetition and animal-level group-profile separation. The legacy generator
matching both figure titles is:

`codex/legacy-consolidation:studies/behavior/sequence_viz/generate_sniff_clustered_slide_figures.py`

Legacy branch checkpoint: `b83298fcfb15ffd5ce3f6dbbd863b1f8e98af03e`.
Generator blob: `1959b1d46558761048917db490b9fd2fc503fe8b`.
The current relocated source is
`studies/sequence_viz/generate_sniff_clustered_slide_figures.py`; it has a different
blob, so do not assume the current file is byte-identical to the presented code.

The recovered variant is **Level3 with unified sniff categories**, not a generic
later22-state/L4 configuration. Its mapping has19named groups plus an unlabelled
tail (20tokens). The source fits10factors with3bout lags, lambda_ortho0.005,
150iterations, seed42, CPU, first min(60,000,available) valid post-injection bouts.
It imports `decode_episodes_in_session` from the Level3 extractor. Outputs include
`fig_level3_slide_duration_vertical.png` and
`fig_level3_slide_self_transitions.png`.

These source/figure matches identify a strong baseline candidate. The exact
historical input bundle, matching saved figure/model and numerical reference
still need binding before tagging a reproducible baseline. The user confirms
presentation date14September; the presentation MD labelled14July contains the
exact supplied movie links, but its filename must not override that. A small
caption check belongs in the reference audit: the
slide bullet reads salinep0.21 while its plot legend reads approximately0.0211.
Do not select a version by obtaining a preferred p-value.

The historical script executes loading/fitting/figure writes at import and
imports notification helpers. Inspect it as text; recover the recipe through
an explicit scoped entry point rather than executing the old presentation script
unmodified. Preserve source and claims as presented; document necessary clock,
mask, overlap, exposure or denominator corrections separately.

## Recommended branching strategy: one new branch

| Ref | Role |
| --- | --- |
| `main` | Existing mainline; preserve its current role during the deadline effort. |
| `codex/modular-refactor` | Current platform plus ongoing broad framework/search research. Preserve current work here before branching. |
| `codex/legacy-consolidation` | Historical reference; inspect/recover specific recipes, not the delivery working branch. |
| **`codex/karen-seqnmf-delivery`** (proposed new) | Two-week delivery, based on a checkpoint of refactor. Scoped to reference seqNMF, Karen adaptation and a small fixed intra/inter set. |

Use a separate Git worktree for the delivery branch. This allows future work on
refactor while delivery remains reproducible and avoids repeatedly changing the
shared directory/branch. Share existing immutable data via explicit paths and
use distinct output directories; do not duplicate raw recordings or models in Git.
`FP_DATA_ROOT`, if needed, is the mirror root containing `data/`.

Before creating the branch, checkpoint the current working tree. The inspected
state is HEAD36473e4on`codex/modular-refactor`, with269status entries including232
untracked. A branch at HEAD alone would not preserve those changes. Inventory and
save the maintained source/config/docs without discarding unrelated edits; keep
ignored empirical artifacts externally and record their paths/fingerprints.
No blanket clean/reset, no raw-data/model staging. This plan has not committed,
staged, switched branches or created worktrees.

Branch delivery from the checkpointed **refactor**, retaining repaired shared
infrastructure. Recover and pin the historical recipe there; do not transplant
the complete old repository or duplicate its shared algorithm implementations.
Promote only needed reviewed fixes between branches. Avoid broad framework merges
into delivery during the final analysis/write-up. Tag the matched reference and
final delivery once their specific evidence is established. Such tags identify
exact source states; they do not confer lab approval on all code in the tree.

## Baseline folder and scientific status: complementary to the branch

Use a small proposed folder such as
`studies/sequence_viz/baselines/seqnmf_sep14/` containing:

- README: what was presented/reviewed, date, scope and limitations.
- Frozen recipe: vocabulary policy, input identity, fit/detection/overlap rules,
  seed and physical/bout time conventions.
- Reference manifest: source revision, external input/model/result paths and hashes.
- Expected small reference outputs plus a scoped reproduction/validation command.
- Status record distinguishing lab-reviewed reference from unreviewed adaptations.

This is an evidence/configuration bundle, not another implementation tree. A
`lab-proven/` copy of source would drift and create ambiguity about which version
owns fixes. A folder by itself also does not isolate changes to shared code;
the branch/worktree supplies that isolation. One maintained implementation and
explicit profiles keep ownership clear.

| Scientific status | Meaning |
| --- | --- |
| Lab-reviewed reference | The specific Jonathan presentation workflow the user identifies; numerical recovery and exact artifact binding must be recorded. |
| Karen adaptation | Same methodological approach with Karen-specific inputs/ontology, fitted templates and validated execution; a new analysis, not automatically lab-approved. |
| Inter/intra extension | Small interpretable readouts on frozen chains, separately documented and checked. |
| Exploratory research | Broad search, HSMM/grammar alternatives, input contrasts, new rankings/verification investigations; preserved on refactor, outside the deadline's required story. |

Code tests establish their tested behavior, not biological validity or lab
acceptance. Label status per method/configuration/result rather than applying a
single “validated” label to an entire branch. Previously viewed data remain useful;
describe the exploratory history briefly rather than implying unusability or a
fresh confirmatory test.

## Scientific deliverable and reuse

Core story: a lab-reviewed sequence-discovery approach identifies interpretable
Karen chain families; occurrence, execution and simple organisation measures
describe PFF/monomer tendencies. PFF/monomer is a treatment/cohort contrast, not a
genotype contrast. Significant effects are not a delivery requirement.

Reuse existing Karen frame caches, reviewed exclusions, annotation groundwork,
seqNMF pilots and organisation code after checking provenance against the selected
reference. `studies/karen_pff_vs_monomer/analysis/GENERAL_CHAIN_ORGANISATION.md`
documents an executed14-library organisation pass. Its outputs are dated evidence,
not automatic final results for a different recipe or exclusion policy. Do not
assume all99recordings are eligible for every intended endpoint or current split.

1. Freeze the reference pipeline and a practical Karen recipe. Transfer the
   procedure and physical definitions, **not Jonathan's numeric syllable IDs,
   learned W templates or chain numbers**. Establish Karen's own vocabulary
   mapping and original-bout/missing-frame/handling support. Preserve the current
   animal splits/reserved animals unless explicitly revised.
2. Fit/project the scoped Karen seqNMF analysis as needed, using a small fixed
   recipe/check set rather than restarting a broad optimisation campaign.
   Check reproducibility, multi-animal support and physical/native-time validity.
3. Perform a targeted qualitative review of the resulting seqNMF catalogue using
   the agreed p100/P80/random movie grouping. The existing147movies span multiple
   methods and do not all need labelling for this seqNMF-focused deliverable.
4. Main readouts: chain recruitment/time occupancy and an animal-level chain-
   profile comparison. Distinguish summed overlapping episode seconds from
   union occupancy, and normalize by valid exposure where appropriate.
5. Intra-chain: a small set of duration/pacing, movement/turn execution and
   observed sequence variants within identified families. Use actual observed
   strings, not lag-wise argmax labels as empirical executions.
6. Inter-chain: repetition/switching/return or conditional transitions, with
   explicit opportunities, gap barriers and a frequency baseline. Reuse completed
   organisation machinery, not a new latent assembly model. Shared-prefix branch
   examples can illustrate intra variation if already supported; they are optional.
7. Compare groups at the animal level with repeated waves handled explicitly.
   Use well-supported untreated observations for the main comparison; sparse late
   waves and L-DOPA/wave/age mixtures need separate interpretation. Report effect
   sizes/uncertainty, support and a bounded multiple-comparison policy. Select
   interpretable measures before examining their best group p-values.

DA remains optional Jonathan WT-only context; Karen has no DA modality in this
analysis. It is not a dependency for completing Karen delivery. Alternative
method championships/new grammar discovery are also not dependencies.

## Two-week schedule and stopping rules

| Days | Result |
| --- | --- |
| 1–2 | Preserve/checkpoint refactor; create delivery worktree; bind slide baseline, recipe and reference evidence; freeze Karen inputs/plan. |
| 3–5 | Working scoped Karen seqNMF result, source/native-time checks and preliminary chain catalogue. |
| 6–7 | Targeted movies/labels; freeze the core catalogue and main story. |
| 8–10 | Bounded inter/intra readouts, animal-level comparisons and necessary sensitivity checks; freeze analyses. |
| 11–14 | Final figures, methods/limitations, reference/run provenance, presentation/write-up and buffer. |

Complete the core Karen application even if an optional extension proves poorly
supported. Keep null/weak results visible. After day10, focus on corrections and
communication rather than adding methods or expanding searches. A final handoff
should include one exact recipe/run manifest, reproducible tables/figures, the
reviewed chain catalogue/videos, concise methods and a status/limitations record.

## Immediate next action

Agree the delivery boundary, then checkpoint the current source and create the
new branch/worktree. Verify the candidate presentation generator against stored
figures/inputs before declaring the baseline recovered. No empirical runs or Git
mutations were launched while preparing this proposal.
