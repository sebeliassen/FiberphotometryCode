# Karen coherence repair and the route to a refreshed catalogue

16 October 2026; delivery worktree, `codex/karen-seqnmf-delivery`. Source is an
uncommitted candidate based on 736f3ca. The preceding feedback report describes
the diagnosis before the repairs below; it is not the current completion state.

## Human preference and catalogue follow-up

The user reports that **ranking repair only is best by far**, while the
phase/cleaned alternative looks comparable to the earlier poor version. This is
an overall comparative judgement, not an invented rating for every family or
sampling mode. The exact feedback and scope are saved in
`qualitative_preference_20261016.json` under the analysis output.
Internal nearest-neighbour gains in an alternative representation do not override
this qualitative judgement of the actual executions.

The preferred view uses the frozen original models and the corrected legacy
ranker: 32 elapsed-time phase samples, original continuous latents, onset-relative
displacement, relative heading, and log total duration with the repaired block
scaling. No whitening, bout-aligned landmarks, velocity features or semantic
agreement term is added to that ranker. Those alternatives remain experimental.

The earlier alternative changed both the representation and the fitted models.
Its poorer appearance does not establish that handling exclusions are harmful.
The next catalogue therefore restores the preferred ranker on the cleaned K10/K18
models, preserving the handling repair. **This combination awaits qualitative
review.** Six original ranking-repair anchor videos remain linked as controls.
Check those before investing in the wider catalogue pass.

Catalogue output:
`outputs/karen_pooled_p80_review_20261015/rank_repaired_catalogue_20261016/`.
The `--catalogue` scope in `render_coherence_comparison.py` prepares all supported
cleaned factors: 28 families, 84 movies, six executions per movie, all three
sampling modes. The viewer defaults to the top ten per K, with all others
accessible, under a distinct package/storage identity. No new fit or parameter
search is needed. The exact source of the previous small-review renderer is
preserved in `source_snapshot_before_catalogue/`, verified against its manifest.

The catalogue is **complete and ready** at
**http://localhost:8766/rank_repaired_catalogue_20261016/** through the existing
Mac SSH tunnel. Its landing page provides the six control comparisons and the
catalogue viewer. Twenty most-common P80 movies (top ten per K) take 90.18 seconds
once / 15m02s at ten plays per file, before notes. All 28 P80 files take 127.10
seconds once; all modes together take 424.85 seconds. Use P80 first and random
checks for families being carried forward; watching all modes is unnecessary.

Catalogue audits passed 84 grid/504 component hashes, full native paths and
latent support, 504 overlay-colour pixel checks (maximum error45/tolerance65),
418 finished-panel darkness checks and natural grid lengths. All 504 selected
identities/distances/percentiles match the `clean`/`corrected_legacy` rank rows;
all 28 uniform random samples were independently reproduced. Browser tests
passed 28-family completeness, top-ten default, all K18 families accessible,
mode separation, movie-hash binding, reference export and old-storage
preservation. Anchor links returned HTTP200 and MP4 Range206 matched exact bytes.
The final candidate passed 878 CPU tests (six existing warnings), architecture,
lint with no new debt, and all 146 folder maps. No new fit was run in this pass.

The selected recipe and provenance are explicit in the catalogue root's
`selected_ranking_recipe.json`, `selected_rank_audit.json` and
`browser_validation.json`; media audit is in `cleaned_ranking_correction/`.
The catalogue's distinct storage key is
`karen-ranking-repair-catalogue-20261016-v1`. Your preference remains scoped to
the previously viewed original-fit examples until this cleaned-fit control is
reviewed; generation and numerical validation do not confer human acceptance.

## Completed work

The latent-IQR units defect is corrected in `render_reference_review.py`.
Independent regression expectations check actual block distances and invariance
to latent feature units. Original models, rankings, movies and exported human
labels remain available. Revised galleries have distinct package/storage keys,
exact movie hashes and read-only copies of the original six-panel notes.

Two new pooled grouped-69 seqNMF models were fitted on the versioned input view:
K10 and K18, length three grouped bouts, lambda 0.3, seed 43, 80 iterations, the
recovered historical update rule and unchanged detection policy. Both include
all 99 recordings, all 21 mice and all five waves, including L-DOPA. No new KPMS
fit, animal split, genotype/treatment selection or DA selection was performed.

| Cleaned model | Detected calls | Union coverage of valid modelled time |
| --- | ---: | ---: |
| K10 | 18,564 | 18.78% |
| K18 | 32,749 | 31.14% |

Coverage is descriptive and counts overlap once. It is not percent behaviour
explained or a judgement of coherence. Original-to-cleaned factors were matched
using normalized template cosine and Hungarian correspondence. Median matches
are 0.99990 (K10) and 0.99999 (K18); minima are 0.68491 and 0.84445. Factor numbers
are not proof of biological identity, and weaker matches require review.

## Handling repair

Existing confidence candidates bracketed two removal/empty/replacement episodes
in `2026-03-15T09_51_36_top_left`. Saved native-frame sheets show free-mouse frames
around the boundaries and hand/cage/empty-arena frames within the intervals.
Conservative analysis exclusions are [56800,62893) and [74166,83015), native
60 Hz. These are buffered analysis boundaries, not exact hand-entry/release times.

The additional excluded elapsed span is 249.03 seconds, most already unmodelled;
**1,096 previously labelled frames / 18.27 seconds** are newly removed from
fitting eligibility. The new view copies only the affected recording and links
unchanged caches. Native rows, frame identities and existing exclusions remain.
No saved new call crosses either added interval.

Six other large residual candidates showed free mice near walls/corners at the
sampled times, so no handling masks were added to them. This is targeted review,
not continuous validation of those intervals or exhaustive handling detection.
There were 82 residual confidence candidates before these two repairs; 74 were
not visually sampled in this pass. Confidence alone cannot identify handling.

## Fixed execution-similarity contrasts

A recent MD prescribing numerical weights was not located. The user explicitly
allowed judgement. The available pasted guidance supports separate posture,
movement and context views, robust scaling, known semantic evidence and retaining
execution variants. The following are bounded starting policies, not optimized
weights or implementation of every proposal in that guidance.

| View | Nominal block weights | Purpose |
| --- | --- | --- |
| Corrected legacy | 25% each: latent posture, onset-relative displacement, heading, log total duration | Isolate the units correction in the established 32 elapsed-time phase descriptor |
| Bout-aligned physical | Posture 45%, movement 45%, tempo 10% | Separate ordered postural/movement progression from bout duration |
| Bout-aligned with semantics | Posture 45%, movement 35%, tempo 10%, known semantics up to 10% | Add modest agreement of human-labelled categories where both executions have known labels |

The two bout-aligned views use onset/middle/offset landmarks in each of the three
actual grouped bouts. There is no video warping, elastic alignment or change to
native event endpoints. Tempo comprises three log bout durations. Movement uses
log1p(speed / 1 native px/s) and asinh(signed angular speed / 1 rad/s). Centroid
and unwrapped heading smoothing, derivatives and phase samples never bridge
missing/excluded native support. Left/right direction is retained.

Posture uses native-unit robust scales and regularized covariance whitening
within its ten-channel block (diagonal shrinkage 0.1). Covariance calibration is
bounded at five robust units; application values are not clipped. There is no
automatic speed regression, latent mirroring or removal of potential phenotype.

One frozen, equal-mouse calibration sample contributes 512 calls per mouse from
the original K10/K18 pool, seed 101, shared across original and cleaned cases.
After feature-unit scaling, each physical block is also normalized by its median
mean-squared distance on 8,192 fixed cross-mouse calibration pairs, seed 2026.
Feature-coordinate counts do not multiply block weights. Nominal weights remain
distinct from observed distance contributions; both are recorded.

Semantic evidence compares ordered human annotation categories at mutually known
landmarks. Raw `x`/unassigned labels are missing evidence, never one shared class.
The semantic weight is multiplied by the fraction of mutually known landmarks;
unavailable weight redistributes to physical blocks. This view retains a physical
medoid, then supplements neighbour distances with semantics; it is not the exact
medoid of the combined missing-aware semantic distance.

Across the original and cleaned family profiles, posture contributes 38.44–65.76%
of summed exemplar distance in the semantic view. The actual contributions sum
to one within 1e-12. These figures are measured diagnostics, not a target that
every family must have identical contributions.

**Spatial context has no numerical weight in this pass.** Verified arena wall/
corner boundaries were unavailable. Screen position or a mouse's occupancy
envelope is not silently substituted for calibrated arena geometry. Context can
be added as an explicit sensitivity view/subtype descriptor after calibration.
This remains a limitation for wall-versus-floor distinctions.

The new weights affect execution similarity, ranking and diagnostics. seqNMF
still fits grouped syllable sequences. Reranking cannot make all assigned
occurrences coherent; any subsequent family split/merge/retrieval change must
be assessed separately with retained coverage and mouse support.

## Quantitative diagnostics and evidence

Three views were evaluated on two frozen models and two cleaned models, without
another hyperparameter or optimizer search. Outputs contain 100,138 eligible
execution rows, 300,414 rank rows and 168 family/profile summaries. Encoded-video
endpoint failures are explicit in `unavailable_calls.csv`, without shifting or
truncating their native calls.

Cross-mouse five-neighbour diagnostics use at most 64 examples per family,
preferentially balanced across mice; final per-family summaries average mice.
Neighbours must come from other mice. Overlapping calls in one candidate
recording cannot cast multiple votes for the same physical event. Physical
cross-family overlap is reported separately in `family_time_overlaps.csv`.
These diagnostics depend on the chosen representation and are not independent
human-coherence labels; adding semantics can improve agreement partly because
the generator already uses a human-grouped vocabulary.

Detector sensitivity at fixed calls/NMS also checks error cutoffs 0.5, 0.6 and
0.68. For the positive original K18_F14 anchor, 0.5 retains only 74/3,066 calls,
whereas 0.6 retains all 3,066. Globally demanding perfect three-stage matches
would remove the user's six positive P80 examples too. No stricter cutoff was
automatically installed or chosen from group effects.

An independent persisted-call audit verifies all 51,313 new calls: saved hashes,
all 99 input label/mask fingerprints, tokens, native endpoints, durations,
contiguous support, activation thresholds, NMS spacing, numerical matching errors
and stage support. Maximum error discrepancy is 1.59e-7, tolerance 2e-7.
This does not establish semantic correctness or held-out generalization.

## Small review, then a new catalogue review pass

Open **http://localhost:8766/coherence_repair_20261016/** on the Mac using the
existing SSH forward. The comparison page links original P80 videos and two
revised viewers:

- `ranking_correction/`: corrected ranking on the frozen original models;
- `cleaned_semantic/`: cleaned pooled models with the bout-aligned semantic view.

Six original anchor families are included: K10_F00/F03/F06 and
K18_F11/F14/F16. The cleaned viewer retains actual factor IDs and records the
matched original family separately. Original six-panel notes are read-only
reference; revised labels identify their new package, model and movie hash.

Both viewers have K10/K18 and p100/P80/random controls. All 36 new movies contain
six executions in two rows of three, native 60 Hz at natural pace, fixed-onset
camera, raw-syllable colour dots and ordered durations. Crop scale is shared
between the two revised versions for each matched anchor. Uniform random uses
a fresh deterministic seed, distinct from the original review's seed.

The two P80 sets total **72.37 seconds once / 12m04s at ten plays per file**.
All new modes total 207.17 seconds once; there is no need to watch every mode
initially. Original reference playback and note-taking are additional.

Start with P80 comparisons, check random for families being carried forward,
and inspect p100 where the core remains uncertain. The cleaned alternative
changes both input exclusions and the similarity descriptor, so that contrast
alone does not causally isolate those two changes. Frozen-model quantitative
profiles provide the corresponding separation; targeted additional movies can
resolve any specific ambiguity.

The purpose is to check the repair and choose a representation for a refreshed
catalogue pass, not to stop at these six families. Next render that catalogue
with the selected recipe, most-common-first and top ten per K by default. All
28 supported cleaned K10/K18 families remain accessible. No family or K setting
is automatically accepted because its exemplar movie looks better. Group and
inter/intra-chain interpretation follows the catalogue review.

## Locations, runtime and checks

Analysis and logs, relative to the delivery worktree:
`outputs/karen_coherence_repair_20261016/`.
Review: `outputs/karen_pooled_p80_review_20261015/coherence_repair_20261016/`.
The original six-panel package remains its sibling `six_panel_20261016/`.

The runs used nohup with 45-minute timeout, two numerical threads, an 8 GiB RSS
guard, 80 GiB free-disk floor and 6 GiB output cap. Fits used the available RTX4090
with 25% JAX allocation; render used four CPU workers and two encoder threads.
One failed rendering preflight lacked the component directory; it is preserved
in the sibling `coherence_repair_20261016_failed_directory_preflight`. The retry
fixed directory creation without changing selection or scientific parameters.

Final validation: 878 CPU tests (six existing deprecation warnings); architecture,
lint with 3,078 reviewed diagnostics/no new debt, and 146 folder maps passed.
Independent media audits passed all 36 grid/216 component hashes, full selected
native paths/latent support, 216 source-colour pixel checks (maximum error 39,
tolerance 65), natural grid lengths and 178 finished-panel darkness checks.
Full encoded frame decoding was performed by the renderer. Isolated browser
tests passed model/mode switching, movie-hash-bound notes, old-storage preservation
and export with original references. Both viewers passed HTTP200 and exact-byte
MP4 Range206. Mac-side continuous playback was not tested by the agent.

Evidence: `final_quantitative_audit.json`, `browser_validation.json`,
`cpu_checks.log`, each viewer's `final_media_audit.json`, `feature_recipe.json`,
launch recipes, `handling/review_decisions.json`, and the saved call/rank tables.

## Subsequent catalogue feedback: goal not met (2026-10-16)

The user's `karen_six_panel_labels (4).json` partially reviews the cleaned-fit
catalogue under the preferred corrected legacy ranker. Its14entries/13families
bind to exact movie hashes:2Similar/6Mixed/1Dissimilar/5comment-only;13P80+1p100,
no random reviews. This does not accept the catalogue or establish a best K.

The user explicitly says the goal is not yet met. The useful result is a set of
positive witnesses and specific within-family distinctions: rest versus wall
investigation, rearing versus grooming, differing p100/P80 modes and a boundary
question. Ranking repair improved selection in the prior comparison; that is
separate from full-family coherence.

Read-only follow-up found80/84 reviewed panel positions match only2/3 template
stages. Different missing stages can admit different actions, but strict3/3 is
not the answer: it removes all6 positive K10_F09 examples. Postfiltering all saved
calls to3/3 retains5.30%(K10) or4.38%(K18) of current union-covered time. Splitting
by missing-stage pattern would also fragment the user's similar K18_F13 core.

Existing449D distances do not consistently reflect these reported distinctions:
between/within-rest mean squared distance ratio .801 for K10_F02; core/exception
separation ratio1.001 for K10_F06. K18_F16 separates better(ratio3.001) but still
assigns both modes to one family. These tiny selected-panel diagnostics are not
held-out tests. This ranker has no explicit velocity block, so velocity overweight
is not established. Historical retrieval uses fixed3bouts, not general insertion/
deletion alignment. Exact native K10_F04 panel3 context/core intervals were audited;
which physical prefix the user meant remains unresolved. No manual trimming.

Evidence and next bounded diagnosis plan:
`outputs/karen_catalogue_feedback_20261016/README.md`, source JSON/SHA audit,
panel support audit, full saved-call support distribution, human-defined pair
distances, boundary witness and reproducible `diagnose.py`. No models, rankings,
movies or production calls changed. Preserve corrected legacy ranking as baseline;
separate membership, resemblance and boundary checks while retaining good variants.
Any refinement must report coverage and mouse-support costs and obtain fresh random
review before acceptance. A small targeted comparison should precede another
catalogue pass; no new broad hyperparameter search is implied by this feedback.
