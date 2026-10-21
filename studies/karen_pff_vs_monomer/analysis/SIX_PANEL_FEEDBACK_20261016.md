# Six-panel feedback: findings and bounded next steps

Date: 16 October 2026. Delivery branch: `codex/karen-seqnmf-delivery`.

The feedback supports pausing wider qualitative review and further K exploration.
Work first on the existing K10/K18 catalogues: correct the gallery scaling,
investigate handling that escaped the current exclusions, then distinguish
execution-ranking problems from overly broad detected families. K22 is parked,
not rejected. There is no new fit, reranking or movie generation in this audit.

## Feedback provenance

Input: `/home/fsp585/.codex/attachments/e60bdc5c-cacf-4ba0-8064-2cba4c76078b/karen_six_panel_labels (1).json`.
Its original bytes and SHA256, exact movie bindings, descriptor contributions and
reproducible diagnostic script are preserved under the delivery worktree's
`outputs/karen_six_panel_feedback_20261016/`.

All 18 new entries bind to the expected package, trial, family, sampling mode,
six-panel count and actual MP4 SHA256. They comprise 12 explicit coherence ratings
(3 Similar, 6 Mixed / unclear, 3 Dissimilar) and six comment-only entries across
17 distinct families. Unfinished `very ` for K10_F06/p100 is not a rating. The
K18_F13/random note comments on P80 coherence separately from the hand in random
panel 2. Embedded earlier three-panel labels remain separate; blank distinctness
and default false follow-up fields are not negative judgements.

These selected examples are diagnostic evidence, not an unbiased comparison of
K settings, independent observations for a significance test, or acceptance of
all occurrences in a positively reviewed family.

## Confirmed scaling defect in the gallery ranking

The user's pose-weighting suspicion has a concrete implementation explanation.
In `render_reference_review.normalize_descriptors`, latent IQRs are estimated on
descriptors already divided by `sqrt(4 * 320)`. Dividing those same descriptors
by those IQRs cancels their intended block normalization. Displacement and duration
scale estimation undo the block factor first; latent scale estimation does not.
Consequently latent squared distances are inflated by **1,280** relative to the
declared equal-block robust scaling.

The audit reconstructs all three modes for the 17 families appearing in the new
feedback: 306 panel positions, 305 unique executions, 92 fingerprinted sessions.
Reconstructed exemplar distances match saved distances within 1.14e-13. Latent
pose accounts for **99.61% of summed squared distance** on these checked pairs
(median pair contribution 99.77%; minimum nonzero pair 94.44%). This is a selected
gallery-pair diagnosis, not a whole-catalogue population estimate.

Applying only the mechanical factor correction to these fixed pairs reduces
latent's summed contribution to 16.78%. That counterfactual is not a reranking,
optimal weight estimate or evidence that corrected neighbours will look coherent.

The defect affects medoid/exemplar choice and p100/P80 example selection. The
seqNMF fits and detected call assignments use the grouped syllable input and are
unchanged by this ranking function. Uniform random selection is from the eligible
call population independently of distance, so its selection is not pose-ranked.
Existing movies and their human labels remain valid descriptions of those exact
files, but their p100/P80 ranks do not implement the advertised balance. Preserve
those files/labels and version the repaired gallery separately.

## Other observations and their implications

| Observation | Evidence and interpretation | Next action |
| --- | --- | --- |
| K18_F14/P80: all approach a wall, rear and descend | Strong positive anchor for the intended kind of theme. Its six examples all have 2/3 stage support, so globally requiring 3/3 would also discard this positive example set. | Preserve this anchor while calibrating ranking and any detection changes. Check fresh random executions before accepting the whole family. |
| K10_F00 mixes stationary grooming/sniffing and wall ascent; K18_F11 mixes running at a wall and rearing/descent | Physical execution can differ despite shared grouped-token support. Wall distance alone cannot separate two behaviours both occurring at the wall. | Examine ordered syllable semantics, movement and postural progression; assess detector permissiveness separately from ranking. |
| K18_F16/P80 is mostly coherent with panels 5/6 outlying | Useful near-positive anchor and explicit within-grid counterexamples. | Use these panels for calibration without treating all six as equivalent positive labels. |
| K10_F09 and K18_F05 vary between floor/centre and wall-associated executions | Current ranker uses onset-relative displacement, not explicit distance to arena boundaries. Locality is absent rather than merely assigned a small explicit weight. | Evaluate verified wall/corner proximity as an added context view. A shared movement family can retain spatial subtypes for intra-chain analysis; do not force every context difference into another chain. |
| K10_F06/P80 looks better than p100 | Compatible with the confirmed pose-dominated ranking and/or multiple execution subtypes. Percentile is proximity under a descriptor, not probability of behavioural correctness. | Revisit with corrected ranking before diagnosing K or altering the generator. |
| K22_F16/p100 looks heterogeneous | One comment-only p100 view cannot establish excessive K. Poor ranking and broad retrieval are alternative explanations. | Park K22; resolve these concrete issues at K10/K18 first. |

## K10_F03 duration spread is real in the saved calls

P80 core durations in panel order are **4.833, 4.700, 14.533, 4.167, 1.850 and
1.867 seconds**. Native endpoints, complete valid support and duration calculations
were checked; the range is not created by playback speed or panel padding.
The generator length is three grouped bouts, not a fixed number of seconds.

For the 14.533-second example, the raw segments are stationary/observe 25,
unassigned 87, stationary/observe 25. The middle raw syllable is 87 (unassigned)
in all six examples. Broad duration and an unassigned stage therefore deserve a
closer look at both the bout representation and detection policy. Do not invent
a semantic label for 87 or impose an arbitrary duration cap from this example.

Elapsed-time phase sampling in the ranker removes absolute duration from the pose
trajectory comparison; duration is retained in its own log-duration block. The
scaling bug made that duration contribution small relative to pose. Correcting
ranking may improve which durations are shown together, but cannot remove wide
duration variation from the assigned family. Assess meaningful timing variants
separately from unrelated sequences admitted by partial template matches.

## Confirmed unexcluded handling witness

K18_F13/random panel 2 binds to mouse 7, session
`2026-03-15T09_51_36_top_left`, native frames **[82750, 82850)** at 60 Hz,
**22:59.167–23:00.833** into the cropped source AVI. The existing eight-frame
contact sheet visibly shows a hand manipulating the tether/mouse during the
active core, including frames 82765, 82788, 82811 and 82833; this is not solely
the 0.5-second context around a valid execution.

All 100 core frames have nonnegative syllable labels and zero reviewed exclusions.
The session's persisted reviewed interval only excludes the initial [0, 15707)
frames. Previously verified mask application does not establish complete detection
of later handling. Pose confidence and human-presence exclusion are different
questions. The witness is also within the pooled fitting input view; hiding its
movie alone would not resolve that exposure.

Next inspect surrounding source frames to determine the full handling interval
and revisit existing within-session handling candidates. Do not infer boundaries
or prevalence from this one contact sheet. Apply any confirmed exclusions in a
new, fingerprinted input view, preserving original artifacts and native gaps.
Then rerun the bounded K10/K18 comparison on that view; there is no need to repeat
the entire 60-configuration capacity sweep to test the repair.

## Recommended sequence

1. **Repair and isolate mechanical issues.** Correct latent scale units with a
   substantive block-contribution regression check. Version all new rankings,
   exemplars and review files; preserve the exact original labels and media.
   Delimit the handling witness and assess related within-session candidates
   before final cleaned-input fitting or behavioural comparisons.
2. **Diagnose coherence at fixed K10/K18.** Start with corrected ranking alone
   on frozen calls to separate sampling from assignment. Then make a small number
   of explicit contrasts using ordered human-labelled syllable content and
   movement/turning, with spatial context as a separate addition. Report actual
   feature contributions rather than assuming equal nominal weights produce
   equal influence. Do not turn this into another metric-by-optimizer-by-K grid.
3. **Change retrieval only where justified.** The current detector accepts
   partial three-stage matches. Check whether negative examples are retained by
   an overly permissive criterion, while preserving the positive K18_F14 anchor
   and supported execution variants. Reranking alone does not improve purity of
   the full assigned family. Record retained time coverage, mouse support,
   within-family consistency and between-family duplication for each contrast.
4. **Return a small before/after review.** Reuse good, bad and near-good anchor
   families, plus fresh uniform random examples to check beyond optimized cores.
   Keep natural playback and the agreed six-panel p100/P80/random layout. A better
   p100 gallery alone is insufficient. Do not request another whole-catalogue
   labelling pass now.
5. **Resume capacity and downstream interpretation afterwards.** Compare K10 and
   K18 using the same repaired policy and retained families, then revisit K22 or
   other hyperparameters only if the corrected catalogue leaves a concrete gap.
   Group differences and inter/intra analyses follow interpretable families;
   they must not reward a setting for separating conditions despite poor coherence.

Completed checks are the feedback/media hash bindings, saved-distance numerical
reconstruction, 92 selected source-cache hashes, full native core support and
duration checks for 305 unique executions, and visual inspection of the single
reported handling contact sheet. These do not establish full-session handling
coverage or improved behavioural coherence. No maintained algorithm was edited;
no fits, exclusions, movie files, browser storage or original rankings changed.
