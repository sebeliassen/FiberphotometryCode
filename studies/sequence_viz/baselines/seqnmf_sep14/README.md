# 14 September presentation: reference recovery

Audit date: 15 October 2026. Presentation date is user-confirmed. **Status: saved
raw-state detector replay matched; original KPMS/native-frame binding verified.
Full presentation recovery remains partial.** See the subsequent
[executed replay and transfer audit](REPLAY_20261015.md).
This folder records provenance and the acceptance procedure, not another model
implementation or a claim that every analysis in the presentation is lab-approved.

## What has actually been recovered

The presentation combines several analysis steps. Preserve their separate inputs,
vocabularies, fit settings, detection rules and display selections.

| Supplied material | Recovered evidence | Confidence and remaining work |
| --- | --- | --- |
| Exact final Theme A/C filenames | Original MP4/PNG pairs; `render_1x3_final_exemplars.py`; panel mice match the script and surviving posters | Renderer traced; exact original recording/start/end selection manifest not yet recovered |
| `final_1x3_theme_B_dash_rank2_panels456.mp4` | Original MP4/PNG; six-panel `candidate_2x3_chain2_dash_rank2_exemplar.mp4`; `render_2x3_right_turn_chains.py` | Final three mice/HUD match parent panels 4–6; three sampled movie frames closely match that row. Exact final editing command remains unlocated |
| Branch variants 27.1%, 22.6%, 10.6% | Original episode CSV, saved model payload, hyperparameter JSON, 10 September fit log | Counts match; subsequent replay from saved W/H matches all 2,136 detections and animal summaries |
| Five-step coverage 14.1%, 46.1%, 76.2% | Original `clustering_rigorous_coverage_statistics.csv`, L1/100, L3/22, L4/14 rows | Exact tabulated values match; coverage is a recurring-sequence census, not evidence that one seqNMF fit explains these fractions. Exact census implementation still needs binding |
| Conservative Level 2 theme usage, 30 sessions | `fig_hidden_slide_level2_mann_whitney.png`; `generate_hidden_slide_mann_whitney_plot.py` | Original visually matches supplied figure, including six displayed p-values and dot/bar arrangement; raw rates/statistics not recomputed |
| Four-group mutual information vs Markov null | `pres_4_groups_vs_markov_null.png`; `generate_presentation_lag_figures.py`; saved `lagged_mi_results_1000s.pkl` | Original visually matches supplied figure and peak annotations; cached numbers/null generation not recomputed |
| Earlier Level 3 duration/self-repetition slide | `generate_sniff_clustered_slide_figures.py` matches unified-sniff labels, titles and stated settings | **Exact numerical artifact unresolved.** Both located duration PNGs with the expected basename show another version; do not call the supplied slide reproduced |

Original media and empirical payloads remain under ignored `compat/` storage. The
machine-readable inventory, SHA-256 hashes of 23 surviving artifacts, nine legacy
source identities, CSV recount and bounded video comparison are saved alongside
this document in [reference_manifest.json](reference_manifest.json), with the
runtime audit result also at:

`outputs/seqnmf_sep14_provenance_20261015/audit.json`

The read-only audit driver is alongside it as `audit.py`. It reads CSVs, hashes
payload bytes, probes MP4 metadata and decodes three frames from two existing
movies. It never unpickles, trains, generates movies, imports the old runners or
sends notifications. Its assertions passed. This is provenance evidence, not
empirical validation of all historical scientific claims.

## Recovered movie lineage

All three final movies are in:

`compat/syllable_generation/analysis/sequence_viz/movies_1x3_final_exemplars/`

| Theme | Original source-syllable path | Displayed mice, left to right | Selection lineage |
| --- | --- | --- | --- |
| A, rear/descent | `[15,14,4,3]` | 1666 KI saline; 1658 KI saline; 1694 KI amphetamine | Final renderer selects original candidate panels 2,4,6 |
| B, right turn/dash | `[2,0,13,2]` | 1680 WT amphetamine; 1775 WT saline; 1771 KI saline | Parent six-panel rank-2 suite, panels 4,5,6 |
| C, sniff/rest | `[8,32,8,32]` | 1775 WT saline; 728 WT saline; 1773 WT saline | Final renderer selects panels 1,2,3 |

The qualitative renderer retrieves exact observed source-syllable strings. It
ranks executions using a PCA representation of resampled latent pose and speed,
within-family/background density ratios, and nearest neighbours with animal
diversity, followed by the documented panel selection. The renderer does **not**
load seqNMF W/H or constrain these exact-string candidates to a saved seqNMF call
catalogue. The provenance link between displayed themes and discovered factors
must therefore be stated explicitly; a representative movie is not itself proof
that its interval was a particular factor detection.

Historical resource loader defaults name KPMS model `amph_prod_k150k_run_12`,
`results.h5`, `video_frame_indexes.npz` and original video files. That is the
**upstream syllabizer**, not the seqNMF model identity. The subsequent replay binds
all historical bout starts/ends and labels to the archived KPMS results and exact
native VFI. Bout end coordinates are inclusive; durations count model rows at
40Hz. Native projection uses existing reviewed Run12 clock evidence with its
timestamp-validation limits. Do not transfer these conventions automatically to Karen.

The B parent movie is in `movies_2x3_right_turn_chains/`. Its lower 378 image rows
match the final movie's body at sampled frames 0,73,146 with mean absolute decoded
channel differences 0.786,0.930,0.768 on a 0–255 scale. Differences are compatible
with re-encoding. Together with the matching mice, HUD and source sequence this
is strong parent-row evidence, not recovery of an unknown editing command or an
exhaustive all-frame identity test.

## The saved seqNMF reference behind the branching example

`optimized_seqnmf/optimized_seqnmf_model_results.pkl` survives alongside its CSVs,
settings and log. Source `optimized_seqnmf/fit_optimized_seqnmf_model.py` writes
W, H, motifs, episodes and hyperparameters to that bundle. Source and run log agree
on K=10, L=3 bout lags, lambda_ortho=1.0, 80 iterations, seed 42, JAX GPU, raw
100-state one-hot input and first 50,000 concatenated bouts. The source loads 30
sessions but the retrieved saved catalogue covers **10 saline sessions/10 mice**
(5 WT/5 KI), 2,136 episodes. “Across all animals” log text does not establish that
all 15 mice contributed to this truncated training prefix.

Factor 1 has 350 episodes. Its source variants `[14,4,3]`, `[14,4,17]`, `[14,4,6]`
occur 95,79,37 times. A denominator detail matters when transferring the branch
analysis: only **307** of the factor's 350 episodes start `[14,4]`. Thus the slide's
27.1/22.6/10.6 are fractions of **the whole factor catalogue**. Strict probabilities
conditional on that root would be 30.94/25.73/12.05% within this catalogue. Neither
automatically estimates continuation probabilities among *all* root occurrences
in the cohort. Preserve the historical numbers and distinguish these estimands
in the new analysis.

The quantitative slide generator is another recipe: unified-sniff 20-token
mapping, K=10, L=3, lambda_ortho=.005, 150 iterations, seed 42, CPU, first up to
60,000 valid post-injection bouts. Its exact saved numerical result remains
unbound. Located same-name PNGs show saline TV .290/p .0140 and amphetamine
TV .353/p .0044; the user's slide shows .253/~.0211 and .312/.0093, with different
chain labels. Filename identity must not override this mismatch.

The additional Level 2 plot counts the three specified theme paths after synonym
mapping and another run-length compression; it is not a new fitted seqNMF model.
Its generator normalizes by recording duration using a hardcoded 40 Hz convention.
The MI figure plots cached real and first-order-null curves; it supplies temporal
context, not a value for L (which here counts bouts) or proof of a particular fit.
Historical interpretive titles are retained as evidence, not independently
endorsed mechanistic conclusions.

## How the baseline will be accepted before the Karen adaptation

1. **Bind exact identities.** Pin source revision/blob, numerical implementation,
   KPMS results, video-index mapping, input session order and training prefix,
   vocabulary/annotation map, seed/backend, W/H, call catalogue and reference
   tables. Resolve the quantitative PNG version or explicitly limit recovered
   scope to the matched reference components. Never substitute the unrelated
   K6/L5/lambda .5 file named `seqnmf_legacy_baseline.json` just because of its name.
2. **Replay saved weights first.** Recover the historical input transformation and
   detector, project the original data without refitting, and compare factor
   templates, call identities, observed strings, counts, source/native boundaries
   and animal summaries. Pin half-open endpoints, boundary/gap policy, exposure,
   masks and overlap definitions. The original fitter concatenates sessions;
   audit boundary-spanning windows rather than silently assuming they are safe.
3. **Check the displayed executions.** Recover each movie's recording key,
   model-row start/end, native frame mapping, stage boundaries and selected
   exemplar/neighbour identity. For B keep the parent and final-edit relationship.
   If those intervals cannot be recovered, label the movie lineage partial rather
   than claiming an exact rerender.
4. **If a refit is necessary, separate it from replay.** Freeze the recipe and
   comparison criteria before running. Match normalized factors allowing
   permutation, scale and allowable temporal shift; compare actual detected
   executions, multi-animal support and animal-level summaries as well as template
   similarity. Report unmatched/split/merged factors. Set numerical tolerances from
   backend precision and the frozen comparison target, not from a desired p-value.
   Several seeds can assess stability if needed; selecting a visually convenient
   seed does not reproduce the original model.
5. **Separate replication from corrections and transfer.** Record historical
   outputs faithfully, then document changes required for current clocks,
   exclusions, valid exposure, overlap and branch denominators. A corrected Karen
   adaptation is a new result. It should preserve the scientific procedure and
   interpretable chain-family concept, with Karen's own syllables and fits.

Use explicit statuses: artifact recovered; renderer traced; model/data bound;
saved-model replay matched; stochastic refit assessed; Karen adaptation reviewed.
The raw-state model/data binding and saved-detector replay are now established.
Independent activation inference/refitting, exact final movie interval manifests,
the unmatched quantitative-slide version and Karen scientific review remain open.
Refactor is now preserved at 6c179ae, with a separate
`codex/karen-seqnmf-delivery` worktree. The initial artifact audit did not change
Git state; the subsequent authorised isolation/replay step did. No model training
or search was needed for either.
