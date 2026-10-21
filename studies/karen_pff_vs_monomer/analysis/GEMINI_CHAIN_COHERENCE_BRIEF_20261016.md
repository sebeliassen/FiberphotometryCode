# Advice request: finding meaningful behavioural chain families

Prepared 16 October 2026. This is a request for scientific and methodological
advice after inspecting the actual local implementation and evidence. Please
challenge our diagnosis as well as propose improvements. We have not achieved
the behavioural goal, and a successful technical audit does not establish it.

## 1. Where to work and what to read first

The active delivery worktree is:

```text
/home/fsp585/Desktop/FiberphotometryCode/.worktrees/karen-seqnmf-delivery
```

Call this directory **W** below; all paths without a leading slash are relative
to W. The parent repository is `/home/fsp585/Desktop/FiberphotometryCode`.
The delivery branch is `codex/karen-seqnmf-delivery`; the parent is currently
`codex/modular-refactor`. Relevant changes are uncommitted, so inspect the working
files rather than assuming HEAD contains the current implementation. Other
branches and older reports contain different approaches and dated checkpoints.

Please start with these, in order:

1. `docs/tasks/CURRENT.md`, especially its newest section. Older sections below
   it are historical, even when they say "complete" or "ready".
2. `outputs/karen_catalogue_feedback_20261016/README.md`: latest partial human
   feedback, concrete failure cases, diagnostics and their limitations.
3. `studies/karen_pff_vs_monomer/analysis/reference_seqnmf.py`: actual fitting
   objective/update and `retrieve` implementation. This is essential.
4. `studies/sequence_viz/recreate_generator_videos.py`: `execution_descriptors`,
   `rank_executions`, `rank_exemplar_neighbors`.
5. `studies/karen_pff_vs_monomer/analysis/render_reference_review.py`:
   `normalize_descriptors`, including the repaired scaling policy.

Use the more detailed file map in section8 as needed. The ignored `outputs/`
directories are available locally but are not Git-tracked. If something cannot
be opened, identify that limit rather than filling in its contents from this brief.
For this consultation, please inspect and advise; do not change source, run new
fits or launch a broad search. We are asking for a defensible next experiment.

## 2. The actual downstream goal

We want a compact, interpretable catalogue of recurring behavioural routines
that covers many distinct kinds of behaviour while keeping similar executions
in the same family. Examples might include rear–sniff–descend–depart, wall
investigation, or an ordered locomotor routine. These are examples of the desired
resolution, not a fixed list the model must reproduce.

Ultimately we want to compare conditions using:

- **Chains themselves:** occurrence, time occupied, duration and other properties.
- **Intra-chain analysis:** variants/modifications inside a shared family,
  such as a walk-away versus run-away ending or a different execution tempo.
- **Inter-chain analysis:** ordering, repetition and combinations of families.

Those analyses require a meaningful common family definition. We must avoid
both combining unrelated stationary activities and fragmenting almost-identical
versions of the same routine. Speed, turning and location can distinguish useful
variants; making them universally irrelevant would lose downstream information.
Conversely, matching speed or near-zero displacement alone must not equate rest,
grooming and active sniffing. The distinction between family identity and an
interesting within-family variant is one of the difficult unresolved questions.

The original motivation was that exact n-grams fragment legitimate variants and
can be dominated by frequent micro-loops. Requiring exact syllable strings would
reintroduce that problem. However, fuzzy matching must preserve the ordered
behavioural core. Reconstruction, coverage, stability and visual similarity are
different properties; none alone defines a successful catalogue.

This is a time-limited delivery effort following a September lab presentation.
Prefer a bounded, interpretable improvement of the existing pipeline when viable.
Please identify explicitly if you think the generator itself is unsuitable;
do not assume every failure can be repaired by another distance function.

## 3. What data we actually have

The active Karen analysis uses one pooled generator per configuration fitted
across **all99 recordings /21 mice /five waves**, including the L-dopa wave.
It is not trained separately for each condition and is not restricted to the
four untreated waves. The study concerns PFF versus monomer longitudinal data.
Repeated recordings from one mouse are not independent biological replicates.

Existing per-recording frame caches contain:

- Native frame indices and syllable labels, including explicit unavailable or
  excluded regions; original video mapping is verified at60Hz for this view.
- Ten continuous latent channels associated with the existing behavioural
  model, plus centroid position and heading. Inspect extraction/provenance
  before assuming a latent channel has a specific physical interpretation.
- An analysis-excluded mask; some caches also contain exported velocity and
  the original static21-keypoint-v2 syllable labels. Their presence does not
  mean they are all used by the selected generator or selected ranker.
- Top-camera video and source links. Raw pose tracking exists elsewhere in the
  project, but the current ranking pipeline consumes these cached arrays.

The current generator input is **one-hot grouped syllable identity in bout time**,
not a continuous concatenation of pose, velocity and location. The selected
vocabulary has69 groups from100 original syllables (`L3_unresolved_separate`).
Adjacent identical grouped syllables are collapsed into bouts. Thus three model
steps are three grouped bouts, not three video frames or a fixed number of seconds.

Human syllable annotations and grouped mappings exist, but many syllables are
unassigned. Unknown labels must remain unknown, not be treated as a shared
behavioural category. An annotation such as "rear" is approximate prior context,
not ground truth that can override the actual movie.

Centroid coordinates are pixels; heading is angular. Calibrated wall/corner/centre
geometry is not currently part of the selected ranking recipe. Such features may
be worth considering, but their calibration and relationship to family identity
need justification. Do not quietly substitute a location proxy.

Two visually confirmed handling intervals were masked in a separate cleaned view
for one recording. Other sampled candidates were not automatically excluded.
This is not exhaustive proof that all remaining handling/tracking errors are gone.
Native gaps, recording boundaries and exclusions must not be bridged when building
sequences, estimating kinematics or interpreting event time.

Jonathan/IFDN is a separate cohort with a separate behavioural vocabulary. It has
processed DA signals; the user's scope is WT DA only as a supplementary downstream
verification modality. Karen does not supply that DA verification here. Do not
compare numeric syllable IDs across models/cohorts, or select Karen chains by DA
or condition differences. Group/DA analysis follows catalogue development.

## 4. Current method and what has already been tried

### Fitting and occurrence retrieval

We adapted the recovered September historical seqNMF implementation. Its
activation-competition term and per-lag template normalization differ from the
modern shared-library implementation and the canonical paper's formulation.
Please inspect the equations in `reference_seqnmf.update`; don't import guarantees
from another implementation because they share the name seqNMF.

An earlier bounded capacity sweep covered K=6/10/14/18/22, penalties .3/3,
seeds42/43/44 and raw/grouped vocabularies. The two active cleaned fits are:

| Setting | K10 | K18 |
| --- | ---: | ---: |
| Grouped vocabulary size | 69 | 69 |
| Template length | 3 bouts | 3 bouts |
| Competition penalty | .3 | .3 |
| Seed / iterations | 43 /80 | 43 /80 |
| Saved detections | 18564 | 32749 |
| Union valid-time coverage | 18.78% | 31.14% |

These are pooled descriptive fits, not held-out generalisation estimates. K is
model capacity, not proof of K distinct meaningful behaviours. We care about good
supported candidates and what additional families buy, not averaging all models
within an algorithm/K to declare one method superior.

Retrieval uses frozen per-factor activation thresholds, local maxima and bout-index
NMS, then error/support checks. In the current three-bout setup:

- Template weight>.15 counts as a supported stage.
- Mean stage support>=.5 admits two of three stages.
- Amplitude-adjusted template error must be<=.68.
- The accepted event spans all three observed bouts, including an unmatched one.
- NMS is applied before final error/support acceptance. A strict postfilter of
  saved calls is not equivalent to all possible alternative retrieval pipelines.

This detector does **not** implement general insertion/deletion alignment.
Accepting an unmatched stage is a substitution allowance. Learned factors and
retrieved calls can overlap; they are not automatically a mutually exclusive
segmentation of all behaviour.

### Ranking and display

The current preferred corrected legacy descriptor has449 coordinates:

| Block | Coordinates | Meaning |
| --- | ---: | --- |
| Latent sequence | 320 | 32 whole-execution elapsed-time samples ×10 channels |
| Displacement | 64 | 32×2 positions relative to onset position/heading |
| Heading | 64 | 32×sin/cos of heading relative to onset |
| Duration | 1 | Log actual event duration |

Robust pooled scales and inverse-square-root block-size factors are applied.
Nominal equal block weights do not ensure equal actual distance contributions.
This selected recipe has **no explicit velocity, angular-velocity, semantic-label
or absolute-location block**. Descriptor phase sampling does not warp video.

A genuine scaling error had estimated latent IQR after block normalization and
thereby cancelled its dimension correction. In a sampled audit this made latent
contributions dominate. This has been repaired, with unit-scaling regressions.
Do not diagnose the current catalogue as though that exact bug is still present.

An actual execution is selected as an animal-balanced squared-distance medoid;
examples are ranked by proximity to it. Proximity to a single exemplar does not
guarantee pairwise coherence or unimodality of its family.

Movies show six executions,2rows×3columns, at natural speed. Camera position and
heading are fixed at onset, with no subsequent following; dots change with the
current raw syllable and ordered syllable strips are visible. There is .5s display
context on each side of the detected event. All six panels use the same mode:

- **p100:** closest exemplar neighbours, drawing from the nearest pool rather
  than literally six identical executions.
- **P80:** near the80th percentile of exemplar proximity; higher means closer.
  This is a distance rank, not80% correctness or an unbiased full-family sample.
- **Random:** seeded uniform sampling without replacement over eligible calls;
  not balanced by mouse. Selection policies are in the grid renderer.

The default interface shows the ten most common supported families per K, with
all families accessible. Viewing every factor of every K is not required.

### Important previous comparison

We compared corrected legacy ranking on frozen original fits against cleaned
fits with a new bout-aligned/whitened physical-semantic descriptor. The user said
**"Ranking repair only is the best by far"**; the phase/cleaned alternative looked
no better than the earlier version. That alternative changed both fit input
exclusions and representation, so it did not isolate the effect of cleanup.

The current full catalogue uses **cleaned fits + corrected legacy ranking**,
providing that missing combination. It is now partially reviewed and not accepted.
The rejected descriptor experiments remain available for inspection, but are not
the selected recipe. Please avoid proposing those same changes as wholly untried.

## 5. Current human evidence and diagnostic counterexamples

The latest uploaded file is copied to:

```text
W/outputs/karen_catalogue_feedback_20261016/labels.original.json
```

It has14 new entries for13 families, all bound to exact current movie hashes:
13 P80 and1 p100. There are2 Similar,6 Mixed/unclear,1 Dissimilar and5 comment-only
entries. No random movie is newly rated. Original/reference package notes are
present in the export but are kept separate from this assessment. These are
partial selected reviews, not a quality estimate for the whole catalogue.

| Case | Human observation | Counterexample to keep in mind |
| --- | --- | --- |
| K10_F09 P80 | Similar exploratory sniffing | All6 examples have only2/3 template support; strict3/3 would discard this positive case |
| K10_F06 P80 | Similar descent/runoff, except panels1/6 | A good core can have bad assignments; current mean distance does not distinguish these exceptions well |
| K10_F02 P80 | Rest-like1–3; wall investigation4–5 | Under the current metric the two sets are not well separated; panel6 is unclassified |
| K18_F16 P80 | Rest-like1–4; wall sniffing5–6 | Five panels miss the template's middle Dash stage; this includes one wall-sniffing panel, so missing stage alone is not the classifier |
| K10_F00 P80 | Mostly rearing, but grooming in2/4 | Human action distinctions can disagree with broad symbolic/physical similarity |
| K10_F03 P80 | Stationary looking/grooming/sniffing | Low displacement is not enough; the user's suspicion of velocity overweight is not established by the actual selected feature set |
| K18_F13 P80 | Similar1/3/4/6, unlike2/5 | The positive core spans different missing-stage patterns; splitting by pattern alone fragments it |
| K18_F14 p100/P80 | Rest-like p100; different small movements at P80 | Distance ranks can sample different appearances inside one family; these comments have no explicit coherence ratings |
| K10_F04 P80 panel3 | Extra action before the target wall behaviour | Need to distinguish display pre-context from actual event boundary; don't hand-trim the clip |

Across these84 panel positions,80 match only2/3 template stages. For K18_F16,
the dominant grouped template is Sniff_Fwd -> Dash_Fwd -> Sniff_Fwd; the rule
admits Sniff -> other -> Sniff. This is a concrete permissive mechanism, not
proof it explains every failure. The full catalogue also mostly contains partial
matches, so the selected-panel frequency alone is not evidence of a causal effect.

A hypothetical3/3 postfilter retains only1147/18564 K10 calls and1480/32749 K18
calls, preserving5.30% and4.38% of their respective existing union-covered time.
This cost and the positive counterexamples rule out a casual blanket tightening.

On exact saved descriptor rows, between-set mean squared distance divided by
within-first-set mean squared distance is .801 for K10_F02(rest first),3.001 for
K18_F16(rest first),1.001 for K10_F06(core first),1.240 for K18_F11(core first)
and1.330 for K18_F13(core first). These are tiny descriptive development checks,
not statistical validation or fitted thresholds. The second set of exceptions
is not assumed to be a coherent cluster. K10_F02 suggests a representation issue;
K18_F16 shows that metric separation need not repair family membership.

K18_F14's original-to-cleaned template cosine is .99999970 with factor14 matched
to factor14. A factor-ID permutation does not explain this particular discrepancy.

For K10_F04 P80 panel3, active native frames are[128747,128879),2.2s at60Hz,
raw55 ->42 ->30; the unmatched template stage is the middle one. Its .5s pre-context
is raw30. Which physical action the user meant by the prefix remains unresolved.
Do not assume this witness establishes an onset bug or a context-only explanation.

## 6. The hard questions we want your help with

Please prioritise these rather than produce a long generic catalogue of methods.

1. **Where is the main bottleneck?** Is this mostly permissive occurrence
   retrieval, an inadequate resemblance representation, an inappropriate fixed
   three-bout template, or the fitted template objective itself? Inspect template
   concentration, support/error geometry, fit history, masks and NMS if useful.
   How would you distinguish these explanations with existing saved artifacts?
   Could two-stage matches merely dominate combinatorially, without being the
   reason for human incoherence? What evidence would falsify our current diagnosis?
2. **What defines a shared behavioural core?** Can we preserve genuine omissions,
   insertions and variants without reducing a routine to any two matching stages?
   Would stage-aware retrieval, local alignment, a duration model or another small
   change help? State what would change the scientific event definition, what can
   be tested on saved candidates, and what requires new inference or fitting.
3. **How should resemblance be measured?** We have latent sequences, displacement,
   heading, duration, derivable kinematics and partial annotations. How can we
   distinguish stationary action types while retaining pace/location variants
   within a common theme? Are better alignment, local/pairwise neighbourhoods,
   multiple prototypes, a different aggregation, or limited label-informed
   constraints justified? Explain scaling/weight choices and why they would
   improve on the already-tried alternatives. A more elaborate metric is not
   automatically better.
4. **When should one family split, merge, reject executions or remain broad?**
   What would distinguish useful supported subthemes from fragmentation? We need
   coherent families across mice, plus interpretable variants for intra analysis.
   Simply producing more clusters or a cleaner nearest-neighbour gallery is not
   sufficient. Could a two-level family/variant catalogue be defensible here,
   without turning the delivery into a new research project?
5. **What is the smallest convincing quantitative test?** Propose a few metrics
   or contrasts covering coherence, coverage, redundancy and mouse support.
   Include positive controls and failure witnesses above. How should we evaluate
   whole-family assignments rather than just the six selected clips? How should
   we use fresh random clips or untouched recordings/mice after development on
   these labels? All-data training is the user's chosen generator design, so
   don't relabel pooled-fit robustness as held-out generalisation.
6. **What would you actually do next?** Rank at most three candidate interventions
   by evidence, implementation/review cost and risk of losing useful behaviour.
   Recommend one small experiment with an interpretable success/failure rule.
   We should not reopen a K×features×weights×optimizers Cartesian search or ask
   the user to label another full catalogue before resolving the same failures.

You can disagree with our assumptions or conclude that an important point remains
unknown. In particular, do not assume correcting ranking fixes assignment, assume
3/3 is gold truth, assume group differences define good chains, or assume the
canonical seqNMF theory establishes properties of this historical implementation.

## 7. Desired form of your answer

- Start with your diagnosis and the evidence supporting it; cite actual local
  source paths/functions or saved artifacts. Distinguish inspected facts,
  deductions, hypotheses and what you could not inspect.
- Give a short table: issue, already addressed/real unresolved concern/marginal,
  decisive check and proposed action. Avoid treating old suggestions as new work.
- Compare up to three practical interventions and recommend a bounded first
  experiment. Specify which variables stay fixed and what result would make us
  abandon the intervention. Account for the positive counterexamples.
- Explain how the recommendation serves meaningful chain, intra-chain and
  inter-chain analyses. Identify any irreversible loss of useful variation or
  change in event definition.
- End with remaining scientific uncertainties that genuinely require human
  judgement, separated from things the code/data can answer.

Advice only at this stage; source access is available so no need to rely only on
this summary. If proposing an expensive or new method, explain why the simpler
saved-artifact tests cannot answer the question first.

## 8. File and evidence map

All paths below are relative to W defined above. Read selectively; the current
problem is the Karen seqNMF catalogue, not the entire historical broad search.

| Purpose | Files/directories |
| --- | --- |
| Current evidence and raw human notes | `outputs/karen_catalogue_feedback_20261016/README.md`, `labels.original.json`, `audit.json`, `panel_template_audit.csv` |
| Reproducible latest diagnostics | Same directory: `diagnose.py`, `full_call_stage_diagnostic.json`, `human_panel_distances.json`, `boundary_audit.json`, `diagnostic_validation.json` |
| Full current review package | `outputs/karen_pooled_p80_review_20261015/rank_repaired_catalogue_20261016/cleaned_ranking_correction/manifest.json`, `index.html`, grid MP4s and `components/` |
| Preferred recipe and exact rank audit | Parent catalogue directory: `selected_ranking_recipe.json`, `selected_rank_audit.json` |
| Current fitted arrays and calls | `outputs/karen_coherence_repair_20261016/fits/K10/` and `fits/K18/`: `model.npz`, `pooled_calls.csv.gz`, `result.json` |
| Event/descriptor join | `outputs/karen_coherence_repair_20261016/eligible_calls.csv.gz` and `descriptors.npz`; descriptor rows align with CSV rows, match an event by `(case, source_index)` |
| Descriptor arrays | NPZ `legacy` is the preferred corrected449D recipe; `posture`, `movement`, `tempo`, `semantic` belong to the experimental alternatives |
| Earlier comparison diagnostics | `outputs/karen_coherence_repair_20261016/family_comparison.csv`, `cross_mouse_neighbours.csv.gz`, `family_time_overlaps.csv`, `detector_sensitivity.csv`, `old_to_repaired_factors.csv` |
| Scaling and provenance | `outputs/karen_coherence_repair_20261016/feature_recipe.json`, `final_quantitative_audit.json`, `qualitative_preference_20261016.json` |
| Cleaned per-recording view | `outputs/karen_coherence_repair_20261016/reviewed_view/`; exact original cache paths/SHA are in catalogue manifest `track_rows` |
| Historical-fit implementation | `studies/karen_pff_vs_monomer/analysis/reference_seqnmf.py`, `run_reference_seqnmf.py`, `run_coherence_repair.py` |
| Capacity sweep and diagnostics | `outputs/karen_pooled_capacity_20261015/`; code `studies/karen_pff_vs_monomer/analysis/reference_capacity.py`, `run_reference_capacity.py` |
| Selected descriptor and ranking | `studies/sequence_viz/recreate_generator_videos.py`; scaling fix in `studies/karen_pff_vs_monomer/analysis/render_reference_review.py` |
| Experimental descriptors, not currently selected | `studies/karen_pff_vs_monomer/analysis/execution_similarity.py` |
| Exact mode selection and rendering | `studies/sequence_viz/render_generator_review_grids.py`, `studies/karen_pff_vs_monomer/analysis/render_six_panel_review.py`, `render_coherence_comparison.py`, `six_panel_gallery.html` |
| Earlier small comparison media | `outputs/karen_pooled_p80_review_20261015/coherence_repair_20261016/ranking_correction/` versus `cleaned_semantic/` |
| Interpretation/history of repair | `studies/karen_pff_vs_monomer/analysis/COHERENCE_REPAIR_20261016.md`, `SIX_PANEL_FEEDBACK_20261016.md` |
| Scaling/representation tests | `tests/unit/test_karen_reference_review.py`, `tests/unit/test_execution_similarity.py` |

Useful external-to-worktree absolute paths:

```text
# Raw100 human syllable annotations:
/home/fsp585/Desktop/FiberphotometryCode/outputs/karen_reviewed_analysis_view_20261012/reviewed_boundaries/syllable_annotations.csv

# Grouped69 annotations and raw members:
/home/fsp585/Desktop/FiberphotometryCode/outputs/karen_clustering_levels_20261012/L3_unresolved_separate/cluster_annotations.csv

# Original frame-cache directory (follow manifest for exact recording/SHA):
/home/fsp585/Desktop/FiberphotometryCode/outputs/karen_reviewed_analysis_view_20261012/reviewed_boundaries/frame_cache

# Raw top-camera video directory (follow manifest bindings for exact source):
/home/fsp585/Desktop/FiberphotometryCode/data/raw/karen_pff_vs_monomer/video_files
```

The Mac viewer was last observed at
`http://localhost:61695/rank_repaired_catalogue_20261016/cleaned_ranking_correction/`
through SSH forwarding. That address is client-specific; use the local files on
the server for inspection. Movies' normal playback is required to judge actions;
contact sheets and scalar metrics alone cannot establish qualitative coherence.

Existing numerical/media/identity checks give confidence that saved calls, native
frames and movie identities are aligned. They do not prove behavioural meaning,
absence of every tracking artifact, whole-family coherence, or correctness of a
future metric. Source tests passed at the prior implementation checkpoint;
the latest human-feedback diagnostics changed no production algorithm.
