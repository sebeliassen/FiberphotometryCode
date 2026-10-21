# Chain viewing specification — 2026-10-15

## Latest user clarification

The selection level applies to **the entire video file**. For a given chain,
prepare a p100 video, a p80 video and a random video. Every panel in a file uses
that file's selection setting. The panels are different executions of the same
chain family. Three or six executions play simultaneously, each at natural pace;
the reviewer can focus on a different panel on each replay.

| Video file | Every panel in the file |
| --- | --- |
| `chain_p100.mp4` | Top exemplar-tier executions |
| `chain_p80.mp4` | Executions near the same within-family P80 target |
| `chain_random.mp4` | Random executions from the eligible family population |

Do not mix exemplar, P80 and random in one video. The preceding independent-clip
package and subsequent mixed-selection grid package are both superseded formats.
Implementation update: the corrected homogeneous-selection renderer is implemented;
six two-cohort previews passed validation, followed by the complete147movie
collection in `outputs/generator_qualitative_movies_20261015`. Final status COMPLETE,
media and viewer validation PASSED, driver exit0. Open its index.html for P80-first
review. Continuous qualitative interpretation remains pending.

## Recovered legacy decision sources

Inspected `codex/legacy-consolidation` directly, without checkout. Its local
relocated copies of the following two documents match the branch byte-for-byte:

- [Three-tier video comparison](../../studies/sequence_viz/clustering_resolution_comparison/CLUSTERING_RESOLUTION_VIDEO_COMPARISON.md:91)
  describes separate exemplar, percentile and ordinary/random movies per family,
  with multiple executions in each movie. The earlier percentile setting was P90.
- [Later P80 specification](../../studies/sequence_viz/clustering_resolution_comparison/P80_SIMILARITY_AND_SAMPLE_SIZE_SCALING.md:20)
  explicitly records the move from P90 to P80; its
  [gallery](../../studies/sequence_viz/clustering_resolution_comparison/P80_SIMILARITY_AND_SAMPLE_SIZE_SCALING.md:74)
  documents three-mouse all-P80 videos. Example: Theme A / Level 1 has target
  similarity 0.463 and panel similarities [0.462, 0.463, 0.469], all in one
  `theme_A_rear_descent_p80.mp4`.
- The actual legacy implementation at
  `studies/behavior/sequence_viz/clustering_resolution_comparison/audit_clustering_discrepancy.py`
  (branch lines 732–790) computes one P80 target per level/family, selects a
  triplet near that target, assigns **every** panel `P80_REPRESENTATIVE`, and
  renders one 1×3 movie with `sync_mode="nothing"`. It prefers distinct animals,
  then falls back to distinct executions if fewer than three animals are available.

The legacy exemplar tier used multiple representative executions; “p100” here is
that top viewing tier, not a request to duplicate one maximum-score occurrence.
Record actual achieved scores/ranks. P80 is a population percentile target, not
an absolute similarity value of 0.80 and not a claim of behavioural correctness.

The latest user request specifies pure random selection; do not silently impose
the historical random gallery's animal stratification. The exact similarity
metric in the new package is a separate issue: the newly implemented medoid/
distance descriptor is not the legacy cosine-similarity or density-ratio method.
Recovering the layout does not establish equivalence of those samplers.

## Retained camera, timing and overlay preferences

Centre and rotate each view once using its own chain-onset position and heading,
then hold that transform fixed throughout context and the full call. No following,
recentering, later heading resets or trajectory inset. Use a common pixel scale
within a family across its displayed selections.

A moving dot marks the actual tracked centroid. Its colour changes with the
current original KPMS syllable and matches the live observed-sequence strip.
Show actual ordered source IDs, available model-bound labels, per-syllable
physical durations and current-stage highlighting. Unknowns remain unknown;
generator templates and HSMM phases are not source syllables.

No time warping, stage waiting or forced alignment. Finished panels become dark
and labelled; do not silently freeze or loop shorter executions. The video ends
at the longest panel. The user's latest onset-locked camera/no-inset preferences
supersede legacy following-camera and trajectory-inset presets. Verified current
native/actual-camera clocks supersede historical hard-coded 40 fps claims.

## Current scope and next step

Keep five saved configurations per cohort, the same 49 supported /68 nominal
fold-0/replicate-0 reference families, held animals and reserved-animal exclusions.
No new model/KPMS fit, broad search or phenotype/DA-based clip selection.

Implementation now writes separate p100, p80 and random files and rejects mixed
selection panels. Six previews were decoded/source/timing/overlay checked before
launching the wider render. Next: confirm full media and viewer checks, then use
the generated P80-first viewer and notes. The source ranking is reused unchanged.
Do not reuse the superseded mixed-grid playback budget for this collection.
