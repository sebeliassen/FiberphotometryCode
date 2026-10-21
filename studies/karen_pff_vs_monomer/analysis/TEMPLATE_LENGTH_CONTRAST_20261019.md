# Controlled Karen template-length review — 19 October 2026

Two five-bout seqNMF generators are fitted and their movies are ready for
qualitative comparison against the existing three-bout generators. The purpose
is to choose settings that produce a useful range of coherent behavioural
families, including meaningful within-family variants. No generator is accepted
from fitting, coverage or interface readiness alone.

## Review entry point

In the delivery worktree, open the served hub:
`outputs/karen_pooled_p80_review_20261015/length_comparison_20261019/`.
Its `START_HERE.md` documents Mac/woody forwarding and feedback exports.

Start with K10, P80 and its three comparison prompts. Use A/B to watch each
six-execution movie at natural pace. A is three bouts; B is five bouts.
The three prompts use six distinct grid files: 40.60 seconds of P80 playback
once, or 6 minutes 46 seconds at ten plays/file, before notes. K18 provides an
optional second capacity check (35.68 seconds of P80 once). Random and p100 are
available for uncertainty; every mode need not be reviewed.

Suggested counterparts use merged native-time overlap within recording. Their
low overlaps are explicitly visible; these are diagnostic prompts, not proven
semantic matches. The deliberate witnesses are not a representative sample of
all families. Browse additional common families when judging breadth/duplication,
then give the overall generator feedback. Labels remain bound to exact movies,
models and source manifests, with separate storage/download names per comparison.

## What was controlled

- The exact frozen pooled input from all 99 recordings/21 mice, including all
  five observation/drug waves, with grouped69 vocabulary and unknown labels
  retained. Native token order, masks, chunk barriers and units were checked.
- K10 and K18 separately; lambda 0.3, seed43 and 80 historical updates for both
  lengths. L5 repacks the same bouts into valid five-bout windows. Existing L3
  models, calls and movies are reused unchanged.
- The preferred corrected legacy ranker and its frozen native scales: latent,
  displacement, heading and duration. No phase-cleaned alternative, new feature
  weights, density-exemplar change or human-label training was introduced.
- Six executions per grid, two rows/three columns, source-syllable colour dot
  and ordered observed labels, 60fps native pace, onset-fixed camera, 0.5 seconds
  context each side. Finished panels become dark. Crop extent is planned per
  family across its three modes; pixel zoom is not globally identical between
  families or lengths. Descriptor resampling does not warp the movies.

## Descriptive results

| K | Length in bouts | Saved calls | Union valid-time coverage | Factors with >=5 calls across >=2 mice |
| --- | --- | ---: | ---: | ---: |
| 10 | 3 | 18,564 | 18.78% | 10 |
| 10 | 5 | 5,201 | 9.32% | 10 |
| 18 | 3 | 32,749 | 31.14% | 18 |
| 18 | 5 | 9,033 | 15.12% | 17 |

These are full saved-fit totals, not only rendered families. L5 K18 factor0
has one eligible call in one mouse, insufficient for six unique executions;
it is explicitly omitted from the movies and retained in totals. The other
27 L5 families have 81 new grids/486 component clips. All 14,234 L5 calls were
eligible under the native-support checks. A family having a movie is not
qualitative acceptance.

The same numerical detection policy was retained: amplitude-adjusted template
error <=0.68, stage support >=0.5, stage weight >0.15, NMS radius two bouts and
per-factor 90th percentile of supported activations >1e-4. L3 can pass two of
three stages; L5 can pass three of five. This does **not** equalize precision,
effective regularisation or minimum discrete matched fraction. L5 calls are
mostly partial: K10 has 4,199 three-stage/881 four-stage/121 five-stage calls;
K18 has 8,283/725/25. Longer templates do not guarantee complete routines.
L5 calls can overlap within factor; coverage counts the union once.

L5 covers about half as much valid time under this policy. Review must establish
whether this buys coherence, useful diversity or clearer variants. It cannot be
called an improvement from reduced detections, nor a failure from lower coverage.
This bounded single-seed/penalty contrast does not establish an optimal length,
held-out generalisation, seed robustness or exhaustive convergence. Group and
DA differences did not select fits or review examples.

## Execution and evidence

Fit output: `outputs/karen_template_length_20261019/`. `launch.json` and
`render_launch.json` record detached commands, logs and scope. Each had a one-hour
timeout, 8GiB RSS cap, 6GiB output cap and 80GiB free-disk floor. Fits used
RTX4090/JAX float32 with highest matmul precision, two numerical threads and
25% GPU preallocation. Rendering used four processes with two encoder threads
each. Both empirical jobs finished; there is no continuing search.

- `quantitative_audit.json`: independent float64 recomputation of all 14,234
  L5 detections, input/payload identities, native token/core/chunk boundaries,
  support, activation thresholds, NMS spacing and union coverage passed.
  Maximum saved float32 error discrepancy was <1.58e-7 (tolerance 3e-7).
- `media_audit.json`: all new grid/component hashes, H264 shapes, 60fps clocks,
  encoded counts/durations, six unique positions, wrapper paths and exact prior
  note identities passed. Renderer fully decoded each selected native source
  and component; independent audit inspected stream metadata rather than
  decoding every grid again. No offscreen-dot/missing-tracking flags occurred.
- Each length page has `browser_validation.json` and `browser_preview.png`:
  actual playback, A/B and mode separation, catalogue guard, persistence,
  distinct downloads and JSON import/export passed. HTTP routes/ranges checked
  separately. Headless interaction used file origin; Mac tunnel not verified.
- `cpu_checks.log`: 885 CPU tests passed, six existing deprecation warnings.
  Source lint/architecture/navigation checks are recorded separately. Tests
  cover execution correctness, not behavioural meaning.

The exact launched renderer is preserved in `source_at_launch/`: its manifest
hash refers to that snapshot, preceding an import-format-only source repair.
The review HTML was built from the separately fingerprinted current builder
and template. Do not conflate launch-time and current-source fingerprints.

Next: use generator feedback to distinguish inadequate temporal context from
mixed membership or representation. Retain useful positive witnesses. If L5
looks promising, check nearby penalty/seed sensitivity before expanding the
catalogue. If both lengths mix the same actions, prioritize membership/core
definitions rather than another broad hyperparameter search.
