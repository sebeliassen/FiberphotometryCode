# Five-theme neighbourhood showcases — 21 October 2026

## Per-theme view memory

The View choice is remembered separately for each theme in this tab session,
including Previous/Next and dropdown navigation. It survives a reload in the
same tab using sessionStorage and falls back to page memory if unavailable.
Feedback JSON/storage is unchanged. Browser checks: tab_view_memory_validation.json.

## Viewer correction — independent rows

The user found the combined overview difficult to review and its player added
large side margins. The default is now three separate three-panel row players,
each with native controls, Replay and Fullscreen. Starting one pauses the others.
Rows use their intrinsic2160×960aspect without a full-width/tall-video constraint.
The View dropdown can isolate one row; the combined overview remains optional.
All45clips, selections and existing feedback identity are unchanged.
`viewer_revision.json` records the template change separately from the frozen
render manifest, preserving previously saved/exported notes. The old template
SHA in that manifest describes the original render checkpoint. New browser
verification is `independent_rows_browser_validation.json`.


The user approved three exemplar neighbourhoods per existing theme, with an
exemplar and two nearby executions per neighbourhood. This produces 45 source
clips, five 3×3 overview movies and 15 optional three-panel row movies. The
purpose is to inspect recurring variations within each theme, including whether
rows differ only in pace or describe different behaviours. These are selected
illustrations, not a new accepted generator or an established three-cluster model.

## Access and viewing

Output in the delivery worktree:
`outputs/karen_pooled_p80_review_20261015/theme_showcases_20261021/`.

The existing woody loopback server serves this folder on port8766. With the
previous local61695 forward still active, open:
`http://localhost:61695/theme_showcases_20261021/`.
Otherwise forward8766 from the Mac and use the same path on localhost8766:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 8766:127.0.0.1:8766 fsp585@woody
```

Choose a theme and play its rows independently. Use View to isolate one row or
open the optional combined overview. Column1 is that row's actual exemplar, columns2–3 its neighbours.
Give optional row coherence/difference notes and an overall theme note. Export
the JSON before changing browser origin or computer. New feedback is isolated
from all previous galleries. The page links to the earlier p100/P80/random view.

Five overviews total73.77seconds per pass (12m18s for ten passes). Viewing all
15 separate rows once totals117.47seconds. These are two presentations of the
same45 executions, not additional independent examples. Longer sensory and
grooming rows intentionally retain their natural durations, including a31s core.

## Frozen population and distance

All7,319 complete theme cores across99recordings/21mice are retained. Theme
definitions, numerical statistics, human annotations, cleaned inputs and masks
are unchanged. No new seqNMF/KPMS fit, search or phenotype comparison occurred.
The existing five-theme atlas remains the statistical reference.

The449D corrected similarity descriptor is unchanged:32 elapsed-phase samples of
10 original latent channels, onset-relative2D displacement, relative-heading
sine/cosine, and log duration. Existing four-block weights and native calibration
scales remain frozen. Independent replay produced exact descriptor hashes for
every theme. Phase sampling affects comparison only; playback is unwarped60Hz.

The saved core index and measured tables use different internal bout offsets in
their occurrence-ID strings. A strict one-to-one join on theme, recording and
half-open native start/end binds all7,319 entries with no missing or extra
intervals. `occurrence_identity_crosswalk.csv.gz` preserves both IDs; identity
strings were not silently substituted and previous files were not rewritten.

## Selection and interpretation

The first anchor minimizes mean squared distance to the nearest execution from
each of five other mice. This prevents repetitions from one prolific mouse alone
defining local density. Subsequent anchors maximize separation relative to local
density within the densest75% of candidates. Identity breaks exact ties. A marked
fallback exists if this pool cannot supply compatible panels; none was needed.

Each anchor is displayed with its nearest compatible neighbours, preferring
different mice. All15 actual rows contain three different mice. Core intervals
cannot overlap anywhere among the nine selected executions of one theme;
pre/post context can overlap. The full statistical population keeps overlapping
cores. Every selected neighbour lies within its anchor's reported local radius.

A local support radius reaches the nearest execution of the fifth other mouse.
The observed balls contain6–18 executions and six mice each. The later anchors'
support balls are geometrically separated from previous anchors. This describes
distinct supported local variations; it does not demonstrate three biological
modes. Tight examples can leave much of the full theme population unrepresented.

Full-population nearest-anchor assignment is recorded separately from local-radius
support. Assignment is descriptive geometry, not accepted behavioural membership.
Plots show every execution's duration, speed, absolute angular velocity and
nearest-anchor distance, with selected clips marked. The summaries use physical
measurements after selection; no group, wave, DA or p-value enters selection.
The HUD percentile is proximity to that row's own anchor within its theme.

## Rendering, resources and evidence

`render_theme_showcases.py` reuses the native video renderer, source labels,
coloured centroid dot, ordered stage strip, verified60Hz mappings, half-second
context and onset-fixed position/heading. One crop is planned across all nine
executions of each theme. Finished panels become dark independently. Overview
dimensions are2160×2880; row movies2160×960; components720×960.

Preparation took10.4seconds. The preview rendered one full theme before the
remaining four. Execution uses three workers, two numerical/encoder threads,
8GiB process-tree RSS guard,80GiB free-disk floor,3GiB output cap and a one-hour
timeout. Logs and launch records are saved in the output folder; no environment
upgrade or accelerator initialization was needed.

- `selection_audit.json`: independent exact descriptor replay, direct float64
  distances, all assignments/support/physical summaries,45 unique native cores,
  cross-mouse rows and nonoverlap passed. Boundary summation differences are
  explicitly recorded; sequential reductions reproduce the stored counts.
- `media_audit.json`: all45 components and20 grids fully decoded; native mappings,
  masks/core order, source imagery, all syllable dots, panel positions and dark
  finishes passed. Independent audit took52.6seconds with peak895MiB RAM.
- Full CPU suite:933passed, six existing warnings. Strict lint passed with
  reviewed historical debt unchanged; architecture201modules/no violations or
  cycles; folder maps synchronized for146directories.
- Browser feedback/playback, portable ZIP and HTTP checks are recorded beside
  these artifacts. Mac-side SSH forwarding is outside the Linux verification.

The earlier numerical atlas, media, labels and portable package remain separate.
Behavioural coherence, usefulness and whether similar themes should merge still
need human judgement.
