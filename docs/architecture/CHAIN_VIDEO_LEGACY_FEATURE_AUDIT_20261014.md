# Historical chain-video features

Read-only audit on 2026-10-14, requested before changing the new screening videos.
Checked `codex/legacy-consolidation` at
`b83298fcfb15ffd5ce3f6dbbd863b1f8e98af03e` with `git show`, without checkout or
executing historical scripts. The current master catalogue is byte-identical to
the legacy branch document (Git blob `904798bf50657513fe823c21aa88b44b978b1e07`).
Checked renderer implementations as well as Markdown; options and dated viewing
presets are distinguished below. No new render, exemplar fitting, model/search
execution or cohort analysis was performed for this audit.

## Concrete features

| Feature | Historical use | Relevance to the current screening |
| --- | --- | --- |
| Canonical exemplar and neighbouring executions | Whole-chain density-ratio representative in panel 1; nearest executions in continuous posture/kinematic space in other panels, preferring different mice. Both 1×3 and 2×3 layouts existed. | Current two ordinary-duration executions plus a different-type comparison have a different purpose. An explicitly labelled exemplar view could complement the ordinary-execution screen. |
| Alternative sampling regimes | Parallel uniform-spread, whole-chain density-ratio and constituent forward-transition exemplar galleries. Some pipelines used posture plus speed; the whole-chain selection and constituent selection were distinct analyses. | Display the selection rule. Selecting visually similar examples is useful for interpretation but does not independently prove generator coherence. Retain ordinary examples and variants. |
| Explicit ordered constituent syllables | Catalogue token strings, actual episode sequences, stage boundaries and human-readable names where annotated; clustered runs showed meta-syllable tokens. | High priority: show each execution's actual source syllables with IDs and model-bound partial labels, leaving unknowns explicit. |
| Active syllable display | Bottom sequence boxes with the current token filled/outlined, completed tokens subdued and future tokens dim. Header showed the active human label or token fallback. | High priority: a chain-active flag alone does not expose internal order. Source syllables and HSMM phases must be separate tracks, not equated. |
| Consistent colours | Shared token colour map across panels; active-stage colour matched the centroid dot and optional trajectory segments. | Useful within the same vocabulary. Numeric syllable IDs/colours must not imply homology across cohorts or different codebooks. |
| Stage durations and progress | Per-stage original duration below the boxes; elapsed/total chain time, playing/waiting/finished status. Optional signed turn annotations and warp multipliers. The wrapper used equal-width token boxes, not duration-proportional box widths. | Useful for detecting fragmentation and unequal execution timing. Actual camera/native clocks remain authoritative. |
| Mouse-centred crop | A close crop followed the centroid; an alternative locked the crop to the onset centroid. Black constant padding was used at crop edges. | High priority for observing head/body/posture changes; a full-arena view remains useful for spatial context. |
| Heading normalisation | Default crop rotated by the heading at chain onset. Diagnostic movies compared this with resetting heading at every syllable onset and marked the reset view. | Onset alignment can make directions comparable. Per-syllable resets are an optional diagnostic and change the apparent continuity. |
| Coloured dot on the animal | Small marker directly on the centroid in the current syllable's colour. | Makes the active syllable visible without searching the lower sequence row. |
| Trajectory inset | Small 2D path rotated to onset heading, segments coloured by token and a marker at the current location. Enabled in Friday/macro-motif examples; deliberately removed in the later 72-video screening preset. | Optional rather than a universal historical requirement. Retain actual spatial support; the old inset auto-scaled each path and did not establish equal distance scales. |
| Metadata and titles | Mouse ID, WT/KI and Saline/Amphetamine in panel captions; family/theme title and selection roles in scripts/catalogues. A blinded-caption option existed. | Keep exact identity available; condition labels can be hidden during initial review. Role labels must say exemplar, ordinary execution, variant or different-type comparison accurately. |
| Similarity annotations and selection statistics | Wrapper displayed a kinematic score when enabled. Catalogues listed candidate counts, animal support, density ratios/dispersion reductions or constituent percentile summaries. | Supporting context; not independent validation when the same metric selected the examples. Score meanings differed between pipelines. |
| Natural-speed playback and context | Pre/post context; documented screening presets used continuous playback without time warping or waiting for other tiles. Historical documentation cited 40 FPS model-index playback; macro documentation also cited an 80 Hz camera policy. | Keep the independently audited current clocks. Do not copy an old FPS constant or presume all historic real-speed claims have equivalent clock validation. |
| Optional temporal alignment | Wrapper supported time warping, waiting for other stages and template-based alignment showing insertions, missing tokens, skips and waits. | Diagnostic views only, visibly marked; natural timing remains necessary for judging execution. |
| Shared-prefix branch movies | Separate renderer aligned examples at the exit of a common prefix, arranged continuation branches by column. | Useful for intra-chain variants and inter/branch questions after shared-prefix identities are supported. |
| Companion review products | Markdown catalogues, MP4 links, poster PNGs and GIF previews; clear/possible/no/corrupt progression rubric plus invariant/difficulty notes. | Useful for a compact review interface and preserving observations without forcing confident behavioural labels. |

The wrapper also had transition-flash controls. The master screening preset
explicitly disabled the extra transition flash; this was not a mandatory feature.
The bare wrapper supported more options than any individual movie preset used.

## Sources and exact scope

- [Master 72-video catalogue](../../studies/sequence_viz/MASTER_72_VIDEO_CATALOG.md:5):
  three selection regimes, shared screening preset and progression rubric. Verified
  against the legacy branch's `syllable_generation/analysis/sequence_viz/` copy.
- [Historical wrapper](../../studies/sequence_viz/chain_grid_movie_wrapper.py:651):
  implemented token display, timing/duration/similarity captions and optional
  blinded labels. Constructor around line 835 declares viewing options; rendering
  around line 1495 applies centroid/onset-heading crop, dot, inset and HUD.
- [Friday exemplar script](../../studies/sequence_viz/render_2x3_friday_seqnmf_exemplars.py:3):
  density exemplar plus other-animal neighbours, six-panel view, inset and HUD
  enabled; selected instance boundaries/tokens passed explicitly.
- [Macro-motif document](../../studies/sequence_viz/clustered_seqnmf/WHOLE_CHAIN_EXEMPLAR_MACRO_MOTIFS.md:22):
  three-panel exemplar layout, token composition and optional trajectory view.
- [Constituent-transition catalogue](../../studies/sequence_viz/screening_24_transition_exemplars/README_24_TRANSITION_EXEMPLARS.md:1):
  alternative exemplar selection, support and per-stage percentile reporting.
- [Decoded phrase-family document](../../studies/sequence_viz/DECODED_SEQNMF_CHAINS_WITH_HUMAN_LABELS.md:10):
  distinguished factor templates from actual retrieved sequences, including
  templates whose lag-wise argmax string rarely or never occurred.
- [Heading-reset diagnostic](../../studies/sequence_viz/run_relational_topology_and_heading_reset_movies.py:212):
  standard onset-heading view versus per-syllable heading-reset view.

This audit inventories presentation/selection mechanisms. Historical statements
about ground truth, zero circularity, causal motor programs or proven
generalizability are not adopted as established evidence. The historical figures,
labels, animal counts and phenotype/DA claims were not reanalysed here.

## Smallest useful adaptation

Before asking the user to review fine behaviour, the most useful additions would
be the **actual ordered syllable sequence, live active-syllable highlighting,
durations and a closer mouse-centred view**. Exemplar-plus-neighbour movies are a
separate useful viewing mode; ordinary executions should remain available to
expose variation. The trajectory inset, additional scores, stage time warping,
heading-reset and branch-centred movies can remain optional.

For feature-created/HSMM/seqNMF chains, distinguish the generator's definition
from the original KPMS sequence observed during each occurrence. A factor does
not necessarily have one fixed syllable string, and an HSMM phase is not a KPMS
syllable. Human labels must be taken from each verified cohort/model vocabulary;
the legacy global annotation map cannot be copied across models. For seqNMF,
show empirical phrase variants rather than presenting a lag-wise argmax template
as an observed chain.

This is an inventory and proposed priority order, not an implemented rendering
change. Existing media, clocks, source inputs and pending qualitative review are
preserved.
