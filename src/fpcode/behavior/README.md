# Behavioural units, method choices and shared analysis

This is the implementation of the
[abstraction plan](../../../docs/architecture/BEHAVIOURAL_ABSTRACTION_PLAN.md).
The directories describe unit responsibilities; resolution and inter/intra
measurement remain separate choices. Context is optional and parallel.

```text
fpcode/
  contracts/
    behavior_units.py       Native intervals, catalogue, occurrences, UnitStream
    behavior_results.py     BehaviorResult envelope, phases, completion, posteriors, derivations
    composition.py          Finite grammar specifications and parse match records
    timelines.py            TimeAxis and clock identity
    pose.py                 PoseSeries, confidence, spatial coordinates
    sessions.py             Recording/subject identity
  behavior/
    observations/
      kinematics.py         Existing physical movement helpers, relocated
      representations.py    Continuous feature/trajectory representations (speed, heading, displacement)
    primitives/
      bouts.py              Existing RLE and gap-aware row boundaries
      counting.py           Existing frame/bout counts
      resolutions.py        Existing explicit classification views/crosswalks
      segmentation.py       Frozen-centroid vector-quantized primitive segmenter
    episodes/
      chains.py             Existing exact motifs, edits and prefix progression
      seqnmf.py             Existing full-template seqNMF and activation calls
      staged.py             Staged probabilistic episode HSMM/HMM with boundary censoring
      grammar.py            Finite grammar primitive-to-episode composition
    assemblies/
      grammar.py            Finite grammar episode-to-assembly composition
    context/
      models.py             Event-context HMM with frozen lower evidence
    analysis/
      organisation.py       Existing successor/run/return/coverage measurements
      transitions.py        Existing transition matrices and adjacent pairs
      variants.py           Existing execution-variant measurements
      phenotyping.py        Existing frame/bout/profile tables, moved whole
      prediction.py         Existing history-prediction baseline
      unit_views.py         Common-stream bridge to organisation measurements
      execution.py          Generic within-unit execution/progression measurements
    unit_adapters.py        Explicit frame/table/window representation adapters
    compatibility.py        Consumer requirements and payload-backed capability reports
  statistics/
    sequence_inference.py   Generic HMM and explicit-duration HSMM inference
    finite_composition.py   Generic bounded chart parser for finite grammars
  workflows/
    behavior_layers.py      Synthetic end-to-end multi-layer comparison workflow
```

`fpcode.kpms` continues to own its fitting backend. `fpcode.tracking` keeps pose
processing. `fpcode.statistics` keeps inference and numerical nulls.
`fpcode.workflows` composes these capabilities. Study policy and old empirical
prototypes remain under `studies`; no cohort fit is implied by this structure.

## Implemented module inventory

All seven planned modules contain bounded real algorithms in place. The remaining
first-pass requirements from the
[implementation handoff](../../../docs/agent-workflows/ANTIGRAVITY_BEHAVIOUR_IMPLEMENTATION_HANDOFF.md)
have been independently verified; exact checks and limits are in
[CURRENT.md](../../../docs/tasks/CURRENT.md). The inventory below describes
available operations. Synthetic verification does not establish validity on
arbitrary recordings.

1. **`observations.representations`**: Continuous kinematic features (`speed`, planar `heading`, circular `angular_velocity`, `centroid_x`, `centroid_y`, and `compute_interval_trajectory_metrics`) from explicit `PoseSeries` and `TimeAxis`. Preserves raw observations, native coordinates, and per-landmark validity without leaking across missing frames or unobserved trajectories.
2. **`primitives.segmentation`**: `CentroidPrimitiveSegmenter` vector-quantized baseline. Fits frozen centroids and scaling strictly on training samples; assigns held-out rows with `-1` for unsupported observations. Emits `UnitStream` via the canonical gap-aware RLE adapter.
3. **`episodes.staged`**: `StagedEpisodeModel` implementing configured categorical explicit-duration HSMM and staged HMM inference. Decodes phase intervals, separates same-routine reentry invocations, handles boundary right-censoring via survival mass, and attaches `PhaseRecord` and `CompletionEvidence` sidecars.
4. **`episodes.grammar`**: `GrammarEpisodeDetector` implementing primitive-to-episode detection from caller-specified finite grammars. Preserves scoped child references, alternative weighted derivations, and gap barriers.
5. **`assemblies.grammar`**: `GrammarAssemblyComposer` implementing episode-to-assembly composition. Reuses the generic finite parser, preserves episode child references, and supports DAG reuse of child types across assembly definitions.
6. **`context.models`**: `EventContextHMM` implementing categorical event-context models. Operates on frozen lower unit streams, measures persistence in event steps, preserves physical gap boundaries, and emits `StatePosteriorRecord` sidecars.
7. **`analysis.execution`**: Generic within-unit execution analytics (`measure_occurrence_execution`, `aggregate_execution_metrics`, `measure_phase_pacing`, `measure_onset_profiles`, `measure_phase_profiles`). Measures durations, exposure, time-weighted physical means, peak latency, displacement, path length, and supported physical/normalized profiles from common streams and rich payloads.

Assembly composition is an optional adjacent component alongside inter-chain
measurements on the same episode stream. Its derivations can describe shared
parts/branches/loops; its intervals can also enter shared inter/intra analyses.
Current grammars are caller-defined with bounded repetition. They do not discover
recursive superroutines or automatically condition grammar choice on context.

Observation features can accompany any unit stream through `FeatureAssociation`;
they do not need to alter KPMS labels, resolution crosswalks or chain detection.
Execution and onset/phase analyses align the original supported frame samples
to each unit's native intervals. Passing features to those analyses is explicit;
detectors do not automatically copy or consume every feature channel.

`extract_trajectory_features(..., arena_bounds=(xmin, xmax, ymin, ymax))` adds
`distance_nearest_corner`, `distance_nearest_edge` and `distance_center`, using
the nose/tail-base midpoint and the pose's position units. Bounds describe an
axis-aligned rectangular arena in the same coordinate frame; calibrate/rectify
coordinates before using them for a rotated or perspective-distorted arena.
Bounds are never estimated from the mouse's observed range. Outside-arena or
unsupported midpoint samples have invalid distance channels, without changing
other feature masks. With no bounds supplied, the original five channels remain.

Explicit `arena_landmarks=("nose", ...)` adds distances for other tracked reference
points with their own confidence/support. `arena_corner_features=True` retains
all four image-coordinate corner distances. A study may label camera-near/far
corners according to chamber placement; these distances do not correct parallax.
`append_landmark_centroid` can add a supported average of caller-selected tracked
points (e.g. ventral paws); it is an anatomical reference, not a reconstructed
ground truth. `robust_position_bounds` provides a declared quantile envelope;
studies must distinguish this empirical proxy from physically calibrated walls.

`workflows.behavior_layers.compare_frozen_episode_methods` compares exact motifs,
frozen seqNMF activation inference and optional configured grammar/staged models
on one recording. Template channel identities must match the primitive catalogue;
features accompany results without influencing detection. Budget failures are
explicit, and raw overlapping candidates remain available for declared downstream
selection. `adapters.behavior_sessions.pose_from_dlc_table` aligns tracked points
by native index, with an explicit reject/mask policy for invalid confidence sentinels.

For physical-time primitive features, provide `native_frames`, or declare
`physical_time_origin_s` and `native_frame_origin` with explicit FPS. The library
does not infer frame zero from FPS alone or round fractional observations into
frames. Features used by execution analysis require a `FeatureAssociation` with
their native mapping. Training normalization remains separate from physical units.

## Common results and capabilities

`UnitStream` remains the method-independent interval representation. `BehaviorResult` wraps `UnitStream` with validated optional payload sidecars:
- `features`: `FeatureAssociation` wrapping `SignalSeries` with recording provenance.
- `phases`: `PhaseRecord` tuples tracking sub-unit constituents.
- `completion`: `CompletionEvidence` tuples recording observed entry/exit, final phase reached, and censoring status.
- `derivations`: `CompositionDerivation` tuples tracking grammar parse trees and child references.
- `posteriors`: `StatePosteriorRecord` storing validated posterior arrays.

`compatibility.inspect_compatibility` derives capabilities directly from validated payloads in `BehaviorResult`. Plain `UnitStream` continues to strictly reject analyses requiring genuine completion, phase, or posterior evidence.

## Dependency and conformance checks

All architectural fences in `ops/repo/dependencies.py` are strictly preserved with zero violations and zero cycles:
- Observations never depend on higher layers.
- Assemblies never import episode detector implementations.
- Shared analyses never import detector fitters.
- Generic numerical inference resides in `fpcode.statistics`.
- Workflow composition resides in `fpcode.workflows.behavior_layers`.
