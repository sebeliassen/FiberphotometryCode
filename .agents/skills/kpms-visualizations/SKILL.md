---
name: kpms-visualizations
description: Generate or validate KPMS syllable grid movies and trajectory figures from results or explicitly reconstructed project/session artifacts, with verified recording and native-frame mappings.
---

# KPMS scientific media

Read [the rendering workflow](../../../docs/agent-workflows/visualization.md) for the actual manifest/schema and canonical CLI/Python entry points. Use results and coordinates directly when possible; a movie request does not require loading a model checkpoint.

Resolve requested artifact type, recordings/syllables, overlay, random/density selection, true label FPS, native video-frame mapping and output. Preserve project policy. Validate matched keys/shapes and eligible bouts; do not silently discard recordings or substitute a sampling mode to make the job succeed.

Frame indexes, model-row positions and timestamps are distinct. Establish clock origin/rate from acquisition/model metadata and known frames. Never infer a 2× conversion from maximum frame index or recording duration. If the mapping is unknown, ask for that specific missing information before dependent rendering.

Prefer CPU for media-only workflows; set backend/device environment before optional JAX/KPMS imports when reconstruction needs them. Honor explicitly requested device choices. Do not downgrade/upgrade dependencies or globally monkeypatch an unrelated backend as a default fix.

Validate a requested small scope with canonical dry-run/preview first, then check artifact size/count, alignment and encoding before a large render. No duration-limit flag exists in the CLI: bound inputs/selection explicitly. Checkpoint or session-pickle reconstruction is an intentional operation on trusted project artifacts, never part of metadata inventory.

Record selection/clock policy and exact outputs/evidence. Use objective state descriptors and the current task record for work spanning sessions.
