# Behavioral Chain Diagnostic Grid Movie Suite

Comprehensive toolkit for visualizing and verifying multi-syllable behavioral sequences (chains) in mouse open-field Keypoint-MoSeq (KPMS) experiments.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Chain Discovery Methods](#2-chain-discovery-methods)
3. [Grid Rendering Options](#3-grid-rendering-options)
4. [Spatial Verification Framework](#4-spatial-verification-framework)
5. [Core Modules](#5-core-modules)
6. [Quickstart Examples](#6-quickstart-examples)
7. [Scientific References](#7-scientific-references)

---

## 1. Design Philosophy

### Why This Toolkit Exists

KPMS assigns a discrete syllable label to every video frame. Behaviorally interesting events are rarely single syllables — they are *chains*: `turn → wall-walk → turn`, `approach → groom → depart`, etc. The problem is that there are many ways to discover such chains in a stream of tokens, and once discovered, many ways to display them that can mislead you. This toolkit addresses both.

### Core Design Decisions

#### No Pausing / No Freeze Frames
Early versions used `time_warp` and `wait_for_others` sync modes that artificially held frames at stage boundaries so the viewer could compare syllable onsets across tiles. In practice this was **confusing** — the mouse appeared to freeze mid-stride and it was difficult to tell real behavior from an encoding artifact. We removed all pausing in favor of **fluid continuous playback** (`sync_mode="nothing"`). A human can simply watch each tile individually and decide whether a pattern exists. The HUD still marks syllable boundaries with a brief color flash.

#### Automated Outlier Rejection (Error Accumulation)
If ~5% of individual syllable bouts are noisy (tracking glitches, micro-flickers, stuck states), the probability of encountering **at least one bad bout** in a chain grows fast:

| Chain length | P(≥1 bad bout) |
|---|---|
| 3 bouts | 14.3% |
| 5 bouts | 22.6% |
| 10 bouts | 40.1% |

This is accumulation of error opportunities, not chaos in the dynamical-systems sense, but the practical effect is similar: one corrupted bout throws off the visual coherence of the entire sequence. We therefore **pre-filter** candidate instances to reject:
- **Micro-flicker bouts**: <5 frames (<125 ms) — too short to be a real postural state
- **Stuck/frozen tracking**: >65 frames (>1.625 s) — likely a tracking failure
- **Velocity jumps**: >22 px/frame centroid displacement — physically impossible mouse movement
- **IQR statistical outliers**: bouts whose kinematics fall outside 1.5×IQR of their syllable population

Surviving instances are then **ranked by centrality** (closeness to the median) so the grid shows the most representative examples first.

#### Solid Black Background Padding
OpenCV's `warpAffine` needs a `borderMode` when the rotated/cropped view extends past the original frame. The default `BORDER_REFLECT` creates mirror-reflection artifacts near arena walls that look like phantom mice. We strictly enforce `BORDER_CONSTANT` with `borderValue=(0, 0, 0)` — clean black padding, no visual confusion.

#### Rodent-Perspective Thinking
Throughout this toolkit we try to respect what matters *from the mouse's viewpoint*:
- **Rotationally agnostic**: A `turn → wall-walk → turn` is the same behavioral pattern whether it happens along the north wall or the south wall. The heading-reset mode and Dvorkin relational topology both capture this.
- **Egocentric boundary encoding**: Following Alexander et al. (2020), walls are encoded relative to the mouse's body axis (`LEFT_FLANK`, `RIGHT_FLANK`, `SNOUT_AHEAD`) rather than compass directions.
- **Motor primitive isolation**: The per-syllable heading reset lets you inspect *what the body is doing* at each stage without it being entangled with the global trajectory direction.

---

## 2. Chain Discovery Methods

The toolkit supports multiple methods for finding recurring chains in the syllable stream. Each answers a slightly different question, and comparing their outputs is the point of the diagnostic movies.

### Method A1: Exact N-grams

**What it does**: Scans the RLE-compressed syllable stream for exact token subsequences (e.g., `[1, 11, 1]`).

**When to use it**: When you have a specific hypothesis about a token sequence. This is the simplest and most interpretable method — if `[1, 11, 1]` recurs 40 times across sessions, every instance contains exactly those three syllables in that order.

**Limitation**: Rigid. Doesn't tolerate insertions, substitutions, or duration variation. A chain that is behaviorally the same but uses syllable 12 instead of 11 (a closely related posture) will be missed entirely.

---

### Method A3: Clustered RLE Sequences

**What it does**: Groups syllables into broader behavioral clusters (e.g., "locomotion cluster", "grooming cluster") using KPMS embedding similarity, then searches for recurring *cluster-level* triplets (e.g., Cluster 01 → Cluster 04 → Cluster 01).

**When to use it**: When you want to find patterns at a coarser grain — "locomotion → stationary → locomotion" regardless of which specific locomotion syllable was used. Useful for discovering higher-order behavioral structure.

**Limitation**: Clustering choices matter. Too few clusters = everything looks the same. Too many = you're back to exact n-grams.

---

### Method B: Weighted Alignment Motif

**What it does**: Uses a weighted edit-distance alignment (substitution costs proportional to KPMS embedding distance between syllable types) to find the best-matching instances of a query motif across all sessions.

**When to use it**: When you want soft matching — "find me things that look like `[6, 17, 2]` but allow similar syllables to substitute in." This handles the biological reality that the same behavior might use slightly different syllable tokens across animals or sessions.

**Limitation**: Requires tuning the substitution cost matrix. More computationally expensive than exact matching.

---

### Method C: Profile HMM Matches

**What it does**: Builds a Profile Hidden Markov Model from a set of aligned chain exemplars. The profile captures position-specific syllable preferences and allows insertions/deletions, analogous to protein sequence alignment in bioinformatics.

**When to use it**: When you have a family of chains that share a common structure but vary in length and composition. The profile HMM naturally handles "this position is usually syllable 6 but occasionally syllable 8" and "some instances have an extra grooming bout inserted between stages 2 and 3."

**Limitation**: Needs multiple exemplars to build the profile. Overkill for simple exact patterns.

---

### Method D: seqNMF Factor Motif (Convolutional Non-negative Matrix Factorization)

**What it does**: Applies sequence NMF (Mackevicius et al., 2019) to the one-hot syllable usage matrix, discovering convolutional motifs that recur across the recording — temporal patterns that the algorithm finds *without being told what to look for*.

**When to use it**: Unsupervised discovery. You don't need a hypothesis. seqNMF will find whatever temporal structure is statistically present. Each "factor" is a motif template; instances are time windows where that factor activates strongly.

**Limitation**: Motifs may be statistically recurrent but ethologically uninteresting. Requires human inspection to determine which factors correspond to meaningful behavior.

---

### Method E: PST Branch-Centred Grid (Probabilistic Suffix Tree)

**What it does**: Builds a variable-order Markov model (PST) of the syllable stream, then identifies branching points — prefixes (e.g., `[14, 7]`) that lead to multiple distinct continuations with different probabilities.

**When to use it**: When you want to understand *decision points* in the behavioral stream. "After `walk → orient`, the mouse usually turns left (60%) but sometimes turns right (25%) or freezes (15%)." The branch grid renders instances of each branch side by side for visual comparison.

**Limitation**: The tree can grow large. Statistical significance of branches depends on having enough data.

---

### Summary: Why Multiple Methods?

No single method is "right." Each captures a different aspect of sequential structure:

| Method | Matching | Discovery | Best for |
|---|---|---|---|
| A1 Exact N-grams | Rigid | Hypothesis-driven | Confirming a specific sequence |
| A3 Clustered RLE | Soft (cluster) | Semi-supervised | Finding coarse behavioral patterns |
| B Weighted Alignment | Soft (distance) | Hypothesis-driven | Tolerating syllable substitutions |
| C Profile HMM | Soft (profile) | Exemplar-driven | Modelling a variable-length family |
| D seqNMF | N/A | Unsupervised | Finding unknown temporal motifs |
| E PST Branch | N/A | Data-driven | Identifying decision/choice points |

The diagnostic movie suite renders the **same** grid format for all methods so you can directly compare how each one carves up the behavioral stream.

---

## 3. Grid Rendering Options

### `sync_mode` — Temporal Playback Strategy

Controls how multiple tiles in the grid handle time relative to each other.

| Value | Behavior | When to use |
|---|---|---|
| `"nothing"` | **(Recommended)** Each tile plays at natural 1.0× speed (40 FPS). Tiles are independent — they don't wait for each other. When a tile's chain ends, it continues into `POST-CHAIN` footage. | Default for all diagnostic movies. Clean, unconfusing, lets humans assess each tile individually. |
| `"time_warp"` | Resamples each stage to the median stage duration across instances, so all tiles hit syllable boundaries simultaneously. | Useful if you specifically need temporal alignment across tiles (e.g., for automated frame-by-frame comparison). Can look unnatural. |
| `"wait_for_others"` | Natural speed within each stage, but holds/freezes at boundaries until the slowest tile catches up. | Compromise between natural speed and aligned boundaries. Can be confusing due to apparent freezing. |

### `centroid_mode` — Camera Tracking

Controls how the camera follows the mouse within each tile.

| Value | Behavior | When to use |
|---|---|---|
| `"follow_centroid"` | **(Recommended)** Camera smoothly tracks the mouse centroid, rotated so the initial heading points upward. The mouse stays roughly centered. | Default. Shows the full trajectory context. |
| `"lock_centroid"` | Fixed crop from KPMS (legacy). The view doesn't track the mouse — it can walk out of frame. | Only for debugging or reproducing original KPMS grid movies. |

### `reset_heading_per_syllable` — Per-Syllable Heading Reset

| Value | Behavior | When to use |
|---|---|---|
| `False` | **(Default)** Camera heading is set once at chain onset and held throughout. The global trajectory is visible — you can see the full path through the arena. | When trajectory shape matters: `turn → wall-walk → turn` is visually distinctive as a trajectory. |
| `True` | Camera heading resets to 0° (facing up) at the start of each new syllable. The HUD displays the relative turn angle between syllables (e.g., `+37°L`, `+77°L`, `Fwd`). | When you want to isolate motor primitives — inspect *what the body is doing* at each stage independently of global direction. Critical for comparing chains like `walk → groom → depart` where the departure direction varies but the motor sequence is the same. |

### `filter_outliers` — Automated Instance Pruning

| Value | Behavior | When to use |
|---|---|---|
| `False` | **(Default)** All provided instances are rendered as-is. | When you want to see everything, or have already pre-filtered. |
| `True` | Applies the 4-stage filter (micro-flicker, stuck, velocity jump, IQR outlier) and ranks survivors by centrality to the median kinematics. Selects the top `rows × cols` most representative instances. | **(Recommended for diagnostics)** When you have many candidate instances (e.g., 40+ n-gram matches) and want a representative grid without visual noise from tracking artifacts. |

### `alignment_mode` — Instance Layout Strategy

| Value | Behavior | When to use |
|---|---|---|
| `"show_variations"` | **(Default)** Each tile shows a different instance of the chain. Standard multi-instance comparison grid. | Most common use case. |
| `"compact_template"` | Shows a single representative instance with a compact overlay of the template tokens. | Quick single-instance preview. |
| `"natural_speed"` | Similar to `show_variations` but emphasizes natural timing without any temporal manipulation. | Fine-grained timing inspection. |

### Other Options

| Parameter | Default | Description |
|---|---|---|
| `rows`, `cols` | 2, 3 | Grid layout. 2×3 = 6 tiles is the standard diagnostic format. |
| `window_size` | 240 | Crop window in pixels around the mouse centroid. |
| `scaled_window_size` | 320 | Output resolution per tile in the rendered video. |
| `pre_sec`, `post_sec` | 0.75, 0.75 | Seconds of footage to show before/after the chain. |
| `fps` | 40 | Output video frame rate (matches KPMS acquisition rate). |
| `show_hud` | True | Overlay showing current syllable token, duration, stage index, and animal metadata. |
| `show_trajectory_inset` | False | (Disabled by default) Small trajectory plot in the corner showing the mouse's path through the arena. |
| `show_similarity_panel` | False | Visual similarity indicator between instances (hidden in screening to avoid uncalibrated bias). |
| `blind_labels` | False | If True, hides genotype/drug labels for blinded assessment. |
| `use_gpu_encoder` | True | Use NVENC hardware encoding via FFmpeg. Falls back to libx264 if unavailable. |

---

## 4. Phase 1 Screening Batches (24 Candidates)

> [!TIP]
> **Methodological Guide**: For an in-depth mathematical and epistemological comparison of whole-chain vs. constituent transition exemplar selection (addressing circularity, motor control primitives, and co-articulation), see [`EXEMPLAR_SELECTION_FRAMEWORKS.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/EXEMPLAR_SELECTION_FRAMEWORKS.md).

The screening suites test **8 sequence discovery methods** across **3 candidate families** each (24 videos total) under a frozen, unconfusing viewing preset:

### A. Uniform Spread Batch (`screening_24_batch/`)
- Samples 6 instances spread evenly across the dataset to evaluate broad generalizability across sessions.
- Catalog: [`screening_24_batch/README_24_CANDIDATES.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_batch/README_24_CANDIDATES.md)

### B. Keypoint-MoSeq Native Density-Ratio Exemplar Batch (`screening_24_exemplars/`)
- Implements Keypoint-MoSeq's native contrastive density-ratio algorithm (`keypoint_moseq.util.sample_instances(mode='density')`):
  1. Continuous 10D SLDS latents are interpolated to $L=20$ timepoints ($200\text{D}$).
  2. 4D PCA is fitted across 400 background episodes sampled from 30 sessions.
  3. The core **Exemplar** maximizes the contrastive density ratio against background behaviors:
     $$\text{Ratio}(x) = \frac{\text{Local Density}(x)}{\text{Global Density}(x)} = \frac{\bar{d}_{\text{global}}(x)}{\bar{d}_{\text{local}}(x)}$$
- Samples from an adaptive neighborhood pool:
  $$k_{\text{pool}} = \min\left(50, \max\left(6, \lceil 0.10 \times N \rceil\right)\right)$$
- Achieves **$54.5\%\text{--}89.4\%$ kinematic dispersion reduction** with local density enrichment up to **$4.52\times$**, while stratifying the 6 tiles across distinct animals.
- Frozen viewing preset: 40 FPS continuous real-speed, follow centroid crop, colored centroid dot on the mouse (active syllable color, no badge dot, no yellow transition flash), solid black padding, token HUD, and trajectory inset removed.
- Catalog: [`screening_24_exemplars/README_24_EXEMPLARS.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_exemplars/README_24_EXEMPLARS.md)

### C. Forward Constituent Transition-Exemplar Batch (`screening_24_transition_exemplars/`)
- Evaluates each constituent syllable locally within its forward transition bucket $\mathcal{B}(S_k \to S_{k+1})$ or coarse cluster transition $\mathcal{B}(S_k \to C_{k+1})$, completely eliminating whole-chain post-hoc filtering.
- Fits 3D PCA on single-syllable latent trajectories ($150\text{D}$) within each forward transition bucket, scoring stage quality by percentile rank to the transition medoid exemplar.
- Identifies **Strict Constituent Chains** where *every constituent stage* ranks in the top 50% ($p_{\min} \ge 0.50$) of its transition bucket (e.g. 17.6% of `[1, 11, 1]`, 17.0% of `[6, 17, 2]`, and 12.6% of `[1, 11, 1, 11]`).
- Ranks candidate chains by composite constituent score $\bar{p} = (\prod p_k)^{1/M}$ and renders the top 6 instances stratified across distinct animals.
- Frozen viewing preset: 40 FPS real-speed, follow centroid crop, mouse centroid dot in active syllable color, no badge dot, no yellow flashes, solid black padding, token HUD, and trajectory inset removed.
- Catalog: [`screening_24_transition_exemplars/README_24_TRANSITION_EXEMPLARS.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_transition_exemplars/README_24_TRANSITION_EXEMPLARS.md)

---

## 5. Core Modules

| File | Role |
|---|---|
| `chain_grid_movie_wrapper.py` | **Primary Engine**: Video frame reading, coordinate normalization, camera rotation, heading reset logic, HUD rendering, outlier filtering, and FFmpeg encoding. Contains `ChainGridMovieGenerator` class and `generate_chain_grid_movie()` convenience wrapper. |
| `run_24_candidate_screening_batch.py` | **Screening Runner (Uniform)**: Generates 24 screening grid movies spread across sessions with automated markdown catalog. |
| `run_24_candidate_exemplar_batch.py` | **Screening Runner (Whole-Chain Density-Ratio Exemplar)**: Generates 24 screening grid movies centered on KPMS contrastive density-ratio whole-chain exemplars. |
| `run_24_candidate_transition_exemplars.py` | **Screening Runner (Constituent Transition Exemplar)**: Generates 24 screening grid movies using forward transition-bucketed constituent scoring (no whole-chain post-hoc filtering). |
| `bout_quality_and_spatial_verifier.py` | **Verification Framework**: `SpatialContextEvaluator` (Dvorkin topology + Alexander EBVs) and `ChainQualityAndContextVerifier` (dual Q/S scoring, weakest-stage attribution). |
| `run_multimethod_diagnostic_movies.py` | **Runner**: Generates the complete 6-method diagnostic suite with fluid playback, outlier filtering, and black padding. |
| `run_all_methods_chain_verification.py` | **Runner**: End-to-end verification across all methods; compares sync modes and camera modes. |
| `run_bout_quality_and_spatial_diagnostics.py` | **Runner**: Side-by-side comparison of `TRUSTED_CLEAN` vs `MOTOR_VARIANT` instances. |
| `run_prioritized_additions_pilot.py` | **Runner**: Demonstrates alignment options (fluid natural speed, compact template, branch-centred grids). |
| `run_relational_topology_and_heading_reset_movies.py` | **Runner**: Relational topology diagnostics and heading-reset comparison movies (standard vs reset side by side). |

---

## 6. Quickstart Examples

### A. Basic Chain Grid Movie (Recommended Defaults)
```python
from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
    load_chain_movie_resources,
    generate_chain_grid_movie
)

resources = load_chain_movie_resources()

instances = [
    {
        "recording_key": "day1_T1_1680.1694.728.e_Arena1234_...",
        "start_frame": 12040,
        "end_frame": 12180,
        "stage_boundaries": [12040, 12075, 12110, 12145, 12180],
        "stage_tokens": ["syll_001", "syll_011", "syll_001", "syll_030"],
        "animal_id": "Mouse 1680",
        "genotype": "WT",
        "drug": "Saline"
    },
    # ... up to 6 instances for a 2×3 grid
]

generate_chain_grid_movie(
    instances=instances,
    output_path="my_chain_grid.mp4",
    resources=resources,
    centroid_mode="follow_centroid",   # Smooth camera tracking
    sync_mode="nothing",              # Fluid natural playback (no pausing)
    reset_heading_per_syllable=False,  # Standard chain-onset alignment
    filter_outliers=True,             # Auto-reject noisy instances
    use_gpu_encoder=True              # NVENC hardware encoding
)
```

### B. Per-Syllable Heading Reset
For inspecting motor primitives independently of global trajectory:
```python
generate_chain_grid_movie(
    instances=instances,
    output_path="my_heading_reset_grid.mp4",
    resources=resources,
    sync_mode="nothing",
    reset_heading_per_syllable=True,   # Camera resets at each syllable onset
    filter_outliers=True,
    use_gpu_encoder=True
)
```
Each syllable is oriented facing UP. The HUD shows signed turn angles (e.g., `1.55s (+37°L)`, `1.40s (+77°L)`, `Fwd`).

### C. Running Quality & Spatial Verification
```python
from syllable_generation.analysis.sequence_viz.bout_quality_and_spatial_verifier import (
    ChainQualityAndContextVerifier
)

verifier = ChainQualityAndContextVerifier(resources["results"])
audit = verifier.verify_chain(instances[0])

print("Classification:", audit["classification"])          # TRUSTED_CLEAN / MOTOR_VARIANT / MEASUREMENT_CORRUPT
print("Weakest Stage:", audit["quality_worst_stage_token"]) # Pinpoints the corrupted stage
print("Excursion Topology:", audit["relational_topology"])  # RELOCATION_BOUNDARY, RETURN_TO_ORIGIN, etc.
print("Egocentric Boundary:", audit["start_ebv"])           # LEFT_FLANK, RIGHT_FLANK, SNOUT_AHEAD, OPEN_ARENA
print("Turn Angles:", audit["stage_turn_angles"])           # Relative degrees per stage
```

---

## 7. Scientific References

1. **Dvorkin, Benjamini, & Golani (2008)**. *Mouse Cognition-Related Behavior in the Open-Field: Emergence of Places of Attraction for Exploration*. **PLOS Computational Biology**, 4(2): e23.
   - Established excursion topology ($A \to A$ origin loop vs $A \to B$ relocation) and history-dependent stopping dynamics.

2. **Alexander, Hinman, et al. (2020)**. *Egocentric boundary vector tuning of the retrosplenial cortex*. **Science Advances**, 6(8): eaaz2322. ([PMID: 32128423](https://pubmed.ncbi.nlm.nih.gov/32128423/))
   - Discovered Egocentric Boundary Cells (EBCs) encoding walls relative to flank/snout, establishing that rodent navigation is egocentrically boundary-relative rather than compass-dependent.

3. **Mackevicius, Bahle, et al. (2019)**. *Unsupervised discovery of temporal sequences in high-dimensional datasets, with applications to neuroscience*. **eLife**, 8: e38471.
   - seqNMF convolutional non-negative matrix factorization for unsupervised motif discovery in neural and behavioral time series.
