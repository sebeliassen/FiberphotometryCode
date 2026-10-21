# V2 Visualization Repair: Handoff (written 2026-10-07 ~17:30)

Source task: repair the V2 selection/rendering pipeline per the Codex audit (`density_sampling_v2_audit.zip` at repo root, extracted copy in the Antigravity scratch dir). The user asked for two things:
1. **Fix** the confirmed defects.
2. **Leave checks and unmeasured hypotheses** in an md for another agent (this file, section 4).

> [!IMPORTANT]
> **State at handoff: only the shared module exists.** The selector, both renderers, the gallery, the runner `.sh` and the tests are **still the pre-repair versions**. No production data has been run. Nothing is certified.

Project root: `syllable_generation/multiview_top_bot_experiments/`. "04/" below means `04_reports_and_figures/`.

---

## 1. Done so far

Created `04/viz_v2_common.py` (import-light). It imports and passes a smoke test in the `kpms_gpu_cuml` env: half-open bouts, 60 fps -> 10+30 frames, 30 fps -> 5+15, grid layout 12 -> 3x4 and 24 -> 4x6. It contains:

| Area | Contents |
|---|---|
| Timing | `frames_from_seconds`, `make_window(fps, pre_s, post_s)` (stores seconds and frames). Defaults: score 1/6 s pre + 0.5 s post, movie 1 s + 2 s, min bout 5/60 s. |
| Bouts | `half_open_bouts(z)` (canonical `[start, end)`, empty-safe), `make_bout_id` (`"{segment}:{onset}"`), `stable_ranks`, `base_session_id`, `get_session_files`. |
| Alignment | `filter_segment_pose` (wraps installed `kpms.util.filter_centroids_headings`, run once per whole segment), `egocentric_window` (NaN-preserving, algebraically equal to KPMS `inverse_rigid_transform`; **equivalence not yet tested**). |
| QC | `QCConfig`, `valid_track_mask` (finite x AND y AND likelihood >= thr), `longest_true_run`, `window_frac_gap`, `summarize_frac_gap`/`summarize_window_qc`, `qc_decision`, `load_fused_tracks`. |
| Selection | `choose_tier` (from the **eligible** count), `density_ratio_scores` (KPMS formula, self-neighbor convention preserved), `order_group_exemplar_first` (explicit exemplar-first, ties by stable ID), `select_tier1_density`, `select_tier2_compact`, `select_tier3_all`. |
| Final-selection safety | `assert_unique_bout_ids`, `selection_ids`, `resolve_display_bouts` (never substitutes; failures become flagged blanks keeping the original ID), `assert_rendered_equals_manifest`, `grid_layout`. |
| Bot pose | `compute_robust_bot_pose`: real paw names (`left_front_paw`, ...); paw fallback returns **heading = NaN** (no arbitrary 0). |
| Provenance | `sha256_*`, `HashCache`, `make_cache_key`, `write_sidecar`, `cache_decision`, `kpms_provenance`, `write_json_atomic`, `NumpyEncoder`. |
| Diagnostics | `rms_point_difference` (scored representation), `confidence_aware_rms` (observed points only, never zero-filled). |

---

## 2. Remaining implementation plan (agreed design; none of it is written)

### 2.1 Selector: rewrite `04/select_cohort_static_21kp_v2_syllable_manifest.py`
- **Heavy imports lazy.** Keep `build_arg_parser()` at module level so the runner wiring can be tested without jax.
- **New args:** `--grid_size`, `--fps` (default read from `cohort_static_21kp_v2/config.yml`, `fps: 60`), `--score_pre_seconds`, `--score_post_seconds`, `--movie_pre_seconds`, `--movie_post_seconds`, `--min_bout_seconds`, `--filter_size 9`, `--conf_threshold` (default from config, 0.5), QC fraction/gap args, `--viz_dir` (alias the old `--output_dir`), `--manifest_name`, `--syllables` (partial smoke run: background PCA must still use all eligible bouts; mark the manifest `partial: true`), `--seed 42`. Drop `--pre/--post`.
- **Per segment:**
  1. Load labels, centroid and heading from `results.h5`; Y from `preprocessed_data.pkl`; VFI from the npz.
  2. Load raw fused tracks for `base_session_id(seg)` (`load_fused_tracks`), cached in memory (~3 GB).
  3. Compute `valid_raw`, then `valid_model = valid_raw[vfi]` (clip indices; out-of-range = invalid).
  4. Filter pose once per segment.
- **Eligibility cascade (first failure wins; count every reason per syllable):** `length_mismatch`/`no_video_pair` (segment level) -> `too_short` -> `window_out_of_segment` -> `model_pose_nonfinite` -> `missing_source_frames` -> `scored_features_nonfinite` -> `qc_dorsal` -> `qc_paws`.
- **Features:** `egocentric_window(Y[a:b], c0_filtered, h0_filtered)` flattened (40x21x2 = 1680 dims). Scored window is `[start-pre, start+post)`.
- **QC:** scored-window metrics gate eligibility. Active-execution metrics `[start, end)` are diagnostics only. Store compactly as 42-float vectors (fraction and gap per keypoint).
- **Tiers from the eligible count.** Record `n_original_bouts` and `n_eligible_bouts`, and keep `total_cohort_bouts` = eligible for gallery compatibility. Add a per-syllable `rejections` dict.
- **Global PCA:** same settings as installed KPMS (4 PCs, 50 neighbors, <=50,000 background samples). Concatenate features into one array; subsample by index (no second full copy). Record `explained_variance_ratio`.
- **Bout record fields:** `bout_id`, `is_anchor`, `anchor_type` (`density_exemplar` | `compact_medoid` | `all_bouts_medoid`), `is_exemplar` (tier 1 only), `is_medoid` (= is_anchor, back-compat), `duration_frames`/`duration_seconds`, `source_top_onset`, `source_top_offset = vfi[end-1]+1` (exclusive), `fraction_score_window_in_label`, `pca_neighbor_rank`, `pca_distance_to_exemplar`, `full_distance_to_anchor`, `qc` (scored and active, per-paw), and `alignment` (raw vs filtered onset heading delta in degrees and centroid shift in px).
- **Manifest top level:** `schema_version: 3`, `fps`, `score_window`, `movie_window`, `trajectory_window` (frames alias), `alignment`, `qc` config, `density` settings, `kpms` provenance (`kpms_provenance()`), grid size, tier definitions.
- **Side outputs (atomic writes, in `viz_dir`):**
  - `neighbor_diagnostics.json`: for each tier-1 syllable, the exemplar plus its 50 neighbors with PCA rank/distance, full-space L2, dorsal/paw RMS (scored) and confidence-aware observed RMS, QC, duration, label fraction, `in_final_G`. Use nearest-neighbor queries only, no NxN matrices.
  - `qc_threshold_sensitivity.json`: coverage over fraction {0.6, 0.8, 0.9, 1.0} x gap {0.05, 0.1, 0.2, inf} s.
  - `selection_rejections.json`.
- Call `assert_unique_bout_ids` before writing.

### 2.2 Grid renderer: edit `04/render_all_cohort_static_21kp_v2_seed101_grid_movies.py`
- Delete the local `compute_robust_bot_pose` and `is_bout_tracking_valid`; import from common.
- **Delete the whole substitution block and the `bouts_by_syllable` indexing** (this also removes a slow pre-pass).
- Use `resolve_display_bouts` with a target validator (label integrity `np.all(z[start:end]==s)`, duration equals `end-start`). Then `assert_rendered_equals_manifest`.
- `grid_rows, grid_cols = grid_layout(manifest["grid_size"])`.
- Movie pre/post come from `manifest["movie_window"]`. Use the filtered per-segment pose (cached by segment) for the top crop and the indicator dot.
- **Badge:** only for `is_anchor`, with text from `anchor_type`.
- **Context padding:** keep edge-padding but mark padded frames visibly (`PAD`).
- **Dropped BotCam frames** (`top_to_bot_frame_continuous` returns -1): currently silently shows `bot_reader[0]`. Replace with the last valid bottom frame, dimmed, with a `BOT DROPPED` badge. This is the same class of silent substitution.
- **Bot orientation:** if the heading is NaN, render unrotated and flag `BOT UNROTATED`; if the pose is invalid, flag `BOT POSE MISSING`. Never an unmarked zero-heading crop.
- **Cache:** replace the "mp4 > 10 KB" test with `make_cache_key` / `cache_decision` plus `write_sidecar`. Per-tile sidecar contents: `bout_id`, status, exact Top frame list, exact Bot frame list (-1 for drops), pad counts, flags. Add `--viz_dir`, `--no_cache`.

### 2.3 Trajectory renderer: rewrite `04/render_all_cohort_static_21kp_v2_seed101_trajectories.py`
- Consume the final manifest. Use observed fused tracks (not the 8.9 GB pickle): low-confidence and missing points are gaps. Use the same filtered onset pose as the selector. Native timing, `fps` from the manifest.
- Keep the existing `superimposed/` and `side_by_side/` PDF/PNG/GIF names. Content is now the **actual anchor execution**, no 15-step interpolation and no 0.6/0.4 blend. Frames after the bout end are marked faded as "after bout end"; pre-onset frames are marked "context".
- New `individual_executions/syllable_XX_individual_executions.{png,pdf}`: G panels in the movie's grid layout, shared axis limits, anchor flagged, bout IDs in titles, post-end segments dashed.
- Provenance sidecar per output.
- GIF delay is quantised to 10 ms units by the format (~20 ms/frame for 60 fps); note it in the docstring.

### 2.4 Gallery: edit `04/build_cohort_static_21kp_v2_seed101_labeling_gallery.py`
- Add `--viz_dir`; take G and windows from the manifest (the "3x4"/"12" strings are hardcoded).
- **Link a movie or plot only when its sidecar cache key matches the current manifest syllable hash**; otherwise show a STALE marker.
- Show eligible/original counts, add a link to the individual-executions PNG, and write a gallery provenance record.
- Also replace its hardcoded `60.0` fps with the manifest value.

### 2.5 Runner: edit `04/run_all_cohort_static_21kp_v2_seed101_visualizations.sh`
- Today it calls the selector with only `--grid_size 12` (so 5/15 defaults silently apply) and forwards `"$@"` to two scripts with different parsers.
- Parse once, then build **separate explicit argument arrays** per script. Pass `--viz_dir` (default `visualizations_cohort_static_21kp_v2_seed101_v3`) and the FPS/window settings to the selector only.
- Echo the resolved settings and verify them against the written manifest before rendering (fail loudly if they differ).

### 2.6 Tests: create `tests/test_v2_visualization_repairs.py` (pytest-style, matching repo)
Turn each reproduction in the audit into an assertion of the **repaired** behavior:
1. A/B/C with A rejected -> A becomes a flagged blank, B and C unchanged, no duplicates; the flag does not transfer.
2. Bout bounds: `[(0,10),(10,50),(50,60)]`; constant sequence; transitions at either boundary; empty sequence; exact 40-frame bout has duration 40.
3. Tier 1: exemplar equals the installed-KPMS ratio argmax on identical inputs. Trick: monkeypatch `np.random.choice` inside `keypoint_moseq.util` to capture `indices[exemplar]` (first arg), then compare with `select_tier1_density`. The group is the exemplar first plus the nearest G-1.
4. 60 identical feature vectors: the exemplar is first and present; the rest ordered by stable ID.
5. QC: good BotCam nose/tail with all four paws missing is rejected; `compute_robust_bot_pose` paw fallback works with real names and returns NaN heading.
6. Heading glitch: an isolated 90 degree glitch inside a segment is removed by `filter_segment_pose`; onset pose is read from the full-segment filter.
7. `egocentric_window` equals `kpms.util.inverse_rigid_transform`.
8. Timing: 60 fps -> 10+30; 30 fps -> 5+15; the runner passes explicit settings (parse the `.sh`); `build_arg_parser` accepts them.
9. `cache_decision`: changing the manifest syllable hash, results hash or code hash forces re-render; an existing large file with no sidecar is **not** reused.
10. `grid_layout`: 24 -> 24 slots.
11. If the production manifest exists: unique IDs, tile 0 is the anchor, all onsets >= `pre_frames`, durations >= minimum. Skip if absent.

Notes:
- Tests needing kpms must `pytest.importorskip`.
- `tests/conftest.py` imports `fiberphotometry.data.session_loading`; confirm pytest collects under the `kpms_gpu_cuml` env first.

### 2.7 Verification runs (after the code is written)
- Use a fresh `--viz_dir`. First a partial selector run and render for syllable 1 and syllable 67 (67 contains early-onset bouts; under the new rules they must be eligible and padded, not rejected), then the full run if those pass.
- Memory is fine here (125 GiB RAM; nothing else heavy was running).
- Run the selector with `kpms_gpu_cuml` and CPU forced (`CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`).

---

## 3. Non-trivial things discovered

1. **Raw confidence is available in the fused files**, not in the preprocessed pickle. `data/cohort_fused_static_21kp_v2/{base_sid}_tandem_21kp.h5` keeps DLC likelihoods verbatim (`static_fusion_engine.py`: "NO anchor-confidence coupling"). Row *i* is TopCam frame *i*. The training ingestion (`run_cohort_static_21kp_v2_production.py`) builds each segment's VFI as `np.arange(s_idx, e_idx)`, so `vfi[segment_frame]` indexes fused rows directly. The camera correspondence, epoch affine, reflection and the `PAW_MAP` bilateral swap are already baked into the fused paw columns, so joining QC through VFI reuses them with no re-derivation.
2. Existing confidence config: `conf_threshold: 0.5` in `cohort_static_21kp_v2/config.yml`; training segmentation used `ventral_conf_thresh=0.4` on the paws. Chosen QC fraction (0.8) and gap (0.1 s) thresholds are **my starting point, not validated**. They must be reviewed against `qc_threshold_sensitivity.json`.
3. **Installed KPMS 0.6.7** (jax_moseq 0.3.3, sklearn 1.7.2, numpy 2.2.6). `sample_instances` confirms: 4 PCs, 50 neighbors, background subsample of 50,000, `local_density = 1/distances.mean(1)` with self included, then `np.argmax(local/global)`, then it **randomly** samples G from the exemplar's 50 neighbors. The V2 "nearest G around the exemplar" is the deliberate deviation to keep. KPMS has no eps; the old V2 selector added `1e-8`; the new code floors only exact-zero means.
4. **Segment-name handling:** `results.h5` keys are segment names `..._segNNN`. Fused files, `find_matched_pairs` and `top_to_bot_frame_continuous` use or normalise the base session id (strip `_seg\d+`).
5. **Tile 0 is not necessarily the exemplar** in the old code (KNN ties can omit it, and renderer substitution inherits `is_medoid`). The renderer's `b_idx == 0 or is_medoid` badge logic therefore must change to `is_anchor` only.
6. The old renderer also pads the movie context by **repeating the segment's edge frame** (`np.pad(..., mode="edge")`), and shows `bot_reader[0]` for dropped Bot frames. Both are silent. Replace with flagged versions (see 2.2).
7. The old selector's `ends` are the last included frame, so durations, `source_top_offset` and the active-indicator are all one frame short. The gallery builder's own bout loop already uses the correct half-open bounds; only selector and renderer were wrong.
8. 7 selected bouts in the old manifest start before frame 60 (syllables 67, 36, 2). Scoring needs only 10 pre-frames at the new setting, so they become eligible; the movie context is padded and flagged, not substituted.
9. Selector memory/time: features are 1680 floats x ~300k+ bouts ~ 2+ GB; avoid `np.array(list)` copies. The 8.9 GB pickle load dominates runtime. Moving the trajectory renderer to observed fused tracks removes that load from the plotting step.
10. `usage_pct` in the old selector counted only sessions with video pairs; the new plan counts all segments for usage (small change; mention in the report).
11. Only plain GIF timing caveat: GIF delay is stored in 10 ms steps.

---

## 4. Checks and hypotheses left for another agent (not done on purpose)

Confirmed-by-source but **prevalence unmeasured** in production; do these after the code lands:

1. **Measure rejection counts** per reason, overall and for syllable 1 and 67; review the QC sensitivity table and decide thresholds.
2. **Syllable-1 neighbor table:** from `neighbor_diagnostics.json` check the exemplar plus 50 neighbors: PCA explained variance, neighborhood radius, full 21kp distances, dorsal vs paw RMS in px, confidence-aware observed RMS, duration and fraction of window in label, whether in the displayed G.
3. **Assert rendered IDs == manifest IDs** in the rendered sidecars for every syllable; count flagged blanks (expected 0).
4. **Four-PC vs full 2D:** if four-PC neighbors disagree in full-keypoint space, investigate projection/weighting before blaming missing 3D. If full-2D executions agree but videos differ, then consider 3D or landmarks not encoded.
5. **Fixed-window scoring vs bout length:** 260 of 1,169 old selections were shorter than the 15-frame post window; check how many selections are shorter than 30 post-frames at the new setting (scored slice then contains the next label).
6. **BotCam crop alignment:** the bottom crop still uses its own nose/tail axis. Validate agreement with the fused body axis. If composing a common crop, use `M_bot = M_top @ H_bot_to_top` with the exact raw-Bot -> fused-Top transform (reflection about `arena_width`, then epoch affine `A, b`). Do not add a second horizontal flip (the renderer currently does `cv2.flip(tile_bot, 1)` after warping; check this against the fusion reflection). Do not alter fusion calibration.
7. **Raw vs filtered onset pose** on actual selected bouts (heading delta distribution in the manifest `alignment` fields).
8. **Video-reader check:** the renderer monkeypatches `OpenCVReader.read` (`_fast_read`, forward gap skip <= 30). Compare against plain sequential decoding at onset and neighbors, including backward seeks, on actual files; record file paths and library versions.
9. **Overlay tracked dorsal/paw points on the synchronized videos** around onset and the active execution (not implemented).
10. **Observed vs scored trajectories:** the trajectory plots will use observed fused points; confirm they match the scored (preprocessed) representation on the selected bouts.
11. Possible separate upstream issues (document, do not fix here): whether fully-missing-paw segments reached training (ingestion only needs one paw >= 0.4 per frame); whether interpolated/preprocessed `Y` hides uncertainty.

---

## 5. Quick-start for the next agent

```bash
cd /home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/04_reports_and_figures
PY=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
$PY -c "import viz_v2_common as v; print(v.half_open_bouts([0]*10+[1]*40+[2]*10))"   # sanity
```
Audit evidence: `density_sampling_v2_audit.zip` (repo root): `density_sampling_v2_audit.md`, `reproduce_dossier_findings.py`, `reproduction_results.json`. Its assertions demonstrate the OLD bugs; the new tests must assert the repaired behavior instead.

Do not touch model results, `preprocessed_data.pkl`, or fusion calibration. The old gallery in `visualizations_cohort_static_21kp_v2_seed101/` stays untouched; all new output goes to `..._seed101_v3/`.
