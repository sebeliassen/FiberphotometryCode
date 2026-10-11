"""Cohort defaults for historical V2 calls; new code uses explicit contexts."""

from pathlib import Path

from fpcode.viz import legacy_defaults as defaults
from fpcode.viz import pose_geometry, provenance
from fpcode.viz.context import BottomPoseLandmarks, LandmarkGroups, SelectionInputs
from fpcode.viz.selection import eligibility
from fpcode.viz.selection.bouts import base_session_id


def landmark_groups():
    return LandmarkGroups(
        dorsal=defaults.DORSAL_IDXS, paws=defaults.PAW_IDXS,
        paw_names=defaults.PAW_BODYPARTS,
        front_paws=defaults.FRONT_PAW_IDXS, hind_paws=defaults.HIND_PAW_IDXS,
    )


def selection_inputs(manifest=None, syllable=None):
    artifacts = {}
    if manifest is not None:
        records = (manifest["syllables"][str(syllable)],) if syllable is not None else manifest["syllables"].values()
        for record in records:
            for bout in record["bouts"]:
                if not bout.get("is_blank"):
                    base = base_session_id(bout["segment_name"])
                    artifacts[base] = defaults.FUSED_DIR / f"{base}_tandem_21kp.h5"
    return SelectionInputs(
        results=defaults.RESULTS_PATH, config=defaults.CONFIG_PATH,
        video_frame_indexes=defaults.VFI_PATH, preprocessed=defaults.PREPROCESSED_PATH,
        calibration_files=defaults.CALIBRATION_FILES, fused_artifacts=artifacts,
        schema_version=defaults.SCHEMA_VERSION,
    )


def fusion_geometry(base_sid, fused_dir=defaults.FUSED_DIR):
    return pose_geometry.fusion_geometry(Path(fused_dir) / f"{base_sid}_tandem_21kp.h5")


def compute_robust_bot_pose(df_bot, bot_scorer, target_idx, search_window=10, top_heading_fallback=None):
    landmarks = BottomPoseLandmarks(
        defaults.BOT_NOSE, defaults.BOT_TAIL,
        defaults.FRONT_PAW_BODYPARTS, defaults.HIND_PAW_BODYPARTS, (486.0, 486.0),
    )
    return pose_geometry.compute_robust_bot_pose(
        df_bot, bot_scorer, target_idx, search_window, top_heading_fallback, landmarks=landmarks,
    )


def summarize_window_qc(valid_window):
    return eligibility.summarize_window_qc(valid_window, groups=landmark_groups())


def summarize_frac_gap(frac, gap, W):
    return eligibility.summarize_frac_gap(frac, gap, W, groups=landmark_groups())


def source_input_hashes(cache):
    return provenance.source_input_hashes(cache, inputs=selection_inputs())


def verify_manifest_inputs(manifest, cache):
    return provenance.verify_manifest_inputs(manifest, cache, inputs=selection_inputs())


def selection_provenance_matches(sidecar, manifest, syllable, code_files=()):
    inputs = selection_inputs(manifest, syllable) if code_files else selection_inputs()
    return provenance.selection_provenance_matches(
        sidecar, manifest, syllable, code_files, inputs=inputs,
    )


def make_cache_key(kind, syllable, manifest, manifest_path, hash_cache, code_files, render_settings):
    return provenance.make_cache_key(
        kind, syllable, manifest, manifest_path, hash_cache, code_files, render_settings,
        inputs=selection_inputs(manifest, syllable),
    )
