"""Compose result, pose and photometry adapters into legacy tandem session exports.

The named legacy policy uses row-prefix camera sampling and a 60 Hz frame clock.
VFI selection, calibrated camera correspondence and gap-aware resampling remain
separate scientific choices; this workflow does not infer them from filenames.
"""

from contextlib import ExitStack
import os
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import pandas as pd

from fpcode.adapters.kpms_results import open_kpms_results, results_to_frame_table
from fpcode.adapters.legacy_sessions import (
    LEGACY_TANDEM_PHOTOMETRY_COLUMNS, attach_export_tables, project_legacy_photometry_table,
)
from fpcode.behavior.kinematics import legacy_nose_tail_kinematics
from fpcode.common.cache_manifest import atomic_path
from fpcode.common.io import safe_load_pickle
from fpcode.common.logging import get_logger
from fpcode.photometry.alignment import combined_photometry_and_syllables

logger = get_logger("fpcode.workflows.session_export")


def _read_prefix_kinematics(path, n_frames, nose):
    table = pd.read_hdf(path)
    scorer = table.columns.levels[0][0]  # Historical export's first-scorer convention.
    anchors = [table.loc[:, [(scorer, part, "x"), (scorer, part, "y")]]
               .iloc[:n_frames].to_numpy(dtype=float) for part in (nose, "tail_base")]
    return legacy_nose_tail_kinematics(*anchors)


def _behavior_table(recording_key, syllables, top, bot, reference):
    index, labels = syllables.index, syllables["syllable"]
    n_frames = len(index)
    heading = syllables["heading"].to_numpy()
    onset = np.concatenate([[True], labels.to_numpy()[1:] != labels.to_numpy()[:-1]])
    data = {
        "frame_index": index.to_numpy(), "name": [recording_key] * n_frames,
        "centroid_x": top["centroid"][:, 0], "centroid_y": top["centroid"][:, 1],
        "heading": heading, "heading_top": top["heading"], "heading_moseq": heading,
        "angular_velocity": top["angular_velocity"], "velocity_px_s": top["velocity"],
        "velocity_px_s_raw": top["velocity_raw"], "syllable": labels,
        "tandem_syllable": labels, "group": ["default"] * n_frames, "onset": onset,
        "centroid_x_top": top["centroid"][:, 0], "centroid_y_top": top["centroid"][:, 1],
        "angular_velocity_top": top["angular_velocity"], "velocity_px_s_top": top["velocity"],
        "velocity_px_s_top_raw": top["velocity_raw"],
        "centroid_x_bot": bot["centroid"][:, 0], "centroid_y_bot": bot["centroid"][:, 1],
        "heading_bot": bot["heading"], "angular_velocity_bot": bot["angular_velocity"],
        "velocity_px_s_bot": bot["velocity"], "velocity_px_s_bot_raw": bot["velocity_raw"],
    }
    if reference is not None and "syllable" in reference.all_data_df.columns:
        data["top_syllable"] = reference.all_data_df["syllable"].iloc[:n_frames].to_numpy()
    return pd.DataFrame(data, index=index)


def _photometry_table(session, index):
    has_fp = isinstance(getattr(session, "photometry", None), dict) and "470" in session.photometry
    if has_fp:
        try:
            return project_legacy_photometry_table(combined_photometry_and_syllables(session), index), True
        except Exception as error:
            logger.warning(f"Photometry fallback for {session.session_id}: {error}")
    return pd.DataFrame(np.nan, index=index, columns=LEGACY_TANDEM_PHOTOMETRY_COLUMNS), False


def _validate_destinations(outputs, sources):
    if len(set(outputs)) != len(outputs):
        raise ValueError("export output paths must be distinct")
    for output in outputs:
        for source in sources:
            same_file = output.exists() and source.exists() and os.path.samefile(output, source)
            if output == source or same_file:
                raise ValueError(f"export output aliases an input: {output}")


def _publish_session_pickles(sessions, outputs):
    # Stage both outputs before publishing either. Each replacement is atomic;
    # two filesystem names cannot be committed as one atomic transaction.
    with ExitStack() as stack:
        primary = stack.enter_context(atomic_path(outputs[0]))
        with primary.open("wb") as stream:
            pickle.dump(sessions, stream)
        if len(outputs) > 1:
            secondary = stack.enter_context(atomic_path(outputs[1]))
            try:
                os.link(primary, secondary)
            except OSError:
                shutil.copyfile(primary, secondary)


def export_tandem_sessions(results_h5, dirty_pkl, output_pkl, *, pairs,
                           output_data_pkl=None, k240k_ref_pkl=None):
    """Export existing session objects using explicit matched tracking inputs.

    Preserves the 23-keypoint recording suffix, nullable label tables, camera
    prefix slicing, historical photometry interpolation, fallback counts and
    pickle class identities. Supplied pair paths bypass filesystem discovery.
    """
    started = time.time()
    results_h5, dirty_pkl, output_pkl = (Path(path).resolve() for path in (results_h5, dirty_pkl, output_pkl))
    for path in (results_h5, dirty_pkl):
        if not path.is_file():
            raise FileNotFoundError(f"Export input not found: {path}")
    pairs = list(pairs)
    pair_map = {pair["sid"]: pair for pair in pairs}
    outputs = [output_pkl] + ([Path(output_data_pkl).resolve()] if output_data_pkl else [])
    sources = [results_h5, dirty_pkl]
    sources.extend(Path(pair[key]).resolve() for pair in pairs for key in ("top_h5", "bot_h5"))
    if k240k_ref_pkl:
        sources.append(Path(k240k_ref_pkl).resolve())
    _validate_destinations(outputs, sources)
    sessions = [session for session in safe_load_pickle(dirty_pkl) if session.session_id in pair_map]
    references = {}
    if k240k_ref_pkl and Path(k240k_ref_pkl).is_file():
        try:
            references = {session.session_id: session for session in safe_load_pickle(k240k_ref_pkl)
                          if session.session_id in pair_map}
        except Exception as error:
            logger.warning(f"Could not load K240K reference: {error}")

    modeled, skipped, present, missing = [], [], 0, 0
    with open_kpms_results(results_h5) as results:
        for session in sessions:
            sid, pair = session.session_id, pair_map[session.session_id]
            recording_key = f"{sid}_tandem_23kp"
            if recording_key not in results:
                skipped.append(sid)
                continue
            syllables = results_to_frame_table(results[recording_key])
            top = _read_prefix_kinematics(pair["top_h5"], len(syllables), "nose")
            bot = _read_prefix_kinematics(pair["bot_h5"], len(syllables), "nose_tip")
            behavior = _behavior_table(recording_key, syllables, top, bot, references.get(sid))
            attach_export_tables(session, recording_key=recording_key, syllables=syllables)
            photometry, available = _photometry_table(session, syllables.index)
            present += available
            missing += not available
            attach_export_tables(session, recording_key=recording_key, syllables=syllables,
                                 all_data=pd.concat([behavior.reset_index(drop=True), photometry], axis=1))
            modeled.append(session)
    _publish_session_pickles(modeled, outputs)
    return {"output_pkl": str(output_pkl), "total_sessions": len(modeled), "skipped_sessions": skipped,
            "photometry_present": present, "photometry_nan": missing,
            "elapsed_seconds": time.time() - started}
