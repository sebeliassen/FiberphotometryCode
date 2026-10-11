import logging
import os

# Relative imports assuming this is in analysis/ulrik_kinematics/lib/
# We need to reach back to analysis/replications/panel_1z/lib or similar for the base loader
# OR use the project base loader.
# Based on context, `syllable_generation.analysis.replications.panel_1z.lib.loader` seems robust.
import sys
from typing import Any, List

import numpy as np
import pandas as pd

from fpcode.common.paths import get_repo_root

# Hack to ensure we can import from the root if needed
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(str((get_repo_root() / 'compat/syllable_generation')))

from analysis.replications.panel_1z.lib import loader as base_loader
from analysis.replications.panel_1z.lib import processing as base_processing

logger = logging.getLogger("Ulrik_Loader")

def load_dataset(project_name: str,
                 base_dir: str = ROOT_DIR,
                 amph_post_injection_delay_min: float = 5.0,
                 amph_phase: str = None,
                 ifdn_phase: str = None) -> List[Any]:
    """
    Load and standardize the dataset for Ulrik Kinematics analysis.
    
    Args:
        project_name: "ifdn" or "amph"
        base_dir: Root directory of the repository
        amph_post_injection_delay_min: Minutes to discard after injection (AMPH only)
        amph_phase: Optional string, 'salineamph' or 'amphamph' to slice data frames
        ifdn_phase: Optional string, '5min_end' to slice data frames starting at 5 minutes
    
    Returns:
        List of processed Session objects.
    """

    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

    # 1. Path Resolution
    if project_name.lower() == "ifdn":
        data_rel = 'data/sessions'
        pkl_path = os.path.join(base_dir, data_rel, "jonathan_ifdn_sessions_v2_w_sylls.pkl")
        if not os.path.exists(pkl_path):
            pkl_path = os.path.join(base_dir, data_rel, "jonathan_ifdn_sessions_w_sylls.pkl")
    elif project_name.lower() == "amph":
        data_rel = 'data/sessions'
        pkl_path = os.path.join(base_dir, data_rel, "jonathan_amph_sessions_v2_w_sylls.pkl")
    elif project_name.lower() == "karen":
        data_rel = 'data/sessions'
        pkl_path = os.path.join(base_dir, data_rel, "sessions_karen_project_w_sylls.pkl")
    else:
        raise ValueError(f"Unknown project: {project_name}")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle not found: {pkl_path}")

    if project_name.lower() == "amph" and amph_phase is None:
        logger.info("Automatically loading AMPH dataset as pre- and post-injection phases...")
        pre_sessions = load_dataset("amph", base_dir=base_dir, amph_post_injection_delay_min=amph_post_injection_delay_min, amph_phase="pre")
        post_sessions = load_dataset("amph", base_dir=base_dir, amph_post_injection_delay_min=amph_post_injection_delay_min, amph_phase="post")

        for s in pre_sessions:
            s.metadata["drug"] = "Saline"
            s.session_id = f"{s.session_id}_pre"
        for s in post_sessions:
            s.metadata["drug"] = "Amphetamine"
            s.session_id = f"{s.session_id}_post"

        all_sessions = pre_sessions + post_sessions
        # Centrally filter out UNK genotypes
        all_sessions = [s for s in all_sessions if str(s.metadata.get("genotype", "UNK")).upper() in ["WT", "KI"]]
        return all_sessions

    logger.info(f"Loading {project_name} from {pkl_path}...")

    # 2. Base Load
    # load_sessions handles the deserialization
    sessions = base_loader.load_sessions(pkl_path=pkl_path)
    logger.info(f"Loaded {len(sessions)} raw sessions.")

    # 3. Enrich (Metadata Standardization)
    # This adds 'group', 'genotype', ensures valid animal_ids
    sessions = base_processing.enrich_sessions(sessions)

    # 4. Project-Specific Filtering
    if project_name.lower() == "amph":
        if amph_phase is not None:
            logger.info(f"Applying Custom Phase Filter: {amph_phase}")
            initial_count = len(sessions)
            valid_sessions = []

            for s in sessions:
                df = getattr(s, "all_data_df", None)
                if df is None or df.empty: continue

                fs = s.metadata.get('fs', 80.0)
                inj_frame = s.metadata.get("injection_frame")

                if inj_frame is None: continue

                if amph_phase == "pre":
                    # Baseline phase: 5 min into session until 5 min before injection
                    start_frame = int(5 * 60 * fs)
                    end_frame = int(inj_frame - 5 * 60 * fs)

                    # Verify slice is mathematically possible
                    if end_frame > start_frame:
                        s.all_data_df = df.iloc[start_frame:end_frame].copy()  # preserves absolute frame index
                        valid_sessions.append(s)
                    else:
                        print(f"  [Warning] Baseline too short for {s.session_id}, skipping 'pre' slicing.")

                elif amph_phase == "post":
                    # Stable drug phase: 15 min after injection until 45 min after injection
                    start_frame = int(inj_frame + 15 * 60 * fs)
                    end_frame = int(inj_frame + 45 * 60 * fs)

                    # Cap at end of dataframe if the session ended early
                    end_frame = min(end_frame, len(df))

                    if end_frame > start_frame:
                        s.all_data_df = df.iloc[start_frame:end_frame].copy()  # preserves absolute frame index
                        valid_sessions.append(s)
                    else:
                        print(f"  [Warning] Data too short for {s.session_id}, skipping 'post' slicing.")

                elif amph_phase == "both":
                    start_frame_pre = int(5 * 60 * fs)
                    end_frame_pre = int(inj_frame - 5 * 60 * fs)
                    start_frame_post = int(inj_frame + 15 * 60 * fs)
                    end_frame_post = int(inj_frame + 45 * 60 * fs)
                    end_frame_post = min(end_frame_post, len(df))

                    if end_frame_pre > start_frame_pre and end_frame_post > start_frame_post:
                        df_pre = df.iloc[start_frame_pre:end_frame_pre]
                        df_post = df.iloc[start_frame_post:end_frame_post]
                        s.all_data_df = pd.concat([df_pre, df_post]).copy()
                        valid_sessions.append(s)
                    else:
                        print(f"  [Warning] Data too short for {s.session_id}, skipping 'both' slicing.")
            sessions = valid_sessions
            logger.info(f"Retained {len(sessions)}/{initial_count} valid sessions after phase filtering.")

        elif amph_post_injection_delay_min is not None:
            logger.info(f"Applying AMPH Post-Injection Filter (delay={amph_post_injection_delay_min} min)...")
            initial_count = len(sessions)

            # This function cuts the dataframe to [injection_frame + delay_frames : ]
            sessions = base_processing.filter_post_injection(sessions, delay_mins=amph_post_injection_delay_min)

            # Verify we didn't lose everyone
            valid_sessions = [s for s in sessions if hasattr(s, "all_data_df") and not s.all_data_df.empty]
            logger.info(f"Retained {len(valid_sessions)}/{initial_count} valid sessions after filtering.")
            sessions = valid_sessions
        else:
            logger.info("Skipping AMPH Post-Injection Filter (Full session loaded).")

    elif project_name.lower() == "ifdn":
        if ifdn_phase == "5min_end":
            logger.info("Applying custom 5min-to-end filter for IFDN...")
            initial_count = len(sessions)
            valid_sessions = []

            for s in sessions:
                df = getattr(s, "all_data_df", None)
                if df is None or df.empty: continue

                signal_col = None
                for col in df.columns:
                    if 'signal' in col and 'zscore' in col:
                        signal_col = col
                        break

                first_valid = 0
                if signal_col is not None:
                    valid_idx = np.where(~df[signal_col].isna())[0]
                    if len(valid_idx) > 0:
                        first_valid = valid_idx[0]

                fs = s.metadata.get("fs", 80.0)
                start_frame = int(first_valid + 5 * 60 * fs)

                if len(df) > start_frame:
                    s.all_data_df = df.iloc[start_frame:].copy()
                    valid_sessions.append(s)
                else:
                    print(f"  [Warning] Data too short for {s.session_id}, skipping.")
            sessions = valid_sessions
            logger.info(f"Retained {len(sessions)}/{initial_count} valid sessions after 5min-end filtering.")
        else:
            logger.info("Applying Acclimation-Aware Filter for IFDN (Skipping 15m, Keeping 30m)...")
            initial_count = len(sessions)
            valid_sessions = []

            for s in sessions:
                df = getattr(s, "all_data_df", None)
                if df is None or df.empty: continue

                # Find the first frame where photometry signal is valid (post-handling)
                # IFDN uses 'signal_0_zscore_datta' typically for one of the channels
                signal_col = None
                for col in df.columns:
                    if 'signal' in col and 'zscore' in col:
                        signal_col = col
                        break

                if signal_col is None:
                    valid_sessions.append(s)
                    continue

                valid_idx = np.where(~df[signal_col].isna())[0]
                if len(valid_idx) == 0:
                    print(f"  [Warning] All NaN for {s.session_id}, skipping.")
                    continue

                first_valid = valid_idx[0]
                fs = s.metadata.get("fs", 80.0)

                # 15 minutes of acclimation
                start_frame = int(first_valid + 15 * 60 * fs)
                # 30 minutes of stable behavior
                end_frame = int(start_frame + 30 * 60 * fs)

                end_frame = min(end_frame, len(df))

                if end_frame > start_frame:
                    s.all_data_df = df.iloc[start_frame:end_frame].copy()
                    valid_sessions.append(s)
                else:
                    print(f"  [Warning] Data too short after acclimation for {s.metadata.get('session_id', 'UNK')}, taking remainder.")
                    if len(df) > start_frame:
                        s.all_data_df = df.iloc[start_frame:].copy()
                        valid_sessions.append(s)

            sessions = valid_sessions
            logger.info(f"Retained {len(sessions)}/{initial_count} valid sessions after acclimation filtering.")

    # Centrally filter out UNK genotypes
    if project_name.lower() == "karen":
        sessions = [s for s in sessions if str(s.metadata.get("genotype", "UNK")).upper() in ["MONOMER", "PFF"]]
    else:
        sessions = [s for s in sessions if str(s.metadata.get("genotype", "UNK")).upper() in ["WT", "KI"]]
    return sessions

def get_metadata_df(sessions) -> pd.DataFrame:
    """
    Extract a clean metadata table for all loaded sessions.
    Useful for checking N and factorial design balance.
    """
    rows = []
    for s in sessions:
        m = s.metadata
        rows.append({
            "Animal": m.get("animal_id"),
            "Genotype": m.get("genotype"),
            "Group": m.get("group"),
            "Drug": m.get("drug", "N/A"),
            "Dose": m.get("dose_mg_kg", "N/A"),
            "Duration_Seconds": len(s.all_data_df) / m.get("fs", 80.0) if hasattr(s, "all_data_df") else 0
        })
    return pd.DataFrame(rows)
