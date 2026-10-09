import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any

# Relative imports assuming this is in analysis/ulrik_kinematics/lib/
# We need to reach back to analysis/replications/panel_1z/lib or similar for the base loader
# OR use the project base loader. 
# Based on context, `syllable_generation.analysis.replications.panel_1z.lib.loader` seems robust.

import sys
# Hack to ensure we can import from the root if needed
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.replications.panel_1z.lib import loader as base_loader
from analysis.replications.panel_1z.lib import processing as base_processing

logger = logging.getLogger("Ulrik_Loader")

def load_dataset(project_name: str, 
                 base_dir: str = ROOT_DIR,
                 amph_post_injection_delay_min: float = 5.0) -> List[Any]:
    """
    Load and standardize the dataset for Ulrik Kinematics analysis.
    
    Args:
        project_name: "ifdn" or "amph"
        base_dir: Root directory of the repository
        amph_post_injection_delay_min: Minutes to discard after injection (AMPH only)
    
    Returns:
        List of processed Session objects.
    """
    
    # 1. Resolve Path
    # Robust handling: If base_dir already ends in 'syllable_generation', don't repeat it.
    if os.path.basename(os.path.normpath(base_dir)) == "syllable_generation":
        data_rel = "data"
    else:
        # Assume base_dir is repo root
        data_rel = os.path.join("syllable_generation", "data")
        
    if project_name.lower() == "ifdn":
        pkl_path = os.path.join(base_dir, data_rel, "jonathan_ifdn_sessions_w_sylls.pkl")
    elif project_name.lower() == "amph":
        pkl_path = os.path.join(base_dir, data_rel, "jonathan_amph_sessions_v2_w_sylls.pkl")
    else:
        raise ValueError(f"Unknown project: {project_name}")
        
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle not found: {pkl_path}")
        
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
        logger.info(f"Applying AMPH Post-Injection Filter (delay={amph_post_injection_delay_min} min)...")
        initial_count = len(sessions)
        
        # This function cuts the dataframe to [injection_frame + delay_frames : ]
        sessions = base_processing.filter_post_injection(sessions, delay_mins=amph_post_injection_delay_min)
        
        # Verify we didn't lose everyone
        valid_sessions = [s for s in sessions if hasattr(s, "all_data_df") and not s.all_data_df.empty]
        logger.info(f"Retained {len(valid_sessions)}/{initial_count} valid sessions after filtering.")
        sessions = valid_sessions
        
    elif project_name.lower() == "ifdn":
        # For IFDN, we might want to ensure a standard length or just keep full sessions?
        # Rigor Manifesto says: "same session duration".
        # However, usually open field is standard length. 
        # Check if we should trim? For now, keep full session to maximize N frames.
        pass
        
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
            "Duration_Seconds": len(s.all_data_df) / m.get("fs", 30.0) if hasattr(s, "all_data_df") else 0
        })
    return pd.DataFrame(rows)
