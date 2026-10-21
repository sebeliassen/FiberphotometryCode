"""Project-owned loading/enrichment policy with explicit implementation owners."""

import logging

from studies.replications_datta.panel_1z.lib import loader as base_loader
from studies.replications_datta.panel_1z.lib import processing as base_processing

logger = logging.getLogger("DeepResearch_Loader")

def load_sessions(project_name: str, apply_filter: bool = True, *, pkl_path=None):
    """
    Load sessions for 'ifdn' or 'amph' using robust panel_1z logic.
    pkl_path selects an explicit trusted pickle; otherwise retain historical discovery.
    - Standardizes regions
    - Enriches with kinematics & metadata
    - Filters AMPH post-injection (5 min default)
    """
    project_name = project_name.lower()

    # 1. Base Load
    # Retain discovery unless a trusted pickle is selected explicitly.
    logger.info(f"Loading sessions for project: {project_name}...")
    sessions = base_loader.load_sessions(project=project_name, pkl_path=pkl_path)

    # 2. Enrich (Metadata, Kinematics)
    logger.info("Enriching sessions...")
    sessions = base_processing.enrich_sessions(sessions)

    # 3. Apply Signal Filtering (optional but recommended for slope analysis)
    if apply_filter:
        logger.info("Applying 1Hz Bessel filter to signals...")
        sessions = base_processing.apply_filter_to_sessions(sessions)

    # 4. Project-Specific Filtering
    if project_name == 'amph':
        logger.info("Applying AMPH Post-Injection Filter (5 min delay)...")
        sessions = base_processing.filter_post_injection(sessions, delay_mins=5.0)

    return sessions
