import logging
import os
import sys

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepResearch_Loader")

# Add Panel 1z path to sys.path
REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation"
PANEL_1Z_LIB = os.path.join(REPO_ROOT, "analysis", "replications", "panel_1z", "lib")

if PANEL_1Z_LIB not in sys.path:
    sys.path.append(PANEL_1Z_LIB)

try:
    import loader as base_loader
    import processing as base_processing
except ImportError as e:
    logger.error(f"Failed to import base loaders from {PANEL_1Z_LIB}: {e}")
    raise

def load_sessions(project_name: str, apply_filter: bool = True):
    """
    Load sessions for 'ifdn' or 'amph' using robust panel_1z logic.
    - Standardizes regions
    - Enriches with kinematics & metadata
    - Filters AMPH post-injection (5 min default)
    """
    project_name = project_name.lower()

    # 1. Base Load
    # base_loader automatically finds the latest pickle
    logger.info(f"Loading latest pickle for project: {project_name}...")
    sessions = base_loader.load_sessions(project=project_name)

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
