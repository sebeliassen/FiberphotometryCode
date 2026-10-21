"""
syllable_generation/videogen/config.py

Configuration constants for video generation.
Includes dimensions, file paths, project-specific overrides, and visual settings.
"""
from pathlib import Path

from fpcode.common.paths import get_repo_root

# --- Dimension Settings ---
# Final video layouts:
# 1. Duo: 2 panels side-by-side (1600x800)
# 2. Trio: Syllable (top-left), Video (top-right), Photometry (bottom full-width) (1600x1600)

QUADRANT_WIDTH = 800
QUADRANT_HEIGHT = 800

# Duo Layout (Side-by-Side)
COMPOSITE_WIDTH = QUADRANT_WIDTH * 2
COMPOSITE_HEIGHT = QUADRANT_HEIGHT

# Trio Layout (Stacked)
TRIO_WIDTH = QUADRANT_WIDTH * 2
TRIO_HEIGHT = QUADRANT_HEIGHT * 2

TARGET_FPS = 30  # Output FPS

# Sampling Rates (all_data_df frequency) vary by project
PROJECT_SAMPLING_RATES = {
    "IFDN": 80,
    "demo_project": 80,
    "jonathan_amph": 80,
    "amph_project": 80,
}

DEFAULT_SAMPLING_RATE = 80

# --- Paths ---
PROJECT_ROOT = Path(str((get_repo_root() / 'compat/syllable_generation')))
KPMS_ROOT = (get_repo_root() / 'vendor/keypoint_moseq_syll')

# Mapping of project names to their trajectory plot directories
PROJECT_TO_PDF_DIR = {
    "jonathan_amph": KPMS_ROOT / "flipped_double_project/double_flip_activity_k100k/trajectory_plots",
    "amph_project": KPMS_ROOT / "amph_project/amph_prod_k120k_run_1 copy/trajectory_plots",
    "IFDN": KPMS_ROOT / "demo_project/full-run/trajectory_plots",
    "demo_project": KPMS_ROOT / "demo_project/full-run/trajectory_plots",
    "flipped_double_project": KPMS_ROOT / "flipped_double_project/double_flip_activity_k100k/trajectory_plots",
    "optimized_flip_project": KPMS_ROOT / "optimized_flip_project/optimized_flip_activity_k20k/trajectory_plots",
}

# Tracking Validations
TRACKING_HEADING_OFFSET = 0 # Radians to add to heading (e.g., np.pi for 180 flip)

DEFAULT_SYLLABLE_PDF_DIR = PROJECT_TO_PDF_DIR["demo_project"]
SYLLABLE_DPI = 300  # High quality for 800x800 grid

def get_syllable_pdf_dir(project_name: str = None) -> Path:
    if project_name and project_name in PROJECT_TO_PDF_DIR:
        return PROJECT_TO_PDF_DIR[project_name]
    return DEFAULT_SYLLABLE_PDF_DIR

# Directory containing the raw AVI behavior videos
BEHAVIOR_VIDEO_DIR = Path("/mnt/hdd/Jonathan_IFxDN/AllTopcams/SplitWithGPU")

# --- Visual Settings ---
BACKGROUND_COLOR = (255, 255, 255) # White for some grids if needed
TEXT_COLOR = (0, 0, 0)
FONT_SCALE = 1.0
FONT_THICKNESS = 2

# Photometry Settings
PLOT_WINDOW_SIZE = 800 # Frames to show (10 seconds at 80Hz)
PLOT_LINE_COLOR = '#2ca02c' # Default fallback
PLOT_BG_COLOR = 'black'
PLOT_TEXT_COLOR = 'white'

BRAIN_REG_TO_COLOR = {
    'VS': 'purple',
    'DMS': 'forestgreen',
    'DLS': 'C0'
}
