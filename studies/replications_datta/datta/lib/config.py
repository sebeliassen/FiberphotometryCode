"""
Configuration parameters for the Datta pipeline.
Defines paths, analysis settings (bins, shuffles), and column definitions.
"""
from pathlib import Path

# --- PATHS ---
# repo_root is resolved relative to this file: datta/lib/config.py -> datta/lib -> datta -> replications -> analysis -> syllable_generation
REPO_ROOT = Path(__file__).resolve().parents[4]
DATTA_ROOT = REPO_ROOT / "analysis/replications/datta"
OUTPUT_ROOT = DATTA_ROOT / "output"
CACHE_DIR = OUTPUT_ROOT / "cache"

# Panel-Specific Output Directories
OUTPUT_DIR = OUTPUT_ROOT / "panel_1e" # Default for compatibility
PANEL_DIRS = {
    "1e": OUTPUT_ROOT / "panel_1e",
    "1e_amph": OUTPUT_ROOT / "panel_1e_amph",
    "1f": OUTPUT_ROOT / "panel_1f"
}

# Ensure dirs exist
for d in [CACHE_DIR] + list(PANEL_DIRS.values()):
    d.mkdir(parents=True, exist_ok=True)

# --- ANALYSIS PARAMETERS ---
BIN_GRID = list(range(1, 61))  # 1s to 60s bins
SEED = 123
FS = 80.0 # Default sampling rate

# --- DEBUG / SUBSET SETTINGS ---
# Set these to True/Small values for rapid verification
DEBUG_MODE = False

if DEBUG_MODE:
    NBOOT = 10
    NSHUFF = 2
    NJOBS = 1  # Serial for debug
    BIN_GRID = [1, 2, 5] # Small subset of bins
else:
    NBOOT = 500
    NSHUFF = 300
    NJOBS = -1

# --- COLUMNS & FEATURES ---
SCALAR_COLS = [
    "velocity_px_s",
    "angular_velocity",
    "acceleration",
    "jerk",
    "transition",
    "duration"
]

SCALAR_MODES = ["mean"]
RATE_METHOD = "cross"
CORR_METHOD = "pearson"

# Regex to identify signal columns (unchanged from datta)
SIGNAL_COL_REGEX = r"^signal_\d+_zscore.*$"
