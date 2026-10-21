from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Analysis Config
WINDOW_SECONDS = 5.0
FPS = 80
WINDOW_FRAMES = int(WINDOW_SECONDS * FPS)  # 400 frames per side
TOTAL_WINDOW_LEN = WINDOW_FRAMES * 2 + 1   # center + left + right

# Plotting Config
BRAIN_REG_TO_COLOR = {
    'VS': 'purple',
    'DMS': 'forestgreen',
    'DLS': 'C0'
}

# Column Regex (Broad enough for zscore, zscore_datta, dff)
SIGNAL_COL_REGEX = r"^signal_\d+_([a-z_]+)$"

# Syllable Config
# Assuming syllables are integers in a column named 'syllable'
