from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict

@dataclass()
class JonathanConfig:
    # Matching / parsing
    tolerance_seconds: float = 2.0
    fiber_col_regex: str = r"^([A-Z]+)(\d+)$"  # e.g., G0, R1, X12

    # Optional manual mapping CSV (looked up under csv_root)
    fiber_map_filename: str = "fiber_map.csv"
    phot_combined_csv_regex: str = r"^photometry_data_combined_(\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2})\.csv$"

    # filenames / discovery
    phot_combined_csv_glob: str = "photometry_data_combined_*.csv"
    phot_combined_csv_regex: str = r"^photometry_data_combined_(\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2})\.csv$"

    # columns / patterns
    signal_col_regex: str = r"^signal_\d+$"   # used by get_signal_cols

    # policy for defaults
    fiber_auto_pair_name: str = "auto_pair"   # metadata['fiber_select_mode'] default label

    def default_effective_indices(self, chamber_index: int) -> list[int]:
        # We need 4 columns per animal to capture G/Iso AND Red signals
        # Left Side:  Green (4k), Red (4k+1)
        # Right Side: Green (4k+2), Red (4k+3)
        k = int(chamber_index)
        return [4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3]

    led_col_name: str = "LedState"  # column holding LED state codes

    # map LED state code -> (letter_key, wavelength_key)
    led_state_map: Dict[int, Tuple[str, str]] = field(default_factory=lambda: {
        1: ("iso", "415"),
        2: ("G",   "470"),
        4: ("R",   "560"),
    })

    combine_freqs: Tuple[str, ...] = ("470",)
    # which letter groups to use when state is 'iso'
    iso_letters: Tuple[str, ...] = ("G", "R")

    # Modified regex: (?:.*_)? allows any prefix ending in an underscore before "Arena"
    h5_filename_regex: str = (
    r"^(?:.*_)?(?P<arena>Arena[0-9and]+)_(?P<camset>[^_]+)_(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_"
    r"(?P<token>.+?)DLC_.*\.h5$"
    )

    video_filename_regex: str = (
    r"^(?:.*_)?(?P<arena>Arena[0-9and]+)_(?P<camset>[^_]+)_(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_"
    r"(?P<token>.+?)\.(?P<ext>avi|mp4)$"
    )
    
    # Canonical chambers we recognize
    canon_chambers: Tuple[str, ...] = ("top_left", "top_right", "bottom_left", "bottom_right")


    # Camera slots by camset
    camset_to_slot: Dict[str, str] = field(default_factory=lambda: {
        "centertopcamvid": "center_cam",
        "lefttopcamvid":   "left_cam",
        "righttopcamvid":  "right_cam",
    })

    # Chamber label to stable index (used earlier in your pipeline)
    chamber_to_index: Dict[str, int] = field(default_factory=lambda: {
        "top_left": 0,
        "bottom_left": 1,
        "bottom_right": 2,
        "top_right": 3,
    })

    camera_token_map: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "lefttopcamvid":  {"upper": "top_left",  "lower": "bottom_left"},
        "righttopcamvid": {"upper": "top_right", "lower": "bottom_right"},
    })

    #"AllTimestampsCombined" discovery ---
    alltimestamps_csv_glob: str = "*AllTimestampsCombined*.csv"
    # Example: CenterTopCam_AllTimestampsCombined2024-12-13T13_37_45.csv
    alltimestamps_csv_regex: str = (
        r"^(?P<camera>.+?)_AllTimestampsCombined"
        r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})\.csv$"
    )

    #time column names (configurable) ---
    phot_system_ts_col: str = "SystemTimestamp"
    allts_system_ts_col: str = "FP3002_System_Timestamp"

    #output column name ---
    frame_index_col_name: str = "frame_idx"

    # output controls ---
    include_ransac: bool = True   # Standard Datta method (renamed from zscore_datta)
    include_wallace: bool = True  # New linear fit method
    include_manual: bool = False  # Simple subtraction (Signal - Iso), no regression
    include_zscore: bool = True   # Legacy/Global method
    include_dff: bool = True      # Legacy global dFF

    include_prominence: bool = True  # nihpp-2025 method
    prominence_basis: str = "wallace" # Bases for prominence: "wallace", "manual", or "ransac"

    def token_to_chamber(self, camset: str, token: str) -> str:
        c = (camset or "").lower()
        t = (token or "").lower()
        # explicit mapping wins
        m = self.camera_token_map.get(c)
        if m and t in m: 
            return m[t]
        # center cams accept canonical names directly
        if c == "centertopcamvid" and t in self.canon_chambers:
            return t
        # fallback: pass-through
        return t

    def cam_slot(self, camset: str) -> str:
        return self.camset_to_slot.get(camset.lower(), camset)

    def chamber_index(self, chamber: str) -> Optional[int]:
        return self.chamber_to_index.get(chamber)

    def is_canon_chamber(self, label: str) -> bool:
        return label in self.canon_chambers

    def fiber_map_path(self, csv_root: Optional[Path]) -> Optional[Path]:
        if not csv_root:
            return None
        p = Path(csv_root) / self.fiber_map_filename
        return p if p.exists() else None


# Import-friendly singleton (simple and practical for notebooks)
JCFG = JonathanConfig()