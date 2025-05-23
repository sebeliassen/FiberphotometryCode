import os
import fnmatch
import re
import pandas as pd
from tqdm import tqdm
from fiberphotometry import config
import numpy as np
import warnings
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple, Pattern, Union
import glob

class DataContainer:
    """Container for storing and retrieving named data items of a uniform type."""
    def __init__(self, data_type: Optional[type] = None) -> None:
        """Initialize the data container with an optional data type."""
        self.data = {}
        self.data_type = data_type

    def add_data(self, name: str, data: Any) -> None:
        """Add a data item with a given name."""
        # Set the data type if not already set
        if self.data_type is None:
            self.data_type = type(data)
        elif not isinstance(data, self.data_type):
            raise TypeError(f"Data must be of type {self.data_type.__name__}, but is of type {type(data)}")

        self.data[name] = data

    def get_data(self, name: str) -> Optional[Any]:
        """Retrieve a data item by name, or None if not present."""
        return self.data.get(name)

    def remove_data(self, name: str) -> None:
        """Remove a data item by name if it exists."""
        if name in self.data:
            del self.data[name]

    def fetch_all_data_names(self) -> List[str]:
        """Return a list of all data item names."""
        return list(self.data.keys())

    def clear_data(self) -> None:
        """Remove all data items."""
        self.data.clear()


class Session:
    def __init__(
        self,
        chamber_id: str,
        trial_dir: str,
        session_guide: pd.Series,
        session_type: str,
        remove_bad_signal_sessions: bool = False
    ) -> None:
        # ─── your existing setup ───────────────────────────────────────────
        self.trial_dir     = trial_dir
        self.trial_id      = os.path.basename(trial_dir)
        self.session_guide = session_guide
        self.session_type  = session_type
        self.remove_bad_signal_sessions = remove_bad_signal_sessions

        # ─── MANDATORY fields ──────────────────────────────────────────────
        self.chamber_id = chamber_id.upper()
        self.setup_id   = session_guide.setup_id
        self.dig_input  = "0" if chamber_id in "ac" else "1"
        self.mouse_id   = session_guide.mouse_id

        # ─── 1) parse all drug_and_dose* columns ───────────────────────────
        drug_cols = [c for c in session_guide.index if c.startswith("drug_and_dose")]
        self.drug_infos = []
        for dc in drug_cols:
            raw = session_guide[dc]
            if isinstance(raw, str) and raw.strip():
                self.drug_infos.append(self._parse_drug_info(raw))
        # if you really only want one dict, you can do:
        # self.drug_info = self.drug_infos[0] if self.drug_infos else {}

        # ─── 2) discover fiber/exclude columns ─────────────────────────────
        fiber_cols   = [c for c in session_guide.index if re.match(r"^fiber\d+$", c)]
        exclude_cols = [c for c in session_guide.index if c.lower().startswith("exclude")]

        # ─── 3) everything else is “optional” metadata ─────────────────────
        reserved = {"setup_id", "mouse_id"} | set(drug_cols) | set(fiber_cols) | set(exclude_cols)
        existing = set(dir(self))
        all_meta  = set(session_guide.index) - reserved

        # Warn about conflicts
        ignored = all_meta & existing
        if ignored:
            warnings.warn(
                f"Ignoring session-guide columns {sorted(ignored)!r} "
                "because they conflict with Session attributes/methods."
            )

        # Preserve original index order when setting
        for col in session_guide.index:
            if col in all_meta and col not in existing:
                setattr(self, col, session_guide[col])

        # ─── 4) build fiber→region & downstream as before ─────────────────
        self.fiber_to_region = self.create_fiber_to_region_dict()
        self.brain_regions   = sorted(self.fiber_to_region.values())

        self.dfs     = DataContainer()

    def _parse_drug_info(self, raw: Any) -> Dict[str, Optional[str]]:
        """
        Take a raw cell value (e.g. "CPT 3.0 mg") and return
        {'name': str, 'dose': str|None, 'metric': str|None}.
        """
        if not isinstance(raw, str) or not raw.strip():
            return {'name': None, 'dose': None, 'metric': None}

        parts = raw.strip().split()
        name = parts[0]
        dose = None
        metric = None

        # if second token is numeric, it's the dose
        if len(parts) >= 2 and re.match(r'^\d+(\.\d+)?$', parts[1]):
            dose = parts[1]
            # optional third token is the metric
            if len(parts) >= 3:
                metric = parts[2]
        # otherwise we just leave dose/metric as None

        return {'name': name, 'dose': dose, 'metric': metric}

    # create_fiber_dict creates dictionary of all fibers used and their corresponding brainregion
    def create_fiber_to_region_dict(self, fiber_pattern: Optional[Pattern[str]] = None) -> Dict[str, Tuple[str, str, str]]:
        """Build a mapping from fiber number to (region, side, fiber_color)."""
        pattern    = fiber_pattern or re.compile(config.FIBER_PATTERN)
        exclude_re = re.compile(r"(?i)^exclude\??$")
        fiber_to_region = {}
        cols = list(self.session_guide.index)

        for idx, col in enumerate(cols):
            m = pattern.match(col)
            if not m:
                continue

            # ensure an 'exclude?' column follows
            if idx + 1 >= len(cols) or not exclude_re.match(cols[idx+1]):
                raise KeyError(
                    f"{self.trial_id}: fiber column '{col}' not followed by an 'exclude?' column."
                )

            val = self.session_guide[col]
            is_excluded = pd.notna(self.session_guide[cols[idx+1]]) and self.remove_bad_signal_sessions
            if pd.isna(val) or is_excluded:
                continue

            # parse region/side
            try:
                region, side = val.split("_", 1)
            except Exception:
                raise ValueError(
                    f"{self.trial_id}: brain-region value {val!r} in column '{col}' "
                    "is not in 'region_side' format (e.g. 'DMS_left', 'mPFC_right')."
                )

            # extract fiber number
            fiber_number = m.group(2) or m.group(3)
            if not fiber_number:
                raise ValueError(
                    f"{self.trial_id}: could not extract fiber number from regex match on column '{col}'."
                )

            # pick color if supplied, else find a fallback _only now_
            if m.group(1):
                fiber_color = m.group(1)
            else:
                try:
                    fiber_color = next(ch for ch in config.LETTER_TO_FREQS if ch != 'iso')
                except StopIteration:
                    raise RuntimeError(
                        f"{self.trial_id}: no non-'iso' entries in config.LETTER_TO_FREQS; "
                        "cannot determine a fallback fiber color for column '{col}'."
                    )

            # guard duplicates
            if fiber_number in fiber_to_region:
                prev = [c for c, v in fiber_to_region.items() if c == fiber_number][0]
                raise ValueError(
                    f"{self.trial_id}: duplicate fiber number '{fiber_number}' in column '{col}'; "
                    f"already defined by column '{prev}'."
                )

            fiber_to_region[fiber_number] = (region, side, fiber_color)

        return fiber_to_region
        
# data_loading.py
def load_all_sessions(
    baseline_dir: str,
    session_type: str,
    first_n_dirs: Optional[int] = None,
    remove_bad_signal_sessions: bool = False,
) -> List[Session]:
    """Load session objects from each trial directory in a baseline directory."""
    RE_TRIAL_DIR = re.compile(r"^(T\d+)_((?:\d+|e)(?:\.(?:\d+|e))*)(_.*)?$")

    # 1. Validate baseline_dir
    if not os.path.exists(baseline_dir):
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")
    if not os.path.isdir(baseline_dir):
        raise NotADirectoryError(f"Baseline path is not a directory: {baseline_dir}")
    if not os.access(baseline_dir, os.R_OK): # Check for read access
        raise PermissionError(f"No read access to baseline directory: {baseline_dir}")

    # 2. List all items and filter for actual directories
    try:
        all_items_in_baseline = os.listdir(baseline_dir)
    except OSError as e:
        raise OSError(f"Could not list contents of baseline directory '{baseline_dir}': {e}")


    subdir_names = [
        item_name for item_name in all_items_in_baseline
        if os.path.isdir(os.path.join(baseline_dir, item_name))
    ]
    # Files or other non-directory items in baseline_dir are silently ignored here.

    # 3. Filter these directory names by the regex
    valid_trial_dirs_info = [] # Will store tuples of (name, match_object)
    dirs_not_matching_regex = []

    for name in subdir_names:
        match = RE_TRIAL_DIR.fullmatch(name)
        if match:
            valid_trial_dirs_info.append({'name': name, 'match': match})
        else:
            dirs_not_matching_regex.append(name)

    # 4. Handle directories that did not match the regex pattern
    #    This is where your choice for B' comes in. For example, for Option B.C (Summary Warn):
    if dirs_not_matching_regex:
        warnings.warn(
            f"The following directories in '{baseline_dir}' did not match the expected "
            f"trial directory format (e.g., T<num>_<segments>[_suffix]) and were skipped:\n"
            f"{', '.join(dirs_not_matching_regex)}",
            UserWarning # Or a custom warning type
        )
        # Depending on verbosity, you might choose to print this or log it instead/additionally.

    # 5. If no valid trial directories were found, raise a pedagogical error
    if not valid_trial_dirs_info:
        error_message = (
            f"No valid trial directories found in '{baseline_dir}'.\n"
            "Trial directory names are expected to follow a pattern like:\n"
            "  T<number>_<segments>[_optionalSuffix]\n"
            "Where <segments> are numbers or the letter 'e', separated by dots (e.g., '123.45.e' or '67' or 'e').\n\n"
            "Examples of valid names:\n"
            "  T1_23.45.e\n"
            "  T102_67.89\n"
            "  T4_e_analysisSet1\n"
            "  T7_123\n\n"
            "Please ensure your directories are named correctly."
        )
        raise ValueError(error_message)

    # 6. Sort the valid trial directories using sort_key_func on the name
    valid_trial_dirs_info.sort(key=lambda x: (int(x['match'].group(1)[1:]),))

    # sorted_subdirs = sorted(subdirs, key=sort_key_func)
    # trial_dirs = [os.path.join(baseline_dir, sd) for sd in sorted_subdirs]

    # 7. Validate and apply first_n_dirs
    if first_n_dirs is not None:
        if not isinstance(first_n_dirs, int) or first_n_dirs <= 0:
            raise ValueError(
                f"'first_n_dirs' must be a positive integer, but got {first_n_dirs}"
            )
        # Slicing handles cases where first_n_dirs > len(valid_trial_dirs_info)
        processed_trial_dirs_info = valid_trial_dirs_info[:first_n_dirs]
    else:
        processed_trial_dirs_info = valid_trial_dirs_info


    all_sessions = []
    for dir_info in tqdm(processed_trial_dirs_info, desc="Processing trial directories"):
        trial_dir_name = dir_info['name']
        match_obj = dir_info['match'] # The regex match object for this directory name
        
        trial_dir_path = os.path.join(baseline_dir, trial_dir_name)

        trial_id        = match_obj.group(1)                # e.g. "T12"
        guide_prefix = f"{trial_id}_trial_guide"         # e.g. "T12_trial_guide"

        # 1. glob for xlsx first, then csv
        xlsx_full_path = os.path.join(trial_dir_path, guide_prefix + ".xlsx")
        csv_full_path  = os.path.join(trial_dir_path, guide_prefix + ".csv")

        has_xlsx = os.path.isfile(xlsx_full_path)
        has_csv = os.path.isfile(csv_full_path)

        # 2. error if none
        if not has_xlsx and not has_csv:
            raise FileNotFoundError(
                f"No trial‐guide found in {trial_dir_path}! "
                f"Expected '{trial_id}_trial_guide.xlsx' or '{trial_id}_trial_guide.csv'."
            )

        # 3. warn if multiple, pick first
        if has_xlsx and has_csv:
            warnings.warn(
                f"Both XLSX and CSV trial‐guide files found for {trial_id} in {trial_dir_path}. "
                f"Using '{os.path.basename(xlsx_full_path)}'."
            )

        guide_path = xlsx_full_path if os.path.isfile(xlsx_full_path) else csv_full_path

        # 4. read with proper engine
        try:
            if guide_path.endswith(".xlsx"):
                current_trial_guide_df = pd.read_excel(
                    guide_path,
                    nrows=4,
                    dtype={"mouse_id": str},
                    index_col=0,
                    engine="openpyxl",
                )
            else:
                current_trial_guide_df = pd.read_csv(
                    guide_path,
                    nrows=4,
                    dtype={"mouse_id": str},
                    index_col=0,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to read trial guide '{guide_path}': {e}")

        # check for mandatory columns
        required_cols = {"setup_id", "mouse_id"}
        missing_cols  = required_cols - set(current_trial_guide_df.columns)
        if missing_cols:
            raise KeyError(
                f"{trial_id}: trial guide at '{guide_path}' is missing required column(s): {missing_cols}"
            )
        
        # Get segments directly from the stored match object's group
        # Group 1 is T<num>, Group 2 is the segments string, Group 3 is optional suffix
        segments = match_obj.group(2).split('.')
        chambers = list("abcd")

        if len(segments) > len(chambers):
            warnings.warn(
                f"{trial_id}: folder name has {len(segments)} segments but only {len(chambers)} chambers; extra segments will be ignored."
                )

        for segment, chamber_id in zip(segments, chambers):
            if segment == 'e':
                continue  # empty session marker
            
            # 1) make sure that chamber row actually exists
            try:
                session_guide = current_trial_guide_df.loc[chamber_id]
            except KeyError:
                raise KeyError(
                    f"{trial_id}: expected chamber '{chamber_id}' not found in trial guide at '{guide_path}'."
                )
            
            # 2) compare IDs
            if session_guide['mouse_id'] != segment:
                raise ValueError(
                    f"{trial_id}/{chamber_id}: mouse_id mismatch—"
                    f"folder says '{segment}', guide says '{session_guide['mouse_id']}'."
                )
            
            new_session = Session(chamber_id, trial_dir_path, session_guide, session_type)
            if len(new_session.brain_regions) > 0 or not remove_bad_signal_sessions:
                all_sessions.append(new_session)

    return all_sessions