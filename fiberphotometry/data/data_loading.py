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
        optional_cols = [c for c in session_guide.index if c not in reserved]
        for col in optional_cols:
            setattr(self, col, session_guide[col])
        # now you’ve got self.task, self.genotype, self.notes, etc., automatically

        # ─── 4) build fiber→region & downstream as before ─────────────────
        self.fiber_to_region = self.create_fiber_to_region_dict()
        self.brain_regions   = sorted(self.fiber_to_region.values())

        self.dfs     = DataContainer()
        self.parsers = []
        #self._build_parsers()

        self.load_all_data()

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

    def load_data(
        self,
        file_pattern: str,
        skip_rows: Optional[int] = None,
        use_cols: Optional[List[str]] = None,
        only_header: bool = False,
        sep: str = ',',
    ) -> Optional[Union[pd.DataFrame, List[str]]]:
        """Load a file matching pattern into a DataFrame or return its header."""
        file_name = next((f for f in os.listdir(self.trial_dir) if fnmatch.fnmatch(f, file_pattern)), None)
        if file_name:
            file_path = os.path.join(self.trial_dir, file_name)
            if only_header:
                df = pd.read_csv(file_path, nrows=0)
                return df.columns.tolist()
            else:
                return pd.read_csv(file_path, skiprows=skip_rows, usecols=use_cols, sep=sep)
        else:
            return None
    
    def load_all_data(self) -> None:
        """Loop through each parser and stash its outputs in self.dfs."""
        p = Path(self.trial_dir)
        for parser in self.parsers:
            for key, df in parser.find_and_parse(p, self).items():
                self.dfs.add_data(key, df)

    # create_fiber_dict creates dictionary of all fibers used and their corresponding brainregion
    def create_fiber_to_region_dict(self, fiber_pattern: Optional[Pattern[str]] = None) -> Dict[str, Tuple[str, str, str]]:
        """Build a mapping from fiber number to (region, side, fiber_color) based on the fiber pattern."""
        # Compile fiber_pattern if not provided
        pattern = fiber_pattern if fiber_pattern is not None else re.compile(config.FIBER_PATTERN)
        fiber_to_region_dict = {}

        # Iterate over the DataFrame index with enumeration for index and column label
        for idx, col in enumerate(self.session_guide.index):
            match = pattern.match(col)
            # Check if the column matches the fiber pattern and the value is not NaN
            if (match 
                and pd.notna(self.session_guide[col])
                #and idx + 1 < len(self.session_guide.index) this should happen, so commented out
                and (pd.isna(self.session_guide.iloc[idx + 1])
                     or not self.remove_bad_signal_sessions)
                ):
                    
                # Extract region and side information
                region, side = self.session_guide[col].split("_")
                
                if match.group(1) and match.group(2):  # Color and number match
                    fiber_color = match.group(1)
                    fiber_number = match.group(2)
                    fiber_to_region_dict[fiber_number] = (region, side, fiber_color)
                # We assume only one non-iso channel if nothing else is mentioned
                elif match.group(3):  # Only number match
                    LETTER_TO_FREQS = config.LETTER_TO_FREQS
                    probable_letter = [channel for channel in LETTER_TO_FREQS.keys() if channel != 'iso'][0]

                    fiber_number = match.group(3)
                    fiber_to_region_dict[fiber_number] = (region, side, probable_letter)
                    
        return fiber_to_region_dict


    def filter_columns(self, col: str) -> bool:
        """Filter columns based on the configured pattern and fiber mapping."""
        pattern = re.compile(config.FILTER_COLUMNS_PATTERN)
        match = pattern.match(col)
        if match:
            captured_number = match.group(1) or match.group(2)
            return captured_number in self.fiber_to_region
        return True
        

# Custom sort key function that extracts the trial number from a directory name
def sort_key_func(dir_name: str) -> Tuple[int, ...]:
    """Extract trial numbers from a directory name for sorting."""
    # Find all numbers following 'T' or before a period '.' and return them as a tuple of integers
    numbers = tuple(map(int, re.findall(r'T(\d+)', dir_name)))
    return numbers

# data_loading.py
def load_all_sessions(
    baseline_dir: str,
    session_type: str,
    first_n_dirs: Optional[int] = None,
    remove_bad_signal_sessions: bool = False,
) -> List[Session]:
    """Load session objects from each trial directory in a baseline directory."""
    subdirs = [
        d for d in os.listdir(baseline_dir) 
        if os.path.isdir(os.path.join(baseline_dir, d))
    ]
    sorted_subdirs = sorted(subdirs, key=sort_key_func)
    trial_dirs = [os.path.join(baseline_dir, sd) for sd in sorted_subdirs]

    if first_n_dirs is None:
        first_n_dirs = len(trial_dirs)

    all_sessions = []
    for trial_dir in tqdm(trial_dirs[:first_n_dirs]):
        trial_id = os.path.basename(trial_dir)
        segments = trial_id.split('_')[1].split('.')  # e.g. "T1_23.25.29.e"
        
        # Find and load a 'trial_guide.xlsx'
        for file in os.listdir(trial_dir):
            if fnmatch.fnmatch(file, 'T*trial_guide.xlsx'):
                current_trial_guide_df = pd.read_excel(
                    os.path.join(trial_dir, file),
                    nrows=4,
                    dtype={"mouse_id": str},
                    index_col=0,
                    engine='openpyxl'
                )
        
        for segment, chamber_id in zip(segments, "abcd"):
            if segment == 'e':
                # Skip sessions marked empty
                continue
            
            session_guide = current_trial_guide_df.loc[chamber_id]
            if session_guide.mouse_id != segment:
                raise Exception(
                    f"The mouse id '{segment}' from the folder name and "
                    f"'{session_guide.mouse_id}' in trial guide do not match."
                )
            
            new_session = Session(chamber_id, trial_dir, session_guide, session_type)
            if len(new_session.brain_regions) > 0 or not remove_bad_signal_sessions:
                all_sessions.append(new_session)

    return all_sessions