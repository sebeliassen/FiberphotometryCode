import os
import fnmatch
import re
import pandas as pd
from tqdm import tqdm
import config
import oft_config
import numpy as np

class DataContainer:
    def __init__(self, data_type=None):
        self.data = {}
        self.data_type = data_type

    def add_data(self, name, data):
        # Set the data type if not already set
        if self.data_type is None:
            self.data_type = type(data)
        elif not isinstance(data, self.data_type):
            raise TypeError(f"Data must be of type {self.data_type.__name__}, but is of type {type(data)}")

        self.data[name] = data

    def get_data(self, name):
        return self.data.get(name)

    def remove_data(self, name):
        if name in self.data:
            del self.data[name]

    def fetch_all_data_names(self):
        return list(self.data.keys())

    def clear_data(self):
        self.data.clear()


class Session:
    def __init__(self, chamber_id, trial_dir, session_guide, session_type):
        self.trial_dir = trial_dir
        self.trial_id = os.path.basename(trial_dir)
        self.session_guide = session_guide
        self.session_type = session_type
        
        self.chamber_id = chamber_id.upper()  # Ensure chamber_id is uppercase
        self.setup_id = session_guide.setup_id
        self.dig_input = "0" if chamber_id in "ac" else "1"

        self.task = session_guide.task
        self.genotype = getattr(session_guide, 'genotype', None)

        # Set up drug information if it exists
        self.drug_info = self._parse_drug_info(session_guide)
        
        self.mouse_id = session_guide.mouse_id
        self.fiber_to_region = self.create_fiber_to_region_dict()
        self.brain_regions = sorted(list(self.fiber_to_region.values()))

        # Initialize DataContainer for DataFrame storage
        self.dfs = DataContainer()
        self.load_all_data(self.session_type)

    def _parse_drug_info(self, session_guide):
        # Retrieve drug and dose information
        drug_and_dose = getattr(session_guide, 'drug_and_dose_1', None)
        
        # Check if drug_and_dose is a non-empty string
        if isinstance(drug_and_dose, str) and drug_and_dose.strip():
            drug_info_list = drug_and_dose.split()
            
            # Verify we have at least name and dose, and optionally a metric
            if len(drug_info_list) >= 2 and re.match(r'^\d+(\.\d+)?$', drug_info_list[1]):
                # If there’s a third element, assume it’s the metric
                drug_info = {'name': drug_info_list[0], 'dose': drug_info_list[1]}
                if len(drug_info_list) == 3:
                    drug_info['metric'] = drug_info_list[2]  # Add metric if provided
                else:
                    drug_info['metric'] = None  # No metric provided
                return drug_info

        # Default if drug information is missing or invalid
        return {'name': drug_and_dose, 'dose': None, 'metric': None}


    def load_data(self, file_pattern, skip_rows=None, use_cols=None, only_header=False, sep=','):
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

    def load_all_data(self, session_type):
        # TODO: merge branches further
        if session_type == 'oft':
            # Load photometry data
            for df_name, df_file_pattern in config.PHOT_DF_PATTERNS.items():
                phot_df = self.load_data(df_file_pattern, use_cols=self.filter_columns)
                self.dfs.add_data(df_name, phot_df)

            for df_name, df_file_pattern in config.TIMESTAMP_DF_PATTERNS.items():
                df = self.load_data(df_file_pattern, sep='\s+')
                self.dfs.add_data(df_name, df)
        elif session_type == 'cpt':
            # Load and store DataFrames in the df_container
            self.dfs.add_data('raw', self.load_data(f'RAW_{self.chamber_id}*.csv', skip_rows=18))
            self.dfs.add_data('ttl', self.load_data(f'DigInput_{self.chamber_id}*.csv'))
            
            # Load bonsai and photwrit data
            for freq in config.RENAME_FREQS:
                bonsai_df = self.load_data(f'c{freq}_bonsai*Setup{self.setup_id}*.csv', use_cols=self.filter_columns)
                phot_df = self.load_data(f'channel{freq}photwrit_Setup{self.setup_id}*.csv', use_cols=self.filter_columns)
                self.dfs.add_data(f'bonsai_{freq}', bonsai_df)
                self.dfs.add_data(f'phot_{freq}', phot_df)
        else:
            raise f'session type need to be either oft or cpt but is {session_type}'


    # create_fiber_dict creates dictionary of all fibers used and their corresponding brainregion
    def create_fiber_to_region_dict(self, fiber_pattern=re.compile(r'([GR])(\d+)|fiber(\d+)')):
        # Initialize an empty dictionary
        fiber_to_region_dict = {}

        # Iterate over the DataFrame index with enumeration for index and column label
        for idx, col in enumerate(self.session_guide.index):
            match = fiber_pattern.match(col)
            # Check if the column matches the fiber pattern and the value is not NaN
            if (match 
                and pd.notna(self.session_guide[col])
                #and idx + 1 < len(self.session_guide.index) this should happen, so commented out
                and pd.isna(self.session_guide.iloc[idx + 1])):
                    
                # Extract region and side information
                region, side = self.session_guide[col].split("_")
                
                if match.group(1) and match.group(2):  # Color and number match
                    fiber_color = match.group(1)
                    fiber_number = match.group(2)
                    fiber_to_region_dict[fiber_number] = (region, side, fiber_color)
                elif match.group(3):  # Only number match
                    fiber_number = match.group(3)
                    fiber_to_region_dict[fiber_number] = (region, side, None)
                    
        return fiber_to_region_dict


    def filter_columns(self, col):
        # Combined pattern to match both "Region" columns and [GR] columns with numbers
        pattern = re.compile(r"[GR](\d+)|Region(\d+)[\w]*")
        match = pattern.match(col)
        
        if match:
            # Extract the captured number from whichever group matched
            captured_number = match.group(1) or match.group(2)
            # Check if this number is in fiber_to_region
            return captured_number in self.fiber_to_region
        else:
            # If the column doesn't match the pattern, include it
            return True
        

# Custom sort key function that extracts the trial number from a directory name
def sort_key_func(dir_name):
    # Find all numbers following 'T' or before a period '.' and return them as a tuple of integers
    numbers = tuple(map(int, re.findall(r'T(\d+)', dir_name)))
    return numbers


# TODO: a bit limited in parameters, recommend additional ones if need be
def load_all_sessions(baseline_dir, session_type, first_n_dirs=None, remove_bad_signal_sessions=False):
    # Get a list of all subdirectories within the baseline directory
    subdirs = [d for d in os.listdir(baseline_dir) if os.path.isdir(os.path.join(baseline_dir, d))]

    # Sort the subdirectories based on the trial number
    sorted_subdirs = sorted(subdirs, key=sort_key_func)

    # Join the sorted subdirectories with the baseline path
    trial_dirs = [os.path.join(baseline_dir, sd) for sd in sorted_subdirs]
    if first_n_dirs is None:
        first_n_dirs = len(trial_dirs)

    all_sessions = []

    for trial_dir in tqdm(trial_dirs[:first_n_dirs]):
        trial_id = os.path.basename(trial_dir)
        segments = trial_id.split('_')[1].split('.')  # Assuming the format is like 'T1_23.25.29.e'
        for file in os.listdir(trial_dir):
            # Check if file ends with 'trial_guide.xlsx'
            if fnmatch.fnmatch(file, '*trial_guide.xlsx'):
                # Load into DataFrame
                current_trial_guide_df = pd.read_excel(os.path.join(trial_dir, file), nrows=4,
                                                    dtype={"mouse_id": str}, index_col=0, engine='openpyxl')
        for segment, chamber_id in zip(segments, "abcd"):
            if segment == 'e':
                continue  # Skip this session as it's marked as empty
            session_guide = current_trial_guide_df.loc[chamber_id]

            if session_guide.mouse_id != segment:
                raise Exception(f"The mouse id '{segment}' from the folder names and trial guide '{session_guide.mouse_id}' do not match")
            
            new_session = Session(chamber_id, trial_dir, session_guide, session_type)
            if len(new_session.brain_regions) > 0 or (remove_bad_signal_sessions == False):
                all_sessions.append(new_session)
    return all_sessions