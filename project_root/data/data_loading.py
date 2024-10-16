import os
import fnmatch
import re
import pandas as pd
from tqdm import tqdm
import config

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
    def __init__(self, chamber_id, trial_dir, session_guide):
        self.trial_dir = trial_dir
        self.trial_id = os.path.basename(trial_dir)
        self.session_guide = session_guide
        
        self.chamber_id = chamber_id.upper()  # Ensure chamber_id is uppercase
        self.setup_id = session_guide.setup_id
        self.dig_input = "0" if chamber_id in "ac" else "1"

        self.task = session_guide.task
        
        #TODO: In the future this should be possible without a try/except clause
        # try:
        #     self.genotype = session_guide.Genotype
        # except AttributeError:
        #     self.genotype = session_guide.notes
        try:
            self.genotype = session_guide.genotype
        except AttributeError:
            self.genotype = None

        #TODO: find a more elegant way of writing/handling session_guide.drug_and_dose_1.endswith('mg/kg')
        #drug_info_list = session_guide.drug_and_dose_1.split()
        #if len(drug_info_list) == 3 and session_guide.drug_and_dose_1.endswith('mg/kg'):
        #    self.drug_info = dict(zip(['name', 'dose', 'metric'], drug_info_list))
        #else:
        #    self.drug_info = {'name': session_guide.drug_and_dose_1, 'dose': None, 'metric': None}
        drug_info_list = session_guide.drug_and_dose_1.split()
        if len(drug_info_list) == 2 and re.match(r'^\d+(\.\d+)?$', drug_info_list[1]):

            self.drug_info = dict(zip(['name', 'dose'], drug_info_list))
        else:
            self.drug_info = {'name': session_guide.drug_and_dose_1, 'dose': None}
        
        self.mouse_id = session_guide.mouse_id
        self.fiber_to_region = self.create_fiber_to_region_dict()
        self.brain_regions = sorted(list(self.fiber_to_region.values()))
        
        # Initialize DataContainer for DataFrame storage
        self.df_container = DataContainer()
        
        self.load_all_data()
        
    def load_data(self, file_pattern, skip_rows=None, use_cols=None, only_header=False):
        file_name = next((f for f in os.listdir(self.trial_dir) if fnmatch.fnmatch(f, file_pattern)), None)
        if file_name:
            file_path = os.path.join(self.trial_dir, file_name)
            if only_header:
                df = pd.read_csv(file_path, nrows=0)
                return df.columns.tolist()
            else:
                return pd.read_csv(file_path, skiprows=skip_rows, usecols=use_cols)
        else:
            return None

    def load_all_data(self):
        # Load and store DataFrames in the df_container
        self.df_container.add_data('raw', self.load_data(f'RAW_{self.chamber_id}*.csv', skip_rows=18))
        self.df_container.add_data('ttl', self.load_data(f'DigInput_{self.chamber_id}*.csv'))
        
        # Load bonsai and photwrit data
        for freq in config.RENAME_FREQS:
            bonsai_df = self.load_data(f'c{freq}_bonsai*Setup{self.setup_id}*.csv', use_cols=self.filter_columns)
            photwrit_df = self.load_data(f'channel{freq}photwrit_Setup{self.setup_id}*.csv', use_cols=self.filter_columns)
            self.df_container.add_data(f'bonsai_{freq}', bonsai_df)
            self.df_container.add_data(f'photwrit_{freq}', photwrit_df)

    # create_fiber_dict creates dictionary of all fibers used and their corresponding brainregion
    def create_fiber_to_region_dict(self, fiber_pattern=re.compile(r'fiber(\d+)')):
        # Initialize an empty dictionary
        fiber_to_region_dict = {}

        for idx, col in enumerate(self.session_guide.index):
            if (fiber_pattern.match(col) 
                and pd.notna(self.session_guide[col])
                #and idx + 1 < len(self.session_guide.index) this should happen, so commented out
                and pd.isna(self.session_guide.iloc[idx + 1])):
                
                fiber_number = fiber_pattern.match(col).group(1)
                fiber_to_region_dict[fiber_number] = self.session_guide[col]

        return fiber_to_region_dict


    # to save memory, all of the columns that contain data from unused fibers, are filtered out pre-loading
    def filter_columns(self, col):
        pattern = re.compile(r"Region(\d+)[\w]*")
        match = pattern.match(col)
        if match:
            # Check if the captured number is in the keys of fiber_dict
            return match.group(1) in self.fiber_to_region
        else:
            # If the column doesn't match the pattern, include it
            return True
        

# Custom sort key function that extracts the trial number from a directory name
def sort_key_func(dir_name):
    # Find all numbers following 'T' or before a period '.' and return them as a tuple of integers
    numbers = tuple(map(int, re.findall(r'T(\d+)', dir_name)))
    return numbers


# TODO: a bit limited in parameters, recommend additional ones if need be
def load_all_sessions(baseline_dir, first_n_dirs=None, remove_bad_signal_sessions=False):
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
                                                    dtype={"mouse_id": str}, index_col=0)
        for segment, chamber_id in zip(segments, "abcd"):
            if segment == 'e':
                continue  # Skip this session as it's marked as empty
            session_guide = current_trial_guide_df.loc[chamber_id]

            if session_guide.mouse_id != segment:
                raise Exception(f"The mouse id '{segment}' from the folder names and trial guide '{session_guide.mouse_id}' do not match")
            
            new_session = Session(chamber_id, trial_dir, session_guide)
            if len(new_session.brain_regions) > 0 or (remove_bad_signal_sessions == False):
                all_sessions.append(new_session)
    return all_sessions