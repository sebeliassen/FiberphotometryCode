
import pandas as pd
import pickle
import os
import sys

# Add the parent directory to sys.path
sys.path.append('/home/fsp585/Desktop/FiberphotometryCode')

def inspect_pkl(path):
    print(f"--- Inspecting {path} ---")
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if it's a list of sessions or a dict
        if isinstance(data, list):
            print(f"Loaded {len(data)} sessions.")
            if len(data) > 0:
                first_session = data[0]
                # print(f"Type of first session: {type(first_session)}")
                # print(f"Dir of first session: {dir(first_session)}")
                
                if hasattr(first_session, 'all_data_df'):
                    df = first_session.all_data_df
                    print("\nall_data_df shape:", df.shape)
                    print("all_data_df columns:", df.columns.tolist())
                    print("\nFirst 3 rows:")
                    print(df.head(3))
                    
                    # Inspect specific columns if they exist
                    cols_to_check = ['drug', 'genotype', 'animal', 'Date']
                    for col in cols_to_check:
                        if col in df.columns:
                            print(f"Unique {col}: {df[col].unique()}")
                elif hasattr(first_session, '__dict__') and 'all_data_df' in first_session.__dict__:
                     df = first_session.all_data_df
                     print("\nall_data_df shape:", df.shape)
                     print("all_data_df columns:", df.columns.tolist())
                else:
                    print("Could not find all_data_df in session object")
        
        elif isinstance(data, dict):
             print(f"Loaded dict with keys: {data.keys()}")
    except Exception as e:
        print(f"Error loading {path}: {e}")
    print("\n")

base_path = '/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data'
files = [
    'jonathan_ifdn_sessions_w_sylls.pkl',
    'jonathan_amph_sessions_w_sylls.pkl'
]

for f in files:
    inspect_pkl(os.path.join(base_path, f))
