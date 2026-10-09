import pandas as pd
import sys

paths = [
    "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_ifdn_sessions_w_sylls.pkl",
    "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl"
]

for p in paths:
    print(f"--- Loading {p} ---")
    try:
        data = pd.read_pickle(p)
        if isinstance(data, list) and len(data) > 0:
            s = data[0]
            print(f"Session: {s.session_id}")
            
            if hasattr(s, 'all_data_df'):
                df = s.all_data_df
                print(f"all_data_df Shape: {df.shape}")
                print(f"all_data_df Columns: {list(df.columns)[:20]}...") 
                
                # Check for specific columns
                has_syll = 'syllable' in df.columns
                has_onset = 'onset' in df.columns
                print(f"Has 'syllable': {has_syll}")
                print(f"Has 'onset': {has_onset}")
                
            else:
                print("No all_data_df found in session.")

    except Exception as e:
        print(f"Error: {e}")
