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
            print(f"Extras keys: {s.extras.keys()}")
            print(f"Photometry keys: {s.photometry.keys()}")
            
            # Check for all_data_df in extras
            if 'all_data_df' in s.extras:
                print(f"all_data_df columns: {list(s.extras['all_data_df'].columns)[:10]}...")
            
            if 'aligned' in s.photometry:
                 print(f"photometry['aligned'] columns: {list(s.photometry['aligned'].columns)[:10]}...")
                 
    except Exception as e:
        print(f"Error: {e}")
