import sys
sys.path.append("/home/fsp585/Desktop/FiberphotometryCode")
import pandas as pd

pkl_path = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_v2_sanitized.pkl"

print(f"Loading {pkl_path}...")
sessions = pd.read_pickle(pkl_path)
print(f"Loaded {len(sessions)} sessions.")

if len(sessions) > 0:
    s = sessions[0]
    print(f"Session 0 Type: {type(s)}")
    print(f"Dir of Session 0: {dir(s)}")
    print(f"Metadata Dict: {s.metadata}")
    
    sylls = s.syllables
    if sylls is None:
        print("s.syllables is None")
    else:
        print(f"s.syllables type: {type(sylls)}")
        print(f"s.syllables len: {len(sylls)}")
        print(f"Sample: {sylls[:10]}")
        
    print(f"Injection Frame: {s.metadata.get('injection_frame')}")
    
    # Check if there is a 'name' that matches moseq_df style
