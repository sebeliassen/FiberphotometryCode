import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Paths
new_pkl_path = "/home/fsp585/Desktop/FiberphotometryCode/monoamine_interactions/data/sessions_dirty.pkl"
ref_pkl_path = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_ifdn_sessions_dirty.pkl"

sys.path.append("/home/fsp585/Desktop/FiberphotometryCode")

def load_pkl(p):
    print(f"Loading {p}...")
    with open(p, "rb") as f:
        return pickle.load(f)

def inspect_session(s, label):
    print(f"\n--- {label} Session Inspection ({s.session_id}) ---")
    print(f"Type: {type(s)}")
    
    # Metadata
    print(f"Metadata Keys: {list(s.metadata.keys())}")
    print(f"Metadata Sample: { {k: s.metadata[k] for k in list(s.metadata.keys())[:5]} }")
    
    # Photometry
    if hasattr(s, "photometry"):
        print(f"Photometry keys: {list(s.photometry.keys())}")
        if "combined_filtered" in s.photometry:
            df = s.photometry["combined_filtered"]
            print(f"Photometry Combined Shape: {df.shape}")
            print(f"Photometry Columns: {list(df.columns)}")
    else:
        print("No 'photometry' attribute")

    # Tracking / Video
    print(f"Tracking Refs DF: {getattr(s, 'tracking_refs_df', 'MISSING')}")
    if hasattr(s, 'tracking_refs_df') and hasattr(s.tracking_refs_df, 'shape'):
         print(f"Tracking Shape: {s.tracking_refs_df.shape}")
         
    # Syllables? (Likely missing in new one)
    if hasattr(s, "syllables"):
        print(f"Syllables: Present")
    else:
        print(f"Syllables: Not present (Expected)")

try:
    new_sessions = load_pkl(new_pkl_path)
    ref_sessions = load_pkl(ref_pkl_path)
    
    print(f"\nTotal New Sessions: {len(new_sessions)}")
    print(f"Total Ref Sessions: {len(ref_sessions)}")
    
    if len(new_sessions) > 1:
        inspect_session(new_sessions[1], "NEW (Session 1)")
        s_new = new_sessions[1]
    elif new_sessions:
        inspect_session(new_sessions[0], "NEW (Session 0)")
        s_new = new_sessions[0]
    
    if ref_sessions:
        # Try to find a roughly similar session if possible, else just the first
        inspect_session(ref_sessions[0], "REF (IFDN)")

    print("\n--- Detailed Comparison ---")
    if new_sessions and ref_sessions:
        s_new = new_sessions[0]
        s_ref = ref_sessions[0]
        
        # Check specific metadata asked by user
        print(f"New Geno: {s_new.metadata.get('genotype')}")
        print(f"Ref Geno: {s_ref.metadata.get('genotype')}")
        
        # Check signals
        print(f"New Signal 0: {s_new.metadata.get('signal_0')}")
        print(f"Ref Signal 0: {s_ref.metadata.get('signal_0')}")

except Exception as e:
    print(f"Comparison failed: {e}")
