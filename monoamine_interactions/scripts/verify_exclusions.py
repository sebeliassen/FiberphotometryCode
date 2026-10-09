
import pickle
import sys
from pathlib import Path

# Add project root to sys.path
project_root = "/home/fsp585/Desktop/FiberphotometryCode"
if project_root not in sys.path:
    sys.path.append(project_root)

# Need to ensure classes are importable for unpickling
try:
    from syllable_generation.loaders.jonathan.loader import SessionRecord
except ImportError:
    pass

# Paths
pkl_path = "/home/fsp585/Desktop/FiberphotometryCode/monoamine_interactions/data/sessions_dirty.pkl"

def load_pkl(p):
    print(f"Loading {p}...")
    with open(p, "rb") as f:
        return pickle.load(f)

def verify_metadata():
    sessions = load_pkl(pkl_path)
    print(f"Loaded {len(sessions)} sessions.")
    
    for s in sessions:
        md = s.metadata
        sid = s.session_id
        aid = md.get("animal_id")
        
        print(f"\nSession: {sid} | Mouse: {aid}")
        
        # Check Fiber Indices
        eff_indices = md.get("fiber_indices_effective")
        print(f"  Effective Fiber Indices: {eff_indices}")
        
        # Check Signals
        for k in ["signal_0", "signal_1", "signal_2", "signal_3"]:
            if k in md:
                print(f"  {k}: {md[k]}")
                if "signal_meta" in md and k in md["signal_meta"]:
                    sm = md["signal_meta"][k]
                    print(f"    - Color: {sm.get('color')}, Hemi: {sm.get('hemisphere')}, Orig: {sm.get('orig_id')}")
                
        # Check Exclusions
        excl = md.get("exclusions")
        if excl:
            print(f"  Exclusions: {excl}")
        else:
            print(f"  Exclusions: None")

if __name__ == "__main__":
    verify_metadata()
