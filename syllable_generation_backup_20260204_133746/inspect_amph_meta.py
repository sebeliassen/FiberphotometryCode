import sys
import os
import pickle

# Add parent directory to path so 'syllable_generation' is importable as a package if needed
# But if 'syllable_generation' is the CWD, then we need to add the parent of CWD
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# Also add CWD just in case
sys.path.append(os.getcwd())

# Attempt to import loader, might need try/except if structure varies
try:
    from analysis.replications.panel_1z.lib import loader
except ImportError:
    # If standard loader fails due to pickle issues, try manual load
    pass

DATA_PATH = "data/jonathan_amph_sessions_v2_w_sylls.pkl"

def load_with_hack():
    # Hack to handle unpickling if classes are missing
    import pickle
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'syllable_generation.datamodel':
                 # Ensure this module is importable or mocked if needed
                 pass
            return super().find_class(module, name)
            
    with open(DATA_PATH, 'rb') as f:
        # Just try standard load first
        return pickle.load(f)

if __name__ == "__main__":
    try:
        data = load_with_hack()
        print(f"Loaded {len(data)} items.")
        s = data[0]
        print("Metadata Keys:", s.metadata.keys())
        print("Injection Frame:", s.metadata.get("injection_frame"))
        
        # Check distribution
        inj_frames = [s.metadata.get("injection_frame") for s in data]
        print(f"Injection Frames found: {len([x for x in inj_frames if x is not None])}")
        
        # Check by Drug
        by_drug = {}
        for s in data:
            drug = s.metadata.get("drug", "UNK")
            inj = s.metadata.get("injection_frame")
            if drug not in by_drug: by_drug[drug] = []
            if inj is not None: by_drug[drug].append(inj)
            
        for drug, frames in by_drug.items():
            print(f"Drug: {drug}, Count: {len(frames)}, Unique Frames: {set(frames)}")
        
    except Exception as e:
        print(f"Error: {e}")
