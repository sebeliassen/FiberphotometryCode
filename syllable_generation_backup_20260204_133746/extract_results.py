
import sys
import os
from pathlib import Path
import keypoint_moseq as kpms

# Boilerplate to ensure we can load local modules if needed, 
# though here we mainly need the installed kpms
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    project_dir = "syllable_generation/syllabizers/keypoint_moseq_syll/amph_project"
    model_name = "amph_full_run"
    
    full_model_path = os.path.join(project_dir, model_name)
    print(f"Loading checkpoint from {full_model_path}...")
    
    # Load the checkpoint
    # kpms.load_checkpoint returns items: model, data, metadata, current_iter
    try:
        model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)
        print(f"Checkpoint loaded. Iteration: {current_iter}")
        
        print("Extracting results...")
        # Matching signature found in syll_results.py:
        # kpms.extract_results(model, metadata, str(project_dir), model_name, ...)
        results = kpms.extract_results(model, metadata, project_dir, model_name)
            
        print("Saving results.h5...")
        # explicitly construct path to check
        save_path = os.path.join(project_dir, model_name, "results.h5")
        print(f"Target path: {save_path}")
        
        print(f"Target path: {save_path}")
        
        # Use explicit path (based on error message hint)
        # "model_name and project_dir are required if path is None"
        # implies if path IS NOT None, we don't need the others.
        try:
             kpms.save_results(results, project_dir=project_dir, model_name=model_name, path=save_path)
             print("Saved using save_results with explicit path.")
        except Exception as e_inner:
             print(f"save_results failed: {e_inner}")
             print("Attempting fallback to save_hdf5...")
             kpms.save_hdf5(save_path, results)
             print("Saved using save_hdf5 fallback.")
        
        if os.path.exists(save_path):
            print(f"SUCCESS: File created at {save_path}")
        else:
            print(f"FAILURE: File NOT found at {save_path}")
            
        print("Done.")
        
    except Exception as e:
        print(f"Error: {e}")
        # Print available attributes of kpms to debug if function names differ
        print("KPMS dir:", dir(kpms))

if __name__ == "__main__":
    main()
