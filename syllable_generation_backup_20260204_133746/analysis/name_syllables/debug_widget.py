import os
import keypoint_moseq as kpms
import pandas as pd
import sys

# --- CONFIGURATION ---
base_path = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll"
project_dir = os.path.join(base_path, "amph_project")
model_name = "amph_prod_k120k_run_1 copy"

print(f"Testing label_syllables for:")
print(f"Project: {project_dir}")
print(f"Model:   {model_name}")

# 1. Load Config
try:
    config = kpms.load_config(project_dir)
    print("Config loaded.")
except Exception as e:
    print(f"FAIL: Logic Error loading config: {e}")
    sys.exit(1)

# 2. Compute/Load moseq_df
# label_syllables needs moseq_df to know about syllable usage/frequencies usually
print("Computing moseq_df...")
try:
    moseq_df = kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True)
    print(f"moseq_df loaded with {len(moseq_df)} rows.")
except Exception as e:
    print(f"FAIL: Could not compute moseq_df: {e}")
    sys.exit(1)
    
# 3. Check Grid Movies Directory
grid_movie_dir = os.path.join(project_dir, model_name, "grid_movies")
if not os.path.exists(grid_movie_dir):
    print(f"FAIL: Grid movie dir not found at {grid_movie_dir}")
    sys.exit(1)
    
movies = os.listdir(grid_movie_dir)
if len(movies) == 0:
    print("FAIL: Grid movie dir is empty.")
    sys.exit(1)
else:
    print(f"Found {len(movies)} grid movies.")

# 4. Dry Run label_syllables
# This usually launches a widget. In a script, it might just return the widget object or print.
# We want to ensure it doesn't Crash looking for files.
print("Attempting to initialize label_syllables...")
try:
    # Note: In a headless environment, this might warn about Ipywidgets, but shouldn't crash 
    # if the logic (finding files) is correct.
    widget_or_obj = kpms.label_syllables(project_dir, model_name, moseq_df)
    print("SUCCESS: label_syllables returned without error.")
    print(f"Returned object type: {type(widget_or_obj)}")
except Exception as e:
    print(f"FAIL: label_syllables crashed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
