
import sys
sys.path.append("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.append("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation")

from analysis.ulrik_kinematics.lib import loader
import pandas as pd

# Load one IFDN session to check cols
sessions = loader.load_dataset("ifdn", base_dir="/home/fsp585/Desktop/FiberphotometryCode")
if sessions:
    s = sessions[0]
    print(f"Columns: {s.all_data_df.columns.tolist()}")
    
    # Check if angular velocity exists or if we need to compute it
    if "angular_velocity" in s.all_data_df.columns:
        print("Found angular_velocity")
    elif "heading" in s.all_data_df.columns:
        print("Found heading (can compute diff)")
    else:
        print("No heading or ang_vel found!")
