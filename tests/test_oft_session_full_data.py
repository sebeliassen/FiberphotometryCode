#!/usr/bin/env python
# coding: utf-8

# ## Creation of sessions

# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[2]:


import sys
from pathlib import Path

# If you launched the notebook from the project root (the folder that contains both "tests/" and "fiberphotometry/"),
# then Path.cwd() is already the project root:
project_root = Path.cwd()

# If for some reason your notebook’s working directory is "tests/", you can step up one level:
if not (project_root / "fiberphotometry").exists():
    project_root = project_root.parent

# Now insert that folder into Python’s import search path:
sys.path.insert(0, str(project_root))

# After this, the import should work:
from fiberphotometry.data.data_loading import load_all_sessions
from fiberphotometry.data.session_loading import populate_containers
from fiberphotometry.data.oft_session_loading import populate_oft_containers


# In[3]:


from fiberphotometry.config import PLOTTING_CONFIG
from fiberphotometry.processing.plotting_setup import PlottingSetup


# In[4]:


path = Path("/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_full/Jonathan_IFxDN_csvs")


# In[5]:


sessions = load_all_sessions(
        baseline_dir=str(path),
        session_type="oft",
        remove_bad_signal_sessions=True
        )


# In[6]:


populate_containers(sessions)


# In[ ]:


populate_oft_containers(sessions, '/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_full/h5_files', 
                        '/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_full/video_files')


# In[ ]:


import pandas as pd
from pathlib import Path
import os
import re

def simplify_filename_details(filename: str) -> str:
    """
    Finds all occurrences of spatial keywords in a filename and joins them.
    It includes 'top' only if it appears in lowercase, to avoid matching
    camera names like 'CenterTopCam'.
    """
    keywords = ['left', 'right', 'bottom', 'top', 'upper', 'lower']
    pattern = r'|'.join(keywords)
    matches = re.finditer(pattern, filename, flags=re.IGNORECASE)
    
    final_words = []
    for match in matches:
        word = match.group(0)
        if word.lower() == 'top':
            if word == 'top':
                final_words.append(word)
        else:
            final_words.append(word)
            
    return '_'.join([w.capitalize() for w in final_words])

# 1. Create a flat list of records
records = []
for s in sessions:
    session_id = f"{s.trial_id}/{s.chamber_id}"
    trial_dir = Path(s.trial_dir)
    
    # A single loop now handles all data types
    for key, df in s.dfs.data.items():
        data_type = "unknown"
        details = "N/A"
        
        if key.startswith('phot_'):
            data_type = "photometry"
            details = "Photometry Data"

        elif key.startswith('tracking_'):
            data_type = "tracking_h5"
            path_col = df.columns[0]
            filename = os.path.basename(df[path_col].iloc[0])
            details = simplify_filename_details(filename)

        elif key.startswith('video_'):
            path_col = df.columns[0]
            filename = os.path.basename(df[path_col].iloc[0])
            # Dynamically get the file extension (e.g., 'avi')
            extension = Path(filename).suffix.lstrip('.')
            data_type = f"video_{extension}"
            details = simplify_filename_details(filename)
        
        elif key.startswith('sync_'):
            data_type = "sync_csv"
            
            # --- THIS BLOCK IS NOW CORRECTED ---
            try:
                # 1. Get the camera name from the new column inside the dataframe
                cam_name = df['camera_name'].iloc[0]
                
                # 2. Now use that correct camera name to find the file path
                path = next(trial_dir.glob(f'{cam_name}_AllTimestampsCombined*.csv'))
                details = simplify_filename_details(path.name)
            except (StopIteration, KeyError, IndexError):
                # Catch errors if column/file is missing or df is empty
                details = "Info not available"
                
        else:
            continue
            
        records.append((session_id, data_type, key, details))

# 2. Convert the list of records into the final DataFrame
summary_df = pd.DataFrame(records, 
                          columns=['Session ID', 'Data Type', 'Source Key', 'Simplified Details'])

# 3. Sort the data for better readability
summary_df = summary_df.sort_values(by=['Session ID', 'Data Type', 'Source Key']).reset_index(drop=True)

# with pd.option_context('display.max_rows', None):
#     display(summary_df)


# In[ ]:


for df_name, df in sessions[0].dfs.data.items():
    print(df_name, df.columns)


# In[ ]:


PlottingSetup(**PLOTTING_CONFIG['cpt']).apply_plotting_setup_to_sessions(sessions)


# In[ ]:


# In your notebook
import pprint # Make sure pprint is imported
from fiberphotometry.data.session_synchronization import synchronize_session_data

print("--- ⏳ Starting synchronization for ALL sessions... ---")

# Loop through every session object in the `sessions` list
for i, s in enumerate(sessions):
    print(f"Processing session {i+1}/{len(sessions)}: {s.trial_id}")
    try:
        # Run the synchronization function on the current session `s`
        synchronize_session_data(s, downsample_video_by=1, tolerance=0.02)
        print(f"✅ Synchronization successful for {s.trial_id}.")
    except Exception as e:
        # Adding a try...except block is good practice in a loop.
        # If one session fails, it won't crash the whole process.
        print(f"❌ ERROR synchronizing session {s.trial_id}: {e}")
    print("-" * 50)

print("\n--- ✅ Synchronization finished for all sessions. ---")

# --- Verification for ALL sessions ---
print("="*50)
print("VERIFICATION OF FINAL DATAFRAME STRUCTURE")
print("="*50)

all_sessions_verified = True
for i, s in enumerate(sessions):
    # Find the synchronized keys for the CURRENT session `s`
    sync_keys = [k for k in s.dfs.data if k.startswith('synchronized_')]
    
    if not sync_keys:
        print(f"❌ Session {i} ({s.trial_id}): FAILED. No synchronized DataFrames found.")
        all_sessions_verified = False
    else:
        print(f"✅ Session {i} ({s.trial_id}): OK. Found {len(sync_keys)} synchronized DataFrame(s).")

if not all_sessions_verified:
    print("\n⚠️ Some sessions failed synchronization. Please review logs above.")
else:
    print("\n--- All sessions were successfully processed. ---")


# --- Detailed structure check of the FIRST session ---
print("\n\n--- Detailed structure check for the first session (sessions[0]) ---")
s0 = sessions[0] # We can safely define s0 here to inspect the first session
sync_keys = sorted([k for k in s0.dfs.data if k.startswith('synchronized_')])

if not sync_keys:
    print("\nNo synchronized DataFrames were found in the first session.")
else:
    for key in sync_keys:
        final_df = s0.dfs.data[key]
        print(f"\n--- Verification for: {key} ---")
        print(f"Total number of rows: {len(final_df)}")
        print("Columns:")
        pprint.pprint(final_df.columns.to_list())


# In[ ]:


from fiberphotometry.kpms.syllable_generator import generate_syllable_models
from fiberphotometry.kpms.syllable_appender import extract_and_append_results

print("Successfully imported syllable analysis modules!")


# In[ ]:


import keypoint_moseq as kpms


# In[ ]:


# # params
curr_frac = 1
latent_dim = 9
n_jax_batches = 24

# --- Define parameters for the FULL run ---
PROJECT_DIR_FULL = f"./kpms_all_sessions_frac_{curr_frac}_ld{latent_dim}_jn{n_jax_batches}"  # Using a new directory for the full analysis
KAPPA = 1.205e+06

# --- Run the model generation process on ALL sessions ---
print("--- 🚀 STARTING SYLLABLE MODEL GENERATION (FULL RUN) ---")

# Capture all three return values from the function
model_names, original_lengths, video_frame_indexes, final_metadata = generate_syllable_models(
    sessions=sessions,
    project_dir=PROJECT_DIR_FULL, # Use the new directory
    kappa=KAPPA,
    subset_fraction=curr_frac,
    num_models=5,               # You can increase this for a more robust model (e.g., 3)
    ar_iters=50,
    fitting_iters=200,
    n_jax_batches=n_jax_batches,
    latent_dim=latent_dim,
    test_mode=False             # <-- Set to False to process all files
)

print(f"\n--- ✅ FULL MODEL GENERATION FINISHED. {len(model_names)} model(s) created. ---")


# In[ ]:


import numpy as np


# In[ ]:


for model_name in model_names:
    print(f"--- 📎 Appending and aligning results for model: {model_name} ---")
    extract_and_append_results(
        sessions=sessions,
        project_dir=PROJECT_DIR_FULL,
        model_name=model_name,
        video_frame_indexes=video_indexes_for_fitting,
        original_lengths=original_lengths,
        metadata=metadata
    )


# In[ ]:


# --- 🔎 VERIFY RESULTS ---
print("\n--- Verifying that syllable data was appended to sessions ---")

# The default key where results are stored
data_key = 'all_results' 

for session in sessions:
    # Check if the new attribute was successfully added
    if hasattr(session, data_key):
        # Access the dictionary of results
        results_dict = getattr(session, data_key)
        
        # Check if the 'syllable' array exists and print its shape
        if 'syllable' in results_dict:
            syllable_shape = results_dict['syllable'].shape
            print(f"✅ Success! Session {session.trial_id} now has '{data_key}' with syllable data shape: {syllable_shape}")
        else:
            print(f"⚠️ Warning: Session {session.trial_id} has '{data_key}' but is missing the 'syllable' key.")
    else:
        print(f"❌ Error: Session {session.trial_id} is missing the '{data_key}' attribute.")


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("--- 📊 Generating data availability plots for all processed sessions ---")

# --- Loop through every session ---
plots_generated = 0
for s in sessions:
    # Check if the current session has the `all_results` attribute
    if hasattr(s, 'all_results'):
        
        # --- Create and display the plot for this specific session ---
        print(f"\nPlotting for session with trial_id: {s.trial_id}")
        
        results_dict = s.all_results
        syllable_array = results_dict['syllable']
        data_exists_mask = ~np.isnan(syllable_array)
        
        fig, ax = plt.subplots(figsize=(18, 4))
        ax.plot(data_exists_mask, color='#007acc', linewidth=1.5)
        
        # Use the current session's ID in the title
        ax.set_title(f"Data Availability for Session {s.trial_id}", fontsize=16)
        ax.set_xlabel("Frame Number (Full Original Timeline)", fontsize=12)
        ax.set_ylabel("Data Status", fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['NaN (Gap)', 'Valid Data'])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.show() # Display the plot for this session before creating the next one
        
        plots_generated += 1

# --- Final Summary ---
if plots_generated == 0:
    print("\n❌ No processed sessions with 'all_results' found in memory to plot.")
else:
    print(f"\n--- ✅ Plot generation complete. {plots_generated} plots created. ---")


# In[ ]:


import pickle

# 1. Bundle all the "juice" (essential, slow-to-generate variables) into one dictionary.
#    This ensures you don't lose the outputs of your long computation.
print("--- Bundling essential variables for saving... ---")
state_to_save = {
    'sessions': sessions,
    'model_names': model_names,
    'video_frame_indexes': video_frame_indexes,
    'original_lengths': original_lengths,
    'project_dir_full': PROJECT_DIR_FULL, # Saving for reference
    'kappa': KAPPA                      # Saving for reference
}
print("✅ Bundle complete.")

# 2. Define a new, more descriptive filename for the complete state.
save_filename = f'full_analysis_state_full_data_{curr_frac}_fst_3_models.pkl'

# 3. Save the entire dictionary to the file.
print(f"\n--- 💾 Saving the complete analysis state to '{save_filename}'... ---")
with open(save_filename, 'wb') as f:
    # We are now saving the 'state_to_save' dictionary, which contains everything.
    pickle.dump(state_to_save, f)

print(f"--- ✅ Save complete. All important data is now in '{save_filename}'. ---")


# ## Loading of sessions

# ### Imports and session loading

# In[1]:


# --- IMPORTS & SETUP ---

# Enable autoreload (optional but good practice)
# %load_ext autoreload
# %autoreload 2

# Standard library & path setup
import sys
from pathlib import Path

# Data handling and analysis
import pandas as pd
import numpy as np
import pickle
import re

# Visualization
import matplotlib.pyplot as plt
import pprint

# Modeling pipeline
import keypoint_moseq as kpms
import jax

# --- 1. SETUP PROJECT PATH ---
print("Setting up project path...")
project_root = Path.cwd()
if not (project_root / "fiberphotometry").exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))
print(f"Project root added to path: {project_root}")

# --- 2. IMPORT CUSTOM MODULES ---
print("\nImporting custom project modules...")
try:
    # core Session class and loader
    from fiberphotometry.data.data_loading import Session, load_all_sessions

    # container for holding session data
    from fiberphotometry.data.session_loading import DataContainer

    # OFT-specific session population
    from fiberphotometry.data.oft_session_loading import populate_oft_containers

    # synchronizing different data streams in a session
    from fiberphotometry.data.session_synchronization import synchronize_session_data

    # KPMS pipeline stages
    from fiberphotometry.kpms.syllable_generator import generate_syllable_models
    from fiberphotometry.kpms.syllable_appender import extract_and_append_results
    from fiberphotometry.kpms.trimming import create_trimmed_coords_and_confs
    from fiberphotometry.kpms.batching import create_batches

    print("✅ All custom modules imported successfully.")
except ImportError as e:
    print(f"❌ Failed to import a custom module: {e}")
    print("   Please ensure your notebook is in the correct project directory.")


# In[2]:


import pickle
import pandas as pd
import numpy as np

# 1. This filename must exactly match the new file where you saved the dictionary.
#load_filename = f'full_analysis_state_full_data_{curr_frac}.pkl'
# load_filename = 'full_analysis_state_full_data_1.pkl'
load_filename = 'full_analysis_state_full_data_1_500_iters.pkl'
print(f"--- 🚀 Loading complete analysis state from '{load_filename}'... ---")

try:
    # Use 'with open' to safely open the file.
    with open(load_filename, 'rb') as f:
        # 2. Load the entire dictionary object that you saved.
        loaded_state = pickle.load(f)

    # 3. Unpack all the variables from the dictionary back into your notebook's workspace.
    sessions = loaded_state['sessions']
    model_names = loaded_state['model_names']
    video_frame_indexes = loaded_state['video_frame_indexes']
    original_lengths = loaded_state['original_lengths']
    PROJECT_DIR_FULL = loaded_state['project_dir_full']
    KAPPA = loaded_state['kappa']

    print(f"--- ✅ Load complete. Unpacked data for {len(sessions)} sessions into memory. ---")

    # 4. A more comprehensive verification to ensure everything loaded correctly.
    if 'sessions' in locals() and 'model_names' in locals() and 'video_frame_indexes' in locals():
         print("\n✅ Verification successful: 'sessions', 'model_names', and other key variables are loaded.")
         print("You can now proceed directly to your analysis and plotting cells.")
    else:
        print("\n⚠️ Verification failed. Some key variables are missing from the loaded data.")

except FileNotFoundError:
    print(f"\n❌ ERROR: The file '{load_filename}' was not found. Make sure it's in the same directory as your notebook.")
except KeyError as e:
    print(f"\n❌ ERROR: A key was missing from the loaded dictionary: {e}. The save file might be from an older version of your code.")


# ### Sanity checks

# ### Syllable syncing

# In[3]:


import pandas as pd

print("--- 🧠 Merging syllable data into synchronized DataFrames ---")

sessions_merged = 0
for i, s in enumerate(sessions):
    try:
        print(f"Processing session {i+1}/{len(sessions)}: {s.trial_id}")
        
        # --- 1. Check if the required data exists ---
        if not hasattr(s, 'all_results'):
            raise KeyError("Attribute 'all_results' not found on this session object.")
        if 'synchronized_data_tracking_0' not in s.dfs.data:
            raise KeyError("DataFrame 'synchronized_data_tracking_0' not found in s.dfs.data.")

        # --- 2. Get the target DataFrame and source data ---
        target_df = s.dfs.data['synchronized_data_tracking_0']
        syllable_array = s.all_results['syllable']

        # --- 3. Perform the length check as you requested ---
        if len(target_df) != len(syllable_array):
            # If lengths don't match, raise an error to be caught by the 'except' block
            raise ValueError(f"Length mismatch! DataFrame has {len(target_df)} rows, but syllable array has {len(syllable_array)} elements.")

        # --- 4. If checks pass, add the new column ---
        # This modifies the DataFrame directly within your session object
        target_df['syllable'] = syllable_array
        print(f"✅ Successfully added 'syllable' column.")
        sessions_merged += 1

    except (KeyError, ValueError) as e:
        # This will catch any errors raised above and print a warning,
        # allowing the loop to continue with the next session.
        print(f"❌ SKIPPING session {s.trial_id}: {e}")
    
    print("-" * 50)

print(f"\n--- ✅ Merge complete. {sessions_merged} out of {len(sessions)} sessions were updated. ---")


# --- Final Verification ---
# Let's check the first session to see if the new column is there.
print("\n--- 🔍 Final Verification of the first session ---")
if 'sessions' in locals() and 'syllable' in sessions[0].dfs.data['synchronized_data_tracking_0'].columns:
    print("Verification successful: 'syllable' column is present.")
    # Display the first few rows of the updated DataFrame
    #display(sessions[0].dfs.data['synchronized_data_tracking_0'].head())
else:
    print("Verification failed: 'syllable' column not found in the first session's DataFrame.")


# ### Model loading and more sanity checks

# In[6]:


PROJECT_DIR = "./kpms_all_sessions_frac_1_ld9_jn24"  # Using a new directory for the full analysis
MODEL_NAME = 'final_model_1.20e+06_idx4'


# In[ ]:


# # define your ten model names
# model_names = [f'final_model_1.20e+06_idx{i}' for i in [0, 1, 2, 4]]

# # compute EML scores and their standard errors
# eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(PROJECT_DIR, model_names)

# # pick the best model
# best_model_name = model_names[np.argmax(eml_scores)]
# print(f"Best model: {best_model_name}")

# # visualize the EML scores across the ensemble
# kpms.plot_eml_scores(eml_scores, eml_std_errs, model_names)


# In[7]:


import keypoint_moseq as kpms
model, data, metadata, current_iter = kpms.load_checkpoint(PROJECT_DIR, MODEL_NAME)
results = kpms.extract_results(model, metadata, PROJECT_DIR, MODEL_NAME, save_results=False)


# In[8]:


model_names = [f'final_model_1.20e+06_idx{i}' for i in [0, 1, 2, 4]]

for model_name in model_names:
    # load the specific checkpoint
    model, data, metadata, current_iter = kpms.load_checkpoint(PROJECT_DIR, model_name)
    
    # extract results for this model
    results = kpms.extract_results(model, metadata, PROJECT_DIR, model_name, save_results=True)


# In[9]:


config = kpms.load_config(PROJECT_DIR)


# In[10]:


import os
import h5py
import numpy as np
import jax.numpy as jnp
from jax_moseq.utils import unbatch
from jax_moseq.models.keypoint_slds import estimate_coordinates

# compute the estimated coordinates
Y_est = estimate_coordinates(
    jnp.array(model['states']['x']),
    jnp.array(model['states']['v']),
    jnp.array(model['states']['h']),
    jnp.array(model['params']['Cd'])
)

# generate a dictionary with reconstructed coordinates for each recording
coordinates_est = unbatch(Y_est, *metadata)


# In[11]:


coordinates_est['Arena1234_CenterTopCamVid_2024-12-13T12_20_06_top_leftDLC_HrnetW32_dataset_fixDec16shuffle10_detector_020_snapshot_020'].shape


# In[12]:


for k, v in model['states'].items():
    print(k, v.shape)


# In[13]:


for k, v in model['params'].items():
    print(k, v.shape)


# In[14]:


for k, v in config.items():
    # try .shape first; getattr(v, 'shape', None) will be None if no .shape
    shape = getattr(v, 'shape', None)
    if shape is None:
        try:
            # if it has a __len__, len(v) will work
            shape = len(v)
        except TypeError:
            # neither .shape nor len → skip
            continue
    print(k, shape)


# In[15]:


for k, v in coordinates_est.items():
    print(k, v.shape)


# ### Trajectory plots

# In[16]:


perm = kpms.reindex_syllables_in_checkpoint(PROJECT_DIR, MODEL_NAME)

config['bodyparts'] = config['use_bodyparts']
kpms.generate_trajectory_plots(coordinates_est, results, PROJECT_DIR, MODEL_NAME, fps=60, min_frequency=0.000, min_duration=1, **config)


# ### Grid movie preprocessing and generation

# In[17]:


import os
import pprint # Make sure pprint is imported for nice printing

def map_syllable_parts_to_videos(video_frame_indexes: dict, video_dir: str) -> dict:
    """
    Maps syllable part keys (e.g., '..._part1') to their corresponding video paths.

    This function is designed to work after keypoint-MoSeq splits a single
    recording into multiple parts during model fitting. It ensures that every key
    in the video_frame_indexes dictionary is mapped to the correct video file.

    Args:
        video_frame_indexes: The dictionary returned by the syllable generation step,
                             where keys may have '_partX' suffixes.
        video_dir: The path to the directory containing the video files.

    Returns:
        A dictionary where keys are the syllable part keys and values are the
        full paths to the single corresponding video file.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]
    
    video_path_map = {}
    
    # Iterate through the keys of the video_frame_indexes dictionary
    for part_key in video_frame_indexes.keys():
        
        # Find the original H5 base name by stripping any '_partX' suffix.
        # We split the string at '_part' and take the first piece.
        h5_base_name = part_key.split('_part')[0]
        
        try:
            # Use the same logic as before to find the video's root name from the H5 base name
            video_root = h5_base_name.split('DLC_')[0]

            # Find the matching video file in the video directory
            found_match = False
            for video_file in video_files:
                if video_file.startswith(video_root):
                    # CRITICAL: Map the *original part_key* to the found video path.
                    # This ensures every part maps back to the same, single video file.
                    video_path_map[part_key] = os.path.join(video_dir, video_file)
                    found_match = True
                    break # Match found, move to the next part_key
            
            if not found_match:
                 print(f"Warning: No matching video found for key: {part_key}")

        except IndexError:
            print(f"Warning: Could not parse '{part_key}'. 'DLC_' delimiter not found. Skipping.")

    return video_path_map


# In[18]:


# --- Define the paths to your directories ---
VIDEO_DIRECTORY = '/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_full/video_files'

# --- Call the NEW function to get the correct mapping ---
# It uses the keys from video_frame_indexes as the source of truth.
h5_to_video_map_corrected = map_syllable_parts_to_videos(
    video_frame_indexes=video_frame_indexes,
    video_dir=VIDEO_DIRECTORY
)

# --- Print the results nicely to verify ---
print("--- Corrected File Mapping Results ---")
pprint.pprint(h5_to_video_map_corrected)


# In[19]:


centroids, headings = kpms.get_centroids_headings(coordinates_est, **config)


# In[20]:


video_frame_indexes


# In[21]:


import numpy as np

# --- 1. Correct the video frame indexes for 30 FPS ---
# We use integer division (//) to ensure the result is a valid frame index.
print("--- Adjusting video_frame_indexes for 30 FPS videos... ---")
# video_frame_indexes_30fps = {
#     key: (arr // 2) for key, arr in video_frame_indexes.items()
# }

# --- 2. (Optional) Verification Step ---
# Let's verify the change for one of the recordings
first_key = list(video_frame_indexes.keys())[0]
print(f"\nVerifying for key: {first_key}")
print(f"  Original indexes (first 5): {video_frame_indexes[first_key][:5]}")
print(f"  Corrected indexes (first 5):  {video_frame_indexes[first_key][:5]}")


# In[ ]:


# --- 1. Align Dictionaries to a Common Set of Keys ---
# The AssertionError indicates a mismatch between the keys in `results` (from the model)
# and `video_frame_indexes` (from the syllable generation step).
# We will create temporary versions of all input dicts containing only the keys they share.

print("--- Aligning input dictionaries to a common set of keys... ---")

# Find the set of keys that exist in BOTH `results` and `video_frame_indexes_30fps`.
keys_results = set(results.keys())
keys_frames = set(video_frame_indexes.keys())
common_keys = keys_results.intersection(keys_frames)

print(f"Original `results` keys: {len(keys_results)}")
print(f"Original `video_frame_indexes` keys: {len(keys_frames)}")
print(f"Found {len(common_keys)} common keys to use for movie generation.")

# For debugging, show the differences (optional, can be commented out)
if len(keys_results) != len(common_keys) or len(keys_frames) != len(common_keys):
    print("\nKeys in results ONLY:", keys_results - keys_frames)
    print("Keys in frames ONLY:", keys_frames - keys_results)
    print("-" * 20)

# Create new, temporary dictionaries filtered to only contain the common keys.
results_common = {k: results[k] for k in common_keys}
video_frame_indexes_common = {k: video_frame_indexes[k] for k in common_keys}
h5_to_video_map_common = {k: h5_to_video_map_corrected[k] for k in common_keys if k in h5_to_video_map_corrected}
coordinates_common = {k: coordinates_est[k] for k in common_keys if k in coordinates_est}
centroids_common = {k: centroids[k] for k in common_keys if k in centroids}
headings_common = {k: headings[k] for k in common_keys if k in headings}


# --- 2. Create a temporary copy of the config to avoid the 'fps' conflict ---
config_for_movies = config.copy()
if 'fps' in config_for_movies:
    del config_for_movies['fps']


# --- 3. Now, call the function with the ALIGNED and CORRECTED data ---
print("\n--- Generating grid movies with aligned data... ---")
kpms.generate_grid_movies(
    results_common,
    PROJECT_DIR,
    MODEL_NAME,
    video_paths=h5_to_video_map_common,
    coordinates=coordinates_common,
    centroids=centroids_common,
    headings=headings_common,
    fps=60,
    video_frame_indexes=video_frame_indexes_common,
    overlay_keypoints=False,
    **config_for_movies
)


# ### Group assignment and syllable labeling

# In[ ]:


import re

# metadata_names is the list/array of all result names (e.g., metadata[0])
metadata_names = metadata[0]

for sess in sessions:
    sess.result_names = []
    h5_path = sess.kpms_h5_path
    for name in metadata_names:
        # strip "_part#" only for matching
        base = re.sub(r'_part\d+$', '', name)
        if base in h5_path and name not in sess.result_names:
            # append the full original name (with part suffix)
            sess.result_names.append(name)


# In[ ]:


import pandas as pd

# build a list of dicts
rows = []
for sess in sessions:
    for name in sess.result_names:
        rows.append({
            "name": name,
            "group": sess.genotype
        })

# make a DataFrame
index_df = pd.DataFrame(rows)

# (optionally) save it back out so your interactive widget picks it up
index_df.to_csv(os.path.join(PROJECT_DIR, "index.csv"), index=False)

# inspect
print(index_df.head())


# In[ ]:


moseq_df = kpms.compute_moseq_df(PROJECT_DIR, MODEL_NAME, smooth_heading=True)


# In[ ]:


moseq_df


# In[ ]:


stats_df = kpms.compute_stats_df(
    PROJECT_DIR,
    MODEL_NAME,
    moseq_df,
    min_frequency=0.005,  # threshold frequency for including a syllable in the dataframe
    groupby=["group", "name"],  # column(s) to group the dataframe by
    fps=60,
)  # frame rate of the video from which keypoints were inferred

stats_df


# In[ ]:


MODEL_NAME


# In[ ]:


kpms.label_syllables(PROJECT_DIR, MODEL_NAME, moseq_df)


# In[ ]:


sessions[0].kpms_h5_path


# In[ ]:


kpms.interactive_group_setting


# ### Plotting setup and more checks

# In[ ]:


from fiberphotometry.config import PLOTTING_CONFIG
from fiberphotometry.processing.plotting_setup import PlottingSetup


# In[ ]:


PlottingSetup(**PLOTTING_CONFIG['cpt']).apply_plotting_setup_to_sessions(sessions)


# In[ ]:


plt.plot(sessions[0].all_results['syllable'])


# In[ ]:


sessions[0].dfs.data.keys()


# In[ ]:


sessions[0].dfs.data['synchronized_data_tracking_0']


# In[ ]:


sessions[0].dfs.data['phot_470']


# ### Syllable based signatures

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.signal import savgol_filter

# --- Analysis & Smoothing Parameters ---
SAMPLING_RATE = 80
WINDOW_SECONDS = 10
SMOOTHING_WINDOW = 25
SMOOTHING_POLY_ORDER = 3
CONFIDENCE_LEVEL = 1.96
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLING_RATE)
TOTAL_SAMPLES = 2 * WINDOW_SAMPLES
SIGNAL_COLS = ['signal_0_470_dff_z', 'signal_1_470_dff_z']

# --- Syllable Mapping Setup ---

# 1. This map orders syllables by frequency (new_index 0 = most frequent).

reindex_map = np.arange(100)
old_id_to_new_idx = {old_id: new_idx for new_idx, old_id in enumerate(reindex_map)}

# 2. Load human-readable labels from CSV.
LABEL_CSV_PATH = '/home/fsp585/Desktop/FiberphotometryCode/tests/kpms_all_sessions_frac_1_ld9_jn24/final_model_1.20e+06_idx3_backup/syll_info.csv' # Adjust path as needed
labels_df = pd.read_csv(LABEL_CSV_PATH)
labels_df['label'] = labels_df['label'].fillna('').str.strip()
new_idx_to_label = {
    row.syllable: row.label
    for _, row in labels_df.iterrows() if row.label
}

# 3. Create a final map from original ID to final label.
syllable_id_to_label = {}
for old_id in range(100):
    new_idx = old_id_to_new_idx.get(old_id)
    if new_idx is not None:
        label = new_idx_to_label.get(new_idx, "non-common")
        syllable_id_to_label[old_id] = label
    else:
        syllable_id_to_label[old_id] = "non-common"


# --- Data Processing Function ---
def process_session_data(session):
    """Processes a single session to extract, smooth, and baseline-correct traces."""
    df = session.dfs.data['synchronized_data_tracking_0']
    
    onsets = df[(df['syllable'].notna()) & (df['syllable'] != df['syllable'].shift(1))]
    
    syllable_traces = {s: {col: [] for col in SIGNAL_COLS} for s in onsets['syllable'].unique()}
    for index, row in onsets.iterrows():
        syllable = row['syllable']
        start_idx = index - WINDOW_SAMPLES
        end_idx = index + WINDOW_SAMPLES
        if start_idx >= 0 and end_idx <= len(df):
            for col in SIGNAL_COLS:
                trace = df.loc[start_idx:end_idx-1, col].values
                if len(trace) == TOTAL_SAMPLES:
                    syllable_traces[syllable][col].append(trace)

    processed_traces = {s: {col: [] for col in SIGNAL_COLS} for s in syllable_traces.keys()}
    for syllable, signals in syllable_traces.items():
        for signal_col, traces in signals.items():
            for trace in traces:
                trace_series = pd.Series(trace)
                interpolated = trace_series.interpolate(method='linear', limit_direction='both')
                smoothed = savgol_filter(interpolated, SMOOTHING_WINDOW, SMOOTHING_POLY_ORDER)
                baseline_value = smoothed[WINDOW_SAMPLES]
                normalized_trace = smoothed - baseline_value
                processed_traces[syllable][signal_col].append(normalized_trace)
                
    return processed_traces

# --- Analysis & Plotting ---

# 1. Process every session
# This part assumes 'sessions' is a pre-loaded list of session objects
processed_list = [process_session_data(s) for s in sessions]

# 2. Select the top 20 syllables to plot
# Since reindex_map is sorted by frequency, we can just take the first 20.
top_20_syllables = reindex_map[:100]
print("Top 20 original syllable IDs:", top_20_syllables.tolist())

# 3. Determine unique brain regions and create time vector
regions = sorted({region for s in sessions for region, _, _ in s.brain_regions})
time_vector = np.linspace(-WINDOW_SECONDS, WINDOW_SECONDS, TOTAL_SAMPLES)

# 4. Plot each syllable
for syl_idx, syllable in enumerate(top_20_syllables): # 'syllable' here is the ORIGINAL integer ID
    final_label = syllable_id_to_label.get(syllable, f"Unknown ID: {syllable}")
    
    fig, axes = plt.subplots(1, len(regions), figsize=(22, 6))
    fig.suptitle(f"Syllable Onset: '{final_label}' (Original ID: {int(syllable)})", fontsize=18)

    for ax, region in zip(axes, regions):
        traces_by_genotype = {'KI': [], 'WT': []}
        for s, proc_data in zip(sessions, processed_list):
            meta = s.signal_meta
            for col in SIGNAL_COLS:
                prefix = "_".join(col.split("_")[:2])
                if meta[prefix][0] == region:
                    traces = proc_data.get(syllable, {}).get(col, [])
                    traces_by_genotype[s.genotype].extend(traces)

        for geno, traces in traces_by_genotype.items():
            if len(traces):
                # skip any traces containing NaNs
                valid_traces = [t for t in traces if not np.isnan(t).any()]
                n_skipped = len(traces) - len(valid_traces)
                if n_skipped > 0:
                    print(f"Skipped {n_skipped} {geno} traces for syllable {syllable}, region {region} due to NaNs")
                if not valid_traces:
                    continue
                arr = np.stack(valid_traces)

                m = arr.mean(axis=0)
                e = sem(arr, axis=0)
                color = 'darkblue' if geno == 'KI' else 'green'
                ax.plot(time_vector, m, label=f"{geno} (n={len(traces)})", color=color)
                ax.fill_between(time_vector,
                                m - CONFIDENCE_LEVEL * e,
                                m + CONFIDENCE_LEVEL * e,
                                color=color, alpha=0.2)
                
                ax.relim()
                ax.autoscale_view()

        ax.set_title(region, fontsize=14)
        ax.axvline(0, color='crimson', linestyle='--', lw=2)
        ax.axhline(0, color='grey', linestyle=':', lw=1)
        ax.set_xlabel("Time from Onset (s)")
        ax.grid(True, linestyle=':')
        ax.legend()

    axes[0].set_ylabel("Normalized Signal (z‑score)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if syl_idx < 30:
        plt.savefig(f'syl_{syl_idx}.png')
    plt.show()


# In[ ]:


kpms.plot_syll_stats_with_sem(
    stats_df,
    PROJECT_DIR,
    MODEL_NAME,
    plot_sig=True,  # whether to mark statistical significance with a star
    thresh=0.05,  # significance threshold
    stat="frequency",  # statistic to be plotted (e.g. 'duration' or 'velocity_px_s_mean')
    order="stat",  # order syllables by overall frequency ("stat") or degree of difference ("diff")
    ctrl_group="a",  # name of the control group for statistical testing
    exp_group="b",  # name of the experimental group for statistical testing
    figsize=(8, 4),  # figure size
    groups=stats_df["group"].unique(),  # groups to be plotted
);


# In[ ]:


normalize = "bigram"  # normalization method ("bigram", "rows" or "columns")

trans_mats, usages, groups, syll_include = kpms.generate_transition_matrices(
    PROJECT_DIR,
    MODEL_NAME,
    normalize=normalize,
    min_frequency=0.010,  # minimum syllable frequency to include
)

kpms.visualize_transition_bigram(
    PROJECT_DIR,
    MODEL_NAME,
    groups,
    trans_mats,
    syll_include,
    normalize=normalize,
    show_syllable_names=True,  # label syllables by index (False) or index and name (True)
)


# In[ ]:


kpms.plot_transition_graph_group(
    PROJECT_DIR,
    MODEL_NAME,
    groups,
    trans_mats,
    usages,
    syll_include,
    layout="circular",  # transition graph layout ("circular" or "spring")
    show_syllable_names=False,  # label syllables by index (False) or index and name (True)
)


# In[ ]:


kpms.plot_transition_graph_group(
    PROJECT_DIR,
    MODEL_NAME,
    groups,
    trans_mats,
    usages,
    syll_include,
    layout="circular",  # transition graph layout ("circular" or "spring")
    show_syllable_names=True,  # label syllables by index (False) or index and name (True)
)


# In[ ]:


kpms.plot_transition_graph_difference(
    PROJECT_DIR, MODEL_NAME, groups, trans_mats, usages, syll_include, layout="circular", show_syllable_names=True
)  # transition graph layout ("circular" or "spring")


# In[ ]:




