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
# In[3]:
from fiberphotometry.config import PLOTTING_CONFIG
from fiberphotometry.data.data_loading import load_all_sessions
from fiberphotometry.data.oft_session_loading import populate_oft_containers
from fiberphotometry.data.session_loading import populate_containers
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


# In[7]:


populate_oft_containers(sessions, '/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_full/h5_files',
                        '/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_full/video_files')


# In[8]:


import os
import re
from pathlib import Path

import pandas as pd


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


# In[9]:


for df_name, df in sessions[0].dfs.data.items():
    print(df_name, df.columns)


# In[10]:


PlottingSetup(**PLOTTING_CONFIG['cpt']).apply_plotting_setup_to_sessions(sessions)


# In[11]:


# In your notebook
import pprint  # Make sure pprint is imported

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


# In[12]:


from fiberphotometry.kpms.syllable_appender import extract_and_append_results
from fiberphotometry.kpms.syllable_generator import generate_syllable_models

print("Successfully imported syllable analysis modules!")


# In[13]:


import keypoint_moseq as kpms

# In[14]:


model_names = [f'final_model_1.20e+06_idx{i}' for i in range(5)]


# In[ ]:


import numpy as np

# # params
curr_frac = 1
latent_dim = 3
n_jax_batches = 24

# --- Define parameters for the FULL run ---
PROJECT_DIR_FULL = f"./kpms_few_bp_150ms_frac_{curr_frac}_ld{latent_dim}_jn{n_jax_batches}"  # Using a new directory for the full analysis
#KAPPA = 1.205e+06
KAPPA = 1.21e+04

# --- Run the model generation process on ALL sessions ---
print("--- 🚀 STARTING SYLLABLE MODEL GENERATION (FULL RUN) ---")

model_names, original_lengths, video_frame_indexes, final_metadata = generate_syllable_models(
    sessions=sessions,
    project_dir=PROJECT_DIR_FULL, # Use the new directory
    kappa=KAPPA,
    subset_fraction=curr_frac,
    num_models=10,               # You can increase this for a more robust model (e.g., 3)
    ar_iters=50,
    fitting_iters=200,
    n_jax_batches=n_jax_batches,
    latent_dim=latent_dim,
    test_mode=False             # <-- Set to False to process all files
)

print(f"\n--- ✅ FULL MODEL GENERATION FINISHED. {len(model_names)} model(s) created. ---")


# In[24]:


from trimming import create_trimmed_coords_and_confs

# In[17]:


# # --- Step 1 & 2: Setup and H5 Path finding (no changes needed) ---
# project_path = Path(PROJECT_DIR_FULL)
# if not project_path.exists():
#     kpms.setup_project(PROJECT_DIR_FULL)
# config = kpms.load_config(PROJECT_DIR_FULL)

# all_h5_paths = []
# for session in sessions:
#     session.kpms_h5_path = None
#     for df_name, df in session.dfs.data.items():
#         if (df_name.startswith('tracking_') and 'h5_path' in df.columns and
#             not df['h5_path'].empty and pd.notna(df['h5_path'].iloc[0]) and
#             'Center' in df['h5_path'].iloc[0]):
#             session.kpms_h5_path = df['h5_path'].iloc[0]
#             all_h5_paths.append(session.kpms_h5_path)
#             break

# # --- Step 3: Load and preprocess data (Consolidated Logic) ---
# coordinates, confidences, _ = kpms.load_keypoints(all_h5_paths, 'deeplabcut')

# # Get ground-truth info from the FULL data
# original_lengths = {key: len(arr) for key, arr in coordinates.items()}
# trimmed_coords, trimmed_confs, _, full_video_frame_indexes = create_trimmed_coords_and_confs(
#     coordinates, confidences, errors=None
# )

# # Prepare the data for fitting, calling the new helper function
# coords_for_fitting = trimmed_coords
# confs_for_fitting = trimmed_confs
# video_indexes_for_fitting = full_video_frame_indexes


# # Format the data for the model
# data, metadata = kpms.format_data(coords_for_fitting, confs_for_fitting, **config)


# In[18]:




# In[19]:


# remember to rename to video_frame_indexes
for model_name in model_names:
    print(f"--- 📎 Appending and aligning results for model: {model_name} ---")
    extract_and_append_results(
        sessions=sessions,
        project_dir=PROJECT_DIR_FULL,
        model_name=model_name,
        video_frame_indexes=video_frame_indexes,
        original_lengths=original_lengths,
        metadata=final_metadata
    )


# In[20]:


print("\n--- Verifying that per-model syllable data was appended to sessions ---")

for session in sessions:
    store = getattr(session, 'kpms_results_by_model', None)

    if not store:
        print(f"❌ Error: Session {getattr(session, 'trial_id', '?')} has no 'kpms_results_by_model' or it's empty.")
        continue

    for model_name, results_dict in store.items():
        # be tolerant of key naming
        key = 'syllables' if 'syllables' in results_dict else ('syllable' if 'syllable' in results_dict else None)

        if key is None:
            print(
                f"⚠️ Warning: Session {getattr(session, 'trial_id', '?')} [{model_name}] "
                f"missing 'syllables'/'syllable' key. Available keys: {list(results_dict.keys())}"
            )
            continue

        arr = results_dict[key]
        nans = float(np.isnan(arr).mean()) if arr.dtype.kind in {'f'} else 0.0

        # quick consistency check: all arrays share the same first dimension
        shapes = {k: v.shape for k, v in results_dict.items()}
        first_dims = {k: v.shape[0] for k, v in results_dict.items()}
        consistent = len(set(first_dims.values())) == 1

        print(
            f"✅ Session {getattr(session, 'trial_id', '?')} [{model_name}] "
            f"'{key}' shape: {arr.shape} | dtype: {arr.dtype} | NaN%: {nans:.1%} | "
            f"consistent first dim across keys: {consistent} (shapes: {shapes})"
        )


# In[21]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("--- 📊 Generating data availability plots for all processed sessions ---")

selected_model = model_names[0]  # <-- pick one, e.g., 'final_model_1' or final_model_names[0]
plots_generated = 0

for s in sessions:
    break
    store = getattr(s, 'kpms_results_by_model', None)
    if not store or selected_model not in store:
        print(f"⚠️ Skipping {getattr(s, 'trial_id', '?')}: no results for model '{selected_model}'.")
        continue

    results_dict = store[selected_model]
    key = 'syllables' if 'syllables' in results_dict else 'syllable'
    syllable_array = results_dict[key]
    data_exists_mask = ~np.isnan(syllable_array)

    print(f"\nPlotting for session: {s.trial_id}  |  model: {selected_model}")
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(data_exists_mask, linewidth=1.5)
    ax.set_title(f"Data Availability — Session {s.trial_id} — {selected_model}", fontsize=16)
    ax.set_xlabel("Frame Number (Full Original Timeline)", fontsize=12)
    ax.set_ylabel("Data Status", fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['NaN (Gap)', 'Valid Data'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.show()
    plots_generated += 1

if plots_generated == 0:
    print("\n❌ No sessions had results for the selected model to plot.")
else:
    print(f"\n--- ✅ Plot generation complete. {plots_generated} plots created. ---")


# In[22]:


import pickle

# 1. Bundle all the "juice" (essential, slow-to-generate variables) into one dictionary.
#    This ensures you don't lose the outputs of your long computation.
print("--- Bundling essential variables for saving... ---")

# keep both keys for compatibility; they can point to the same list
state_to_save = {
    'sessions': sessions,                           # contains kpms_results_by_model per session
    'final_model_names': model_names,         # new, explicit
    'model_names': model_names,               # legacy alias for older notebooks
    'video_frame_indexes': video_frame_indexes,
    'original_lengths': original_lengths,
    'project_dir_full': PROJECT_DIR_FULL,
    'kappa': KAPPA
}

print("✅ Bundle complete.")

save_filename = 'kpms_few_bp_150ms_frac_1_ld3_jn24_test.pkl'
print(f"\n--- 💾 Saving the complete analysis state to '{save_filename}'... ---")
with open(save_filename, 'wb') as f:
    pickle.dump(state_to_save, f)
print(f"--- ✅ Save complete. All important data is now in '{save_filename}'. ---")


# ## Loading of sessions

# ### Imports and session loading

# In[25]:


PROJECT_DIR = "./kpms_few_bp_150ms_frac_1_ld3_jn24"  # Using a new directory for the full analysis
MODEL_NAME = 'final_model_1.21e+04_idx0'


# In[26]:


# --- IMPORTS & SETUP ---

# Enable autoreload (optional but good practice)
# %load_ext autoreload
# %autoreload 2

# Standard library & path setup
import pickle
import pprint
import sys
from pathlib import Path

# Modeling pipeline
# Visualization
import matplotlib.pyplot as plt
import numpy as np

# Data handling and analysis
import pandas as pd

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

    # OFT-specific session population
    from fiberphotometry.data.oft_session_loading import populate_oft_containers

    # container for holding session data
    from fiberphotometry.data.session_loading import DataContainer

    # synchronizing different data streams in a session
    from fiberphotometry.data.session_synchronization import synchronize_session_data
    from fiberphotometry.kpms.batching import create_batches
    from fiberphotometry.kpms.syllable_appender import extract_and_append_results

    # KPMS pipeline stages
    from fiberphotometry.kpms.syllable_generator import generate_syllable_models
    from fiberphotometry.kpms.trimming import create_trimmed_coords_and_confs

    print("✅ All custom modules imported successfully.")
except ImportError as e:
    print(f"❌ Failed to import a custom module: {e}")
    print("   Please ensure your notebook is in the correct project directory.")


# In[ ]:


import pickle

import numpy as np
import pandas as pd

#load_filename = 'final_model_1.21e+04_idx0.pkl'
load_filename = 'kpms_few_bp_150ms_frac_1_ld3_jn24.pkl'
print(f"--- 🚀 Loading complete analysis state from '{load_filename}'... ---")

with open(load_filename, 'rb') as f:
    loaded_state = pickle.load(f)

# unpack
sessions = loaded_state['sessions']
PROJECT_DIR_FULL = loaded_state.get('project_dir_full')
KAPPA = loaded_state.get('kappa')
video_frame_indexes = loaded_state['video_frame_indexes']
original_lengths = loaded_state['original_lengths']
final_model_names = loaded_state.get('final_model_names', loaded_state.get('model_names', []))

print(f"--- ✅ Base state loaded. Sessions: {len(sessions)}. Candidate models: {len(final_model_names)} ---")

# ---- materialize a single-model view into session.all_results ----
def _materialize_single_model_view(sessions, model_name):
    updated, missing = 0, []
    for s in sessions:
        store = getattr(s, 'kpms_results_by_model', None)
        if not store or model_name not in store:
            missing.append(getattr(s, 'trial_id', '?'))
            # ensure no stale legacy attribute lingers
            if hasattr(s, 'all_results'):
                delattr(s, 'all_results')
            continue

        res = store[model_name]

        # tolerate 'syllables' vs 'syllable' naming by creating a local alias if needed
        if 'syllable' not in res and 'syllables' in res:
            # make a shallow copy so we don't mutate the stored dict
            res = {**res, 'syllable': res['syllables']}

        # ensure no stale attr, then set the single-model projection
        if hasattr(s, 'all_results'):
            delattr(s, 'all_results')
        setattr(s, 'all_results', res)
        updated += 1
    return updated, missing

if 'MODEL_NAME' not in globals():
    raise NameError("Please set MODEL_NAME before loading (e.g., MODEL_NAME = final_model_names[0]).")

updated, missing = _materialize_single_model_view(sessions, MODEL_NAME)

print(f"✅ Materialized single-model view: {updated} sessions now have 'all_results' for '{MODEL_NAME}'.")
if missing:
    print(f"⚠️ {len(missing)} sessions lacked results for '{MODEL_NAME}': {missing[:6]}{' ...' if len(missing)>6 else ''}")

# quick sanity: do we actually have syllable vectors accessible like before?
any_ok = any(
    hasattr(s, 'all_results') and isinstance(s.all_results, dict) and
    (('syllable' in s.all_results) or ('syllables' in s.all_results))
    for s in sessions
)
print("🔎 Post-load check:", "OK ✅" if any_ok else "Issue ❌ — no sessions expose syllables via all_results")


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
        print("✅ Successfully added 'syllable' column.")
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

# In[5]:


# # define your ten model names
# model_names = [f'final_model_1.20e+06_idx{i}' for i in range(5)]

# # compute EML scores and their standard errors
# eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(PROJECT_DIR, model_names)

# # pick the best model
# best_model_name = model_names[np.argmax(eml_scores)]
# print(f"Best model: {best_model_name}")

# # visualize the EML scores across the ensemble
# kpms.plot_eml_scores(eml_scores, eml_std_errs, model_names)


# In[6]:


model, data, metadata, current_iter = kpms.load_checkpoint(PROJECT_DIR, MODEL_NAME)
results = kpms.extract_results(model, metadata, PROJECT_DIR, MODEL_NAME, save_results=False)


# In[7]:


# model_names = [f'final_model_1.20e+06_idx{i}' for i in range(5)]

# for model_name in model_names:
#     # load the specific checkpoint
#     model, data, metadata, current_iter = kpms.load_checkpoint(PROJECT_DIR, model_name)

#     # extract results for this model
#     results = kpms.extract_results(model, metadata, PROJECT_DIR, model_name, save_results=True)


# In[8]:


config = kpms.load_config(PROJECT_DIR)


# In[9]:


import os

import jax.numpy as jnp
import numpy as np
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch

# compute the estimated coordinates
Y_est = estimate_coordinates(
    jnp.array(model['states']['x']),
    jnp.array(model['states']['v']),
    jnp.array(model['states']['h']),
    jnp.array(model['params']['Cd'])
)

# generate a dictionary with reconstructed coordinates for each recording
coordinates_est = unbatch(Y_est, *metadata)


# In[10]:


coordinates_est['Arena1234_CenterTopCamVid_2024-12-13T12_20_06_top_leftDLC_HrnetW32_dataset_fixDec16shuffle10_detector_020_snapshot_020'].shape


# In[11]:


for k, v in model['states'].items():
    print(k, v.shape)


# In[12]:


for k, v in model['params'].items():
    print(k, v.shape)


# In[13]:


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


# In[14]:


for k, v in coordinates_est.items():
    print(k, v.shape)


# ### Trajectory plots

# In[15]:


#perm = kpms.reindex_syllables_in_checkpoint(PROJECT_DIR, MODEL_NAME)

config['bodyparts'] = config['use_bodyparts']
kpms.generate_trajectory_plots(coordinates_est, results, PROJECT_DIR, MODEL_NAME, fps=60, min_frequency=0.000, min_duration=1, **config)


# ### Grid movie preprocessing and generation

# In[16]:


import os
import pprint  # Make sure pprint is imported for nice printing


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


# In[17]:


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


# In[18]:


centroids, headings = kpms.get_centroids_headings(coordinates_est, **config)


# In[19]:


video_frame_indexes


# In[20]:


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

# In[23]:



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


# In[24]:


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


# In[25]:


moseq_df = kpms.compute_moseq_df(PROJECT_DIR, MODEL_NAME, smooth_heading=True)


# In[26]:


moseq_df


# In[27]:


stats_df = kpms.compute_stats_df(
    PROJECT_DIR,
    MODEL_NAME,
    moseq_df,
    min_frequency=0.005,  # threshold frequency for including a syllable in the dataframe
    groupby=["group", "name"],  # column(s) to group the dataframe by
    fps=60,
)  # frame rate of the video from which keypoints were inferred

stats_df


# In[28]:


MODEL_NAME


# In[30]:


PROJECT_DIR


# In[29]:


kpms.label_syllables(PROJECT_DIR, MODEL_NAME, moseq_df)


# In[31]:


sessions[0].kpms_h5_path


# In[ ]:


kpms.interactive_group_setting


# ### Plotting setup and more checks

# In[32]:


from fiberphotometry.config import PLOTTING_CONFIG
from fiberphotometry.processing.plotting_setup import PlottingSetup

# In[33]:


PlottingSetup(**PLOTTING_CONFIG['cpt']).apply_plotting_setup_to_sessions(sessions)


# In[34]:


plt.plot(sessions[0].all_results['syllable'])


# In[35]:


sessions[0].dfs.data.keys()


# In[36]:


sessions[0].dfs.data['synchronized_data_tracking_0']


# In[37]:


sessions[0].dfs.data['phot_470']


# ### Syllable based signatures

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import sem

# --- Analysis & Smoothing Parameters ---
SAMPLING_RATE = 80
WINDOW_SECONDS = 10
SMOOTHING_WINDOW = 25
SMOOTHING_POLY_ORDER = 3
CONFIDENCE_LEVEL = 1.96
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLING_RATE)
TOTAL_SAMPLES = 2 * WINDOW_SAMPLES
SIGNAL_COLS = ['signal_0_470_dff_z', 'signal_1_470_dff_z']

# --- Syllable Label Setup (no reindexing) ---
LABEL_CSV_PATH = '/home/fsp585/Desktop/FiberphotometryCode/tests/kpms_all_sessions_frac_1_ld9_jn24/final_model_1.20e+06_idx1/syll_info.csv'
labels_df = pd.read_csv(LABEL_CSV_PATH)
labels_df['label'] = labels_df['label'].fillna('').str.strip()

# Map original syllable ID to label; fallback to "non-common" if missing/empty
syllable_id_to_label = {
    int(row.syllable): row.label if row.label else "non-common"
    for _, row in labels_df.iterrows()
}

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
processed_list = [process_session_data(s) for s in sessions]

# 2. Gather all syllables present across sessions
all_syllables = sorted({
    syll for proc in processed_list for syll in proc.keys()
})
print("Syllables to plot:", all_syllables)

# 3. Determine unique brain regions and hemispheres from sessions' metadata
regions = sorted({region for s in sessions for region, _, _ in s.brain_regions})
hemispheres = sorted({
    s.signal_meta["_".join(col.split("_")[:2])][1]
    for s in sessions
    for col in SIGNAL_COLS
    if "_".join(col.split("_")[:2]) in s.signal_meta
})
time_vector = np.linspace(-WINDOW_SECONDS, WINDOW_SECONDS, TOTAL_SAMPLES)

# 4. Plot each syllable with hemisphere rows (top=left, bottom=right) and region columns
for syl_idx, syllable in enumerate(all_syllables):
    final_label = syllable_id_to_label.get(int(syllable), f"Unknown ID: {syllable}")

    hemis_order = ['left', 'right']  # top row = left, bottom = right
    n_rows = 2
    n_cols = len(regions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols + 2, 8))  # width scales with regions
    fig.suptitle(f"Syllable Onset: '{final_label}' (Original ID: {int(syllable)})", fontsize=18)

    time_vector = np.linspace(-WINDOW_SECONDS, WINDOW_SECONDS, TOTAL_SAMPLES)

    for row_idx, hemisphere in enumerate(hemis_order):
        for col_idx, region in enumerate(regions):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            traces_by_genotype = {'KI': [], 'WT': []}

            for s, proc_data in zip(sessions, processed_list):
                meta = s.signal_meta
                for col in SIGNAL_COLS:
                    prefix = "_".join(col.split("_")[:2])
                    if prefix not in meta:
                        continue
                    region_meta = meta[prefix][0]
                    hemi_meta = meta[prefix][1]
                    if region_meta == region and hemi_meta == hemisphere:
                        traces = proc_data.get(syllable, {}).get(col, [])
                        traces_by_genotype[s.genotype].extend(traces)

            for geno, traces in traces_by_genotype.items():
                if not traces:
                    continue
                valid_traces = [t for t in traces if not np.isnan(t).any()]
                n_skipped = len(traces) - len(valid_traces)
                if n_skipped > 0:
                    print(f"Skipped {n_skipped} {geno} traces for syllable {syllable}, region {region}, hemisphere {hemisphere} due to NaNs")
                if not valid_traces:
                    continue

                arr = np.stack(valid_traces)
                m = arr.mean(axis=0)
                e = sem(arr, axis=0)
                lower = m - CONFIDENCE_LEVEL * e
                upper = m + CONFIDENCE_LEVEL * e

                color = 'darkblue' if geno == 'KI' else 'green'
                ax.plot(time_vector, m, label=f"{geno} (n={len(traces)})", color=color)
                ax.fill_between(time_vector, lower, upper, color=color, alpha=0.2)

            ax.set_title(f"{region} | {hemisphere}", fontsize=13)
            ax.axvline(0, color='crimson', linestyle='--', lw=2)
            ax.axhline(0, color='grey', linestyle=':', lw=1)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time from Onset (s)")
            if col_idx == 0:
                ax.set_ylabel("Normalized Signal (z-score)")
            ax.grid(True, linestyle=':')
            ax.legend(fontsize='small')

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
)


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

