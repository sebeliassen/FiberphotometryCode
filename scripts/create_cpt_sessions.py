"""
This is the main.py file where all of the boiler-plate loading and preprocessing is done.
A lot of the code for forming these operations are put into the data_loading.py for loading 
in all of the sessions from the files.

The preprocessing is mainly done by the setup_utils.py, which contains two main classes called
the Renamer and the Syncer. 

Renamer.rename_sessions_data() uses the regex patterns, to regularize
the names for the different dataframes. Renamer.rename_sessions_fiber_to_brain_region() fetches
the columns contained in the fiberphotometry dfs (photwrit and bonsai as of now), and renames the
columns to the brainregions that the fibers point to.

Syncer.apply_sync_to_all_sessions() applies the same synchronization logic as in the original CPT code. 
This will of course need to change if we don't have any raw_df for instance.
"""

import os
import sys
# Ensure the repository root (one level above scripts/) is on sys.path so fiberphotometry can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pickle
import shutil
from fiberphotometry.data.data_loading import load_all_sessions
from fiberphotometry.data.renamer import Renamer
from fiberphotometry.data.syncer  import Syncer
from fiberphotometry.data.timepoints import create_event_idxs_container_for_sessions
from fiberphotometry.processing.plotting_setup import PlottingSetup
from fiberphotometry.processing.signal_info_setup import assign_sessions_signal_info
from fiberphotometry.config import *
from tqdm import tqdm

def _download_to_tmp(baseline_dir):
    print("Attempting to download or copy from remote to /tmp...")

    # Construct a local tmp folder name
    tmp_basename = os.path.basename(os.path.normpath(baseline_dir))
    local_tmp_dir = os.path.join("/tmp", tmp_basename)
    os.makedirs(local_tmp_dir, exist_ok=True)

    # Get a list of immediate subdirectories
    sub_dirs = [
        entry.path
        for entry in os.scandir(baseline_dir)
        if entry.is_dir()
    ]

    with tqdm(total=len(sub_dirs), desc="Downloading folders", unit="folder") as pbar:
        for sub_dir in sub_dirs:
            dest_path = os.path.join(local_tmp_dir, os.path.basename(sub_dir))
            shutil.copytree(sub_dir, dest_path, dirs_exist_ok=True)
            pbar.update(1)

    # Now point baseline_dir to the local path for further processing
    new_baseline_dir = local_tmp_dir
    print(f"Using local path {new_baseline_dir} for processing.")
    return new_baseline_dir

def load_and_prepare_sessions(
    baseline_dir,
    session_type,
    first_n_dirs=None,
    load_from_pickle=False,
    remove_bad_signal_sessions=False,
    pickle_name=None
):
    if pickle_name is None:
        pickle_name = '/sessions.pickle'
    else:
        pickle_name = f'/{pickle_name}.pickle'

    pickle_filepath = baseline_dir + pickle_name

    # if load_from_pickle:
    #     try:
    #         with open(pickle_filepath, 'rb') as f:
    #             sessions = pickle.load(f)
    #             return sessions
    #     except (FileNotFoundError, IOError):
    #         print(f"Pickle file not found at {pickle_filepath}.")
    #         baseline_dir = _download_to_tmp(baseline_dir)
    # elif not load_from_pickle:
    #     print("load_from_pickle is False.")
    #     baseline_dir = _download_to_tmp(baseline_dir)

    # 1) load raw sessions
    sessions = load_all_sessions(
        baseline_dir,
        session_type,
        first_n_dirs=first_n_dirs,
        remove_bad_signal_sessions=remove_bad_signal_sessions
    )

    # 2) strip & pick your TTL column in one go
    for sess in sessions:
        Renamer.finalize_ttl_for_session(sess)

    # 3) rename your photometry to brain‐region names
    Renamer.rename_sessions_fiber_to_brain_region(sessions, LETTER_TO_FREQS)

    # 4) now sync on that single TTL_ts column
    for session in sessions:
        Syncer.sync_session(session)

    plotting_setup = PlottingSetup(**PLOTTING_CONFIG['cpt'])
    plotting_setup.apply_plotting_setup_to_sessions(sessions)

    create_event_idxs_container_for_sessions(sessions, actions_attr_dict, reward_attr_dict)
    assign_sessions_signal_info(sessions)

    return sessions

def create_pickle(src, dst=None, overwrite=False, first_n_dirs=None):
    """
    Creates a pickle file of the processed sessions.

    Parameters:
    - src (str): Source directory for loading and processing sessions.
    - dst (str or None): Destination path for the pickle file. If None, saves in the source directory.
    - overwrite (bool): Whether to overwrite the pickle file if it already exists.

    Returns:
    - sessions (list): The processed sessions.
    """
    # Process the sessions
    sessions = load_and_prepare_sessions(src, 'cpt', load_from_pickle=False, remove_bad_signal_sessions=True, first_n_dirs=first_n_dirs)
    print(sessions)

    # Determine destination path
    if dst is None:
        dst = os.path.join(src, "sessions.pickle")

    # Check if file exists and handle overwrite flag
    if os.path.exists(dst) and not overwrite:
        print(f"Pickle file already exists at {dst}. Use overwrite=True to replace it.")
        return sessions

    # Save sessions to pickle
    with open(dst, "wb") as f:
        pickle.dump(sessions, f)
        print(f"Pickle file saved at {dst}.")

    return sessions
