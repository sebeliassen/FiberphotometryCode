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
from fiberphotometry.data.data_loading import load_all_sessions
from fiberphotometry.data.renamer import Renamer
from fiberphotometry.data.syncer  import Syncer
from fiberphotometry.data.timepoints import create_event_idxs_container_for_sessions
from fiberphotometry.processing.plotting_setup import PlottingSetup
from fiberphotometry.processing.signal_info_setup import assign_sessions_signal_info
from fiberphotometry.config import *


def load_and_prepare_sessions(baseline_dir, first_n_dirs=None, load_from_pickle=False, 
                              remove_bad_signal_sessions=False, pickle_name=None):
    if pickle_name is None:
        pickle_name = '/sessions.pickle'
    else:
        pickle_name = f'/{pickle_name}.pickle'
    if load_from_pickle:
        with open(baseline_dir + pickle_name, 'rb') as f:
            sessions = pickle.load(f)
        return sessions

    # load in sessions
    sessions = load_all_sessions(baseline_dir, first_n_dirs, remove_bad_signal_sessions=remove_bad_signal_sessions)

    # rename columns of dfs
    Renamer.rename_sessions_data(sessions, TTL_PATTERNS)
    Renamer.rename_sessions_fiber_to_brain_region(sessions, LETTER_TO_FREQS)

    # add sync columns to dfs
    Syncer.apply_sync_to_all_sessions(sessions)

    # apply attributes used for plotting
    plotting_setup = PlottingSetup(**PLOTTING_CONFIG)
    plotting_setup.apply_plotting_setup_to_sessions(sessions)

    # add event_idxs
    create_event_idxs_container_for_sessions(sessions, actions_attr_dict, reward_attr_dict)
    assign_sessions_signal_info(sessions)
    return sessions