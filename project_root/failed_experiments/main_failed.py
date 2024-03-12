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

import pickle
from data.data_loading import load_all_sessions
from data.setup_utils import Renamer, Syncer
from data.timepoints import create_timepoint_container_for_sessions
from processing.plotting_setup import PlottingSetup
from analysis.timepoint_analysis import add_aucs, get_signal_around_timepoint
from analysis.performance_funcs import d_prime
from config import *
import matplotlib.pyplot as plt


def load_and_prepare_sessions(baseline_dir, first_n_dirs=None, load_from_pickle=False, 
                              remove_bad_signal_sessions=False):
    if load_from_pickle:
        with open(baseline_dir + '/sessions.pickle', 'rb') as f:
            sessions = pickle.load(f)
        return sessions

    # load in sessions
    sessions = load_all_sessions(baseline_dir, first_n_dirs, remove_bad_signal_sessions=remove_bad_signal_sessions)

    # rename columns of dfs
    Renamer.rename_sessions_data(sessions, RENAME_PATTERNS)
    Renamer.rename_sessions_fiber_to_brain_region(sessions, RENAME_FREQS)

    # add sync columns to dfs
    Syncer.apply_sync_to_all_sessions(sessions)

    # apply attributes used for plotting
    plotting_setup = PlottingSetup(**PLOTTING_CONFIG)
    plotting_setup.apply_plotting_setup_to_sessions(sessions)

    # add timepoints
    create_timepoint_container_for_sessions(sessions, actions_attr_dict, reward_attr_dict)

    return sessions

import matplotlib.pyplot as plt
import numpy as np


sessions = load_and_prepare_sessions("../Baseline", load_from_pickle=False, remove_bad_signal_sessions=True)
# -50 to +100 frames

# Set the flag for normalization
normalize_between_minima = False

auc_results = {i: dict() for i in list(range(len(sessions)))}
for i, session in enumerate(sessions):
    for action_type in ['hit', 'mistake']:
        if i in (24, 29):  # Skip bad signals
            continue

        # Define minima_indices for specific sessions
        elif i == 9:    
            minima_indices = np.array([200, 350])
        else:
            minima_indices = np.array([150, 300])

        # Filter out sessions based on brain region and available data
        current_brain_region = session.brain_regions[-1]
        if (current_brain_region[0] != 'V') or (len(session.timepoints_container.get_data(action_type)) == 0):
            continue

        # Get the signal data
        xs, ys = get_signal_around_timepoint(session, action_type, current_brain_region)
        
        # Normalize the signal if the flag is set
        if normalize_between_minima:
            # Find the lowest point between the minima
            min_val_between_minima = np.min(ys[minima_indices[0]:minima_indices[1]+1])
            # Subtract the lowest value from all values in the range to normalize
            ys = ys - min_val_between_minima

        # Calculate AUC using the trapezoidal rule for the signal between the minima
        auc = np.trapz(ys[minima_indices[0]:minima_indices[1]+1], xs[minima_indices[0]:minima_indices[1]+1])

        auc_results[i][action_type] = auc

auc_results = {session_idx: info for session_idx, info in auc_results.items() if len(info.keys()) == 2}

from scipy.stats import linregress

# Your data
xs = []
ys = []

for session_idx, info in auc_results.items():
    curr_d_prime = d_prime(sessions[session_idx])
    y_res = info['hit'] / info['mistake']
    xs.append(curr_d_prime)
    ys.append(y_res)

slope, intercept, r_value, p_value, std_err = linregress(xs, ys)

r_squared = r_value**2
regression_line = [slope * x + intercept for x in xs]

plt.scatter(xs, ys)
plt.xlabel('d_prime')
plt.ylabel('hit_auc - mistake_auc')
plt.plot(xs, regression_line, color='red', label=f'Linear Regression (R² = {r_squared:.2f})')
plt.legend()
plt.show()


# session_results = {}

# for idx, session in enumerate(sessions):
#     info = {}
#     for attr, auc_results in session.auc_results.items():
#         if auc_results:
#             _, _, lst = zip(*auc_results)
#             info[attr] = sum(lst)/len(lst) 
#     info['d_prime'] = d_prime(session)
#     session_results[idx] = info 

          