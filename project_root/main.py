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
from processing.timepoint_analysis import plot_session_events_and_signal, aggregate_signals
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


# sessions = load_and_prepare_sessions("../Baseline", load_from_pickle=True, remove_bad_signal_sessions=True)
# for session in sessions:
#     session.d_prime = d_prime(session) 
#     session.c_score = c_score(session) 
#     session.participation = participation(session) 

# sorted_d_prime_sessions = sorted(sessions, key=lambda sesh: sesh.d_prime)
# sorted_d_prime_sessions = [sesh for sesh in sorted_d_prime_sessions if sesh.brain_regions[-1][0] == 'V' \
#                            and sesh.d_prime != -1]

# sorted_c_score_sessions = sorted(sessions, key=lambda sesh: sesh.d_prime)
# sorted_c_score_sessions = [sesh for sesh in sorted_c_score_sessions if sesh.brain_regions[-1][0] == 'V' \
#                            and sesh.c_score != -1]

# sorted_participation_sessions = sorted(sessions, key=lambda sesh: sesh.d_prime)
# sorted_participation_sessions = [sesh for sesh in sorted_participation_sessions if sesh.brain_regions[-1][0] == 'V' \
#                                  and sesh.participation != -1]


# fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), ncols=2)
# aggregate_signals(sorted_d_prime_sessions[:5], 'hit', ['VS_left', 'VS_right'], ax1, brain_reg_name='bad')
# aggregate_signals(sorted_d_prime_sessions[-5:], 'hit', ['VS_left', 'VS_right'], ax2, brain_reg_name='good')
# fig.suptitle(f'hit_signals_r{5}')
# plt.show()

