# Patterns for loading session data files by session type.
DATA_PATTERNS = {
    'cpt': {
        'raw':  { 'parser': 'csv', 'pattern': 'RAW_{chamber_id}*.csv' },
        'ttl':  { 'parser': 'csv', 'pattern': 'DigInput_{chamber_id}*.csv' },
        'phot': { 'parser': 'csv', 'pattern': 'photometry_data_combined_Setup{setup_id}*.csv' },
    },
}

# Plotting configuration parameters
PLOTTING_CONFIG = {
    # cno 5 minutes, stages 10 minutes
    'cpt': {
        'baseline_duration':20,    # in minutes before trial start
        'trial_length': 30,         # in minutes after trial start
        'fps': 20,                  # frames per second
        'fit_window_start': 11,     # in minutes before trial start
        'fit_window_end': 1,         # in minutes before trial start
    },
    'oft': {
        'baseline_duration': 20,             # in minutes before injection starts
        'trial_length': 20,       # in minutes after injection ends
        'fps': 20,                           # frames per second
        'fit_window_start': 16,              # in minutes before injection start
        'fit_window_end': 1                  # in minutes before injection start
    }
}

SESSION_CONFIG = {
'skip_sequence_check': True
}

# Frequency settings
COMBINED_SPLIT = {
    'cpt': {
        1: '415',   # Bonsai LedState==1 → 415 Hz channel
        2: '470',   # Bonsai LedState==2 → 470 Hz channel
    }
}
LETTER_TO_FREQS = {'iso': '415', 'G': '470'}
FREQS_USED = ['415', '470']

PHOT_DF_PATTERNS = {
    'phot_415': 'channel415*.csv',
    'phot_470': 'channel470*.csv',
}

TIMESTAMP_DF_PATTERNS = {
    'cam': 'BaslerTrack*.csv',
    'bonsai_415': 'c415_bonsaiTS*.csv',
    'bonsai_470': 'c470_bonsaiTS*.csv'
}

peak_interval_config = {
    'interval_start': 10 * 20,
    'interval_end': 10 * 20,
}

# item_attr_dict = {"Hit": "hit",
#                     "Mistake": "mistake",
#                     "Missed Hit": "miss",                    
#                     "Correct Rejection": "cor_reject"}

# actions_attr_dict = {"Hit": "hit",
#                     "Mistake": "mistake", 
#                     "Missed Hit": "miss",                    
#                     "Correction Trial Correct Rejection": "cor_reject", 
#                     "Correct Rejection": "cor_reject"}

# reward_attr_dict = {"Reward Collected Start ITI": "reward_collect"}

item_attr_dict = {}

# actions_attr_dict = {
#     'Correct_Counter':      'global_correct_hit',      # maps Correct_Counter_<n> → global_correct_hit
#     'FIXED_RATIO_COUNTER':  'fixed_ratio_touch'        # maps FIXED_RATIO_COUNTER_<m> → fixed_ratio_touch
# }

actions_attr_dict = {
}

reward_attr_dict = {}

attr_interval_dict = {'hit': (-2.5, 5),
                      'miss': (-2.5, 2),
                      'cor_reject': (-2.5, 2),
                      'mistake': (-2.5, 2.5),
                      'reward_collect': (-3, 4.5),
                      'before_dispimg_mistake': (-2.5, 2.5),
                      'before_dispimg_hit': (-2.5, 2.5),
                      'before_dispimg_cor_reject': (-2.5, 2.5),
                      'before_dispimg_miss': (-2.5, 2.5),
                      'iti_touch': (-2.5, 5),
                      'dispimg': (-2.5, 2.5)}

# all_brain_regions = ['VS', 'DMS', 'DLS']
all_brain_regions = ['LH', 'mPFC']
all_event_types = ['hit', 'mistake', 'miss', 'cor_reject', 'reward_collect', 
                   'before_dispimg_mistake', 'before_dispimg_hit', 'before_dispimg_cor_reject', 'before_dispimg_miss', 'dispimg']
all_metrics = ['c_score', 'd_prime', 'participation', 'disp_to_hit_time', 'hit_to_reward_time', 'num_center_touches']

response_metric_idxs = {0: 'slope_up',
                        1: 'slope_down',
                        2: 'maximal_value',
                        3: 'peak_timing',
                        4: 'auc'}
  
# Patterns for fiber column matching and column filtering moved from code into config
# TODO: we need to find a standardization of the pattern below
FIBER_PATTERN = r'([GR])(\d+)|fiber(\d+)'
FILTER_COLUMNS_PATTERN = r'[GR](\d+)|Region(\d+)[\w]*'
  # Combined photometry CSV patterns by session type (single-file per-session dumps)
COMBINED_PATTERNS = {
    'cpt': 'photometry_data_combined*.csv'
}


SYNC = {
    # --- Column Name Identification ---
    # Define which columns contain the necessary timestamps in different dataframes.
    # The code will try these in order and use the first one found.

    # Column in the 'raw' DataFrame (from ABET II) holding the event time used for sync.
    'raw_time_col': 'Evnt_Time',

    # Columns to check for the primary timestamp in the TTL DataFrame (DigInput*.csv).
    # The Renamer's finalize_ttl_for_session should ideally produce 'TTL_ts'
    # from the highest priority source found (e.g., FP3002.Seconds, SystemTimestamp).
    # List potential raw names as fallbacks if renamer didn't run or target not found.
    'ttl_time_col': 'SystemTimestamp',

    # Columns to check for the primary timestamp in Photometry DataFrames (channel* / photdata*).
    'phot_time_col': 'SystemTimestamp',

    'sec_zero_col': 'sec_from_zero',
    'sec_trial_col': 'sec_from_trial_start',

    # --- Processing Parameters ---

    # Truncate all processed photometry streams to the length of the shortest one?
    'truncate_streams': True,

    # --- Required Reference for Sync Logic ---
    # Which photometry frequency stream's start time is used as the reference point
    # for calculating the offset? Typically the main signal channel (e.g., '470').
    'reference_phot_freq': '470',
}
