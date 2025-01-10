# config.py

# Plotting configuration parameters
PLOTTING_CONFIG = {
    'cpt': {
        'baseline_duration': 20,    # in minutes before trial start
        'trial_length': 30,         # in minutes after trial start
        'fps': 20,                  # frames per second
        'fit_window_start': 11,     # in minutes before trial start
        'fit_window_end': 1         # in minutes before trial start
    },
    'oft': {
        'baseline_duration': 20,             # in minutes before injection starts
        'trial_length': 20,       # in minutes after injection ends
        'fps': 20,                           # frames per second
        'fit_window_start': 11,              # in minutes before injection start
        'fit_window_end': 1                  # in minutes before injection start
    }
}


RENAME_PATTERNS = [
    ('bonsai', {"pattern": "(CombiTimestamps)?_\\d+_?[AB].?", "replacement": ""}),
    ('bonsai', {"pattern": "FP3002_Timestamp?(_\\d+)?", "replacement": "Timestamp_FP3002"}),
    ('bonsai', {"pattern": "TimestampBonsai?(_\\d+)?", "replacement": "Timestamp_Bonsai"}),
    ('bonsai', {"pattern": r"FrameCount_\d\d\d", "replacement": "FrameCount"}),
    ('ttl', {"pattern": "_Timestamp_AND_Seconds_", "replacement": "_"}),
    ('ttl', {"pattern": r'_DigInput[012]', "replacement": ""}),
    ('ttl', {"pattern": 'FP3002.(Seconds|Value)', "replacement": r"\1_FP3002"}),
    ('ttl', {"pattern": 'BonsaiTimestamp', "replacement": "Timestamp_Bonsai"})
]
# TimestampBonsai_415
RENAME_FREQS = ['415', '470', '560']
#LETTER_TO_FREQS = {'iso': '415', 'G': '470', 'R': '560'}
LETTER_TO_FREQS = {'iso': '415', 'G': '470'}


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

item_attr_dict = {"Hit": "hit",
                    "Mistake": "mistake",
                    "Missed Hit": "miss",                    
                    "Correct Rejection": "cor_reject"}

actions_attr_dict = {"Hit": "hit",
                    "Mistake": "mistake", 
                    "Missed Hit": "miss",                    
                    "Correction Trial Correct Rejection": "cor_reject", 
                    "Correct Rejection": "cor_reject"}

reward_attr_dict = {"Reward Collected Start ITI": "reward_collect"}


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