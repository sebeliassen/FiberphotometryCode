# config.py

# Plotting configuration parameters
PLOTTING_CONFIG = {
    'baseline_duration': 20,  # in minutes
    'trial_length': 30,       # in minutes
    'fps': 20,                # frames per second
    'fit_window_start': 11,   # in minutes before trial start
    'fit_window_end': 1       # in minutes before trial start
}

RENAME_PATTERNS = [
    ('bonsai', {"pattern": "(CombiTimestamps)?_\\d+_?[AB].?", "replacement": ""}),
    ('bonsai', {"pattern": "FP3002_Timestamp", "replacement": "Timestamp_FP3002"}),
    ('bonsai', {"pattern": "TimestampBonsai", "replacement": "Timestamp_Bonsai"}),
    ('ttl', {"pattern": r'_DigInput[01]', "replacement": ""}),
    ('ttl', {"pattern": 'FP3002.(Seconds|Value)', "replacement": r"\1_FP3002"}),
    ('ttl', {"pattern": 'BonsaiTimestamp', "replacement": "Timestamp_Bonsai"})
]

RENAME_FREQS = ['415', '470', '560']

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
                      'before_dispimg_hit': (-2.5, 2.5)}

all_brain_regions = ['VS', 'DMS', 'DLS']
all_event_types = ['hit', 'mistake', 'miss', 'cor_reject', 'reward_collect']
all_metrics = ['c_score', 'd_prime', 'participation']