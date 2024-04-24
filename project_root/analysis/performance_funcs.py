from config import * 
from scipy.stats import norm
from data.data_loading import DataContainer
from utils import count_session_events
import numpy as np


def set_event_counts(mouse):
        event_counts = {}

        for event_type in actions_attr_dict.values():
            total = count_session_events(mouse.sessions, event_type)
            event_counts[event_type] = total

        mouse.event_counts = event_counts


def calculate_rate(event_counts, numerator_event, denominator_event):
    #Helper method to calculate rates, handling division by zero.#
    numerator = event_counts[numerator_event]
    denominator = numerator + event_counts[denominator_event]
    return numerator / denominator if denominator else 0


# attach hit_rate and such to the mouse
def get_hr_and_far(event_counts):
    # TODO: this method is to be updated such that it returns more than just rates
    # Calculates performance metrics such as hit rate and false alarm rate
    hit_rate = calculate_rate(event_counts, 'hit', 'miss')
    false_alarm_rate = calculate_rate(event_counts, 'mistake', 'cor_reject')

    if hit_rate in (0, 1) or false_alarm_rate in (0, 1):
            return None, None

    # Calculate z-scores
    z_hit = norm.ppf(hit_rate)
    z_false_alarm = norm.ppf(false_alarm_rate)
    return z_hit, z_false_alarm

# attach hit_rate and such to the mouse
def d_prime(event_counts):
    z_hit, z_false_alarm = get_hr_and_far(event_counts)
    if z_hit and z_false_alarm:
        return z_hit - z_false_alarm
    else:
        return None

# attach hit_rate and such to the mouse
def c_score(event_counts):
    z_hit, z_false_alarm = get_hr_and_far(event_counts)
    if z_hit and z_false_alarm:
        return - (z_hit + z_false_alarm) / 2
    else:
        return None


def participation(event_counts):
    return event_counts['hit'] + event_counts['mistake']


def total_hits(event_counts):
    return event_counts['hit']

def total_mistakes(event_counts):
    return event_counts['mistake']


def hit_rate(event_counts):
    return calculate_rate(event_counts, 'hit', 'miss')


def false_alarm_rate(event_counts):
    return calculate_rate(event_counts, 'mistake', 'cor_reject')

def get_avg_time_from_disp_to_hit_to_reward(sessions):
    dispimg_idxs = []
    hit_idxs = []
    reward_idxs = []
    
    for session in sessions:
        event_idxs_data = session.event_idxs_container.data
        phot_df = session.df_container.get_data("photwrit_470")
        phot_times = phot_df['SecFromZero_FP3002'].values
        
        curr_dispimg_idxs = event_idxs_data['before_dispimg_hit']
        curr_hit_idxs = event_idxs_data['hit']
        curr_reward_idxs = event_idxs_data['reward_collect']

        min_len = min(len(curr_dispimg_idxs), len(curr_hit_idxs), len(curr_reward_idxs))

        curr_dispimg_idxs = curr_dispimg_idxs[:min_len]
        curr_hit_idxs = curr_hit_idxs[:min_len]
        curr_reward_idxs = curr_reward_idxs[:min_len]

        dispimg_idxs.extend(curr_dispimg_idxs)
        hit_idxs.extend(curr_hit_idxs)
        reward_idxs.extend(curr_reward_idxs)

    dispimg_idxs = np.array(dispimg_idxs)
    hit_idxs = np.array(hit_idxs)
    reward_idxs = np.array(reward_idxs)

    # Perform element-wise comparison
    if not ((dispimg_idxs < hit_idxs) & (hit_idxs < reward_idxs)).all():
        raise ValueError("Indices are not in order")

    return (np.mean(phot_times[hit_idxs] - phot_times[dispimg_idxs]),
            np.mean(phot_times[reward_idxs] - phot_times[hit_idxs])) 

def add_performance_container(mouse):
    metric_container = DataContainer(float)
    set_event_counts(mouse)
    
    # Define a dictionary with metric names and their corresponding functions
    metrics = {
        'd_prime': d_prime,
        'c_score': c_score,
        'participation': participation,
        'total_hits': total_hits,
        'total_mistakes': total_mistakes,
        'hit_rate': hit_rate,
        'false_alarm_rate': false_alarm_rate,
    }
    
    # Loop through the dictionary, calculating and adding each metric
    for metric_name, metric_function in metrics.items():
        metric_val = metric_function(mouse.event_counts)
        if metric_val is None:
             metric_val = -1.0
        elif type(metric_val) == int:
             metric_val = float(metric_val)
        metric_container.add_data(metric_name, metric_val)

    disp_to_hit_time, hit_to_reward_time = get_avg_time_from_disp_to_hit_to_reward(mouse.sessions)
    metric_container.add_data('disp_to_hit_time', disp_to_hit_time) 
    metric_container.add_data('hit_to_reward_time', hit_to_reward_time) 
    num_center_touches = sum(
        session.df_container.get_data("raw")['Item_Name'].str.contains("Centre_Touches_during_ITI").sum()
        for session in mouse.sessions)
    
    metric_container.add_data('num_center_touches', float(num_center_touches))
    mouse.metric_container = metric_container