from config import * 
from scipy.stats import norm
from data.data_loading import DataContainer


def set_event_counts(mouse):
        event_counts = {}

        for event_type in actions_attr_dict.values():
            total = sum(len(mouse_session.timepoints_container.data.get(event_type, []))
                        for mouse_session in mouse.sessions)
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
    mouse.metric_container = metric_container