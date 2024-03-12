from collections import defaultdict
from itertools import chain, product
from analysis.timepoint_analysis import aggregate_signals, collect_signals, get_signal_around_event
from analysis.performance_funcs import add_performance_container
from data.mouse import create_mice_dict
from config import attr_interval_dict
import random
from utils import count_session_events
import numpy as np


from collections import defaultdict
from itertools import chain, product

class MiceAnalysis:
    def __init__(self, sessions):
        self.mice_dict = create_mice_dict(sessions)
        self.set_all_metrics(sessions)
        self.cumulative_events_by_metric = self.calculate_cumulative_events(attr_interval_dict)

    def set_all_metrics(self, sessions):
        for mouse in self.mice_dict.values():
            add_performance_container(mouse)
        self.all_performance_metrics = list(self.mice_dict.values())[0].metric_container.data.keys()
        self.all_response_metrics = list({metric[-1] for metric in sessions[0].response_metrics.keys()})

    def calculate_cumulative_events(self, attr_interval_dict):
        cumulative_events_by_metric = {}
        for metric in self.all_performance_metrics:
            # Cumulative events data for the current metric
            cumulative_events = defaultdict(list)

            for mouse in sorted(self.mice_dict.values(), key=lambda m: m.metric_container.data[metric]):
                # Generate all combinations of sessions, brain regions (without postfix), and event types
                # TODO: hardcoded for now, should not be
                all_brain_regions = ['VS', 'DMS', 'DLS']
                combinations = product(mouse.sessions, all_brain_regions, attr_interval_dict.keys())

                for session, brain_region, event_type in combinations:
                    if brain_region not in '_'.join(session.brain_regions):
                        continue
                    curr_cumsum = cumulative_events[(brain_region, event_type)]

                    prev_sum = curr_cumsum[-1][1] if curr_cumsum else 0
                    key = (event_type, brain_region, self.all_response_metrics[0])

                    addend = len(session.response_metrics[key]) if key in session.response_metrics else 0
                    curr_cumsum.append((mouse.mouse_id, prev_sum + addend))

            cumulative_events_by_metric[metric] = cumulative_events

        return cumulative_events_by_metric

    def sample_high_and_low_sessions(self, metric, brain_region, event_type,
                                     n=1000, min_mice=3, max_mice=3):
        mouse_ids, cumsums = zip(*self.cumulative_events_by_metric[metric][(brain_region, event_type)])

        if max_mice > len(mouse_ids) // 2:
            max_mice = len(mouse_ids) // 2
            print(f"max_mice was set to {max_mice} because it was too high in relation to the number of mice.")

        low_mouse_ids = set()
        high_mouse_ids = set()

        from_left_idxs = []
        for i in range(len(mouse_ids) - 1):
            if mouse_ids[i] != mouse_ids[i + 1]:
                from_left_idxs.append(i)
    
        from_right_idxs = [i + 1 for i in from_left_idxs[::-1]]
        
        for from_left_idx, from_right_idx in zip(from_left_idxs, from_right_idxs):
            if len(low_mouse_ids) >= max_mice:
                break

            left_mouse_id = mouse_ids[from_left_idx]
            left_cumsum = cumsums[from_left_idx]

            right_mouse_id = mouse_ids[from_right_idx]
            right_cumsum = cumsums[-1] - cumsums[from_right_idx]

            low_mouse_ids.add(left_mouse_id)
            high_mouse_ids.add(right_mouse_id)
            if left_cumsum >= n and right_cumsum >= n and len(low_mouse_ids) >= min_mice:
                break

        # Initialize empty lists for the lowest and highest sessions
        lowest_sessions = []
        highest_sessions = []

        # Iterate through low mouse IDs and extend the lowest_sessions list with filtered sessions
        for mouse_id in low_mouse_ids:
            filtered_sessions = [session for session in self.mice_dict[mouse_id].sessions if brain_region in '_'.join(session.brain_regions)]
            lowest_sessions.extend(filtered_sessions)

        # Iterate through high mouse IDs and extend the highest_sessions list with filtered sessions
        for mouse_id in high_mouse_ids:
            filtered_sessions = [session for session in self.mice_dict[mouse_id].sessions if brain_region in '_'.join(session.brain_regions)]
            highest_sessions.extend(filtered_sessions)

        return lowest_sessions, highest_sessions
    
    def sample_response_metrics(self, metric, brain_region, event_type, n=200):
        low_sessions, high_sessions = \
            self.sample_high_and_low_sessions(metric, brain_region, event_type, n=n)

        low_vals = defaultdict(list)
        high_vals = defaultdict(list)

        for session in low_sessions:
            for response_metric in self.all_response_metrics:
                key = (event_type, brain_region, response_metric)
                values = session.response_metrics[key]
                values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
                low_vals[response_metric].extend(values)

        for session in high_sessions:
            for response_metric in self.all_response_metrics:
                key = (event_type, brain_region, response_metric)
                values = session.response_metrics[key]
                values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
                high_vals[response_metric].extend(values)

        low_sampled_vals = defaultdict(list)
        high_sampled_vals = defaultdict(list)

        for response_metric in low_vals.keys():
            low_vals_by_metric = low_vals[response_metric]
            high_vals_by_metric = high_vals[response_metric]
            final_n = min(n, len(low_vals_by_metric), len(high_vals_by_metric))            
            low_sampled_vals[response_metric] = random.sample(low_vals_by_metric, final_n)
            high_sampled_vals[response_metric] = random.sample(high_vals_by_metric, final_n)
        
        return low_sampled_vals, high_sampled_vals
    
    def sample_phot_signals(self, metric, brain_region, event_type, n=200):
        low_sessions, high_sessions = \
            self.sample_high_and_low_sessions(metric, brain_region, event_type, n=n)
        
        regions_to_aggregate = [f'{brain_region}_{suffix}' for suffix in ['left', 'right']]

        final_n = min(n, count_session_events(low_sessions, event_type), 
                         count_session_events(high_sessions, event_type))
        low_out = aggregate_signals(low_sessions, event_type, regions_to_aggregate, n=final_n)
        high_out = aggregate_signals(high_sessions, event_type, regions_to_aggregate, n=final_n)

        return low_out, high_out, low_sessions, high_sessions