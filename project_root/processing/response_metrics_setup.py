from analysis.timepoint_analysis import get_signal_around_event
from config import attr_interval_dict

def add_response_metrics_to_sessions(sessions):
    for session in sessions: 
        for brain_regions in session.brain_regions:
            for event_type in attr_interval_dict.keys():
                if len(session.event_idxs_container.data.get(event_type, [])) == 0:
                    continue
                get_signal_around_event(session, event_type, brain_regions, add_response_metrics_to_session=True)