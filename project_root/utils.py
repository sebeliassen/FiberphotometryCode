from config import * 

interval_start = peak_interval_config["interval_start"]
interval_end = peak_interval_config["interval_end"]

def find_start_end_idxs(event_type):
    fps = PLOTTING_CONFIG['fps']

    start_time, end_time = attr_interval_dict[event_type]
    start_event_idx = int(start_time * fps + interval_start)
    end_event_idx = int(end_time * fps + interval_start)

    return start_event_idx, end_event_idx

def find_session_by_trial_mouse_id(sessions, trial_id, mouse_id):
    if not isinstance(mouse_id, int):
        raise TypeError("mouse_id must be an int")
    
    if not isinstance(trial_id, int):
        raise TypeError("mouse_id must be an int")

    for session in sessions:
        curr_trial_id = int(session.trial_id.split("_")[0][1:])
        curr_mouse_id = int(session.mouse_id)
        if curr_trial_id == trial_id and curr_mouse_id == mouse_id:
            return session
    return None

def count_session_events(sessions, event_type):
    return sum(len(session.event_idxs_container.data.get(event_type, [])) for session in sessions)