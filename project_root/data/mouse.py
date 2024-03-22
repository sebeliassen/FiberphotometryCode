from config import actions_attr_dict
from collections import defaultdict
from utils import count_session_events

class Mouse:
    def __init__(self, mouse_sessions):
        all_mouse_ids = {mouse_session.mouse_id for mouse_session in mouse_sessions}
        if len(all_mouse_ids) != 1:
            raise AssertionError("Only sessions with the same mouse_id can be used for instantiation")
        
        self.mouse_id = all_mouse_ids.pop()  # Assuming all IDs are the same, pop the single element
        self.sessions = mouse_sessions
        self.event_counts = self.set_event_counts()

    def set_event_counts(self):
        event_counts = {}

        for event_type in actions_attr_dict.values():
            total = count_session_events(self.sessions, event_type)
            event_counts[event_type] = total

        return event_counts
    

def create_mice_dict(sessions):
    all_mouse_sessions = defaultdict(list)

    for session in sessions:
        all_mouse_sessions[session.mouse_id].append(session)

    all_mice = {}
    for mouse_id, mouse_sessions in all_mouse_sessions.items():
        all_mice[mouse_id] = Mouse(mouse_sessions)
    return all_mice