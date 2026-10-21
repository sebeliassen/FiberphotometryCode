_SESSIONS = None
_PROJECT_DIR = None

def set_context(sessions, project_dir_full=None):
    global _SESSIONS, _PROJECT_DIR
    _SESSIONS = sessions
    _PROJECT_DIR = project_dir_full

def get_sessions():
    return _SESSIONS

def get_project_dir():
    return _PROJECT_DIR
