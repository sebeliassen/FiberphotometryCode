from pathlib import Path

import pytest

from fiberphotometry.data.data_loading import load_all_sessions

# adjust to your project layout
SRC = Path("/Users/fsp585/Desktop/GetherLabCode/FiberphotometryCode/src_oft")

@pytest.fixture(scope="module")
def raw_sessions():
    """Discover & load up to 30 'cpt' sessions (but do NOT populate .dfs yet)."""
    all_dirs = [
        d for d in SRC.iterdir()
        if d.is_dir() and any(c.is_dir() and c.name.startswith("T1_")
                              for c in d.iterdir())
    ][:30]

    loaded = []
    for src in all_dirs:
        loaded.extend(load_all_sessions(
            baseline_dir=str(src),
            session_type="cpt",
            first_n_dirs=3,
            remove_bad_signal_sessions=True
        ))
    return loaded

def test_raw_sessions_load(raw_sessions):
    # raw_sessions should be a non‐empty list (or whatever property you expect)
    assert isinstance(raw_sessions, list)
    assert len(raw_sessions) > 0


# @pytest.fixture(scope="module")
# def sessions(raw_sessions):
#     """Take those raw sessions and fill in their .dfs containers."""
#     populate_containers(raw_sessions)
#     return raw_sessions
