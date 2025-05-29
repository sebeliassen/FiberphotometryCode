import warnings
import pytest
from pathlib import Path
from collections import defaultdict

from fiberphotometry.config import PLOTTING_CONFIG
from fiberphotometry.data.data_loading import DataContainer, load_all_sessions
from fiberphotometry.data.session_loading import populate_containers
from fiberphotometry.data.syncer import sync_session
from fiberphotometry.data.timepoint_processing import add_event_idxs_to_session
from fiberphotometry.processing.plotting_setup import PlottingSetup
from fiberphotometry.processing.signal_info_setup import assign_sessions_signal_info

# adjust to your project layout
SRC = Path("/Users/fsp585/Desktop/GetherLabCode/FiberphotometryCode/src")

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

@pytest.fixture(scope="module")
def sessions(raw_sessions):
    """Take those raw sessions and fill in their .dfs containers."""
    populate_containers(raw_sessions)
    return raw_sessions

def actions_attr_dict_for(sess):
    if "Marco" in sess.trial_dir:
        return {
            'Correct_Counter':     'global_correct_hit',
            'FIXED_RATIO_COUNTER':'fixed_ratio_touch'
        }
    elif "Habituation" in sess.trial_dir:
        return {}
    else:
        return {
            "Hit":      "hit",
            "Mistake":  "mistake",
            "Missed Hit":"miss",
            "Correction Trial Correct Rejection":"cor_reject",
            "Correct Rejection":"cor_reject"
        }

def reward_attr_dict_for(sess):
    if "Marco" in sess.trial_dir or "Habituation" in sess.trial_dir:
        return {}
    else:
        return {"Reward Collected Start ITI":"reward_collect"}

def test_sync_and_timepoints(sessions):
    """
    For each session:
      1) sync the timebases
      2) build the event index container
      3) apply the plotting‐setup guards
      4) assign the signal_info
    Collect all failures, then report them with session identifiers.
    """
    errors = []

    for sess in sessions:
        try:
            # 1) Sync
            sync_session(sess)

            # 2) Build event‐indices
            sess.event_idxs_container = DataContainer(data_type=list)
            add_event_idxs_to_session(
                sess,
                actions_attr_dict_for(sess),
                reward_attr_dict_for(sess)
            )

            # 3) Plotting setup
            PlottingSetup(**PLOTTING_CONFIG['cpt']) \
                .apply_plotting_setup_to_sessions([sess])

            # 4) Signal‐info
            assign_sessions_signal_info([sess])

            # (Optional) Quick sanity check
            assert hasattr(sess, 'signal_info'), (
                f"{sess.trial_dir!r}/{sess.mouse_id!r}: "
                "signal_info attribute not set"
            )

        except Exception as exc:
            # Capture **any** exception along with session IDs
            errors.append((sess, exc))

    for sess in sessions:
        # each value in sess.signal_info is a dict with a 'signal_idx_ranges' list
        per_pair_counts = [len(info['signal_idx_ranges']) 
                        for info in sess.signal_info.values()]
        total_events = sum(per_pair_counts)
        print(
            f"{sess.trial_dir!r} (mouse {sess.mouse_id!r}): "
            f"{total_events} events across "
            f"{len(per_pair_counts)} region×event pairs"
        )
    if errors:
        lines = []
        for sess, exc in errors:
            lines.append(
                f"Session trial_dir={sess.trial_dir!r}, mouse_id={sess.mouse_id!r} "
                f"failed: {exc!s}"
            )
        pytest.fail(
            f"{len(errors)} session(s) failed during sync→timepoint processing:\n"
            + "\n".join(lines)
        )

# map each SPECIAL_PROCESSORS key to the set of suffix prefixes you expect
EXPECTED_SUFFIXES = {
    'rewardDelay':             {'RD'},
    'probabilisticReward_50%': {'PR'},
    'varITI':                  {'VI'},
    'varStimDur':              {'SD'},
    'Fixed_Ratio_baseline':    {'CC','FR'},
}

# @pytest.mark.usefixtures("sessions")
# def test_event_suffixes(sessions):
#     """
#     For each session whose task endswith one of the SPECIAL_PROCESSORS keys,
#     confirm that after processing, the only suffix codes present are exactly
#     the ones we expect for that processor.
#     """
#     wrong = []  # collect mismatches

#     # we’ll need an actions/rewards dict for each session, same as your other tests:
#     def dicts_for(sess):
#         if "Marco" in sess.trial_dir:
#             return (
#                 {'Correct_Counter':'global_correct_hit', 'FIXED_RATIO_COUNTER':'fixed_ratio_touch'},
#                 {}
#             )
#         elif "Habituation" in sess.trial_dir:
#             return {}, {}
#         else:
#             return (
#                 {"Hit":"hit","Mistake":"mistake","Missed Hit":"miss",
#                  "Correction Trial Correct Rejection":"cor_reject",
#                  "Correct Rejection":"cor_reject"},
#                 {"Reward Collected Start ITI":"reward_collect"}
#             )

#     # sort keys longest-first so 'Fixed_Ratio_baseline' wins over any substring
#     proc_keys = sorted(SPECIAL_PROCESSORS.keys(), key=len, reverse=True)

#     for sess in sessions:
#         # 1) find which processor (if any) applies
#         match = next((k for k in proc_keys if sess.task.endswith(k)), None)

#         # decide what suffix‐set we expect
#         if match is None:
#             expected = set()             # standard tasks should have NO suffixes
#         else:
#             expected = EXPECTED_SUFFIXES[match]

#         # sync + index
#         sync_session(sess)
#         sess.event_idxs_container = DataContainer(data_type=list)
#         actions_dict, reward_dict = dicts_for(sess)
#         add_event_idxs_to_session(sess, actions_dict, reward_dict)

#         # pull out all Item_Name values after processing
#         df = sess.events_of_interest_df
#         # extract suffix codes
#         codes = df['Item_Name'].map(lambda nm: split_event_suffix(nm)[1])
#         actual = {c for c in codes if c is not None}

#         if actual != expected:
#             wrong.append((sess, expected, actual))

#     # if any mismatched, fail with detail
#     if wrong:
#         lines = []
#         for sess, exp, act in wrong:
#             lines.append(
#                 f"{sess.trial_dir!r}: expected suffixes {exp}, found {act}"
#             )
#         pytest.fail("Suffix-test failures:\n" + "\n".join(lines))