from collections import defaultdict
import warnings
import pandas as pd
import pytest
from pathlib import Path

from fiberphotometry.data.data_loading    import DataContainer, load_all_sessions
from fiberphotometry.data.session_loading import populate_containers
from fiberphotometry.data.syncer          import sync_session
from fiberphotometry.data.timepoint_processing import add_event_idxs_to_session
from fiberphotometry.data.handlers        import SPECIAL_PROCESSORS, split_event_suffix

# adjust to your project layout
SRC = Path("/Users/fsp585/Desktop/GetherLabCode/FiberphotometryCode/src")

@pytest.fixture(scope="module")
def sessions():
    """
    Load up to 2 folders under SRC that have T1_* trials,
    then populate their .dfs containers.
    """
    all_dirs = [
        d for d in SRC.iterdir()
        if d.is_dir() and any(c.is_dir() and c.name.startswith("T1_") for c in d.iterdir())
    ][:30]

    loaded = []
    for src in all_dirs:
        # if ('Marco' not in str(src)) and ('RewardDelay' not in str(src)):
        #     continue
        loaded.extend(load_all_sessions(
            baseline_dir=str(src),
            session_type="cpt",
            first_n_dirs=3,
            remove_bad_signal_sessions=True
        ))

    # fill in each session's dfs
    populate_containers(loaded)
    return loaded


# def test_sync_and_timepoints(sessions):
#     """
#     For each session:
#      1) sync all streams
#      2) build the event index container
#     Collect any failures, then report them all at once with session IDs.
#     """
#     errors = []

#     for sess in sessions:
#         try:
#             # 1) sync the timebases, should not raise
#             sync_session(sess)

#             # 2) build the event_idxs_container with the right dicts
#             #    (your existing per-‘Marco’ logic)
#             if "Marco" in sess.trial_dir:
#                 actions_attr_dict = {
#                     'Correct_Counter':     'global_correct_hit',
#                     'FIXED_RATIO_COUNTER': 'fixed_ratio_touch'
#                 }
#                 reward_attr_dict = {}
#             elif "Habituation" in sess.trial_dir:
#                 actions_attr_dict = {}
#                 reward_attr_dict = {}
#             else:
#                 actions_attr_dict = {
#                     "Hit": "hit",
#                     "Mistake": "mistake",
#                     "Missed Hit": "miss",
#                     "Correction Trial Correct Rejection": "cor_reject",
#                     "Correct Rejection": "cor_reject"
#                 }
#                 reward_attr_dict = {"Reward Collected Start ITI": "reward_collect"}

#             # reset and populate
#             sess.event_idxs_container = DataContainer(data_type=list)
#             add_event_idxs_to_session(sess, actions_attr_dict, reward_attr_dict)

#         except Exception as exc:
#             # collect the failure and keep going
#             errors.append((sess, exc))

#     if errors:
#         # build a nice multi-line message
#         lines = []
#         for sess, exc in errors:
#             lines.append(
#                 f"Session trial_id={sess.trial_dir!r}, mouse_id={sess.mouse_id!r} "
#                 f"failed: {exc!s}"
#             )
#         pytest.fail(
#             f"{len(errors)} session(s) failed during sync/timepoint processing:\n"
#             + "\n".join(lines)
#         )

# def test_sync_and_timepoints(sessions):
#     """
#     For each session:
#      1) sync all streams
#      2) build the event index container
#     Then at the end, print an aggregate histogram of all event-idx counts
#     (filtering out any event with total count <= 1).
#     """
#     errors = []
#     # accumulator for all sessions
#     all_counts = defaultdict(int)

#     for sess in sessions:
#         try:
#             sync_session(sess)

#             # choose your dicts…
#             if "Marco" in sess.trial_dir:
#                 actions_attr_dict = {
#                     'Correct_Counter':     'global_correct_hit',
#                     'FIXED_RATIO_COUNTER': 'fixed_ratio_touch'
#                 }
#                 reward_attr_dict = {}
#             elif "Habituation" in sess.trial_dir:
#                 actions_attr_dict = {}
#                 reward_attr_dict = {}
#             else:
#                 actions_attr_dict = {
#                     "Hit": "hit",
#                     "Mistake": "mistake",
#                     "Missed Hit": "miss",
#                     "Correction Trial Correct Rejection": "cor_reject",
#                     "Correct Rejection": "cor_reject"
#                 }
#                 reward_attr_dict = {"Reward Collected Start ITI": "reward_collect"}

#             sess.event_idxs_container = DataContainer(data_type=list)
#             add_event_idxs_to_session(sess, actions_attr_dict, reward_attr_dict)

#             # introspect the container’s internal dict of event→indices
#             data_dict = next(
#                 (v for v in vars(sess.event_idxs_container).values()
#                  if isinstance(v, dict)),
#                 {}
#             )
#             # accumulate counts
#             for key, idx_list in data_dict.items():
#                 all_counts[key] += len(idx_list)

#         except Exception as exc:
#             errors.append((sess, exc))

#     # after looping all sessions, fail if any errors
#     if errors:
#         lines = [
#             f"Session {s.trial_dir!r}, mouse {s.mouse_id!r} failed: {e}"
#             for s, e in errors
#         ]
#         pytest.fail("Errors in sync/timepoint processing:\n" + "\n".join(lines))

#     # build and print aggregate histogram
#     # filter out counts <= 1
#     filtered = {k: v for k, v in all_counts.items() if v > 1}
#     if not filtered:
#         print("No events with total count > 1 found across all sessions.")
#         return

#     # sort descending by count
#     sorted_events = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)
#     print("\nAggregate event-index counts (total > 1):")
#     for name, cnt in sorted_events:
#         print(f"{name:30s} {cnt:4d}")

# map each SPECIAL_PROCESSORS key to the set of suffix prefixes you expect
EXPECTED_SUFFIXES = {
    'rewardDelay':             {'RD'},
    'probabilisticReward_50%': {'PR'},
    'varITI':                  {'VI'},
    'varStimDur':              {'SD'},
    'Fixed_Ratio_baseline':    {'CC','FR'},
}

@pytest.mark.usefixtures("sessions")
def test_event_suffixes(sessions):
    """
    For each session whose task endswith one of the SPECIAL_PROCESSORS keys,
    confirm that after processing, the only suffix codes present are exactly
    the ones we expect for that processor.
    """
    wrong = []  # collect mismatches

    # we’ll need an actions/rewards dict for each session, same as your other tests:
    def dicts_for(sess):
        if "Marco" in sess.trial_dir:
            return (
                {'Correct_Counter':'global_correct_hit', 'FIXED_RATIO_COUNTER':'fixed_ratio_touch'},
                {}
            )
        elif "Habituation" in sess.trial_dir:
            return {}, {}
        else:
            return (
                {"Hit":"hit","Mistake":"mistake","Missed Hit":"miss",
                 "Correction Trial Correct Rejection":"cor_reject",
                 "Correct Rejection":"cor_reject"},
                {"Reward Collected Start ITI":"reward_collect"}
            )

    # sort keys longest-first so 'Fixed_Ratio_baseline' wins over any substring
    proc_keys = sorted(SPECIAL_PROCESSORS.keys(), key=len, reverse=True)

    for sess in sessions:
        # 1) find which processor (if any) applies
        match = next((k for k in proc_keys if sess.task.endswith(k)), None)

        # decide what suffix‐set we expect
        if match is None:
            expected = set()             # standard tasks should have NO suffixes
        else:
            expected = EXPECTED_SUFFIXES[match]

        # sync + index
        sync_session(sess)
        sess.event_idxs_container = DataContainer(data_type=list)
        actions_dict, reward_dict = dicts_for(sess)
        add_event_idxs_to_session(sess, actions_dict, reward_dict)

        # pull out all Item_Name values after processing
        df = sess.events_of_interest_df
        # extract suffix codes
        codes = df['Item_Name'].map(lambda nm: split_event_suffix(nm)[1])
        actual = {c for c in codes if c is not None}

        if actual != expected:
            wrong.append((sess, expected, actual))

    # if any mismatched, fail with detail
    if wrong:
        lines = []
        for sess, exp, act in wrong:
            lines.append(
                f"{sess.trial_dir!r}: expected suffixes {exp}, found {act}"
            )
        pytest.fail("Suffix-test failures:\n" + "\n".join(lines))