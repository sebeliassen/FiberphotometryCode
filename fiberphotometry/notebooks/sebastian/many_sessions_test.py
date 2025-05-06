import numpy as np
from scripts.create_cpt_sessions import load_and_prepare_sessions
from fiberphotometry.data.syncer import Syncer as NewSyncer
from old_module.data.setup_utils import Renamer as OldRenamer, Syncer as OldSyncer  # import your hard-coded ones
from old_module.config import RENAME_PATTERNS as OLD_RENAME_PATTERNS

def compare_one_session(baseline_dir, session_idx=0):
    # 1) Load raw data without any processing
    sessions = load_and_prepare_sessions(
        baseline_dir,
        session_type='cpt',
        first_n_dirs=1,
        load_from_pickle=False,
        remove_bad_signal_sessions=False
    )
    sess_new = sessions[session_idx]

    # 2) Clone it for “old” processing
    import copy
    sess_old = copy.deepcopy(sess_new)

    # 3) Run the old renamer + sync
    OldRenamer.rename_sessions_data([sess_old], OLD_RENAME_PATTERNS)
    OldSyncer.sync_session(sess_old)

    # 4) Run the new renamer + sync
    #    (assuming your new Renamer.finalize_ttl_for_session is hooked in load_and_prepare_sessions)
    NewSyncer.sync_session(sess_new)

    # 5) Grab the key arrays
    raw = sess_new.dfs.get_data('raw')
    phot_old = sess_old.dfs.get_data('phot_470')  # or 415
    phot_new = sess_new.dfs.get_data('phot_470')

    # 6) Extract the intermediate arrays
    print("=== RAW Evnt_Time[0:5] ===")
    print(raw['Evnt_Time'].values[:5])

    print("\n=== OLD SecFromZero_FP3002 head ===")
    print(phot_old['SecFromZero_FP3002'].values[:5])

    print("\n=== OLD SecFromTrialStart_FP3002 head ===")
    print(phot_old['SecFromTrialStart_FP3002'].values[:5])

    print("\n=== NEW SecFromZero head ===")
    print(phot_new['SecFromZero'].values[:5])

    print("\n=== NEW SecFromTrialStart head ===")
    print(phot_new['SecFromTrialStart'].values[:5])

    # 7) Compare the “first-true-trial” index:
    old_idx = int(np.searchsorted(
        phot_old['SecFromTrialStart_FP3002'].values, 0, side='right'
    ))
    new_idx = int(np.searchsorted(
        phot_new['SecFromTrialStart'].values, 0, side='right'
    ))
    print(f"\nold trial_start_idx = {old_idx}, new trial_start_idx = {new_idx}")

if __name__ == "__main__":
    compare_one_session("/Users/fsp585/Desktop/GetherLabCode/FiberphotometryCode/Baseline")
