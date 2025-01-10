# phot_df has [FrameCounter, Timestamp, cam_frame_num, BonsaiTrackingTimestamp]
import pandas as pd

def adjust_names(s):
    first_second, mouse_ids, hand_dir, _ = s.split('_')
    id_one, id_two = mouse_ids.split('&')

    if hand_dir == 'LeftHand':
        return f'{first_second}_{id_one}'
    elif hand_dir == 'RightHand':
        return f'{first_second}_{id_two}'


def add_inj_times(sessions):
    timepoints = pd.read_csv('/Users/fsp585/Desktop/GetherLabCode/FiberphotometryCode/timepointListTable_withColNames.txt')
    timepoints['File_Name_adjusted'] = timepoints['File_Name'].apply(adjust_names)
    first_second = sessions[0].trial_dir.split('/')[-2].split()[0]
    mask = timepoints['File_Name_adjusted'].apply(lambda s: s.startswith(first_second))

    timepoints = timepoints[mask]
    timepoints.loc[:, 'mouse_id'] = timepoints['File_Name_adjusted'].apply(lambda s: s.split('_')[-1])
    timepoints = timepoints.set_index('mouse_id', drop=True)
    for idx, session in enumerate(sessions):
        session.inj_start_time = int(timepoints.loc[session.mouse_id, 'tOutForInject'])
        session.inj_end_time = int(timepoints.loc[session.mouse_id, 'tBackAfterInject'])