from collections import defaultdict
from data.data_loading import DataContainer


def add_timepoints_to_session(session, actions_attr_dict, reward_attr_dict):
    # Initialize dictionaries
    before_dispimg_timepoints_dict = {attr: [] for attr in actions_attr_dict.values()}
    after_dispimg_timepoints_dict = {attr: [] for attr in actions_attr_dict.values()}
    timepoints_lists_dict = {attr: [] for attr in actions_attr_dict.values()}

    # Define events of interest including 'Display Image'
    events_of_interest = ['Display Image'] + list(actions_attr_dict.keys())

    # Get raw data and filter for events of interest
    item_df = session.df_container.get_data("raw")[["Item_Name"]]
    for reward_item, reward_attr in reward_attr_dict.items():
        timepoints_lists_dict[reward_attr] = list(item_df[item_df["Item_Name"] == reward_item].index)
    filtered_df = item_df[item_df["Item_Name"].isin(events_of_interest)].reset_index()
    session.events_of_interest_df = filtered_df

    prev_item_name = ""
    action_keys = set(actions_attr_dict.keys())

    # Iterate through the filtered DataFrame
    for i, row in filtered_df.iterrows():
        item_name = row['Item_Name']
        # Check for the interlocking pattern and raise an exception if it's not met
        if ({item_name, prev_item_name} <= action_keys) or (item_name == prev_item_name):
            raise Exception(f"'{item_name}' cannot follow '{prev_item_name}' in the chain of events")
        
        if item_name == 'Display Image':
            prev_item_name = item_name
            continue
        elif item_name in actions_attr_dict:
            attr = actions_attr_dict[item_name]
            # Save the timepoint for each action event
            timepoints_lists_dict[attr].append(row['index'])  # Using DataFrame index as timepoint

            # Assign before and after 'Display Image' timepoints
            if i > 0:
                before_dispimg_timepoints_dict[attr].append(filtered_df.iloc[i - 1]['index'])  # Previous row index
            if i < len(filtered_df) - 1:
                after_dispimg_timepoints_dict[attr].append(filtered_df.iloc[i + 1]['index'])  # Next row index
        prev_item_name = item_name

    # Utilize another instance of DataContainer specifically for timepoints
    for attr, timepoints in timepoints_lists_dict.items():
        session.timepoints_container.add_data(attr, timepoints)
    for attr, timepoints in before_dispimg_timepoints_dict.items():
        session.timepoints_container.add_data(f'before_dispimg_{attr}', timepoints)
    for attr, timepoints in after_dispimg_timepoints_dict.items():
        session.timepoints_container.add_data(f'after_dispimg_{attr}', timepoints)


def create_timepoint_container_for_sessions(sessions, actions_attr_dict, reward_attr_dict):
    for session in sessions:
        session.timepoints_container = DataContainer(data_type=list)
        add_timepoints_to_session(session, actions_attr_dict, reward_attr_dict)