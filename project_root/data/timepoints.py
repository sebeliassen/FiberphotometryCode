from collections import defaultdict
from data.data_loading import DataContainer


def add_event_idxs_to_session(session, actions_attr_dict, reward_attr_dict):
    # Initialize dictionaries
    before_dispimg_event_idxs_dict = {attr: [] for attr in actions_attr_dict.values()}
    after_dispimg_event_idxs_dict = {attr: [] for attr in actions_attr_dict.values()}
    event_idxs_lists_dict = {attr: [] for attr in actions_attr_dict.values()}

    # Define events of interest including 'Display Image'
    events_of_interest = ['Display Image'] + list(actions_attr_dict.keys())

    # Get raw data and filter for events of interest
    item_df = session.dfs.get_data("raw")[["Item_Name"]]
    for reward_item, reward_attr in reward_attr_dict.items():
        event_idxs_lists_dict[reward_attr] = list(item_df[item_df["Item_Name"] == reward_item].index)
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
            # Save the event_idx for each action event
            event_idxs_lists_dict[attr].append(row['index'])  # Using DataFrame index as event_idx

            # Assign before and after 'Display Image' event_idxs
            if i > 0:
                before_dispimg_event_idxs_dict[attr].append(filtered_df.iloc[i - 1]['index'])  # Previous row index
            if i < len(filtered_df) - 1:
                after_dispimg_event_idxs_dict[attr].append(filtered_df.iloc[i + 1]['index'])  # Next row index
        prev_item_name = item_name

    # Utilize another instance of DataContainer specifically for event_idxs
    for attr, event_idxs in event_idxs_lists_dict.items():
        session.event_idxs_container.add_data(attr, event_idxs)
    for attr, event_idxs in before_dispimg_event_idxs_dict.items():
        session.event_idxs_container.add_data(f'before_dispimg_{attr}', event_idxs)
    for attr, event_idxs in after_dispimg_event_idxs_dict.items():
        session.event_idxs_container.add_data(f'after_dispimg_{attr}', event_idxs)
    
    iti_touch_idxs = list(item_df[item_df["Item_Name"] == "Centre_Touches_during_ITI"].index)
    session.event_idxs_container.add_data('iti_touch', iti_touch_idxs)

    display_img_idxs = list(item_df[item_df["Item_Name"] == "Display Image"].index)
    session.event_idxs_container.add_data('dispimg', display_img_idxs)

def create_event_idxs_container_for_sessions(sessions, actions_attr_dict, reward_attr_dict):
    for session in sessions:
        session.event_idxs_container = DataContainer(data_type=list)
        add_event_idxs_to_session(session, actions_attr_dict, reward_attr_dict)