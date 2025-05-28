from collections import defaultdict
import warnings
import pandas as pd
from fiberphotometry.data.data_loading import DataContainer
from fiberphotometry.config import SESSION_CONFIG

from fiberphotometry.data.handlers import (
    SPECIAL_PROCESSORS,
    split_event_suffix,
    get_event_key,
    is_reward_event,
    base_event_name,
    _ensure_columns
)

_sorted_processors = sorted(
    SPECIAL_PROCESSORS.items(),
    key=lambda kv: len(kv[0]),
    reverse=True
)

def add_event_idxs_to_session(session, actions_attr_dict: dict, reward_attr_dict: dict):
    """
    Build and store all event index lists for one session.

    Steps:
      1) Validate session interface
      2) Pick the correct SpecialEventProcessor (if any)
      3) Load raw data and ensure required columns
      4) Filter to only the Item_Name values we care about
      5) Build a small events_table for debugging/inspection
      6) Apply the special-processor renaming logic (or drop Arg1_Value)
      7) Flag which rows are rewards
      8) Iterate action rows to collect before/after indices
      9) Iterate reward rows to collect reward indices
     10) Add static ITI-touch and Display-Image indices
     11) Push all lists into session.event_idxs_container
    """
    skip_seq = SESSION_CONFIG.get('skip_sequence_check', False)

    # 1) Validate session interface
    if not hasattr(session, 'dfs') or not hasattr(session.dfs, 'get_data'):
        raise AttributeError(f"Session {session!r} must have dfs.get_data method")
    if not isinstance(session.task, str):
        raise TypeError(f"Session.task must be a string, got {type(session.task)}")


    # 2) Pick the right processor by task-suffix (longest match first)
    matches = [proc for suffix, proc in _sorted_processors
               if session.task.endswith(suffix)]
    if len(matches) > 1:
        raise ValueError(f"Multiple SPECIAL_PROCESSORS match task {session.task!r}: "
                         f"{[type(p).__name__ for p in matches]}")
    processor = matches[0] if matches else None

    if processor is None and not actions_attr_dict and not reward_attr_dict:
        # Create empty tables so downstream code still has something
        session.events_table = pd.DataFrame(
            columns=['raw_idx','Item_Name','event_base','suffix_code','suffix_num']
        )
        session.events_of_interest_df = pd.DataFrame(
            columns=['Item_Name','is_reward']
        )
        return

    # 3) Load raw DataFrame and check columns
    raw = session.dfs.get_data('raw')
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("session.dfs.get_data('raw') must return a DataFrame")
    _ensure_columns(raw, require_arg1=True)   # ensure both columns exist
    item_df = raw[['Item_Name', 'Arg1_Value']]

    # 4) Filter to relevant Item_Name values
    if not isinstance(actions_attr_dict, dict) or not isinstance(reward_attr_dict, dict):
        raise TypeError("actions_attr_dict and reward_attr_dict must be dicts")
    events = ['Display Image', *actions_attr_dict.keys(), *reward_attr_dict.keys()]
    if processor:
        triggers = processor.triggers if hasattr(processor, 'triggers') else [processor.trigger]
        events.extend(triggers)

    filtered = item_df[item_df['Item_Name'].isin(events)].reset_index()
    if filtered.empty:
        warnings.warn(
            f"No rows in raw data matched expected events {events!r}\nsession={session.trial_dir, session.mouse_id}",
            UserWarning, stacklevel=2
        )

    # 5) Build events_table for raw-row metadata
    try:
        base, code, num = zip(*filtered['Item_Name'].map(split_event_suffix))
    except Exception as e:
        raise RuntimeError(f"split_event_suffix failed on filtered Item_Names,\nsession={session.trial_dir, session.mouse_id}") from e
    events_tbl = (
        filtered
        .assign(
            event_base  = base,
            suffix_code = [c.lower() if c else None for c in code],
            suffix_num  = pd.to_numeric(num, errors='coerce')
        )
        .rename(columns={'index':'raw_idx'})
        [['raw_idx','Item_Name','event_base','suffix_code','suffix_num']]
    )
    session.events_table = events_tbl

    # 6) Apply special processing or simply drop Arg1_Value
    if processor:
        filtered = processor.apply(filtered)
    else:
        # If no processor, ensure Arg1_Value can be dropped
        if 'Arg1_Value' not in filtered:
            raise RuntimeError("No processor and missing 'Arg1_Value' to drop")
        filtered = filtered.drop(columns=['Arg1_Value'])

    # 7) Flag reward rows on the processed Item_Name
    suffix = getattr(processor, 'suffix_prefix', None)
    filtered['is_reward'] = filtered['Item_Name'].apply(
        lambda name: is_reward_event(name, reward_attr_dict, suffix)
    )
    # Store for debugging or later inspection
    session.events_of_interest_df = filtered

    # 8) Collect indices for action (non-reward) rows
    event_idxs = defaultdict(list)
    before_dispimg = defaultdict(list)
    after_dispimg = defaultdict(list)
    # seed keys for actions
    for attr in actions_attr_dict.values():
        event_idxs.setdefault(attr, [])


    action_df = filtered[~filtered['is_reward']].reset_index(drop=True)
    previous_event = ''  # Keep track of last event for sequence checks
    for i, row in action_df.iterrows():
        name = row['Item_Name']
        base_name = base_event_name(name)
        prev_base = base_event_name(previous_event)

        # Ensure valid sequencing: no identical or back-to-back same actions
        if not skip_seq:
            if ({base_name, prev_base} <= set(actions_attr_dict.keys())) or base_name == prev_base:
                raise Exception(f"'{name}' cannot follow '{previous_event}' in the chain")

        # Skip Display Image in action indexing
        if base_name == 'Display Image':
            previous_event = name
            continue

        if base_name in actions_attr_dict:
            # Derive full (suffixed) and base attribute keys
            suffixed_attr = get_event_key(name, actions_attr_dict)
            base_attr = actions_attr_dict.get(base_name, base_name)

            # Store the event index under both keys if they differ
            if suffixed_attr and suffixed_attr != base_attr:
                event_idxs[suffixed_attr].append(row['index'])
                event_idxs[base_attr].append(row['index'])
            else:
                event_idxs[base_attr].append(row['index'])

            # Record before/after Display Image indices for base attribute
            if i > 0:
                before_dispimg[base_attr].append(action_df.iloc[i-1]['index'])
            if i < len(action_df) - 1:
                after_dispimg[base_attr].append(action_df.iloc[i+1]['index'])

            # Also record for suffixed attribute if different
            if suffixed_attr and suffixed_attr != base_attr:
                if i > 0:
                    before_dispimg[suffixed_attr].append(action_df.iloc[i-1]['index'])
                if i < len(action_df) - 1:
                    after_dispimg[suffixed_attr].append(action_df.iloc[i+1]['index'])

        previous_event = name

    # 9) Collect indices for reward rows
    reward_df = filtered[filtered['is_reward']].reset_index(drop=True)
    for _, row in reward_df.iterrows():
        name = row['Item_Name']
        if is_reward_event(name, reward_attr_dict, suffix):
            reward_key = get_event_key(name, reward_attr_dict) or reward_attr_dict.get(base_event_name(name))
            if reward_key:
                event_idxs[reward_key].append(row['index'])

    # 10) Add static ITI-touch and Display-Image indices
    iti_touch_indices = item_df[item_df['Item_Name'] == 'Centre_Touches_during_ITI'].index.tolist()
    disp_image_indices = item_df[item_df['Item_Name'] == 'Display Image'].index.tolist()
    session.event_idxs_container.add_data('iti_touch', iti_touch_indices)
    session.event_idxs_container.add_data('dispimg', disp_image_indices)

    # 11) Store all collected indices in the session container
    for attr_key, idx_list in event_idxs.items():
        session.event_idxs_container.add_data(attr_key, idx_list)
    for attr_key, idx_list in before_dispimg.items():
        session.event_idxs_container.add_data(f'before_dispimg_{attr_key}', idx_list)
    for attr_key, idx_list in after_dispimg.items():
        session.event_idxs_container.add_data(f'after_dispimg_{attr_key}', idx_list)


def create_event_idxs_container_for_sessions(sessions, actions_attr_dict: dict, reward_attr_dict: dict):
    """
    Initialize an event_idxs_container on each session and populate it.
    """
    for session in sessions:
        session.event_idxs_container = DataContainer(data_type=list)
        add_event_idxs_to_session(session, actions_attr_dict, reward_attr_dict)