from collections import defaultdict
from data.data_loading import DataContainer
import re  # assuming re is available in your environment
from abc import abstractmethod, ABC
from pandas import DataFrame
import pandas as pd
from fiberphotometry.config import SESSION_CONFIG

# Define the pattern once for reuse
EVENT_RE = re.compile(r'^(.*?)(_([A-Z]{2})(\d+(?:\.\d+)?))$')

def split_event_suffix(name: str):
    """
    Split off a trailing _XX<digits> suffix.
    Returns (base_name, code, number) or (name, None, None).
    """
    m = EVENT_RE.fullmatch(name)
    if not m:
        return name, None, None
    base, _, code, num = m.groups()
    return base, code, num

def base_event_name(name: str) -> str:
    """Return the base name without suffix."""
    base, _, _ = split_event_suffix(name)
    return base

class SpecialEventProcessor(ABC):
    def __init__(self, trigger: str, converter, suffix_prefix: str):
        self.trigger = trigger
        self.converter = converter
        self.suffix_prefix = suffix_prefix

    def apply(self, df: DataFrame) -> DataFrame:
        # 1) Pick off the rows to rename
        mask = df['Item_Name'] == self.trigger

        if mask.any():
            df.loc[mask, 'Item_Name'] = (
                df.loc[mask]
                  .apply(lambda r: f"{self.trigger}_{self.converter(r['Arg1_Value'])}", 
                         axis=1)
            )

        # 2) Delegate any per‐trigger suffix logic
        return self.process(df)

    @abstractmethod
    def process(self, df: DataFrame) -> DataFrame:
        pass

class AdjacentProcessor(SpecialEventProcessor):
    def _apply_adjacent(self, df: DataFrame) -> DataFrame:
        # (same body, but use self.trigger/self.suffix_prefix)
        if df.iloc[0]["Item_Name"].startswith(f"{self.trigger}_"):
            df = df.iloc[1:].reset_index(drop=True)
        mask = df["Item_Name"].str.startswith(f"{self.trigger}_")
        for i in df.index[mask]:
            value = df.at[i, "Item_Name"].split("_")[-1]
            suffix = f"_{self.suffix_prefix}{value}"
            if i > 0:
                df.at[i-1,  "Item_Name"] += suffix
            if i < len(df) - 1:
                df.at[i+1,  "Item_Name"] += suffix
        return df.drop(index=df.index[mask]).reset_index(drop=True)

    def process(self, df: DataFrame) -> DataFrame:
        return self._apply_adjacent(df)
    
class RunningProcessor(SpecialEventProcessor):
    """Carry suffix forward until next trigger"""
    def _apply_running(self, df: DataFrame) -> DataFrame:
        """
        In 'running' mode, use the special row's value as a running suffix that is appended
        to all subsequent rows (except 'Display Image') until the next special row is encountered,
        then drop the special rows.
        """
        current_suffix = None
        indices_to_drop = []
        for i in range(len(df)):
            item_name = df.loc[i, 'Item_Name']
            # If this is the trigger row, update suffix and mark for removal
            if item_name.startswith(f"{self.trigger}_"):
                value = item_name.rsplit('_', 1)[-1]
                current_suffix = f"_{self.suffix_prefix}{value}"
                indices_to_drop.append(i)
            else:
                # Append the current suffix to non-display-image rows
                if current_suffix and item_name != 'Display Image':
                    df.loc[i, 'Item_Name'] += current_suffix
        # Drop all trigger rows and reset index
        return df.drop(index=indices_to_drop).reset_index(drop=True)

    def process(self, df: DataFrame) -> DataFrame:
        return self._apply_running(df)
    
class RenameOnlyProcessor(SpecialEventProcessor):
    """
    Just renames trigger rows to include their Arg1_Value _with_ the two-letter prefix,
    then leaves the rest to CompositeProcessor.apply() (which will drop Arg1_Value).
    """
    def apply(self, df: DataFrame) -> DataFrame:
        # only rename the rows matching our trigger
        mask = df['Item_Name'] == self.trigger
        if mask.any():
            df.loc[mask, 'Item_Name'] = (
                df.loc[mask]
                  .apply(
                      lambda r: f"{self.trigger}_{self.suffix_prefix}{self.converter(r['Arg1_Value'])}",
                      axis=1
                  )
            )
        return df

    def process(self, df: DataFrame) -> DataFrame:
        # no extra logic here (CompositeProcessor will drop Arg1_Value for us)
        return df
    
class FixedRatioProcessor(SpecialEventProcessor):
    def process(self, df):
        # drop the “_0” resets
        zero_mask = df['Item_Name'].str.endswith('_0')
        return df[~zero_mask]

class CompositeProcessor:
    """
    Chains multiple SpecialEventProcessor-like objects in sequence.
    Exposes a unified .triggers list so that callers never have to know
    whether they’re dealing with a simple or composite processor.
    """
    def __init__(self, processors: list[SpecialEventProcessor]):
        # underlying processors
        self.processors = processors

    def apply(self, df: DataFrame) -> DataFrame:
        for proc in self.processors:
            df = proc.apply(df)

        # only now do we drop the shared Arg1_Value column
        if 'Arg1_Value' in df.columns:
            df = df.drop(columns=['Arg1_Value'])

        return df
    
    @property
    def triggers(self) -> list[str]:
        """
        The full list of Item_Name values this composite wants to see
        (i.e. each sub-processor’s trigger).
        """
        return [p.trigger for p in self.processors]
    
    @property
    def suffix_prefix(self):
        """
        CompositeProcessor doesn’t itself have a single suffix_prefix—
        if you ever need to know, dig into the individual processors.
        """
        return None

SPECIAL_PROCESSORS = {
    'rewardDelay':             AdjacentProcessor('Current_Reward_Delay', int,  'RD'),
    'probabilisticReward_50%': AdjacentProcessor('Feeder #1',           str,  'PR'),
    'varITI':                  RunningProcessor(  'Current_ITI',        int,  'VI'),
    'varStimDur':              RunningProcessor(  'stimulus_duration',  str,  'SD'),
}

SPECIAL_PROCESSORS["Fixed_Ratio_baseline"] = CompositeProcessor([
    RenameOnlyProcessor("Correct_Counter",     int, "CC"),
    RenameOnlyProcessor("FIXED_RATIO_COUNTER", int, "FR"),
])

def get_event_key(event_name: str, mapping: dict) -> str:
    base, code, num = split_event_suffix(event_name)
    if code and base in mapping:
        return mapping[base] + f"_{code.lower()}{num}"
    return mapping.get(event_name)


def is_reward_event(name: str, reward_dict, suffix_prefix = None) -> bool:
    """
    Returns True if `name` exactly matches one of the keys in reward_dict,
    or (if suffix_prefix is provided) if it starts with key + "_" + suffix_prefix.
    """
    if suffix_prefix:
        return any(name == k or name.startswith(f"{k}_{suffix_prefix}") 
                   for k in reward_dict)
    return name in reward_dict

def add_event_idxs_to_session(session, actions_attr_dict: dict, reward_attr_dict: dict):
    skip_seq = SESSION_CONFIG.get('skip_sequence_check', False)
    """
    Orchestrates filtering, special processing, and index collection.
    """
    # 7.1 Determine if a special processor applies based on task suffix
    processor = next(
        (p for suffix, p in SPECIAL_PROCESSORS.items() if session.task.endswith(suffix)),
        None
    )

    # 7.2 Load and filter raw event data
    raw = session.dfs.get_data('raw')
    item_df = raw[['Item_Name', 'Arg1_Value']]
    events = ['Display Image', *actions_attr_dict.keys(), *reward_attr_dict.keys()]
    if processor:
        # if this is a CompositeProcessor, grab all its triggers,
        # otherwise fall back to the single .trigger
        if hasattr(processor, 'triggers'):
            events.extend(processor.triggers)
        else:
            events.append(processor.trigger)
    filtered = item_df[item_df['Item_Name'].isin(events)].reset_index()

    #  — build a tidy table of every event row we’ll index later
    base, code, num = zip(*filtered['Item_Name'].map(split_event_suffix))
    events_tbl = (
        filtered
        .assign(
            event_base  = base,
            suffix_code = [c.lower() if c else None for c in code],
            suffix_num  = pd.to_numeric(num, errors='coerce')
        )
        .rename(columns={'index':'raw_idx'})   # preserve the original raw index
        [['raw_idx','Item_Name','event_base','suffix_code','suffix_num']]
    )
    session.events_table = events_tbl

    # 7.3 Apply special processing or drop Arg1_Value
    if processor:
        filtered = processor.apply(filtered)
    else:
        filtered = filtered.drop(columns=['Arg1_Value'])

    # 7.4 Flag reward rows
    suffix = getattr(processor, 'suffix_prefix', None)
    filtered['is_reward'] = filtered['Item_Name'].apply(
        lambda name: is_reward_event(name, reward_attr_dict, suffix)
    )
    # Store for debugging or later inspection
    session.events_of_interest_df = filtered

    names = session.events_of_interest_df["Item_Name"]

    # 7.5 Prepare containers for indices
    event_idxs = defaultdict(list)
    before_dispimg = defaultdict(list)
    after_dispimg = defaultdict(list)
    # Pre-seed keys for action attributes
    for attr in actions_attr_dict.values():
        event_idxs.setdefault(attr, [])

    # 7.6 Process action events (non-reward rows)
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

    # 7.7 Process reward events
    reward_df = filtered[filtered['is_reward']].reset_index(drop=True)
    for _, row in reward_df.iterrows():
        name = row['Item_Name']
        if is_reward_event(name, reward_attr_dict, suffix):
            reward_key = get_event_key(name, reward_attr_dict) or reward_attr_dict.get(base_event_name(name))
            if reward_key:
                event_idxs[reward_key].append(row['index'])

    # 7.8 Add ITI touch and Display Image indices
    iti_touch_indices = item_df[item_df['Item_Name'] == 'Centre_Touches_during_ITI'].index.tolist()
    disp_image_indices = item_df[item_df['Item_Name'] == 'Display Image'].index.tolist()
    session.event_idxs_container.add_data('iti_touch', iti_touch_indices)
    session.event_idxs_container.add_data('dispimg', disp_image_indices)

    # 7.9 Store all collected indices in session container
    for attr_key, idx_list in event_idxs.items():
        session.event_idxs_container.add_data(attr_key, idx_list)
    for attr_key, idx_list in before_dispimg.items():
        session.event_idxs_container.add_data(f'before_dispimg_{attr_key}', idx_list)
    for attr_key, idx_list in after_dispimg.items():
        session.event_idxs_container.add_data(f'after_dispimg_{attr_key}', idx_list)


def create_event_idxs_container_for_sessions(sessions, actions_attr_dict: dict, reward_attr_dict: dict):
    for session in sessions:
        session.event_idxs_container = DataContainer(data_type=list)
        add_event_idxs_to_session(session, actions_attr_dict, reward_attr_dict)