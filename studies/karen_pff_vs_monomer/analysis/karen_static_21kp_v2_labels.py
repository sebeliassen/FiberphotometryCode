#!/usr/bin/env python3
"""
karen_static_21kp_v2_labels.py
Ground-truth human annotations and ethological semantics for the 100 syllables
in the Full Cohort Static 21kp V2 production model (Seed 101).
"""

from typing import Dict

KAREN_21KP_LABELS: Dict[int, Dict[str, str]] = {
    0: {"name": "x", "notes": "", "category": "Unassigned"},
    1: {"name": "rear", "notes": "", "category": "Rearing / Postural"},
    2: {"name": "x", "notes": "", "category": "Unassigned"},
    3: {"name": "dash", "notes": "walking pace", "category": "Locomotion / Dash"},
    4: {"name": "dash", "notes": "fast", "category": "Locomotion / Dash"},
    5: {"name": "groom", "notes": "often also sniff", "category": "Grooming / Maintenance"},
    6: {"name": "look right", "notes": "slightly turn right as well", "category": "Orientation / Glance"},
    7: {"name": "x", "notes": "", "category": "Unassigned"},
    8: {"name": "turn left", "notes": "", "category": "Steering / Turn"},
    9: {"name": "groom", "notes": "sometimes sniff too; not enough bottom acc to accurately determine all the time", "category": "Grooming / Maintenance"},
    10: {"name": "walk right", "notes": "close to walk straight", "category": "Locomotion / Walk"},
    11: {"name": "x", "notes": "", "category": "Unassigned"},
    12: {"name": "bad syllable", "notes": "maybe a dump state or something?", "category": "Dump / Outlier"},
    13: {"name": "short dash", "notes": "take one step while looking left", "category": "Locomotion / Stepped Dash"},
    14: {"name": "dash", "notes": "", "category": "Locomotion / Dash"},
    15: {"name": "sniff", "notes": "after walking forward; observant syllable", "category": "Investigation / Sniff"},
    16: {"name": "back off", "notes": "sometimes descent; sometimes back-pedal", "category": "Locomotion / Retreat"},
    17: {"name": "x", "notes": "", "category": "Unassigned"},
    18: {"name": "dump syllable?", "notes": "", "category": "Dump / Outlier"},
    19: {"name": "short dash", "notes": "while looking left", "category": "Locomotion / Stepped Dash"},
    20: {"name": "turn right", "notes": "", "category": "Steering / Turn"},
    21: {"name": "x", "notes": "", "category": "Unassigned"},
    22: {"name": "back off", "notes": "reverse dash, retreat", "category": "Locomotion / Retreat"},
    23: {"name": "walk left", "notes": "", "category": "Locomotion / Walk"},
    24: {"name": "run", "notes": "", "category": "Locomotion / Sprint"},
    25: {"name": "observe", "notes": "very still syllable", "category": "Investigation / Arrest"},
    26: {"name": "x", "notes": "", "category": "Unassigned"},
    27: {"name": "x", "notes": "", "category": "Unassigned"},
    28: {"name": "scratch", "notes": "licking behind (circular pose), distinct from grooming paws", "category": "Grooming / Scratch"},
    29: {"name": "dumpster syllable", "notes": "median is strongly a rear, but way too multimodal", "category": "Dump / Postural"},
    30: {"name": "x", "notes": "", "category": "Unassigned"},
    31: {"name": "x", "notes": "", "category": "Unassigned"},
    32: {"name": "dump syllable?", "notes": "multimodal, but primarily grooming/scratching", "category": "Dump / Groom"},
    33: {"name": "x", "notes": "", "category": "Unassigned"},
    34: {"name": "turn right", "notes": "", "category": "Steering / Turn"},
    35: {"name": "x", "notes": "", "category": "Unassigned"},
    36: {"name": "x", "notes": "", "category": "Unassigned"},
    37: {"name": "rest", "notes": "", "category": "Quiescence / Rest"},
    38: {"name": "run", "notes": "", "category": "Locomotion / Sprint"},
    39: {"name": "x", "notes": "", "category": "Unassigned"},
    40: {"name": "x", "notes": "", "category": "Unassigned"},
    41: {"name": "sniff right", "notes": "observant syllable", "category": "Investigation / Sniff"},
    42: {"name": "dash", "notes": "", "category": "Locomotion / Dash"},
    43: {"name": "sniff", "notes": "stretch to sniff, observant", "category": "Investigation / Sniff"},
    44: {"name": "x", "notes": "", "category": "Unassigned"},
    45: {"name": "dash", "notes": "", "category": "Locomotion / Dash"},
    46: {"name": "turn right", "notes": "", "category": "Steering / Turn"},
    47: {"name": "x", "notes": "", "category": "Unassigned"},
    48: {"name": "look left", "notes": "", "category": "Orientation / Glance"},
    49: {"name": "bad syllable", "notes": "rear and sniff", "category": "Dump / Complex"},
    50: {"name": "sniff", "notes": "", "category": "Investigation / Sniff"},
    51: {"name": "sniff", "notes": "", "category": "Investigation / Sniff"},
    52: {"name": "sniff", "notes": "", "category": "Investigation / Sniff"},
    53: {"name": "look ahead", "notes": "very short dash", "category": "Orientation / Glance"},
    54: {"name": "back off", "notes": "", "category": "Locomotion / Retreat"},
    55: {"name": "walk right", "notes": "", "category": "Locomotion / Walk"},
    56: {"name": "look left", "notes": "slight sniff", "category": "Orientation / Glance"},
    57: {"name": "turn right", "notes": "", "category": "Steering / Turn"},
    58: {"name": "scrunch", "notes": "", "category": "Postural / Scrunch"},
    59: {"name": "scratch", "notes": "", "category": "Grooming / Scratch"},
    60: {"name": "scrunch", "notes": "", "category": "Postural / Scrunch"},
    66: {"name": "look right", "notes": "", "category": "Orientation / Glance"},
    67: {"name": "scratch", "notes": "", "category": "Grooming / Scratch"},
    71: {"name": "turn left", "notes": "", "category": "Steering / Turn"},
    75: {"name": "walk right", "notes": "", "category": "Locomotion / Walk"},
    78: {"name": "groom", "notes": "", "category": "Grooming / Maintenance"},
    80: {"name": "dash right", "notes": "", "category": "Locomotion / Dash"},
    82: {"name": "walk left", "notes": "", "category": "Locomotion / Walk"},
    85: {"name": "turn right", "notes": "", "category": "Steering / Turn"},
    88: {"name": "look right", "notes": "", "category": "Orientation / Glance"},
    90: {"name": "look right", "notes": "sometimes groom", "category": "Orientation / Glance"},
    94: {"name": "dash left", "notes": "", "category": "Locomotion / Dash"},
    96: {"name": "look right", "notes": "", "category": "Orientation / Glance"},
}

def get_label(syllable_id: int) -> str:
    """Returns human readable label string: name (or Syllable XX if unknown)."""
    if syllable_id in KAREN_21KP_LABELS:
        name = KAREN_21KP_LABELS[syllable_id]["name"]
        if name and name != "x":
            return f"S{syllable_id:02d}_{name.replace(' ', '_')}"
    return f"S{syllable_id:02d}"

def get_chain_label(tokens: list) -> str:
    """Returns human readable chain string from token list."""
    return " -> ".join([get_label(t) for t in tokens])
