"""Karen V2 annotation hierarchy, copied explicitly from the dated master study.

These are hand-curated semantic aggregations, not fitted distance clusters.
The original master mapping stays intact. Unknown-preserving variants are an
explicit sensitivity policy, with dump members also kept individually distinct.
"""

import numpy as np
import pandas as pd

MESO = {
    "Dash_Fwd": [3, 4, 14, 42, 45], "Sprint": [24, 38], "Dash_Lat": [80, 94],
    "Stepped_Dash": [13, 19], "Walk_R": [10, 55, 75], "Walk_L": [23, 82],
    "Turn_R": [20, 34, 46, 57, 85], "Turn_L": [8, 71],
    "Look_R": [6, 66, 88, 90, 96], "Look_L": [48, 56], "Look_Fwd": [53],
    "Sniff_Fwd": [15, 43, 50, 51, 52], "Sniff_Lat": [41], "Stationary": [25],
    "Rear": [1], "Retreat": [16, 22, 54], "Scrunch": [58, 60],
    "Groom_Paw": [5, 9, 78], "Groom_Body": [28, 59, 67], "Rest": [37],
    "Dump_State": [12, 18, 29, 32, 49],
}
BROAD = {
    "Locomotion_Dash": [3, 4, 13, 14, 19, 24, 38, 42, 45, 80, 94],
    "Locomotion_Walk": [10, 23, 55, 75, 82], "Steering_Turn": [8, 20, 34, 46, 57, 71, 85],
    "Orientation_Look": [6, 48, 53, 56, 66, 88, 90, 96], "Sniff_Exploration": [15, 41, 43, 50, 51, 52],
    "Stationary_Arrest": [25], "Rearing_Wall": [1], "Retreat_Backoff": [16, 22, 54],
    "Postural_Scrunch": [58, 60], "Grooming_Maintain": [5, 9, 78], "Scratching_Body": [28, 59, 67],
    "Quiescent_Rest": [37], "Dump_Outlier": [12, 18, 29, 32, 49],
}


def _group_map(groups, *, preserve_unresolved=False):
    groups = {name: ids for name, ids in groups.items() if not (preserve_unresolved and name.startswith("Dump"))}
    used = {sid for ids in groups.values() for sid in ids}
    other = sorted(set(range(100)) - used)
    if preserve_unresolved:
        for sid in other:
            groups[("Dump" if sid in [12, 18, 29, 32, 49] else "Unassigned") + f"_S{sid:02d}"] = [sid]
    else:
        groups["Other_Unassigned"] = other
    mapping = np.zeros(100, int)
    for index, ids in enumerate(groups.values()):
        mapping[ids] = index
    return mapping, list(groups)


def build_hierarchy(annotations):
    raw = annotations.set_index("syllable_id")
    synonyms = np.arange(100)
    synonyms[[4, 14, 42, 45]], synonyms[[51, 52]], synonyms[60] = 3, 50, 58
    _, synonyms = np.unique(synonyms, return_inverse=True)
    specs = {"L1": (np.arange(100), [f"S{i:02d} {raw.loc[i, 'description']}" for i in range(100)]),
             "L2": (synonyms, [])}
    for cluster in range(synonyms.max() + 1):
        ids = np.flatnonzero(synonyms == cluster)
        specs["L2"][1].append(" + ".join(f"S{i:02d} {raw.loc[i, 'description']}" for i in ids))
    specs["L3"], specs["L4"] = _group_map(MESO), _group_map(BROAD)
    specs["L3_unresolved_separate"] = _group_map(MESO, preserve_unresolved=True)
    specs["L4_unresolved_separate"] = _group_map(BROAD, preserve_unresolved=True)
    mappings, memberships, tables = {}, [], {}
    for level, (mapping, names) in specs.items():
        mappings[level] = mapping
        rows = []
        for cluster, name in enumerate(names):
            ids = np.flatnonzero(mapping == cluster)
            categories = raw.loc[ids, "category"].tolist()
            unresolved = any(c.startswith(("Unassigned", "Dump")) for c in categories)
            rows.append({"syllable_id": cluster, "description": name, "category": name,
                         "raw_members": " ".join(map(str, ids)), "unresolved": unresolved})
            for sid in ids:
                memberships.append({"level": level, "raw_syllable_id": sid, "cluster_id": cluster,
                                    "cluster_name": name, "raw_description": raw.loc[sid, "description"],
                                    "raw_category": raw.loc[sid, "category"], "unresolved": unresolved})
        tables[level] = pd.DataFrame(rows)
    return mappings, tables, pd.DataFrame(memberships)
