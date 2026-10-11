from typing import List

import numpy as np
import pandas as pd

from fpcode.behavior.transitions import adjacent_pairs


def get_group_transition_matrix(sessions, top_n: int = 15, ordered_syllables: List[int] = None, normalize: str = 'row') -> pd.DataFrame:
    """
    Aggregate transition matrices across sessions.
    If ordered_syllables is provided, uses them as rows/cols (ignore top_n).
    """
    import pandas as pd

    # 1. Identify syllables to use
    if ordered_syllables is not None:
        top_sylls = ordered_syllables
    else:
        all_sylls = []
        for s in sessions:
            if "syllable" in s.all_data_df.columns:
                all_sylls.append(s.all_data_df["syllable"].dropna())

        if not all_sylls: return pd.DataFrame()
        global_counts = pd.concat(all_sylls).value_counts()
        top_sylls = global_counts.head(top_n).index.tolist()

    # 2. Build aggregate matrix
    # Fix 1: Calculate Per-Session Matrix first, then Average (Avoid Pseudoreplication)

    session_matrices = []

    for s in sessions:
        if "syllable" not in s.all_data_df.columns: continue
        seq = s.all_data_df["syllable"].fillna(-1).to_numpy()

        # Keep gap sentinels while collapsing bouts; filter adjacent pairs only.
        pairs = adjacent_pairs(seq, collapse_runs=True)
        if not len(pairs): continue

        # Count bigrams for this session
        local_counts = pd.DataFrame(0.0, index=top_sylls, columns=top_sylls)

        syll_to_idx = {syll: i for i, syll in enumerate(top_sylls)}
        mapped = np.array([[syll_to_idx.get(x, -1) for x in pair] for pair in pairs])

        # Bigrams (vectorized)
        srcs = mapped[:, 0]
        dsts = mapped[:, 1]

        # Filter valid pairs
        valid_mask = (srcs != -1) & (dsts != -1)
        valid_srcs = srcs[valid_mask]
        valid_dsts = dsts[valid_mask]

        # Add to local counts
        np.add.at(local_counts.values, (valid_srcs, valid_dsts), 1)

        # Normalize to Probability Matrix
        if normalize == 'global':
            total_sum = local_counts.values.sum()
            local_prob = local_counts / max(1.0, total_sum)
        else:
            row_sums = local_counts.sum(axis=1)
            local_prob = local_counts.div(row_sums.replace(0, 1), axis=0)

        session_matrices.append(local_prob)

    if not session_matrices:
        return pd.DataFrame(0.0, index=top_sylls, columns=top_sylls)

    # Average across sessions
    # element-wise mean
    concatenated = np.stack([m.values for m in session_matrices])
    mean_matrix_vals = np.nanmean(concatenated, axis=0) # Use nanmean just in case

    mean_matrix = pd.DataFrame(mean_matrix_vals, index=top_sylls, columns=top_sylls)

    return mean_matrix

def get_group_usage(sessions, top_n: int = 20, min_instances: int = 0) -> pd.DataFrame:
    """
    Calculate average syllable usage across animal groups.
    Returns a long-form DF suitable for seaborn.
    """
    import pandas as pd

    from studies.kinematics_ulrik.lib.metrics import get_syllable_usage
    rows = []
    for s in sessions:
        usage = get_syllable_usage(s, min_instances=min_instances)
        animal = s.metadata.get("animal_id")
        geno = s.metadata.get("genotype")
        drug = s.metadata.get("drug", "N/A")

        for syll, freq in usage.items():
            rows.append({
                "Animal": animal,
                "Genotype": geno,
                "Drug": drug,
                "Syllable": syll,
                "Frequency": freq
            })
    return pd.DataFrame(rows)

