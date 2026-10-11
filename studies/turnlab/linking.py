def add_session_idx(merged_all, sessions, session_base_col='session_base', h5_attr='kpms_h5_path',
                    strict=True, use_fallback=True):
    """
    Add a 'session_idx' column to merged_all by matching session_base to the stem of sessions[i].kpms_h5_path.
    - session_base_col: column in merged_all containing the expected stem (basename without '.h5')
    - h5_attr: attribute on each session that holds the full .h5 path
    - strict: if True, assert that all rows were matched (bijective assumption)
    - use_fallback: if True, try a light substring fallback for any unmatched stems

    Returns: merged_all (same object) with 'session_idx' added.
    """
    # 1) Build stem -> index map from sessions
    stem_to_idx = {}
    dups = []
    for i, s in enumerate(sessions):
        h5 = getattr(s, h5_attr, '') or ''
        fname = h5.rsplit('/', 1)[-1]            # basename
        stem = fname[:-3] if fname.endswith('.h5') else fname
        if stem in stem_to_idx:
            dups.append(stem)
        stem_to_idx[stem] = i

    if dups:
        print(f"⚠️ Duplicate stems found (unexpected for bijection): {dups[:5]}{' ...' if len(dups) > 5 else ''}")

    # 2) Direct map
    merged_all['session_idx'] = merged_all[session_base_col].map(stem_to_idx)

    # 3) Optional fallback for any misses (substring containment in either direction)
    if use_fallback:
        missing = merged_all.loc[merged_all['session_idx'].isna(), session_base_col].unique().tolist()
        if missing:
            for base in missing:
                cands = [idx for stem, idx in stem_to_idx.items() if (base in stem) or (stem in base)]
                if len(cands) == 1:
                    stem_to_idx[base] = cands[0]
            merged_all['session_idx'] = merged_all['session_idx'].fillna(
                merged_all[session_base_col].map(stem_to_idx)
            )

    # 4) Sanity / strictness
    n_missing = int(merged_all['session_idx'].isna().sum())
    if n_missing and strict:
        examples = merged_all.loc[merged_all['session_idx'].isna(), session_base_col].unique()[:5]
        raise ValueError(f"{n_missing} rows unmatched. Examples: {list(examples)}")
    elif n_missing:
        print(f"⚠️ {n_missing} rows unmatched; leaving session_idx as NaN for those rows.")

    return merged_all
