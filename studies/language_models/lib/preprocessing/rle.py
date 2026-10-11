import numpy as np


def collapse_transitions(session, mapping: dict = None) -> np.ndarray:
    """
    Applies Run-Length Encoding (RLE) to the syllable stream.
    Collapses consecutive duplicate frames (e.g., [walk, walk, walk, rear] -> [walk, rear]).
    
    Args:
        session: Session object with s.all_data_df.
        mapping: Optional dictionary to map syllable IDs to semantic strings first.
        
    Returns:
        np.ndarray: Collapsed syllable/label sequence.
    """
    df = session.all_data_df
    if "syllable" not in df.columns:
        return np.array([])

    raw_seq = df["syllable"].fillna(-1).to_numpy()
    if len(raw_seq) == 0:
        return np.array([])

    # Translate to semantic labels if mapping is provided
    if mapping is not None:
        semantic_seq = np.array([mapping.get(int(x), "Other") for x in raw_seq])
    else:
        semantic_seq = raw_seq.astype(int)

    # Run-length encode
    mask = np.concatenate(([True], semantic_seq[1:] != semantic_seq[:-1]))
    collapsed = semantic_seq[mask]
    return collapsed
