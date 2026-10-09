from typing import List, Dict, Tuple

def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """Calculate frequencies of adjacent pairs."""
    pairs = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i+1])
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs

def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """Merge all occurrences of the given pair with a new token ID."""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def train_bpe_tokenizer(sequences: List[List[int]], vocab_size: int = 100) -> Dict[int, Tuple[int, int]]:
    """
    Trains a Byte-Pair Encoding (BPE) tokenizer on the behavioral sentences
    to merge the most frequent adjacent syllable transitions.
    
    Returns:
        Dict[int, Tuple[int, int]]: Merging rules mapping new tokens to (token_a, token_b).
    """
    # Flattens sequences with a special separator
    flat_ids = []
    for seq in sequences:
        flat_ids.extend(seq)
        flat_ids.append(-100)  # Separator token
        
    num_merges = vocab_size - len(set(flat_ids))
    merges = {}
    
    current_token_idx = max(flat_ids) + 1
    for i in range(num_merges):
        stats = get_stats(flat_ids)
        if not stats:
            break
        best_pair = max(stats, key=stats.get)
        if stats[best_pair] < 2:  # Stop if no pair occurs more than once
            break
        flat_ids = merge(flat_ids, best_pair, current_token_idx)
        merges[current_token_idx] = best_pair
        current_token_idx += 1
        
    return merges

def apply_bpe_merges(sequences: List[List[int]], merges: Dict[int, Tuple[int, int]]) -> List[List[int]]:
    """
    Applies BPE merges to a list of sequences.
    
    Args:
        sequences: List of list of syllable IDs.
        merges: BPE merging rules.
        
    Returns:
        List[List[int]]: Compressed sequences.
    """
    new_sequences = []
    for seq in sequences:
        flat_seq = list(seq)
        for token_idx, pair in merges.items():
            flat_seq = merge(flat_seq, pair, token_idx)
        new_sequences.append(flat_seq)
    return new_sequences
