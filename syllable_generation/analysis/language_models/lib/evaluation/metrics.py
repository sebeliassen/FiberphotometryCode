import numpy as np
from typing import List, Tuple

def calculate_perplexity(model, sentences: List[List[str]]) -> float:
    """
    Computes sequence perplexity ( surprise score ) across behavioral sentences.
    Higher perplexity means behavior transitions are less grammatical/structured.
    """
    log_probs = []
    total_tokens = 0
    
    n = model.n
    for sentence in sentences:
        if len(sentence) < n:
            continue
        for i in range(len(sentence) - n + 1):
            ngram = sentence[i:i+n]
            context = ngram[:-1]
            target = ngram[-1]
            prob = model.get_probability(context, target)
            log_probs.append(np.log2(prob))
            total_tokens += 1
            
    if total_tokens == 0:
        return np.nan
        
    avg_log_prob = np.sum(log_probs) / total_tokens
    perplexity = 2 ** (-avg_log_prob)
    return perplexity

def calculate_levenshtein_distance(seq_a: List[str], seq_b: List[str]) -> float:
    """ Computes Edit Distance (Levenshtein) between discrete sentences """
    size_x = len(seq_a) + 1
    size_y = len(seq_b) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq_a[x-1] == seq_b[y-1]:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1] + 1,
                    matrix[x, y-1] + 1
                )
    return matrix[size_x - 1, size_y - 1]
