"""Visualization selection / density; historical V2 behavior preserved."""

from __future__ import annotations

from fpcode.viz.legacy_defaults import DENSITY_N_NEIGHBORS
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
import numpy as np


def choose_tier(n_eligible: int, G: int) -> str:
    """Tier from the ELIGIBLE count (determined after QC, never before)."""
    if n_eligible <= 0:
        return "no_eligible_bouts"
    if n_eligible > DENSITY_N_NEIGHBORS:
        return "tier_1_density"
    if n_eligible > G:
        return "tier_2_compact_neighborhood"
    return "tier_3_all_bouts"


def density_ratio_scores(X_pca: np.ndarray, global_nbrs, n_neighbors: int = DENSITY_N_NEIGHBORS):
    """
    Installed-KPMS density ratio (keypoint_moseq.util.sample_instances):

        local_density  = 1 / mean(kNN distances within the syllable, self included)
        global_density = 1 / mean(kNN distances to the background fit, self not guaranteed)
        ratio          = local_density / global_density

    The self-neighbor convention is KPMS's: the query point is part of the
    syllable's own fit, so its first neighbor is itself at distance 0.  A zero
    mean distance (all points identical) is floored to avoid a division error;
    KPMS itself has no floor, so any difference only appears in that degenerate
    case.
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_pca)
    distances, indices = nbrs.kneighbors(X_pca)
    local_mean = np.maximum(distances.mean(axis=1), np.finfo(np.float64).tiny)
    g_dist, _ = global_nbrs.kneighbors(X_pca)
    global_mean = np.maximum(g_dist.mean(axis=1), np.finfo(np.float64).tiny)
    ratio = (1.0 / local_mean) / (1.0 / global_mean)
    return ratio, distances, indices, 1.0 / local_mean, 1.0 / global_mean


def order_group_exemplar_first(
    exemplar: int, cand_idx: Sequence[int], cand_dist: Sequence[float], stable_rank: np.ndarray, G: int
) -> List[int]:
    """
    The display group: the scored exemplar FIRST, then its nearest candidates in
    increasing distance; exactly equal distances are ordered by stable bout ID.
    The exemplar is inserted explicitly (a tie in the KNN query can otherwise
    omit it) and removed from the remaining candidates (no duplicate).
    """
    others = [(float(d), int(stable_rank[i]), int(i)) for i, d in zip(cand_idx, cand_dist) if int(i) != int(exemplar)]
    others.sort()
    return [int(exemplar)] + [i for _, _, i in others][: max(0, G - 1)]


def select_tier1_density(
    X_pca: np.ndarray, global_nbrs, stable_rank: np.ndarray, G: int, n_neighbors: int = DENSITY_N_NEIGHBORS
) -> Dict[str, Any]:
    """Tier 1 (N > 50): KPMS density-ratio exemplar, then nearest G around it."""
    ratio, distances, indices, local_d, global_d = density_ratio_scores(X_pca, global_nbrs, n_neighbors)
    exemplar = int(np.argmax(ratio))
    nb_idx = [int(i) for i in indices[exemplar]]
    nb_dist = [float(d) for d in distances[exemplar]]
    chosen = order_group_exemplar_first(exemplar, nb_idx, nb_dist, stable_rank, G)
    return {
        "anchor": exemplar,
        "chosen": chosen,
        "neighbor_idx": nb_idx,
        "neighbor_dist": nb_dist,
        "density_ratio": float(ratio[exemplar]),
        "local_density": float(local_d[exemplar]),
        "global_density": float(global_d[exemplar]),
    }


def _medoid_and_order(D: np.ndarray, members: Sequence[int], stable_rank: np.ndarray) -> List[int]:
    """Medoid of ``members`` first, then the rest by (distance to medoid, stable ID)."""
    members = [int(m) for m in members]
    sub = D[np.ix_(members, members)]
    sums = sub.sum(axis=1)
    best = min(range(len(members)), key=lambda k: (float(sums[k]), int(stable_rank[members[k]])))
    medoid = members[best]
    rest = sorted((m for m in members if m != medoid), key=lambda m: (float(D[medoid, m]), int(stable_rank[m])))
    return [medoid] + rest


def select_tier2_compact(trajs: np.ndarray, stable_rank: np.ndarray, G: int) -> Dict[str, Any]:
    """Tier 2 (G < N <= 50): candidate sets of each bout + its G-1 nearest; lowest mean pairwise distance wins."""
    from sklearn.metrics import pairwise_distances

    N = len(trajs)
    D = pairwise_distances(trajs)
    best_score, best_set = None, None
    for i in range(N):
        order = sorted((j for j in range(N) if j != i), key=lambda j: (float(D[i, j]), int(stable_rank[j])))
        cand = [i] + order[: G - 1]
        sub = D[np.ix_(cand, cand)]
        score = float(sub[np.triu_indices(G, k=1)].mean())
        if best_score is None or score < best_score - 1e-12:
            best_score, best_set = score, cand
    chosen = _medoid_and_order(D, best_set, stable_rank)
    return {"anchor": chosen[0], "chosen": chosen, "compactness_score": best_score}


def select_tier3_all(trajs: np.ndarray, stable_rank: np.ndarray) -> Dict[str, Any]:
    """Tier 3 (N <= G): every eligible bout once, medoid first; blanks are added by the caller."""
    N = len(trajs)
    if N == 1:
        return {"anchor": 0, "chosen": [0]}
    from sklearn.metrics import pairwise_distances

    D = pairwise_distances(trajs)
    chosen = _medoid_and_order(D, list(range(N)), stable_rank)
    return {"anchor": chosen[0], "chosen": chosen}
