import numpy as np

def calculate_dtw_distance(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    """
    Computes multi-dimensional Dynamic Time Warping (DTW) distance 
    between two continuous 10D PCA postural trajectories.
    
    Args:
        traj_a: Array of shape (L1, 10) representing continuous pose path.
        traj_b: Array of shape (L2, 10) representing continuous pose path.
        
    Returns:
        float: Aligned postural distance.
    """
    L1 = len(traj_a)
    L2 = len(traj_b)
    
    # Cost matrix
    dtw_matrix = np.full((L1 + 1, L2 + 1), np.inf)
    dtw_matrix[0, 0] = 0.0
    
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            # Compute Euclidean distance in 10D PCA space
            cost = np.linalg.norm(traj_a[i-1] - traj_b[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # Insertion
                dtw_matrix[i, j-1],    # Deletion
                dtw_matrix[i-1, j-1]   # Match
            )
            
    return dtw_matrix[L1, L2]
