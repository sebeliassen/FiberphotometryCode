import os
import pickle
from typing import List, Tuple, Dict, Any, Optional

class SessionDataset:
    """
    Generalized session dataset provider for behavioral strategy modeling.
    Supports dynamic cohort grouping across genotypes (e.g., WT, KI, SCHIZO), 
    treatments (e.g., Saline, AMPH), and custom cohort metadata tags.
    """
    def __init__(self, cache_path: Optional[str] = None):
        self.clusterer = None
        self.all_win_inputs: List[Any] = []
        self.session_window_meta: List[Tuple] = []
        self.labels: List[str] = []
        self.n_windows_per_session: List[int] = []
        self.cache_path = cache_path
        
        if cache_path:
            self.load_from_cache(cache_path)

    def load_from_cache(self, cache_path: str) -> None:
        """
        Loads dataset sliding windows and metadata from a precomputed pickle cache.
        """
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
        with open(cache_path, "rb") as f:
            self.clusterer, self.all_win_inputs, self.session_window_meta = pickle.load(f)
            
        self.cache_path = cache_path
        self.labels = [meta[5] for meta in self.session_window_meta]
        self.n_windows_per_session = [meta[0] for meta in self.session_window_meta]

    def get_cohort_labels(self) -> List[str]:
        """
        Returns group/cohort labels for each session (e.g., 'WT_Saline', 'KI_AMPH', 'SCHIZO_Saline').
        """
        return self.labels

    def get_window_inputs(self) -> List[Any]:
        """
        Returns all sliding window inputs across sessions.
        """
        return self.all_win_inputs

    def get_num_sessions(self) -> int:
        """
        Returns total number of sessions in the dataset.
        """
        return len(self.labels)

    def filter_by_cohort(self, target_cohorts: List[str]) -> Dict[str, Any]:
        """
        Subsets session window data to only specified target cohort tags.
        """
        target_set = set(target_cohorts)
        subset_inputs = []
        subset_meta = []
        
        win_offset = 0
        for meta in self.session_window_meta:
            n_win = meta[0]
            group_label = meta[5]
            if group_label in target_set:
                subset_inputs.extend(self.all_win_inputs[win_offset : win_offset + n_win])
                subset_meta.append(meta)
            win_offset += n_win
            
        return {
            "window_inputs": subset_inputs,
            "session_meta": subset_meta
        }
