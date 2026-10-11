import numpy as np


class TraceAccumulator:
    def __init__(self, window_size):
        self.window_size = window_size
        self.count = 0
        self.sum = np.zeros(window_size, dtype=np.float64)
        self.sum_sq = np.zeros(window_size, dtype=np.float64)

    def add(self, trace):
        """
        Add a single trace or batch of traces (N, T).
        Trace must be of length window_size.
        """
        # Ensure numpy
        if not isinstance(trace, np.ndarray):
            trace = np.array(trace)

        if trace.ndim == 1:
            if len(trace) != self.window_size:
                raise ValueError(f"Trace length {len(trace)} != window {self.window_size}")
            self.sum += trace
            self.sum_sq += trace**2
            self.count += 1
        elif trace.ndim == 2:
            # Batch update
            N, T = trace.shape
            if T != self.window_size:
                 raise ValueError(f"Trace length {T} != window {self.window_size}")

            self.sum += np.sum(trace, axis=0)
            self.sum_sq += np.sum(trace**2, axis=0)
            self.count += N

    def get_stats(self):
        """
        Returns dictionary with mean, std, sem, ci_95
        """
        if self.count == 0:
            nan_arr = np.full(self.window_size, np.nan)
            return {
                "n": 0,
                "mean": nan_arr,
                "std": nan_arr,
                "sem": nan_arr,
                "ci_95": nan_arr
            }

        mean = self.sum / self.count

        # Variance = E[X^2] - (E[X])^2
        # Use abs to prevent floating point errors returning negative zeros
        var = np.abs((self.sum_sq / self.count) - (mean ** 2))
        std = np.sqrt(var)

        # SEM = Std / sqrt(N)
        sem = std / np.sqrt(self.count)

        # 95% CI = 1.96 * SEM (assuming Normal approx for Mean)
        ci_95 = 1.96 * sem

        return {
            "n": self.count,
            "mean": mean,
            "std": std,
            "sem": sem,
            "ci_95": ci_95
        }
