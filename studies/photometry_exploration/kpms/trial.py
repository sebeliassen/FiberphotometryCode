class DFStore(dict):
    """A dict that only accepts pandas-like DataFrames as values."""
    @staticmethod
    def _looks_like_pandas_df(obj):
        # Heuristic: works for pandas DataFrame without importing pandas
        return hasattr(obj, "columns") and hasattr(obj, "dtypes") and hasattr(obj, "shape")

    def __setitem__(self, key, value):
        if not self._looks_like_pandas_df(value):
            raise TypeError("dfs can only contain pandas DataFrames.")
        super().__setitem__(str(key), value)

    def update(self, mapping=(), **kwargs):
        items = dict(mapping, **kwargs).items()
        for k, v in items:
            self[k] = v  # routes through __setitem__ for validation


class Trial:
    """
    Simple container for a single trial.

    Attributes:
        base_dir (str): Base directory for the trial.
        trial_id (str): Identifier of the trial.
        mouse_id (str): Identifier of the mouse.
        dfs (DFStore): Dict-like store that accepts DataFrames only.
        tracking (str): Path to a .csv tracking file.
        video (str): Path to a .mp4 video file.
    """

    def __init__(self, base_dir, trial_id, mouse_id, tracking=None, video=None):
        self.base_dir = str(base_dir)
        self.trial_id = str(trial_id)
        self.mouse_id = str(mouse_id)

        # --- Add more attributes here if needed (e.g., "session_date", "genotype") ---

        self.dfs = DFStore()
        self.tracking = None
        self.video = None

        if tracking is not None:
            self.set_tracking(tracking)
        if video is not None:
            self.set_video(video)

    def set_tracking(self, path):
        path = str(path)
        if not path.lower().endswith(".csv"):
            raise ValueError("tracking must be a path to a .csv file.")
        self.tracking = path
        return self

    def set_video(self, path):
        path = str(path)
        if not path.lower().endswith(".mp4"):
            raise ValueError("video must be a path to a .mp4 file.")
        self.video = path
        return self

    # Convenience helpers
    def put_df(self, name, df):
        """Insert a single DataFrame into dfs."""
        self.dfs[name] = df
        return self

    def put_dfs(self, mapping):
        """Insert many DataFrames at once: {'name': df, ...}."""
        self.dfs.update(mapping)
        return self

    def __getitem__(self, name):
        return self.dfs[name]

    def __repr__(self):
        keys = ", ".join(sorted(self.dfs.keys()))
        return (
            f"Trial(base_dir={self.base_dir!r}, trial_id={self.trial_id!r}, "
            f"mouse_id={self.mouse_id!r}, dfs=[{keys}], "
            f"tracking_set={self.tracking is not None}, video_set={self.video is not None})"
        )
