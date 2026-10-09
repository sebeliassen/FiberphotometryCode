def generate_grid_movies(
    results,
    project_dir=None,
    model_name=None,
    output_dir=None,
    video_dir=None,
    video_paths=None,
    video_frame_indexes=None,
    pre=1.0,
    post=2.0,
    rows=4,
    cols=6,
    filter_size=9,
    min_frequency=0.005,
    min_duration=3,
    dot_radius=4,
    dot_color=(255, 255, 255),
    quality=7,
    window_size=None,
    coordinates=None,
    centroids=None,
    headings=None,
    bodyparts=None,
    use_bodyparts=None,
    sampling_options={},
    video_extension=None,
    max_video_size=1920,
    skeleton=[],
    overlay_keypoints=False,
    keypoints_only=False,
    keypoints_scale=1,
    fps=None,
    plot_options={},
    use_dims=[0, 1],
    keypoint_colormap="autumn",
    **kwargs,
):
    """Generate grid movies for a modeled dataset.

    Grid movies show many instances of a syllable and are useful in
    figuring out what behavior the syllable captures
    (see :py:func:`keypoint_moseq.viz.grid_movie`). This method
    generates a grid movie for each syllable that is used sufficiently
    often (i.e. has at least `rows*cols` instances with duration
    of at least `min_duration` and an overall frequency of at least
    `min_frequency`). The grid movies are saved to `output_dir` if
    specified, or else to `{project_dir}/{model_name}/grid_movies`.
    A subset of parameters are documented below. See
    :py:func:`keypoint_moseq.viz.grid_movie` for the remaining parameters.

    Parameters
    ----------
    results: dict
        Dictionary containing modeling results for a dataset (see
        :py:func:`keypoint_moseq.fitting.extract_results`)

    project_dir: str, default=None
        Project directory. Required to save grid movies if `output_dir`
        is None.

    model_name: str, default=None
        Name of the model. Required to save grid movies if
        `output_dir` is None.

    output_dir: str, default=None
        Directory where grid movies should be saved. If None, grid
        movies will be saved to `{project_dir}/{model_name}/grid_movies`.

    video_dir: str, default=None
        Directory containing videos of the modeled data (see
        :py:func:`keypoint_moseq.io.find_matching_videos`). Either  `video_dir`
        or `video_paths` must be provided unless `keypoints_only=True`.

    video_paths: dict, default=None
        Dictionary mapping recording names to video paths. The recording
        names must correspond to keys in the results dictionary. Either
        `video_dir` or `video_paths` must be provided unless
        `keypoints_only=True`.

    video_frame_indexes: dict, default=None
        Dictionary mapping recording names to arrays of video frame indexes.
        This is useful when the original keypoint coordinates used for modeling
        corresponded to a subset of frames from each video (i.e. if videos were
        trimmed or coordinates were downsampled).

    filter_size: int, default=9
        Size of the median filter applied to centroids and headings

    min_frequency: float, default=0.005
        Minimum frequency of a syllable to be included in the grid movies.

    min_duration: int, default=3
        Minimum duration of a syllable instance to be included in the
        grid movie for that syllable.

    sampling_options: dict, default={}
        Dictionary of options for sampling syllable instances (see
        :py:func:`keypoint_moseq.util.sample_instances`).

    coordinates: dict, default=None
