"""Render requested artifacts from validated, explicitly associated input data."""

from pathlib import Path

import numpy as np

from fpcode.common.config import load_yaml_config


def load_render_inputs(results_path, manifest_path=None):
    import h5py

    results_path = Path(results_path)
    if results_path.is_dir():
        results_path = results_path / "results.h5"
    results = {}
    with h5py.File(results_path, "r") as handle:
        for key, group in handle.items():
            if isinstance(group, h5py.Group) and "syllable" in group:
                results[key] = {name: item[()] for name, item in group.items() if isinstance(item, h5py.Dataset)}
    if not results:
        raise ValueError("Results contain no recording groups with syllable labels")
    for key, record in results.items():
        labels = np.asarray(record["syllable"])
        if labels.ndim != 1 or not labels.size or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
            raise ValueError(f"Invalid syllable labels for recording {key}")

    manifest = load_yaml_config(manifest_path) if manifest_path is not None else {}
    allowed = {"fps", "sampling_mode", "coordinates", "video_paths", "video_frame_indexes", "render_options"}
    unknown = set(manifest) - allowed
    if unknown:
        raise ValueError(f"Unknown render manifest settings: {', '.join(sorted(unknown))}")
    base = Path(manifest_path).resolve().parent if manifest_path is not None else results_path.parent

    def paths(name):
        mapping = manifest.get(name, {})
        if not isinstance(mapping, dict):
            raise ValueError(f"Render {name} must map recording names to paths")
        unknown_keys = set(mapping) - set(results)
        if unknown_keys:
            raise ValueError(f"Unknown recordings in {name}: {', '.join(sorted(unknown_keys))}")
        return {key: Path(path) if Path(path).is_absolute() else base / path for key, path in mapping.items()}

    coordinates = {key: record["coordinates"] for key, record in results.items() if "coordinates" in record}
    for key, path in paths("coordinates").items():
        coordinates[key] = np.load(path, allow_pickle=False)
    for key, value in coordinates.items():
        if value.ndim != 3 or value.shape[-1] not in (2, 3) or len(value) != len(results[key]["syllable"]):
            raise ValueError(f"Coordinates for {key} must have shape (label frames, keypoints, 2 or 3)")
    indexes = {key: record["video_frame_indexes"] for key, record in results.items() if "video_frame_indexes" in record}
    for key, path in paths("video_frame_indexes").items():
        indexes[key] = np.load(path, allow_pickle=False)
    for key, value in indexes.items():
        value = np.asarray(value)
        if value.ndim != 1 or len(value) != len(results[key]["syllable"]) or not np.issubdtype(value.dtype, np.integer) or (value < 0).any() or (np.diff(value) <= 0).any():
            raise ValueError(f"Invalid original video frame mapping for {key}")
    fps = float(manifest.get("fps", 40))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Render fps must be finite and positive")
    mode = manifest.get("sampling_mode", "density")
    if mode not in {"density", "random"}:
        raise ValueError("sampling_mode must be 'density' or 'random'")
    options = manifest.get("render_options", {})
    if not isinstance(options, dict):
        raise ValueError("render_options must be a mapping")
    return {"path": results_path, "results": results, "coordinates": coordinates,
            "video_paths": {key: str(path) for key, path in paths("video_paths").items()},
            "video_frame_indexes": indexes or None, "fps": fps, "sampling_mode": mode,
            "render_options": options}


def plan_rendering(inputs, render_type, grid_size, syllables=None):
    """Validate artifact requirements and eligible bouts without loading a model."""
    from fpcode.behavior.bouts import extract_bouts
    from fpcode.viz.grid_movies import exact_grid_layout

    if render_type not in {"grid_movies", "trajectories", "all"}:
        raise ValueError(f"Unsupported rendering type: {render_type}")
    rows, cols = exact_grid_layout(grid_size)
    results = inputs["results"]
    available = set(np.concatenate([record["syllable"] for record in results.values()]).tolist())
    requested = sorted(available if syllables is None else set(syllables))
    if not requested or not set(requested) <= available:
        raise ValueError(f"Requested syllables must exist in results; available: {sorted(available)}")
    if set(inputs["coordinates"]) != set(results):
        raise ValueError("Rendering requires coordinates for every recording in results or the render manifest")
    bouts = {key: extract_bouts(record["syllable"], min_length=3) for key, record in results.items()}
    if render_type in {"trajectories", "all"}:
        present = {bout.syllable for values in bouts.values() for bout in values}
        missing = set(requested) - present
        if missing and syllables is not None:
            raise ValueError(f"No bouts of at least 3 frames for syllables: {sorted(missing)}")
        requested = [key for key in requested if key in present]
    if render_type in {"grid_movies", "all"}:
        videos = inputs["video_paths"]
        if set(videos) != set(results) or any(not Path(path).is_file() for path in videos.values()):
            raise ValueError("Grid movies require a render manifest with video_paths for every recording")
        for key, record in results.items():
            if "centroid" not in record or "heading" not in record:
                raise ValueError(f"Grid movie results require centroid and heading arrays for {key}")
            if np.shape(record["centroid"]) != (len(record["syllable"]), 2) or np.shape(record["heading"]) != (len(record["syllable"]),):
                raise ValueError(f"Invalid centroid/heading shape for {key}")
        options = inputs["render_options"]
        pre = round(float(options.get("pre", 1.0)) * inputs["fps"])
        post = round(float(options.get("post", 2.0)) * inputs["fps"])
        minimum_duration = options.get("min_duration", 3)
        if pre < 0 or post < 0 or not isinstance(minimum_duration, int) or minimum_duration < 1:
            raise ValueError("Render context must be nonnegative and min_duration a positive integer")
        frequencies = {key: 0 for key in requested}
        counts = {key: 0 for key in requested}
        total = sum(len(record["syllable"]) for record in results.values())
        for key, record in results.items():
            for syllable in requested:
                frequencies[syllable] += np.count_nonzero(record["syllable"] == syllable)
            for bout in extract_bouts(record["syllable"], min_length=minimum_duration):
                if bout.syllable in counts and bout.onset >= pre and bout.onset < len(record["syllable"]) - post:
                    counts[bout.syllable] += 1
        minimum_frequency = float(options.get("min_frequency", .005))
        if not np.isfinite(minimum_frequency) or not 0 <= minimum_frequency <= 1:
            raise ValueError("min_frequency must be between zero and one")
        missing = [key for key in requested if counts[key] < rows * cols or frequencies[key] / total < minimum_frequency]
        if missing and syllables is not None:
            raise ValueError(f"Requested syllables lack enough eligible bouts for the grid: {missing}")
        requested = [key for key in requested if key not in missing]
    if not requested:
        raise ValueError("No syllables have eligible bouts for the requested rendering")
    return rows, cols, requested


def render_model_visualizations(results_path, render_type, grid_size, syllables=None, *, manifest=None, output_dir=None):
    inputs = load_render_inputs(results_path, manifest)
    rows, cols, requested = plan_rendering(inputs, render_type, grid_size, syllables)
    results = inputs["results"]
    output_dir = Path(output_dir) if output_dir is not None else inputs["path"].parent / "renders"
    files = []
    if render_type in {"grid_movies", "all"}:
        from fpcode.viz.grid_movies import generate_grid_movies

        summary = generate_grid_movies(
            results, output_dir=output_dir / "grid_movies", video_paths=inputs["video_paths"],
            coordinates=inputs["coordinates"], video_frame_indexes=inputs["video_frame_indexes"],
            fps=inputs["fps"], rows=rows, cols=cols, syllables=requested,
            sampling_mode=inputs["sampling_mode"], **inputs["render_options"],
        )
        if summary.get("error") or not summary.get("files"):
            raise RuntimeError(summary.get("error", "Grid movie backend produced no movies"))
        files.extend(summary["files"])
    if render_type in {"trajectories", "all"}:
        from fpcode.viz.trajectories import plot_syllable_trajectories

        if set(inputs["coordinates"]) != set(results):
            raise ValueError("Trajectory plots require coordinates for every recording in results or the render manifest")
        labels = {key: record["syllable"] for key, record in results.items()}
        for syllable in requested:
            files.append(str(plot_syllable_trajectories(inputs["coordinates"], labels, syllable,
                             output_path=output_dir / "trajectories" / f"syllable{syllable}.png")))
    if not files or any(not Path(path).is_file() or not Path(path).stat().st_size for path in files):
        raise RuntimeError("Rendering did not produce the requested nonempty artifacts")
    return {"output_dir": str(output_dir), "render_type": render_type, "grid_size": grid_size,
            "syllables": requested, "files": files}
