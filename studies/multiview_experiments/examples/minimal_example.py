import os

# Force EGL for offscreen rendering.
os.environ["PYOPENGL_PLATFORM"] = "egl"

from vispy import app, scene

app.use_app('egl')

import imageio
import numpy as np


def create_video_from_positions(positions, video_filename='rotation_video_precomputed.mp4',
                                fps=60, rotation_degrees=360, canvas_size=(800, 600), zoom_factor=0.5):
    """
    Create a video from a precomputed positions array.

    Parameters:
      positions: numpy.ndarray
          Array of shape (n_timesteps, n_points, 3) containing the positions.
      video_filename: str
          The output filename.
      fps: int
          Frames per second.
      rotation_degrees: float
          Total degrees of rotation over the video.
      canvas_size: tuple
          The size (width, height) of the offscreen canvas.
      zoom_factor: float
          Multiplier for the camera distance. Lower values zoom in further, higher values zoom out.
          Default is 1.0.
    """
    n_timesteps = positions.shape[0]

    # Normalize positions so that the minimum for each axis is 0.
    min_coords = positions.min(axis=(0, 1))
    positions = positions - min_coords

    # Calculate overall extent from the normalized positions.
    max_coords = positions.max(axis=(0, 1))
    overall_extent = np.max(max_coords)

    # Calculate the rotation per frame.
    angle_per_frame = rotation_degrees / n_timesteps

    # Create an offscreen SceneCanvas.
    canvas = scene.SceneCanvas(show=False, size=canvas_size, bgcolor='white')
    view = canvas.central_widget.add_view()

    # Center the camera at the overall mean of all points.
    # overall_center = positions.reshape(-1, 3).mean(axis=0)
    camera = scene.cameras.TurntableCamera(up='z', fov=45)
    # camera.center = overall_center
    camera.center = np.array([0.20, 0.20, 0])


    # Adjust camera distance for zoom level.
    # A lower zoom_factor brings the camera closer, effectively "zooming in."
    camera.distance = overall_extent * zoom_factor
    view.camera = camera

    # Create a Markers visual for the points.
    markers = scene.visuals.Markers()
    markers.set_data(positions[0], face_color='red', size=4)
    view.add(markers)

    # --- Add Axes using LineVisual ---
    axis_length = 3.0

    # X-axis (red)
    x_axis = scene.visuals.Line(
        pos=np.array([[0, 0, 0], [axis_length, 0, 0]], dtype=np.float64),
        color=(1, 0, 0, 1),
        width=4,
        connect='strip'
    )
    x_axis.parent = view.scene

    # Y-axis (green)
    y_axis = scene.visuals.Line(
        pos=np.array([[0, 0, 0], [0, axis_length, 0]], dtype=np.float64),
        color=(0, 1, 0, 1),
        width=4,
        connect='strip'
    )
    y_axis.parent = view.scene

    # Z-axis (blue)
    z_axis = scene.visuals.Line(
        pos=np.array([[0, 0, 0], [0, 0, axis_length]], dtype=np.float64),
        color=(0, 0, 1, 1),
        width=4,
        connect='strip'
    )
    z_axis.parent = view.scene
    # --- End axes ---

    # --- Add Grid Lines on the XY plane (z = 0) ---
    grid_color = (0.8, 0.8, 0.8, 1)
    grid_spacing = 0.4
    grid_extent = 1.2

    # Vertical grid lines.
    for x in np.arange(-grid_extent, grid_extent + grid_spacing, grid_spacing):
        line = scene.visuals.Line(
            pos=np.array([[x, -grid_extent, 0], [x, grid_extent, 0]], dtype=np.float64),
            color=grid_color,
            width=1,
            connect='strip'
        )
        line.parent = view.scene

    # Horizontal grid lines.
    for y in np.arange(-grid_extent, grid_extent + grid_spacing, grid_spacing):
        line = scene.visuals.Line(
            pos=np.array([[-grid_extent, y, 0], [grid_extent, y, 0]], dtype=np.float64),
            color=grid_color,
            width=1,
            connect='strip'
        )
        line.parent = view.scene
    # --- End grids ---

    # Create the video.
    with imageio.get_writer(video_filename, fps=fps) as writer:
        for i in range(n_timesteps):
            # Rotate the camera.
            camera.azimuth = i * angle_per_frame
            # Update marker positions.
            markers.set_data(positions[i], face_color='red', size=10)
            # Render the offscreen canvas.
            frame = canvas.render()
            writer.append_data(frame)
            if i % 30 == 0:
                print(f"Rendered timestep {i+1}/{n_timesteps}")

    canvas.close()
    print(f"Video saved as '{video_filename}'")


if __name__ == '__main__':
    # If run directly, use a built-in simulation with sine movement.
    fps = 30
    duration = 30  # seconds
    n_timesteps = fps * duration
    amplitude = 0.5
    frequency = 0.5
    base_points = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 0, 1]
    ], dtype=np.float64)
    n_points = base_points.shape[0]

    # Precompute positions with sine modulation along the z-axis.
    all_points = np.empty((n_timesteps, n_points, 3), dtype=np.float64)
    for i in range(n_timesteps):
        t = i / fps
        pts = base_points.copy()
        pts[:, 2] += amplitude * np.sin(2 * np.pi * frequency * t)
        all_points[i] = pts

    # Adjust the zoom_factor here to zoom in/out.
    create_video_from_positions(all_points, zoom_factor=0.8)
