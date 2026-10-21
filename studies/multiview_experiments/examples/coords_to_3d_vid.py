import time

import matplotlib.pyplot as plt
import numpy as np
from vispy import scene


def linear_interp_nans_1d(arr):
    """Interpolate linearly to fill in 1D NaNs."""
    n = arr.shape[0]
    x = np.arange(n)
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) == 0:
        return np.zeros(n)
    return np.interp(x, valid, arr[valid])

def fill_nans_linear(coords):
    """
    Fill NaN values in a coordinate array of shape (T, 27, 3)
    via linear interpolation along the time axis.
    """
    T, N, D = coords.shape
    filled = np.copy(coords)
    for i in range(N):
        for d in range(D):
            filled[:, i, d] = linear_interp_nans_1d(filled[:, i, d])
    return filled


def plot_three_frames(coords):
    """
    Plot three consecutive frames from a (T, 27, 3) coordinate array.
    This function sets up a VisPy SceneCanvas, creates marker and line visuals,
    updates them for three consecutive frames, renders each frame, and shows the result
    using matplotlib.
    """
    T, N, D = coords.shape

    # Compute global bounds using nan-aware functions.
    global_min = np.nanmin(coords.reshape(-1, 3), axis=0)
    global_max = np.nanmax(coords.reshape(-1, 3), axis=0)
    center = (global_min + global_max) / 2.0
    extent = global_max - global_min
    max_extent = np.max(extent)

    print("Global min:", global_min)
    print("Global max:", global_max)
    print("Center:", center)
    print("Max extent:", max_extent)

    # Define bodyparts (the order must match your coords) and a minimal skeleton.
    bodyparts = [
        'nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip',
        'left_eye', 'right_eye', 'neck', 'mid_back', 'mouse_center',
        'mid_backend', 'mid_backend2', 'mid_backend3', 'tail_base',
        'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'left_shoulder',
        'left_midside', 'left_hip', 'right_shoulder', 'right_midside',
        'right_hip', 'tail_end', 'head_midpoint'
    ]
    # For simplicity, here we define a minimal skeleton that connects a few keypoints.
    skeleton_names = [
        ['nose', 'head_midpoint'],
        ['head_midpoint', 'neck'],
        ['neck', 'mid_back'],
        ['mid_back', 'mouse_center'],
        ['mouse_center', 'tail_base']
    ]
    name_to_idx = {name: i for i, name in enumerate(bodyparts)}
    skeleton_edges = []
    for edge in skeleton_names:
        try:
            i = name_to_idx[edge[0]]
            j = name_to_idx[edge[1]]
            skeleton_edges.append([i, j])
        except KeyError:
            pass
    skeleton_edges = np.array(skeleton_edges)

    # Set up the VisPy scene.
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    canvas.bgcolor = 'white'
    view = canvas.central_widget.add_view()

    # Set up the camera.
    view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=30, elevation=30)
    view.camera.center = center
    # Zoom in closer; adjust the distance as needed.
    view.camera.distance = 5.0 * max_extent
    # Optionally, let the camera auto-compute a good range:
    # view.camera.set_range(x=(global_min[0], global_max[0]),
    #                       y=(global_min[1], global_max[1]),
    #                       z=(global_min[2], global_max[2]))

    # Create marker visual (for points).
    points = scene.visuals.Markers()
    points.set_data(coords[0], face_color='red', edge_color='black', size=50)
    view.add(points)

    # Create line visual (for the skeleton).
    lines = scene.visuals.Line(connect='segments', color='blue', width=5)
    view.add(lines)

    # Give the scene a moment to update.
    canvas.update()
    time.sleep(0.1)

    # Render and display three consecutive frames.
    start_idx = 100_000  # Change this index as needed
    rendered_images = []
    for idx in range(start_idx, start_idx + 3):
        current_coords = coords[idx]
        points.set_data(current_coords, face_color='red', edge_color='black', size=50)
        segs = np.empty((len(skeleton_edges)*2, 3))
        segs[0::2] = current_coords[skeleton_edges[:, 0]]
        segs[1::2] = current_coords[skeleton_edges[:, 1]]
        lines.set_data(segs)

        # Update the canvas and render the frame.
        canvas.update()
        # A short sleep can ensure the scene is fully updated.
        time.sleep(0.05)
        img = canvas.render(size=(800,600))
        rendered_images.append(img)

        # Display the frame using matplotlib.
        plt.figure(figsize=(8,6))
        plt.imshow(img)
        plt.title(f"Frame {idx}")
        plt.axis('off')
        plt.show()

    return rendered_images

# Example usage:
# Assuming you have your coords array of shape (T, 27, 3):
# coords = np.load('your_coords.npy')
# images = plot_three_frames(coords)

def plot_nose_coordinates(coords, sample_rate=100):
    T, N, D = coords.shape
    nose_coords = coords[::sample_rate, 0, :]
    time_axis = np.arange(0, T, sample_rate)

    plt.figure(figsize=(10,6))
    plt.plot(time_axis, nose_coords[:, 0], label='nose_x')
    plt.plot(time_axis, nose_coords[:, 1], label='nose_y')
    plt.plot(time_axis, nose_coords[:, 2], label='nose_z')
    plt.xlabel("Frame")
    plt.ylabel("Coordinate Value")
    plt.title(f"Nose Coordinates over Time (sampled every {sample_rate} frames)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_plot = "nose_coordinates.png"
    plt.savefig(output_plot)
    plt.show()
    print(f"Nose coordinate plot saved to {output_plot}")

# Example usage:
# Assuming you have a NumPy array 'coords' of shape (218227, 27, 3)
# coords = np.load('your_coords.npy')
# create_3d_vid(coords)


def minimal_example():
    # Create an artificial coordinate array of shape (3, 27, 3)
    # Here we generate coordinates uniformly between 0 and 100.
    T = 3    # 3 frames
    N = 27   # 27 bodyparts
    D = 3    # 3 dimensions (x, y, z)
    coords = np.random.uniform(0, 100, size=(T, N, D))

    # Compute global bounds.
    global_min = np.nanmin(coords.reshape(-1, 3), axis=0)
    global_max = np.nanmax(coords.reshape(-1, 3), axis=0)
    center = (global_min + global_max) / 2.0
    extent = global_max - global_min
    max_extent = np.max(extent)

    print("Global min:", global_min)
    print("Global max:", global_max)
    print("Center:", center)
    print("Max extent:", max_extent)

    # Define bodyparts names (here just dummy names "bp0", "bp1", ... "bp26")
    bodyparts = ['bp' + str(i) for i in range(27)]
    # Create a minimal skeleton: simply connect each bodypart to the next one.
    skeleton_edges = np.array([[i, i+1] for i in range(26)])

    # --- Set up the VisPy scene.
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    canvas.bgcolor = 'white'
    view = canvas.central_widget.add_view()

    # Set up a TurntableCamera.
    view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=30, elevation=30)
    view.camera.center = center
    # Increase the distance so the entire spread is visible.
    view.camera.distance = 5.0 * max_extent
    # Optionally set the view range exactly.
    view.camera.set_range(x=(global_min[0], global_max[0]),
                          y=(global_min[1], global_max[1]),
                          z=(global_min[2], global_max[2]))

    # --- Create visuals.
    # Markers for points.
    points = scene.visuals.Markers()
    # Set initial data for frame 0 with a large size for visibility.
    points.set_data(coords[0], face_color='red', edge_color='black', size=50)
    view.add(points)

    # Lines for the skeleton.
    lines = scene.visuals.Line(connect='segments', color='blue', width=5)
    view.add(lines)

    # Give the scene a moment to update.
    canvas.update()
    time.sleep(0.1)

    # Render three consecutive frames and display each.
    for idx in range(T):
        current_coords = coords[idx]
        # Update marker positions.
        points.set_data(current_coords, face_color='red', edge_color='black', size=50)

        # Build skeleton line segments.
        segs = np.empty((skeleton_edges.shape[0]*2, 3))
        segs[0::2] = current_coords[skeleton_edges[:, 0]]
        segs[1::2] = current_coords[skeleton_edges[:, 1]]
        lines.set_data(segs)

        canvas.update()
        time.sleep(0.1)
        # Render the scene.
        img = canvas.render(size=(800,600))

        # Display the frame using matplotlib.
        plt.figure(figsize=(8,6))
        plt.imshow(img)
        plt.title(f"Frame {idx}")
        plt.axis('off')
        plt.show()

    return coords  # Return the artificial data (for debugging if needed)

# Run the minimal example.
