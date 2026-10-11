"""Bounded manual checks of real KPMS fitting/rendering on tiny synthetic inputs.

Run with the project's KPMS Python environment. This is deliberately outside
the default pytest suite. A fresh CPU-only subprocess owns all backend patches;
its temporary checkpoint/videos are removed when the check finishes.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def check_training(directory):
    from types import SimpleNamespace

    import h5py
    import jax
    import keypoint_moseq as kpms
    import numpy as np

    from fpcode.kpms.backend.patching import kpms_optimizations
    from fpcode.kpms.init_model import batched_init_model_by_video
    from fpcode.kpms.training import _fit_phase

    assert jax.default_backend() == "cpu"
    observations = np.random.default_rng(21).normal(scale=0.2, size=(2, 16, 3, 2))
    observations += np.array([[10., 0.], [0., 4.], [-10., 0.]])
    observations += np.arange(16)[None, :, None, None] * 0.05
    data = {"Y": observations, "mask": np.ones((2, 16)), "conf": np.full((2, 16, 3), .9)}
    options = dict(
        anterior_idxs=[0], posterior_idxs=[2], PCA_fitting_num_frames=32,
        error_estimator={"slope": -.5, "intercept": .25},
        obs_hypparams={"sigmasq_0": .1, "sigmasq_C": .1, "nu_sigma": 1e5, "nu_s": 5},
        ar_hypparams={"latent_dim": 2, "nlags": 1, "S_0_scale": .01, "K_0_scale": 10.},
        trans_hypparams={"num_states": 2, "gamma": 10., "alpha": 5.7, "kappa": 10.},
        cen_hypparams={"sigmasq_loc": .5},
    )
    model = batched_init_model_by_video(data, None, options, params_subset_size=1, batch_size=1, seed=21)
    assert model["states"]["x"].shape == (2, 16, 2)
    assert np.asarray(model["noise_prior"]).shape == (2, 16, 3)
    metadata = (np.array(["tiny-a", "tiny-b"]), np.array([[0, 16], [0, 16]]))
    backend = SimpleNamespace(kpms=kpms)
    checkpoint = directory / "tiny" / "checkpoint.h5"
    original = kpms.fitting._wrapped_resample
    with kpms_optimizations():
        for current, target, ar_only in ((0, 1, True), (1, 2, False)):
            model = _fit_phase(
                backend, model, data, metadata, directory, checkpoint, current=current, target=target,
                model_name="tiny", ar_only=ar_only, save_every_n_iters=1,
                generate_progress_plots=False, parallel_message_passing=False,
            )
    assert kpms.fitting._wrapped_resample is original
    assert all(np.isfinite(np.asarray(value)).all() for value in model["states"].values())
    with h5py.File(checkpoint, "r") as handle:
        assert set(handle["model_snapshots"]) == {"0", "1", "2"}
    print("PASS: real batched initialization, optimized AR/SLDS updates, checkpoints and patch cleanup", flush=True)


def check_rendering(directory):
    import cv2
    import numpy as np

    from fpcode.viz.grid_movies import generate_grid_movies

    video = directory / "tiny.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 64))
    assert writer.isOpened(), "The installed OpenCV backend cannot encode the synthetic source video"
    try:
        for index in range(96):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.circle(frame, (32 + index % 5, 32), 4, (100, 220, 40), -1)
            writer.write(frame)
    finally:
        writer.release()
    centroids = np.full((96, 2), 32.)
    coordinates = centroids[:, None, :].copy()
    before = coordinates.copy()
    result = generate_grid_movies(
        {"tiny": {"syllable": np.tile(np.repeat([0, 1], 8), 6), "centroid": centroids,
                  "heading": np.zeros(96)}}, output_dir=directory / "movies",
        video_paths={"tiny": str(video)}, coordinates={"tiny": coordinates}, fps=20,
        rows=1, cols=1, syllables=[1], sampling_mode="random", min_duration=3,
        pre=.1, post=.3, window_size=32, max_video_size=64,
    )
    assert result["movies_count"] == 1, result
    assert set(Path(result["output_dir"]).glob("*.mp4")) == {directory / "movies/syllable1.mp4"}
    capture = cv2.VideoCapture(result["files"][0])
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
        ok, first = capture.read()
        assert ok and first.shape[:2] == (32, 32)
    finally:
        capture.release()
    np.testing.assert_array_equal(coordinates, before)
    print("PASS: real subset sampling, grid generation, MP4 encoding and eight-frame decode", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=("training", "rendering", "all"), default="all")
    parser.add_argument("--timeout", type=float, default=60., help="Maximum worker runtime in seconds")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 0 < args.timeout < float("inf"):
        parser.error("timeout must be finite and positive")
    if not args.worker:
        environment = {**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "",
                       "JAX_ENABLE_X64": "true", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                       "OMP_NUM_THREADS": "2", "NUMBA_NUM_THREADS": "2"}
        try:
            return subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--check", args.check],
                env=environment, timeout=args.timeout, check=False,
            ).returncode
        except subprocess.TimeoutExpired:
            print(f"FAIL: runtime smoke check exceeded {args.timeout:g} seconds", file=sys.stderr)
            return 1
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    with tempfile.TemporaryDirectory(prefix="fpcode-runtime-smoke-") as location:
        directory = Path(location)
        if args.check in {"training", "all"}:
            check_training(directory)
        if args.check in {"rendering", "all"}:
            check_rendering(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
