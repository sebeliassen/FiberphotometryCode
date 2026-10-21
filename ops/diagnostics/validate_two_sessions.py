"""Explicit, bounded KPMS training validation on two real session excerpts.

Reads an existing project, fits 2 AR + 2 SLDS updates in a fresh process, and
keeps evidence under outputs/<run-id>. This never runs in the default suite.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def select_two_sessions(data, metadata, frames):
    """Copy valid prefixes from different sessions without joining fragments."""
    import numpy as np

    if not isinstance(data, dict) or not {"Y", "mask", "conf"} <= data.keys():
        raise ValueError("Preprocessed input requires Y, mask and conf arrays")
    if not isinstance(metadata, (tuple, list)) or len(metadata) != 2:
        raise ValueError("Metadata must contain recording names and segment bounds")
    names, bounds = np.asarray(metadata[0]), np.asarray(metadata[1])
    shape = data["Y"].shape
    if (len(shape) != 4 or shape[-1] != 2 or names.shape != (shape[0],)
            or bounds.shape != (shape[0], 2) or bounds.dtype.kind not in "iu"
            or data["mask"].shape != shape[:2] or data["conf"].shape != shape[:3]):
        raise ValueError("Inconsistent two-dimensional observation, confidence or metadata shapes")
    selected, sessions = [], set()
    for index, (name, bound) in enumerate(zip(names, bounds)):
        name = name.decode() if isinstance(name, bytes) else str(name)
        base = re.sub(r"_seg\d+(?=_|$)", "", name)
        if (base in sessions or bound[0] < 0 or bound[1] < bound[0]
                or bound[1] - bound[0] < frames or shape[1] < frames):
            continue
        prefix = slice(0, frames)
        if (not np.all(np.asarray(data["mask"][index, prefix]) == 1)
                or not np.isfinite(np.asarray(data["Y"][index, prefix])).all()
                or not np.isfinite(np.asarray(data["conf"][index, prefix])).all()):
            continue
        selected.append(index)
        sessions.add(base)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise ValueError(f"Need two distinct sessions with {frames} valid prefix frames each")
    sample = {key: np.stack([np.asarray(data[key][index, :frames]).copy() for index in selected])
              for key in ("Y", "mask", "conf")}
    sample_bounds = bounds[selected].copy()
    sample_bounds[:, 1] = sample_bounds[:, 0] + frames
    sample_names = np.asarray([names[index].decode() if isinstance(names[index], bytes)
                               else str(names[index]) for index in selected])
    identities = [hashlib.sha256(name.encode()).hexdigest()[:12] for name in sorted(sessions)]
    return sample, (sample_names, sample_bounds), identities


def _signatures(paths):
    return {str(path): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                       "inode": path.stat().st_ino} for path in paths}


def validate(args, directory):
    # Imports and runtime configuration are owned by this fresh worker process.
    import gc

    import h5py
    import jax
    import jax.numpy as jnp
    import keypoint_moseq as kpms
    import numpy as np
    import yaml

    from fpcode.common.cache_manifest import atomic_write_json
    from fpcode.kpms.backend.patch_registry import temporary_patches
    from fpcode.kpms.checkpoints import patched_load_checkpoint
    from fpcode.kpms.evaluation import evaluate_checkpoints, read_checkpoint_identity
    from fpcode.kpms.results import extract_and_export_results
    from fpcode.workflows.training import fit_model

    started = time.monotonic()
    source = args.project.resolve()
    inputs = [source / name for name in ("preprocessed_data.pkl", "config.yml", "pca.p")]
    before = _signatures(inputs)
    report = {"status": "RUNNING", "source_inputs": before, "requested_device": args.device,
              "seed": args.seed, "frames_per_session": args.frames, "distinct_sessions": 2}
    atomic_write_json(directory / "validation.json", report)
    try:
        expected_platform = "gpu" if args.device == "cuda" else "cpu"
        if jax.default_backend() != expected_platform or not jax.config.jax_enable_x64:
            raise RuntimeError(f"Expected float64 {expected_platform} backend; found {jax.devices()}")
        matrix = jnp.arange(16, dtype=jnp.float64).reshape(4, 4)
        answer = jax.jit(lambda values: values @ values.T)(matrix)
        answer.block_until_ready()
        np.testing.assert_allclose(answer, np.arange(16).reshape(4, 4) @ np.arange(16).reshape(4, 4).T)
        if any(device.platform != expected_platform for device in answer.devices()):
            raise RuntimeError("The compiled validation calculation used the wrong device")
        if before[str(inputs[0])]["size"] > args.max_source_mib * 1024**2:
            raise ValueError("Preprocessed input exceeds the explicit size limit; use a pilot project")
        with inputs[0].open("rb") as handle:
            loaded = pickle.load(handle)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            data, metadata = loaded
        elif isinstance(loaded, dict) and {"data", "metadata"} <= loaded.keys():
            data, metadata = loaded["data"], loaded["metadata"]
        else:
            raise ValueError("Preprocessed pickle must contain data and metadata")
        sample, sample_metadata, identities = select_two_sessions(data, metadata, args.frames)
        del loaded, data, metadata
        gc.collect()
        with (directory / "preprocessed_data.pkl").open("wb") as handle:
            pickle.dump((sample, sample_metadata), handle)
        shutil.copyfile(inputs[2], directory / "pca.p")
        settings = yaml.safe_load(inputs[1].read_text())
        settings["verbose"] = False
        (directory / "config.yml").write_text(yaml.safe_dump(settings))
        cfg = dict(num_ar_iters=2, num_slds_iters=2, save_every_n_iters=1,
                   use_memory_patch=True, use_batched_init=True, location_aware=False,
                   init_kappa=1e6, slds_kappa=1.5e15,
                   data_path=str(directory / "preprocessed_data.pkl"), pca_path=str(directory / "pca.p"))
        atomic_write_json(directory / "training_request.json", cfg)
        original = kpms.fitting._wrapped_resample
        loaders = (kpms.load_checkpoint, kpms.io.load_checkpoint, kpms.fitting.load_checkpoint)
        updates = []

        def observe(function, observations, model, *positional, **options):
            result = original(function, observations, model, *positional, **options)
            resident = sum(isinstance(value, jax.Array) and any(
                device.platform == expected_platform for device in value.devices()
            ) for value in jax.tree_util.tree_leaves(result))
            if not resident:
                raise RuntimeError("A completed update has no arrays on the requested device")
            updates.append({"ar_only": options.get("ar_only"), "resident_arrays": resident})
            return result

        with temporary_patches([(kpms.fitting, "_wrapped_resample", observe)]):
            checkpoint = fit_model(cfg, directory / "model", seed=args.seed, device=args.device)
        if (kpms.fitting._wrapped_resample is not original or
                (kpms.load_checkpoint, kpms.io.load_checkpoint, kpms.fitting.load_checkpoint) != loaders):
            raise RuntimeError("Training failed to restore backend patches")
        if [entry["ar_only"] for entry in updates] != [True, True, False, False]:
            raise RuntimeError(f"Expected exactly 2 AR + 2 SLDS updates: {updates}")
        identity = read_checkpoint_identity(checkpoint)
        if (identity["iteration"] != 4 or identity["n_recordings"] != 2
                or identity["active_frames"] != 2 * args.frames
                or identity["iterations"] != [0, 1, 2, 3, 4]):
            raise RuntimeError(f"Unexpected checkpoint endpoint or dataset: {identity}")
        with h5py.File(checkpoint, "r") as handle:
            def check_finite(name, dataset):
                if isinstance(dataset, h5py.Dataset) and dataset.dtype.kind in "fiu":
                    if not np.isfinite(dataset[()]).all():
                        raise RuntimeError(f"Nonfinite checkpoint values: {name}")
            handle["model_snapshots/4/states"].visititems(check_finite)
            handle["model_snapshots/4/params"].visititems(check_finite)
        checkpoint_before = _signatures([checkpoint])
        evaluated = evaluate_checkpoints([checkpoint])
        if next(iter(evaluated.values()))["evaluated_frames"] != 2 * (args.frames - identity["nlags"]):
            raise RuntimeError("Evaluation did not respect the AR lag offset")
        model, _, stored_metadata, iteration = patched_load_checkpoint(path=str(checkpoint))
        extracted = extract_and_export_results(model, stored_metadata, directory, "model", save_results=True)
        if iteration != 4 or set(extracted) != set(sample_metadata[0]):
            raise RuntimeError("Reload or extraction changed recording identity")
        for result in extracted.values():
            if len(result["syllable"]) != args.frames or not np.isfinite(result["latent_state"]).all():
                raise RuntimeError("Extracted results have incorrect length or nonfinite states")
        if _signatures([checkpoint]) != checkpoint_before:
            raise RuntimeError("Evaluation or extraction modified the source checkpoint")
        report.update(status="PASS", device=str(jax.devices()[0]), jax=jax.__version__,
                      session_hashes=identities, observation_shape=list(sample["Y"].shape),
                      ar_updates=2, slds_updates=2, observed_updates=updates,
                      checkpoint_identity=identity, evaluation=evaluated,
                      checkpoint_unchanged_during_evaluation=True)
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["source_files_unchanged"] = _signatures(inputs) == before
        if not report["source_files_unchanged"]:
            report.update(status="FAIL", error="Original project artifacts changed")
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        atomic_write_json(directory / "validation.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(report["error"])
    print(f"PASS: two sessions, {args.frames} frames each, 2 AR + 2 SLDS on {report['device']}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="Existing pilot KPMS project, read-only")
    parser.add_argument("--output-root", type=Path, help="Defaults to repository outputs/")
    parser.add_argument("--frames", type=int, default=256, help="Prefix frames per session (8..2048)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--timeout", type=float, default=180, help="Maximum fresh worker runtime in seconds")
    parser.add_argument("--max-source-mib", type=int, default=1024, help="Maximum input pickle size before loading")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 8 <= args.frames <= 2048 or args.seed < 0 or args.max_source_mib <= 0:
        parser.error("Require 8..2048 frames, a nonnegative seed and a positive input size limit")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    if args.worker_output:
        validate(args, args.worker_output)
        return 0
    from fpcode.common.cache_manifest import atomic_write_json
    from fpcode.common.paths import get_repo_root

    source = args.project.resolve()
    for name in ("preprocessed_data.pkl", "config.yml", "pca.p"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"Required input missing: {source / name}")
    if (source / "preprocessed_data.pkl").stat().st_size > args.max_source_mib * 1024**2:
        raise ValueError("Input exceeds the size limit; select a pilot project")
    output = (args.output_root or get_repo_root() / "outputs").resolve()
    if output == source or source in output.parents:
        raise ValueError("Validation output must be outside the original project")
    directory = output / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                          + f"-two-session-validation-{uuid.uuid4().hex[:8]}")
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, directory / "validation_script.py")
    environment = {**os.environ, "JAX_PLATFORMS": "cuda" if args.device == "cuda" else "cpu",
                   "JAX_ENABLE_X64": "true", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   "OMP_NUM_THREADS": "2", "NUMBA_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
    if args.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--project", str(source),
               "--frames", str(args.frames), "--device", args.device, "--seed", str(args.seed),
               "--max-source-mib", str(args.max_source_mib), "--worker-output", str(directory)]
    print(f"Validation evidence: {directory}", flush=True)
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            result = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            atomic_write_json(directory / "validation.json", {"status": "TIMEOUT", "timeout": args.timeout})
            print(f"FAIL: validation exceeded {args.timeout:g} seconds; see worker.log", file=sys.stderr)
            return 1
    record = directory / "validation.json"
    if result and (not record.is_file() or json.loads(record.read_text()).get("status") == "RUNNING"):
        atomic_write_json(record, {"status": "FAIL", "worker_exit_code": result})
    print(f"{'PASS' if result == 0 else 'FAIL'}: see {record}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
