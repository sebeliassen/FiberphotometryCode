"""Read-only environment and explicit-input diagnostics; no backend initialization."""

import argparse
import json
import platform
import sys
from pathlib import Path

from fpcode.common.environment import initialized_devices, package_versions
from fpcode.common.paths import get_data_root, get_repo_root, resolve_data_path, resolve_path

REQUIREMENTS = {
    "cpu": ("numpy", "scipy", "pandas", "h5py", "matplotlib", "PyYAML", "tables",
            "pytest", "scikit-learn", "Pillow", "imageio", "imageio-ffmpeg", "ruff"),
    "kpms": ("jax", "jaxlib", "jax-moseq", "keypoint-moseq", "numba", "psutil", "dynamax"),
    "render": ("numpy", "h5py", "matplotlib", "Pillow", "imageio", "imageio-ffmpeg"),
}


def diagnose(require=(), inputs=(), data_inputs=()):
    names = sorted({name for group in REQUIREMENTS.values() for name in group}
                   | {"fiberphotometry-suite", "opencv-python", "opencv-python-headless"})
    versions = package_versions(names)
    issues = []
    for group in require:
        for name in REQUIREMENTS[group]:
            if versions[name] is None:
                issues.append(f"Missing {group} distribution: {name}")
        if group in {"cpu", "render"} and not any(versions[n] for n in ("opencv-python", "opencv-python-headless")):
            issues.append(f"Missing {group} distribution: OpenCV (opencv-python or opencv-python-headless)")
    files = []
    for values, resolver, kind in ((inputs, resolve_path, "repository"), (data_inputs, resolve_data_path, "data")):
        for value in values:
            path = resolver(value)
            exists = path.exists()
            files.append({"requested": str(value), "path": str(path), "root": kind, "exists": exists})
            if not exists:
                issues.append(f"Missing input: {path}")
    return {
        "schema_version": 1, "ok": not issues, "read_only": True,
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "roots": {"repository": str(get_repo_root()), "data_mirror": str(get_data_root()),
                  "data_path_example": str(resolve_data_path("data/models")),
                  "convention": "FP_DATA_ROOT is a mirror root; relative data arguments include data/."},
        "packages": versions, "backend": initialized_devices(), "inputs": files,
        "requirements_checked": list(require), "issues": issues,
        "limits": ["Installed versions do not prove compatibility or GPU readiness.",
                   "Only explicitly named inputs are checked; files are not decoded or unpickled.",
                   "No dependency installation, directory creation or backend initialization is performed."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable report")
    parser.add_argument("--require", choices=tuple(REQUIREMENTS), action="append", default=[])
    parser.add_argument("--input", type=Path, action="append", default=[], help="Input relative to repository root, or absolute")
    parser.add_argument("--data-input", type=Path, action="append", default=[], help="Input relative to FP_DATA_ROOT, including data/, or absolute")
    args = parser.parse_args(argv)
    report = diagnose(args.require, args.input, args.data_input)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Interpreter: {report['python']['executable']}")
        print(f"Repository root: {report['roots']['repository']}")
        print(f"Data mirror root: {report['roots']['data_mirror']}")
        print(report["roots"]["convention"])
        print("Checks passed." if report["ok"] else "\n".join(report["issues"]))
        print("Dependency presence is metadata-only; optional backends were not initialized.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
