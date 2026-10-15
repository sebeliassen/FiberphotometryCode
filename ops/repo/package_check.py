"""Build a source-only wheel and check it in a temporary isolated installation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
RESOURCE_NAMES = ("static_arena_transforms.json", "epoch_arena_transforms.json", "ttl_cohort_offsets.json")


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> int:
    env = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_NO_INDEX="1", PIP_CONFIG_FILE=os.devnull,
    )
    env.pop("PYTHONPATH", None)
    env.pop("FIBERPHOTOMETRY_ROOT", None)
    env.pop("FP_DATA_ROOT", None)
    with tempfile.TemporaryDirectory(prefix="fpcode-wheel-") as temporary:
        workspace = Path(temporary)
        source = workspace / "source"
        source.mkdir()
        shutil.copytree(ROOT / "src/fpcode", source / "src/fpcode", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for filename in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(ROOT / filename, source / filename)
        wheels = workspace / "wheels"
        run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheels), str(source)],
            cwd=workspace, env=env,
        )
        wheel, = wheels.glob("*.whl")
        with ZipFile(wheel) as archive:
            names = set(archive.namelist())
            assert "fpcode/py.typed" in names, "Type marker missing from wheel"
            for module in ("contracts/behavior_units.py", "behavior/unit_adapters.py", "behavior/compatibility.py",
                           "behavior/observations/kinematics.py", "behavior/primitives/bouts.py",
                           "behavior/episodes/seqnmf.py", "behavior/analysis/unit_views.py",
                           "behavior/episodes/staged.py", "behavior/assemblies/grammar.py", "behavior/context/models.py"):
                assert "fpcode/" + module in names, f"Behaviour layer module missing from wheel: {module}"
            for name in RESOURCE_NAMES:
                assert f"fpcode/compat/multiview/resources/{name}" in names, f"Packaged calibration resource missing: {name}"
        installation = workspace / "venv"
        # Reuse installed scientific dependencies without installing/upgrading them.
        # The package itself is installed into this venv and its origin is checked.
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(installation)
        python = installation / "bin/python"
        run([str(python), "-m", "pip", "install", "--no-deps", "--ignore-installed", str(wheel)], cwd=workspace, env=env)
        check = "\n".join((
            "from pathlib import Path",
            "from importlib.resources import files",
            "import json, sys, fpcode",
            f"assert Path(fpcode.__file__).is_relative_to(Path({str(installation)!r}))",
            "from fpcode.tracking.geometry import rot2d",
            "assert rot2d(0).shape == (2, 2)",
            "from fpcode.contracts import FrameSpan, SessionKey, UnitOccurrence, UnitStream, UnitType",
            "from fpcode.behavior.analysis.unit_views import measure_unit_organisation",
            "from fpcode.behavior.episodes.seqnmf import SeqNMF",
            "from fpcode.behavior.sequences.seqnmf import SeqNMF as old_seqnmf",
            "from fpcode.behavior.context import models",
            "assert SeqNMF is old_seqnmf and models.IMPLEMENTATION_STATUS == 'implemented'",
            "stream = UnitStream(SessionKey('synthetic'), 'episode', 'model', 'video', 10.,",
            "                    (FrameSpan(0, 20),), (UnitType('a'),),",
            "                    (UnitOccurrence('one', 'a', FrameSpan(0, 10)), UnitOccurrence('two', 'a', FrameSpan(10, 20))))",
            "assert measure_unit_organisation(stream, gap_seconds=0)['counts']['self_pairs'] == 1",
            "resources = files('fpcode.compat.multiview').joinpath('resources')",
            f"for name in {RESOURCE_NAMES!r}:",
            "    assert json.loads(resources.joinpath(name).read_text())",
            "assert not {'jax', 'keypoint_moseq', 'studies', 'syllable_generation'} & set(sys.modules)",
        ))
        run([str(python), "-I", "-B", "-c", check], cwd=workspace, env=env)
        for verb in ("train", "evaluate", "fuse", "render", "export", "doctor"):
            run([str(installation / f"bin/fpcode-{verb}"), "--help"], cwd=workspace, env=env)
        diagnostic = subprocess.run([str(installation / "bin/fpcode-doctor"), "--json"],
                                    cwd=workspace, env=env, check=True, capture_output=True, text=True)
        assert json.loads(diagnostic.stdout)["backend"]["inspection"] == "not_initialized"
        print("Wheel resources/type marker, canonical import, six CLI help commands and read-only doctor passed outside checkout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
