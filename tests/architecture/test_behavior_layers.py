"""Behavior unit choices cannot acquire dependencies through legacy reexports."""

import pytest

from ops.repo.dependencies import audit


def _package(tmp_path, modules):
    package = tmp_path / "fpcode"
    for name, source in modules.items():
        path = package / (name.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return package


@pytest.mark.parametrize(("layer", "allowed"), [
    ("observations", ("observations",)),
    ("primitives", ("observations", "primitives")),
    ("episodes", ("observations", "primitives", "episodes")),
    ("assemblies", ("observations", "assemblies")),
    ("context", ("observations", "primitives", "context")),
    ("analysis", ("observations", "primitives", "analysis")),
])
def test_behavior_layer_has_explicit_lower_dependencies(tmp_path, layer, allowed):
    modules = {f"behavior.{target}.support": "" for target in allowed}
    modules.update({
        "contracts.units": "", "common.infrastructure": "", "alignment.clocks": "",
        "statistics.summary": "", "behavior.unit_adapters": "", "behavior.compatibility": "",
    })
    modules[f"behavior.{layer}.choice"] = "\n".join(
        [f"import fpcode.behavior.{target}.support" for target in allowed]
        + ["import fpcode.contracts.units", "import fpcode.common.infrastructure",
           "import fpcode.alignment.clocks", "import fpcode.statistics.summary",
           "import fpcode.behavior.unit_adapters", "import fpcode.behavior.compatibility"]
    )
    report = audit(_package(tmp_path, modules))
    assert report["violations"] == []
    assert report["cycles"] == []


@pytest.mark.parametrize(("source", "target"), [
    ("observations", "primitives"), ("primitives", "episodes"),
    ("episodes", "assemblies"), ("episodes", "analysis"),
    ("episodes", "context"), ("assemblies", "episodes"),
    ("assemblies", "primitives"), ("context", "episodes"),
    ("context", "assemblies"), ("analysis", "episodes"),
    ("analysis", "assemblies"), ("analysis", "context"),
])
def test_behavior_layer_fences_cover_relative_imports(tmp_path, source, target):
    package = _package(tmp_path, {
        f"behavior.{source}.choice": f"from ..{target} import detector\n",
        f"behavior.{target}.detector": "",
    })
    report = audit(package)
    assert any(f"forbidden behavior layer dependency fpcode.behavior.{target}" in value
               for value in report["violations"])
    assert report["cycles"] == []


@pytest.mark.parametrize("layer", ["observations", "episodes", "assemblies", "context", "analysis"])
@pytest.mark.parametrize("target", ["segmentation", "segmentation.internal"])
def test_other_unit_layers_cannot_import_primitive_detector_internals(tmp_path, layer, target):
    package = _package(tmp_path, {
        f"behavior.{layer}.choice": f"import fpcode.behavior.primitives.{target}\n",
        f"behavior.primitives.{target}": "",
    })
    assert audit(package)["violations"] == [
        f"fpcode.behavior.{layer}.choice:1: primitive detector dependency fpcode.behavior.primitives.{target}",
    ]


@pytest.mark.parametrize("layer", ["episodes", "context", "analysis"])
def test_lower_unit_helpers_remain_available_without_detector_internals(tmp_path, layer):
    package = _package(tmp_path, {
        f"behavior.{layer}.choice":
            "import fpcode.behavior.primitives.bouts\n"
            "import fpcode.behavior.primitives.counting\n"
            "import fpcode.behavior.primitives.resolutions\n",
        "behavior.primitives.bouts": "",
        "behavior.primitives.counting": "",
        "behavior.primitives.resolutions": "",
    })
    assert audit(package)["violations"] == []


def test_primitive_methods_may_share_primitive_detector_implementation(tmp_path):
    package = _package(tmp_path, {
        "behavior.primitives.choice": "from .segmentation import fit\n",
        "behavior.primitives.segmentation": "def fit(): pass\n",
    })
    assert audit(package)["violations"] == []


@pytest.mark.parametrize("source", [
    "import importlib as loader\nloader.import_module('fpcode.behavior.episodes.detector')\n",
    "from importlib import import_module as load\nload('..episodes.detector', __package__)\n",
    "from importlib import import_module\n"
    "import_module(name='.detector', package='fpcode.behavior.episodes')\n",
])
def test_analysis_literal_dynamic_import_cannot_load_detector(tmp_path, source):
    package = _package(tmp_path, {
        "behavior.analysis.choice": source,
        "behavior.episodes.detector": "",
    })
    assert audit(package)["violations"] == [
        "fpcode.behavior.analysis.choice:2: forbidden behavior layer dependency fpcode.behavior.episodes.detector",
    ]


def test_analysis_initializer_is_not_a_detector_escape(tmp_path):
    package = _package(tmp_path, {
        "behavior.analysis.__init__": "from ..episodes.detector import fit\n",
        "behavior.analysis.measure": "from . import fit\n",
        "behavior.episodes.detector": "def fit(): pass\n",
    })
    assert any(value.startswith("fpcode.behavior.analysis:1: forbidden behavior layer dependency")
               for value in audit(package)["violations"])


@pytest.mark.parametrize("caller", ["behavior.analysis.choice", "workflows.measure", "kpms.consumer"])
def test_canonical_caller_cannot_use_old_detector_facade(tmp_path, caller):
    package = _package(tmp_path, {
        caller: "from fpcode.behavior.sequences.seqnmf import fit\n",
        "behavior.sequences.seqnmf": "from fpcode.behavior.episodes.seqnmf import fit\n",
        "behavior.episodes.seqnmf": "def fit(): pass\n",
    })
    assert audit(package)["violations"] == [
        f"fpcode.{caller}:1: canonical caller must import the behavior owner, not its facade: "
        "fpcode.behavior.sequences.seqnmf",
    ]


def test_behavior_public_aggregate_cannot_hide_dependency_but_module_target_is_explicit(tmp_path):
    package = _package(tmp_path, {
        "behavior.__init__": "from .observations.kinematics import smooth\n",
        "behavior.observations.kinematics": "def smooth(): pass\n",
        "behavior.observations.__init__": "",
        "behavior.primitives.good": "from fpcode.behavior import observations\n",
        "behavior.primitives.bad": "from fpcode.behavior import smooth\n",
    })
    assert audit(package)["violations"] == [
        "fpcode.behavior.primitives.bad:1: canonical caller must import the behavior owner, "
        "not its facade: fpcode.behavior",
    ]


def test_named_behavior_facade_only_forwards_to_its_actual_owner(tmp_path):
    modules = {
        "behavior.sequences.chains": "from fpcode.behavior.episodes.chains import detect\n",
        "behavior.episodes.chains": "def detect(): pass\n",
        "behavior.episodes.seqnmf": "",
    }
    package = _package(tmp_path, modules)
    assert audit(package)["violations"] == []
    (package / "behavior/sequences/chains.py").write_text("import fpcode.behavior.episodes.seqnmf\n")
    assert audit(package)["violations"] == [
        "fpcode.behavior.sequences.chains:1: behavior facade must forward to a named owner: "
        "fpcode.behavior.episodes.seqnmf",
    ]


def test_old_behavior_facade_cannot_contain_detector_implementation(tmp_path):
    package = _package(tmp_path, {"behavior.sequences.seqnmf": "def fit(): pass\n"})
    assert audit(package)["violations"] == [
        "fpcode.behavior.sequences.seqnmf:1: behavior facade must only forward names",
    ]


@pytest.mark.parametrize("module", ["unit_adapters", "compatibility"])
def test_behavior_bridges_stay_independent_of_detectors(tmp_path, module):
    package = _package(tmp_path, {
        f"behavior.{module}": "import fpcode.behavior.episodes.chains\n",
        "behavior.episodes.chains": "",
    })
    assert audit(package)["violations"] == [
        f"fpcode.behavior.{module}:1: behavior foundation dependency fpcode.behavior.episodes.chains",
    ]


def test_unit_adapter_reuses_exact_gap_aware_bout_helper(tmp_path):
    package = _package(tmp_path, {
        "behavior.unit_adapters": "from fpcode.behavior.primitives.bouts import run_boundaries\n",
        "behavior.primitives.bouts": "def run_boundaries(): pass\n",
    })
    assert audit(package)["violations"] == []


@pytest.mark.parametrize(("bridge", "target"), [
    ("unit_adapters", "detector"), ("unit_adapters", "resolutions"),
    ("compatibility", "bouts"),
])
def test_bout_helper_exception_does_not_grant_other_bridge_dependencies(tmp_path, bridge, target):
    package = _package(tmp_path, {
        f"behavior.{bridge}": f"import fpcode.behavior.primitives.{target}\n",
        f"behavior.primitives.{target}": "",
    })
    assert audit(package)["violations"] == [
        f"fpcode.behavior.{bridge}:1: behavior foundation dependency fpcode.behavior.primitives.{target}",
    ]


def test_behavior_layer_rules_do_not_expand_top_level_permissions(tmp_path):
    package = _package(tmp_path, {
        "behavior.episodes.choice": "import fpcode.kpms.training\nimport studies.cohort\n",
        "kpms.training": "",
    })
    assert audit(package)["violations"] == [
        "fpcode.behavior.episodes.choice:1: forbidden dependency fpcode.kpms.training",
        "fpcode.behavior.episodes.choice:2: upward import studies.cohort",
    ]
