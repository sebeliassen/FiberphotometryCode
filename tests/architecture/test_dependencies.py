from pathlib import Path

from ops.repo.dependencies import audit

ROOT = Path(__file__).resolve().parents[2]


def test_dependency_boundaries_and_cycles():
    report = audit(ROOT / "src/fpcode")
    assert report["module_count"] > 20
    assert not report["violations"], "\n".join(report["violations"])
    assert not report["cycles"], report["cycles"]


def test_audit_catches_relative_layer_violation_and_cycle(tmp_path):
    package = tmp_path / "fpcode"
    (package / "common").mkdir(parents=True)
    (package / "behavior").mkdir()
    (package / "common/a.py").write_text("from ..behavior import b\n")
    (package / "behavior/b.py").write_text("from ..common import a\n")
    report = audit(package)
    assert any("forbidden dependency" in item for item in report["violations"])
    assert report["cycles"]


def test_audit_does_not_hide_syntax_errors_or_assigned_parse_args(tmp_path):
    package = tmp_path / "fpcode"
    package.mkdir()
    (package / "bad.py").write_text("def broken(\n")
    (package / "cli.py").write_text("if True:\n    args = parser.parse_args()\n")
    (package / "safe.py").write_text("if __name__ == '__main__':\n    args = parser.parse_args()\n")
    report = audit(package)
    assert len(report["violations"]) == 2


def test_audit_catches_literal_dynamic_boundaries_missing_targets_and_cycles(tmp_path):
    package = tmp_path / "fpcode"
    (package / "common").mkdir(parents=True)
    (package / "behavior").mkdir()
    (package / "common/a.py").write_text(
        "import importlib as loader\n"
        "loader.import_module('fpcode.behavior.b')\n"
        "loader.import_module(name='fpcode.common.missing')\n"
        "loader.import_module('studies.private_cohort')\n")
    (package / "behavior/b.py").write_text(
        "from importlib import import_module as load\n"
        "load('.a', 'fpcode.common')\n")
    report = audit(package)
    assert any("forbidden dependency fpcode.behavior.b" in item for item in report["violations"])
    assert any("missing dynamic import target fpcode.common.missing" in item for item in report["violations"])
    assert any("upward import studies.private_cohort" in item for item in report["violations"])
    assert report["cycles"] == [["fpcode.behavior.b", "fpcode.common.a", "fpcode.behavior.b"]]


def test_kpms_cannot_depend_on_workflow_exports(tmp_path):
    package = tmp_path / "fpcode"
    (package / "kpms").mkdir(parents=True)
    (package / "workflows").mkdir()
    (package / "kpms/export.py").write_text("from fpcode.workflows.session_export import export_sessions\n")
    (package / "kpms/training.py").write_text("from fpcode.workflows.session_export import export_sessions\n")
    (package / "workflows/session_export.py").write_text("def export_sessions(): pass\n")
    report = audit(package)
    assert len(report["violations"]) == 2
    assert any(item.startswith("fpcode.kpms.export:") for item in report["violations"])
    assert any(item.startswith("fpcode.kpms.training:") for item in report["violations"])


def test_generic_visualization_cannot_import_cohort_defaults_or_wrappers(tmp_path):
    package = tmp_path / "fpcode"
    (package / "viz").mkdir(parents=True)
    (package / "viz/legacy_defaults.py").write_text("COHORT = 'historical'\n")
    (package / "viz/legacy_io.py").write_text("def load(): pass\n")
    (package / "viz/legacy_api.py").write_text("from .legacy_defaults import COHORT\n")
    (package / "viz/pose_geometry.py").write_text("from fpcode.viz import legacy_defaults\n")
    (package / "viz/provenance.py").write_text(
        "from importlib import import_module\nimport_module('fpcode.viz.legacy_io')\n")
    (package / "viz/media.py").write_text("from .legacy_api import load\n")
    report = audit(package)
    assert len(report["violations"]) == 3
    assert all("implicit cohort dependency" in item for item in report["violations"])
    assert not report["cycles"]
