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
