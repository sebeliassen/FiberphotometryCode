"""Historical facades cannot relax the boundaries of actual algorithms."""

from ops.repo.dependencies import audit


def test_domain_algorithm_cannot_import_historical_profile(tmp_path):
    package = tmp_path / "fpcode"
    (package / "tracking").mkdir(parents=True)
    (package / "compat").mkdir()
    (package / "tracking/engine.py").write_text("from fpcode.compat.schemas import PathConfig\n")
    (package / "compat/schemas.py").write_text("class PathConfig: pass\n")
    result = audit(package)
    assert result["violations"] == [
        "fpcode.tracking.engine:1: forbidden dependency fpcode.compat.schemas",
    ]


def test_named_facade_only_forwards_to_its_specific_owner(tmp_path):
    package = tmp_path / "fpcode"
    (package / "kpms").mkdir(parents=True)
    (package / "compat").mkdir()
    (package / "compat/kpms_export.py").write_text("def export(): pass\n")
    (package / "kpms/export.py").write_text(
        "import importlib\nimportlib.import_module('fpcode.compat.kpms_export')\n")
    assert audit(package)["violations"] == []
    (package / "kpms/export.py").write_text(
        "import importlib\nimportlib.import_module('fpcode.compat.kpms_export')\n"
        "def calculate(): return 1\n")
    assert "compatibility facade must only forward names" in audit(package)["violations"][0]
    (package / "kpms/export.py").write_text("import fpcode.compat.other\n")
    assert "forbidden dependency fpcode.compat.other" in audit(package)["violations"][0]
