"""Recovered ignored source must survive a clean checkout without numerical edits."""

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class NumericalDefinitions(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def definition_hash(source):
    tree = ast.parse(source)
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    normalized = NumericalDefinitions().visit(ast.Module(body=definitions, type_ignores=[]))
    def portable(node):
        if isinstance(node, ast.AST):
            return {"node": type(node).__name__, **{name: portable(value) for name, value in ast.iter_fields(node)
                                                   if name != "type_params"}}
        if isinstance(node, list):
            return [portable(value) for value in node]
        return node

    payload = json.dumps(portable(normalized), sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_recovered_definitions_match_preserved_source():
    manifest = json.loads((ROOT / "registry/SOURCE_RECOVERY_20261021.json").read_text())
    assert len(manifest["files"]) == 20
    for row in manifest["files"]:
        path = ROOT / row["path"]
        assert definition_hash(path.read_text()) == row["definition_ast_sha256"], row["path"]


def test_ignore_rules_admit_source_and_keep_payloads_excluded():
    sources = ("studies/photometry_exploration/data/data_loading.py",
               "studies/language_models/lib/data/dataset.py",
               "studies/language_models/lib/models/discrete.py")
    payloads = ("data/example.py", "studies/photometry_exploration/data/recording.csv",
                "studies/language_models/lib/data/cache.pkl", "studies/language_models/lib/models/model.h5")
    for relative in sources + payloads:
        result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", relative], cwd=ROOT,
                                capture_output=True, check=False)
        assert result.returncode == (0 if relative in payloads else 1), relative


def test_data_loading_source_imports_in_fresh_interpreter_without_model_backends(tmp_path):
    code = f"""
import sys
from pathlib import Path
sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'src')!r}]
from studies.photometry_exploration.data.data_loading import DataContainer
from studies.language_models.lib.data.dataset import SessionDataset
import studies.photometry_exploration.data.data_loading as loading
assert Path(loading.__file__).is_relative_to(Path({str(ROOT)!r}))
container = DataContainer(int)
container.add_data('one', 3)
assert container.get_data('one') == 3
try:
    container.add_data('wrong', 'text')
except TypeError:
    pass
else:
    raise AssertionError('typed container accepted invalid data')
container.remove_data('one')
assert container.fetch_all_data_names() == []
assert SessionDataset().get_cohort_labels() == []
assert not {{'jax', 'keypoint_moseq', 'torch', 'cuml'}} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
