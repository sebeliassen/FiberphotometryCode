"""Saved baseline ASTs guard relocated numerical algorithms without JAX."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTENTIONAL_CHANGES = {
    "runtime": {"set_batch_sizes", "precompute_optimal_batch_config", "apply_optimizations",
                "restore_original_kernels", "kpms_optimizations"},
    "visualization": {"HashCache", "hash_files", "write_json_atomic"},
}


def test_extracted_numerical_bodies_match_pre_refactor_baseline():
    fixture = json.loads((ROOT / "tests/golden/module_extraction.json").read_text())
    checked = 0
    for group in ("runtime", "visualization"):
        for name, record in fixture[group].items():
            if name in INTENTIONAL_CHANGES[group]:
                continue
            path = ROOT / "src/fpcode" / (record["module"].replace(".", "/") + ".py")
            definitions = {node.name: node for node in ast.parse(path.read_text()).body
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            digest = hashlib.sha256(ast.dump(definitions[name], include_attributes=False).encode()).hexdigest()
            assert digest == record["ast_sha256"], (group, name, path)
            checked += 1
    assert checked == 85
