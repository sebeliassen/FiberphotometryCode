"""Saved baseline ASTs guard relocated numerical algorithms without JAX."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTENTIONAL_CHANGES = {
    "runtime": {"set_batch_sizes", "precompute_optimal_batch_config", "apply_optimizations",
                "restore_original_kernels", "kpms_optimizations"},
    # Raw finite-coordinate/confidence support now delegates to tracking.quality;
    # independent mask expectations cover that single intentional wrapper change.
    "visualization": {
        "HashCache", "hash_files", "write_json_atomic", "valid_track_mask",
        # These APIs now receive landmark and file contexts. Independent small
        # geometry/QC/provenance/HDF tests characterize both explicit contexts
        # and historical V2 defaults; the saved baseline remains untouched.
        "fusion_geometry", "compute_robust_bot_pose", "summarize_window_qc",
        "summarize_frac_gap", "choose_tier", "source_input_hashes",
        "verify_manifest_inputs", "selection_provenance_matches", "make_cache_key",
        "media_file_identity", "load_fused_tracks",
    },
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
    assert checked == 73
