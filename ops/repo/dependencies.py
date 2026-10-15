"""Read-only AST dependency audit, including relative imports and cycles."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path

FORBIDDEN = {"studies", "syllable_generation", "archive", "tests", "scratch", "core", "pipelines"}
# Historical modules may forward only to their named compatibility owner.
# Domain implementations never import compat directly. Workflows can select an
# explicit historical profile; existing adapter facades retain old call defaults.
COMPATIBILITY_FACADES = {
    "fpcode.common.io": "fpcode.compat.io",
    "fpcode.common.schemas": "fpcode.compat.schemas",
    "fpcode.adapters.multiview_compatibility": "fpcode.compat.multiview.profile",
    "fpcode.kpms.export": "fpcode.compat.kpms_export",
    "fpcode.photometry.alignment": "fpcode.compat.photometry.frame_merge",
    "fpcode.photometry.processing": "fpcode.compat.photometry.acquisition_alignment",
    "fpcode.viz.v2": "fpcode.compat.viz.v2",
    "fpcode.viz.legacy_api": "fpcode.compat.viz.legacy_api",
    "fpcode.viz.legacy_defaults": "fpcode.compat.viz.legacy_defaults",
    "fpcode.viz.legacy_io": "fpcode.compat.viz.legacy_io",
}
# Reusable visualization algorithms receive their landmarks and input files
# explicitly. Historical defaults may be selected only by compatibility APIs.
EXPLICIT_VIZ_MODULES = {
    "fpcode.viz.context", "fpcode.viz.geometry", "fpcode.viz.pose_geometry",
    "fpcode.viz.media", "fpcode.viz.provenance",
    "fpcode.viz.selection.bouts", "fpcode.viz.selection.density",
    "fpcode.viz.selection.eligibility", "fpcode.viz.selection.exemplars",
}
COHORT_VIZ_MODULES = {
    "fpcode.viz.legacy_defaults", "fpcode.viz.legacy_api", "fpcode.viz.legacy_io",
    "fpcode.compat.viz.legacy_defaults", "fpcode.compat.viz.legacy_api", "fpcode.compat.viz.legacy_io",
}
LAYERS = {
    "contracts": {"contracts"},
    "common": {"common", "contracts"},  # schemas retains a compatibility facade
    "adapters": {"adapters", "contracts", "common"},
    "alignment": {"alignment", "contracts", "common"},
    "statistics": {"statistics", "contracts", "common"},
    "behavior": {"behavior", "contracts", "alignment", "statistics", "common"},
    "tracking": {"tracking", "contracts", "adapters", "alignment", "statistics", "common"},
    "photometry": {"photometry", "contracts", "adapters", "alignment", "behavior", "statistics", "common"},
    "kpms": {"kpms", "contracts", "adapters", "alignment", "behavior", "tracking", "statistics", "common"},
    "viz": {"viz", "contracts", "adapters", "alignment", "behavior", "tracking", "statistics", "common"},
    "compat": set(),
    "workflows": set(),
    "cli": set(),
}
LAYERS["workflows"] = set(LAYERS) - {"cli"}
LAYERS["cli"] = set(LAYERS)
LAYERS["compat"] = set(LAYERS) - {"cli"}

# These are unit choices inside the behavior capability, rather than an
# alternative to the top-level dependency contract. Workflows compose detectors
# and measurements; detector implementations do not reach into one another's
# higher layers or recover that permission through an aggregate public import.
BEHAVIOR_LAYERS = {
    "observations": {"observations"},
    "primitives": {"observations", "primitives"},
    "episodes": {"observations", "primitives", "episodes"},
    "assemblies": {"observations", "assemblies"},
    "context": {"observations", "primitives", "context"},
    "analysis": {"observations", "primitives", "analysis"},
}
# Higher unit choices may reuse primitive views and interval/counting helpers,
# but fitted primitive detector internals belong behind explicit result adapters.
BEHAVIOR_PRIMITIVE_DETECTORS = {"fpcode.behavior.primitives.segmentation"}
BEHAVIOR_FOUNDATIONS = {"fpcode.behavior.unit_adapters", "fpcode.behavior.compatibility"}
# Label-to-interval conversion reuses the canonical gap-aware RLE helper rather
# than implementing a second segmentation policy. This is a single adapter
# dependency, not permission to select primitive detectors or resolutions.
BEHAVIOR_FOUNDATION_DEPENDENCIES = {
    "fpcode.behavior.unit_adapters": {"fpcode.behavior.primitives.bouts"},
    "fpcode.behavior.compatibility": set(),
}
# Old paths remain import-compatible, but new shared callers name the owner.
# Timeline reconstruction and historical tiling are retained adapters, not
# forwarding facades or detector implementations in the new unit layers.
BEHAVIOR_FACADES = {
    "fpcode.behavior": {
        "fpcode.behavior.observations.kinematics", "fpcode.behavior.primitives.bouts",
        "fpcode.behavior.analysis.transitions", "fpcode.behavior.sequences.timelines",
        "fpcode.behavior.sequences.tiling", "fpcode.behavior.syllables",
    },
    "fpcode.behavior.syllables": {
        "fpcode.behavior.primitives.bouts", "fpcode.behavior.sequences.timelines",
        "fpcode.behavior.sequences.tiling",
    },
    "fpcode.behavior.sequences": {
        "fpcode.behavior.sequences.timelines", "fpcode.behavior.sequences.tiling",
    },
    "fpcode.behavior.kinematics": {"fpcode.behavior.observations.kinematics"},
    "fpcode.behavior.bouts": {"fpcode.behavior.primitives.bouts"},
    "fpcode.behavior.resolutions": {"fpcode.behavior.primitives.resolutions"},
    "fpcode.behavior.variants": {"fpcode.behavior.analysis.variants"},
    "fpcode.behavior.phenotyping": {"fpcode.behavior.analysis.phenotyping"},
    "fpcode.behavior.transitions": {"fpcode.behavior.analysis.transitions"},
    "fpcode.behavior.sequences.counting": {"fpcode.behavior.primitives.counting"},
    "fpcode.behavior.sequences.chains": {"fpcode.behavior.episodes.chains"},
    "fpcode.behavior.sequences.seqnmf": {"fpcode.behavior.episodes.seqnmf"},
    "fpcode.behavior.sequences.organisation": {"fpcode.behavior.analysis.organisation"},
    "fpcode.behavior.sequences.prediction": {"fpcode.behavior.analysis.prediction"},
}


def _behavior_dependency_violation(name, imported):
    if name in BEHAVIOR_FACADES:
        if imported.startswith("fpcode.") and imported not in BEHAVIOR_FACADES[name]:
            return f"behavior facade must forward to a named owner: {imported}"
        return None
    if imported in BEHAVIOR_FACADES and not name.startswith("fpcode.compat."):
        return f"canonical caller must import the behavior owner, not its facade: {imported}"
    if name in BEHAVIOR_FOUNDATIONS:
        if imported.startswith("fpcode.") and (imported not in BEHAVIOR_FOUNDATIONS
                                               and imported not in BEHAVIOR_FOUNDATION_DEPENDENCIES[name]
                                               and imported.split(".")[1] not in {"contracts", "common"}):
            return f"behavior foundation dependency {imported}"
        return None
    own_parts, target_parts = name.split("."), imported.split(".")
    if len(own_parts) < 3 or own_parts[:2] != ["fpcode", "behavior"]:
        return None
    own_layer = own_parts[2]
    if own_layer not in BEHAVIOR_LAYERS or target_parts[:2] != ["fpcode", "behavior"]:
        return None
    if own_layer != "primitives" and any(
            imported == detector or imported.startswith(detector + ".")
            for detector in BEHAVIOR_PRIMITIVE_DETECTORS):
        return f"primitive detector dependency {imported}"
    if imported in BEHAVIOR_FOUNDATIONS:
        return None
    if len(target_parts) < 3 or target_parts[2] not in BEHAVIOR_LAYERS[own_layer]:
        return f"forbidden behavior layer dependency {imported}"
    return None


def module_name(path, package_root):
    parts = list(path.relative_to(package_root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _cycles(graph):
    completed, active, stack, found = set(), set(), [], []

    def visit(node):
        if node in active:
            found.append(stack[stack.index(node):] + [node])
            return
        if node in completed:
            return
        active.add(node)
        stack.append(node)
        for target in sorted(graph.get(node, ())):
            visit(target)
        stack.pop()
        active.remove(node)
        completed.add(node)

    for node in sorted(graph):
        visit(node)
    return found


def _top_level_parse_calls(tree):
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.lines = []

        def visit_FunctionDef(self, node):
            for value in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
                if value is not None:
                    self.visit(value)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_If(self, node):
            if ast.unparse(node.test) in ("__name__ == '__main__'", "'__main__' == __name__"):
                return
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"parse_args", "parse_known_args"}:
                self.lines.append(node.lineno)
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    return visitor.lines


def _importlib_aliases(tree):
    libraries, loaders = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            libraries.update(alias.asname or alias.name.split(".")[0] for alias in node.names
                             if alias.name == "importlib" or alias.name.startswith("importlib."))
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib" and not node.level:
            loaders.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
    return libraries, loaders


def _literal_import_call(node, aliases, module, package):
    libraries, loaders = aliases
    function = node.func
    recognized = (isinstance(function, ast.Name) and function.id in loaders
                  or isinstance(function, ast.Attribute) and function.attr == "import_module"
                  and isinstance(function.value, ast.Name) and function.value.id in libraries)
    if not recognized:
        return None
    keywords = {item.arg: item.value for item in node.keywords}
    argument = node.args[0] if node.args else keywords.get("name")
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        return None
    imported = argument.value
    if imported.startswith("."):
        context = node.args[1] if len(node.args) > 1 else keywords.get("package")
        if isinstance(context, ast.Constant) and isinstance(context.value, str):
            base = context.value
        elif isinstance(context, ast.Name) and context.id in {"__package__", "__name__"}:
            base = package if context.id == "__package__" else module
        else:
            return None  # Computed package identities need human review.
        imported = importlib.util.resolve_name(imported, base)
    return imported


def audit(package_root):
    package_root = Path(package_root)
    paths = {module_name(path, package_root): path for path in sorted(package_root.rglob("*.py"))}
    if not paths:
        raise ValueError(f"no Python modules in {package_root}")
    graph = {name: set() for name in paths}
    violations = []
    for name, path in paths.items():
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeError) as error:
            violations.append(f"{name}: cannot parse: {error}")
            continue
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        import_aliases = _importlib_aliases(tree)
        facade_forwarded = False
        for node in ast.walk(tree):
            dynamic = False
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                try:
                    base = (importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                            if node.level else node.module or "")
                except (ImportError, ValueError) as error:
                    violations.append(f"{name}:{node.lineno}: invalid relative import: {error}")
                    continue
                child_modules = [f"{base}.{alias.name}" for alias in node.names
                                 if f"{base}.{alias.name}" in paths]
                # A named behavior submodule is an explicit target, unlike a
                # symbol reexported by the aggregate compatibility initializer.
                # Preserve the normal initializer edge for all other imports.
                candidates = ([base] if not (base.startswith("fpcode.behavior")
                                             and len(child_modules) == len(node.names)) else []) + child_modules
            elif isinstance(node, ast.Call):
                try:
                    imported = _literal_import_call(node, import_aliases, name, package)
                except (ImportError, ValueError) as error:
                    violations.append(f"{name}:{node.lineno}: invalid dynamic import: {error}")
                    continue
                if imported is None:
                    continue
                candidates, dynamic = [imported], True
            else:
                continue
            for imported in candidates:
                if COMPATIBILITY_FACADES.get(name) == imported:
                    facade_forwarded = True
                if name in EXPLICIT_VIZ_MODULES and imported in COHORT_VIZ_MODULES:
                    violations.append(f"{name}:{node.lineno}: implicit cohort dependency {imported}")
                if imported.split(".")[0] in FORBIDDEN:
                    violations.append(f"{name}:{node.lineno}: upward import {imported}")
                if imported.startswith("fpcode."):
                    own_layer = name.split(".")[1] if "." in name else None
                    target_layer = imported.split(".")[1]
                    if (own_layer in LAYERS and target_layer not in LAYERS[own_layer]
                            and COMPATIBILITY_FACADES.get(name) != imported):
                        violations.append(f"{name}:{node.lineno}: forbidden dependency {imported}")
                    if dynamic and imported not in paths:
                        violations.append(f"{name}:{node.lineno}: missing dynamic import target {imported}")
                    behavior_violation = _behavior_dependency_violation(name, imported)
                    if behavior_violation:
                        violations.append(f"{name}:{node.lineno}: {behavior_violation}")
                if imported in paths and imported != name:
                    graph[name].add(imported)
        violations.extend(f"{name}:{line}: argument parsing at import time"
                          for line in _top_level_parse_calls(tree))
        if facade_forwarded or name in BEHAVIOR_FACADES:
            # An old public path cannot gain permission to implement algorithms
            # simply by being listed as a compatibility facade.
            for node in tree.body:
                if (isinstance(node, ast.ClassDef)
                        or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name not in {"__getattr__", "__dir__"}):
                    kind = "behavior" if name in BEHAVIOR_FACADES else "compatibility"
                    violations.append(f"{name}:{node.lineno}: {kind} facade must only forward names")
    return {"module_count": len(paths), "edges": {name: sorted(edges) for name, edges in graph.items()},
            "violations": violations, "cycles": _cycles(graph)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[2] / "src/fpcode")
    report = audit(parser.parse_args(argv).package)
    print(json.dumps(report, indent=2))
    return int(bool(report["violations"] or report["cycles"]))


if __name__ == "__main__":
    raise SystemExit(main())
