"""Read-only AST dependency audit, including relative imports and cycles."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path

FORBIDDEN = {"studies", "syllable_generation", "archive", "tests", "scratch", "core", "pipelines"}
# Historical entry points may delegate upward; actual domain implementations
# retain their normal boundary. Keep each exception explicit and narrow.
COMPATIBILITY_IMPORTS = {("fpcode.kpms.export", "fpcode.workflows.session_export")}
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
}
LAYERS = {
    "contracts": {"contracts"},
    "common": {"common", "contracts"},  # schemas retains a compatibility facade
    "adapters": {"adapters", "contracts", "common"},
    "alignment": {"alignment", "contracts", "common"},
    "statistics": {"statistics", "contracts", "common"},
    "behavior": {"behavior", "contracts", "alignment", "statistics", "common"},
    "tracking": {"tracking", "contracts", "adapters", "alignment", "behavior", "common"},
    "photometry": {"photometry", "contracts", "adapters", "alignment", "behavior", "statistics", "common"},
    "kpms": {"kpms", "contracts", "adapters", "alignment", "behavior", "photometry", "tracking", "statistics", "common"},
    "viz": {"viz", "contracts", "adapters", "alignment", "behavior", "tracking", "kpms", "statistics", "common"},
    "workflows": set(),
    "cli": set(),
}
LAYERS["workflows"] = set(LAYERS) - {"cli"}
LAYERS["cli"] = set(LAYERS)


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
                candidates = [base] + [f"{base}.{alias.name}" for alias in node.names
                                       if f"{base}.{alias.name}" in paths]
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
                if name in EXPLICIT_VIZ_MODULES and imported in COHORT_VIZ_MODULES:
                    violations.append(f"{name}:{node.lineno}: implicit cohort dependency {imported}")
                if imported.split(".")[0] in FORBIDDEN:
                    violations.append(f"{name}:{node.lineno}: upward import {imported}")
                if imported.startswith("fpcode."):
                    own_layer = name.split(".")[1] if "." in name else None
                    target_layer = imported.split(".")[1]
                    if (own_layer in LAYERS and target_layer not in LAYERS[own_layer]
                            and (name, imported) not in COMPATIBILITY_IMPORTS):
                        violations.append(f"{name}:{node.lineno}: forbidden dependency {imported}")
                    if dynamic and imported not in paths:
                        violations.append(f"{name}:{node.lineno}: missing dynamic import target {imported}")
                if imported in paths and imported != name:
                    graph[name].add(imported)
        violations.extend(f"{name}:{line}: argument parsing at import time"
                          for line in _top_level_parse_calls(tree))
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
