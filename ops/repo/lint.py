"""Check maintained source with strict library lint and explicit study lint debt."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "registry/LINT_BASELINE.json"
STRICT_PATHS = (
    "src/fpcode", "pipelines", "ops/backup", "ops/repo/dependencies.py",
    "ops/repo/lint.py", "ops/repo/package_check.py",
)
MAINTAINED_PATHS = ("studies", "tests", "ops")


def diagnostics(paths: tuple[str, ...]) -> list[dict]:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format", "json", *paths],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def fingerprint(item: dict) -> tuple[str, str, str]:
    return (str(Path(item["filename"]).relative_to(ROOT)), item["code"], item["message"])


def summarize(items: list[dict]) -> Counter:
    # Line changes do not hide or invent debt. Repeated diagnostics still count.
    return Counter(fingerprint(item) for item in items)


def report(items: list[dict], label: str) -> None:
    print(label)
    for item in items:
        path, code, message = fingerprint(item)
        position = item["location"]
        print(f"  {path}:{position['row']}:{position['column']}: {code} {message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true", help="Explicitly replace the reviewable debt inventory")
    args = parser.parse_args(argv)
    strict = diagnostics(STRICT_PATHS)
    if strict:
        report(strict, "Strict library/entry-point lint failed:")
        return 1
    current_items = diagnostics(MAINTAINED_PATHS)
    current = summarize(current_items)
    version = subprocess.check_output([sys.executable, "-m", "ruff", "--version"], text=True).strip()
    if args.write_baseline:
        payload = {
            "schema_version": 1,
            "ruff_version": version,
            "scope": list(MAINTAINED_PATHS),
            "diagnostics": [
                {"path": path, "code": code, "message": message, "count": count}
                for (path, code, message), count in sorted(current.items())
            ],
        }
        BASELINE.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Recorded {sum(current.values())} existing diagnostics in {BASELINE.relative_to(ROOT)}; review its diff.")
        return 0
    if not BASELINE.exists():
        print("Missing lint debt inventory; create and review it with make lint-baseline.", file=sys.stderr)
        return 1
    payload = json.loads(BASELINE.read_text())
    if payload.get("schema_version") != 1 or payload.get("scope") != list(MAINTAINED_PATHS):
        print("Lint inventory schema/scope mismatch.", file=sys.stderr)
        return 1
    if payload["ruff_version"] != version:
        print(f"Use {payload['ruff_version']} to match the reviewed lint inventory; found {version}.", file=sys.stderr)
        return 1
    baseline = Counter({
        (item["path"], item["code"], item["message"]): item["count"]
        for item in payload["diagnostics"]
    })
    added = current - baseline
    if added:
        remaining = added.copy()
        new_items = []
        for item in current_items:
            key = fingerprint(item)
            if remaining[key]:
                new_items.append(item)
                remaining[key] -= 1
        report(new_items, "New maintained-source lint debt:")
        return 1
    resolved = sum((baseline - current).values())
    print(f"Strict lint passed; maintained source has {sum(current.values())} reviewed diagnostics and no new debt.")
    if resolved:
        print(f"{resolved} inventoried diagnostics resolved; a reviewed baseline refresh can remove them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
