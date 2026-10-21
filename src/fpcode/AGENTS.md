# Shared library

The [capability contract](../../docs/architecture/UNIT_CONTRACT.md) and `ops/repo/dependencies.py` define ownership and allowed imports. The library must not import studies, old runners, tests or archive material. Pass project policy explicitly; named historical facades have narrow exceptions, not permission to add new implicit defaults.

Keep base-package/configuration/CLI imports free of backend initialization and output writes. Optional numerical backends belong behind explicit operations. Patches must restore each target namespace's previous value on success and failure.

Preserve units, clocks, validity and canonical record identity. Moving a function is not evidence that its numerical behavior is unchanged. Use independent small expectations for changed algorithms and retain existing golden/identity checks.

Check relevant tests, full functional tests for shared behavior, architecture, lint, generated maps and installed-wheel behavior when exports/packaging change. Intentional reexports must survive lint fixes. Runtime/kernel changes also need appropriately scoped empirical evidence before claims about scientific parity or performance.
