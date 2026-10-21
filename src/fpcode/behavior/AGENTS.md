# Behavioural abstraction structure

Read [the module inventory and interfaces](README.md) and the repository
[capability contract](../../../docs/architecture/UNIT_CONTRACT.md).

- Unit abstraction, classification resolution, detector choice and inter/intra
  measurement are separate axes. Context remains optional and parallel.
- Shared records belong in `fpcode.contracts`. Preserve native clocks, validity,
  catalogue/model identity, intervals and original method outputs through adapters.
- Planned modules document their intended capabilities and expose no fake fitting
  API. Implement missing stage/completion/uncertainty payloads honestly when needed;
  the interval contract cannot invent them.
- Higher methods consume common lower results. Shared analyses do not import
  detector internals; `ops/repo/dependencies.py` enforces the declared boundaries.
- Old imports forward to canonical owners. Patching numerical globals requires the
  canonical module. Preserve callable identity, numerical fixtures and historical
  lookup paths when refining this structure.
- Study prototypes remain study-owned until their numerical/scientific assumptions
  are audited. Generic code receives cohort policy explicitly.
- Use relevant interface/architecture tests during development; batch required
  final gates at a meaningful checkpoint. No empirical fits follow from imports,
  placeholders, shape tests or a structural relocation.
