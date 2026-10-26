# Preserved IFdN session enrichment policy

This directory owns the historical Jonathan IFdN enrichment implementation from
`syllable_generation/projects/jonathan_openfield_example`. Its seven source-policy
files under `overrides/` are copied byte-for-byte from pinned commit `969c893`;
the historical copies remain in `compat/`. These are the original project choices,
not newly selected cohort defaults. `studies/oft_baseline/overrides` belongs to a
different policy with a different metadata schema and must not be substituted.

The direct builder imports this physical owner. Its eight original stages, two
enrichment calls, channel mapping and numerical processing are preserved. Use the
unambiguous leaf API:

```python
from studies.session_building.jonathan.build import build
```

The package-level `jonathan.build` name can refer to a module or a function,
depending on import order. Use the leaf API above or `build_sessions` for an unambiguous callable.
Importing the builder or its historical leaf proxies does not execute a build,
read policy CSVs/NPZs, probe videos or initialize modeling backends.

Historical lookup behavior is retained. `load_animal_map(ctx)` first checks this
directory's `overrides/animals.csv`, then `ctx.csv_root/animals.csv`, then
`ctx.csv_root/Jonathan_IFxDN_csvs/animals.csv`. Missing animal metadata prints the
existing diagnostic and returns an empty map. The optional `overrides_dir` passed
to `enrich()` selects exclusions and automatic masks; it does not change animal
metadata lookup. Manual interval endpoints remain inclusive and are unioned with
automatic frame exclusions. `skip_exclusions=True` skips both exclusion sources.

The [policy regression tests](../../../tests/unit/test_jonathan_builder_policy.py)
retain the seven original file checksums and controlled synthetic checks.
They do not validate ignored `auto_mask.npz` files, arbitrary saved sessions or a
complete empirical build. No empirical inputs were loaded during consolidation.
