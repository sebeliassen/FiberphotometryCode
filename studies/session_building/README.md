# Session building

`build_sessions_unified.build_sessions` composes the Jonathan stages with explicit
project policy owners. It passes the selected configuration through the existing
`BuildContext.cfg`, including into numerical process workers. Importing the module
or requesting help does not import scientific backends or build a cohort.

| Project argument | Enrichment and overrides | Configuration | Default discovery |
| --- | --- | --- | --- |
| `jonathan_full`, `jonathan_openfield_example` | `studies/oft_baseline/ifdn_policy` | Fresh default `JonathanConfig` | H5 |
| `jonathan_oft_baseline` | `studies/oft_baseline` | Baseline `config.JCFG` | Baseline hybrid |
| `jonathan_juvenile_intruder` | `studies/juvenile_intruder` | Juvenile `config.JCFG` | Baseline hybrid |
| `karen_project` | `studies/karen_pff_vs_monomer/analysis/enrich.py`; overrides in `session_policy/overrides` | Karen `session_policy.config.JCFG` | Karen hybrid |

The original project argument remains in session metadata. IFdN and monoamine
baseline tables have different schemas and are kept separate. Source and override
ownership follows the active canonical modules; `FIBERPHOTOMETRY_ROOT` is not an
external project plugin search path. Adding another project requires explicit
source and policy review.

`discovery_mode="h5"` forces the shared H5 stage with the selected context policy.
`discovery_mode="hybrid"` selects Karen's tracker for Karen and the historical
monoamine baseline tracker for other supported projects. That fallback deliberately
retains its own baseline discovery policy. It does not adopt juvenile or IFdN
settings merely because hybrid mode was forced. The remaining shared stages use
the selected context policy. Changing those discovery semantics needs separate
scientific review.

The order remains discovery, video references, photometry references, enrichment,
photometry realization, enrichment, splitting and numerical processing. A selected
config module exposing `OPTIMAL_FLIP_IDS` retains the optional flagging hook before
splitting. No flipped project policy is fabricated. API defaults remain
`force_dirty=True, overwrite=True`; CLI flags default to false. A cache hit with
`overwrite=False` returns before project resolution. Only load a trusted selected
pickle: cache loading is explicit deserialization, not a metadata check.

Shared stages retain their historical default policy for direct callers without a
`cfg` field; numerical processing retains the live config singleton fallback.
Explicit context policy takes precedence. Synthetic tests cover actual shared
stages, sequential cohorts, failure isolation, cold entry points, and two bounded
fresh process workers. They do not establish full empirical cohort equivalence.
Production joblib parallelism retains its existing behavior; the tests cap workers
and numerical threads rather than launching an unbounded production build.

[AMPH source recovery](../oft_amph/README.md) preserves its distinct original
source and tables. Its canonical and historical APIs expose metadata helpers;
enrichment and mask execution fail explicitly because the original enrichment is
incomplete and its frame-clock heuristic is unresolved. AMPH remains outside this
builder's supported project mapping. A flipped source owner has not been recovered.
Both IDs fail explicitly before building, as do unknown projects, unless the
historical cache fast path applies. Named trusted artifact
reconstruction, empirical/feedback recovery and remaining exporter/PPE consumers
still gate retirement of legacy roots. All legacy roots and symlinks are retained.

For the direct IFdN builder, use the stable lazy package callable
`from studies.session_building.jonathan import build_sessions`, or the unchanged
explicit leaf API `from studies.session_building.jonathan.build import build`.
Both refer to the same function accepting a `BuildContext`. This differs from
the unified `build_sessions_unified.build_sessions` orchestration API above.
The historical package-level `build` name remains import-order-sensitive; its
retirement requires a separate compatibility decision. See [the package API](jonathan/README.md).
