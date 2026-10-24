# AMPH source and metadata boundary

This owner recovers AMPH metadata helpers and source tables. AMPH enrichment and
mask execution raise `AMPHPolicyError` immediately. The historical AMPH adapter
now targets this owner; it previously loaded the IFdN cohort's enrichment instead.
The unified session builder still does not support `jonathan_openfield_amph`.

The original [source snapshot](historical/enrich_969c893.py.txt) is preserved as
non-importable text, byte for byte from commit
`969c893b8b6ba7ddd42af178f36aff3406f84cea`, path
`syllable_generation/projects/jonathan_openfield_amph/enrich.py`. Its SHA256 is
`4fcb471c6dba2269b0017cf6e6adccb5f47cd677a2f901fbe65c20d2ff326814`:
12,768 bytes with no trailing newline. The named historical February backup is
identical. A later formatting revision retains all ten original function ASTs.
No complete later AMPH implementation was found in the audited reachable history;
that search does not establish what exists in external backups.

The preserved source has several unresolved behaviors:

- `enrich` ends after assigning entry-frame metadata, has no return statement and
  never calls its automatic or manual exclusion helpers. It cannot serve as the
  list-returning session-build stage without a new, specified implementation.
- An unmatched automatic-mask key can leave `indices` unbound or reuse the prior
  session's indices. There is no validated missing-key policy in this snapshot.
- Manual exclusions and injection metadata infer a factor of two from a maximum
  index below 300,000; injection delay assumes 80 Hz. Array positions, native video
  frames and photometry frame indices need explicit coordinate and clock mapping.

Appending a return or exclusion tail, choosing mask order, or replacing those
clocks would select scientific behavior beyond source relocation. The active
module therefore gates `enrich`, `apply_auto_masks` and `apply_manual_exclusions`
before loading inputs or changing sessions, including empty and dirty calls.
It does not import, execute or dynamically reconstruct the frozen source.

Seven original helper ASTs are retained: CSV animal/session/exclusion parsers,
automatic-mask loading, region labels/columns and `_val`. Their original fields,
interval parsing, error handling and lookup precedence are preserved. The unused
historical `regex` import is omitted; no dependency is added. A single documented
`noqa: E722` retains the original session parser's broad exception handling rather
than changing that behavior during recovery. Other formatting changes preserve ASTs.

The five files under `overrides` are exact copies of tracked AMPH source tables:
`animals.csv`, `sessions.csv`, `exclusions.csv`, `exclusions_test_ifdn.csv` and
`reviewed_sessions.txt`. The old copies remain. `load_animal_map` still prefers
its own adjacent `animals.csv` over `ctx.csv_root/animals.csv`; an exclusions
override root does not change that animal-map precedence. IFdN and monoamine
baseline tables are separate policies.

These helpers read files when explicitly called. In particular,
`load_auto_mask_data` loads a selected NPZ; it is not an inert metadata probe.
No empirical auto-mask, model or recording was copied or read for this recovery.
The ignored historical `auto_mask.npz` link and external payload remain untouched.

Synthetic tests cover metadata fields, precedence and interval parsing, immediate
execution errors without side effects, exact source/table hashes, cold imports,
both legacy/canonical import orders and the physical adapter. They establish this
source and metadata boundary, not a working AMPH cohort analysis or equivalence
of historical empirical artifacts. Execution needs a complete stage contract,
explicit clock/coordinate policy and separately scoped validation on selected inputs.
