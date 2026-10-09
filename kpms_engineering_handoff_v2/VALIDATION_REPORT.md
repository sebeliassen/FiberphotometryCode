# Revision 2 validation report

Prepared September 29, 2026.

| Check | Result |
|---|---|
| First source ZIP preservation | 100/100 files byte-identical |
| Auxiliary source ZIP preservation | 101/101 files byte-identical |
| Archive comparison | 4 changed files, 1 added report, 0 removed files |
| Both original text attachments | SHA-256 matches preserved copies |
| Python parse and in-memory compile | 168/168 passed across both snapshots under Python 3.12.14 |
| Shell syntax, without execution | 4/4 passed with bash -n |
| Local links in new handoff documents | 208 resolve |
| Embedded Antigravity prompt | Matches the standalone prompt |
| Final ZIP CRC and bytes | Verified after creation by packaging script |

## Warnings in unchanged source

- `original_bundle/diagnostics_and_benchmarks/profile_full_dataset_granular.py:524: invalid escape sequence '\p'`
- `auxiliary_bundle/diagnostics_and_benchmarks/profile_full_dataset_granular.py:524: invalid escape sequence '\p'`

## Scope

No supplied Python/shell file was changed, imported or executed. No GPU benchmark, installed environment inspection, timing, model validation or OOM reproduction was performed. Official upstream sources were checked to corroborate mechanisms; they do not identify the user's installed source. The new incident contains a traceback excerpt, not a complete raw log or allocator profile.

The 84-file latest Python snapshot and 84-file previous snapshot deliberately coexist for provenance. Repeated filenames are revisions, not additional patches to activate. Source differences are in AUXILIARY_DIFF.patch. The dossier and prompt now prioritize fit-entry residency and correcting measurement semantics before allocator or custom-kernel work.
