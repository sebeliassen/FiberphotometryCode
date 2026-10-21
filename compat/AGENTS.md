# Historical namespaces

Root `compat` preserves old top-level import/file namespaces. `src/fpcode/compat` owns historical library implementations and profiles. Do not conflate the two.

Preserve module/function/class object identity where tested, historical pickle qualified names and both import orders. Directory symlinks can load the same file twice under different module names; retarget proxies to the canonical module rather than copying definitions or weakening `is` assertions.

Physical adapters can serve `runpy`, file/spec loaders and direct scripts. A global import finder is not a replacement for those filesystem obligations. Some operational scripts expose `run_pipeline` rather than `main`.

Reconcile `registry/SHIMS.tsv` and source/data move ledgers when targets change. Review historical migration utilities before running them; their old namespaces are not the current tree. Validate affected identity and direct-execution checks before retiring an alias. Retaining an import path proves compatibility only for the serialized artifacts actually checked.
