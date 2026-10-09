# KPMS scaling handoff — revision 2

Updated September 29, 2026 with the auxiliary archive and reported later-sweep OOM.

**Read [AUXILIARY_REVIEW.md](AUXILIARY_REVIEW.md) first.** The new first experiment is to check whether re-entering `fit_model` uploads host `s` back to the GPU. The report also identifies iteration-count, timing, checkpoint and configuration issues that must be resolved before calling the failure proven fragmentation.

- [KPMS_ENGINEERING_DOSSIER.md](KPMS_ENGINEERING_DOSSIER.md): consolidated dossier, including the auxiliary review and updated prompt.
- [ANTIGRAVITY_PROMPT.md](ANTIGRAVITY_PROMPT.md): ready-to-paste investigation instructions.
- `auxiliary_bundle/`: all 101 files from the newest supplied archive, unchanged — 84 Python, 2 shell, 15 Markdown.
- `original_bundle/`: all 100 files from the preceding archive, unchanged — 84 Python, 2 shell, 14 Markdown.
- [AUXILIARY_DIFF.patch](AUXILIARY_DIFF.patch): exact differences in the four changed files; not a proposed fix.
- [CODE_INDEX.md](CODE_INDEX.md): latest source index. [BASELINE_CODE_INDEX.md](BASELINE_CODE_INDEX.md): previous snapshot index.
- `SOURCE_MANIFEST.json`, `DEPENDENCY_INVENTORY.json`, [VALIDATION_REPORT.md](VALIDATION_REPORT.md): provenance, import inventory, static/packaging checks and limits.
- `source_notes/`: original pasted contextual documents, preserved unchanged.

This bundle does not modify the supplied Python or shell code and does not claim to fix or reproduce the GPU failure. Both snapshots are review exports with the same missing live-repository dependencies described in the dossier. Preserve their original import layout when preparing an experiment; do not run both patch revisions together.

The new CPU-scales behavior applies to the large-batch sampler and cache restoration in two runners. Initialization, the small-batch path, fitting wrappers and later consumers require a complete residency audit. The incident's fragmentation diagnosis and 46-second steady-state interpretation remain unverified. Current official documentation is cited as supporting reference material, not evidence of the user's installed version.
