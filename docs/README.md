# Documentation index

The maintained entry points below describe the current package. Existing reports
and study notes keep their original scientific context; they are not merged into
one replacement report. No historical document needs to be deleted to use this index.

## Current guidance

| Document | Use |
| --- | --- |
| [Repository README](../README.md) | Installation, package entry points and fast checks |
| [Capability contract](architecture/UNIT_CONTRACT.md) | Module responsibilities and allowed dependencies |
| [Validation](architecture/VALIDATION.md) | Fast tests, CPU rendering checks and bounded two-session GPU validation |
| [Configuration guide](../configs/README.md) | Active profiles, legacy configuration links and photometry policies |
| [Recovery](recovery.md) | Existing backup evidence and restoration procedures |
| [Design decision](decisions/0001-capability-modules-and-compatible-scientific-policies.md) | Why scientific policies and compatibility APIs stay explicit |

The [October 9 handoff](../handoff.md) is a dated record. It is preserved and is
not the current task list or latest validation result. Current test selection is
defined in `pyproject.toml`; each explicit runtime check writes its own evidence
under `outputs/<run-id>/validation.json`.

## Historical engineering and scientific references

These documents remain useful records of the environment and investigations at
the time they were written. Their paths, counts, commands and issue statuses may
precede later repairs; consult the maintained guidance above for current usage.

| Topic | Existing references |
| --- | --- |
| Earlier architecture, operation and storage snapshots | [Architecture](ARCHITECTURE.md), [runbook](RUNBOOK.md), [data layout](DATA_LAYOUT.md), [known issues](KNOWN_ISSUES.md), [test split plan](TEST_SPLIT_PLAN.md) |
| Refactoring and backup history | [Scaffolding](reports/SCAFFOLDING_REPORT.md), [refactoring](reports/REFACTORING_CHANGE_REPORT.md), [validation](reports/VALIDATION_REPORT.md), [backup](reports/BACKUP_REPORT.md) |
| KPMS performance and memory | [Scaling dossier](reports/KPMS_SCALING_BOTTLENECKS_AND_OPTIMIZATIONS_DOSSIER.md), [patching and memory](reports/KPMS_MONKEYPATCHING_AND_MEMORY_SCALING_GUIDE.md), [compaction](reports/KPMS_CHECKPOINT_COMPACTION_AND_ROLLING_STORAGE_SPEC.md), [OOM resolution](reports/KPMS_OOM_REAPER_AND_COMPACTION_RESOLUTION_REPORT.md), [live crash analysis](reports/KPMS_SCALING_ADDENDUM_LIVE_CRASH_ANALYSIS.md) |
| Model comparisons and production | [8-keypoint briefing](reports/KPMS_8KP_TRANSITION_AND_PRODUCTION_BRIEFING.md), [8 versus 17 keypoints](reports/KPMS_8KP_VS_17KP_CONTROLLED_COMPARISON_REPORT.md), [seed 404 selection](reports/KPMS_SEED_404_EML_SELECTION_AND_SWE_HARDENING_REPORT.md), [tandem dossier](reports/TANDEM_MULTIVIEW_KPMS_ENGINEERING_DOSSIER_AND_EXPERT_REVIEW_REQUEST.md), [historical reproduction](KPMS_HISTORICAL_REPRODUCTION_BENCHMARK_RESULTS.md), [speedup reconciliation](KPMS_SPEEDUP_RECONCILIATION_AND_BENCHMARK_PLAN.md) |
| Tracking and multimodal science | [Integration pipeline](DATA_INTEGRATION_PIPELINE.md), [orientation and storage](KAREN_BOTCAM_ORIENTATION_AND_STORAGE.md), [Karen overview](KAREN_PROJECT_SYSTEM_OVERVIEW_AND_LITERATURE_REQUEST.md), [multiview alignment](TANDEM_KPMS_MULTIVIEW_ALIGNMENT_AND_EMPIRICAL_BENCHMARK.md), [ventral transfer learning](VENTRAL_TRANSFER_LEARNING_AND_PRODUCTION_PIPELINE.md), [sequence atlas](SEQNMF_CHAIN_ANOMALY_ATLAS_FRAMEWORK.md), [postural ceiling](reports/unique_information_postural_ceiling.md) |
| Dated working material | [Presentation](reports/monday_14_7_2026_presentation.md), [daily notes](reports/SUMMARY_FOR_TOMORROW.md), [earlier scaffolding instructions](reports/SWE_ORGANIZATION_BOT_PROMPT.md) |

## Study documentation and preservation

Study instructions belong alongside their analysis under [studies](../studies/).
Many existing notes still live under [syllable_generation](../syllable_generation/)
and some records remain under [reports](../reports/). Those locations are retained
until a document's owning study and incoming links have been checked. Frozen
bundles are indexed by [archive/ARCHIVE_INDEX.md](../archive/ARCHIVE_INDEX.md).

New broadly reusable guidance belongs here; study hypotheses, exclusions and
cohort-specific notes belong with their study. Consolidation should handle a
verified duplicate or superseded instruction with an explicit link, rather than
discarding earlier measurements or combining different scientific baselines.
