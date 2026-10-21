# From photometry analysis to behavioural routines

Handoff presentation for Sebastian Hammer Eliassen · 21 October 2026

Prepared from Git history, selected source files, the supplied combined lab reports, and the delivery-worktree records. The reports are historical evidence, including plans and draft text; their requests were not treated as instructions. Scientific models and analyses were not rerun for this presentation. Subsequent source preservation and integration checks are recorded in [current task state](../tasks/CURRENT.md) and the [main-promotion checkpoint](../../registry/checkpoints/main_promotion_20261021.json).

## Slide 1 — The overall contribution

- Built a reusable workflow connecting experimental recordings, photometry, tracking, behavioural syllables and recurring action sequences.
- Turned project-specific analyses into shared tools that could support several lab collaborations.
- Worked on both scientific interpretation and the computing needed to make those analyses practical.
- Current destination: using seqNMF-informed themes to study how behaviour is organised and performed over longer timescales.

Suggested opening: “The common thread in my work has been connecting neural signals to behaviour, while making the analysis reusable across projects.”

## Slide 2 — Where it started: photometry analysis, 2024–2025

- Developed CPT response and performance analyses: event-aligned signals, response metrics, participant comparisons and relationships between performance and fluorescence.
- Combined CPT and open-field workflows and supported both older single-channel and newer dual-sensor recordings.
- Added legacy-data conversion, session loading and synchronisation, signal processing, error handling and tests.
- Supported analysis across genotype, dose, brain region, task stage and time within a session.

Evidence: main-branch history and reports from December 2024 through June 2025. The historical data standard was described as provisional, pending agreement with experimental users.

## Slide 3 — What this enabled for the lab

- Sam/Jeppe: CPT performance comparisons, probe-test analyses, response relationships and initial mixed-model work.
- Leonie: manuscript/thesis plots covering NE responses, drugs, genotypes, regions, task stages and session performance.
- Jonathan/Søren: behavioural syllables, aligned dopamine signatures and comparisons of WT and IFxDN/DAT-KI cohorts.
- Additional contributions: Marco’s fixed-ratio analyses, tutorial notebooks, synchronised visualisations for Lundbeck, and Woody setup with Søren and Jan.

Speaker note: distinguish completed analysis support from proposed papers, follow-up experiments and broader scientific conclusions. RUBUS access/connectivity was tested; production cluster use is not established by the reports. DDSA proposal work with Sadegh and Ulrik is an optional additional contribution to mention.

## Slide 4 — From signals to measurable behaviour

- Extended the workflow to DeepLabCut tracking, multi-camera alignment, 3D pose and Keypoint-MoSeq syllables.
- Created synchronised movies combining video, pose, syllable labels and photometry to inspect alignment and interpretation.
- Developed turn-centred analysis, including turn detection and event-aligned neural responses.
- Trained bottom-camera tracking and explored combining body and paw information to improve behavioural descriptions.

Speaker note: 3D and tandem models were investigated because extra coordinates do not automatically yield clearer syllables. Gait discrimination and improved group separation remain empirical questions.

## Slide 5 — Making larger datasets practical

- Optimised Keypoint-MoSeq computation, CPU/GPU workload division and memory-aware batching.
- Improved handling of large datasets, interrupted runs and model storage.
- September report benchmark: approximately 18 hours to 41 minutes on the amph dataset, around 26–27× faster in that comparison.
- Reported completed-model compaction: approximately 75% smaller checkpoints and around 1.5 TB recovered; began a KPMS Studio interface for less technical users.

Speaker note: performance/storage numbers are historical reported measurements, not newly benchmarked or universal guarantees. The GUI was still in development; do not present a complete validated user workflow or upstream acceptance of the optimisations as established.

## Slide 6 — Why the refactors mattered

- Research had grown beyond the old photometry repository into many scripts, datasets, experiments and duplicated utilities.
- Scaffolding introduced inventories, recovery records, packaging, configuration and clearer source/data separation.
- Legacy consolidation extracted reusable mechanisms while preserving historical callers.
- Modular refactoring clarified shared-library, study, operations and compatibility ownership, with tests and dependency checks.

Practical benefit: a successor can change a shared method or a study analysis without needing to rediscover the entire workspace history.

## Slide 7 — The actual branch progression

Historical source checkpoints, before staged integration onto main:

| Stage | Recorded revision | Role in the handoff |
| --- | --- | --- |
| `main` before integration | `67468c1`, 8 Oct 2025 | Older photometry framework, OFT/KPMS integration and Turnlab; not the complete pre-refactor research workspace |
| `scaffolding-refactor` | `0891de1`, 9 Oct 2026 | Captured previously local research and added organisation, packaging, migration and recovery machinery |
| `codex/legacy-consolidation` | `b83298f`, 11 Oct 2026 | Intermediate shared-library extraction and study consolidation; already contains substantial refactoring |
| `codex/modular-refactor` | `15df93d`, 15 Oct 2026 | Newer capability architecture plus behavioural-layer/search research; source later preserved at `31326cc` |
| `codex/karen-seqnmf-delivery` | `736f3ca`, 16 Oct 2026; source preserved at `2c43b75`, 21 Oct | Isolated Karen seqNMF delivery; 45 reviewed source, test and documentation files captured from later working changes |

The first four tips form an ancestry chain. Delivery branches from modular research checkpoint `6c179ae`, before the last two modular documentation commits. The `pre-scaffolding-20261009` snapshot is a better baseline for the full research workspace than old `main`.

The initial review used local refs. Subsequent work refreshed remote refs, preserved both source states, and promoted reviewed changes through sequential pull requests with normal merges. Consult [current task state](../tasks/CURRENT.md) and the [main-promotion checkpoint](../../registry/checkpoints/main_promotion_20261021.json) for exact main revisions, completed promotions and validation. Ignored empirical outputs and browser feedback still require separate preservation.

## Slide 8 — The scientific move: “What comes after the syllable?”

- Individual syllables describe short actions; longer sequences may reveal routines and behavioural organisation.
- Explored predictive history, language-inspired representations and shMoSeq; the tested broader states were difficult to interpret consistently across animals.
- Exact sequence counts were too brittle to small variations and could be dominated by repetition.
- Moved toward seqNMF to identify recurring sequence structure, then used observed paths and movies to refine interpretable themes.

Historical amph examples included rearing/sniffing sequences, running/turning sequences and grooming/resting sequences. Early WT/KI differences were exploratory, rather than transferable evidence for Karen’s cohort.

## Slide 9 — What the seqNMF workflow can now examine

- Recruitment: how often a theme occurs and how much valid recording time it occupies.
- Within-chain execution: timing, movement, substitutions, interruptions and branching after eligible roots.
- Between-chain organisation: repetition, continuation, transitions, returns and preceding context.
- Behavioural meaning: compare selected exemplars, nearby executions and random examples while preserving actual video timing.

Longer-term extension: align photometry to chain starts, repetition, switching and execution differences. Slower fluorescence variation is not automatically a validated tonic-dopamine measure.

## Slide 10 — Current Karen delivery

- Pooled existing data across 99 recordings and 21 mice; compared vocabularies, component capacities and three/five-bout templates.
- Produced five provisional refined themes: travel–turn–travel; orientation–sniff–orientation; sniff–observe–sniff; rear–back-off; groom–reorient–groom.
- Retrieved 7,319 complete overlapping cores, with physical union coverage of 8.57% of valid time, and generated tables, figures and review galleries.
- Latest showcase: three independently playable rows per theme, each showing an exemplar and two neighbours from three different mice.

Speaker note: these are human-refined observed cores informed by pooled seqNMF catalogues, not automatic acceptance of all factor detections. Overlapping core counts are not independent observations. Selected close neighbours illustrate local consistency rather than whole-population quality.

## Slide 11 — What is established and what remains open

- Delivered: computational workflow, frozen theme definitions, numerical readouts, media, review tools and recorded provenance/audits.
- Still open: behavioural coherence, whether sensory themes should merge, general generator quality and independent confirmation.
- Latest statistical record: 8,300 finite primary tests, 146 family-level BH results and zero global-BH results; 144 family results are within-group changes and only two are group differences.
- Pooled fitting included all mice, so it supplies no held-out-animal validation; sparse transitions, late-wave sample size and L-DOPA/wave confounding constrain interpretation.

Speaker note: the narrower exploratory atlas is delivered. Broad disease findings or a generally accepted chain generator are not established. Earlier search/catalogue p-values must not be attached to the newest themes.

## Slide 12 — How the next person should take over

- Use integrated main for new code, and preserve the original source checkpoints, empirical inputs, frozen model/occurrence artifacts, output packages and exported human feedback.
- Start by opening the five-theme showcase and reviewing behavioural coherence; record acceptance, merge or refinement decisions before further group interpretation.
- Reproduce one frozen theme from input/mask/native frame identity through measurement and movie output, then extend to the full atlas.
- Continue a small set of mouse-level intra/inter-chain questions with explicit definitions and correction policy; pursue photometry integration as a subsequent extension.

Suggested closing: “The next person inherits a functioning exploratory workflow and concrete themes to review. The immediate task is to decide which behavioural descriptions are useful, verify the frozen analysis, and then build the next scientific comparison on that foundation.”

# Practical handoff appendix

## Where to start

All paths below are relative to the named checkout.

Current source: integrated main, with the local integration worktree at `/home/fsp585/Desktop/FiberphotometryCode/.worktrees/main-integration`. Check [current task state](../tasks/CURRENT.md) before continuing; the worktree branch and remote main may advance separately.

- `docs/tasks/CURRENT.md`: accumulated dated task states; read newest entries first.
- `docs/architecture/UNIT_CONTRACT.md`: current ownership and scientific/software boundaries.
- `ARCHITECTURE_INDEX.md`: navigation into implementation owners.
- `src/fpcode`: reusable library; `studies/<project>`: scientific policy and project analyses; `ops/<purpose>`: operations; `configs`: profiles; `compat`/`archive`: historical interfaces and preserved material.

Preserved primary runtime/viewer checkout: `/home/fsp585/Desktop/FiberphotometryCode`, on `codex/modular-refactor`. It retains the original shared-library/runtime setup; do not switch or clean it while existing viewers or provenance depend on it.

Preserved delivery checkout and frozen empirical outputs: `/home/fsp585/Desktop/FiberphotometryCode/.worktrees/karen-seqnmf-delivery`. Its reviewed source is also integrated onto main; ignored outputs remain at their original locations.

- `docs/tasks/CURRENT.md`: dated Karen scientific delivery records; consult integrated main's current task state for subsequent source integration.
- `studies/karen_pff_vs_monomer/analysis/EXPLORATORY_THEME_DELIVERY_20261021.md`: latest atlas methods, findings, reproduction and limitations.
- `studies/karen_pff_vs_monomer/analysis/THEME_NEIGHBOURHOOD_SHOWCASES_20261021.md`: latest selection/media/review design.
- `outputs/karen_theme_delivery_20261021/report/index.html`: numerical report.
- `outputs/karen_pooled_p80_review_20261015/theme_showcases_20261021/index.html`: latest qualitative review gallery.
- `outputs/karen_pooled_p80_review_20261015/theme_delivery_20261021.zip`: portable report/media package, approximately 166 MiB.
- `outputs/karen_pooled_p80_review_20261015/theme_showcases_20261021.zip`: portable latest showcase, approximately 128 MiB. Extract and open `index.html`.

## Implementation owners to transfer

Under `studies/karen_pff_vs_monomer/analysis/` on integrated main, with original delivery source and outputs preserved separately:

- Fitting/reference catalogue: `reference_seqnmf.py`, `run_reference_seqnmf.py`, `run_reference_capacity.py` and their saved recipes.
- Refined observed themes: `theme_candidates.py`, frozen `library.json` and occurrence index.
- Measurements/statistics: `run_theme_delivery.py`, `theme_measurements.py`, `theme_organisation.py`, `theme_statistics.py`, `theme_baselines.py`.
- Review/media: `execution_similarity.py`, `render_theme_delivery.py`, `theme_neighbourhoods.py`, `render_theme_showcases.py`, `theme_showcase_gallery.html`.

Two different seqNMF implementations exist. Delivery’s `reference_seqnmf.py` preserves the recovered September update policy, with boundary safeguards. Shared `src/fpcode/behavior/episodes/seqnmf.py` uses a different data-dependent competition penalty. Their parameters and outputs are not interchangeable. The older presentation also combines different vocabularies/fits and selected observed-path movies; it is not a single universal recipe.

## Preservation priorities before departure

1. Retain the captured primary and delivery source checkpoints (`31326cc` and `2c43b75`) and verified recovery bundles. Their manifests and restore checks are recorded in `docs/tasks/CONSOLIDATION_PROGRESS_20261021.md` and primary `outputs/consolidation_20261021`. Preserve later work separately; do not reset or clean the original worktrees.
2. Preserve the shared-library provenance of the frozen 21 October atlas execution: its editable install resolved shared `fpcode` imports from the primary checkout. Consult `runtime_library_sources.json` and the output source snapshots for actual paths/hashes. For new runs, verify import origins against the selected integrated checkout.
3. Transfer ignored empirical material separately: tracking/frame caches, video, masks, mappings, fitted models, source arrays and frozen inputs. Git and the portable viewing ZIPs are not a full reproducibility backup.
4. Export browser feedback JSON from every relevant gallery before changing machine or browser origin. Local autosave is not shared lab storage.
5. Record environment, dependency versions, storage mounts, path configuration and access arrangements. `FP_DATA_ROOT` points to the mirror containing `data/`, not to `data/` itself.
6. Choose named maintainers for the shared framework and Karen scientific analysis. Suggested roles are a software owner, a scientific interpretation owner and an input/storage owner; no assignments have been made here.

## Suggested first handover session

- Demonstrate one photometry notebook to explain the original use case.
- Show synchronised tracking/syllable video to explain the input and clock conventions.
- Open the latest five-theme gallery and discuss one convincing and one ambiguous example.
- Trace a theme into its frozen occurrence table, physical measurements and statistical row.
- Agree on the small next question, review decisions and who owns each follow-up.

The original delivery records report 933 CPU tests passed, architecture checks with no violations/cycles, and media/browser audits. These remain dated evidence for that delivery. Exact subsequent source-integration checks and their limits are recorded in the [main-promotion checkpoint](../../registry/checkpoints/main_promotion_20261021.json); CPU checks do not replace empirical or media validation. A successor should first validate their own setup with the read-only doctor and then choose focused checks before empirical execution.

# Evidence used

- Supplied attachment: `/home/fsp585/.codex/attachments/ec256bc8-3a60-4c1c-9fa7-4ced749a1d86/gether_samlet.txt`. Includes reports from late 2024 through September 2026, a blank template, overlapping September versions and proposed future work. The more detailed July–September report provides the attributed benchmark/storage figures.
- Original local branch histories/tree listings for main, scaffolding-refactor, legacy-consolidation, modular-refactor and Karen delivery; merge-base checks established ancestry. Subsequent refreshed remote refs, pull-request promotions and recovery evidence are recorded in the [main-promotion checkpoint](../../registry/checkpoints/main_promotion_20261021.json).
- Selected original session-loading/Turnlab source and current seqNMF/theme source.
- `docs/reports/REFACTOR_PRESENTATION_AND_AGENT_WORKFLOWS_20261011.md`: historical refactor comparison, interpreted alongside current contract/status rather than as current completion claims.
- Delivery `studies/sequence_viz/baselines/seqnmf_sep14/REPLAY_20261015.md`: saved-reference replay matched 2,136 calls while documenting native-boundary and endpoint issues. It supersedes the earlier recovery README’s unreplayed status for that specific saved reference.
- Delivery current task state and the exploratory-theme/showcase reports named above.
