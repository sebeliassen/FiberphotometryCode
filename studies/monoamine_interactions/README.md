# Monoamine Interactions

A research project focused on multi-modal analysis of neural signals during behavior.

Current project policy lives in explicit study owners. The baseline configuration
is [`studies/oft_baseline/config.py`](../oft_baseline/config.py); the juvenile
configuration is [`studies/juvenile_intruder/config.py`](../juvenile_intruder/config.py).
The baseline hybrid discovery function in `steps/tracking.py` imports its own
configuration directly, without a bare `config` import or path injection. Both
historical policies retain the `(50000, 250000)` frame range, optional video token
and Arena chamber fallback. Their configuration instances remain independent.

The layout and setup instructions below describe the historical workspace.
Current project aliases under `projects/` point to the corresponding study owners.
The unified session builder still needs separate dynamic-import and configuration
isolation work before its complete workflow is validated. No empirical build is
implied by the filename-only synthetic discovery tests; see the
[baseline policy notes](../oft_baseline/README.md).

## Project Structure

```
monoamine_interactions/
├── loaders/               # Shared data building pipelines
│   └── build_sessions.py
├── data/                  # Shared data storage
├── analysis/              # Shared notebooks/scripts
├── context/               # Shared documentation
├── diagnostics/           # Shared QC outputs
├── scripts/               # Global utility scripts (setup, validation)
├── jonathan_oft_baseline/ # Sub-project: Open field baseline experiment
│   ├── enrich.py          # Project-specific logic
│   └── overrides/         # Project-specific metadata
└── README.md
```

## Setup Instructions

1) **Set up Python Environment**:
   Use the `kpms_gpu` conda environment.

2) **Prepare Data**:
   ```bash
   cd scripts
   ./scripts/setup_videos.sh run # Copy videos to HDD
   cd ../jonathan_oft_baseline
   ./setup_data.sh run           # Prepare CSVs and placeholders
   ./create_dummy_h5.sh         # Satisfy session discovery
   ```

3) **Build Sessions**:
   ```bash
   cd jonathan_oft_baseline/loaders
   conda run -n kpms_gpu python build_sessions.py
   ```

## Design Principles
- **Generic Core**: Logic is imported from the sibling `syllable_generation` repo.
- **Flat Overrides**: Use `overrides/*.csv` for manual corrections without changing code.
- **Sub-Project Isolation**: Each experiment has its own folder and data.
