# Monoamine Interactions

A research project focused on multi-modal analysis of neural signals during behavior.

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
