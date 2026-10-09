# ==============================================================================
# FiberphotometryCode Automation Makefile
# Reads environment paths dynamically with graceful defaults.
# ==============================================================================

PYTHON ?= $(shell which python3 2>/dev/null || echo /home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python)
PYTEST ?= $(shell which pytest 2>/dev/null || echo /home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/pytest)
RUFF ?= $(shell which ruff 2>/dev/null || echo /home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/ruff)

.PHONY: help test lint inventory backup verify-backup dry-run

help:
	@echo "Available Makefile targets:"
	@echo "  make test          - Run fast CPU tests (unit and synthetic regression)"
	@echo "  make lint          - Lint code files using ruff"
	@echo "  make inventory     - Re-generate ARTIFACT_INVENTORY.tsv"
	@echo "  make backup        - Execute Phase 0 Tier-A physical backup to /mnt/hdd"
	@echo "  make verify-backup - Verify SHA-256 hashes between source and /mnt/hdd backup"
	@echo "  make dry-run       - Execute read-only dry-runs (EML & calibration)"

test:
	@echo "Running fast CPU test suite via unified pyproject.toml configuration..."
	$(PYTEST) -v

lint:
	@echo "Running ruff linter across core codebase..."
	$(RUFF) check core

inventory:
	@echo "Recomputing repository artifact inventory..."
	$(PYTHON) scripts/generate_inventory.py

backup:
	@echo "Executing Tier-A physical backup..."
	$(PYTHON) scripts/run_phase0_backup.py

verify-backup:
	@echo "Verifying Tier-A cryptographic hashes against live originals..."
	$(PYTHON) scripts/verify_tier_a_backup.py

dry-run:
	@echo "Executing EML ranking dry-run..."
	$(PYTHON) syllable_generation/multiview_top_bot_experiments/03_evaluation_and_eml/compute_tandem_cohort_eml_rankings.py --dry-run
	@echo "Executing calibration validity audit..."
	$(PYTHON) syllable_generation/multiview_top_bot_experiments/01_data_and_fusion/audit_calibration_validity.py
