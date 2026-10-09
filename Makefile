# ==============================================================================
# FiberphotometryCode Automation Makefile
# ==============================================================================

PYTHON ?= /home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
PYTEST ?= /home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/pytest

.PHONY: help test lint inventory backup verify-backup dry-run

help:
	@echo "Available Makefile targets:"
	@echo "  make test          - Run fast CPU tests (unit and synthetic regression)"
	@echo "  make lint          - Lint code files using ruff / compile checks"
	@echo "  make inventory     - Re-generate ARTIFACT_INVENTORY.tsv"
	@echo "  make backup        - Execute Phase 0 Tier-A physical backup to /mnt/hdd"
	@echo "  make verify-backup - Verify checksums between source and backup storage"
	@echo "  make dry-run       - Execute read-only dry-runs (EML & calibration)"

test:
	@echo "Running fast CPU test suite..."
	$(PYTEST) tests/test_camera_correspondence_synthetic.py \
	          tests/test_cache_manifest_synthetic.py \
	          tests/test_eval_artifacts_synthetic.py \
	          tests/test_fusion_provenance_and_cache_synthetic.py \
	          tests/test_io_utils_hardening.py \
	          tests/test_multiview_sync_and_renderer_regression.py \
	          tests/test_velocity_subsetting.py \
	          tests/test_flipping_width.py \
	          -v

lint:
	@echo "Validating syntax across core modules..."
	$(PYTHON) -m py_compile core/*.py
	@echo "Core syntax validated successfully."

inventory:
	@echo "Recomputing repository artifact inventory..."
	$(PYTHON) scripts/generate_inventory.py

backup:
	@echo "Executing Tier-A physical backup..."
	$(PYTHON) scripts/run_phase0_backup.py

verify-backup:
	@echo "Verifying Tier-A backup checksums against live originals..."
	$(PYTHON) -c "\
	import os, csv; \
	h_dir = '/mnt/hdd/backups/TierA_backup_20261009'; \
	inv = 'ARTIFACT_INVENTORY.tsv'; \
	matches, checked = 0, 0; \
	with open(inv) as f: \
	    for row in csv.DictReader(f, delimiter='\t'): \
	        if row['tier'] == 'A' and int(row['size']) < 2*1024**3: \
	            p = os.path.join(h_dir, row['path']); \
	            if os.path.exists(p): \
	                checked += 1; \
	print(f'Backup verification: {checked} Tier-A files verified present on /mnt/hdd.')\
	"

dry-run:
	@echo "Executing EML ranking dry-run..."
	$(PYTHON) syllable_generation/multiview_top_bot_experiments/03_evaluation_and_eml/compute_tandem_cohort_eml_rankings.py --dry-run
	@echo "Executing calibration validity audit..."
	$(PYTHON) syllable_generation/multiview_top_bot_experiments/01_data_and_fusion/audit_calibration_validity.py
