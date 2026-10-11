PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff
BACKUP_DEST ?= /mnt/hdd/backups/FiberphotometryCode
SNAPSHOT ?=
RESTORE_DEST ?=
VALIDATION_PROJECT ?=
VALIDATION_ARGS ?=

.PHONY: help test lint architecture smoke-runtime validate-two-sessions backup-plan backup verify-backup restore-backup shims-report metrics

help:
	@echo "test, lint, architecture, smoke-runtime, validate-two-sessions, backup-plan, backup, verify-backup, restore-backup, shims-report, metrics"

test:
	$(PYTEST) -q

lint:
	$(RUFF) check src/fpcode pipelines ops/backup ops/repo/dependencies.py

architecture:
	$(PYTHON) -B ops/repo/dependencies.py

# Explicit optional gate, separate from the fast functional suite.
smoke-runtime:
	$(PYTHON) -B ops/diagnostics/smoke_runtime.py

validate-two-sessions:
	@test -n "$(VALIDATION_PROJECT)" || (echo 'Supply VALIDATION_PROJECT=/path/to/pilot/project'; exit 2)
	$(PYTHON) -B ops/diagnostics/validate_two_sessions.py --project "$(VALIDATION_PROJECT)" $(VALIDATION_ARGS)

backup-plan:
	$(PYTHON) -B ops/backup/snapshot.py plan --root "$(CURDIR)" --policy configs/backup.yaml

backup:
	$(PYTHON) -B ops/backup/snapshot.py create --root "$(CURDIR)" --destination "$(BACKUP_DEST)" --policy configs/backup.yaml

verify-backup:
	@test -n "$(SNAPSHOT)" || (echo 'Supply SNAPSHOT=/path/to/sealed/snapshot'; exit 2)
	$(PYTHON) -B ops/backup/snapshot.py verify --snapshot "$(SNAPSHOT)"

restore-backup:
	@test -n "$(SNAPSHOT)" -a -n "$(RESTORE_DEST)" || (echo 'Supply SNAPSHOT and a NEW RESTORE_DEST'; exit 2)
	$(PYTHON) -B ops/backup/snapshot.py restore --snapshot "$(SNAPSHOT)" --destination "$(RESTORE_DEST)"

shims-report:
	$(PYTHON) -B ops/repo/shims_report.py

metrics:
	$(PYTHON) -B ops/repo/metrics.py
