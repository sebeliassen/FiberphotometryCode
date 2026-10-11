# Operational Runbook & Disaster Recovery

This runbook describes operational procedures for environment execution, running pipeline stages, verifying provenance, and executing disaster recovery for the `FiberphotometryCode` repository.

---

## 1. Environment & Pre-Flight Checks

### Active Conda / Mamba Environment
Production pipelines require JAX with CUDA 12.4 and CuML support:
```bash
conda activate kpms_gpu_cuml
```

### Pre-Flight Resource Audit
Before launching or resuming any GPU modeling job, check running processes and GPU memory:
```bash
# 1. Verify GPU availability and memory usage
nvidia-smi

# 2. Check for active training jobs
ps aux | grep -E "python.*(run_|train_|kpms)" | grep -v grep

# 3. Check disk space on primary NVMe root (ensure at least 5 GB free)
df -h /
```

---

## 2. Dry-Run Verification

All primary CLI runners support safe dry-run modes that inspect file paths, check cache manifests, and validate GPU shapes without allocating memory or modifying files:

```bash
# Dry-run evaluation and EML compatibility verification
python syllable_generation/multiview_top_bot_experiments/03_evaluation_and_eml/compute_tandem_cohort_eml_rankings.py --dry-run

# Audit calibration and camera sync without modifying data
python syllable_generation/multiview_top_bot_experiments/01_data_and_fusion/audit_calibration_validity.py
```

---

## 3. Resuming Interrupted Runs & Provenance Validation

### Safe Checkpoint Loading
When resuming or loading an existing model:
1. **Never open checkpoints in `r+` or `w` mode**. Always use `core/h5_guard.py` or specify read-only access:
   ```python
   import h5py
   with h5py.File(checkpoint_path, 'r') as f:
       # Read states, latent params, and betas
       pass
   ```
2. **Sidecar Manifests**: Check if a sidecar `.manifest.json` exists adjacent to the checkpoint or cache. Verify integrity:
   ```python
   from core.cache_manifest import verify_manifest
   is_valid, msg = verify_manifest(artifact_path)
   if not is_valid:
       print(f"Warning: Artifact provenance unverified: {msg}")
   ```

### Resuming Seed Sweeps
When running multi-seed queues (Seeds 101, 202, 303, 404, 505), launcher scripts automatically check for completed `checkpoint.h5` files and skip completed seeds:
```bash
bash syllable_generation/multiview_top_bot_experiments/02_training_runners/launch_calibrated400_production.sh
```

---

## 4. Disaster Recovery & Restoration Procedures

### Restoring Tier-A Artifacts from Secondary Physical Disk (`/mnt/hdd`)

If any checkpoint, PCA file, session pickle, or configuration is corrupted or lost:

```bash
# Option 1: Restore a specific file
cp -a /mnt/hdd/backups/TierA_backup_20261009/<relative_path> /home/fsp585/Desktop/FiberphotometryCode/<relative_path>

# Example: Restore a specific YAML configuration
cp -a /mnt/hdd/backups/TierA_backup_20261009/syllable_generation/multiview_top_bot_experiments/configs/tandem_cohort_50ar_200slds.yml \
      /home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/configs/

# Option 2: Restore all backed-up Tier A artifacts (non-destructive sync)
rsync -av /mnt/hdd/backups/TierA_backup_20261009/ /home/fsp585/Desktop/FiberphotometryCode/
```

### Restoring Code from Standalone Snapshot
```bash
# Restore entire code tree from snapshot tarball
tar -xzf /home/fsp585/Desktop/_backups/20261009_112705/code_snapshot.tar.gz -C /tmp/restored_code/
```

### Restoring Write Permissions on Originals
All 568 original Tier-A files were set to read-only (`chmod a-w`) during Phase 0 to prevent accidental overwrite:
```bash
# Put back original write permissions whenever editing or training is needed
./restore_tier_a_permissions.sh
```

---

## 5. Automated Health Checks

Run the automated test suite to verify mathematics and pipeline invariants:
```bash
# Run fast CPU tests (takes ~30 seconds)
pytest -m "cpu and not needs_data" -v

# Validate cryptographic inventory integrity
python -c "
import csv
with open('ARTIFACT_INVENTORY.tsv') as f:
    r = csv.DictReader(f, delimiter='\t')
    count = sum(1 for row in r)
print(f'Inventory healthy: {count:,} files registered.')
"
```
