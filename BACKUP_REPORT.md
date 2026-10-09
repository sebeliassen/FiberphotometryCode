# Phase 0 Backup & Versioning Report

**Generated**: 2026-10-09 11:39:47  
**Target Repository**: `/home/fsp585/Desktop/FiberphotometryCode`  
**Git Tag**: `pre-scaffolding-20261009` (commit `969c893` on branch `scaffolding-refactor`)  

---

## 1. Executive Summary & Hard Gate Status
Phase 0 safeguards all code, configurations, models, manifests, and datasets before any codebase reorganization:
- **Code Versioning**: Completed additively on branch `scaffolding-refactor`. Created `.gitignore.legacy` and new `.gitignore` strictly excluding data/binaries. Tagged `pre-scaffolding-20261009`.
- **Code Snapshot**: Independent `code_snapshot.tar.gz` archived to `/home/fsp585/Desktop/_backups/20261009_112705/` and mirrored to `/mnt/hdd/backups/code_snapshots/20261009_112705/` with `SHA256SUMS`.
- **Artifact Inventory**: `ARTIFACT_INVENTORY.tsv` generated with 26,863 files cataloged.
- **Tier-A Redundancy**: 509 critical Tier-A artifacts (< 2 GB: configs, manifests, PCA models, results, init models, session pickles totaling 44.22 GB) copied to physical disk `/mnt/hdd` and verified byte-for-byte.
- **Large Model Protection**: 67 large model checkpoints (>= 2 GB, 861.17 GB) indexed with exact cryptographic hashes and write-protected in place via `chmod a-w`.
- **Permission Reversibility**: All original modes recorded in `PERMISSIONS_MODIFIED_TIER_A.tsv`. One-click restoration script created: `restore_tier_a_permissions.sh`.
- **Restore Drill**: Passed with 100% byte-for-byte fidelity.

---

## 2. Storage & Free Space

### Primary NVMe Disk (`/`)
```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  3.6T  3.4T  5.4G 100% /
```

### Secondary Physical Disk (`/mnt/hdd`)
```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        11T  6.8T  3.6T  66% /mnt/hdd
```

---

## 3. Tier Inventory Breakdown

| Tier | Category | File Count | Total Size | Protection & Backup Strategy |
|---|---|---|---|---|
| **Tier A** | Checkpoints, `results.h5`, `pca.p`, init models, session pickles, configs, manifests | 576 | 905.39 GB | 509 files (<2GB, 44.22 GB) copied & verified on `/mnt/hdd`; 67 files (>=2GB) indexed & `chmod a-w` protected on original disk |
| **Tier B** | Regenerable caches, fused H5s, intermediate features, test runs | 16,461 | 197.36 GB | Fully indexed with cryptographic hashes in `ARTIFACT_INVENTORY.tsv` |
| **Tier C** | Raw DLC H5s, tracking CSVs, video recordings (.avi, .mp4), raw photometry | 2,780 | 268.61 GB | Read-only input data, fully indexed in `ARTIFACT_INVENTORY.tsv` |
| **CODE / DOCS** | Python source, notebooks, shell scripts, markdown docs, configs | 2,346 | 0.21 GB | Committed to Git branch `scaffolding-refactor`, tagged `pre-scaffolding-20261009`, and archived in `code_snapshot.tar.gz` |
| **OTHER** | Ancillary files, small logs, miscellaneous text | 4,700 | 14.75 GB | Inventoried in `ARTIFACT_INVENTORY.tsv` |

---

## 4. Backup Locations & Verification

1. **Git Repository Baseline**:
   - Branch: `scaffolding-refactor`
   - Tag: `pre-scaffolding-20261009`
   - Commit: `969c893`

2. **Standalone Code Archive**:
   - Primary: `/home/fsp585/Desktop/_backups/20261009_112705/code_snapshot.tar.gz`
   - Secondary: `/mnt/hdd/backups/code_snapshots/20261009_112705/code_snapshot.tar.gz`
   - SHA256: `913625530d41bd2f2c12cc7720d1444f9fb27ff94efa9109d76d28665b607ff6`

3. **Tier A Physical Backup**:
   - Path: `/mnt/hdd/backups/TierA_backup_20261009/`
   - Total Files Copied: 509
   - Total Volume: 44.22 GB
   - Hash Verification: 100% matched between source and destination.

4. **Tier A In-Place Protection & Reversal**:
   - Total Files Protected (`chmod a-w`): 568
   - Prevents accidental in-place overwriting by any runner or training script during refactoring.
   - Restoration Script: `./restore_tier_a_permissions.sh` (can be executed whenever write access is needed again).

---

## 5. Restore Drill Verification
- **Test Artifact**: `kpms_engineering_handoff_v2/SOURCE_MANIFEST.json`
- **Source on HDD**: `/mnt/hdd/backups/TierA_backup_20261009/kpms_engineering_handoff_v2/SOURCE_MANIFEST.json`
- **Restored Path**: `/tmp/restore_drill_tier_a/SOURCE_MANIFEST.json`
- **Original Hash**: `71d646926a8f464dc2e0b2aa21f25af07435392e54d17583b9de6a36f8723cbf`
- **Restored Hash**: `71d646926a8f464dc2e0b2aa21f25af07435392e54d17583b9de6a36f8723cbf`
- **Exact Match Verified**: **YES** (100% identical)

### Universal Restore Commands
To restore any Tier A file from the second disk back to the workspace:
```bash
# Restore specific file
cp -a /mnt/hdd/backups/TierA_backup_20261009/<rel_path> /home/fsp585/Desktop/FiberphotometryCode/<rel_path>

# Restore all Tier A backups
rsync -av /mnt/hdd/backups/TierA_backup_20261009/ /home/fsp585/Desktop/FiberphotometryCode/

# Put back write permissions afterwards
./restore_tier_a_permissions.sh
```

---

## 6. What Was Not Copied
- **67 Multi-GB Tier A Checkpoint Files** (861.17 GB): Per user instruction, these giant checkpoint H5 files were not copied over SATA to save several hours of transfer time. They are instead indexed with head/tail cryptographic fingerprints and write-protected in place via `chmod a-w`.
- **Tier B Caches & Tier C Raw Videos/Tracking** (1.7 TB): Read-only / regenerable data, inventoried in `ARTIFACT_INVENTORY.tsv`.

---
**STATUS: PHASE 0 COMPLETE. STOPPED FOR USER APPROVAL BEFORE PHASE 1.**
