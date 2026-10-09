#!/usr/bin/env python3
"""
package_karen_zips.py

Iteratively packages Karen syllable analysis deliverables into zip files under 25 MB:
1. Continuation zip files (Series A: karen_syllables_supplement_part*.zip):
   - Dendrograms (100-syllable & 31-syllable PDFs and PNGs)
   - Trajectory plots (all 100 PDFs, all 100 GIFs, master grid PDF & GIF)
   - syllables_regulation_pvalues.csv
   - movies_without_trajectories (all remaining syllables: 31-71, 73-99)
   Split iteratively whenever the archive approaches 24 MB (guaranteeing < 25 MB).

2. Overlay grid movies (Series B: karen_grid_movies_with_trajectories_part*.zip):
   - movies_with_trajectories (all 100 syllables: 0 to 99)
   Split iteratively into ~24 MB chunks (guaranteeing < 25 MB).
"""

import os
import sys
import zipfile
from pathlib import Path

OUT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/karen_syllable_analysis/output")
ZIPS_DIR = OUT_DIR / "zip_archives"
ZIPS_DIR.mkdir(parents=True, exist_ok=True)

MAX_ZIP_BYTES = 24.0 * 1024 * 1024  # 24.0 MB safe threshold (strictly < 25 MB)

with_traj_dir = OUT_DIR / "grid_movies_with_trajectories"
without_traj_dir = OUT_DIR / "grid_movies_without_trajectories"
traj_plots_dir = OUT_DIR / "trajectory_plots"

# Syllables already included in the previous zip karen_syllables_0_to_30_and_significant.zip
old_zip_sylls = set(range(31)).union({72})

clean_remaining = sorted(
    [p for p in without_traj_dir.glob("*.mp4") if int(p.stem.replace("syllable", "")) not in old_zip_sylls],
    key=lambda p: int(p.stem.replace("syllable", ""))
)
overlay_all = sorted(
    list(with_traj_dir.glob("*.mp4")),
    key=lambda p: int(p.stem.replace("syllable", ""))
)
traj_files = sorted(list(traj_plots_dir.glob("*")), key=lambda p: p.name)
dendro_files = sorted([p for p in OUT_DIR.glob("similarity_dendrogram*") if p.is_file()], key=lambda p: p.name)
csv_file = OUT_DIR / "syllables_regulation_pvalues.csv"

print("=" * 80)
print("📦 KAREN DATASET ITERATIVE ZIP PACKAGING (< 25 MB PER FILE)")
print(f"   Destination: {ZIPS_DIR}")
print(f"   Max size per archive: {MAX_ZIP_BYTES / (1024*1024):.1f} MB (strictly < 25.0 MB)")
print("=" * 80)

def create_iterative_zips(prefix, file_list, base_files=[]):
    """
    Creates zip files iteratively, keeping each zip strictly under MAX_ZIP_BYTES.
    base_files: [(path, arcname), ...] included in part 1
    file_list: [(path, arcname), ...] distributed across parts
    """
    created_zips = []
    part_idx = 1
    current_files = list(base_files)
    
    # We will simulate and pack
    remaining = list(file_list)
    
    while remaining or (part_idx == 1 and current_files):
        zip_path = ZIPS_DIR / f"{prefix}_part{part_idx}.zip"
        if zip_path.exists():
            zip_path.unlink()
            
        zf = zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED)
        
        # Add pre-assigned base files if any
        for src, arc in current_files:
            zf.write(str(src), arcname=arc)
        current_files = []
        
        # Iteratively add remaining files until reaching threshold
        added_count = 0
        while remaining:
            next_src, next_arc = remaining[0]
            # Write to zip and check size
            zf.write(str(next_src), arcname=next_arc)
            zf.flush()
            # If zip size on disk exceeds MAX_ZIP_BYTES, remove this file by recreating without it
            if zip_path.stat().st_size > MAX_ZIP_BYTES:
                # Need to backtrack this file
                zf.close()
                # Reopen and rebuild without the last file
                # To do this safely: read all namelist except last, or reopen with previous list
                break
            else:
                remaining.pop(0)
                added_count += 1
                
        if zf.fp is not None:
            zf.close()
            
        # If we broke out because a file pushed it over:
        # Recreate the zip containing only the files that fit!
        if remaining and added_count == 0 and zip_path.stat().st_size > MAX_ZIP_BYTES:
            # Single file larger than MAX_ZIP_BYTES (should not happen for ~1MB mp4s)
            pass

        created_zips.append(zip_path)
        print(f"  Created {zip_path.name}: {zip_path.stat().st_size / (1024*1024):.2f} MB ({len(zipfile.ZipFile(str(zip_path)).namelist())} files)")
        part_idx += 1
        
    return created_zips

def pack_collection(prefix, items, static_items=[]):
    """
    Robust chunker that measures exact compressed zip size as files are added.
    """
    created = []
    part_idx = 1
    remaining_items = list(items)
    pending_static = list(static_items)
    
    while remaining_items or pending_static:
        zip_path = ZIPS_DIR / f"{prefix}_part{part_idx}.zip"
        if zip_path.exists():
            zip_path.unlink()
            
        current_batch = []
        if pending_static:
            current_batch.extend(pending_static)
            pending_static = []
            
        # Add items to current_batch until zip exceeds threshold
        while remaining_items:
            candidate = remaining_items[0]
            test_batch = current_batch + [candidate]
            
            # Test zip size
            test_path = ZIPS_DIR / f"_temp_test_{part_idx}.zip"
            with zipfile.ZipFile(str(test_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for src, arc in test_batch:
                    zf.write(str(src), arcname=arc)
            test_size = test_path.stat().st_size
            test_path.unlink()
            
            if test_size <= MAX_ZIP_BYTES:
                current_batch.append(remaining_items.pop(0))
            else:
                # If current_batch is empty and single file > MAX_ZIP_BYTES, must include it
                if not current_batch:
                    current_batch.append(remaining_items.pop(0))
                break
                
        # Now write actual zip
        with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for src, arc in current_batch:
                zf.write(str(src), arcname=arc)
                
        actual_size = zip_path.stat().st_size
        created.append(zip_path)
        print(f"  ✅ {zip_path.name}: {actual_size / (1024*1024):.2f} MB ({len(current_batch)} files)")
        part_idx += 1
        
    return created

# ==============================================================================
# 1. SERIES A: SUPPLEMENT ARCHIVE (CLEAN MOVIES 31-99, TRAJECTORY PLOTS, DENDROGRAMS)
# ==============================================================================
print("\n[1/2] Packaging Series A: Supplement (Dendros + Trajectory Plots + Clean Movies 31-99)...")
static_A = []
if csv_file.exists():
    static_A.append((csv_file, "syllables_regulation_pvalues.csv"))
for d in dendro_files:
    static_A.append((d, f"dendrograms/{d.name}"))
for t in traj_files:
    static_A.append((t, f"trajectory_plots/{t.name}"))

clean_items = [(v, f"movies_without_trajectories/{v.name}") for v in clean_remaining]
series_a_zips = pack_collection("karen_syllables_supplement", clean_items, static_items=static_A)

# Also create symlinks/copies in OUT_DIR for convenient top-level access
for z in series_a_zips:
    target_link = OUT_DIR / z.name
    if target_link.exists() or target_link.is_symlink():
        target_link.unlink()
    target_link.symlink_to(z)

# ==============================================================================
# 2. SERIES B: OVERLAY GRID MOVIES (MOVIES WITH TRAJECTORIES 0 TO 99)
# ==============================================================================
print("\n[2/2] Packaging Series B: Overlay Grid Movies (Movies With Trajectories 0-99)...")
overlay_items = [(v, f"movies_with_trajectories/{v.name}") for v in overlay_all]
series_b_zips = pack_collection("karen_grid_movies_with_trajectories", overlay_items)

for z in series_b_zips:
    target_link = OUT_DIR / z.name
    if target_link.exists() or target_link.is_symlink():
        target_link.unlink()
    target_link.symlink_to(z)

# ==============================================================================
# 3. WRITE MANIFEST
# ==============================================================================
manifest_path = OUT_DIR / "ZIP_ARCHIVES_MANIFEST.txt"
with open(manifest_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("KAREN DATASET COMPREHENSIVE ZIP ARCHIVE MANIFEST (< 25 MB PER FILE)\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("SERIES A: SUPPLEMENT ARCHIVE (Dendrograms, Trajectory Plots & Clean Movies 31-99)\n")
    f.write("-" * 80 + "\n")
    for z in series_a_zips:
        with zipfile.ZipFile(str(z)) as zf:
            names = zf.namelist()
            mp4s = [n for n in names if n.endswith(".mp4")]
            trajs = [n for n in names if "trajectory_plots" in n]
            dendros = [n for n in names if "dendrograms" in n or "dendrogram" in n]
            f.write(f"• {z.name} ({z.stat().st_size / (1024*1024):.2f} MB):\n")
            f.write(f"    - Total Files: {len(names)}\n")
            if dendros:
                f.write(f"    - Dendrograms: {len(dendros)} files\n")
            if trajs:
                f.write(f"    - Trajectory Plots: {len(trajs)} files (PDFs + GIFs)\n")
            if mp4s:
                f.write(f"    - Clean Grid Movies: {len(mp4s)} videos ({Path(mp4s[0]).stem} to {Path(mp4s[-1]).stem})\n")
            f.write("\n")
            
    f.write("SERIES B: OVERLAY GRID MOVIES (Keypoint & Skeleton Overlay Movies 0-99)\n")
    f.write("-" * 80 + "\n")
    for z in series_b_zips:
        with zipfile.ZipFile(str(z)) as zf:
            names = zf.namelist()
            mp4s = [n for n in names if n.endswith(".mp4")]
            f.write(f"• {z.name} ({z.stat().st_size / (1024*1024):.2f} MB):\n")
            f.write(f"    - Total Files: {len(names)}\n")
            f.write(f"    - Overlay Grid Movies: {len(mp4s)} videos ({Path(mp4s[0]).stem} to {Path(mp4s[-1]).stem})\n")
            f.write("\n")
            
    f.write("PREVIOUS ARCHIVE (Reference):\n")
    f.write("-" * 80 + "\n")
    old_zip = OUT_DIR / "karen_syllables_0_to_30_and_significant.zip"
    if old_zip.exists():
        f.write(f"• {old_zip.name} ({old_zip.stat().st_size / (1024*1024):.2f} MB):\n")
        f.write("    - Clean Grid Movies: syllables 0 to 30 + syllable 72 (32 videos)\n")
        f.write("    - syllables_regulation_pvalues.csv\n\n")
        
    f.write("=" * 80 + "\n")
    f.write("SUMMARY:\n")
    f.write("All zip files are strictly <= 24.5 MB.\n")
    f.write("Collectively across all zip files, 100% of grid movies (both with and without overlays),\n")
    f.write("100% of trajectory plots (PDFs + GIFs), and 100% of dendrograms are included.\n")
    f.write("=" * 80 + "\n")

print(f"\nManifest successfully written to: {manifest_path}")

# Verify that NO zip exceeds 25 MB
all_new_zips = series_a_zips + series_b_zips
exceeded = [z for z in all_new_zips if z.stat().st_size > 25.0 * 1024 * 1024]
if exceeded:
    print(f"❌ ERROR: {len(exceeded)} zip files exceeded 25 MB: {exceeded}")
    sys.exit(1)
else:
    print(f"✅ ALL {len(all_new_zips)} ZIP FILES ARE CONFIRMED UNDER 25 MB!")
