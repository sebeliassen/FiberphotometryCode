import os
import glob
from PIL import Image, ImageChops
import numpy as np

def compare_images(dir1, dir2):
    files1 = set([os.path.relpath(p, dir1) for p in glob.glob(os.path.join(dir1, '**/*.png'), recursive=True)])
    files2 = set([os.path.relpath(p, dir2) for p in glob.glob(os.path.join(dir2, '**/*.png'), recursive=True)])
    
    common = files1.intersection(files2)
    only1 = files1 - files2
    only2 = files2 - files1
    
    print(f"Comparing {len(common)} common PNG files...")
    if only1: print(f"Files only in new dir: {len(only1)}")
    if only2: print(f"Files only in old dir: {len(only2)}")
    
    diffs = []
    
    for f in sorted(list(common)):
        p1 = os.path.join(dir1, f)
        p2 = os.path.join(dir2, f)
        
        try:
            img1 = Image.open(p1).convert('RGB')
            img2 = Image.open(p2).convert('RGB')
            
            if img1.size != img2.size:
                print(f"Size mismatch for {f}: {img1.size} vs {img2.size}")
                continue
                
            arr1 = np.array(img1).astype(float)
            arr2 = np.array(img2).astype(float)
            
            mae = np.mean(np.abs(arr1 - arr2))
            diffs.append((f, mae))
            
            if mae > 0.0:
                print(f"Diff > 0: {f} (MAE: {mae:.2f})")
                
        except Exception as e:
            print(f"Error comparing {f}: {e}")
            
    print("\nSummary:")
    exact_matches = sum(1 for d in diffs if d[1] == 0)
    print(f"Exact pixel matches: {exact_matches} / {len(common)}")
    if exact_matches < len(common):
        print("Note: Small differences are expected for plots like Random Forest outputs, UMAP/t-SNE/PCA, or Saliency maps because of stochasticity in the algorithms.")
        for d in sorted([d for d in diffs if d[1] > 0], key=lambda x: x[1], reverse=True):
            print(f"  - {d[0]}: MAE={d[1]:.2f} (max 255.0)")

if __name__ == '__main__':
    dir_new = '/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/pnas_replication'
    dir_old = '/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/pnas_replication_old'
    compare_images(dir_new, dir_old)
