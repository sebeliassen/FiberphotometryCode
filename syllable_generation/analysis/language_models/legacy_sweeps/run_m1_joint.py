"""
Run Method 1 (Slow Syllable Slicing) Bayesian Optimization
with joint weights w1=1.0, w3=1.0 on the AMPH pre-injection dataset.
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.ulrik_kinematics.lib import loader
from analysis.language_models.lib.preprocessing import optimization
from analysis.language_models.lib.visualization import metrics_viz

if __name__ == "__main__":
    print("*** Loading AMPH dataset (pre-phase)... ***")
    sessions = loader.load_dataset("amph", base_dir=ROOT_DIR, amph_phase="pre")
    print(f"Loaded {len(sessions)} AMPH sessions.")
    
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/bouts")
    os.makedirs(out_dir, exist_ok=True)
    
    # We run with joint weights: 1.0 for decodability, 1.0 for compression
    weights = (1.0, 1.0)
    print(f"\n>>> Running Method 1 Optimization for JOINT GOAL (weights={weights}) <<<")
    
    df = optimization.bayesian_optimize_m1(
        sessions, 
        cache_name="amph", 
        n_iter=250,
        n_init=50,
        weights=weights,
        xi=0.1
    )
    
    print("\nMethod 1 Joint Optimization Results:")
    print(df.to_markdown(index=False, floatfmt=".6f"))
    
    opt_path = os.path.join(out_dir, "method1_joint_optimization_sweep.png")
    # For visualization, we can use the same M1 plotting tool
    metrics_viz.plot_optimization_results_m1(df, opt_path)
    print(f"Finished joint optimization. Plot saved to {opt_path}")
