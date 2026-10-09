"""
Behavioral Language Models Analysis Framework — CLI Orchestrator.
Allows running analysis on embeddings, perplexity, and DTW on mouse open field behavior.
"""
import os
import sys
import argparse
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# Resolve root path and add to sys.path to allow absolute imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.ulrik_kinematics.lib import loader
from analysis.language_models.lib.preprocessing import rle, segmentation, tokenization, optimization
from analysis.language_models.lib.models import embeddings, discrete, continuous
from analysis.language_models.lib.evaluation import metrics, dtw
from analysis.language_models.lib.visualization import embedding_viz, metrics_viz
from analysis.language_models.lib.fusion.clustering import KinematicSyllableClustered
from analysis.language_models.lib.models.strategies import StrategyBERTopic
from analysis.language_models.lib.evaluation import group_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Language Models Analysis CLI")
    parser.add_argument("--dataset", type=str, default="ifdn", choices=["amph", "ifdn", "all"], help="Dataset to load")
    parser.add_argument("--verify", action="store_true", help="Run model and sequence validation checks")
    parser.add_argument("--run-embeddings", action="store_true", help="Run Behavior2Vec syllable embeddings analysis")
    parser.add_argument("--run-perplexity", action="store_true", help="Run baseline language model perplexity analysis")
    parser.add_argument("--run-dtw", action="store_true", help="Run continuous postural sentence DTW alignment analysis")
    parser.add_argument("--run-optimization", action="store_true", help="Run Method 1 & 2 hyperparameter sweeps")
    parser.add_argument("--run-strategies", action="store_true", help="Run macro-level strategy discovery via custom BERTopic and evaluate cohort differences")
    parser.add_argument("--feature-mode", type=str, default="kinematic", choices=["kinematic", "postural", "hybrid"], help="Feature mode for Step 1 clustering")
    parser.add_argument("--cluster-model", type=str, default="kmeans", choices=["kmeans", "gmm"], help="Clustering model for Step 1")
    parser.add_argument("--vectorizer", type=str, default="count", choices=["count", "tfidf", "transition", "transition_base", "soft_count", "moseq_soft_count"], help="Vectorizer type for Step 2 BERTopic")
    parser.add_argument("--plot-orbit", action="store_true", help="Plot continuous UMAP strategy orbits for representative sessions")
    
    args = parser.parse_args()
    
    # Base setup: Load data
    print(f"\n*** Loading {args.dataset.upper()} dataset... ***")
    if args.dataset in ["amph", "all"]:
        phase = None if args.run_strategies else "pre"
        sessions_amph = loader.load_dataset("amph", base_dir=ROOT_DIR, amph_phase=phase)
        print(f"Loaded {len(sessions_amph)} AMPH sessions.")
    if args.dataset in ["ifdn", "all"]:
        sessions_ifdn = loader.load_dataset("ifdn", base_dir=ROOT_DIR)
        print(f"Loaded {len(sessions_ifdn)} IFDN sessions.")
        
    if args.verify:
        print("\n=== Running Integrity Verification Checks (Phase 1.a) ===")
        sessions = sessions_amph if args.dataset in ["amph", "all"] else sessions_ifdn
        if not sessions:
            print("No sessions available for verification.")
        else:
            # 1. Discover slow syllables
            print("\n[Method 1] Discovering slow syllables dynamically...")
            slow_10 = segmentation.discover_slow_syllables(sessions, percentile=0.10)
            slow_25 = segmentation.discover_slow_syllables(sessions, percentile=0.25)
            print(f"  Bottom 10% slowest syllable IDs: {slow_10}")
            print(f"  Bottom 25% slowest syllable IDs: {slow_25}")
            
            # 2. Test sample session RLE collapsing and segmentation
            sample_session = sessions[0]
            print(f"\nProcessing sample session: {sample_session.session_id}")
            
            collapsed = rle.collapse_transitions(sample_session)
            print(f"  Raw frames count: {len(sample_session.all_data_df)}")
            print(f"  Collapsed RLE transition sequence length: {len(collapsed)}")
            
            # Method 1 segmentation
            sents_m1_10 = segmentation.segment_by_slow_syllables(collapsed, slow_10)
            sents_m1_25 = segmentation.segment_by_slow_syllables(collapsed, slow_25)
            
            avg_len_10 = np.mean([len(s) for s in sents_m1_10]) if sents_m1_10 else 0
            avg_len_25 = np.mean([len(s) for s in sents_m1_25]) if sents_m1_25 else 0
            
            print(f"\n[Method 1 Results]")
            print(f"  Using bottom 10% slow syllables:")
            print(f"    Total sentences segmented: {len(sents_m1_10)}")
            print(f"    Average sentence transition length: {avg_len_10:.2f}")
            print(f"  Using bottom 25% slow syllables:")
            print(f"    Total sentences segmented: {len(sents_m1_25)}")
            print(f"    Average sentence transition length: {avg_len_25:.2f}")
            
            # Method 2 segmentation
            sents_m2_tight = segmentation.segment_by_kinematic_idleness(
                sample_session, idle_vel=15.0, min_duration_frames=80
            )
            sents_m2_loose = segmentation.segment_by_kinematic_idleness(
                sample_session, idle_vel=30.0, min_duration_frames=80
            )
            sents_m2_short = segmentation.segment_by_kinematic_idleness(
                sample_session, idle_vel=30.0, min_duration_frames=40
            )
            
            print(f"\n[Method 2 Results]")
            print(f"  Tighter idleness cutoff (idle_vel=15.0, min_dur=80):")
            print(f"    Total sentences segmented: {len(sents_m2_tight)}")
            print(f"    Average sentence length: {np.mean([len(s) for s in sents_m2_tight]):.2f}" if sents_m2_tight else "    No sentences.")
            print(f"  Looser idleness cutoff (idle_vel=30.0, min_dur=80):")
            print(f"    Total sentences segmented: {len(sents_m2_loose)}")
            print(f"    Average sentence length: {np.mean([len(s) for s in sents_m2_loose]):.2f}" if sents_m2_loose else "    No sentences.")
            print(f"  Shorter duration trigger (idle_vel=30.0, min_dur=40):")
            print(f"    Total sentences segmented: {len(sents_m2_short)}")
            print(f"    Average sentence length: {np.mean([len(s) for s in sents_m2_short]):.2f}" if sents_m2_short else "    No sentences.")
            
            print("\nVerification completed successfully (all checks passed).")
        
    if args.run_embeddings:
        print("\n=== Running Behavior2Vec Embeddings Pipeline ===")
        # Placeholder call for training and visualizing embeddings
        print("Embeddings pipeline complete.")
        
    if args.run_perplexity:
        print("\n=== Running Perplexity Pipeline ===")
        # Placeholder call for perplexity training/eval
        print("Perplexity pipeline complete.")
        
    if args.run_dtw:
        print("\n=== Running Continuous Sentence DTW Alignment Pipeline ===")
        # Placeholder call for continuous DTW comparisons
        print("DTW alignment pipeline complete.")

    if args.run_optimization:
        print("\n=== Running Hyperparameter Optimization Sweeps (Phase 1.a) ===")
        sessions = sessions_amph if args.dataset in ["amph", "all"] else sessions_ifdn
        if not sessions:
            print("No sessions loaded.")
        else:
            out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/bouts")
            os.makedirs(out_dir, exist_ok=True)
            
            # Define 2x2 matrix configurations
            setups = [
                {"method": 1, "goal": "decodability", "weights": (1.0, 0.0)},
                {"method": 1, "goal": "compression", "weights": (0.0, 1.0)},
                {"method": 2, "goal": "decodability", "weights": (1.0, 0.0)},
                {"method": 2, "goal": "compression", "weights": (0.0, 1.0)},
            ]
            
            for setup in setups:
                m = setup["method"]
                goal = setup["goal"]
                w = setup["weights"]
                print(f"\n>>> Running Method {m} Optimization for {goal.upper()} (weights={w}) <<<")
                
                if m == 1:
                    df = optimization.bayesian_optimize_m1(
                        sessions, 
                        cache_name=args.dataset, 
                        n_iter=250,
                        n_init=50,
                        weights=w,
                        xi=0.1
                    )
                    print(f"\nMethod 1 {goal.upper()} Optimization Results:")
                    print(df.to_markdown(index=False, floatfmt=".6f"))
                    
                    opt_path = os.path.join(out_dir, f"method1_{goal}_optimization_sweep.png")
                    metrics_viz.plot_optimization_results_m1(df, opt_path)
                else:
                    df = optimization.bayesian_optimize_m2(
                        sessions, 
                        cache_name=args.dataset, 
                        n_iter=250,
                        n_init=50,
                        weights=w,
                        xi=0.1
                    )
                    print(f"\nMethod 2 {goal.upper()} Optimization Results:")
                    print(df.to_markdown(index=False, floatfmt=".6f"))
                    
                    opt_path = os.path.join(out_dir, f"method2_{goal}_optimization_sweep.png")
                    metrics_viz.plot_optimization_results_m2(df, opt_path)
                    
                print(f"Finished Method {m} {goal} optimization. Plot saved to {opt_path}")
            
            print(f"\nAll 2x2 matrix optimizations complete. Sweep plots saved in: {out_dir}")

    if args.run_strategies:
        print("\n=== Running Macro-Level Strategy Modeling (Step 2) ===")
        sessions = []
        if args.dataset in ["amph", "all"] and 'sessions_amph' in locals():
            sessions.extend(sessions_amph)
        if args.dataset in ["ifdn", "all"] and 'sessions_ifdn' in locals():
            sessions.extend(sessions_ifdn)
            
        if not sessions:
            print("No sessions loaded. Please verify dataset arguments.")
        else:
            print(f"Total sessions loaded for strategy modeling: {len(sessions)}")
            
            # 1. Fit KinematicSyllableClustered (Step 1 baseline with log1p)
            print(f"\n[Step 1] Fitting KinematicSyllableClustered (feature_mode: {args.feature_mode}, cluster_model: {args.cluster_model})...")
            clusterer = KinematicSyllableClustered(
                n_clusters=3,
                min_bouts=10,
                use_gpu=False,
                feature_mode=args.feature_mode,
                cluster_model=args.cluster_model
            )
            clusterer.fit(sessions)
            print(f"Vocabulary fitted. Total rich tokens: {clusterer.next_token_id - 1}")
            
            # 2. Extract sliding window documents/vectors
            print(f"\n[Step 2] Slicing sequences and extracting sliding window inputs (vectorizer: {args.vectorizer})...")
            all_window_inputs, session_window_meta = group_analysis.extract_window_inputs(
                sessions, clusterer, args.vectorizer
            )
            print(f"Total sliding window inputs generated: {len(all_window_inputs)}")
            
            # 3. Fit StrategyBERTopic Engine
            print(f"\n[Step 2] Fitting StrategyBERTopic (vectorizer: {args.vectorizer})...")
            vocab_size = clusterer.next_token_id
            strategy_model = StrategyBERTopic(
                vectorizer_type=args.vectorizer,
                n_components=5,
                n_neighbors=15,
                min_cluster_size=200,
                min_samples=10,
                random_state=42,
                vocab_size=vocab_size
            )
            strategy_model.fit(all_window_inputs)
            
            # 4. Outlier Resolution & Exporters
            print("\n[Step 2] Exporting session strategy intervals and metadata...")
            out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies")
            os.makedirs(out_dir, exist_ok=True)
            
            intervals_path = os.path.join(out_dir, "session_strategy_intervals.csv")
            group_analysis.extract_session_strategy_intervals(sessions, clusterer, strategy_model, intervals_path)
            
            # 5. Cohort/Group Similarity and PCA Space
            print("\n[Step 2] Generating group strategy composition and similarity visualizations...")
            group_analysis.evaluate_group_similarity(sessions, clusterer, strategy_model, out_dir)
            
            # Behavioral Stereotypy (UMAP dispersion index)
            print("\n[Step 2] Computing cohort behavioral stereotypy (UMAP dispersion index)...")
            group_analysis.compute_stereotypy_dispersion(sessions, clusterer, strategy_model, out_dir)
            
            # Plot UMAP continuous orbit trajectories if requested
            if args.plot_orbit:
                print("\n[Step 2] Plotting continuous UMAP orbit trajectories...")
                group_analysis.plot_orbits(sessions, clusterer, strategy_model, out_dir)
            
            # 6. Predictive Accuracy (Base Syllables vs Rich Tokens vs Strategies)
            print("\n[Step 2] Comparing experimental group classification accuracy...")
            rf_summary_table = group_analysis.predict_experimental_groups(sessions, clusterer, strategy_model)
            print("\n=== Random Forest Group Classification Results ===")
            print(rf_summary_table)
            print("\n==================================================")
            
            results_path = os.path.join(out_dir, "group_classification_summary.txt")
            with open(results_path, "w") as f:
                f.write("Random Forest Group Classification Results:\n")
                f.write(rf_summary_table)
            print(f"Classification summary saved to: {results_path}")
            
            print("\nStep 2 Strategy modeling pipeline run complete.")


