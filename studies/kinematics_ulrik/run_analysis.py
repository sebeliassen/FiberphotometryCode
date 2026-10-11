"""
Ulrik Kinematics Extended Pipeline — CLI Entry Point.

Output directory layout:
  output/global_metrics/     — Legacy exploration plots (run_ifdn, run_amph)
  output/pnas_replication/   — PNAS figure replications (per genotype/phase)
  output/{module}/           — Module-specific discovery plots
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    sys.path.append(os.path.join(root_dir, "syllable_generation"))

from analysis.ulrik_kinematics.lib import loader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ulrik Kinematics Extended Pipeline")
    parser.add_argument("--dataset", type=str, default="amph", choices=["amph", "ifdn", "karen", "all"], help="Dataset to load")
    parser.add_argument("--run-all", action="store_true", help="Run the entire legacy pipeline")
    parser.add_argument("--verify-motifs", action="store_true", help="Run the Motifs module verification test")
    parser.add_argument("--run-motifs-discovery", action="store_true", help="Run the Motif biological discovery phase")
    parser.add_argument("--test-significance", action="store_true", help="Test motif significance against a Markov Random Walk")
    parser.add_argument("--run-phased-discovery", action="store_true", help="Run clustering, motifs, micro, and networks on specific phases (salineamph and amphamph)")
    parser.add_argument("--verify-temporal", action="store_true", help="Run the Temporal module verification test")
    parser.add_argument("--run-temporal-discovery", action="store_true", help="Run the Temporal biological discovery phase")
    parser.add_argument("--verify-clustering", action="store_true", help="Run the Clustering topological invariance test")
    parser.add_argument("--run-clustering-discovery", action="store_true", help="Run the UMAP Genotype isolation discovery phase")
    parser.add_argument("--verify-micro", action="store_true", help="Run the Micro-Kinematics variance test")
    parser.add_argument("--run-micro-discovery", action="store_true", help="Run the Micro-Kinematics biological discovery phase")
    parser.add_argument("--verify-networks", action="store_true", help="Run the Networks Module variance test")
    parser.add_argument("--run-network-discovery", action="store_true", help="Run the Network graphical discovery phase")
    parser.add_argument("--run-pnas-replication", action="store_true", help="Run the PNAS paper replication plots")
    parser.add_argument("--run-neural-replication", action="store_true", help="Run the neural replication (Kinematic vs Neural, RFC)")
    parser.add_argument("--run-full-temporal", action="store_true", help="Run the full-session (saline + amph) temporal dynamics")
    parser.add_argument("--run-timing-stats", action="store_true", help="Run the regional timing stats (VS vs DS) and t-test")
    parser.add_argument("--verify-timing-stats", action="store_true", help="Run the regional timing stats with a broken temporal relationship to verify test failure")
    parser.add_argument("--timing-mode", type=str, default="strict", choices=["strict", "naive", "both"], help="Peak finding mode for regional timing stats (strict interior only, naive with bound fallback, or both)")
    parser.add_argument("--run-unique-info", action="store_true", help="Run the Unique Information Pipeline (kinematic stripping & residual decoding)")
    parser.add_argument("--verify-unique-info", action="store_true", help="Verify the Unique Information Pipeline on a fast subset")
    parser.add_argument("--run-temporal-saliency", action="store_true", help="Run the Temporal Saliency and Pre-Onset Planning Pipeline")
    parser.add_argument("--verify-temporal-saliency", action="store_true", help="Verify the Temporal Saliency Pipeline on a fast subset")
    parser.add_argument("--run-temporal-saliency-extended", action="store_true", help="Run the Extended Temporal Saliency Pipeline (-1s to +1s)")
    parser.add_argument("--verify-temporal-saliency-extended", action="store_true", help="Verify the Extended Temporal Saliency Pipeline on a fast subset")
    parser.add_argument("--run-saliency-modeling", action="store_true", help="Run the Biophysical Saliency Modeling & Synaptic Integration Pipeline")
    parser.add_argument("--verify-saliency-modeling", action="store_true", help="Verify the Biophysical Saliency Modeling Pipeline on a fast subset")
    parser.add_argument("--stripping-model", type=str, default="random_forest", choices=["linear", "random_forest", "gradient_boosting", "mlp"], help="Model type to use for kinematic stripping")

    args = parser.parse_args()

    if args.run_full_temporal:
        from analysis.ulrik_kinematics.lib import temporal
        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Full-Session Temporal Discovery ***")
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=None)
            temporal.run_full_session_discovery(sessions_amph, project_name="AMPH")

    if args.verify_motifs:
        from analysis.ulrik_kinematics.lib import motifs

        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Verification ***")
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
            motifs.verify_motifs_module(sessions_amph)

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for Verification ***")
            sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
            motifs.verify_motifs_module(sessions_ifdn)

    if args.run_motifs_discovery and not args.run_phased_discovery:
        from analysis.ulrik_kinematics.lib import motifs
        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Motif Discovery ***")
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
            motifs.run_discovery(sessions_amph, project_name="AMPH", style="standard")
            motifs.run_discovery(sessions_amph, project_name="AMPH", style="alphabet")

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for Motif Discovery ***")
            sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
            motifs.run_discovery(sessions_ifdn, project_name="IFDN", style="standard")
            motifs.run_discovery(sessions_ifdn, project_name="IFDN", style="alphabet")

    if args.test_significance:
        from analysis.ulrik_kinematics.lib import motifs
        if args.dataset in ["amph", "all"]:
            print("\n*** Testing Motif Significance for AMPH Dataset ***")
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
            motifs.check_significance(sessions_amph, project="AMPH", style="standard")

    if args.verify_temporal:
        from analysis.ulrik_kinematics.lib import temporal
        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Temporal Verification ***")
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
            temporal.verify_temporal_module(sessions_amph)

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for Temporal Verification ***")
            sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
            temporal.verify_temporal_module(sessions_ifdn)

    if args.run_temporal_discovery:
        from analysis.ulrik_kinematics.lib import temporal
        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Temporal Discovery ***")
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
            temporal.run_discovery(sessions_amph, project_name="AMPH")

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for Temporal Discovery ***")
            sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
            temporal.run_discovery(sessions_ifdn, project_name="IFDN")

    if args.verify_clustering:
        from analysis.ulrik_kinematics.lib import clustering
        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Clustering Verification ***")
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
            clustering.verify_clustering_module(sessions_amph)

    if args.run_clustering_discovery:
        from analysis.ulrik_kinematics.lib import clustering
        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for UMAP Discovery ***")
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
            clustering.run_discovery(sessions_amph, project_name="AMPH")
        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for UMAP Discovery ***")
            sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
            clustering.run_discovery(sessions_ifdn, project_name="IFDN")
        if args.dataset in ["karen", "all"]:
            print("\n*** Loading KAREN Dataset for UMAP Discovery ***")
            sessions_karen = loader.load_dataset("karen", base_dir=root_dir)
            clustering.run_discovery(sessions_karen, project_name="KAREN")

    if args.verify_micro:
        from analysis.ulrik_kinematics.lib import micro_kinematics
        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Micro Verification ***")
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
            micro_kinematics.verify_micro_module(sessions_amph)

    if args.run_micro_discovery and not args.run_phased_discovery:
        from analysis.ulrik_kinematics.lib import micro_kinematics
        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Micro Discovery ***")
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
            micro_kinematics.run_discovery(sessions_amph, project_name="AMPH", style="alphabet")
            micro_kinematics.run_discovery(sessions_amph, project_name="AMPH", style="standard")

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for Micro Discovery ***")
            sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
            micro_kinematics.run_discovery(sessions_ifdn, project_name="IFDN", style="alphabet")
            micro_kinematics.run_discovery(sessions_ifdn, project_name="IFDN", style="standard")

        if args.dataset in ["karen", "all"]:
            print("\n*** Loading KAREN Dataset for Micro Discovery ***")
            sessions_karen = loader.load_dataset("karen", base_dir=root_dir)
            micro_kinematics.run_discovery(sessions_karen, project_name="KAREN", style="alphabet")
            micro_kinematics.run_discovery(sessions_karen, project_name="KAREN", style="standard")

    if args.verify_networks:
        from analysis.ulrik_kinematics.lib import networks
        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Networks Verification ***")
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
            networks.verify_networks_module(sessions_amph)

    if args.run_network_discovery:
        from analysis.ulrik_kinematics.lib import networks
        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Networks Discovery ***")
            sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_post_injection_delay_min=5.0)
            networks.run_discovery(sessions_amph, project_name="AMPH", style="alphabet")
            networks.run_discovery(sessions_amph, project_name="AMPH", style="standard")

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for Networks Discovery ***")
            sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
            networks.run_discovery(sessions_ifdn, project_name="IFDN", style="alphabet")
            networks.run_discovery(sessions_ifdn, project_name="IFDN", style="standard")

    if args.run_phased_discovery:
        from analysis.ulrik_kinematics.lib import clustering, micro_kinematics, motifs

        run_all = not any([
            args.run_clustering_discovery,
            args.run_motifs_discovery,
            args.run_micro_discovery,
        ])

        for phase in ["pre", "post"]:
            print("\n" + "#"*60)
            print(f" PHASED DISCOVERY: {phase.upper()}")
            print("#"*60)
            sessions = loader.load_dataset("amph", base_dir=root_dir, amph_phase=phase)
            project_name = f"AMPH_{phase}"

            if run_all or args.run_clustering_discovery:
                clustering.run_discovery(sessions, project_name=project_name)

            for style in ["alphabet", "standard"]:
                if run_all or args.run_motifs_discovery:
                    motifs.run_discovery(sessions, project_name=project_name, style=style)
                if run_all or args.run_micro_discovery:
                    micro_kinematics.run_discovery(sessions, project_name=project_name, style=style)


    if args.run_all or not any(vars(args).values()):
        from analysis.ulrik_kinematics.lib import legacy_viz, quantitative_viz
        print("\n*** Running Full Legacy Pipeline ***")
        if args.dataset in ["ifdn", "all"]:
            legacy_viz.run_ifdn()
        if args.dataset in ["amph", "all"]:
            legacy_viz.run_amph()
        if args.dataset in ["karen", "all"]:
            legacy_viz.run_karen()

        legacy_viz.run_manual_viz(dataset=args.dataset)
        quantitative_viz.run_quantitative_viz(dataset=args.dataset)
        quantitative_viz.run_advanced_kinematics_viz(dataset=args.dataset)

    global_amph_kinematics = None
    global_ifdn_kinematics = None

    if args.run_pnas_replication or args.run_neural_replication:
        from analysis.ulrik_kinematics.lib import neural_pnas, pnas_replication

        if args.dataset in ["amph", "all"]:
            print("\n*** Computing Global AMPH Kinematic Similarity (Both Phases, All Genotypes) ***")
            sessions_global_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase="both")
            top_sylls_global = neural_pnas._get_top_syllables(sessions_global_amph, k=20)
            _dist, _valid, _Z = pnas_replication.get_kinematic_similarity(
                sessions_global_amph, "AMPH", top_sylls_global, pre=5, post=15
            )
            global_amph_kinematics = (_dist, _valid, _Z, top_sylls_global)
            pnas_replication.plot_fig3h_kinematic_similarity(sessions_global_amph, "AMPH", global_kinematics=global_amph_kinematics)

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Computing Global IFDN Kinematic Similarity (All Genotypes) ***")
            sessions_global_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)
            top_sylls_global = neural_pnas._get_top_syllables(sessions_global_ifdn, k=20)
            _dist, _valid, _Z = pnas_replication.get_kinematic_similarity(
                sessions_global_ifdn, "IFDN", top_sylls_global, pre=5, post=15
            )
            global_ifdn_kinematics = (_dist, _valid, _Z, top_sylls_global)
            pnas_replication.plot_fig3h_kinematic_similarity(sessions_global_ifdn, "IFDN", global_kinematics=global_ifdn_kinematics)

    if args.run_pnas_replication:
        if args.dataset in ["amph", "all"]:
            for phase in ["pre", "post", "both"]:
                print(f"\n*** Loading AMPH Dataset for PNAS Replication ({phase.upper()}) ***")
                sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase=phase)

                for geno in ["WT", "KI"]:
                    sessions_geno = [s for s in sessions_amph if s.metadata.get("genotype", "").upper() == geno]
                    if sessions_geno:
                        pnas_replication.run_replication(sessions_geno, f"AMPH_{phase}_{geno}", global_kinematics=global_amph_kinematics)

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for PNAS Replication ***")
            sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)

            for geno in ["WT", "KI"]:
                sessions_geno = [s for s in sessions_ifdn if s.metadata.get("genotype", "").upper() == geno]
                if sessions_geno:
                    pnas_replication.run_replication(sessions_geno, f"IFDN_{geno}", global_kinematics=global_ifdn_kinematics)

    if args.run_neural_replication:
        if args.dataset in ["amph", "all"]:
            for phase in ["pre", "post", "both"]:
                print(f"\n*** Loading AMPH Dataset for Neural Replication ({phase.upper()}) ***")
                sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase=phase)

                for geno in ["WT", "KI"]:
                    sessions_geno = [s for s in sessions_amph if s.metadata.get("genotype", "").upper() == geno]
                    if sessions_geno:
                        neural_pnas.run_replication(sessions_geno, f"AMPH_{phase}_{geno}", global_kinematics=global_amph_kinematics)

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for Neural Replication ***")
            sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)

            for geno in ["WT", "KI"]:
                sessions_geno = [s for s in sessions_ifdn if s.metadata.get("genotype", "").upper() == geno]
                if sessions_geno:
                    neural_pnas.run_replication(sessions_geno, f"IFDN_{geno}", global_kinematics=global_ifdn_kinematics)

    if args.run_timing_stats or args.verify_timing_stats:
        from analysis.ulrik_kinematics.lib import regional_timing
        from analysis.ulrik_kinematics.lib.saliency_modeling import notify_ntfy

        is_verification = args.verify_timing_stats
        modes = ["strict", "naive"] if getattr(args, "timing_mode", "strict") == "both" else [getattr(args, "timing_mode", "strict")]

        notify_ntfy(f"⏱️ Timing Stats: Starting peak timing analysis for dataset {args.dataset} [Modes: {modes}].")

        for mode in modes:
            suffix = "_naive" if mode == "naive" else ""
            if args.dataset in ["amph", "all"]:
                for phase in ["pre", "post", "both"]:
                    print(f"\n*** Loading AMPH Dataset for Regional Timing Stats ({phase.upper()}) [Mode: {mode.upper()}] ***")
                    sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase=phase)

                    for geno in ["WT", "KI"]:
                        sessions_geno = [s for s in sessions_amph if s.metadata.get("genotype", "").upper() == geno]
                        if sessions_geno:
                            regional_timing.calculate_peak_timing_stats(sessions_geno, f"AMPH_{phase}_{geno}{suffix}", is_verification=is_verification, mode=mode)

            if args.dataset in ["ifdn", "all"]:
                print(f"\n*** Loading IFDN Dataset for Regional Timing Stats [Mode: {mode.upper()}] ***")
                sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)

                for geno in ["WT", "KI"]:
                    sessions_geno = [s for s in sessions_ifdn if s.metadata.get("genotype", "").upper() == geno]
                    if sessions_geno:
                        regional_timing.calculate_peak_timing_stats(sessions_geno, f"IFDN_{geno}{suffix}", is_verification=is_verification, mode=mode)

        notify_ntfy(f"📊 Timing Stats: Finished peak timing statistics and plots for dataset {args.dataset}!")

    if args.run_unique_info or args.verify_unique_info:
        from analysis.ulrik_kinematics.lib import unique_information

        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Unique Information Pipeline ***")
            if args.verify_unique_info:
                sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase="both")[:2]
                print(f"  [Verification Mode] Loaded {len(sessions_amph)} sessions.")
            else:
                sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase="both")

            unique_information.run_pipeline(sessions_amph, "AMPH", model_type=args.stripping_model)

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for Unique Information Pipeline ***")
            if args.verify_unique_info:
                sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)[:2]
                print(f"  [Verification Mode] Loaded {len(sessions_ifdn)} sessions.")
            else:
                sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)

            unique_information.run_pipeline(sessions_ifdn, "IFDN", model_type=args.stripping_model)

    if args.run_temporal_saliency or args.verify_temporal_saliency:
        from analysis.ulrik_kinematics.lib import temporal_saliency

        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Temporal Saliency Pipeline ***")
            if args.verify_temporal_saliency:
                sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase="both")[:2]
                print(f"  [Verification Mode] Loaded {len(sessions_amph)} sessions.")
            else:
                sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase="both")

            temporal_saliency.run_saliency_pipeline(sessions_amph, "AMPH", model_type=args.stripping_model)

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for Temporal Saliency Pipeline ***")
            if args.verify_temporal_saliency:
                sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)[:2]
                print(f"  [Verification Mode] Loaded {len(sessions_ifdn)} sessions.")
            else:
                sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)

            temporal_saliency.run_saliency_pipeline(sessions_ifdn, "IFDN", model_type=args.stripping_model)

    if args.run_temporal_saliency_extended or args.verify_temporal_saliency_extended:
        from analysis.ulrik_kinematics.lib import temporal_saliency

        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Extended Temporal Saliency Pipeline ***")
            if args.verify_temporal_saliency_extended:
                sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase="both")[:2]
                print(f"  [Verification Mode] Loaded {len(sessions_amph)} sessions.")
            else:
                sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase="both")

            temporal_saliency.run_saliency_pipeline(sessions_amph, "AMPH", window_half_s=1.0, model_type=args.stripping_model)

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for Extended Temporal Saliency Pipeline ***")
            if args.verify_temporal_saliency_extended:
                sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)[:2]
                print(f"  [Verification Mode] Loaded {len(sessions_ifdn)} sessions.")
            else:
                sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)

            temporal_saliency.run_saliency_pipeline(sessions_ifdn, "IFDN", window_half_s=1.0, model_type=args.stripping_model)

    if args.run_saliency_modeling or args.verify_saliency_modeling:
        from analysis.ulrik_kinematics.lib import saliency_modeling

        is_verify = args.verify_saliency_modeling

        if args.dataset in ["amph", "all"]:
            print("\n*** Loading AMPH Dataset for Saliency Modeling Pipeline ***")
            if is_verify:
                sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase="both")[:2]
                print(f"  [Verification Mode] Loaded {len(sessions_amph)} sessions.")
            else:
                sessions_amph = loader.load_dataset("amph", base_dir=root_dir, amph_phase="both")

            saliency_modeling.run_modeling_pipeline(sessions_amph, "AMPH")

        if args.dataset in ["ifdn", "all"]:
            print("\n*** Loading IFDN Dataset for Saliency Modeling Pipeline ***")
            if is_verify:
                sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)[:2]
                print(f"  [Verification Mode] Loaded {len(sessions_ifdn)} sessions.")
            else:
                sessions_ifdn = loader.load_dataset("ifdn", base_dir=root_dir)

            saliency_modeling.run_modeling_pipeline(sessions_ifdn, "IFDN")

