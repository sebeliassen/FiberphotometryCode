#!/usr/bin/env python3
"""
Orchestrator for Panel 1z Refactor.
Modular, CLI-driven pipeline for analyzing neural/behavioral data.
"""

import sys
import argparse
import logging
from pathlib import Path
import importlib

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# Fix for unpickling 'syllable_generation.datamodel'
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local imports (Lazy loaded in main to avoid circular deps during dev)
from analysis.replications.panel_1z import lib
from analysis.replications.panel_1z import modules

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Panel1z")

# Registry of available modules
# We will populate this dynamically or statically
AVAILABLE_MODULES = [
    'peak_timing',
    'responsiveness', 
    'specificity',
    'fingerprint',
    'vigor_waveforms',
    'vigor_valence',
    'stereotypy',
    'geometry',
    'hyperactivity'
]

def main():
    parser = argparse.ArgumentParser(description="Modular Panel 1z Analysis Pipeline")
    
    # Project & Data
    parser.add_argument("--project", type=str, default="ifdn", choices=["ifdn", "amph", "all"],
                        help="Project dataset to analyze (default: ifdn)")
    parser.add_argument("--pkl-path", type=str, default=None,
                        help="Override path to sessions pickle")
    
    # Analysis Options
    parser.add_argument("--modules", type=str, nargs="+", default=["all"],
                        help=f"Modules to run. Choices: {AVAILABLE_MODULES} or 'all'")
    parser.add_argument("--filter-1hz", action="store_true",
                        help="Apply 1Hz Bessel High-pass filter to signals")
    parser.add_argument("--split-injection-delay", type=float, default=0.0,
                        help="Minutes after injection to start analysis (only for AMPH)")
    
    # Filters
    parser.add_argument("--min-trials", type=int, default=100,
                        help="Minimum trials per syllable for inclusion")
    
    args = parser.parse_args()

    # 1. Resolve Modules
    if "all" in args.modules:
        modules_to_run = AVAILABLE_MODULES
    else:
        modules_to_run = [m for m in args.modules if m in AVAILABLE_MODULES]
        
    if not modules_to_run:
        logger.error(f"No valid modules selected from {args.modules}. Available: {AVAILABLE_MODULES}")
        sys.exit(1)
        
    logger.info(f"🚀 Starting Analysis [{args.project.upper()}]")
    logger.info(f"   Modules: {modules_to_run}")
    logger.info(f"   Filtering: {'ON (1Hz Bessel)' if args.filter_1hz else 'OFF'}")
    if args.split_injection_delay > 0:
        logger.info(f"   Time Split: Post-Injection + {args.split_injection_delay} mins")

    # 2. Load Sessions
    try:
        from analysis.replications.panel_1z.lib import loader
        sessions = loader.load_sessions(project=args.project, pkl_path=args.pkl_path)
    except Exception as e:
        logger.error(f"Failed to load sessions: {e}")
        sys.exit(1)

    if not sessions:
        logger.error("No sessions loaded. Exiting.")
        sys.exit(1)

    # 3. Pre-Processing (Optional)
    # Some modules might need kinematics pre-calculated
    try:
        from analysis.replications.panel_1z.lib import processing
        
        # [NEW] Injection Filtering (Before Enrichment so stats are correct)
        if args.project == "amph" and args.split_injection_delay > 0:
             logger.info("✂️  Filtering Post-Injection...")
             sessions = processing.filter_post_injection(sessions, args.split_injection_delay)
        
        # Enrich data (e.g. baking region names, calc kinematics)
        # Note: We do this IN PLACE on the session objects
        logger.info("🔧 Pre-processing sessions...")
        sessions = processing.enrich_sessions(sessions)
        
        if args.filter_1hz:
             logger.info("🔧 Applying 1Hz Global Filter...")
             sessions = processing.apply_filter_to_sessions(sessions)
        
    except ImportError:
        logger.warning("shared processing lib not found. Skipping enrichment.")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)

    # 4. Config for Modules
    # [FIX] Output dir based on project
    out_dir_name = f"output_{args.project}"
    
    config = {
        'project': args.project,
        'filter_1hz': args.filter_1hz,
        'min_trials': args.min_trials,
        'output_dir': Path(__file__).parent / out_dir_name,
        'repo_root': REPO_ROOT
    }
    
    # Ensure output dir exists
    config['output_dir'].mkdir(exist_ok=True)

    # 5. Run Modules
    for mod_name in modules_to_run:
        logger.info(f"▶️  Running Module: {mod_name}")
        try:
            # Dynamic Import
            mod = importlib.import_module(f"analysis.replications.panel_1z.modules.{mod_name}")
            
            # Create module-specific output dir
            mod_output = config['output_dir'] / mod_name
            mod_output.mkdir(exist_ok=True)
            
            # Update config for this module
            mod_config = config.copy()
            mod_config['output_dir'] = mod_output
            
            # Execute
            if hasattr(mod, 'run'):
                mod.run(sessions, mod_config)
            else:
                logger.error(f"Module {mod_name} has no 'run()' function.")
                
        except ImportError as e:
            logger.error(f"Could not import module {mod_name}: {e}")
        except Exception as e:
            logger.error(f"Error running module {mod_name}: {e}", exc_info=True)

    logger.info("✅ All requested modules completed.")

if __name__ == "__main__":
    main()
