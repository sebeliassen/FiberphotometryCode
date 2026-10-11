"""
Digest Results Wrapper (AMPH Panel 1e).

This script configures the global `config.OUTPUT_DIR` to point to the AMPH-specific directory 
(`output/panel_1e_amph`) before invoking the core digestion logic (`digest_results.main`).

It enables the reuse of the common digestion code (statistics, chance bands) for the AMPH dataset 
without code duplication.
"""
from lib import config

# Override Output Dir
config.OUTPUT_DIR = config.PANEL_DIRS["1e_amph"]

import digest_results

if __name__ == "__main__":
    print(f"Digesting AMPH results in {config.OUTPUT_DIR}...")
    digest_results.main()
