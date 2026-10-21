import os
import sys

# Path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from studies.behavior.language_models.lib.evaluation.photometry_peth import align_photometry_to_strategy_switches


def main():
    print("=== Pipeline 3: Fiberphotometry PETH Alignment (Stub Runner) ===")

    # Call stub module interface
    res = align_photometry_to_strategy_switches(
        intervals_df=None,
        photometry_data_dict={},
        window_before_sec=5.0,
        window_after_sec=5.0
    )

    print(f"Photometry PETH alignment runner completed: {res['message']}")

if __name__ == "__main__":
    main()
