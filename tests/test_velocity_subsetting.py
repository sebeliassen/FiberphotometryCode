
import os
import sys

import numpy as np

# Add parent directory to path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'syllable_generation/syllabizers/keypoint_moseq_syll')))

from utils.trimming import subset_by_activity


def test_subset_by_activity_logic():
    # Create synthetic data: 1 Bodypart, T=10
    # Coordinates: Static then High Movement then Static
    # T=0-2: (0,0) -> Vel=0
    # T=3-6: Moves 10 units/frame -> Vel=10
    # T=7-9: (40,0) -> Vel=0

    T = 10
    coords = np.zeros((T, 1, 2))

    # Movement
    coords[0] = [0,0]
    coords[1] = [0,0]
    coords[2] = [0,0]
    coords[3] = [10,0] # Vel=10
    coords[4] = [20,0] # Vel=10
    coords[5] = [30,0] # Vel=10
    coords[6] = [40,0] # Vel=10
    coords[7] = [40,0] # Vel=0
    coords[8] = [40,0]
    coords[9] = [40,0]

    # Confidences (all 1)
    confs = np.ones((T, 1))

    # VFI (0..9)
    vfi = np.arange(T)

    data_coords = {'test': coords}
    data_confs = {'test': confs}
    data_vfi = {'test': vfi}

    # Subset fraction 0.4 (Window size 4)
    # The high velocity window is indices [3, 4, 5, 6] (Velocities at 2,3,4,5 are involved?)
    # Wait, velocity at t is ||C_{t+1} - C_t||.
    # V[0] = 0
    # V[1] = 0
    # V[2] = 10 (2->3)
    # V[3] = 10 (3->4)
    # V[4] = 10 (4->5)
    # V[5] = 10 (5->6)
    # V[6] = 0 (6->7)

    # We want to capture the movement.
    # Max rolling sum of 4 frames.
    # Window [2,3,4,5] has sum 40.
    # Window [3,4,5,6] has sum 30.

    # Let's see what the function does.

    sub_c, sub_conf, sub_v = subset_by_activity(data_coords, data_confs, data_vfi, frac=0.4)

    assert len(sub_c['test']) == 4

    np.testing.assert_array_equal(sub_v['test'], [2, 3, 4, 5])
    np.testing.assert_array_equal(sub_c['test'], coords[[2, 3, 4, 5]])
    np.testing.assert_array_equal(sub_conf['test'], confs[[2, 3, 4, 5]])

    # Expected: The window covering the high velocity steps.
    # Ideally it should include indices 2,3,4,5

    # Let's run it and see. This test is to verify it works meaningfully.
    pass

if __name__ == "__main__":
    test_subset_by_activity_logic()
    print("Test passed (logic check printed above)")
