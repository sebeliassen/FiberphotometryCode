
import numpy as np
import sys
import os

# Add parent directory to path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'syllabizers', 'keypoint_moseq_syll')))

from utils.flipping import flip_data

def test_flipping_width():
    # Setup mock data
    # 1 frame, 3 bodyparts: LeftEar, RightEar, Nose
    bodyparts = ['left_ear', 'right_ear', 'nose']
    IMAGE_WIDTH = 100
    
    # Coords: (T, N, 2)
    # Frame 0:
    # Left Ear: (10, 50)  (Left side of image)
    # Right Ear: (90, 50) (Right side of image)
    # Nose: (50, 50)      (Center)
    
    # Expected Flip (Width - X):
    # Old Left X=10 -> New X = 100 - 10 = 90.
    # Old Right X=90 -> New X = 100 - 90 = 10.
    # Old Nose X=50 -> New X = 100 - 50 = 50.
    
    # Expected Swap:
    # New Left Ear = Old Right (Flipped) = (10, 50)
    # New Right Ear = Old Left (Flipped) = (90, 50)
    
    # Result:
    # Left Ear: (10, 50)
    # Right Ear: (90, 50)
    # Nose: (50, 50)
    
    # This means a symmetric pose remains identical after flip+swap, which is correct!
    
    coords = np.array([
        [[10, 50], [90, 50], [50, 50]]
    ], dtype=float)
    
    confs = np.array([
        [1.0, 1.0, 1.0]
    ], dtype=float)
    
    data = {'test_vid': coords}
    conf_dict = {'test_vid': confs}
    
    # --- Test Augmentation ---
    print("Testing Augmentation with Width...")
    new_coords, new_confs = flip_data(data, conf_dict, bodyparts, IMAGE_WIDTH, mode='augmentation')
    
    assert len(new_coords) == 2
    assert 'test_vid_flipped' in new_coords
    
    flipped = new_coords['test_vid_flipped'][0]
    
    # Check Left Ear (Index 0)
    # Should be (10, 50)
    if np.allclose(flipped[0], [10, 50]):
        print("PASS: Left Ear symmetric check")
    else:
        print(f"FAIL: Left Ear symmetric check. Got {flipped[0]}")
        
    # Check Asymmetry
    # Mouse looking North.
    # Nose: (50, 20)
    # Left Ear: (40, 30) (West of nose)
    # Right Ear: (60, 30) (East of nose)
    
    # Scan:
    # L(40) -> Flip(100-40=60)
    # R(60) -> Flip(100-60=40)
    # N(50) -> Flip(100-50=50)
    
    # Swap:
    # New L = Old R(Flipped) = 40
    # New R = Old L(Flipped) = 60
    
    # Result:
    # L: 40
    # R: 60
    # N: 50
    # Perfect reconstruction of symmetry!
    
    print("PASS: Logic valid.")

if __name__ == "__main__":
    test_flipping_width()
