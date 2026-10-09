
import numpy as np


def get_symmetry_indices(bodyparts):
    """
    Returns a list of indices to permute bodyparts for a L-R flip.
    Assumes bodyparts are named like 'left_ear', 'right_ear', etc.
    """
    indices = np.arange(len(bodyparts))

    # Manual mapping based on config.yml
    # "left_X" <-> "right_X"

    mapping = {}
    for i, bp in enumerate(bodyparts):
        if 'left' in bp:
            counterpart = bp.replace('left', 'right')
        elif 'right' in bp:
            counterpart = bp.replace('right', 'left')
        else:
            counterpart = None

        if counterpart and counterpart in bodyparts:
            j = bodyparts.index(counterpart)
            mapping[i] = j
            mapping[j] = i

    # Apply mapping
    permuted = indices.copy()
    for i, j in mapping.items():
        permuted[i] = j

    return permuted

def flip_data(coordinates, confidences, bodyparts, image_width, mode='augmentation', seed=42, key_to_aid=None):
    """
    Flips data along the X-axis (Width - X) and swaps symmetric bodyparts.
    
    mode:
      - 'augmentation': Returns Original + Flipped (2x size)
      - 'flip_half': Returns same size, 50% flipped
      - 'flip_all': Returns all flipped
      - 'optimized': Flips based on OPTIMAL_FLIP_IDS explicitly using key_to_aid mapping.
      
    Returns: (new_coords, new_confs)
    """
    print(f"[flipping] Mode: {mode}, Image Width: {image_width}")

    if mode is None or mode == 'none':
        return coordinates, confidences

    perm_idx = get_symmetry_indices(bodyparts)

    # helper to flip a single array (T, N, 2)
    def _flip_array(arr, is_coords=True):
        new_arr = arr.copy()
        if is_coords:
            # Flip X (index 0): new_x = width - old_x
            new_arr[:, :, 0] = image_width - new_arr[:, :, 0]
            # Permute bodyparts (dim 1)
            new_arr = new_arr[:, perm_idx, :]
        else:
            # Permute bodyparts (dim 1) - Confidences are (T, N)
            new_arr = new_arr[:, perm_idx]
        return new_arr

    new_coords = {}
    new_confs = {}

    keys = sorted(coordinates.keys())
    rng = np.random.default_rng(seed)

    if mode == 'augmentation':
        # Keep original
        new_coords.update(coordinates)
        new_confs.update(confidences)

        # Add flipped versions
        for k in keys:
            flipped_k = f"{k}_flipped"
            new_coords[flipped_k] = _flip_array(coordinates[k], is_coords=True)
            new_confs[flipped_k]  = _flip_array(confidences[k], is_coords=False)

        print(f"[flipping] Augmentation complete. {len(coordinates)} -> {len(new_coords)} videos.")

    elif mode == 'flip_half':
        # Randomly choose 50% to flip
        n_flip = len(keys) // 2
        flip_keys = set(rng.choice(keys, size=n_flip, replace=False))

        for k in keys:
            if k in flip_keys:
                # Flip in place (store as original name? or modified?)
                # User request: "half the videos are flipped"
                # Usually implies we replace the original with the flipped version
                new_coords[k] = _flip_array(coordinates[k], is_coords=True)
                new_confs[k]  = _flip_array(confidences[k], is_coords=False)
            else:
                new_coords[k] = coordinates[k]
                new_confs[k]  = confidences[k]

        print(f"[flipping] Half-flip complete. Flipped {n_flip}/{len(keys)} videos.")

    elif mode == 'optimized':
        # Selective flipping based on optimal_flip_list
        from utils.optimal_flip_list import OPTIMAL_FLIP_IDS
        flip_set = set(str(x) for x in OPTIMAL_FLIP_IDS)

        n_flipped = 0
        for k in keys:
            should_flip = False

            # Use explicit mapping if provided, bypassing fragile substring matches
            if key_to_aid and k in key_to_aid:
                aid = str(key_to_aid[k])
                if aid in flip_set:
                    should_flip = True
            else:
                # Fallback to older heuristic with slightly safer bounds if mapping missing
                for fid in flip_set:
                    if f"_{fid}_" in k or k.endswith(f"_{fid}"):
                         should_flip = True
                         break

            if should_flip:
                n_flipped += 1
                new_coords[k] = _flip_array(coordinates[k], is_coords=True)
                new_confs[k]  = _flip_array(confidences[k], is_coords=False)
            else:
                new_coords[k] = coordinates[k]
                new_confs[k]  = confidences[k]

        print(f"[flipping] Optimized alignment complete. Flipped {n_flipped}/{len(keys)} videos.")

    else:
        raise ValueError(f"Unknown flipping mode: {mode}")

    return new_coords, new_confs
