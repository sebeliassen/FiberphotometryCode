import numpy as np

def create_batches(coordinates, confidences, errors=None, max_product_size=1e7):
    batches_coords = []
    batches_confs = []
    batches_errors = None if errors is None else []
    current_batch_coords = {}
    current_batch_confs = {}
    current_batch_errors = {} if errors is not None else None
    current_batch_product = 0

    for key, coords in coordinates.items():
        prod_size = np.prod(coords.shape)

        if current_batch_product + prod_size > max_product_size and current_batch_coords:
            batches_coords.append(current_batch_coords)
            batches_confs.append(current_batch_confs)
            if errors is not None:
                batches_errors.append(current_batch_errors)

            current_batch_coords = {}
            current_batch_confs = {}
            current_batch_errors = {} if errors is not None else None
            current_batch_product = 0

        current_batch_coords[key] = coords
        current_batch_confs[key] = confidences[key]
        if errors is not None:
            current_batch_errors[key] = errors[key]
        current_batch_product += prod_size

    if current_batch_coords:
        batches_coords.append(current_batch_coords)
        batches_confs.append(current_batch_confs)
        if errors is not None:
            batches_errors.append(current_batch_errors)

    return batches_coords, batches_confs, batches_errors