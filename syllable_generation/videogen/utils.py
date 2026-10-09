
import cv2
import numpy as np

def resize_and_pad(img: np.ndarray, target_w: int, target_h: int, pad_value: int = 0) -> np.ndarray:
    """
    Resizes image to fit within target dimensions while preserving aspect ratio.
    Pads the remaining area with pad_value.
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    
    # Interpolation: use area for downscaling, linear for upscaling
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    
    # Create canvas with 3 channels if input has 3 channels, else 1
    if len(img.shape) == 3:
        canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    else:
        canvas = np.full((target_h, target_w), pad_value, dtype=np.uint8)
    
    # Center the image
    y_off = (target_h - nh) // 2
    x_off = (target_w - nw) // 2
    
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas
