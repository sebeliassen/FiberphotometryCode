"""
syllable_generation/videogen/grids/syllable.py

Generates the Syllable Plot grid.
Loads pre-generated trajectory plots (PDFs) based on syllable ID.
Features:
- Image Caching: Converts and caches PDFs to avoid disk I/O on repeated frames.
- Sticky Display: Retains the last valid syllable image if a subsequent PDF is missing, preventing flickering.
"""

from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path

from .. import config, utils


class SyllableGridGenerator:
    def __init__(self, project_name: str = None, pdf_dir_override: Path = None):
        self.project_name = project_name
        if pdf_dir_override:
            self.pdf_dir = pdf_dir_override
        else:
            self.pdf_dir = config.get_syllable_pdf_dir(project_name)
        print(f"SyllableGridGenerator initialized for project '{project_name}'. PDF Dir: {self.pdf_dir}")
        self.pdf_cache = {} # Cache for converted PDF images (Syllable ID -> np.ndarray)
        self.last_valid_img = None
        self.placeholder_img = None

    def get_syllable_grid(self, syllable_id: int) -> np.ndarray:
        """
        Returns the image for the given syllable ID.
        Checks cache -> loads PDF -> converts -> caches.
        If PDF missing, returns the last successfully loaded syllable image (Sticky behavior).
        """
        # 1. Check Cache
        if syllable_id in self.pdf_cache:
            img = self.pdf_cache[syllable_id]
            # If it's the cached placeholder, we don't treat it as "last_valid"
            # for future sticky calls, but we return it if it was specifically cached.
            # Actually, let's just return it.
            if not np.array_equal(img, self.placeholder_img):
                self.last_valid_img = img
            return img

        # 2. Try to load PDF
        pdf_path = self.pdf_dir / f"Syllable{syllable_id}.pdf"

        try:
            if pdf_path.exists():
                # Convert PDF to image
                # poppler_path might be needed on Windows, but on Linux usually auto-detected if installed
                images = convert_from_path(str(pdf_path), dpi=config.SYLLABLE_DPI)
                if images:
                    pil_img = images[0]
                    # Convert PIL to CV2 (RGB -> BGR)
                    open_cv_image = np.array(pil_img)
                    open_cv_image = open_cv_image[:, :, ::-1].copy()

                    # Resize to quadrant and add label
                    final_img = self._process_image(open_cv_image, f"Syllable {syllable_id}")
                    self.pdf_cache[syllable_id] = final_img
                    self.last_valid_img = final_img
                    return final_img
        except Exception as e:
            print(f"Error loading PDF for Syllable {syllable_id}: {e}")

        # 3. Fallback / Sticky Behavior
        if self.last_valid_img is not None:
            # We don't cache this as the current syllable's entry because we'd rather
            # keep trying or just use the state naturally?
            # Actually, to be consistent with "placeholder", let's return last_valid.
            return self.last_valid_img

        # 4. Final Placeholder (if we haven't seen any valid syllable yet)
        placeholder = self._get_placeholder(syllable_id)
        # We don't cache this because we might find a valid syllable later that can be "sticky"
        return placeholder

    def _process_image(self, img: np.ndarray, label: str) -> np.ndarray:
        """Resizes to quadrant (keeping aspect ratio) and adds label text overlay."""
        # Resizes with Aspect Ratio preservation and pads with white (matching PDF background)
        img_resized = utils.resize_and_pad(img, config.QUADRANT_WIDTH, config.QUADRANT_HEIGHT, pad_value=255)

        # Overlay Text
        return self._add_label(img_resized, label)

    def _get_placeholder(self, syllable_id: int) -> np.ndarray:
        """Generates a text-based placeholder frame."""
        canvas = np.full((config.QUADRANT_HEIGHT, config.QUADRANT_WIDTH, 3), 200, dtype=np.uint8) # Light grey
        text = f"Syllable {syllable_id}"
        sub_text = "(PDF Not Found)"

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Center Text
        (tw, th), _ = cv2.getTextSize(text, font, 2.0, 3)
        cv2.putText(canvas, text, ((config.QUADRANT_WIDTH - tw)//2, config.QUADRANT_HEIGHT//2 - 20),
                    font, 2.0, (0,0,0), 3, cv2.LINE_AA)

        (tw2, th2), _ = cv2.getTextSize(sub_text, font, 1.0, 2)
        cv2.putText(canvas, sub_text, ((config.QUADRANT_WIDTH - tw2)//2, config.QUADRANT_HEIGHT//2 + 40),
                    font, 1.0, (0,0,200), 2, cv2.LINE_AA)

        return canvas

    def _add_label(self, img: np.ndarray, text: str) -> np.ndarray:
        """Adds text label with background box to top-left corner."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.5
        thick = 3
        margin = 20

        (tw, th), base = cv2.getTextSize(text, font, scale, thick)

        # Background box
        cv2.rectangle(img, (margin, margin), (margin + tw + 20, margin + th + 20 + base), (255,255,255), -1)
        cv2.rectangle(img, (margin, margin), (margin + tw + 20, margin + th + 20 + base), (0,0,0), 1)

        # Text
        cv2.putText(img, text, (margin + 10, margin + 10 + th), font, scale, (0,0,0), thick, cv2.LINE_AA)
        return img

