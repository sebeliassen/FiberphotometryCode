"""
syllable_generation/videogen/grids/photometry.py

Generates the Photometry Trace grid.
Uses Matplotlib (Agg backend) to render rolling Z-score traces for active sensors.
Support dynamic layout (L/R hemisphere separation) and color-coding by brain region.
"""

import matplotlib

matplotlib.use('Agg') # Non-interactive backend
from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .. import config


class PhotometryGridGenerator:
    def __init__(self):
        # We can reuse the figure to avoid memory leaks/overhead?
        # Creating a figure per frame is slow.
        # Using one figure and clearing acts faster but requires care.
        # For Side-by-Side, we can have 1 Figure with 2 Subplots.

        # Setup Figure
        # dpi=100. 1600x800 total. 2 panels side by side -> each 800x800.
        # So Figure size e.g. 16x8 inches at 100dpi.
        self.dpi = 100
        w_inch = config.COMPOSITE_WIDTH / self.dpi
        h_inch = config.COMPOSITE_HEIGHT / self.dpi

        self.fig = plt.figure(figsize=(w_inch, h_inch), dpi=self.dpi)
        self.canvas = FigureCanvasAgg(self.fig)

        # Background
        self.fig.patch.set_facecolor(config.PLOT_BG_COLOR)

        # Subplots: 1 row, 2 cols
        # ax1 = Left, ax2 = Right
        self.axs = self.fig.subplots(1, 2)

        # Adjust layout
        self.fig.tight_layout(pad=3.0)

    def generate_frame(
        self,
        df_window: pd.DataFrame,
        current_frame_idx: int,
        signal_meta: Dict[str, Any],
        is_trio: bool = False
    ) -> np.ndarray:
        """
        Draws the photometry traces for the current window.
        """
        # Define dimensions
        target_width = config.TRIO_WIDTH if is_trio else config.COMPOSITE_WIDTH
        target_height = config.QUADRANT_HEIGHT # Both are 800

        self._clear_axes()

        # Identify Green Signals (orig_letter='G')
        green_signals = []
        for key, meta in signal_meta.items():
            if meta.get('orig_letter', '').upper() == 'G':
                green_signals.append((key, meta))

        green_signals.sort(key=lambda x: x[0])

        # Dynamic Axis Assignment
        # Heuristic: Map Left Hemisphere to Left Axis (ax0), Right to Right (ax1).
        sig_to_ax = {}

        # Try to find Left and Right
        l_sig = next((s for s in green_signals if s[1].get('hemisphere') == 'L'), None)
        r_sig = next((s for s in green_signals if s[1].get('hemisphere') == 'R'), None)

        if l_sig:
            sig_to_ax[l_sig[0]] = self.axs[0]
        if r_sig:
            sig_to_ax[r_sig[0]] = self.axs[1]

        # Fallback: if we have 2 signals but couldn't map L/R cleanly (?)
        # Or if we have >2 green signals? (Rare).
        # If we missed one simply because it wasn't L/R labeled (e.g. unknown), assign to empty ax.
        used_axs = set(sig_to_ax.values())
        unassigned_sigs = [s for s in green_signals if s[0] not in sig_to_ax]

        for sig in unassigned_sigs:
            if self.axs[0] not in used_axs:
                sig_to_ax[sig[0]] = self.axs[0]
                used_axs.add(self.axs[0])
            elif self.axs[1] not in used_axs:
                sig_to_ax[sig[1]] = self.axs[1]
                used_axs.add(self.axs[1])

        # Plot
        # For X-axis, we use relative index in the window?
        # Let's use the actual index from the dataframe window.
        xs = df_window.index.values

        for sig_key, ax in sig_to_ax.items():
            meta = signal_meta[sig_key]

            # Get Data
            col_name = f"{sig_key}_zscore" # Or just signal_0 -> signal_0_zscore?
            # Check likely column names
            if col_name not in df_window.columns:
                col_name = sig_key # Fallback

            if col_name in df_window.columns:
                ys = df_window[col_name].values

                # Convert X-axis to relative seconds
                # Centers plot at 0s (current frame)
                sampling_rate = config.DEFAULT_SAMPLING_RATE
                xs_seconds = (xs - current_frame_idx) / sampling_rate

                # Determine Color
                reg = meta.get('brain_reg')
                line_color = config.BRAIN_REG_TO_COLOR.get(reg, config.PLOT_LINE_COLOR)

                # Plot Trace
                ax.plot(xs_seconds, ys, color=line_color, linewidth=2)

                # Title: Region - Hemisphere
                title = f"{reg} - {meta.get('hemisphere','?')}"
                ax.set_title(title, color=config.PLOT_TEXT_COLOR, fontsize=16, fontweight='bold')

                # Vertical Line at current time (Always 0 because it's center-aligned)
                ax.axvline(x=0, color='white', linestyle='--', alpha=0.7)

                # Explicitly set X limits for +/- 5 seconds
                ax.set_xlim(-5, 5)
                ax.set_xlabel("Seconds relative to curr", color=config.PLOT_TEXT_COLOR)
            else:
                 # Missing column
                 ax.text(0.5, 0.5, "Data Missing", color='red', ha='center', va='center')

        # Styling for all axes
        for ax in self.axs:
            ax.set_facecolor(config.PLOT_BG_COLOR)
            ax.tick_params(colors=config.PLOT_TEXT_COLOR, which='both')
            for spine in ax.spines.values():
                spine.set_color(config.PLOT_TEXT_COLOR)

            # Hide X ticks? Or show frame numbers?
            # Scrolling X-axis is distracting.
            # Maybe just keep clean.

        # Check for empty axes (missing sensors)
        # If an axis wasn't used, blank it or put "No Sensor"
        for ax in self.axs:
             if not ax.lines and not ax.texts:
                 ax.text(0.5, 0.5, "No Signal", color='grey', ha='center', va='center', transform=ax.transAxes, fontsize=14)

        # Render
        self.canvas.draw()

        # Convert to OpenCV image (BGR)
        img = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize to EXACT dimensions just in case of slight DPI-rounding mismatch
        img = cv2.resize(img, (target_width, target_height))

        return img

    def _clear_axes(self):
        for ax in self.axs:
            ax.clear()

    def close(self):
        plt.close(self.fig)
