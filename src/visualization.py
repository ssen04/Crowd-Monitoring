# src/visualization.py
"""
Visualization Module

Responsible for creating the final 2x2 dashboard view.
Takes YOLO output, density map, surge map, and an info box, and arranges them together in one combined window.
Combines all panels (YOLO, Density, Surge, Info) into a 2x2 grid display.
"""

import cv2
import numpy as np

def make_info_panel(w, h, lines):
    """
    Create a rectangular dark-colored box that shows text information.

    w = width of the panel
    h = height of the panel
    lines = list of strings to display (e.g., counts, FPS, risk level)
    """
    # create a blank panel of size (h x w), filled with dark gray (30,30,30)
    # means we have 3 color channels (RGB)
    panel = np.full((h, w, 3), (30,30,30), dtype=np.uint8)

    # loop through each line of text
    for i, line in enumerate(lines):
        # write each line of text on the panel,starting at y = 30 for line 0, then +25 pixels for each next line
        cv2.putText(panel, line, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # return the finished info panel
    return panel

def combine_panels(yolo_view, density_view, surge_view, info_panel):
    """
    Takes four images (YOLO output, density map, surge map, info panel)
    and puts them together into a 2x2 grid layout.
    """
    # create top row
    top = np.hstack((yolo_view, density_view))
    # create bottom row
    bottom = np.hstack((surge_view, info_panel))
    # stack
    return np.vstack((top, bottom))
