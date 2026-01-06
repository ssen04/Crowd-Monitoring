# src/dynamic_zoom.py
"""
Dynamic Zoom Module

Automatically zooms into areas with high temporal surge intensity.
Useful for localized crowd risk inspection.

Usage:
    zoomer = DynamicZoomer()
    zoom_frame = zoomer.zoom_to_surge_area(frame, surge_map)
"""

import cv2
import numpy as np

class DynamicZoomer:
    def __init__(self, zoom_factor=5.0, min_crop=80):
        """
        Look at the surge map – this is basically a heatmap showing where crowds are growing or “moving fast”.
        Find the hotspot – the point in the surge map with the highest value (the most activity).
        Crop a small rectangle around that hotspot – the size depends on zoom_factor and min_crop.
        Resize that crop to full frame – this creates a zoom-in effect.

        Parameters:
          zoom_factor: how much to zoom (larger = stronger zoom)
          min_crop: minimum size (in pixels) of the zoom window
        """
        self.zoom_factor = zoom_factor # store how much to zoom
        self.min_crop = min_crop # store minimum crop size

    def zoom_to_surge_area(self, frame, surge_map):
        # if input is invalid, return original frame
        if surge_map is None or frame is None:
            return frame

        # get height and width of the frame
        H, W = frame.shape[:2]

        # normalise the surge map to range [0, 1] for consistency
        norm = cv2.normalize(surge_map, None, 0, 1, cv2.NORM_MINMAX)

        # find location of the highest surge intensity in the map
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(norm)

        # if max intensity is too low, no need to zoom -> return original frame
        if maxVal < 0.2:  # no visible surge detected
            return frame

        # get coordinates of highest surge
        cx, cy = maxLoc

        # decode crop window size based on zoom factor and frame size
        crop_w = max(int(W / self.zoom_factor), self.min_crop) # width of zoom window
        crop_h = max(int(H / self.zoom_factor), self.min_crop) # height of zoom window

        # calculate top-left corner of crop window, ensure it stays inside frame
        x1 = int(np.clip(cx - crop_w // 2, 0, W - crop_w))
        y1 = int(np.clip(cy - crop_h // 2, 0, H - crop_h))
        # bottom-right corner
        x2, y2 = x1 + crop_w, y1 + crop_h

        # crop the region of interest (ROI) from the frame
        roi = frame[y1:y2, x1:x2]
        # resize cropped region back to full frame size for zoom effect
        zoomed = cv2.resize(roi, (W, H), interpolation=cv2.INTER_CUBIC)

        # overlay for visual cue - zoomed already resized the cropped region to full frame size - overlay is the original frame
        # cv2.addWeighted(zoomed, 0.85, overlay, 0.15, 0) blends them - rectangle was on the original frame, after resizing and blending - invisible
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        blended = cv2.addWeighted(zoomed, 0.85, overlay, 0.15, 0)

        return blended
