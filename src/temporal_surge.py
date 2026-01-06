# src/temporal_surge.py
"""
Temporal Surge Detection

Detects sudden density changes over time.
Used for early crowd warning or surge detection.
"""

import cv2
import numpy as np

class TemporalSurgeDetector:
    def __init__(self, alpha=0.3, scale_factor=8):
        """
        Parameters:
          alpha: how much we blend with the previous frame.
               - smaller alpha == slow smoothing
               - larger alpha == fast smoothing
          scale_factor: CSRNet outputs a smaller map than the actual frame. This number tells us how much to resize it back
        """
        self.prev_density = None # store the previous frame's density map
        self.alpha = alpha # smoothing strength
        self.scale_factor = scale_factor # how much to enlarge the output later

    def compute_surge(self, density):
        """
        Compute temporal surge map from (how much density increased compared to last frame).
        """
        # if this is the first frame == there is no previous density to compare with
        #  store it and return a surge map full of zeros
        if self.prev_density is None:
            self.prev_density = density.copy().astype(np.float32)
            return np.zeros_like(density, dtype=np.float32)

        # compute (density - prev_density), but only keep positive values
        # this means? = only detect increase not decrease
        # positive difference  == increase in density
        diff = np.maximum(density.astype(np.float32) - self.prev_density, 0)

        # exponential moving average smoothing
        # update the previous density using a smooth moving average
        # to prevents noisy flickering
        self.prev_density = (
            self.alpha * density.astype(np.float32) # new frame contribution
            + (1 - self.alpha) * self.prev_density # previous frame contribution
        )

        # normalise the surge map to values between 0 and 1.
        diff = cv2.normalize(diff, None, 0, 1.0, cv2.NORM_MINMAX)

        # CSRNet density map is small
        # resize the surge map back to the full input resolution = upsample surge map to match full frame res
        full_h = int(density.shape[0] * self.scale_factor)
        full_w = int(density.shape[1] * self.scale_factor)
        diff_resized = cv2.resize(diff, (full_w, full_h), interpolation=cv2.INTER_LINEAR)

        # final surge map at full resolution
        return diff_resized

    def colorize_surge(self, surge):
        """
        Convert the surge map into a colored heatmap for visualization.
        """
        # normalise again from 0–255 and convert to uint8 for OpenCV color map
        surge_u8 = (255 * cv2.normalize(surge, None, 0, 1, cv2.NORM_MINMAX)).astype(np.uint8)
        # apply color map (Inferno → yellow/orange/red)
        return cv2.applyColorMap(surge_u8, cv2.COLORMAP_INFERNO)