# src/segmentation.py
"""
Zone Segmentation on Density Maps

Clusters spatial regions by crowd intensity.
Used for per-zone risk analysis.
"""

import cv2
import numpy as np

def segment_density_map(density_map, num_segments=4):
    """
    Converts a density map into segmented zones
    It normalizes the density map to 0–255.
    Converts it into a list of points for k-means clustering.
    Clusters the density values into K segments (zones) — e.g., 4 clusters could correspond to “low”, “medium”, “high”, “very high” density.
    Maps cluster indices into a visually meaningful range.
    Generates a color visualization of zones using a heatmap (Jet colormap).

    Parameters:
        density_map : density map from CSRNet - 2D numpy array (H x W), the crowd density map
        num_segments: number of zones to cluster (default 4)

    Returns:
        labels_remap : 2D array of zone labels (0,1,...)
        color_vis   : color visualization of the segmented zones
        """
    # normalise the density map to range [0, 255] as uint8 for OpenCV
    norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # get the height (h) and width (w) of the density map
    h, w = norm.shape
    # flatten the 2D image into a 1D column vector for clustering
    Z = norm.reshape(-1, 1).astype(np.float32)

    # ensure the number of clusters is between 2 and 8
    # clustering with just 1 cluster would be meaningless
    # if number of clusters is too high (like 50), segmentation becomes noisy and harder to interpret
    K = max(2, min(num_segments, 8))

    # perform K-means clustering on pixel intensity values
    # - Z : data points
    # - K : number of clusters
    # - None: placeholder for initial labels (let OpenCV handle)
    # - Termination criteria: stop after 20 iterations or when accuracy < 1.0
    # - 10 : number of times K-means runs with different initializations
    # - cv2.KMEANS_RANDOM_CENTERS: pick initial cluster centers randomly
    _, labels, centers = cv2.kmeans(Z, K, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                    10, cv2.KMEANS_RANDOM_CENTERS)
    # reshape labels back to 2D (h x w) to match original density map
    labels = labels.reshape(h, w).astype(np.uint8)
    # sort cluster centers by intensity value so that lower intensity = lower label
    order = np.argsort(centers.flatten())
    # create remap array to reorder labels according to intensity
    remap = np.zeros_like(order)
    # apply remapping to labels so cluster 0 = lowest density, cluster K-1 = highest density
    for i, v in enumerate(order): remap[v] = i
    labels_remap = remap[labels]

    # norm labels to [0, 255] for visualization as an image
    label_u8 = cv2.normalize(labels_remap.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    # Apply a color map (here: JET) to visualize zones with different colors
    color_vis = cv2.applyColorMap(label_u8, cv2.COLORMAP_JET)

    # Return labels_remap: the zone numbers for each pixel, color_vis: a colorful image showing the zones
    return labels_remap, color_vis
