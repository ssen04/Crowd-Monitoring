"""
Milestone 5 — Final Integrated System (7-Panel Layout)
-------------------------------------------------------
Top-left:     YOLOv5 object detection & tracking
Top-right:    CSRNet density heatmap
Middle-left:  Temporal surge (Δ density)
Middle-right: Dynamic zoom of surge region
Bottom-left:  Original video feed
Bottom-middle: Emergency safe path
Bottom-right: Info panel (legend + metrics)
"""

import cv2
import time
import numpy as np
import sys
import os

from src.yolo_detection import YOLODetector
from src.deepsort_tracker import PeopleTracker
from src.csrnet_density import DensityEstimator
from src.temporal_surge import TemporalSurgeDetector
from src.segmentation import segment_density_map
from src.dynamic_zoom import DynamicZoomer
from src.visualization import make_info_panel
from src.emergency_path import EmergencyPathFinder

# ---------------- CONFIG ----------------
# new york video
# DEFAULT_VIDEO_PATH = "/Users/sukanya/PycharmProjects/CMPT742_Project/datasets/mot/crowd times square.mp4"
# DEFAULT_VIDEO_PATH = "/Users/sukanya/PycharmProjects/CMPT742_Project/datasets/mot/41315-429396382_tiny.mp4"
WEIGHTS_PATH = "/Users/sukanya/PycharmProjects/CMPT742_Project/models/CSRNet_pytorch/weights/PartAmodel_best.pth.tar"

# High crowd
DEFAULT_VIDEO_PATH = "/Users/sukanya/PycharmProjects/CMPT742_Project/datasets/mot/High Crowd.mp4"

# Risk calculation weights (matching milestone4_split_view.py)
ALPHA = 0.6  # Density weight
BETA = 0.4  # Surge/motion weight


# ---------------- main ----------------
def main(video_path=None):
    """
    Main entry point for standalone
    """

    # --- determine which video to use --- #
    if video_path is None:
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
        else:
            video_path = DEFAULT_VIDEO_PATH

    # --- validate video path --- #
    if not os.path.exists(video_path):
        print(f"Error!: Video path not found → {video_path}")
        return

    print(f"\nStarting Final Integrated Crowd Analysis System (7-Panel Layout)...")
    print(f"Video Source: {video_path}\n")

    # --- initialise all processing modules --- #
    # yolo handles: person detection
    yolo = YOLODetector()
    # tracker handles: tracking
    tracker = PeopleTracker()  # internally initialise DeepSort(max_age=15)
    # csr handles: CSRNet density predictor
    csr = DensityEstimator(WEIGHTS_PATH)
    # surge_detector handles: temporal surge - difference of density
    surge_detector = TemporalSurgeDetector()
    # zoomer handles: dynamic zooming - focus on hotspots
    zoomer = DynamicZoomer(zoom_factor=3.0, min_crop=100)
    # path finder handles: finding best emergency path
    path_finder = EmergencyPathFinder(grid_resolution=30)

    # --- open video --- #
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # count frames processed
    frame_idx = 0
    # initialise start time
    start_time = time.time()

    # --- main loop --- #
    # process video frame-by-frame
    while True:
        # read 1 frame from video
        ret, frame = cap.read()
        # break if we reach the end
        if not ret:
            break
        frame_idx += 1

        # --- YOLO detection & tracking --- #
        # detections_raw = [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
        detections_raw, yolo_view = yolo.detect_people(frame)
        # DeepSORT expects format ([x1,y1,x2,y2], confidence_score, class_label)
        detections = []
        for det in detections_raw:
            # YOLO sometimes returns 4, 5, or 6 fields depending on version
            # to make this compatible with multiple yolo versions - currently using yolov5
            if len(det) >= 6:
                x1, y1, x2, y2, conf, cls = det[:6]
            elif len(det) == 5:
                x1, y1, x2, y2, conf = det
                cls = "person"
            else:
                # only coordinates provided
                x1, y1, x2, y2 = det
                conf = 1.0
                cls = "person"
            # append box coords to detections
            detections.append(([float(x1), float(y1), float(x2), float(y2)], float(conf), cls))

        # track detected people over time - IDs to follow the same person across frames
        # tracked below is not being used currently but useful - count unique people over time/ Associate detected people with other panels
        tracked = tracker.update(frame, detections)

        # --- CSRNet Density map --- #
        # get density map
        density = csr.get_density_map(frame)
        # convert raw density map to colored heatmap
        _, density_color = segment_density_map(density)

        # ensure CSRNet output matches frame size = resize heatmap to match original frame
        density_color = cv2.resize(density_color, (frame.shape[1], frame.shape[0]))

        # If density_color is grayscale, convert to BGR
        # ensures density heatmap has 3 channels (BGR)
        if len(density_color.shape) == 2 or density_color.shape[2] == 1:
            density_color = cv2.cvtColor(density_color, cv2.COLOR_GRAY2BGR)

        # Overlay density map on original frame (80% transparent - alpha = 0.2)
        density_overlay = cv2.addWeighted(frame, 0.2, density_color, 0.5, 0)
        density_color = density_overlay

        # --- Temporal surge --- #
        # surge = where density changed rapidly (crowd moving in/out)
        surge_map = surge_detector.compute_surge(density)
        surge_view_raw = surge_detector.colorize_surge(surge_map)

        # resize
        surge_view_raw = cv2.resize(surge_view_raw, (frame.shape[1], frame.shape[0]))

        # ensure 3-channel
        if len(surge_view_raw.shape) == 2 or surge_view_raw.shape[2] == 1:
            surge_view_raw = cv2.cvtColor(surge_view_raw, cv2.COLOR_GRAY2BGR)

        # overlay surge heatmap on original frame (90% transparent)
        # surge_overlay = cv2.addWeighted(frame, 0.1, surge_view_raw, 0.4, 0)
        # surge_view = surge_overlay
        surge_view = surge_view_raw

        # --- dynamic zoom --- #
        zoomed_view = zoomer.zoom_to_surge_area(frame, surge_map)

        # --- emergency safe path --- #
        path, path_view = path_finder.find_safe_path(frame, density, surge_map, alpha=ALPHA, beta=BETA)

        # --- Info panel --- #
        fps = frame_idx / (time.time() - start_time + 1e-6)
        info = [
            f"Frame: {frame_idx}",
            f"FPS: {fps:.2f}",

            "Legend:",
            "Top-left -> YOLOv5 (Detection + Tracking)",
            "Top-right -> CSRNet Density Heatmap",
            "Middle-left -> Temporal Surge (ΔDensity)",
            "Middle-right -> Dynamic Zoom (localized)",
            "Bottom-left -> Original Video",
            "Bottom-right -> Best emergency path",
            "",
            "Press 'Q' to exit"
        ]

        # create the info panel image
        h, w = frame.shape[:2]
        info_panel = make_info_panel(w // 3, h // 3, info)

        # --- resize all panels to equal size (W/3 x H/3) --- #
        panel_w = w // 3
        panel_h = h // 3

        yolo_view = cv2.resize(yolo_view, (panel_w, panel_h))
        density_color = cv2.resize(density_color, (panel_w, panel_h))
        surge_view = cv2.resize(surge_view, (panel_w, panel_h))
        zoomed_view = cv2.resize(zoomed_view, (panel_w, panel_h))
        frame_resized = cv2.resize(frame, (panel_w, panel_h))
        path_view = cv2.resize(path_view, (panel_w, panel_h))
        info_resized = cv2.resize(info_panel, (panel_w, panel_h))

        # create blank panel for alignment
        blank_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

        # --- combine panels (3x3 grid, using 7 panels) ---#
        # horizontal stack for 3 cells and then vertical stack the 3 rows on top on one another
        row1 = np.hstack((yolo_view, density_color, blank_panel))
        row2 = np.hstack((surge_view, zoomed_view, blank_panel))
        row3 = np.hstack((frame_resized, path_view, info_resized))
        grid = np.vstack((row1, row2, row3))

        # --- display final dashboard --- #
        cv2.imshow("Crowd Analysis System (7-Panel Integrated)", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Finished processing. Exiting now")


if __name__ == "__main__":
    # Support both GUI launch and direct terminal run
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()