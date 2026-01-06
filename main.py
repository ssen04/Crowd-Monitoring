"""
Milestone 5 — Final Integrated System (6-Panel Layout)
-------------------------------------------------------
Top-left:     YOLOv5 object detection & tracking
Top-right:    CSRNet density heatmap
Middle-left:  Temporal surge (Δ density)
Middle-right: Dynamic zoom of surge region
Bottom-left:  Original video feed
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

# ---------------- config ---------------- #

# Moderate
# DEFAULT_VIDEO_PATH = "/Users/sukanya/PycharmProjects/CMPT742_Project/datasets/mot/switch 2.mp4"
# Low crowd
# DEFAULT_VIDEO_PATH = "/Users/sukanya/PycharmProjects/CMPT742_Project/datasets/mot/Low Crowd 742.mp4"
# High crowd
DEFAULT_VIDEO_PATH = "/Users/sukanya/PycharmProjects/CMPT742_Project/datasets/mot/High Crowd.mp4"
# DEFAULT_VIDEO_PATH = "/Users/sukanya/PycharmProjects/CMPT742_Project/datasets/mot/crowd times square.mp4"

# Other
# DEFAULT_VIDEO_PATH = "/Users/sukanya/PycharmProjects/CMPT742_Project/datasets/mot/41315-429396382_tiny.mp4"
# DEFAULT_VIDEO_PATH = "/Users/sukanya/PycharmProjects/CMPT742_Project/datasets/mot/Moderate 2.mp4"

# Use part A for dense crowds and part B for moderate to low crowds
WEIGHTS_PATH = "/Users/sukanya/PycharmProjects/CMPT742_Project/models/CSRNet_pytorch/weights/PartAmodel_best.pth.tar"


# ---------------- main ---------------- #
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

    print(f"\nStarting Final Integrated Crowd Analysis System (6-Panel Layout)...")
    print(f"Video Source: {video_path}\n")

    # --- initialise all processing modules --- #
    # yolo handles: person detection
    yolo = YOLODetector()
    # tracker handles: tracking
    tracker = PeopleTracker() # internally initialise DeepSort(max_age=15)
    # csr handles: CSRNet density predictor
    csr = DensityEstimator(WEIGHTS_PATH)
    # surge_detector handles: temporal surge - difference of density
    surge_detector = TemporalSurgeDetector()
    # zoomer handles: dynamic zooming - focus on hotspots
    zoomer = DynamicZoomer(zoom_factor=3.0, min_crop=100)

    # --- open video --- #
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    frame_idx = 0 # count frames processed
    start_time = time.time() # initialise start time

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
        tracked = tracker.update(frame, detections) # this calls deepsort PeopleTracker.update()

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
        surge_overlay = cv2.addWeighted(frame, 0.1, surge_view_raw, 0.4, 0)
        surge_view = surge_overlay

        # to see only surge comment out the above line and uncomment this
        # surge_view = surge_view_raw

        # --- dynamic zoom --- #
        zoomed_view = zoomer.zoom_to_surge_area(frame, surge_map)

        # --- Automatic switching between YOLO and CSRNet --- #
        # YOLO detections
        yolo_count = len(detections)
        # CSRNet estimated crowd count
        csrnet_count = np.sum(density)

        # if crowd is dense -> YOLO performs poorly -> switch to CSRNet
        # threshold currently set to 35
        if yolo_count >= 35 or csrnet_count >= 35:
            USE_YOLO = False
            print("YOLO Count:", yolo_count)
            print("CSRNet Count:", csrnet_count)
        else:
            USE_YOLO = True
            print("YOLO Count:", yolo_count)
            print("CSRNet Count:", csrnet_count)

    # human-readable panel label
        active_method_text = "YOLO (Sparse Mode)" if USE_YOLO else "CSRNet (Dense Mode)"

        # --- dim inactive panel --- # (20% brightness)
        if USE_YOLO:
            # if current mode is YOLO, dim CSRNet panel
            density_color = cv2.addWeighted(density_color, 0.2, np.zeros_like(density_color), 0, 0)
        else:
            # if current mode is CSRNet, dim YOLO panel
            yolo_view = cv2.addWeighted(yolo_view, 0.2, np.zeros_like(yolo_view), 0, 0)

        # --- Info panel --- #
        fps = frame_idx / (time.time() - start_time + 1e-6)
        info = [
            f"Frame: {frame_idx}",
            f"FPS: {fps:.2f}",
            f"Active Model: {active_method_text}",

            "Legend:",
            "Top-left -> YOLOv5 (Detection + Tracking)",
            "Top-right -> CSRNet Density Heatmap",
            "Middle-left -> Temporal Surge (ΔDensity)",
            "Middle-right -> Dynamic Zoom (localized)",
            "Bottom-left -> Original Video",
            "Bottom-right -> Info Panel",
            "",
            "Press 'Q' to exit"
        ]

        # create the info panel image
        h, w = frame.shape[:2]
        info_panel = make_info_panel(w // 2, h // 3, info)

        # --- resize all 6 panels to equal size (W/2 x H/3) --- #
        yolo_view = cv2.resize(yolo_view, (w // 2, h // 3))
        density_color = cv2.resize(density_color, (w // 2, h // 3))
        surge_view = cv2.resize(surge_view, (w // 2, h // 3))
        zoomed_view = cv2.resize(zoomed_view, (w // 2, h // 3))
        frame_resized = cv2.resize(frame, (w // 2, h // 3))
        info_resized = cv2.resize(info_panel, (w // 2, h // 3))

        # --- Combine panels (3x2 grid) --- #
        # horizontal stack for 2 cells and then vertical stack the 3 rows on top on one another
        row1 = np.hstack((yolo_view, density_color))
        row2 = np.hstack((surge_view, zoomed_view))
        row3 = np.hstack((frame_resized, info_resized))
        grid = np.vstack((row1, row2, row3))

        # --- Display final dashboard --- #
        cv2.imshow("Crowd Analysis System (6-Panel Integrated)", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup when done
    cap.release()
    cv2.destroyAllWindows()
    print("Finished processing. Exiting now")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()