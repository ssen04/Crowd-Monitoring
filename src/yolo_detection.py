# src/yolo_detection.py
"""
YOLO Detection Module

Responsible for:
 - Loading YOLOv5 pretrained model
 - Running person detection on each frame
 - Returning bounding boxes and annotated frame
"""

from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name="yolov5s.pt", conf_threshold=0.35):
        # load the YOLO model
        self.model = YOLO(model_name)
        # min confidence score for keeping a detection
        # Detections below this confidence are not used
        self.conf_threshold = conf_threshold

    def detect_people(self, frame):
        """
        Runs YOLO detection on a single frame.
        Returns:
          - list of [x1, y1, x2, y2] bounding boxes
          - annotated frame for visualization ( YOLO library is drawing boxes automatically)
        """

        # run the YOLO model on the frame.
        # imgsz = 640 -> YOLO internally resizes the image to 640x640
        # conf = confidence threshold
        # model returns a list of results
        # YOLO always returns a list, even for one image -> [0] means “get the first result,” which is the one we want
        # results = self.model(frame, imgsz=640, conf=self.conf_threshold)[0]
        results = self.model(frame, imgsz=640, conf=self.conf_threshold, classes=[0])[0] # only person class

        # store all detected bounding boxes
        detections = []

        # safety check: make sure YOLO returned a 'boxes' attribute
        if hasattr(results, "boxes"):
            # loop over each detected box.
            for box in results.boxes:
                # box.cls stores the predicted class ID (tensor)
                # convert to int -> 0 means "person" in COCO dataset.
                cls = int(box.cls.cpu().numpy())
                # only keep detections where the class is "person".
                if cls == 0:  # person class (COCO class index 0)
                    # extract bounding box coordinates (x1,y1,x2,y2)
                    # YOLO gives this as a tensor -> convert to numpy, then int
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

                    # add the box to our list
                    detections.append([x1, y1, x2, y2])

        # return - the list of person bounding boxes + A YOLO-generated image with the detections drawn on it
        return detections, results.plot()
