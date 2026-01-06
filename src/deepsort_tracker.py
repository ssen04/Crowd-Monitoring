# src/deepsort_tracker.py
"""
DeepSORT Tracker Module

Keeps track of detected people across frames - not being used currently but can be useful for future development
"""

from deep_sort_realtime.deepsort_tracker import DeepSort

class PeopleTracker:
    def __init__(self, max_age=15):
        # initialise the DeepSort tracker instance
        # max_age=15 === a person can disappear from the frame for up to 15 frames and still be considered the same person when they reappear
        # If they are missing for more than 15 frames, DeepSORT will remove the track, and if they reappear later, they will get a new track ID
        self.tracker = DeepSort(max_age=max_age)

    def update(self, frame, detections):
        """
        Update the tracker with new detections for the current frame.

        Takes YOLO detections as input and returns tracked objects with IDs.
        detections: list of [x1, y1, x2, y2]
        """
        # update tracker with current detections and frame
        # returns a list of Track objects
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # initialise list to store confirmed tracked objects
        tracked_boxes = []

        # loop through all returned tracks
        for t in tracks:
            # only include confirmed tracks (not tentative or lost)
            if t.is_confirmed():
                # convert track bbox to [left, top, width, height] format
                bbox = t.to_ltwh()  # left, top, width, height
                # append tuple of (track_id, bbox) to the list
                tracked_boxes.append((t.track_id, bbox))

        # return all confirmed tracked objects for this frame
        return tracked_boxes
