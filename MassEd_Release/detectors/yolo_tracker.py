
from ultralytics import YOLO
import cv2
import sys
import os

# Add parent directory to path to import config if run directly (optional safety)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import MODEL_PATH, TRACKER_TYPE, CONF_THRESHOLD

class YOLOTracker:
    def __init__(self):
        print(f"Loading YOLO model: {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH)
        self.unique_ids = set()
        
    def detect_and_track(self, frame):
        """
        Runs tracking on the frame.
        Returns:
            results: Raw YOLO results
            positions: List of (id, cx, cy)
            current_count: Number of people currently tracked
            unique_count: Total unique people seen so far
        """
        # Run inference with tracking
        # persist=True is important for tracking to work across frames
        results = self.model.track(frame, persist=True, tracker=TRACKER_TYPE, 
                                   conf=CONF_THRESHOLD, classes=[0], verbose=False) # class 0 is person
        
        positions = []
        current_count = 0
        
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                positions.append((int(track_id), cx, cy))
                self.unique_ids.add(int(track_id))
                
            current_count = len(positions)
            
        return results, positions, current_count, len(self.unique_ids)
