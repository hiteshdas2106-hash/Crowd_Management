
import cv2
from .config import TARGET_WIDTH, TARGET_HEIGHT

class VideoLoader:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Resize for consistent processing
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        return frame

    def release(self):
        self.cap.release()
