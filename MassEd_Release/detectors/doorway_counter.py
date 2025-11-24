import cv2
import numpy as np

class DoorwayCounter:
    """
    Tracks people crossing a virtual line to count entries and exits.
    """
    def __init__(self, line_points):
        """
        Initialize the doorway counter.
        
        Args:
            line_points (list): [(x1, y1), (x2, y2)] defining the doorway line.
        """
        self.line_points = line_points
        self.enter_count = 0
        self.exit_count = 0
        
        # Store previous positions of IDs to detect crossing
        # {id: (x, y)}
        self.previous_positions = {}
        
        # Keep track of IDs that have already been counted to avoid double counting
        # (Optional, but good for robustness if they linger on the line)
        # For simplicity in this version, we rely on the crossing logic.
        
        # List of IDs currently passing (for visualization/debug)
        self.passing_ids = []

    def update(self, detections):
        """
        Update counts based on new detections.
        
        Args:
            detections (list): List of tuples (id, x, y) from the tracker.
        """
        current_ids = set()
        self.passing_ids = []
        
        for (tid, cx, cy) in detections:
            current_ids.add(tid)
            
            if tid in self.previous_positions:
                prev_cx, prev_cy = self.previous_positions[tid]
                
                # Check for line intersection
                if self._intersect(self.line_points[0], self.line_points[1], (prev_cx, prev_cy), (cx, cy)):
                    # Determine direction
                    # We use the cross product to determine which side of the line the point is on.
                    # Vector AB = B - A
                    # Vector AP = P - A
                    # Cross product (ABxAP) z-component tells us the side.
                    
                    # Let's define "Enter" as crossing from Side A to Side B
                    # and "Exit" as crossing from Side B to Side A.
                    # We need to know which side is which. 
                    # Convention: 
                    # Line is defined as P1 -> P2.
                    # "Left" of the vector P1->P2 is one side, "Right" is the other.
                    
                    val_prev = self._ccw(self.line_points[0], self.line_points[1], (prev_cx, prev_cy))
                    val_curr = self._ccw(self.line_points[0], self.line_points[1], (cx, cy))
                    
                    # If signs are different, they are on opposite sides (which we already know from intersect)
                    # But we need to know direction.
                    
                    # Assuming:
                    # Positive result means "Counter-Clockwise" / Left
                    # Negative result means "Clockwise" / Right
                    
                    # If moving from Right (neg) to Left (pos) -> ENTER
                    # If moving from Left (pos) to Right (neg) -> EXIT
                    # (This is arbitrary, user can swap points to swap direction)
                    
                    if val_prev < 0 and val_curr > 0:
                        self.exit_count += 1
                        self.passing_ids.append(tid)
                    elif val_prev > 0 and val_curr < 0:
                        self.enter_count += 1
                        self.passing_ids.append(tid)
                        
            # Update position
            self.previous_positions[tid] = (cx, cy)
            
        # Clean up missing IDs
        # (Remove IDs that haven't been seen in a while to save memory)
        # For this simple version, we'll just keep the ones currently seen
        # plus maybe a small buffer if needed, but strict cleanup is safer for now
        # to prevent "teleporting" crossings if an ID is lost and reassigned.
        
        # Actually, if we remove immediately, we lose tracking if detection flickers.
        # But YOLOTracker usually handles ID persistence.
        # Let's remove IDs that are NOT in the current frame.
        # A more robust way would be to keep them for N frames.
        
        keys_to_remove = []
        for tid in self.previous_positions:
            if tid not in current_ids:
                keys_to_remove.append(tid)
        
        for tid in keys_to_remove:
            del self.previous_positions[tid]

    def get_counts(self):
        """
        Return the current counts.
        """
        return {
            "enter": self.enter_count,
            "exit": self.exit_count,
            "inside": self.enter_count - self.exit_count,
            "passing_ids": self.passing_ids
        }

    def draw(self, frame):
        """
        Draw the line and counts on the frame.
        """
        p1, p2 = self.line_points
        
        # Draw line
        cv2.line(frame, p1, p2, (0, 255, 255), 2)
        
        # Draw endpoints
        cv2.circle(frame, p1, 5, (0, 0, 255), -1)
        cv2.circle(frame, p2, 5, (0, 0, 255), -1)
        
        # Draw counts
        counts = self.get_counts()
        text = f"In: {counts['enter']} | Out: {counts['exit']} | Inside: {counts['inside']}"
        
        # Background for text
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (10, 10), (10 + w + 20, 10 + h + 20), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def _ccw(self, A, B, C):
        """
        Check relative position of C to vector AB.
        Returns positive if C is to the left, negative if to the right, 0 if collinear.
        """
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

    def _intersect(self, A, B, C, D):
        """
        Return true if line segments AB and CD intersect.
        """
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)
