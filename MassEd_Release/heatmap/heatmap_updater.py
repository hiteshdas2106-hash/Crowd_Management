
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import TARGET_WIDTH, TARGET_HEIGHT, DECAY_FACTOR, BLOB_RADIUS, BLOB_INTENSITY

class HeatmapUpdater:
    def __init__(self):
        self.heatmap_accum = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)
        
    def update(self, positions):
        """
        Updates the heatmap with new positions.
        positions: List of (id, cx, cy)
        """
        # Decay existing heat
        self.heatmap_accum *= DECAY_FACTOR
        
        # Add new heat
        # Create a temporary mask for the current frame's blobs to add efficiently
        temp_mask = np.zeros_like(self.heatmap_accum)
        
        for _, cx, cy in positions:
            # Ensure coordinates are within bounds
            if 0 <= cx < TARGET_WIDTH and 0 <= cy < TARGET_HEIGHT:
                # Draw a filled circle on the temp mask
                # We use a simple circle for speed, but could use a true Gaussian kernel if needed
                # For visual "Gaussian-like" effect, we can draw multiple concentric circles or apply blur
                cv2.circle(temp_mask, (cx, cy), BLOB_RADIUS, (1,), -1)
        
        # Blur the temp mask to make it look like Gaussian blobs
        temp_mask = cv2.GaussianBlur(temp_mask, (0, 0), sigmaX=BLOB_RADIUS/2)
        
        # Add to accumulator
        self.heatmap_accum += temp_mask * BLOB_INTENSITY
        
        # Clamp values to avoid overflow (though float32 is large, we want controlled visualization)
        # We don't strictly clamp to 1.0 here to allow "hotter" spots, but we will normalize for display
        
    def get_heatmap_image(self):
        """
        Returns the colorized heatmap image.
        """
        # Normalize to 0-255
        # We use a dynamic max to adjust to the current heat levels, or a fixed max for absolute heat
        # Let's use a fixed max for stability, or dynamic with a minimum floor
        
        max_val = np.max(self.heatmap_accum)
        if max_val == 0:
            norm_map = self.heatmap_accum
        else:
            # Scale so that the hottest point is 255, but avoid flickering by using a rolling max or fixed cap
            # For this implementation, we'll just normalize to 255 for best visualization
            norm_map = (self.heatmap_accum / max_val) * 255
            
        norm_map = norm_map.astype(np.uint8)
        
        # Apply colormap
        color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        return color_map

    def get_raw_accum(self):
        return self.heatmap_accum
