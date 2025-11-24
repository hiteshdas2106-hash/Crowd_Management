
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import (TARGET_WIDTH, TARGET_HEIGHT, GRID_ROWS, GRID_COLS, 
                          SAFE_THRESHOLD, WARNING_THRESHOLD,
                          COLOR_SAFE, COLOR_WARNING, COLOR_DANGER)

class DangerZoneDetector:
    def __init__(self):
        self.cell_w = TARGET_WIDTH // GRID_COLS
        self.cell_h = TARGET_HEIGHT // GRID_ROWS
        
    def analyze(self, heatmap_accum):
        """
        Analyzes the heatmap accumulator to determine zone safety.
        Returns:
            zone_statuses: List of dicts with zone info
            global_status: Overall status string
            grid_overlay: Image with grid drawn
        """
        zone_statuses = []
        max_zone_heat = 0
        
        # Create a transparent overlay for the grid
        grid_overlay = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x1 = c * self.cell_w
                y1 = r * self.cell_h
                x2 = x1 + self.cell_w
                y2 = y1 + self.cell_h
                
                # Extract zone from accumulator
                zone_heat = heatmap_accum[y1:y2, x1:x2]
                total_heat = np.sum(zone_heat)
                
                # Normalize heat by area to make it comparable to thresholds
                # Or just use raw sum if thresholds are tuned for it.
                # Let's use average heat per pixel in the zone for resolution independence
                avg_heat = np.mean(zone_heat)
                
                # Determine status
                if avg_heat < 0.1: # Noise floor
                    status = "SAFE"
                    color = COLOR_SAFE
                elif avg_heat < 1.0: # Arbitrary low value, need to tune based on accumulator scale
                    # Since accumulator grows indefinitely if not decayed enough, 
                    # we need to be careful. 
                    # Let's use the thresholds from config which are likely raw values or counts.
                    # Given the heatmap logic (adding ~0.5 per frame, decaying 0.99),
                    # Steady state for 1 person = 0.5 / (1 - 0.99) = 50.
                    # So 50 is roughly 1 person standing still.
                    # Let's use the sum, but scaled down.
                    # Actually, let's just use the config thresholds against the max value in the zone
                    # or the average.
                    pass

                # Let's stick to the config thresholds and assume they refer to the MAX value in the zone
                # or the MEAN value. Let's use MEAN * 100 for easier numbers.
                score = avg_heat * 100
                
                if score < SAFE_THRESHOLD:
                    status = "SAFE"
                    color = COLOR_SAFE
                elif score < WARNING_THRESHOLD:
                    status = "WARNING"
                    color = COLOR_WARNING
                else:
                    status = "DANGER"
                    color = COLOR_DANGER
                
                zone_statuses.append({
                    "row": r, "col": c,
                    "score": score,
                    "status": status
                })
                
                if status != "SAFE":
                    max_zone_heat = max(max_zone_heat, score)
                
                # Draw grid rect
                cv2.rectangle(grid_overlay, (x1, y1), (x2, y2), color, 2)
                # Draw status text
                cv2.putText(grid_overlay, f"{status}", (x1 + 10, y1 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Determine global status
        if max_zone_heat > WARNING_THRESHOLD:
            global_status = "DANGER"
        elif max_zone_heat > SAFE_THRESHOLD:
            global_status = "WARNING"
        else:
            global_status = "SAFE"
            
        return zone_statuses, global_status, grid_overlay
