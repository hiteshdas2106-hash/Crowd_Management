
import cv2
from .config import COLOR_TEXT

def draw_text(img, text, pos, color=COLOR_TEXT, scale=0.6, thickness=2):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def draw_bbox(img, box, label=None, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        draw_text(img, label, (x1, y1 - 10), color=color)

def overlay_heatmap(frame, heatmap_color, alpha=0.6):
    return cv2.addWeighted(heatmap_color, alpha, frame, 1 - alpha, 0)

def draw_dashboard(frame, stats):
    """
    Draws a polished dashboard overlay.
    stats: dict containing 'total', 'current', 'enter', 'exit', 'status'
    """
    h, w = frame.shape[:2]
    
    # Dashboard background (Top bar)
    # Semi-transparent black
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "MassEd.ex Monitor", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Metrics
    # Layout: Total | Current | Enter | Exit | Status
    
    # Helper to draw metric
    def draw_metric(label, value, x, color=(255, 255, 255)):
        cv2.putText(frame, label, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, str(value), (x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    start_x = 250
    gap = 120
    
    draw_metric("TOTAL", stats['total'], start_x)
    draw_metric("CURRENT", stats['current'], start_x + gap)
    draw_metric("ENTER", stats['enter'], start_x + gap*2, (0, 255, 0))
    draw_metric("EXIT", stats['exit'], start_x + gap*3, (0, 100, 255))
    
    # Status needs color based on value
    status = stats['status']
    s_color = (0, 255, 0)
    if status == "WARNING": s_color = (0, 255, 255)
    if status == "DANGER": s_color = (0, 0, 255)
    
    draw_metric("STATUS", status, start_x + gap*4, s_color)

