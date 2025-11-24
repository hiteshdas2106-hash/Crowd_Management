
import cv2

# System Config
WINDOW_NAME = "MassEd.ex System"
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Tracker Config
MODEL_PATH = "yolov8n.pt"
TRACKER_TYPE = "bytetrack.yaml"  # or 'botsort.yaml'
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

# Heatmap Config
HEATMAP_ALPHA = 0.6  # Overlay transparency
DECAY_FACTOR = 0.99  # How fast heat fades (0.0 to 1.0)
BLOB_RADIUS = 40
BLOB_INTENSITY = 0.5  # Heat added per frame per person

# Danger Zones
GRID_ROWS = 3
GRID_COLS = 3
SAFE_THRESHOLD = 50
WARNING_THRESHOLD = 150
# Above WARNING is DANGER

# Colors (B, G, R)
COLOR_SAFE = (0, 255, 0)
COLOR_WARNING = (0, 255, 255)
COLOR_DANGER = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)

# Volunteer Tracking Config
VENUE_MAP_PATH = "venue_layout.png"

# API Config
API_HOST = "0.0.0.0"
API_PORT = 5001
API_DEBUG = True

# Volunteer Visualization
VOLUNTEER_MARKER_RADIUS = 8
VOLUNTEER_COLOR_OK = (0, 255, 0)      # Green
VOLUNTEER_COLOR_WARNING = (0, 255, 255)  # Yellow
VOLUNTEER_COLOR_DANGER = (0, 0, 255)   # Red

# Alert Settings
AUTO_ALERT_DANGER = True  # Auto-alert on danger zone entry
AUTO_ALERT_WARNING = False  # Auto-alert on warning zone entry
ALERT_COOLDOWN_SECONDS = 30  # Minimum time between duplicate alerts
