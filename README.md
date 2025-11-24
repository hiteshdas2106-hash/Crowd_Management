# MassEd.ex Venue Monitor

A real-time crowd monitoring and management system using YOLOv8 and computer vision.

## Features
- **Real-time People Tracking**: Uses YOLOv8 and ByteTrack for accurate people counting.
- **Heatmap Generation**: Visualizes crowd density over time.
- **Danger Zone Detection**: Automatically identifies overcrowded areas.
- **Volunteer Management**: Simulates and tracks volunteer positions for crowd control.
- **Flow Analysis**: Visualizes movement patterns with flow vectors.

## Installation

1.  **Clone the repository** (or download the source code).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install `ultralytics` and `lapx` separately if you encounter issues.*

## Usage

Run the main monitor script:

```bash
python venue_monitor.py
```

### Controls
- **Gate Selection**: Click two points to define the entry/exit gate.
- **Keyboard Shortcuts**:
    - `q`: Quit
    - `z`: Draw a new zone (Click points, Enter to finish)
    - `x`: Delete last zone
    - `d`: Toggle analytics dashboard
    - `f`: Toggle flow vectors
    - `e`: Toggle EVACUATION MODE
    - `1-7`: Select a volunteer
    - `w`: Set waypoint for selected volunteer (Click on map)

## Configuration
Adjust settings in `utils/config.py` to change video source, thresholds, and colors.
