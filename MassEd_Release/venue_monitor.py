import cv2
import numpy as np
import time
import os

from utils.config import VIDEO_SOURCE
from utils.video_loader import VideoLoader
from detectors.yolo_tracker import YOLOTracker
from heatmap.heatmap_updater import HeatmapUpdater
from heatmap.danger_zone_detector import DangerZoneDetector
from detectors.doorway_counter import DoorwayCounter
from utils.logger import DataLogger
from collections import deque
import json
from datetime import datetime

class VenueMonitor:
    def __init__(self, venue_layout_path):
        self.venue_layout = cv2.imread(venue_layout_path)
        if self.venue_layout is None:
            raise ValueError(f"Could not load venue layout from {venue_layout_path}")
        
        self.venue_h, self.venue_w = self.venue_layout.shape[:2]
        print(f"Venue layout loaded: {self.venue_w}x{self.venue_h}")
        
        # Gate line points (will be set interactively)
        self.gate_line = None
        self.gate_points = []
        
        # Volunteers
        self.volunteers = {}  # {id: {'x': x, 'y': y, 'name': name}}
        self.selected_volunteer = None
        self.waypoint_mode = False  # For setting waypoints
        self._init_volunteers()
        
        # Multi-Zone System
        self.zones = []  # List of {'name': str, 'points': [(x,y)...], 'capacity': int, 'color': (B,G,R)}
        self.zone_mode = False
        self.current_zone_points = []
        
        # Flow Analysis
        self.person_trails = {}  # {person_id: deque of (x, y, timestamp)}
        self.flow_vectors_enabled = False
        self.trail_length = 30  # frames to keep
        
        # Evacuation System
        self.evacuation_mode = False
        self.evacuation_routes = []  # Pre-defined safe paths
        
        # Analytics
        self.analytics_enabled = False
        self.session_start_time = time.time()
        self.peak_count = 0
        self.total_dwell_time = 0
        self.alert_log = deque(maxlen=10)  # Last 10 alerts
        
        # Data Recording
        self.recording = False
        self.session_data = []
    
    def _init_volunteers(self):
        """Initialize some test volunteers"""
        import random
        # Create 7 volunteers at random positions
        self.volunteers = {}
        for i in range(1, 8):
            self.volunteers[i] = {
                'x': random.randint(50, self.venue_w - 50),
                'y': random.randint(50, self.venue_h - 50),
                'name': f'V{i}',
                'vx': random.choice([-2, -1, 0, 1, 2]),  # velocity x
                'vy': random.choice([-2, -1, 0, 1, 2]),   # velocity y
                'waypoint': None,  # Target waypoint (x, y)
                'instruction': None  # Text instruction
            }
        self.selected_volunteer = 1  # Start with V1 selected
        print(f"Initialized {len(self.volunteers)} volunteers")
        
    def select_gate_line(self):
        """Interactive gate line selection"""
        print("\n=== GATE LINE SELECTION ===")
        print("Click TWO points to define the entry/exit gate line")
        print("Press 'r' to reset, 'q' to finish")
        
        clone = self.venue_layout.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.gate_points) < 2:
                    self.gate_points.append((x, y))
                    cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                    
                    if len(self.gate_points) == 2:
                        cv2.line(clone, self.gate_points[0], self.gate_points[1], (0, 255, 0), 3)
                        print(f"Gate line defined: {self.gate_points[0]} -> {self.gate_points[1]}")
                    
                    cv2.imshow("Select Gate Line", clone)
        
        cv2.namedWindow("Select Gate Line")
        cv2.setMouseCallback("Select Gate Line", mouse_callback)
        
        while True:
            cv2.imshow("Select Gate Line", clone)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                self.gate_points = []
                clone = self.venue_layout.copy()
                print("Reset. Click two points again.")
            elif key == ord('q'):
                if len(self.gate_points) == 2:
                    self.gate_line = self.gate_points
                    break
                else:
                    print("Please select 2 points before quitting!")
        
        cv2.destroyWindow("Select Gate Line")
        print(f"Gate line set: {self.gate_line}\n")
    
    def _count_people_in_zone(self, zone, people_positions):
        """Count how many people are inside a given zone polygon"""
        count = 0
        points = np.array(zone['points'], dtype=np.int32)
        for (tid, x, y) in people_positions:
            if cv2.pointPolygonTest(points, (x, y), False) >= 0:
                count += 1
        return count
    
    def _calculate_flow_vector(self, person_id):
        """Calculate movement vector for a person based on their trail"""
        if person_id not in self.person_trails or len(self.person_trails[person_id]) < 2:
            return None
        
        trail = list(self.person_trails[person_id])
        # Compare current position with position 5 frames ago
        if len(trail) < 5:
            return None
        
        current = trail[-1]
        past = trail[-5]
        
        dx = current[0] - past[0]
        dy = current[1] - past[1]
        
        # Only return if movement is significant
        if abs(dx) > 2 or abs(dy) > 2:
            return (dx, dy)
        return None
    
    def _auto_dispatch_volunteers(self, danger_zones):
        """Automatically assign volunteers to danger zones"""
        if not danger_zones:
            return
        
        # Find idle volunteers
        idle_volunteers = [v_id for v_id, v_data in self.volunteers.items() if not v_data['waypoint']]
        
        if not idle_volunteers:
            return
        
        # Assign to danger zones
        for i, zone in enumerate(danger_zones[:len(idle_volunteers)]):
            v_id = idle_volunteers[i]
            # Send to center of danger zone
            zone_points = np.array(zone['points'])
            center_x = int(np.mean(zone_points[:, 0]))
            center_y = int(np.mean(zone_points[:, 1]))
            
            self.volunteers[v_id]['waypoint'] = (center_x, center_y)
            self.volunteers[v_id]['instruction'] = f"URGENT: Go to {zone['name']}"
            self._add_alert(f"Dispatched {self.volunteers[v_id]['name']} to {zone['name']}", "HIGH")
    
    def _add_alert(self, message, priority="INFO"):
        """Add an alert to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.alert_log.append({
            'time': timestamp,
            'message': message,
            'priority': priority
        })
        print(f"[{priority}] {timestamp}: {message}")
    
    def _save_session_data(self):
        """Save session data to JSON file"""
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        data = {
            'session_start': datetime.fromtimestamp(self.session_start_time).isoformat(),
            'session_end': datetime.now().isoformat(),
            'peak_count': self.peak_count,
            'zones': self.zones,
            'alerts': list(self.alert_log),
            'data_points': self.session_data
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Session data saved to {filename}")
    
    def initialize(self):
        """Initialize components for monitoring"""
        print("Initializing monitoring system...")
        self.loader = VideoLoader(VIDEO_SOURCE)
        self.tracker = YOLOTracker()
        self.heatmap_updater = HeatmapUpdater()
        self.zone_detector = DangerZoneDetector()
        self.logger = DataLogger(filename="venue_session_log.csv")
        
        # Initialize doorway counter with the gate line
        if self.gate_line:
            self.doorway_counter = DoorwayCounter(self.gate_line)
        else:
            print("WARNING: No gate line defined. Skipping doorway counting.")
            self.doorway_counter = None
            
        print("System initialized.")

    def process_frame(self):
        """Process a single frame and return the annotated image"""
        start_time = time.time()
        
        # 1. Get frame from camera
        frame = self.loader.get_frame()
        if frame is None:
            return None, 0, 0, {'enter': 0, 'exit': 0, 'inside': 0}, "UNKNOWN"
        
        cam_h, cam_w = frame.shape[:2]
        
        # 2. Track people
        results, positions, current_count, unique_count = self.tracker.detect_and_track(frame)
        
        # 3. Update heatmap (in camera coordinates)
        self.heatmap_updater.update(positions)
        heatmap_gray = self.heatmap_updater.get_raw_accum()
        
        # 4. Map positions to venue coordinates
        venue_positions = []
        for (tid, cx, cy) in positions:
            vx = int((cx / cam_w) * self.venue_w)
            vy = int((cy / cam_h) * self.venue_h)
            venue_positions.append((tid, vx, vy))
            
            # Update person trails for flow analysis
            if tid not in self.person_trails:
                self.person_trails[tid] = deque(maxlen=self.trail_length)
            self.person_trails[tid].append((vx, vy, time.time()))
        
        # Clean up old trails
        active_ids = {tid for tid, _, _ in venue_positions}
        self.person_trails = {tid: trail for tid, trail in self.person_trails.items() if tid in active_ids}
        
        # Update peak count
        if current_count > self.peak_count:
            self.peak_count = current_count
        
        # Analyze zones if defined
        zone_stats = []
        danger_zones = []
        if self.zones:
            for zone in self.zones:
                count = self._count_people_in_zone(zone, venue_positions)
                capacity = zone.get('capacity', 50)
                utilization = count / capacity if capacity > 0 else 0
                
                status = "SAFE"
                if utilization > 0.9:
                    status = "DANGER"
                    danger_zones.append(zone)
                elif utilization > 0.7:
                    status = "WARNING"
                
                zone_stats.append({
                    'zone': zone,
                    'count': count,
                    'capacity': capacity,
                    'utilization': utilization,
                    'status': status
                })
        
        # Auto-dispatch in evacuation mode
        if self.evacuation_mode and danger_zones:
            self._auto_dispatch_volunteers(danger_zones)
        
        # Update volunteer positions (random movement OR waypoint navigation)
        import random
        import math
        for v_id, v_data in self.volunteers.items():
            if v_data['waypoint']:
                # Navigate to waypoint
                wx, wy = v_data['waypoint']
                dx = wx - v_data['x']
                dy = wy - v_data['y']
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 5:  # Reached waypoint
                    v_data['waypoint'] = None
                    v_data['instruction'] = "Arrived!"
                    print(f"✓ {v_data['name']} reached destination!")
                else:
                    # Move towards waypoint
                    speed = 3
                    v_data['x'] += int((dx / dist) * speed)
                    v_data['y'] += int((dy / dist) * speed)
            else:
                # Random walk
                v_data['x'] += v_data['vx']
                v_data['y'] += v_data['vy']
                
                # Bounce off walls
                if v_data['x'] <= 0 or v_data['x'] >= self.venue_w:
                    v_data['vx'] = -v_data['vx']
                    v_data['x'] = max(0, min(self.venue_w, v_data['x']))
                
                if v_data['y'] <= 0 or v_data['y'] >= self.venue_h:
                    v_data['vy'] = -v_data['vy']
                    v_data['y'] = max(0, min(self.venue_h, v_data['y']))
                
                # Occasionally change direction
                if random.random() < 0.02:  # 2% chance per frame
                    v_data['vx'] = random.choice([-2, -1, 0, 1, 2])
                    v_data['vy'] = random.choice([-2, -1, 0, 1, 2])
        
        # 5. Update doorway counter (in venue coordinates)
        if self.doorway_counter:
            self.doorway_counter.update(venue_positions)
            doorway_counts = self.doorway_counter.get_counts()
        else:
            doorway_counts = {'enter': 0, 'exit': 0, 'inside': 0}
        
        # 6. Analyze danger zones
        zone_statuses, global_status, grid_overlay = self.zone_detector.analyze(heatmap_gray)
        
        # 7. Log data
        self.logger.log(unique_count, current_count, doorway_counts['enter'], doorway_counts['exit'], global_status)
        
        # 8. Visualization
        # Start with venue layout
        display = self.venue_layout.copy()
        
        # Resize heatmap to venue size
        heatmap_color = self.heatmap_updater.get_heatmap_image()
        heatmap_resized = cv2.resize(heatmap_color, (self.venue_w, self.venue_h))
        
        # Overlay heatmap
        display = cv2.addWeighted(display, 0.6, heatmap_resized, 0.4, 0)
        
        # Resize and overlay danger grid
        grid_resized = cv2.resize(grid_overlay, (self.venue_w, self.venue_h))
        display = cv2.addWeighted(display, 1.0, grid_resized, 0.3, 0)
        
        # Draw tracked people on venue
        for (tid, vx, vy) in venue_positions:
            cv2.circle(display, (vx, vy), 3, (0, 255, 0), -1)
            cv2.putText(display, f"{tid}", (vx, vy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw volunteers
        for v_id, v_data in self.volunteers.items():
            vx, vy = v_data['x'], v_data['y']
            
            # Draw path to waypoint if exists
            if v_data['waypoint']:
                wx, wy = v_data['waypoint']
                cv2.line(display, (vx, vy), (wx, wy), (255, 255, 0), 2)  # Yellow line
                cv2.circle(display, (wx, wy), 8, (255, 255, 0), 2)  # Yellow target circle
                cv2.circle(display, (wx, wy), 3, (255, 255, 0), -1)
            
            # Highlight selected volunteer
            if v_id == self.selected_volunteer:
                cv2.circle(display, (vx, vy), 15, (0, 255, 255), 3)  # Yellow ring
                cv2.circle(display, (vx, vy), 10, (255, 0, 255), -1)  # Magenta fill
            else:
                # Color based on status
                if v_data['waypoint']:
                    cv2.circle(display, (vx, vy), 10, (0, 165, 255), -1)  # Orange (has mission)
                else:
                    cv2.circle(display, (vx, vy), 10, (255, 0, 0), -1)  # Blue (idle)
            
            cv2.circle(display, (vx, vy), 10, (255, 255, 255), 2)  # White border
            
            # Label
            cv2.putText(display, v_data['name'], (vx-10, vy-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Show instruction if exists
            if v_data['instruction']:
                cv2.putText(display, v_data['instruction'], (vx-30, vy+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw zones
        for zone_stat in zone_stats:
            zone = zone_stat['zone']
            points = np.array(zone['points'], dtype=np.int32)
            
            # Color based on status
            if zone_stat['status'] == "DANGER":
                color = (0, 0, 255)  # Red
            elif zone_stat['status'] == "WARNING":
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green
            
            # Draw polygon
            cv2.polylines(display, [points], True, color, 3)
            
            # Fill with transparency
            overlay = display.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
            
            # Zone label
            center_x = int(np.mean(points[:, 0]))
            center_y = int(np.mean(points[:, 1]))
            label = f"{zone['name']}: {zone_stat['count']}/{zone_stat['capacity']}"
            cv2.putText(display, label, (center_x-50, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw current zone being created
        if self.zone_mode and len(self.current_zone_points) > 0:
            for i, pt in enumerate(self.current_zone_points):
                cv2.circle(display, pt, 5, (255, 0, 255), -1)
                if i > 0:
                    cv2.line(display, self.current_zone_points[i-1], pt, (255, 0, 255), 2)
        
        # Draw flow vectors
        if self.flow_vectors_enabled:
            for tid, vx, vy in venue_positions:
                vector = self._calculate_flow_vector(tid)
                if vector:
                    dx, dy = vector
                    # Scale for visibility
                    end_x = int(vx + dx * 3)
                    end_y = int(vy + dy * 3)
                    cv2.arrowedLine(display, (vx, vy), (end_x, end_y), (255, 255, 0), 2, tipLength=0.3)
        
        # Draw gate line and counts
        if self.gate_line:
            cv2.line(display, self.gate_line[0], self.gate_line[1], (0, 255, 255), 3)
            mid_x = (self.gate_line[0][0] + self.gate_line[1][0]) // 2
            mid_y = (self.gate_line[0][1] + self.gate_line[1][1]) // 2
            
            # Gate stats
            gate_text = f"IN:{doorway_counts['enter']} OUT:{doorway_counts['exit']} INSIDE:{doorway_counts['inside']}"
            cv2.rectangle(display, (mid_x-100, mid_y-30), (mid_x+100, mid_y+10), (0, 0, 0), -1)
            cv2.putText(display, gate_text, (mid_x-90, mid_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Dashboard overlay
        self._draw_dashboard(display, unique_count, current_count, doorway_counts, global_status)
        
        # Analytics Dashboard (right side)
        if self.analytics_enabled:
            panel_x = self.venue_w - 250
            panel_y = 70
            panel_w = 240
            panel_h = 200
            
            # Semi-transparent panel
            overlay = display.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            
            # Title
            cv2.putText(display, "ANALYTICS", (panel_x + 10, panel_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Stats
            y_offset = panel_y + 50
            line_height = 25
            
            session_time = int(time.time() - self.session_start_time)
            stats_text = [
                f"Session: {session_time//60}m {session_time%60}s",
                f"Peak Count: {self.peak_count}",
                f"Zones: {len(self.zones)}",
                f"Volunteers: {len([v for v in self.volunteers.values() if v['waypoint']])} active",
                f"Alerts: {len(self.alert_log)}"
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(display, text, (panel_x + 10, y_offset + i * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Alert log
            alert_y = y_offset + len(stats_text) * line_height + 10
            cv2.putText(display, "Recent Alerts:", (panel_x + 10, alert_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            for i, alert in enumerate(list(self.alert_log)[-3:]):
                color = (0, 0, 255) if alert['priority'] == "HIGH" else (0, 165, 255) if alert['priority'] == "MEDIUM" else (255, 255, 255)
                msg = alert['message'][:25] + "..." if len(alert['message']) > 25 else alert['message']
                cv2.putText(display, f"{alert['time']}: {msg}", 
                           (panel_x + 10, alert_y + 20 + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Mode indicators (bottom)
        indicators = []
        if self.waypoint_mode:
            indicators.append(("WAYPOINT MODE", (0, 255, 255)))
        if self.zone_mode:
            indicators.append((f"ZONE MODE: {len(self.current_zone_points)} points (Enter to finish)", (255, 0, 255)))
        if self.evacuation_mode:
            indicators.append(("EVACUATION MODE ACTIVE", (0, 0, 255)))
        if self.flow_vectors_enabled:
            indicators.append(("Flow Vectors ON", (255, 255, 0)))
        
        for i, (text, color) in enumerate(indicators):
            y_pos = self.venue_h - 40 - (i * 35)
            cv2.rectangle(display, (10, y_pos), (400, y_pos + 30), color, -1)
            cv2.putText(display, text, (15, y_pos + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return display, current_count, unique_count, doorway_counts, global_status

    def run(self):
        """Main monitoring loop (Legacy mode)"""
        self.initialize()
        
        print("System ready. Starting monitoring...")
        print("\n=== KEYBOARD CONTROLS ===")
        print("  BASIC:")
        print("    q: Quit | s: Save session data")
        print("  VOLUNTEERS:")
        print("    1-7: Select volunteer | w: Waypoint mode | c: Clear waypoint")
        print("    Arrow Keys: Manual move | r: Randomize positions")
        print("  ZONES:")
        print("    z: Zone drawing mode | n: Name last zone | x: Delete last zone")
        print("  ANALYSIS:")
        print("    f: Toggle flow vectors | d: Toggle analytics dashboard")
        print("    e: EMERGENCY evacuation mode | a: Auto-dispatch volunteers")
        print("========================\n")
        
        # Mouse callback for waypoints and zones
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.zone_mode:
                    # Add point to current zone
                    self.current_zone_points.append((x, y))
                    print(f"Zone point {len(self.current_zone_points)}: ({x}, {y})")
                elif self.waypoint_mode:
                    if self.selected_volunteer in self.volunteers:
                        self.volunteers[self.selected_volunteer]['waypoint'] = (x, y)
                        self.volunteers[self.selected_volunteer]['instruction'] = f"Go to ({x}, {y})"
                        print(f"✓ Set waypoint for {self.volunteers[self.selected_volunteer]['name']}: ({x}, {y})")
                        self.waypoint_mode = False  # Auto-disable after setting
        
        cv2.namedWindow("Venue Monitor")
        cv2.setMouseCallback("Venue Monitor", mouse_callback)
        
        while True:
            display, current_count, unique_count, doorway_counts, global_status = self.process_frame()
            if display is None:
                print("End of video stream.")
                break
            
            # Display
            cv2.imshow("Venue Monitor", display)
            
            # Keyboard controlse output
            sel_v = self.volunteers.get(self.selected_volunteer, {})
            print(f"Count: {current_count} | Total: {unique_count} | IN: {doorway_counts['enter']} | OUT: {doorway_counts['exit']} | Status: {global_status} | Selected: {sel_v.get('name', 'None')} ({sel_v.get('x', 0)}, {sel_v.get('y', 0)})")
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7')]:
                self.selected_volunteer = int(chr(key))
            elif key == ord('w'):
                self.waypoint_mode = not self.waypoint_mode
                status = "ENABLED" if self.waypoint_mode else "DISABLED"
                print(f"Waypoint mode {status}")
            elif key == ord('c'):
                if self.selected_volunteer in self.volunteers:
                    self.volunteers[self.selected_volunteer]['waypoint'] = None
                    self.volunteers[self.selected_volunteer]['instruction'] = None
                    print(f"Cleared waypoint for {self.volunteers[self.selected_volunteer]['name']}")
            elif key == ord('r'):
                # Randomize all volunteer positions
                import random
                for v_id in self.volunteers:
                    self.volunteers[v_id]['x'] = random.randint(50, self.venue_w - 50)
                    self.volunteers[v_id]['y'] = random.randint(50, self.venue_h - 50)
                    self.volunteers[v_id]['vx'] = random.choice([-2, -1, 0, 1, 2])
                    self.volunteers[v_id]['vy'] = random.choice([-2, -1, 0, 1, 2])
            
            # Zone controls
            elif key == ord('z'):
                self.zone_mode = not self.zone_mode
                if self.zone_mode:
                    self.current_zone_points = []
                    print("Zone drawing mode ENABLED. Click points, press Enter to finish, Esc to cancel.")
                else:
                    self.current_zone_points = []
                    print("Zone drawing mode DISABLED")
            elif key == 13 and self.zone_mode:  # Enter key
                if len(self.current_zone_points) >= 3:
                    # Create new zone
                    zone_name = f"Zone {len(self.zones) + 1}"
                    self.zones.append({
                        'name': zone_name,
                        'points': self.current_zone_points.copy(),
                        'capacity': 50,  # Default capacity
                        'color': (0, 255, 0)
                    })
                    self._add_alert(f"Created {zone_name} with {len(self.current_zone_points)} points", "INFO")
                    self.current_zone_points = []
                    self.zone_mode = False
                else:
                    print("Need at least 3 points to create a zone!")
            elif key == 27 and self.zone_mode:  # Escape key
                self.current_zone_points = []
                self.zone_mode = False
                print("Zone creation cancelled")
            elif key == ord('x'):
                if self.zones:
                    removed = self.zones.pop()
                    print(f"Deleted {removed['name']}")
                else:
                    print("No zones to delete")
            elif key == ord('n'):
                if self.zones:
                    new_name = input("Enter zone name: ")
                    self.zones[-1]['name'] = new_name
                    print(f"Renamed last zone to: {new_name}")
            
            # Analysis controls
            elif key == ord('f'):
                self.flow_vectors_enabled = not self.flow_vectors_enabled
                status = "ENABLED" if self.flow_vectors_enabled else "DISABLED"
                print(f"Flow vectors {status}")
            elif key == ord('d'):
                self.analytics_enabled = not self.analytics_enabled
                status = "ENABLED" if self.analytics_enabled else "DISABLED"
                print(f"Analytics dashboard {status}")
            elif key == ord('e'):
                self.evacuation_mode = not self.evacuation_mode
                if self.evacuation_mode:
                    self._add_alert("EVACUATION MODE ACTIVATED", "HIGH")
                    print("🚨 EVACUATION MODE ACTIVATED 🚨")
                else:
                    print("Evacuation mode deactivated")
            elif key == ord('a'):
                # Manual auto-dispatch
                if zone_stats:
                    danger = [zs['zone'] for zs in zone_stats if zs['status'] == "DANGER"]
                    if danger:
                        self._auto_dispatch_volunteers(danger)
                    else:
                        print("No danger zones to dispatch to")
                else:
                    print("No zones defined")
            elif key == ord('s'):
                self._save_session_data()
            
            # Arrow key controls (existing)
            elif key == 82 or key == 0:  # Up arrow
                if self.selected_volunteer in self.volunteers:
                    self.volunteers[self.selected_volunteer]['y'] = max(0, self.volunteers[self.selected_volunteer]['y'] - 10)
                    self.volunteers[self.selected_volunteer]['waypoint'] = None  # Cancel waypoint on manual move
            elif key == 84 or key == 1:  # Down arrow
                if self.selected_volunteer in self.volunteers:
                    self.volunteers[self.selected_volunteer]['y'] = min(self.venue_h, self.volunteers[self.selected_volunteer]['y'] + 10)
                    self.volunteers[self.selected_volunteer]['waypoint'] = None
            elif key == 81 or key == 2:  # Left arrow
                if self.selected_volunteer in self.volunteers:
                    self.volunteers[self.selected_volunteer]['x'] = max(0, self.volunteers[self.selected_volunteer]['x'] - 10)
                    self.volunteers[self.selected_volunteer]['waypoint'] = None
            elif key == 83 or key == 3:  # Right arrow
                if self.selected_volunteer in self.volunteers:
                    self.volunteers[self.selected_volunteer]['x'] = min(self.venue_w, self.volunteers[self.selected_volunteer]['x'] + 10)
                    self.volunteers[self.selected_volunteer]['waypoint'] = None
        
        self.loader.release()
        cv2.destroyAllWindows()
        print("\nMonitoring stopped.")
    
    def _draw_dashboard(self, frame, total, current, gate_counts, status):
        """Draw dashboard overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "VENUE MONITOR", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Metrics
        x = 250
        gap = 110
        
        cv2.putText(frame, f"TOTAL: {total}", (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"CURRENT: {current}", (x + gap, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"IN: {gate_counts['enter']}", (x + gap*2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"OUT: {gate_counts['exit']}", (x + gap*3, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
        
        # Status
        s_color = (0, 255, 0) if status == "SAFE" else (0, 255, 255) if status == "WARNING" else (0, 0, 255)
        cv2.putText(frame, f"STATUS: {status}", (x + gap*4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 2)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venue_layout_path = os.path.join(script_dir, "venue_layout.png")
    
    if not os.path.exists(venue_layout_path):
        print(f"ERROR: Venue layout not found at {venue_layout_path}")
        return
    
    monitor = VenueMonitor(venue_layout_path)
    
    # Interactive gate selection
    monitor.select_gate_line()
    
    # Start monitoring
    monitor.run()

if __name__ == "__main__":
    main()
