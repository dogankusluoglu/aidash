import threading
import time
import logging
import os
import serial
import pynmea2
import gpxpy
import gpxpy.gpx
import folium
from folium import plugins
from datetime import datetime
import math
import json
import numpy as np
import cv2

try:
    from gps import gps, WATCH_ENABLE, WATCH_NEWSTYLE
    GPSD_AVAILABLE = True
except ImportError:
    GPSD_AVAILABLE = False

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class GPSManager:
    def __init__(self, config=None):
        self.config = config if config else {}
        self.port = self.config.get('port', '/dev/ttyACM0')
        self.baud = self.config.get('baud', 9600)
        self.use_gpsd = self.config.get('use_gpsd', True) and GPSD_AVAILABLE
        
        self.latest_fix = {
            "lat": None,
            "lon": None,
            "alt": None,
            "speed_kmh": 0.0,
            "timestamp": None,
            "fix_quality": 0,
            "hdop": 99.9
        }
        # route_points stores: (timestamp, lat, lon, alt, speed_kmh, vehicle_counts, tdi)
        self.route_points = []
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.callbacks = []
        
        # Analytics state
        self.total_distance = 0.0
        self.max_speed = 0.0
        self.idle_time = 0.0
        self.hard_braking_events = []
        self.hard_accel_events = []
        self.max_tdi = 0.0
        self.peak_traffic_frame = None
        self.peak_traffic_count = 0
        
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger("GPSManager")
        self.logger.setLevel(logging.INFO)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.logger.info("GPS Manager started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.logger.info("GPS Manager stopped.")

    def subscribe(self, callback):
        self.callbacks.append(callback)

    def get_latest_fix(self):
        with self.lock:
            return self.latest_fix.copy()

    def record_frame_stats(self, frame, detections, pts):
        """
        Called by inference_worker to correlate current AI detections with GPS.
        """
        counts = self._count_vehicles(detections)
        total_vehicles = sum(counts.values())
        
        with self.lock:
            fix = self.latest_fix.copy()
            if fix['lat'] is not None:
                # Calculate TDI = vehicle_count / max(speed, 1.0)
                speed = max(fix['speed_kmh'], 1.0)
                tdi = total_vehicles / speed
                
                # Update peak traffic frame
                if total_vehicles > self.peak_traffic_count:
                    self.peak_traffic_count = total_vehicles
                    self.peak_traffic_frame = frame.copy()
                
                # Check for hard braking/accel if we have points
                if self.route_points:
                    prev_p = self.route_points[-1]
                    dt = time.time() - prev_p[0]
                    if dt > 0:
                        accel = (fix['speed_kmh'] - prev_p[4]) / (dt * 3.6) # km/h/s to m/s^2 approx
                        if accel < self.config.get('hard_brake_threshold', -3.0):
                            self.hard_braking_events.append((time.time(), accel))
                        elif accel > self.config.get('hard_accel_threshold', 2.5):
                            self.hard_accel_events.append((time.time(), accel))

                # Distance accumulation
                if self.route_points:
                    prev_p = self.route_points[-1]
                    dist = haversine(prev_p[1], prev_p[2], fix['lat'], fix['lon'])
                    self.total_distance += dist
                
                if fix['speed_kmh'] > self.max_speed:
                    self.max_speed = fix['speed_kmh']
                
                if fix['speed_kmh'] < 1.0:
                    if self.route_points:
                        self.idle_time += (time.time() - self.route_points[-1][0])

                if tdi > self.max_tdi:
                    self.max_tdi = tdi

                # Store point with AI context
                point = (time.time(), fix['lat'], fix['lon'], fix['alt'], 
                         fix['speed_kmh'], counts, tdi)
                
                # Throttle to 1Hz
                if not self.route_points or (point[0] - self.route_points[-1][0] >= 1.0):
                    self.route_points.append(point)

    def _count_vehicles(self, results):
        counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
        if not results:
            return counts
            
        # Standard COCO indices for YOLO
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        vehicle_map = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        
        nms_key = next((k for k in results.keys() if 'nms' in k.lower()), None)
        if nms_key:
            detections = results[nms_key]
            if len(detections) > 0:
                for cls_id, class_detections in enumerate(detections[0]):
                    if cls_id in vehicle_map:
                        label = vehicle_map[cls_id]
                        for det in class_detections:
                            if len(det) >= 5 and det[4] >= self.config.get('threshold', 0.2):
                                counts[label] += 1
        return counts

    def _run(self):
        if self.use_gpsd:
            try:
                self._run_gpsd()
            except Exception as e:
                self.logger.error(f"gpsd connection failed: {e}. Falling back to serial.")
                self._run_serial()
        else:
            self._run_serial()

    def _run_gpsd(self):
        self.logger.info("Connecting to gpsd...")
        try:
            session = gps(mode=WATCH_ENABLE | WATCH_NEWSTYLE)
        except Exception as e:
            self.logger.error(f"Could not connect to gpsd: {e}")
            self._run_serial()
            return

        while self.running:
            try:
                report = session.next()
                if report['class'] == 'TPV':
                    fix = {
                        "lat": getattr(report, 'lat', None),
                        "lon": getattr(report, 'lon', None),
                        "alt": getattr(report, 'alt', None),
                        "speed_kmh": getattr(report, 'speed', 0.0) * 3.6,
                        "timestamp": getattr(report, 'time', None),
                        "fix_quality": getattr(report, 'mode', 0),
                        "hdop": getattr(report, 'hdop', 99.9)
                    }
                    self._update_fix(fix)
            except StopIteration:
                self.logger.error("gpsd session ended.")
                break
            except Exception as e:
                self.logger.error(f"Error in gpsd loop: {e}")
                time.sleep(1)

    def _run_serial(self):
        self.logger.info(f"Connecting to serial GPS on {self.port} at {self.baud}...")
        ser = None
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            while self.running:
                try:
                    line = ser.readline().decode('ascii', errors='replace')
                    if line.startswith('$'):
                        try:
                            msg = pynmea2.parse(line)
                            fix = self._parse_nmea(msg)
                            if fix:
                                self._update_fix(fix)
                        except pynmea2.ParseError:
                            continue
                except Exception as e:
                    self.logger.error(f"Serial read error: {e}")
                    time.sleep(1)
        except Exception as e:
            self.logger.error(f"Serial GPS failed: {e}")
        finally:
            if ser:
                ser.close()

    def _parse_nmea(self, msg):
        fix = {}
        if isinstance(msg, pynmea2.types.talker.GGA):
            fix = {
                "lat": msg.latitude,
                "lon": msg.longitude,
                "alt": msg.altitude,
                "fix_quality": int(msg.gps_qual),
                "hdop": float(msg.horizontal_dil),
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
            }
        elif isinstance(msg, pynmea2.types.talker.RMC):
            fix = {
                "lat": msg.latitude,
                "lon": msg.longitude,
                "speed_kmh": float(msg.spd_over_grnd) * 1.852 if msg.spd_over_grnd else 0.0,
                "timestamp": datetime.combine(msg.datestamp, msg.timestamp).isoformat() if msg.datestamp and msg.timestamp else None,
                "fix_quality": 1 if msg.status == 'A' else 0
            }
        if not fix or fix.get('lat') is None:
            return None
        return fix

    def _update_fix(self, fix):
        with self.lock:
            for k, v in fix.items():
                if v is not None:
                    self.latest_fix[k] = v
            # Legacy simple recording for background
            if not self.route_points and self.latest_fix['lat'] is not None:
                 point = (time.time(), self.latest_fix['lat'], self.latest_fix['lon'], 
                         self.latest_fix['alt'], self.latest_fix['speed_kmh'], {}, 0.0)
                 self.route_points.append(point)

    def _get_vibe_score(self, avg_speed, avg_tdi):
        if len(self.route_points) < 5:
            return "Too Short to Vibe"
            
        hard_events = len(self.hard_braking_events) + len(self.hard_accel_events)
        
        if avg_speed > 70 and avg_tdi < 0.05:
            return "Hope You Were Watching For Cameras" if hard_events < 2 else "Arrest Imminent (unless you've got that R100 driver's license)"
        if avg_speed > 50 and avg_tdi < 0.1:
            return "Nice and Easy"
        if avg_tdi > 0.5:
            return "Definitely Uses Phone While Driving" if avg_speed < 15 else "Which Joe Rogan Podcast Are You Listening To?"
        if hard_events > 5:
            return "Whiplash Addict"
        if 20 <= avg_speed <= 50:
            return "\"FUeL iS sO ExPEnSiVe\" ahh driver"
        return "Coffee Cup Holder Enthusiast"

    def generate_summary(self, session_dir):
        if not self.route_points:
            self.logger.warning("No data for summary.")
            return

        # 1. Export GPX
        self.export_gpx(os.path.join(session_dir, "route.gpx"))
        
        # 2. Save Peak Frame
        peak_frame_path = "peak_traffic.jpg"
        if self.peak_traffic_frame is not None:
            cv2.imwrite(os.path.join(session_dir, peak_frame_path), self.peak_traffic_frame)

        # 3. Process Analytics
        valid_points = [p for p in self.route_points if p[1] is not None]
        avg_speed = sum(p[4] for p in valid_points) / len(valid_points)
        avg_tdi = sum(p[6] for p in valid_points) / len(valid_points)
        vibe = self._get_vibe_score(avg_speed, avg_tdi)
        
        stats = {
            "distance_km": round(self.total_distance, 2),
            "max_speed_kmh": round(self.max_speed, 1),
            "avg_speed_kmh": round(avg_speed, 1),
            "idle_min": round(self.idle_time / 60, 1),
            "vibe": vibe,
            "hard_braking": len(self.hard_braking_events),
            "hard_accel": len(self.hard_accel_events),
            "peak_vehicles": self.peak_traffic_count
        }

        # 4. Generate Map Fragment
        map_html = self._render_map_fragment(valid_points)

        # 5. Prepare Chart Data
        chart_data = {
            "labels": [datetime.fromtimestamp(p[0]).strftime("%H:%M:%S") for p in valid_points],
            "speed": [round(p[4], 1) for p in valid_points],
            "vehicles": [sum(p[5].values()) for p in valid_points]
        }

        # 6. Render Dashboard
        self._render_dashboard(session_dir, stats, map_html, chart_data, peak_frame_path)

    def _render_map_fragment(self, points):
        try:
            start_lat, start_lon = points[0][1], points[0][2]
            m = folium.Map(location=[start_lat, start_lon], zoom_start=15, tiles="cartodbdark_matter")
            
            # Speed-Colored Route
            for i in range(len(points)-1):
                p1, p2 = points[i], points[i+1]
                speed = p2[4]
                color = "red" if speed < 20 else "yellow" if speed < 60 else "green"
                folium.PolyLine([(p1[1], p1[2]), (p2[1], p2[2])], color=color, weight=5, opacity=0.8).add_to(m)
            
            # HeatMap Layer
            heat_data = [[p[1], p[2], p[6]] for p in points if p[6] > 0]
            if heat_data:
                plugins.HeatMap(heat_data, name="Traffic Density", min_opacity=0.3).add_to(m)
            
            folium.Marker([points[0][1], points[0][2]], popup="Start", icon=folium.Icon(color='green')).add_to(m)
            folium.Marker([points[-1][1], points[-1][2]], popup="End", icon=folium.Icon(color='red')).add_to(m)
            
            return m._repr_html_()
        except Exception as e:
            self.logger.error(f"Map fragment error: {e}")
            return "<div class='error'>Map generation failed</div>"

    def _render_dashboard(self, session_dir, stats, map_html, chart_data, peak_frame_path):
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Drive Summary - {stats['vibe']}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
        .dashboard {{ max-width: 1200px; margin: 0 auto; display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: #1e1e1e; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .card.full {{ grid-column: 1 / -1; }}
        h1, h2 {{ color: #bb86fc; }}
        .stat-val {{ font-size: 2em; font-weight: bold; color: #03dac6; }}
        .stat-label {{ color: #999; text-transform: uppercase; font-size: 0.8em; }}
        .map-container {{ height: 500px; border-radius: 12px; overflow: hidden; }}
        .peak-img {{ width: 100%; border-radius: 8px; margin-top: 10px; border: 1px solid #333; }}
        .vibe-badge {{ background: #3700b3; color: white; padding: 5px 15px; border-radius: 20px; font-size: 1.2em; }}
    </style>
</head>
<body>
    <div style="text-align: center; margin-bottom: 40px;">
        <h1>Drive Summary: <span class="vibe-badge">{stats['vibe']}</span></h1>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <div class="stat-label">Total Distance</div>
            <div class="stat-val">{stats['distance_km']} km</div>
        </div>
        <div class="card">
            <div class="stat-label">Avg / Max Speed</div>
            <div class="stat-val">{stats['avg_speed_kmh']} / {stats['max_speed_kmh']} km/h</div>
        </div>
        <div class="card">
            <div class="stat-label">Idle Time</div>
            <div class="stat-val">{stats['idle_min']} min</div>
        </div>
        <div class="card">
            <div class="stat-label">Driving Events</div>
            <div style="display: flex; gap: 20px; margin-top: 10px;">
                <div>🛑 {stats['hard_braking']} Hard Brakes</div>
                <div>🚀 {stats['hard_accel']} Rapid Accels</div>
            </div>
        </div>

        <div class="card full">
            <h2>Route & Traffic Density</h2>
            <div class="map-container">{map_html}</div>
        </div>

        <div class="card full">
            <h2>Telemetry Timeline</h2>
            <canvas id="telemetryChart" height="100"></canvas>
        </div>

        <div class="card">
            <h2>Peak Traffic Moment</h2>
            <p>Spotted {stats['peak_vehicles']} vehicles at once.</p>
            <img src="{peak_frame_path}" class="peak-img" alt="Peak Traffic POV">
        </div>
    </div>

    <script>
        const ctx = document.getElementById('telemetryChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(chart_data['labels'])},
                datasets: [
                    {{
                        label: 'Speed (km/h)',
                        data: {json.dumps(chart_data['speed'])},
                        borderColor: '#03dac6',
                        yAxisID: 'y',
                        tension: 0.3
                    }},
                    {{
                        label: 'Vehicles Detected',
                        data: {json.dumps(chart_data['vehicles'])},
                        borderColor: '#cf6679',
                        yAxisID: 'y1',
                        tension: 0.3,
                        backgroundColor: 'rgba(207, 102, 121, 0.2)',
                        fill: true
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ type: 'linear', display: true, position: 'left', grid: {{ color: '#333' }} }},
                    y1: {{ type: 'linear', display: true, position: 'right', grid: {{ drawOnChartArea: false }} }}
                }},
                plugins: {{ legend: {{ labels: {{ color: '#fff' }} }} }}
            }}
        }});
    </script>
</body>
</html>
"""
        with open(os.path.join(session_dir, "drive_summary.html"), "w") as f:
            f.write(html_template)
        self.logger.info(f"Dashboard generated at {os.path.join(session_dir, 'drive_summary.html')}")

    def export_gpx(self, path):
        if not self.route_points: return
        gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)
        for p in self.route_points:
            gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(p[1], p[2], elevation=p[3], time=datetime.fromtimestamp(p[0])))
        with open(path, 'w') as f:
            f.write(gpx.to_xml())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mgr = GPSManager()
    mgr.start()
    time.sleep(5)
    mgr.stop()
