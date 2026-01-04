import threading
import queue
import time
import json
import logging
import yaml
import cv2
import os
import signal
import numpy as np
from pipeline import DashcamPipeline
from inference import HailoInference
from storage import StorageManager
from power import PowerManager
from gps_manager import GPSManager
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class DashcamApp:
    def __init__(self, config_path=None):
        if config_path is None:
            # Resolve config.yaml relative to this file's location
            # (main.py is in src/, config.yaml is in the parent directory)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config.yaml")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self._setup_logging()
        self.frame_queue = queue.Queue(maxsize=5)
        self.display_queue = queue.Queue(maxsize=2)
        self.metadata_queue = queue.Queue()
        self.running = False
        
        # Components
        self.pipeline = DashcamPipeline(self.config)
        self.inference = None
        try:
            self.inference = HailoInference(self.config['inference']['model_path'])
        except Exception as e:
            self.logger.error(f"Inference initialization failed: {e}. Running without AI.")

        self.power = PowerManager(self.config['power']['ignition_gpio'], self.stop)
        self.gps = GPSManager(self.config.get('gps', {}))
        self.current_video_file = None
        self.metadata_accumulator = []
        self.meta_lock = threading.Lock()

    def _on_new_fragment(self, location):
        with self.meta_lock:
            # If we were already recording, flush the previous segment's metadata
            if self.current_video_file:
                self._flush_to_file(self.current_video_file)
            
            self.current_video_file = location
            self.metadata_accumulator = []
            self.logger.info(f"Switched metadata tracking to: {location}")

    def _flush_to_file(self, video_path):
        json_path = video_path.replace(".mp4", ".json")
        try:
            with open(json_path, "w") as f:
                json.dump(self.metadata_accumulator, f)
            self.logger.info(f"Saved metadata to {json_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metadata to {json_path}: {e}")

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/dashcam.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("DashcamApp")

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR
        
        if self.frame_queue.full():
            # Drop frame if inference is slow
            return Gst.FlowReturn.OK
            
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        # Extract frame data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            # We assume RGB from the pipeline videoconvert
            # Need to get dimensions from caps
            struct = caps.get_structure(0)
            width = struct.get_int("width")[1]
            height = struct.get_int("height")[1]
            
            frame = np.ndarray(
                (height, width, 3),
                buffer=map_info.data,
                dtype=np.uint8
            ).copy() # Copy to avoid buffer issues
            
            pts = buffer.pts
            self.frame_queue.put((frame, pts))
            buffer.unmap(map_info)
            
        return Gst.FlowReturn.OK

    def inference_worker(self):
        self.logger.info("Starting inference worker...")
        frame_count = 0
        start_time = time.time()
        while self.running:
            try:
                frame, pts = self.frame_queue.get(timeout=1)
                
                # Pre-processing: Resize to 640x640 for Hailo
                input_w = self.config['inference']['input_width']
                input_h = self.config['inference']['input_height']
                resized_frame = cv2.resize(frame, (input_w, input_h))
                
                # Hailo models expect RGB, but our pipeline is now BGR for preview/OpenCV
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                # Run inference
                results = None
                if self.inference:
                    results = self.inference.run(frame_rgb)
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        self.logger.info(f"Inference running at {fps:.2f} FPS")

                    # Draw detections on the frame before recording/display
                    if results:
                        nms_key = next((k for k in results.keys() if 'nms' in k.lower()), None)
                        if nms_key:
                            detections = results[nms_key]
                            if len(detections) > 0:
                                for cls_id, class_detections in enumerate(detections[0]):
                                    for det in class_detections:
                                        if len(det) < 5: continue
                                        y1, x1, y2, x2, score = det[:5]
                                        if score < self.config['inference']['threshold']:
                                            continue
                                        
                                        h, w = frame.shape[:2]
                                        ix1, iy1 = int(x1 * w), int(y1 * h)
                                        ix2, iy2 = int(x2 * w), int(y2 * h)
                                        
                                        label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else f"ID {cls_id}"
                                        color = (0, 255, 0)
                                        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
                                        cv2.putText(frame, f"{label} {score:.2f}", (ix1, iy1 - 10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Overlay GPS data if enabled
                    gps_fix = self.gps.get_latest_fix()
                    if self.config.get('gps', {}).get('overlay_enabled', True):
                        if gps_fix['lat'] is not None:
                            gps_text = f"Lat: {gps_fix['lat']:.5f} Lon: {gps_fix['lon']:.5f} Spd: {gps_fix['speed_kmh']:.1f} km/h"
                            h, w = frame.shape[:2]
                            cv2.putText(frame, gps_text, (10, h - 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Store metadata with timestamp
                    metadata = {
                        "pts": pts,
                        "timestamp": time.time(),
                        "objects": results, # List of detected objects
                        "gps": gps_fix
                    }
                    with self.meta_lock:
                        self.metadata_accumulator.append(metadata)

                    # Correlate GPS with AI detections for analytics
                    self.gps.record_frame_stats(frame, results, pts)

                # Push the frame with detections to the recording pipeline
                self.pipeline.push_recorded_frame(frame, pts)

                # Push to display queue with detections
                if not self.display_queue.full():
                    self.display_queue.put((frame.copy(), results))
                
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in inference worker: {e}")

    def _update_display(self):
        if not self.running:
            cv2.destroyAllWindows()
            return False
            
        try:
            # Get multiple frames if they are backed up, but only show the last one
            frame_data = None
            while True:
                try:
                    frame_data = self.display_queue.get_nowait()
                except queue.Empty:
                    break
            
            if frame_data is None:
                return True
                
            frame, results = frame_data
            cv2.imshow("Hailo Dashcam Preview", frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.logger.error(f"Error in display update: {e}")
            
        return True

    def storage_worker(self):
        self.logger.info("Starting storage cleanup worker...")
        while self.running:
            try:
                self.storage.clean_up()
                # Also save metadata from queue to files
                self._flush_metadata()
                time.sleep(10) # Run every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in storage worker: {e}")

    def _flush_metadata(self):
        # In a real dashcam, we'd associate this with the current video segment.
        # For simplicity, we'll write to a "current_metadata.json" or similar.
        # A more robust way is to listen for splitmuxsink signals for new files.
        pass

    def start(self):
        self.running = True
        self.logger.info("Dashcam starting...")
        
        # Create a unique session directory to avoid overwriting
        session_id = time.strftime("%Y%m%d_%H%M%S")
        self.config['video']['recordings_path'] = os.path.join(
            self.config['video'].get('recordings_path', 'recordings'),
            session_id
        )
        os.makedirs(self.config['video']['recordings_path'], exist_ok=True)
        self.logger.info(f"Recordings will be saved to: {self.config['video']['recordings_path']}")

        self.storage = StorageManager(
            self.config['video']['recordings_path'], 
            self.config['storage']['max_usage_percent']
        )

        # Start workers
        self.inf_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inf_thread.start()
        
        self.storage_thread = threading.Thread(target=self.storage_worker, daemon=True)
        self.storage_thread.start()
        
        # UI update in GLib main loop
        cv2.namedWindow("Hailo Dashcam Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hailo Dashcam Preview", 1280, 720)
        GLib.timeout_add(30, self._update_display)
        
        self.power.start_monitoring()
        self.gps.start()
        
        # Build and start pipeline
        self.pipeline.build_pipeline()
        self.pipeline.set_appsink_callback(self._on_new_sample)
        self.pipeline.new_fragment_callback = self._on_new_fragment
        
        try:
            self.pipeline.start()
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
        finally:
            self.stop()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.logger.info("Dashcam stopping...")
        
        # Final flush of metadata
        if self.current_video_file:
            self._flush_to_file(self.current_video_file)
            
        self.gps.stop()
        # Export session route data and advanced dashboard
        session_dir = self.config['video']['recordings_path']
        self.gps.generate_summary(session_dir)

        self.pipeline.stop()
        if self.inference:
            self.inference.release()
        self.logger.info("Dashcam stopped.")

if __name__ == "__main__":
    # Ensure we are running from the project root so relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    app = DashcamApp()
    
    # Global signal handler
    def signal_handler(sig, frame):
        print("\nStopping dashcam...")
        app.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app.start()

