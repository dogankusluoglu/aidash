# Hailo 8 Dashcam Project (aidash)

![Highly precise COCO object detection](https://github.com/dogankusluoglu/aidash/blob/main/hq720.jpg?raw=true)

## Overview
This is an AI-powered dashcam application designed for the **Raspberry Pi 5** using the **Hailo-8** AI accelerator. It captures high-definition video from a Raspberry Pi camera, runs real-time object detection (YOLO), and saves the video with AI bounding boxes permanently "burned" into the recording.

---

## Quick Start

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/aidash.git
    cd aidash
    ```
2. **Create directories**: In the project root, run:
    ```bash
    mkdir models recordings logs
    ```
2.  **Install Dependencies** (see **Prerequisites** below): Ensure you have the HailoRT (included in the `hailo-all` package as of Raspbian Trixie) and GStreamer development libraries installed on your RPi 5. Ensure you add the relevant model in the `models/` directory and update `config.yaml` accordingly.  
3.  **Run**:
    ```bash
    cd src
    python3 main.py
    ```

## Prerequisites

Before running this project, you need to set up your Raspberry Pi 5 with the Hailo-8 AI Kit and install the necessary software.

### 1. Hardware Setup
- **Raspberry Pi 5** (8GB recommended).
- **Hailo-8** or **Hailo-8L** AI Kit (8L untested, however it should work well with YOLOv8n).
- **Raspberry Pi Camera Module 3** (or any `libcamerasrc` compatible camera).
- **USB GPS Module** (e.g., U-blox 7) connected to `/dev/ttyACM0` or handled by `gpsd`.

### 2. Enable PCIe Gen 3
The Raspberry Pi 5 PCIe interface is disabled by default and runs at Gen 2 speeds. To enable Gen 3 for maximum performance:

1. Edit the boot configuration:
   ```bash
   sudo nano /boot/firmware/config.txt
   ```
2. Add the following lines at the end:
   ```ini
   # Enable PCIe 1x
   dtparam=pciex1
   # Force PCIe Gen 3 speeds
   dtparam=pciex1_gen=3
   ```
3. Reboot your Pi:
   ```bash
   sudo reboot
   ```

### 3. Install Hailo Software Suite
Install the `hailo-all` package which includes the HailoRT, GStreamer plugins, and firmware:
```bash
sudo apt update
sudo apt install hailo-all
sudo reboot
```
Verify the installation by checking if the Hailo device is detected:
```bash
hailortcli scan
```

### 4. System Dependencies
Install GStreamer, Python bindings, and GPS tools:
```bash
sudo apt install -y \
    python3-gi python3-gi-cairo gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 gstreamer1.0-plugins-bad \
    gstreamer1.0-libcamera \
    gpsd gpsd-clients python3-gps
```

### 5. Python Libraries
Install the required Python packages:
```bash
pip install opencv-python pyyaml numpy pyserial pynmea2 gpxpy folium
```

### 6.1. Retrained AI Models
I've included the models I've trained on different datasets in `retrained_models/`:
 - `yolov11s_bdd100k.hef` - Poor retraining attempt of YOLOv11s using [this dataset](https://www.kaggle.com/datasets/a7madmostafa/bdd100k-yolo/data). Hailo AI Software Suite is only available for Linux, otherwise the Docker container must be used via WSL. It being 2am, I didn't bother struggling to get my Nvidia GPU drivers working in a Docker container nested within WSL. Falling back on the CPU results in a optimization level of 0 with only 64 images, causing lower accuracy and framerate (~17 FPS). Just keeping it here for science, I guess.

### 6.2. AI Models
The rest of the `.hef` model files are large and excluded from the repository. You must download them manually or use the provided script for the voice model.

#### Manual Download (Recommended)
1.  **Create the models directory**:
    ```bash
    mkdir -p models
    ```
2.  **Download .hef files**: Visit the [Hailo Model Explorer](https://hailo.ai/products/hailo-software/model-explorer-vision/) and download the pre-compiled models for the Hailo-8.
    - Recommended for RPi 5: **YOLOv8n** (High FPS) or **YOLOv11s** (Higher Accuracy).
3.  **Place the files**: Move the downloaded `.hef` files into the `models/` folder. Ensure the filenames match your `config.yaml` (e.g., `models/yolov11s.hef`).

#### Voice Recognition Model (Optional) (TODO)
If you plan to use the voice assistant features in the future or want to get it running on your own, run the download script:
```bash
bash scripts/download_models.sh
```


## Configuration

Tweak `config.yaml` to adjust resolution, FPS, detection thresholds, storage limits, and model:

```yaml
video:
  width: 1920
  height: 1080
  fps: 30
inference:
  model_path: "models/yolov11s.hef"
  threshold: 0.2
storage:
  max_usage_percent: 90
```

---

## Tech Stack
- **Hardware**: Raspberry Pi 5 8GB, Hailo-8 AI Kit (26 TOPS), Raspberry Pi Camera Module 3 (Wide-Angle 120 degree lens), GPS/GLONASS U-blox7 USB GPS module.
- **OS**: Raspberry Pi OS Trixie.
- **Language**: Python 3.13+.
- **Video Framework**: GStreamer 1.0 (using `libcamerasrc` and `x264enc`).
- **AI Inference**: HailoRT (`hailo_platform`), YOLOv8n/YOLOv11s/Retrained YOLOv11 with BDD100k (in progress) (.hef format).
- **Computer Vision**: OpenCV (for drawing overlays and preview).

## Key Features
- **Real-time Detection**: Runs YOLOv8n at ~30 FPS on the Hailo-8 and YOLOv11s at ~20 FPS.
- **Integrated Recording**: Detections are drawn on the frames in Python and pushed back into GStreamer for encoding, so overlays are saved in the `.mp4`.
- **Real-time GPS Telemetry**: Overlays live Latitude, Longitude, and Speed (km/h) onto the video recording and preview.
- **Advanced Post-Drive Analytics**: 
    - **Interactive Dashboard**: Generates a standalone `drive_summary.html` with speed-colored route maps, traffic heatmaps, and telemetry charts (Chart.js).
    - **Trip Stats**: Calculates total distance, idle time, and max/avg speeds.
    - **Driving Behavior**: Detects hard braking and rapid acceleration events.
    - **Traffic Density**: Correlates AI vehicle counts with GPS speed to index traffic congestion.
    - **Drive "Vibe" Score**: Categorizes your trip (e.g., "Arrest Imminent", "Coffee Cup Holder Enthusiast"). These values need to be adjusted.
- **Dual-Pipeline Architecture**:
    - **Source Pipeline**: Camera -> GStreamer -> Appsink -> Python.
    - **Record Pipeline**: Python -> Appsrc -> GStreamer -> MP4 File.
- **Unique Session Management**: Every recording session is saved in a unique timestamped folder (`recordings/YYYYMMDD_HHMMSS/`) to prevent overwriting.
- **Circular Buffer**: Automatically deletes oldest recordings when disk usage exceeds a threshold (e.g., 90%).
- **Graceful Shutdown**: Handles `Ctrl+C` and window close signals to ensure GStreamer finishes writing the MP4 header, preventing file corruption.
- **Metadata Logging**: Saves raw detection results in a companion `.json` file for every video segment.

## Project Structure
- `src/main.py`: The entry point. Manages threads for inference, storage cleanup, and the GLib main loop for GUI/GStreamer.
- `src/pipeline.py`: Defines the dual GStreamer pipelines and the frame "bridging" logic.
- `src/inference.py`: Wraps the HailoRT API for model loading and post-processing.
- `src/storage.py`: Manages disk space and circular recording logic.
- `src/gps_manager.py`: Handles GPS ingestion (gpsd/serial), analytics calculations, and HTML dashboard generation.
- `src/power.py`: Monitors GPIO for ignition-based auto-shutdown.
- `config.yaml`: Central configuration for resolution, FPS, paths, and detection thresholds.

## Current State
- The system is optimized for RPi 5 (using software `x264enc` as RPi 5 lacks a hardware H.265 encoder).
- Native format is `YUY2` from the camera, converted to `BGR` for OpenCV compatibility.
- Inference happens on an `RGB` conversion of the resized frame.

## To-do

- [x] Integrated GPS tracking and real-time HUD.
- [x] Post-drive analytics dashboard and interactive route mapping.
- [ ] Retrained YOLOv11 with BDD100k dataset - **in progress** - YOLOv8n is pretty average, v11n is definitely good enough, but it can't hurt to try be better
- [ ] Live AI voice assistant - integrated with the real time video, not sure where to start for that yet. I'd likely never even use this feature, my subwoofer kills all frequencies above 80Hz at any volume. Whisper model will be raving.
- [ ] Number plate recognition - I'm being real ambitious with how much computation power I have. 
- [ ] Car make, model and year range detection - taking dataset suggestions


