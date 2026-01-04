import gi
import logging
import os

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, GstVideo

class DashcamPipeline:
    def __init__(self, config):
        self.config = config
        Gst.init(None)
        self.src_pipeline = None
        self.record_pipeline = None
        self.appsink = None
        self.appsrc = None
        self.bus = None
        self.loop = None
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger("DashcamPipeline")
        self.logger.setLevel(logging.INFO)

    def build_pipeline(self):
        v_width = self.config['video']['width']
        v_height = self.config['video']['height']
        v_fps = self.config['video']['fps']
        v_codec = self.config['video']['codec']
        v_path = self.config['video']['recordings_path']
        v_segment_time = self.config['video']['segment_time']
        
        os.makedirs(v_path, exist_ok=True)
        parser = "h264parse" if "264" in v_codec else "h265parse"

        # 1. Source Pipeline: Camera -> Python (BGR format for OpenCV)
        src_str = (
            f"libcamerasrc name=src ! "
            f"video/x-raw,format=YUY2,width={v_width},height={v_height},framerate={v_fps}/1 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink name=appsink emit-signals=true sync=false"
        )
        
        # 2. Record Pipeline: Python -> File
        record_str = (
            f"appsrc name=appsrc format=time is-live=true do-timestamp=true ! "
            f"video/x-raw,format=BGR,width={v_width},height={v_height},framerate={v_fps}/1 ! "
            f"videoconvert ! {v_codec} ! {parser} ! "
            f"splitmuxsink name=sink location={v_path}/video_%05d.mp4 max-size-time={v_segment_time * 1000000000}"
        )

        self.logger.info(f"Building source pipeline: {src_str}")
        self.src_pipeline = Gst.parse_launch(src_str)
        
        self.logger.info(f"Building record pipeline: {record_str}")
        self.record_pipeline = Gst.parse_launch(record_str)
        
        self.appsink = self.src_pipeline.get_by_name("appsink")
        self.appsrc = self.record_pipeline.get_by_name("appsrc")
        
        # Monitor the recording bus for fragment signals
        record_bus = self.record_pipeline.get_bus()
        record_bus.add_signal_watch()
        record_bus.connect("message", self._on_bus_message)

    def push_recorded_frame(self, frame, pts):
        if not self.appsrc:
            return
            
        data = frame.tobytes()
        buffer = Gst.Buffer.new_allocate(None, len(data), None)
        buffer.fill(0, data)
        
        # Maintain the original PTS
        buffer.pts = pts
        buffer.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, self.config['video']['fps'])
        
        self.appsrc.emit("push-sample", Gst.Sample.new(buffer, self.appsrc.get_property("caps"), None, None))

    def set_appsink_callback(self, callback):
        if self.appsink:
            self.appsink.connect("new-sample", callback)

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ELEMENT:
            s = message.get_structure()
            if s and s.get_name() == "splitmuxsink-fragment-opened":
                location = s.get_string("location")
                self.logger.info(f"New video fragment opened: {location}")
                if hasattr(self, 'new_fragment_callback') and self.new_fragment_callback:
                    self.new_fragment_callback(location)
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.logger.error(f"Record Pipeline Error: {err.message}")

    def start(self):
        self.logger.info("Starting pipelines...")
        self.record_pipeline.set_state(Gst.State.PLAYING)
        self.src_pipeline.set_state(Gst.State.PLAYING)
        self.loop = GObject.MainLoop()
        self.loop.run()

    def stop(self):
        self.logger.info("Stopping pipelines...")
        
        if self.src_pipeline:
            self.src_pipeline.set_state(Gst.State.NULL)
            self.src_pipeline = None

        if self.record_pipeline:
            self.logger.info("Sending EOS to record pipeline...")
            self.record_pipeline.send_event(Gst.Event.new_eos())
            bus = self.record_pipeline.get_bus()
            msg = bus.timed_pop_filtered(Gst.SECOND * 3, Gst.MessageType.EOS | Gst.MessageType.ERROR)
            self.record_pipeline.set_state(Gst.State.NULL)
            self.record_pipeline = None
            
        if self.loop and self.loop.is_running():
            self.loop.quit()
