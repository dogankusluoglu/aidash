import numpy as np
import logging
from hailo_platform import (HEF, VDevice, InferVStreams, InputVStreamParams,
                            OutputVStreamParams, FormatType, ConfigureParams, HailoStreamInterface)

class HailoInference:
    def __init__(self, model_path, threshold=0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.hef = HEF(self.model_path)
        self.target = None
        self.network_group = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        self.input_vstream_info = None
        self.output_vstream_infos = None
        self._setup_logger()
        self._init_hailo()

    def _setup_logger(self):
        self.logger = logging.getLogger("HailoInference")
        self.logger.setLevel(logging.INFO)

    def _init_hailo(self):
        self.logger.info("Initializing Hailo device...")
        self.target = VDevice()
        
        # Configure network group
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        
        # Prepare streams
        self.input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, format_type=FormatType.FLOAT32)
        
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_infos = self.hef.get_output_vstream_infos()
        
        # Activate network group
        self.network_group_active = self.network_group.activate()
        self.network_group_active.__enter__()
        
        # Pre-create infer pipeline for performance
        self.infer_pipeline = InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params)
        self.infer_pipeline.__enter__()
        
        self.logger.info(f"Loaded model: {self.model_path}")
        self.logger.info(f"Input stream: {self.input_vstream_info.name}, Shape: {self.input_vstream_info.shape}")

    def release(self):
        if hasattr(self, 'infer_pipeline'):
            self.infer_pipeline.__exit__(None, None, None)
        if hasattr(self, 'network_group_active'):
            self.network_group_active.__exit__(None, None, None)
        if self.target:
            self.target.release()

    def run(self, frame):
        """
        Runs inference on a single frame.
        Expects frame to be in RGB format and already resized to input shape.
        """
        # Prepare input data
        input_data = {self.input_vstream_info.name: np.expand_dims(frame, axis=0).astype(np.uint8)}
        
        outputs = self.infer_pipeline.infer(input_data)
        
        return self._postprocess(outputs)

    def _postprocess(self, outputs):
        """
        Post-process YOLOv8 outputs. 
        Recursively converts numpy arrays to lists for JSON serialization.
        """
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
            
        return convert(outputs)

if __name__ == "__main__":
    # Test initialization
    import yaml
    import os
    
    # Ensure we can find config.yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))
    
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    try:
        inf = HailoInference(cfg['inference']['model_path'])
        print("Hailo initialization successful")
    except Exception as e:
        print(f"Hailo initialization failed: {e}")

