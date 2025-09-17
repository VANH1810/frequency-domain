import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.utils.shared_memory as shm
from tritonclient import utils
import queue
from modules.rtmpose.rtm_utils import preprocess_det_fast, postprocess_det_fast, draw_boxes, crop_boxes_fast
import threading
import time 
from modules.common.timer import Timer
from modules.common.logger import get_app_logger

DTYPE = np.float32

class RTMDetClient:
    """Triton gRPC client (System-SHM) cho mô hình RTMDet với hỗ trợ Multi-threading."""

    def __init__(
        self,
        camera_name: str="cam1",
        url: str = "localhost:8001",
        model: str = "rtmdet",
        input_name: str = "image",
        output_name: str = "output",
        shm_prefix: str = "rtmdet",
        model_version: str = "2",
        queue_size: int = 5
    ):
        self.url = url
        self.model = model
        self.input_name = input_name
        self.output_name = output_name
        self.in_region = f"{shm_prefix}_{camera_name}_in"
        self.out_region = f"{shm_prefix}_{camera_name}_out"
        self.queue_size = queue_size
        self.running = True
        self.model_version = model_version
        self.logger = get_app_logger(camera_name, __name__)
        
        
        self.shm_lock = threading.Lock()
        self.results_by_frame = {}  
        self.conf_thresh = 0.5
        self.iou_thresh = 0.45
        self.client = grpcclient.InferenceServerClient(url, verbose=False)
        
        
        self.out_shape = (1, 8400, 6)
        self.out_bytes = np.prod(self.out_shape) * DTYPE().itemsize

        self._cleanup_shared_memory()
        
            
    def _cleanup_shared_memory(self):
        """Clean up existing shared memory regions"""
        try:
            self.client.unregister_system_shared_memory()
            self.client.unregister_cuda_shared_memory()
        except Exception as e:
            print(f"Warning when removing old shared memory: {e}")

    def _create_and_register(self, region_name, byte_size):
        handle = shm.create_shared_memory_region(region_name, f"/{region_name}", byte_size)
        self.client.register_system_shared_memory(region_name, f"/{region_name}", byte_size)
        return handle
    
    @Timer("pose rtmdet preprocess")
    def _preprocess_step(self, img): 
        return preprocess_det_fast(img)
    
    @Timer("pose rtmdet inferece")
    def _infer_step(self, inp): 
        with self.shm_lock:
            in_bytes = inp.nbytes
            try:
                for region in [self.in_region, self.out_region]:
                    self.client.unregister_system_shared_memory(region)
            except:
                pass
            
            in_handle = self._create_and_register(self.in_region, in_bytes)
            out_handle = self._create_and_register(self.out_region, self.out_bytes)
            
            shm.set_shared_memory_region(in_handle, [inp])
            
            inputs = [grpcclient.InferInput(self.input_name, inp.shape, "FP32")]
            inputs[0].set_shared_memory(self.in_region, in_bytes)
            outputs = [grpcclient.InferRequestedOutput(self.output_name)]
            outputs[0].set_shared_memory(self.out_region, self.out_bytes)
            
            self.client.infer(self.model, model_version = self.model_version,  inputs=inputs, outputs=outputs)
            det = shm.get_contents_as_numpy(out_handle, DTYPE, self.out_shape).copy()            
            
            for region, handle in [(self.in_region, in_handle), (self.out_region, out_handle)]:                    
                self.client.unregister_system_shared_memory(region)
                shm.destroy_shared_memory_region(handle)
            
            return det
    
    @Timer("pose rtmdet postprocess")
    def _postprocess_step(self, det, img_w, img_h, off_x, off_y):         
        return postprocess_det_fast(det, img_w, img_h, (off_x, off_y), conf_thre=self.conf_thresh, iou_thre=self.iou_thresh)
    
                
    @Timer("pose rtmdet detect total time")
    def detect(self, img: np.ndarray, frame_index: int, conf_thresh: float, iou_thresh: float):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        inp, (off_x, off_y) = self._preprocess_step(img)  
        inp = np.ascontiguousarray(inp).astype(np.float32)
        det = self._infer_step(inp)
        img_h, img_w = img.shape[:2]
        boxes = self._postprocess_step(det, img_w, img_h, off_x, off_y)
        return frame_index, boxes

    def __del__(self):
        try:
            self.client.unregister_system_shared_memory(self.in_region)  
            self.client.unregister_system_shared_memory(self.out_region)
        except Exception as e:
            self.logger.error(f"Error cleaning up SHM: {e}")


if __name__ == "__main__":
    client = RTMDetClient(url="localhost:8001", use_threads=True)
    frame_index = 0      
    img = cv2.imread("/workspace/images_test/frame_12.jpg")
    frame_index, boxes = client.detect(img, frame_index)
    print(boxes)
    client.close()
