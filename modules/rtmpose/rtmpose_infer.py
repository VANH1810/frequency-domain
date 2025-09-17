import cv2
import numpy as np
import threading
import queue
import time
import tritonclient.grpc as grpcclient
import tritonclient.utils.shared_memory as shm
from tritonclient import utils
from modules.rtmpose.rtm_utils import preprocess_pose_batch_fast, postprocess_pose_batch_fast, draw_keypoints
from modules.common.timer import Timer
from modules.common.logger import get_app_logger
import concurrent

class RTMPoseClient:
    def __init__(
        self,
        camera_name: str = "cam1",
        url: str = "localhost:8001",
        model: str = "rtmpose",
        input_name: str = "input",
        output_x: str = "simcc_x",
        output_y: str = "simcc_y",
        shm_prefix: str = "rtmpose", 
        model_type: str = "normal",
        model_version: str = "3",
        use_threads: bool = False,
        queue_size: int = 5,
        max_batch_size: int = 10,
        num_infer_threads: int = 2
    ):
        self.url = url
        self.model = model
        self.model_version = model_version
        self.model_type = model_type
        self.input_name = input_name
        self.output_x = output_x
        self.output_y = output_y
        self.shm_prefix = shm_prefix
        self.camera_name = camera_name
        self.use_threads = use_threads
        self.queue_size = queue_size
        self.max_batch_size = max_batch_size
        self.num_infer_threads = num_infer_threads

        self.dtype = np.float32
        self.client = grpcclient.InferenceServerClient(url, verbose=False)
        self.logger = get_app_logger(camera_name, __name__)
        self.running = True
        
        self.shm_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.results_by_frame = {}
        
        # Thread pool for parallel batch processing
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_infer_threads)

    def _bootstrap_shapes(self, batch):
        if self.model_type == "mini":
            return (batch, 17, 384), (batch, 17, 512)
        return (batch, 17, 576), (batch, 17, 768)

    def _create_and_register(self, region_name, byte_size):
        handle = shm.create_shared_memory_region(region_name, f"/{region_name}", byte_size)
        self.client.register_system_shared_memory(region_name, f"/{region_name}", byte_size)
        return handle
    
    @Timer("pose rtmpose prerocess step")
    def _preprocess_step(self, frames):
        return preprocess_pose_batch_fast(frames, self.model_type)
    
    @Timer("pose rtmpose infer step")
    def _infer_step(self, inp, batch_id):
        out_shape_x, out_shape_y = self._bootstrap_shapes(inp.shape[0])
        x_bytes = np.prod(out_shape_x) * self.dtype().itemsize
        y_bytes = np.prod(out_shape_y) * self.dtype().itemsize
        in_bytes = inp.nbytes

        in_region = f"{self.shm_prefix}_{self.camera_name}_in_{batch_id}"
        x_region = f"{self.shm_prefix}_{self.camera_name}_x_{batch_id}"
        y_region = f"{self.shm_prefix}_{self.camera_name}_y_{batch_id}"

        #init 
        in_handle = None
        x_handle = None
        y_handle = None
        
        with self.shm_lock:
            try:
                for region in [in_region, x_region, y_region]:
                    try:
                        self.client.unregister_system_shared_memory(region)
                    except:
                        pass

                in_handle = self._create_and_register(in_region, in_bytes)
                x_handle = self._create_and_register(x_region, x_bytes)
                y_handle = self._create_and_register(y_region, y_bytes)

                shm.set_shared_memory_region(in_handle, [inp])

                inputs = [grpcclient.InferInput(self.input_name, inp.shape, "FP32")]
                inputs[0].set_shared_memory(in_region, in_bytes)

                outputs = [
                    grpcclient.InferRequestedOutput(self.output_x),
                    grpcclient.InferRequestedOutput(self.output_y)
                ]
                outputs[0].set_shared_memory(x_region, x_bytes)
                outputs[1].set_shared_memory(y_region, y_bytes)

                self.client.infer(self.model, model_version = self.model_version, inputs=inputs, outputs=outputs)

                simcc_x = shm.get_contents_as_numpy(x_handle, self.dtype, out_shape_x).copy()
                simcc_y = shm.get_contents_as_numpy(y_handle, self.dtype, out_shape_y).copy()

                for region, handle in [(in_region, in_handle), (x_region, x_handle), (y_region, y_handle)]:
                    self.client.unregister_system_shared_memory(region)
                    shm.destroy_shared_memory_region(handle)

                return simcc_x, simcc_y
            except Exception as e:
                self.logger.error(f"Infer error: {e}")
                raise
            
    @Timer("pose rtmpose postprocess step")                
    def _postprocess_step(self, simcc_x, simcc_y, img_shapes, offsets):
        return postprocess_pose_batch_fast(simcc_x, simcc_y, img_shapes, offsets, self.model_type)
    

    def _process_batch(self, batch_id, batch, inp, offsets, is_preprocessed):
        """Process a single batch from preprocessing through postprocessing"""
        try:
            # Preprocess
            img_shapes = [frame.shape[:2] for frame in batch]
            if not is_preprocessed:
                inp, offsets = self._preprocess_step(batch)            
                inp = np.ascontiguousarray(inp).astype(np.float32)
            
            # Inference
            simcc_x, simcc_y = self._infer_step(inp, batch_id)
            
            # Postprocess
            kps_list = self._postprocess_step(simcc_x, simcc_y, img_shapes, offsets)
            
            # Store results
            with self.result_lock:
                self.results_by_frame[batch_id] = kps_list
                
            return kps_list
        except Exception as e:
            self.logger.error(f"Batch processing error for batch {batch_id}: {e}")
            return []

    def estimate_pose(self, frames, inp_p, offsets_p, is_preprocessed, frame_index=0):
        if self.use_threads:
            # Clear previous results
            with self.result_lock:
                self.results_by_frame.clear()
                
            # Split frames into batches
            batches = [frames[i:i + self.max_batch_size] for i in range(0, len(frames), self.max_batch_size)]
            batches_inp = [inp_p[i:i + self.max_batch_size] for i in range(0, len(frames), self.max_batch_size)]
            batch_offsets = [offsets_p[i:i + self.max_batch_size] for i in range(0, len(frames), self.max_batch_size)]
            batch_ids = [frame_index * 1000 + i for i in range(len(batches))]
            
            # Process batches in parallel using thread pool
            futures = []
            for idx, batch, inp, offsets in zip(batch_ids, batches, batches_inp, batch_offsets):
                future = self.thread_pool.submit(self._process_batch, idx, batch, inp, offsets, is_preprocessed)
                futures.append(future)
            
            # Collect results in the original order
            results = []
            for future in concurrent.futures.as_completed(futures):
                batch_result = future.result()
                if batch_result:
                    results.append(batch_result)
            
            # Ensure results are in the correct order based on batch_ids
            ordered_results = []
            for idx in batch_ids:
                with self.result_lock:
                    if idx in self.results_by_frame:
                        ordered_results.append(self.results_by_frame[idx])
            
            # Flatten the list of keypoints
            all_keypoints = []
            for result in ordered_results:
                all_keypoints.extend(result)
                
            return frame_index, all_keypoints
        else:
            # Non-threaded version (original implementation)
            out_shape_x, out_shape_y = self._bootstrap_shapes(len(frames))
            self.x_bytes = np.prod(out_shape_x) * self.dtype().itemsize
            self.y_bytes = np.prod(out_shape_y) * self.dtype().itemsize

            img_shapes = [frame.shape[:2] for frame in frames]
            inp = inp_p
            offsets = offsets_p
            if not is_preprocessed:
                inp, offsets = self._preprocess_step(frames)
                inp = np.ascontiguousarray(inp).astype(np.float32)
            simcc_x, simcc_y = self._infer_step(inp, frame_index)
            kps_list = self._postprocess_step(simcc_x, simcc_y, img_shapes, offsets)
            return frame_index, kps_list

    def close(self):
        if self.use_threads:
            self.running = False
            # Shut down the thread pool
            self.thread_pool.shutdown(wait=False)