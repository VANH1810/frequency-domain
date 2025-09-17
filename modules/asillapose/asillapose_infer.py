import os
import cv2
import torch
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.utils.shared_memory as shm
from tritonclient import utils
import threading
import queue
from modules.asillapose.post_processor import resize_letter_box, transform_image, postprocess, get_result
from modules.common.logger import get_app_logger
from modules.common.config import get_app_config

class AsillaPoseClient:
    def __init__(self, camera_name="cam1", model_name="pose_origin_fp16_new", 
                 server_infer_port=8001, infer_width = 1472, infer_height=896,
                 verbose=False, max_queue_size=10, num_workers=2):
        """
        Initialize the AsillaPoseClient with real-time processing capabilities
        
        Args:
            camera_name (str): Name of the camera
            model_name (str): Model name on Triton server
            verbose (bool): Enable/disable detailed logging
            max_queue_size (int): Maximum size of input and output queues
            num_workers (int): Number of worker threads for parallel processing
        """
        self.camera_name = camera_name
        self.logger = get_app_logger(camera_name, __name__)
        self.config_reader = get_app_config(camera_name)
        self.model_name = model_name
        self.verbose = verbose
        
        # Initialize Triton client
        self.triton_client = grpcclient.InferenceServerClient(url=f"localhost:{server_infer_port}", verbose=verbose)
        
        # Cleanup existing shared memory
        self._cleanup_shared_memory()
        
        # Input size for the model
        self.w_infer = infer_width
        self.h_infer = infer_height
        self.input_size = (self.h_infer, self.w_infer)
        
        # Queue for input and output
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        
        # Threading for parallel processing
        self.num_workers = num_workers
        self.workers = []
        self.stop_event = threading.Event()
        
        # Start worker threads
        self._start_workers()
        
        
    
    def _cleanup_shared_memory(self):
        """Clean up existing shared memory regions"""
        try:
            self.triton_client.unregister_system_shared_memory()
            self.triton_client.unregister_cuda_shared_memory()
        except Exception as e:
            self.logger.warning(f"Warning when removing old shared memory: {e}")
    
    def _start_workers(self):
        """Start worker threads for parallel inference"""
        for idx in range(self.num_workers):
            worker = threading.Thread(target=self._worker_process, args=(idx,), daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker_process(self, idx_worker):
        """Worker thread to process inference tasks"""
        while not self.stop_event.is_set():
            try:
                # Get batch of images with frame indices
                batch_data = self.input_queue.get(timeout=1)
                frm_idx, images = batch_data
                
                # Perform inference
                try:
                    outputs = self.infer(frm_idx, images, idx_worker)
                    self.output_queue.put((frm_idx, outputs, images))
                except Exception as e:
                    self.logger.error(f"Inference error: {e}")
                
                self.input_queue.task_done()
            except queue.Empty:
                continue
    
    def preprocess(self, images):
        """
        Preprocess input images with error handling and logging
        
        Args:
            images (list): List of images to process
            
        Returns:
            tuple: (input_tensors, scales, offsets)
        """
        if not images:
            self.logger.warning("No input images provided")
            return [], [], []
            
        input_tensors = []
        scales = []
        offsets = []
        
        padding_color=self.config_reader.getint("asillapose", "padding_color", fallback=0)
        padding_type=self.config_reader.get("asillapose", "padding_type", fallback="center")
        interpolation=cv2.INTER_LINEAR
        
        for image in images:
            try:
                input_tensor, scale, (top, _, left, _) = resize_letter_box(image, 
                                                                            self.input_size, 
                                                                            padding_color=padding_color, 
                                                                            padding_type=padding_type, 
                                                                            interpolation=interpolation)
                input_tensor = transform_image(input_tensor)
                input_tensors.append(input_tensor)
                scales.append(scale)
                offsets.append((top, left))
            except Exception as e:
                self.logger.error(f"Preprocessing error for image: {e}")
        
        return input_tensors, scales, offsets

    def infer(self, frm_idx, images, idx_worker):
        """
        Perform inference on a batch of images with shared memory optimization
        
        Args:
            frm_idx (int): Frame index for logging
            images (list): List of images for pose detection
            
        Returns:
            tensor: Processed inference results
        """
        self.logger.info(f"Pose infer frame {frm_idx}")
        
        if not images:
            return []
        
            
        batch_size = len(images)
        input_tensors, self.scales, self.offsets = self.preprocess(images)
        
        if not input_tensors:
            return []
        
        input0_data = np.stack(input_tensors, 0)
        input_byte_size = input0_data.size * input0_data.itemsize
        output_byte_size = input_byte_size * 2
        shm_ip_handle = None
        shm_op_handle = None
        
        try:
            # Create input shared memory
            shm_ip_handle = shm.create_shared_memory_region(f"input_{self.camera_name}_{idx_worker}_pose_data", 
                                                            f"input_{self.camera_name}_{idx_worker}_pose", 
                                                            input_byte_size)
            shm.set_shared_memory_region(shm_ip_handle, [input0_data])
            self.triton_client.register_system_shared_memory(
                f"input_{self.camera_name}_{idx_worker}_pose_data", 
                f"input_{self.camera_name}_{idx_worker}_pose", 
                input_byte_size
            )
            
            # Create output shared memory
            shm_op_handle = shm.create_shared_memory_region(
                f"output_{self.camera_name}_{idx_worker}_pose_data", 
                f"output_{self.camera_name}_{idx_worker}_pose", 
                output_byte_size
            )
            self.triton_client.register_system_shared_memory(
                f"output_{self.camera_name}_{idx_worker}_pose_data", 
                f"output_{self.camera_name}_{idx_worker}_pose", 
                output_byte_size
            )
            
            # Prepare input and output based on mode
            # GRPC mode
            inputs = [grpcclient.InferInput("images", [batch_size, 3, self.h_infer, self.w_infer], "FP32")]
            inputs[0].set_shared_memory(f"input_{self.camera_name}_{idx_worker}_pose_data", input_byte_size)
            outputs = [grpcclient.InferRequestedOutput("outputs")]
            outputs[0].set_shared_memory(f"output_{self.camera_name}_{idx_worker}_pose_data", output_byte_size)
            
            # Perform inference
            results = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
            
            # Process results with safer attribute access
            # grpc mode
            output1 = results.get_output("outputs")
            output_datatype = output1.datatype
            output_shape = output1.shape
            
            if output1 is None:
                raise RuntimeError("Inference output is missing")
            
            # Get output data 
            output1_data = shm.get_contents_as_numpy(
                shm_op_handle,
                utils.triton_to_np_dtype(output_datatype),
                output_shape,
            )
            
            # Convert and process final results
            outputs_final = torch.from_numpy(output1_data)
            
            conf_thres=self.config_reader.getfloat("asillapose", "conf_thres", fallback=0.01)
            oks_thres=self.config_reader.getfloat("asillapose", "oks_thres", fallback=0.85)
            max_det=self.config_reader.getint("asillapose", "max_det", fallback=1280)
            
            outputs_final = self.process_no_e2e(outputs_final, conf_thres, oks_thres, max_det)
            
            return outputs_final
            
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()  # Add this to get full stack trace
            return []
        
        finally:
            # Clean up shared memory
            self._cleanup_shared_memory_handles(shm_ip_handle, shm_op_handle, frm_idx, idx_worker)
    
    def _cleanup_shared_memory_handles(self, shm_ip_handle, shm_op_handle, frm_idx, idx_worker):
        """
        Clean up shared memory handles
        
        Args:
            shm_ip_handle: Input shared memory handle
            shm_op_handle: Output shared memory handle
            frm_idx (int): Frame index for logging
        """
        try:
            if shm_ip_handle:
                self.triton_client.unregister_system_shared_memory(f"input_{self.camera_name}_{idx_worker}_pose_data")
                shm.destroy_shared_memory_region(shm_ip_handle)
            if shm_op_handle:
                self.triton_client.unregister_system_shared_memory(f"output_{self.camera_name}_{idx_worker}_pose_data")
                shm.destroy_shared_memory_region(shm_op_handle)
                
            self.logger.debug(f"Pose shared memory released frame {frm_idx}.")
        except Exception as e:
            self.logger.error(f"Error cleaning up shared memory: {e}")
    
    def submit_batch(self, frm_idx, images):
        """
        Submit a batch of images for processing
        
        Args:
            frm_idx (int): Frame index
            images (list): List of images to process
        """
        try:
            self.input_queue.put((frm_idx, images), timeout=0.1)
        except queue.Full:
            self.logger.warning("Input queue full, skipping batch")
    
    def get_results(self, timeout=0.2):
        """
        Get processed results from the output queue
        
        Args:
            timeout (float): Maximum wait time for results
            
        Returns:
            tuple or None: (frame_index, outputs, images) or None if no results
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop worker threads and clean up resources"""
        self.stop_event.set()
        for worker in self.workers:
            worker.join()
        self._cleanup_shared_memory()
    
    
    @staticmethod
    def process_no_e2e(outputs, conf_thres=0.01, oks_thres=0.85, max_det=1280):
        """
        Process model output
        
        Args:
            outputs (tensor): Output tensor from the model
            
        Returns:
            tensor: Processed results
        """
        
        return postprocess(
            predictions=outputs,
            conf_thres=conf_thres,
            oks_thres=oks_thres,
            nc=1,
            max_det=max_det,
            kpt_sigmas=None,
        )
        
    def get_box_kps(self, outputs, images):
        _objects = []
        for i, image in enumerate(images):
            _, scale, (top, _, left, _) = resize_letter_box(image, self.input_size)
            boxes, scores, _, kps = get_result(outputs[i], scale, top, left)
            _objects.append((boxes, scores, kps))
        return _objects

