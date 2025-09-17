import tritonclient.grpc as grpcclient
import json
import numpy as np
import time
from modules.common.logger import get_app_logger
import tritonclient.utils.shared_memory as shm
from tritonclient import utils

class GCNClient:
    def __init__(self, camera_name="cam1" , 
                 model_name="action_reg", 
                 server_infer_port=8001,
                 shm_prefix = "gcndet",
                 num_cls=6,
                 verbose=False):
        """
        Initialize GCNClient to communicate with Triton inference server.
        
        :param server_url: Triton server URL.
        :param model_name: Name of the model deployed on Triton.
        :param config_path: Path to the JSON config file for input formatting.
        """
        self.camera_name = camera_name
        self.logger = get_app_logger(camera_name, __name__)
        self.model_name = model_name
        self.gcn18_req = {"pose_info": {
                                        "all_poses": None, 
                                        "image_height": 360,
                                        "image_width": 640,
                                        "joint_num": 14,
                                        # "output_dim": 24, #4: GCN 18 , 24: GCN 14
                                        "output_dim": num_cls,
                                        "sequence_number_pose": 20
                                        }
                         }
        
        self.input_name = "input_data"
        self.output_name = "gcn_postprocess_output"
        self.in_region = f"{shm_prefix}_{camera_name}_in"
        self.out_region = f"{shm_prefix}_{camera_name}_out"
        
         # Initialize client based on the mode
        self.client = grpcclient.InferenceServerClient(url=f"localhost:{server_infer_port}", verbose=verbose)
        
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

    def infer(self, frm_idx, flatten_kps, height, width):
        """
        Perform inference on the provided flatten keypoints.
        
        :param flatten_kps: A numpy array (K, 1) containing keypoints.
        :return: Inference result from Triton.
        """
        self.logger.info(f"GCN infer frame {frm_idx}")
        if len(flatten_kps.shape) != 2 or flatten_kps.shape[1] != 1:
            raise ValueError(f"Expected shape (K,1), but got {flatten_kps.shape}")

        # Update input data in the request template
        self.gcn18_req["pose_info"]["all_poses"] = flatten_kps.tolist()
        self.gcn18_req["pose_info"]["image_height"] = height
        self.gcn18_req["pose_info"]["image_width"] = width
        input_string = json.dumps(self.gcn18_req).encode("utf-8")
        
        # Convert bytes to numpy array and put it in a list
        input_np = np.array([input_string], dtype=np.object_)
        
        input_serialized = utils.serialize_byte_tensor(input_np)
        in_bytes = utils.serialized_byte_size(input_serialized)
        out_bytes = in_bytes * 2
        in_handle = None
        out_handle = None
        
        try:            
            try:
                for region in [self.in_region, self.out_region]:
                    self.client.unregister_system_shared_memory(region)
            except:
                pass
            
            # Create input shared memory
            in_handle = self._create_and_register(self.in_region, in_bytes)
            shm.set_shared_memory_region(in_handle, [input_serialized])
            
            out_handle = self._create_and_register(self.out_region, out_bytes)
            
            inputs = [grpcclient.InferInput(self.input_name, [1], "BYTES")]
            inputs[0].set_shared_memory(self.in_region, in_bytes)
            
            outputs = [grpcclient.InferRequestedOutput(self.output_name)]
            outputs[0].set_shared_memory(self.out_region, out_bytes)
                

            results = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
            
            output1 = results.get_output(self.output_name)
            if output1 is None:
                raise RuntimeError("Inference output is missing")
                
            # grpc mode
            output1_data = shm.get_contents_as_numpy(
                out_handle,
                utils.triton_to_np_dtype(output1.datatype),
                output1.shape,
            )
            if output1.datatype == "BYTES":                        
                output1_data = np.array([x.decode("utf-8") for x in output1_data.flatten()]) 
                
            final_output = json.loads(output1_data[0])
            return final_output

        except Exception as e:
            raise RuntimeError(f"Inference error: {e}")
        
        finally:
            if in_handle:
                self.client.unregister_system_shared_memory(self.in_region)
                shm.destroy_shared_memory_region(in_handle)
            if out_handle:
                self.client.unregister_system_shared_memory(self.out_region)
                shm.destroy_shared_memory_region(out_handle)
                
            self.logger.debug(f"GCN shared memory released frame {frm_idx}.")
                
    def __del__(self):
        """Destructor to ensure resource cleanup when the object is deleted"""
        try:
            self.client.unregister_system_shared_memory(self.in_region)
            self.client.unregister_cuda_shared_memory(self.out_region)
        except:
            pass