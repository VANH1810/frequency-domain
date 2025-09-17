import os
import json
import onnxruntime
import numpy as np
import pandas as pd
from tqdm import tqdm

T = 20
WINDOW_SIZE = 20
STRIDE      = 3
connect_joint = np.array([1,1,1,2,3,1,5,6,1,8,9,1,11,12])
inputs = 'JVB'

def norm_data2(personList, imagesize=np.array([1920, 1080])):
    data = personList.copy().astype(float)
    idxs = np.arange(personList.shape[0])
    for i, poselist in enumerate(personList):
        idx = poselist[:,:,0] > 0
        pose = poselist[idx]
        #1. Scale
        w=(np.max(pose[:,0])-np.min(pose[:,0]))
        h=(np.max(pose[:,1])-np.min(pose[:,1]))
        rates = imagesize / np.array([w, h])
        rate = min(rates)
        pose = pose*rate

        #2. Move to center
        #2.1 Get center of person
        xg=(np.max(pose[:,0])+np.min(pose[:,0]))/2
        yg=(np.max(pose[:,1])+np.min(pose[:,1]))/2
        center = np.array([xg, yg])
        #2.2 Get distance bitween img center and person center
        img_size = imagesize/2
        distance = center - img_size
        #2.3 move center to cordinate root, scale to 0~1
        pose_normalize = np.zeros(poselist.shape, dtype=np.float32)
        pose_normalize[idx, 0]= (pose[:, 0] - distance[0])/imagesize[0]
        pose_normalize[idx, 1]= (pose[:, 1] - distance[1])/imagesize[1]

        data[i] = pose_normalize

    return data, idxs

def processing_data(data_process):
    data = np.array(data_process)
    # (C, max_frame, V, M) -> (I, C*2, T, V, M)
    data = data.transpose(3,1,2,0)
    joint, velocity, bone = multi_input(data[:,:T,:,:])
    data_new = []
    if 'J' in inputs:
        data_new.append(joint)
    if 'V' in inputs:
        data_new.append(velocity)
    if 'B' in inputs:
        data_new.append(bone)
    data_new = np.stack(data_new, axis=0)
    return data_new

def _get_data( data):
    #print("DEBUG get data:", data.shape)
    data = data.transpose(2,0,1)    
    # print(data.shape)
    #print("data:", data)     
    data = np.expand_dims(data, axis=3)
    # data = data.transpose(3,2,1,0)

    #print("DEBUG get _get_data:", data.shape)
    joint, velocity, bone = multi_input(data[:,:T,:,:])
    data_new = []
    if 'J' in inputs:
        data_new.append(joint)
    if 'V' in inputs:
        data_new.append(velocity)
    if 'B' in inputs:
        data_new.append(bone)
    data_new = np.stack(data_new, axis=0)
    return data_new

def multi_input( data):
    C, T, V, M = data.shape
    joint = np.zeros((C*2, T, V, M))
    velocity = np.zeros((C*2, T, V, M))
    bone = np.zeros((C*2, T, V, M))
    joint[:C,:,:,:] = data
    for i in range(V):
        joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
    for i in range(T-2):
        velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
        velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
    for i in range(len(connect_joint)):
        bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,connect_joint[i],:]
    bone_length = 0
    for i in range(C):
        bone_length += bone[i,:,:,:] ** 2
    bone_length = np.sqrt(bone_length) + 0.0001
    for i in range(C):
        bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
    return joint, velocity, bone

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


selected_ids = {
    "nose":0,
    "l_sho":5, "r_sho":6,
    "l_elb":7, "r_elb":8,
    "l_wri":9, "r_wri":10,
    "l_hip":11,"r_hip":12,
    "l_knee":13,"r_knee":14,
    "l_ank":15,"r_ank":16
}

label_list = [
    "Standing_still",
    "Sitting_still",
    "Walking",
    "Fall_down",
    "Lying",
    "Staggering"
]

group_map = {
    "Fall_down":  "Falling",
    "Staggering": "Staggering"
    # Các nhãn khác sẽ thành "Other"
}
def extract_14_from_17(kp17):
    """
    kp17: np.array shape (17,3) order = COCO
    returns np.array shape (14,3) in your model order
    """
    out = np.zeros((14,3), dtype=kp17.dtype)
    out[0]  = kp17[selected_ids["nose"]]
    # neck:
    out[1]  = (kp17[selected_ids["l_sho"]] + kp17[selected_ids["r_sho"]]) / 2
    out[2]  = kp17[selected_ids["r_sho"]]
    out[3]  = kp17[selected_ids["r_elb"]]
    out[4]  = kp17[selected_ids["r_wri"]]
    out[5]  = kp17[selected_ids["l_sho"]]
    out[6]  = kp17[selected_ids["l_elb"]]
    out[7]  = kp17[selected_ids["l_wri"]]
    out[8]  = kp17[selected_ids["r_hip"]]
    out[9]  = kp17[selected_ids["r_knee"]]
    out[10] = kp17[selected_ids["r_ank"]]
    out[11] = kp17[selected_ids["l_hip"]]
    out[12] = kp17[selected_ids["l_knee"]]
    out[13] = kp17[selected_ids["l_ank"]]
    return out

if __name__ == '__main__':
    # --- model ---
    model_file = 'EfficentGCN-B4_asilla_secom_20250521.onnx'
    model = onnxruntime.InferenceSession(model_file, providers=['CPUExecutionProvider'])
    input_name = model.get_inputs()[0].name

    input_folder  = 'data/client_v4_video/5FPS_all_csv'
    output_folder = 'client_v4_gcn_predictions'
    os.makedirs(output_folder, exist_ok=True)

    FRAME_SIZE = np.array([1920, 1080])      

    csv_paths = []
    for dirpath, dirnames, filenames in os.walk(input_folder):
        for fname in filenames:
            if fname.lower().endswith('.csv'):
                csv_paths.append(os.path.join(dirpath, fname))


    for path_in in csv_paths:
        fname = os.path.basename(path_in)
        if not fname.lower().endswith('.csv'):
            continue

        print("Processing", path_in)
        df = pd.read_csv(path_in)

        records = []
        for pid, grp in df.groupby('PID'):
            grp = grp.sort_values('FrameID')

            # 1) Đọc full 14×3
            raw17 = np.stack([
                extract_14_from_17(np.array(json.loads(s), dtype=float).reshape(-1,3))
                for s in grp['Pose']
            ], axis=0).astype(np.float32)       # (n_frames, 14, 3)

            # 2) Tách x,y và thêm chiều M=1
            coords = raw17[..., :2]             # (n_frames, 14, 2)
            personList = coords[..., np.newaxis, :]  # (n_frames, 14, 1, 2)

            # 3) Normalize trên từng frame
            normed, _ = norm_data2(personList, FRAME_SIZE)
            # normed.shape == (n_frames, 14, 1, 2)

            # 4) Lấy lại coords đã norm và bỏ chiều M
            normed_coords = normed[:, :, 0, :]  # (n_frames, 14, 2)
            n_frames = len(normed_coords)

            # sliding window
            for start in range(0, n_frames - WINDOW_SIZE + 1, STRIDE):
                end = start + WINDOW_SIZE
                window = normed_coords[start:end]  # (WINDOW_SIZE, 14, 2)

                # build input
                npy = _get_data(window)                     # (I, C*2, T, V, M)
                npy = np.expand_dims(npy, 0).astype(np.float32)  # (1, I, C*2, T, V, M)

                # infer
                outputs = model.run(None, {input_name: npy})[0]  # logits (1, C)
                probs   = softmax(outputs, axis=1)
                pred    = np.argmax(probs, axis=1)[0]
                pred_p  = probs[0, pred]

                orig_label = label_list[pred]
                new_label  = group_map.get(orig_label, "Other")

                print(f"PID={pid}, frames [{start:2d}–{end-1:2d}] → "
                    f"{orig_label} ⇒ {new_label} (prob={pred_p:.3f})")

                records.append({
                    "Frame_Range": f"{start}-{end-1}",
                    "PID": pid,
                    "Label": orig_label,
                    "Prob": pred_p
                })
            
            out_df = pd.DataFrame(records, columns=["Frame_Range","PID","Label","Prob"])
            base, _ = os.path.splitext(fname)
            out_csv  = os.path.join(output_folder, f"{base}_predictions.csv")
            out_df.to_csv(out_csv, index=False)
            print(f"Saved predictions for {fname} → {out_csv}")


