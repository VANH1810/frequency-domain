from frequency_pipeline import WaveletFeatureExtractor
import os
import pandas as pd
import numpy as np
import json
import pywt
from tqdm import tqdm
import pickle
def compute_dwt_coeffs( signal):
    level=3
    coeffs = pywt.wavedec(signal - np.mean(signal), 'db1', level=level)
    feats = []
    # for c in coeffs:
    #     feats.extend([
    #         np.sum(c**2),
    #         np.mean(c),
    #         np.std(c),
    #         np.max(np.abs(c)),
    #     ])
    # return np.array(feats)
    names = [f'cA{level}'] + [f'cD{l}' for l in range(level, 0, -1)]
    return {name: coeff for name, coeff in zip(names, coeffs)}

def main(csv_path, frame_size):
    # Sửa lại đây cho đúng csv/video của bạn
    CSV_PATH   = csv_path
    FRAME_SIZE = frame_size

    # Tạo thư mục output dựa trên tên video
    csv_basename = os.path.splitext(os.path.basename(CSV_PATH))[0]
    base_output_dir = os.path.join('client_v4_feat')
    out_dir         = os.path.join(base_output_dir, csv_basename)
    feats_dir       = os.path.join(out_dir)

    # os.makedirs(feats_dir, exist_ok=True)
    name_dwt = {0:'cA', 1:'cD1',2:'cD2', 3:'cD3'}
    # 1) Vẽ plots
    df = pd.read_csv(CSV_PATH)
    for pid, grp in df.groupby('PID'):
        grp = grp.sort_values('FrameID')
        # lấy 4 giá trị đầu của bbox
        pid_plots = os.path.join(feats_dir, f"pid_{pid}")
        # os.makedirs(pid_plots, exist_ok=True)

        extractor = WaveletFeatureExtractor(fs=30, wavelet='db1',
                                        dwt_level=3, save_dir=pid_plots)

        all_bboxes = np.stack([json.loads(s)[:4] for s in grp['Bbox']])
        all_keypoints = np.stack([
            np.array(json.loads(s), dtype=float).reshape(-1,3)
            for s in grp['Pose']
        ])
        dist_arr, ang_arr, bbox_feats = extractor.extract_time_features(
            all_keypoints, all_bboxes, FRAME_SIZE)
        dist_path = os.path.join(pid_plots, 'dist_arr.npy')
        angle_path = os.path.join(pid_plots, 'angle_arr.npy')        
        bbox_path = os.path.join(pid_plots, 'bbox_arr.npy')
        np.save(dist_path, dist_arr)
        np.save(angle_path, ang_arr)
        np.save(bbox_path, bbox_path)
        # dwt_dist_all  
        result = {}
        arrays = {'dist': dist_arr, 'ang': ang_arr, 'bbox': bbox_feats}
        for feat_name, arr in arrays.items():
            for idx in range(arr.shape[1]):
                sig = arr[:, idx]
                coeff_dict = compute_dwt_coeffs(sig)
                result[f'{feat_name}_{idx}'] = coeff_dict
        dwt_path = os.path.join(pid_plots, 'dwt_data.pkl')
        with open(dwt_path, 'wb') as write:
            pickle.dump(result, write)
if __name__ == '__main__':
    frame_size = (1920, 1080)
    csv_folder = 'data/client_v4_video/5FPS_all_csv'
    csv_paths = []
    for root, folders, files in os.walk(csv_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_paths.append(os.path.join(root, file))
    
    for csv_path in tqdm(csv_paths):
        main(csv_path, frame_size)
