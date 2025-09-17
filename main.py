from frequency_pipeline import WaveletFeatureExtractor
import pywt
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import numpy as np

def process_csv_to_json_dwt(
    csv_path: str,
    frame_size: tuple,
    output_json: str,
    fs: float = 5.0,
    wavelet: str = 'db1',
    dwt_level: int = 3
):
    df = pd.read_csv(csv_path)
    print()
    extractor = WaveletFeatureExtractor(fs=fs, wavelet=wavelet, dwt_level=dwt_level)

    result = {}
    for pid, group in df.groupby('PID'):
        grp = group.sort_values('FrameID')
        # Parse bbox
        all_bboxes = np.stack([ json.loads(s)[0:4] for s in grp['Bbox'] ])
        # Parse pose
        all_keypoints = np.stack([
            np.array(json.loads(s), dtype=float).reshape(-1, 3)[:, :2]
            for s in grp['Pose']
        ])
        # Extract DWT coeffs (nested dict of ndarray)
        feats = extractor.extract_dwt_features(all_keypoints, all_bboxes, frame_size)
        # Convert ndarray -> list for JSON
        pid_dict = {}
        for feat_name, coeffs in feats.items():
            pid_dict[feat_name] = {k: v.tolist() for k, v in coeffs.items()}
        result[int(pid)] = pid_dict

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved DWT coefficients JSON to {output_json}")

if __name__ == '__main__':

    csv_root     = 'data/client_v4_video/5FPS_all_csv' 
    output_root  = 'output-wavelet' 
    base_dir   = os.path.dirname(csv_root)
    base_name  = os.path.basename(base_dir)
    frame_size   = (1920, 1080)
    fs           = 5.0
    wavelet      = 'db1'
    dwt_level    = 3


    for root, _, files in os.walk(csv_root):
        for fname in files:
            if not fname.endswith('.csv'):
                continue

            # Đường dẫn file CSV gốc
            csv_path = os.path.join(root, fname)

            # Relpath tính từ csv_root, giữ nguyên cấu trúc con
            rel_dir = os.path.relpath(root, csv_root)

            # Kết hợp: outputs/ + client_v2_video/ + (cấu trúc con)
            out_dir = os.path.join(output_root, base_name, rel_dir)
            os.makedirs(out_dir, exist_ok=True)

            # Tạo tên file JSON
            json_name   = f"{os.path.splitext(fname)[0]}_dwt_features.json"
            output_json = os.path.join(out_dir, json_name)

            # Chạy xử lý
            process_csv_to_json_dwt(
                csv_path=csv_path,
                frame_size=frame_size,
                output_json=output_json,
                fs=fs,
                wavelet=wavelet,
                dwt_level=dwt_level
            )
            print(f"→ Đã xử lý: {csv_path}")
            print(f"  → Lưu JSON tại: {output_json}\n")
    

