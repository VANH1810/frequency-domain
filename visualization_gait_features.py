import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from frequency_pipeline import WaveletFeatureExtractor 
from helper import _draw_limbs, _draw_pid
from gait_features import calculate_all_gait_features, ALL_FEATURE_KEYS, JOINT_INDICES, calculate_jerk

def annotate_video_from_csv(video_path, csv_path, out_path, save_frames_dir=None):
    # --- Load DataFrame ---
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    frame_groups = df.groupby('FrameID')

    # --- Open video & writer ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))

    # --- Annotate frame by frame ---
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = frame.copy()
        if frame_idx in frame_groups.groups:
            group = frame_groups.get_group(frame_idx)
            for _, row in group.iterrows():
                pid = int(row['PID'])
                raw = json.loads(row['Bbox'])
                x0, y0, w, h = map(int, raw[:4])     # chỉ lấy 4 giá trị đầu
                x1, y1 = x0 + w, y0 + h
                cv2.rectangle(img, (x0, y0), (x1, y1), (0,255,0), 2)
                #cv2.putText(img, f"ID{pid}", (x0, y0-5),
                #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                pts = np.array(json.loads(row['Pose']), dtype=float).reshape(-1,3)
                img = _draw_limbs(pts[:,:2],img)
                img = cv2.putText(img, f"ID{pid}", (x0, y0-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                #img = _draw_pid(img, [x0,y0,x1,y1],pid)
                #for xk, yk, _ in pts:
                #    cv2.circle(img, (int(xk), int(yk)), 3, (0,0,255), -1)

        writer.write(img)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved annotated video to {out_path}")

def main(csv_path, video_path, frame_size):
    CSV_PATH   = csv_path
    VIDEO_PATH = video_path
    FRAME_SIZE = frame_size

    # Tạo thư mục output dựa trên tên video
    video_basename = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    base_output_dir = os.path.join('client_v4_timeseries')
    out_dir         = os.path.join(base_output_dir, video_basename)
    plots_dir       = os.path.join(out_dir, 'plots')
    frames_dir      = os.path.join(out_dir, 'frames')
    annotated_mp4   = f"{video_basename}_annotated.mp4"
    out_video_path  = os.path.join(out_dir, annotated_mp4)

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # 1) Vẽ plots (wavelet + gait time‐series)
    df = pd.read_csv(CSV_PATH)
    csv_basename = os.path.splitext(os.path.basename(CSV_PATH))[0]

    for pid, grp in df.groupby('PID'):
        grp = grp.sort_values('FrameID')
        pid_plots = os.path.join(plots_dir, f"pid_{pid}")
        os.makedirs(pid_plots, exist_ok=True)

        # --- A) Wavelet features ---
        extractor = WaveletFeatureExtractor(fs=5, wavelet='db1',
                                            dwt_level=3, save_dir=pid_plots)
        all_bboxes = np.stack([json.loads(s)[:4] for s in grp['Bbox']])
        all_keypoints = np.stack([
            np.array(json.loads(s), dtype=float).reshape(-1,3)
            for s in grp['Pose']
        ])

        # dist_arr, ang_arr, bbox_feats = extractor.extract_time_features(
        #     all_keypoints, all_bboxes, FRAME_SIZE)
        # extractor.plot_all_features(dist_arr, ang_arr, bbox_feats)

        # --- B) Gait features time‐series ---
        # 1. Chuẩn bị pose chỉ lấy x,y
        pose_xy = all_keypoints[:, :, :2]   # shape = (T, J, 2)

        # # 2. Khởi tạo dict để chứa time‐series của mỗi feature
        ts_features = {feat: [] for feat in ALL_FEATURE_KEYS}
        window_size = 5
        ts_frames   = []
        # # 3. Tính dần từng time‐step
        for start in range(0, len(pose_xy), window_size):
            end = min(start + window_size, len(pose_xy))
            window = pose_xy[start:end]                  # shape ≤ (5, J, 2)
            feats  = calculate_all_gait_features(window, JOINT_INDICES, fps=5)

            # lưu kết quả
            for feat in ALL_FEATURE_KEYS:
                ts_features[feat].append(feats.get(feat, np.nan))
            ts_frames.append(end)

        # 4. Vẽ line‐plot cho từng feature và lưu ảnh
        for feat, values in ts_features.items():
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(ts_frames, values, marker='.', linestyle='-')
            ax.set_title(feat, pad=8)
            ax.set_xlabel('Frame Index')
            ax.set_ylabel(feat)
            plt.tight_layout()
            fig.savefig(os.path.join(pid_plots, f'{feat}.png'))
            plt.close(fig)

    # 2) Annotate video
    # annotate_video_from_csv(
    #     video_path=VIDEO_PATH,
    #     csv_path=CSV_PATH,
    #     out_path=out_video_path,
    #     save_frames_dir=frames_dir
    # )

if __name__ == '__main__':
    csv_folder = 'data/client_v4_video/5FPS_all_csv/'
    video_folder = 'data/client_v4_video/5FPS_all/'
    csv_paths = []
    for root, folders, files in os.walk(csv_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_paths.append(os.path.join(root, file))

    for csv_path in csv_paths:
        video_path = csv_path.replace(csv_folder, video_folder)
        video_path = video_path.replace('.csv','.MP4')
        print(video_path, csv_path)
        main(csv_path=csv_path, video_path = video_path, frame_size=(1920,1080))
        # main(
        #     csv_path='clientdata_eval_csv/falling/20250218_171252_67_tentou_front.csv',
        #     video_path='clientdata_eval/falling/20250218_171252_67_tentou_front.mp4',
        #     frame_size=(1920, 1080)
        # )
    print('finish !!!!!!!!!!!!!!!!')

