import os
import json
import cv2
import numpy as np
import pandas as pd
from frequency_pipeline import WaveletFeatureExtractor 
from helper import _draw_limbs, _draw_pid

def annotate_video_from_csv(video_path, csv_path, out_path):
    """
    Annotate video with bounding boxes and poses from CSV.
    """

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
    fps    = cap.get(cv2.CAP_PROP_FPS) or 5.0

    # --- Prepare writer ---
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

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
                x0, y0, w, h = map(int, raw[:4])     
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

    video_basename = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    base_output_dir = os.path.join('client_v1_video')
    out_dir         = os.path.join(base_output_dir, video_basename)
    plots_dir       = os.path.join(out_dir, 'plots')
    annotated_mp4   = f"{video_basename}_annotated.mp4"
    out_video_path  = os.path.join(out_dir, annotated_mp4)

    os.makedirs(plots_dir, exist_ok=True)
    # 1) Vẽ plots
    df = pd.read_csv(CSV_PATH)
    for pid, grp in df.groupby('PID'):
        grp = grp.sort_values('FrameID')
        # lấy 4 giá trị đầu của bbox
        pid_plots = os.path.join(plots_dir, f"pid_{pid}")
        os.makedirs(pid_plots, exist_ok=True)

        extractor = WaveletFeatureExtractor(fs=5, wavelet='db1',
                                        dwt_level=3, save_dir=pid_plots)

        all_bboxes = np.stack([json.loads(s)[:4] for s in grp['Bbox']])
        all_keypoints = np.stack([
            np.array(json.loads(s), dtype=float).reshape(-1,3)
            for s in grp['Pose']
        ])
        dist_arr, ang_arr, bbox_feats = extractor.extract_time_features(
            all_keypoints, all_bboxes, FRAME_SIZE)
        extractor.plot_all_features(dist_arr, ang_arr, bbox_feats)

    # 2) Annotate video
    annotate_video_from_csv(
        video_path =VIDEO_PATH,
        csv_path =CSV_PATH,
        out_path =out_video_path,
    )

if __name__ == '__main__':
    csv_folder = 'data/client_v1_video/client_v1_video_csv'
    video_folder = 'data/client_v1_video'

    csv_files = {os.path.splitext(f)[0]: os.path.join(csv_folder, f)
                 for f in os.listdir(csv_folder) if f.endswith('.csv')}

    # for root, folders, files in os.walk(csv_folder):
    #     for file in files:
    #         if file.endswith('.csv'):
    #             csv_paths.append(os.path.join(root, file))
                
    for root, folders, files in os.walk(video_folder):
        for file in files:
            if file.endswith('.mp4'):
                name_no_ext = os.path.splitext(file)[0]

                if name_no_ext in csv_files:
                    csv_path = csv_files[name_no_ext]
                    video_path = os.path.join(root, file)

                    print(f"CSV: {csv_path}")
                    print(f"Video: {video_path}")

                    main(csv_path=csv_path, video_path=video_path, frame_size=(1920,1080))
    print('finish !!!!!!!!!!!!!!!!')

