from frequency_pipeline import WaveletFeatureExtractor
import os
import pandas as pd
import numpy as np
import json
import pywt
from tqdm import tqdm
import pickle

# =========================
# DWT helper
# =========================
def compute_dwt_coeffs(signal):
    level = 3
    coeffs = pywt.wavedec(signal - np.mean(signal), 'db1', level=level)
    names = [f'cA{level}'] + [f'cD{l}' for l in range(level, 0, -1)]
    return {name: coeff for name, coeff in zip(names, coeffs)}


# =========================
# Tính bbox từ Pose
# =========================
# ---------- Guard ----------
def is_nan_or_empty(s):
    if s is None: return True
    if isinstance(s, float) and np.isnan(s): return True
    if isinstance(s, str) and s.strip() == "": return True
    return False

def parse_xy_score_to_kpts3(xy_str, score_str, frame_size, default_conf=1.0):
    """
    xy_str: chuỗi JSON của POSES -> flat [x1,y1,x2,y2,...] hoặc [[x,y], ...]
    score_str: chuỗi JSON của SCORE -> [c1, c2, ..., cK] (có thể None)
    Trả về (K,3): [x,y,conf]
    """
    W, H = frame_size

    # --- Parse XY ---
    xy = json.loads(str(xy_str))  # an toàn với dấu ngoặc kép trong CSV
    xy = np.array(xy, dtype=float)

    if xy.ndim == 1:
        n = xy.size
        if n % 2 != 0:
            raise ValueError(f"POSES length {n} không chia hết cho 2.")
        kpts_xy = xy.reshape(-1, 2)  # (K,2)
    elif xy.ndim == 2 and xy.shape[1] == 2:
        kpts_xy = xy  # (K,2)
    elif xy.ndim == 2 and xy.shape[0] == 2:  # (2,K) -> transpose
        kpts_xy = xy.T
    else:
        raise ValueError(f"POSES shape không hợp lệ: {xy.shape}")

    K = kpts_xy.shape[0]

    # --- Parse SCORE thành conf ---
    if score_str is not None and str(score_str).strip() not in ("", "nan", "None"):
        sc = json.loads(str(score_str))
        sc = np.array(sc, dtype=float).reshape(-1)
        if sc.size < K:
            # thiếu -> pad
            pad = np.full((K - sc.size,), float(default_conf), dtype=float)
            conf = np.concatenate([sc, pad], axis=0)
        elif sc.size > K:
            # thừa -> cắt
            conf = sc[:K]
        else:
            conf = sc
    else:
        conf = np.full((K,), float(default_conf), dtype=float)

    # --- Ghép [x,y,conf] + clip về khung ảnh ---
    kpts = np.concatenate([kpts_xy, conf[:, None]], axis=1)  # (K,3)
    kpts[:, 0] = np.clip(kpts[:, 0], 0, W - 1)
    kpts[:, 1] = np.clip(kpts[:, 1], 0, H - 1)
    return kpts  # (K,3)

# ---------- BBox với guard ----------
def bbox_from_pose_array(kpts_frame, frame_size, conf_thr=0.0):
    W, H = frame_size
    if kpts_frame is None or kpts_frame.ndim != 2 or kpts_frame.shape[1] < 2:
        return [0.0, 0.0, 2.0, 2.0]

    if kpts_frame.shape[1] >= 3 and conf_thr > 0.0:
        mask = kpts_frame[:, 2] >= conf_thr
        pts = kpts_frame[mask][:, :2] if mask.any() else kpts_frame[:, :2]
    else:
        pts = kpts_frame[:, :2]

    xs = np.clip(pts[:, 0], 0, W - 1)
    ys = np.clip(pts[:, 1], 0, H - 1)
    x0, y0 = float(xs.min()), float(ys.min())
    x1, y1 = float(xs.max()), float(ys.max())
    w = max(2.0, x1 - x0)
    h = max(2.0, y1 - y0)
    return [x0, y0, w, h]

def main(csv_path, frame_size):

    CSV_PATH   = csv_path
    FRAME_SIZE = frame_size

    # Thư mục output theo tên CSV
    csv_basename = os.path.splitext(os.path.basename(CSV_PATH))[0]
    base_output_dir = os.path.join('JPRecorded-feats-newv2')
    out_dir         = os.path.join(base_output_dir, csv_basename)
    feats_dir       = os.path.join(out_dir)
    os.makedirs(feats_dir, exist_ok=True)
    name_dwt = {0:'cA', 1:'cD1',2:'cD2', 3:'cD3'}

    df = pd.read_csv(CSV_PATH)

    frame_col = 'FrameID' if 'FrameID' in df.columns else ('FRAMEID' if 'FRAMEID' in df.columns else None)
    if frame_col is None:
        raise ValueError("CSV cần có cột 'FrameID' hoặc 'FRAMEID'.")
    if 'PID' not in df.columns:
        raise ValueError("CSV cần có cột 'PID'.")
    if 'POSES' not in df.columns:
        raise ValueError("CSV cần có cột 'Pose' (list [x,y,conf]*K).")

    pose_col_candidates  = ['POSES', 'POSE', 'Pose']
    score_col_candidates = ['SCORE', 'Score', 'SCORES']

    pose_col  = next((c for c in pose_col_candidates  if c in df.columns), None)
    score_col = next((c for c in score_col_candidates if c in df.columns), None)
    if pose_col is None:
        raise ValueError(f"Không tìm thấy cột POSES. Thử một trong: {pose_col_candidates}")

    # df['PID_orig'] = df.get('PID', 1)
    # df['PID'] = 1
    for pid, grp in df.groupby('PID'):
        grp = grp.sort_values(frame_col)

        pid_plots = os.path.join(feats_dir, f"pid_{pid}")
        os.makedirs(pid_plots, exist_ok=True)

        extractor = WaveletFeatureExtractor(
            fs=15, wavelet='db1', dwt_level=3, save_dir=pid_plots
        )

        # all_bboxes = np.stack([json.loads(s)[:4] for s in grp['Bbox']])
        # Parse Pose: (T, K, 3)

        # poses_list = []
        # for s in grp['POSES']:
        #     kf = parse_pose_entry(s, FRAME_SIZE, default_conf=1.0)  # <-- thay vì reshape(-1,3)
        #     poses_list.append(kf)
        # all_keypoints = np.stack(poses_list, axis=0)  # (T,K,3)

        # # Tạo bbox từ pose
        # all_bboxes = np.stack(
        #     [bbox_from_pose_array(kf, FRAME_SIZE, conf_thr=0.0) for kf in all_keypoints],
        #     axis=0
        # )  # (T,4)

        poses_list  = []
        bboxes_list = []
        bad_rows    = []

        for _, row in grp.iterrows():
            try:
                kf = parse_xy_score_to_kpts3(
                    row[pose_col],
                    row[score_col] if score_col in grp.columns else None,
                    FRAME_SIZE,
                    default_conf=1.0
                )
                poses_list.append(kf)
                bboxes_list.append(bbox_from_pose_array(kf, FRAME_SIZE, conf_thr=0.30))
            except Exception as e:
                bad_rows.append((int(row[frame_col]), str(e)))

        if not poses_list:
            print(f"[WARN] PID {pid}: toàn bộ frame lỗi. Ví dụ lỗi: {bad_rows[:2]}")
            continue

        # Nếu số keypoint mỗi frame khác nhau -> cắt về K_min để stack an toàn
        K_min = min(k.shape[0] for k in poses_list)
        poses_list  = [k[:K_min, :] for k in poses_list]
        bboxes_list = bboxes_list  # bbox đã (T,4), không phụ thuộc K

        all_keypoints = np.stack(poses_list, axis=0)                 # (T, K_min, 3)
        all_bboxes    = np.stack(bboxes_list, axis=0).astype(float)  # (T, 4)

        if bad_rows:
            print(f"[INFO] PID {pid}: kept {len(poses_list)} frames, skipped {len(bad_rows)} (ví dụ: {bad_rows[:2]})")

        # Trích xuất đặc trưng theo thời gian
        wr_feats, wr_names = extractor.extract_wr_frame_features(
            all_keypoints, all_bboxes, FRAME_SIZE
        )

        # --- Lưu time series ---
        np.save(os.path.join(pid_plots, 'wr_feats.npy'), wr_feats)
        with open(os.path.join(pid_plots, 'wr_feat_names.json'), 'w', encoding='utf-8') as f:
            json.dump(wr_names, f, ensure_ascii=False, indent=2)

        # Lưu thêm dạng CSV để dễ xem/plot nhanh
        pd.DataFrame(wr_feats, columns=wr_names).to_csv(
            os.path.join(pid_plots, 'wr_feats.csv'), index=False
        )

        # Mỗi kênh là một file .npy riêng:
        for i, name in enumerate(wr_names):
            np.save(os.path.join(pid_plots, f'wr_{name}.npy'), wr_feats[:, i])


if __name__ == '__main__':
    frame_size = (1920, 1080)
    # TH1: chạy 1 file
    # main('path/to/your.csv', frame_size)

    # TH2: quét cả thư mục
    csv_folder = 'JPrecorded_track'
    csv_paths = []
    for root, folders, files in os.walk(csv_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_paths.append(os.path.join(root, file))

    for csv_path in tqdm(csv_paths):
        main(csv_path, frame_size)
