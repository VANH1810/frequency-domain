#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import pywt
import re
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# =========================
# CẤU HÌNH
# =========================
FP_RATE      = 15         # fps
WINDOW_SEC   = 3          # độ dài cửa sổ (giây)
STEP_SEC     = 0.5        # bước trượt (giây)
DIST1_IDX    = 1          # index dist1 trong dist_arr
DIST9_IDX    = 9          # index dist9 trong dist_arr
ANGLE0_IDX   = 0          # index angle0 trong dist_arr
BBOX_SPEED_IDX = 3        # index speed trong bbox_arr (0-based)

WAVELET      = 'db2'      # wavelet ngắn, ổn với cửa sổ 20 mẫu
DWT_LEVEL    = 3          # mục tiêu level=3 (tự hạ nếu dữ liệu ngắn)

# =========================
# LOADER
# =========================
def load_all_features(base_dir):
    """
    Cấu trúc thư mục như trước: base_dir/<video>/<pid_dir>/*.npy
    Mỗi pid_dir có các file .npy, ví dụ: dist_arr.npy, bbox_arr.npy, v.v.
    """
    data = {}
    for video in os.listdir(base_dir):
        p_v = os.path.join(base_dir, video)
        if not os.path.isdir(p_v): 
            continue
        data[video] = {}
        for pid_dir in os.listdir(p_v):
            p_p = os.path.join(p_v, pid_dir)
            if not os.path.isdir(p_p): 
                continue
            pid = pid_dir.split('_',1)[1] if '_' in pid_dir else pid_dir
            data[video][pid] = {}
            for fn in os.listdir(p_p):
                if not fn.endswith('.npy'): 
                    continue
                arr = np.load(os.path.join(p_p, fn))
                arr2d = arr.reshape(-1,1) if arr.ndim==1 else arr
                data[video][pid][fn[:-4]] = arr2d
    return data

def load_all_wr_features(base_dir):
    """
    Tìm trong base_dir/<video>/<pid_dir>/wr_feats.npy (+ wr_feat_names.json nếu có).
    Trả về:
      data[video][pid] = {
         'wr_feats': np.ndarray (T, 10),
         'wr_names': list[str] (len=10) hoặc None
      }
    """
    data = {}
    for video in os.listdir(base_dir):
        p_v = os.path.join(base_dir, video)
        if not os.path.isdir(p_v):
            continue
        data[video] = {}
        for pid_dir in os.listdir(p_v):
            p_p = os.path.join(p_v, pid_dir)
            if not os.path.isdir(p_p):
                continue
            pid = pid_dir.split('_',1)[1] if '_' in pid_dir else pid_dir

            wr_path = os.path.join(p_p, 'wr_feats.npy')
            if not os.path.exists(wr_path):
                continue
            arr = np.load(wr_path)  # (T, 10)
            names_path = os.path.join(p_p, 'wr_feat_names.json')
            wr_names = None
            if os.path.exists(names_path):
                try:
                    with open(names_path, 'r', encoding='utf-8') as f:
                        wr_names = json.load(f)
                except Exception:
                    wr_names = None

            data[video][pid] = {'wr_feats': arr, 'wr_names': wr_names}
    return data


FEATURE_COL_REGEX = r'^f\d+$'

def get_feature_cols(df: pd.DataFrame):
    """
    Chỉ lấy cột đặc trưng dạng 'f' + số (f0, f1, ...).
    Không thể dính 'frame_range' vì không match ^f\\d+$.
    """
    cols = df.filter(regex=FEATURE_COL_REGEX).columns.tolist()
    if not cols:
        raise RuntimeError("Không tìm thấy cột đặc trưng dạng f<digits> (vd: f0, f1, ...).")
    # Debug an toàn:
    assert all(re.match(FEATURE_COL_REGEX, c) for c in cols), cols
    return cols


# =========================
# WAVELET FEATURES (NO FFT)
# =========================
def _detrend(x):
    x = np.asarray(x, float)
    return x - np.mean(x)

def _pad_energies_to_L3(energies, level_available):
    """
    energies của [A_L, D_L, D_{L-1}, ..., D_1] -> map về [E_A3, E_D3, E_D2, E_D1]
    Nếu L<3, điền 0 cho băng thiếu; A_L gán vào A3.
    """
    # unpack energies
    # energies[0] = E_A_L
    # energies[1:] = [E_D_L, E_D_{L-1}, ..., E_D_1]
    E_A_L = energies[0]
    D_list = energies[1:]

    E_A3 = E_A_L
    E_D3 = D_list[0] if level_available >= 3 else 0.0
    if level_available >= 3:
        E_D2 = D_list[1] if len(D_list) >= 2 else 0.0
        E_D1 = D_list[-1]  # D1 là phần tử cuối
    elif level_available == 2:
        # D_list = [E_D2, E_D1]
        E_D2 = D_list[0]
        E_D1 = D_list[1] if len(D_list) > 1 else 0.0
    else:  # level_available == 1
        E_D2 = 0.0
        E_D1 = D_list[0] if len(D_list) > 0 else 0.0

    return E_A3, E_D3, E_D2, E_D1

def dwt_energy_feats(x, wavelet=WAVELET, level=DWT_LEVEL):
    """
    Đặc trưng Wavelet-only cho 1 kênh (cửa sổ 4s ~ 20 mẫu):
      - Năng lượng & tỷ lệ năng lượng theo băng (A3,D3,D2,D1)
      - Chỉ số R12 = E_D1 / (E_D2 + eps)
      - HiLo = (E_D1+E_D2)/(E_A3+E_D3 + eps)
      - Entropy theo băng
      - Sparsity (L1/L2) cho D1, D2
      - Big-fraction (tỷ lệ hệ số lớn) cho D1, D2
    Output: vector 15 chiều (ổn định)
    """
    x = _detrend(x)
    wave = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(x), wave.dec_len)
    L = max(1, min(level, max_level))  # ít nhất 1

    coeffs = pywt.wavedec(x, wavelet=wavelet, level=L, mode='symmetric')  # [A_L, D_L, ..., D_1]
    Es = [float(np.sum(c*c)) for c in coeffs]
    E_A3, E_D3, E_D2, E_D1 = _pad_energies_to_L3(Es, L)
    E_tot = E_A3 + E_D3 + E_D2 + E_D1 + 1e-9

    r_A3, r_D3, r_D2, r_D1 = E_A3/E_tot, E_D3/E_tot, E_D2/E_tot, E_D1/E_tot
    R12  = E_D1 / (E_D2 + 1e-9)
    HiLo = (E_D1 + E_D2) / (E_A3 + E_D3 + 1e-9)

    # band entropy (4 dải)
    p = np.array([E_A3, E_D3, E_D2, E_D1], float) / E_tot
    band_entropy = float(-np.sum(p * np.log(p + 1e-12)))

    # L1/L2 và big-fraction cho D1, D2: lấy coefficient thực từ coeffs hiện có
    # Tìm D1 và D2 thực tế (nếu không có, dùng mảng rỗng)
    # coeffs = [A_L, D_L, D_{L-1}, ..., D_1]
    D1_coeff = coeffs[-1] if len(coeffs) >= 2 else np.array([])
    if L >= 2:
        D2_coeff = coeffs[-2]
    else:
        D2_coeff = np.array([])

    def L1L2(c):
        c = np.asarray(c, float)
        if c.size == 0: 
            return 0.0
        L1 = float(np.sum(np.abs(c))) + 1e-12
        L2 = float(np.sqrt(np.sum(c*c))) + 1e-12
        return L1 / L2

    def bigfrac(c):
        c = np.asarray(c, float)
        if c.size == 0:
            return 0.0
        a = np.abs(c)
        med = np.median(a)
        mad = np.median(np.abs(a - med)) + 1e-9
        thr = med + 1.4826 * mad
        return float(np.mean(a > thr))

    D1_L1L2 = L1L2(D1_coeff)
    D2_L1L2 = L1L2(D2_coeff)
    D1_bigfrac = bigfrac(D1_coeff)
    D2_bigfrac = bigfrac(D2_coeff)

    return np.array([
        E_A3, E_D3, E_D2, E_D1,
        r_A3, r_D3, r_D2, r_D1,
        R12, HiLo, band_entropy,
        D1_L1L2, D2_L1L2,
        D1_bigfrac, D2_bigfrac
    ], dtype=float)

import numpy as np
import pandas as pd
import json
import os

# ---- core stats helpers ----
def _q(x, p):  # percentile robust
    return float(np.percentile(np.asarray(x, float), p))

def _std(x):
    return float(np.std(np.asarray(x, float)))

def _mean(x):
    return float(np.mean(np.asarray(x, float)))

def _zcr(x):
    x = np.asarray(x, float)
    m = np.mean(x)
    s = np.sign(x - m)
    return float(np.mean(s[1:] * s[:-1] < 0)) if x.size >= 2 else 0.0

def _ac1(x):
    x = np.asarray(x, float)
    if x.size < 3: return 0.0
    x0, x1 = x[:-1], x[1:]
    m0, m1 = np.mean(x0), np.mean(x1)
    s0, s1 = np.std(x0) + 1e-12, np.std(x1) + 1e-12
    return float(np.mean((x0-m0)*(x1-m1)) / (s0*s1))

def _cv(x):
    x = np.asarray(x, float)
    mu = np.mean(x)
    return float(np.std(x) / (abs(mu) + 1e-12))

# ---- channel grouping on wr_names (10 kênh theo thứ tự bạn đã lưu) ----
DEFAULT_WR_NAMES = [
    "cos_knee_L","cos_knee_R",
    "hip_flex_L","hip_flex_R",
    "arm_swing_L","arm_swing_R",
    "ankleLR_over_torso",
    "wristLR_over_shoulder",
    "midhip_y_local_norm",
    "speed",
]

OSC_NAMES = {"cos_knee_L","cos_knee_R","hip_flex_L","hip_flex_R","arm_swing_L","arm_swing_R"}
RATIO_NAMES = {"ankleLR_over_torso","wristLR_over_shoulder"}
COM_NAMES = {"midhip_y_local_norm"}
SPEED_NAMES = {"speed"}

def stat_core_features_window_wr(win: np.ndarray, wr_names=None):
    """
    win: (wlen, C=10)
    wr_names: list tên kênh theo cùng thứ tự cột.
    Trả về (vec, names): vector 44D + tên cột.
    """
    if wr_names is None: wr_names = DEFAULT_WR_NAMES
    C = win.shape[1]
    assert C == len(wr_names), f"Mismatch channels: win has {C}, names {len(wr_names)}"

    vec_parts, name_parts = [], []

    for i, name in enumerate(wr_names):
        x = win[:, i]

        if name in OSC_NAMES:
            q05 = _q(x, 5); q95 = _q(x, 95)
            std = _std(x); zcr = _zcr(x); ac1 = _ac1(x)
            vec_parts += [q05, q95, std, zcr, ac1]
            name_parts += [f"{name}_q05", f"{name}_q95", f"{name}_std", f"{name}_zcr", f"{name}_ac1"]

        elif name in RATIO_NAMES:
            mu = _mean(x); std = _std(x); cv = _cv(x); q95 = _q(x, 95)
            vec_parts += [mu, std, cv, q95]
            name_parts += [f"{name}_mean", f"{name}_std", f"{name}_cv", f"{name}_q95"]

        elif name in COM_NAMES:
            std = _std(x); iqr = _q(x, 95) - _q(x, 5); ac1 = _ac1(x)
            vec_parts += [std, iqr, ac1]
            name_parts += [f"{name}_std", f"{name}_iqr95_5", f"{name}_ac1"]

        elif name in SPEED_NAMES:
            mu = _mean(x); std = _std(x); q95 = _q(x, 95)
            vec_parts += [mu, std, q95]
            name_parts += [f"{name}_mean", f"{name}_std", f"{name}_q95"]

        else:
            # fallback an toàn
            mu = _mean(x); std = _std(x)
            vec_parts += [mu, std]
            name_parts += [f"{name}_mean", f"{name}_std"]

    return np.array(vec_parts, float), name_parts

WR_WAVELET_CHANNELS = [
    "cos_knee_L","cos_knee_R",
    "hip_flex_L","hip_flex_R",
    "arm_swing_L","arm_swing_R",
    "midhip_y_local_norm","speed",
]


def extract_wavelet_features_window(win, win_bbox=None,
                                    dist1_idx=DIST1_IDX, dist9_idx=DIST9_IDX, angle0_idx=ANGLE0_IDX,
                                    use_bbox_speed=True, bbox_speed_idx=BBOX_SPEED_IDX):
    """Trả về vector đặc trưng Wavelet-only cho 4 kênh: dist1, dist9, angle0, bbox_speed (nếu có)."""
    feats = []
    for idx in [dist1_idx, dist9_idx, angle0_idx]:
        x = np.asarray(win[:, idx], float)
        feats.append(dwt_energy_feats(x, wavelet=WAVELET, level=DWT_LEVEL))

    if use_bbox_speed and (win_bbox is not None) and (win_bbox.shape[1] > bbox_speed_idx):
        sp = np.asarray(win_bbox[:, bbox_speed_idx], float)
        feats.append(dwt_energy_feats(sp, wavelet=WAVELET, level=DWT_LEVEL))
    else:
        feats.append(np.zeros(15, float))  # cố định chiều, nếu thiếu
    
    return np.hstack(feats)  # 4 * 15 = 60 chiều

def wavelet_feature_names():
    base = [
        "E_A3","E_D3","E_D2","E_D1",
        "r_A3","r_D3","r_D2","r_D1",
        "R12","HiLo","band_entropy",
        "D1_L1L2","D2_L1L2",
        "D1_bigfrac","D2_bigfrac"
    ]
    names = []
    for k in ["d1","d9","a0","bbox_spd"]:
        names += [f"{k}_{n}" for n in base]
    return names  # total 60

def extract_wavelet_features_window_wr(win):
    """
    win: (wlen, C) với C=10 kênh wr.
    Trả về vector 15*C (mặc định 150 chiều) = dwt_energy_feats cho từng kênh rồi nối.
    """
    C = win.shape[1]
    vecs = []
    for i in range(C):
        x = np.asarray(win[:, i], float)
        vecs.append(dwt_energy_feats(x, wavelet=WAVELET, level=DWT_LEVEL))
    return np.hstack(vecs)

def wavelet_feature_names_wr(wr_names=None):
    base = [
        "E_A3","E_D3","E_D2","E_D1",
        "r_A3","r_D3","r_D2","r_D1",
        "R12","HiLo","band_entropy",
        "D1_L1L2","D2_L1L2",
        "D1_bigfrac","D2_bigfrac"
    ]
    if wr_names is None or len(wr_names) == 0:
        # fallback tên kênh chuẩn theo thứ tự đã extract:
        wr_names = [
            "cos_knee_L","cos_knee_R",
            "hip_flex_L","hip_flex_R",
            "arm_swing_L","arm_swing_R",
            "ankleLR_over_torso",
            "wristLR_over_shoulder",
            "midhip_y_local_norm",
            "speed",
        ]
    names = []
    for k in wr_names:
        names += [f"{k}_{n}" for n in base]
    return names  # 15 * len(wr_names)

# =========================
# DATASET TẠO CỬA SỔ
# =========================
def build_dataset_wavelet(feats_data):
    """
    Trả về DataFrame:
      - cột đặc trưng số (60 cột)
      - meta: vid, pid, frame_range
    """
    records = []
    wlen = int(WINDOW_SEC * FP_RATE)
    step = int(STEP_SEC   * FP_RATE)

    for vid, pid_dict in feats_data.items():
        for pid, feats in pid_dict.items():
            arr = feats.get('dist_arr')
            arr_bbox = feats.get('bbox_arr')
            if arr is None or arr.shape[0] < wlen:
                continue
            T = arr.shape[0]
            for st in range(0, T - wlen + 1, step):
                ed = st + wlen
                win = arr[st:ed, :]
                win_bbox = arr_bbox[st:ed, :] if arr_bbox is not None else None

                vec = extract_wavelet_features_window(
                    win, win_bbox,
                    dist1_idx=DIST1_IDX, dist9_idx=DIST9_IDX, angle0_idx=ANGLE0_IDX,
                    use_bbox_speed=True, bbox_speed_idx=BBOX_SPEED_IDX
                )

                rec = { f'f{i}': vec[i] for i in range(len(vec)) }
                rec.update({
                    'vid': vid,
                    'pid': pid,
                    'frame_range': f"{st}-{ed-1}"
                })
                records.append(rec)

    df = pd.DataFrame.from_records(records)
    return df

def build_dataset_wavelet_wr(feats_data, fps=FP_RATE,
                             window_sec=WINDOW_SEC, step_sec=STEP_SEC,
                             channels=WR_WAVELET_CHANNELS):
    """
    Tạo DataFrame từ wr_feats (T×C), chỉ với đặc trưng Wavelet (15 trị/kênh).
    - cột f0..fN (N = 15 * len(channels))
    - meta: vid, pid, frame_range
    """
    import numpy as np, pandas as pd, os, json
    records = []
    wlen = int(window_sec * fps)
    step = int(step_sec   * fps)

    for vid, pid_dict in feats_data.items():
        for pid, pack in pid_dict.items():
            arr = pack.get('wr_feats', None)     # (T, 10)
            if arr is None or arr.shape[0] < wlen:
                continue
            wr_names = pack.get('wr_names', None)
            if not wr_names:  # fallback theo thứ tự đã lưu
                wr_names = [
                    "cos_knee_L","cos_knee_R",
                    "hip_flex_L","hip_flex_R",
                    "arm_swing_L","arm_swing_R",
                    "ankleLR_over_torso",
                    "wristLR_over_shoulder",
                    "midhip_y_local_norm",
                    "speed",
                ]

            # chọn cột theo channels
            sel_idx = [wr_names.index(c) for c in channels if c in wr_names]
            speed_idx = wr_names.index('speed') if 'speed' in wr_names else (len(wr_names)-1)   
            if not sel_idx: 
                continue
            arr_sel = arr[:, sel_idx]  # (T, C_sel)

            T = arr_sel.shape[0]
            for st in range(0, T - wlen + 1, step):
                ed = st + wlen
                win = arr_sel[st:ed, :]  # (wlen, C_sel)

                # Wavelet features: ghép 15 trị/kênh
                vec_parts = []
                for i in range(win.shape[1]):
                    vec_parts.append(dwt_energy_feats(win[:, i], wavelet=WAVELET, level=DWT_LEVEL))
                vec = np.hstack(vec_parts)
                sp = arr[st:ed, speed_idx]
                speed_mean = float(np.mean(sp))
                speed_p95  = float(np.percentile(sp, 95))
                speed_std  = float(np.std(sp))
                rec = {f"f{i}": vec[i] for i in range(vec.size)}
                rec.update({
                    'vid': vid, 'pid': pid, 'frame_range': f"{st}-{ed-1}",
                    'speed_mean': speed_mean, 'speed_p95': speed_p95, 'speed_std': speed_std,   # <<< NEW
                })
                records.append(rec)

    df = pd.DataFrame.from_records(records)

    # (tuỳ chọn) lưu map tên cột để debug/feature importance
    if len(records) > 0:
        base = [
            "E_A3","E_D3","E_D2","E_D1",
            "r_A3","r_D3","r_D2","r_D1",
            "R12","HiLo","band_entropy",
            "D1_L1L2","D2_L1L2","D1_bigfrac","D2_bigfrac"
        ]
        names = []
        for ch in [c for c in channels if c in wr_names]:
            names += [f"{ch}_{n}" for n in base]
        colmap = {f"f{i}": names[i] for i in range(len(names))}
        os.makedirs("feature_name_maps", exist_ok=True)
        with open(os.path.join("feature_name_maps", "wr_wavelet_names.json"), "w", encoding="utf-8") as f:
            json.dump(colmap, f, ensure_ascii=False, indent=2)
    return df


# =========================
# CHỌN TRAIN WALKING & TRAIN SVM
# =========================
def is_walking_video_name(vid_name: str):
    """
    Heuristic: true nếu tên video ám chỉ đi bộ.
    (Bạn có thể bổ sung pattern theo dataset: 'walking', 'hokou', v.v.)
    """
    s = vid_name.lower()
    return ('walking' in s) or ('hokou' in s) or ('walk' in s)

def train_one_class_svm_on_walking(df_train: pd.DataFrame):
    feat_cols = get_feature_cols(df_train)

    print(feat_cols)
    walk_mask = df_train['vid'].apply(is_walking_video_name)
    X_walk = (df_train.loc[walk_mask, feat_cols]
                        .apply(pd.to_numeric, errors='coerce')
                        .fillna(0.0)
                        .values)
    if X_walk.shape[0] == 0:
        raise RuntimeError("Không tìm thấy dữ liệu Walking để train. Kiểm tra rule is_walking_video_name().")

    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    )
    pipe.fit(X_walk)
    return pipe, walk_mask.sum()

def oversample_walking(df_train: pd.DataFrame, factor: int = 3):
    walk_mask = df_train['vid'].str.startswith("Walk")
    df_walk = df_train[walk_mask]
    df_nonwalk = df_train[~walk_mask]  # nếu có dữ liệu khác (thường sẽ bỏ)

    # Lặp lại dữ liệu Walking
    df_walk_oversampled = pd.concat([df_walk] * factor, ignore_index=True)

    return pd.concat([df_walk_oversampled, df_nonwalk], ignore_index=True)

# =========================
# DỰ ĐOÁN & XUẤT KẾT QUẢ
# =========================
def build_video_offsets_from_dir(pose_dir: str, csv_ext=('.csv',)):
    """
    Trả về:
      offsets[(vid,pid)] = start_offset (đã quy về 1-based)
      vid_info[vid] = {'max': max_FRAMEID (1-based), 'min_offset': min start_offset trong video}
    Quy ước:
      - Nếu pose của video là 1-based (đa số PID có min FRAMEID = 1) → start_offset = min_FRAMEID
      - Nếu pose là 0-based → start_offset = min_FRAMEID + 1 (đưa về 1-based)
    """
    offsets = {}
    vid_info = {}

    for fn in os.listdir(pose_dir):
        if not fn.lower().endswith(csv_ext):
            continue
        vid = canon_vid(os.path.splitext(fn)[0])
        p = os.path.join(pose_dir, fn)
        dfp = pd.read_csv(p)
        if 'FRAMEID' not in dfp.columns or 'PID' not in dfp.columns:
            print(f"[WARN] {fn} thiếu FRAMEID/PID, bỏ qua."); continue

        dfp['pid'] = dfp['PID'].astype(str).str.strip()
        dfp['FRAMEID'] = pd.to_numeric(dfp['FRAMEID'], errors='coerce').fillna(0).astype(int)

        mins_by_pid = dfp.groupby('pid')['FRAMEID'].min()
        maxF = int(dfp['FRAMEID'].max())
        # suy luận 1-based: đa số min == 1
        one_based = (pd.Series(mins_by_pid.values).value_counts().idxmax() == 1)

        starts = {}
        for pid, mn in mins_by_pid.items():
            start_offset = int(mn if one_based else mn + 1)  # quy về 1-based
            offsets[(vid, str(pid))] = start_offset
            starts[str(pid)] = start_offset

        vid_info[vid] = {
            'max': maxF if one_based else maxF + 1,      # quy về 1-based
            'min_offset': min(starts.values()) if starts else 1
        }

    return offsets, vid_info

def remap_frame_ranges_by_offset(df_out: pd.DataFrame, pose_dir: str,
                                 keep_original: bool = True,
                                 clamp_right: bool = True):
    """
    - Map frame_range 0-based của model → toạ độ frame thực 1-based của VIDEO:
        mapped = [st0 + start_offset, ed0 + start_offset]
    - KHÔNG clamp trái theo PID → độ dài cửa sổ giữ nguyên.
    - Chỉ clamp phải theo max frame của video (nếu cần).
    """
    offsets, vid_info = build_video_offsets_from_dir(pose_dir)

    df = df_out.copy()
    df['vid'] = df['vid'].apply(canon_vid)
    df['pid'] = df['pid'].astype(str).str.strip()

    if keep_original and 'frame_range_model' not in df.columns:
        df['frame_range_model'] = df['frame_range']

    new_ranges = []
    for vid, pid, fr in zip(df['vid'], df['pid'], df['frame_range']):
        st0, ed0 = _parse_frame_range(fr)          # 0-based của model
        # lấy offset cho (vid,pid); nếu thiếu dùng min_offset của video; nếu vẫn thiếu → 1
        start_off = offsets.get((vid, pid), None)
        if start_off is None:
            start_off = vid_info.get(vid, {}).get('min_offset', 1)
        st = st0 + start_off
        ed = ed0 + start_off

        if clamp_right:
            vmax = vid_info.get(vid, {}).get('max', None)
            if vmax is not None:
                ed = min(ed, vmax)

        new_ranges.append(_fmt_range(st, ed))

    df['frame_range'] = new_ranges
    return df

def apply_low_speed_rule(df_in: pd.DataFrame,
                         thr_mean: float = 0.010,
                         thr_p95:  float = 0.030,
                         thr_std:  float = 0.015,
                         col_mean: str = 'speed_mean',
                         col_p95:  str = 'speed_p95',
                         col_std:  str = 'speed_std') -> pd.DataFrame:
    df = df_in.copy()
    if not all(c in df.columns for c in [col_mean, col_p95, col_std]):
        # Nếu chưa có meta tốc độ thì bỏ qua
        return df

    is_run = df['Pred'].eq('Run')
    low_p95  = df[col_p95]  <= thr_p95
    low_mean = df[col_mean] <= thr_mean
    low_std  = df[col_std]  <= thr_std

    flip_mask = is_run & (low_p95 | (low_mean & low_std))
    df.loc[flip_mask, 'Pred'] = 'Walk'
    return df


def predict_and_export(df_any: pd.DataFrame, model, out_dir="output_csv",
                       out_suffix="_runwalk_eval.csv",
                       gold_csv_path=None,
                       pose_dir=None):
    feat_cols = get_feature_cols(df_any)
    X = df_any.loc[:, feat_cols].apply(pd.to_numeric, errors='raise').values

    y_pred = model.predict(X)
    labels = np.where(y_pred == -1, 'Run', 'Walk')

    df_out = df_any[['vid','pid','frame_range']].copy()
    df_out['vid'] = df_out['vid'].apply(canon_vid)
    df_out['pid'] = df_out['pid'].astype(str).str.strip()

    scores = model.named_steps['oneclasssvm'].decision_function(
        model.named_steps['standardscaler'].transform(X)
    )
    df_out['score'] = scores
    df_out['Pred']  = labels

    # >>> căn frame_range theo pose_dir (nhiều CSV, mỗi vid 1 file)
    if pose_dir is not None:
        df_out = remap_frame_ranges_by_offset(
            df_out, pose_dir=pose_dir, keep_original=True, clamp_right=True
        )

    # (tuỳ chọn) gắn TrueLabel từ gold
    if gold_csv_path is not None and os.path.exists(gold_csv_path):
        df_out = add_true_label_by_overlap(df_out, gold_csv_path)

    df_out['score'] = scores
    df_out['Pred']  = labels

    # gộp meta tốc độ sang df_out (nếu df_any có)
    for col in ['speed_mean','speed_p95','speed_std']:
        if col in df_any.columns:
            df_out[col] = df_any[col].values

    # >>> RULE: flip Run -> Walk khi tốc độ thấp
    if apply_low_speed:
        df_out = apply_low_speed_rule(
            df_out,
            thr_mean=speed_thr_mean,
            thr_p95=speed_thr_p95,
            thr_std=speed_thr_std
        )

    # >>> POST: đa số láng giềng ±k
    if apply_neighbor_post:
        df_out = flip_by_neighbor_majority_per_pid(df_out, k=neighbor_k)
    os.makedirs(out_dir, exist_ok=True)
    for vid, grp in df_out.groupby('vid'):
        cols = ['frame_range','pid','Pred','score']
        if 'TrueLabel' in df_out.columns: cols += ['TrueLabel']
        if 'frame_range_model' in df_out.columns: cols += ['frame_range_model']
        out_path = os.path.join(out_dir, f"{vid}{out_suffix}")
        grp[cols].to_csv(out_path, index=False)
        print(f"Saved {out_path}")

    return df_out

# =========================
# METRICS: EVENT-LEVEL (INTERVAL)
# =========================
def canon_vid(name: str):
    s = str(name).strip()
    s = os.path.basename(s)                         # bỏ path nếu có
    s = re.sub(r'\.(mp4|mov|avi|mkv)$', '', s, flags=re.IGNORECASE)  # bỏ đuôi video
    return s
    
def _parse_frame_range(fr_str: str):
    # "st-ed" -> (int(st), int(ed))
    s = str(fr_str).strip()
    a, b = s.split('-')
    return int(a), int(b)

def _overlap(st1, ed1, st2, ed2):
    # inclusive overlap
    return max(st1, st2) <= min(ed1, ed2)

def _fmt_range(st, ed):  # in ra dạng "st-ed"
    return f"{int(st)}-{int(ed)}"

from collections import defaultdict

def build_pose_index_from_dir(pose_dir: str, csv_ext=('.csv',)):
    """
    Trả về:
      - pose_idx: dict[(vid,pid)] -> {'min': mn, 'max': mx}
      - one_based_map: dict[vid] -> True/False (FRAMEID 1-based?)
    """
    pose_idx = {}
    one_based_map = {}

    for fn in os.listdir(pose_dir):
        if not fn.lower().endswith(csv_ext):
            continue
        vid = canon_vid(os.path.splitext(fn)[0])  # lấy tên file làm vid
        p = os.path.join(pose_dir, fn)
        dfp = pd.read_csv(p)

        # cột bắt buộc
        if 'FRAMEID' not in dfp.columns or 'PID' not in dfp.columns:
            print(f"[WARN] {fn} thiếu FRAMEID/PID, bỏ qua.")
            continue

        dfp['pid'] = dfp['PID'].astype(str).str.strip()
        dfp['FRAMEID'] = pd.to_numeric(dfp['FRAMEID'], errors='coerce').fillna(0).astype(int)

        # suy luận 1-based cho file này
        mins_this_file = dfp.groupby('pid')['FRAMEID'].min().tolist()
        file_one_based = (pd.Series(mins_this_file).value_counts().idxmax() == 1)
        one_based_map[vid] = file_one_based

        # min/max theo pid
        g = dfp.groupby('pid')['FRAMEID']
        for pid, s in g:
            mn, mx = int(s.min()), int(s.max())
            pose_idx[(vid, pid)] = {'min': mn, 'max': mx}

    return pose_idx, one_based_map

def align_frame_ranges_to_pose_dir(df_out: pd.DataFrame,
                                   pose_dir: str,
                                   keep_original: bool = True,
                                   force_one_based: bool = None):
    """
    - Đổi frame_range sang hệ chỉ mục của pose (per-vid 1-based/0-based).
    - Clamp theo [min,max] của từng (vid,pid) từ pose CSV của chính video đó.
    """
    pose_idx, one_based_map = build_pose_index_from_dir(pose_dir)

    df = df_out.copy()
    df['vid'] = df['vid'].apply(canon_vid)
    df['pid'] = df['pid'].astype(str).str.strip()

    if keep_original and 'frame_range_model' not in df.columns:
        df['frame_range_model'] = df['frame_range']

    new_ranges = []
    drop_mask = []

    for (vid, pid, fr) in zip(df['vid'], df['pid'], df['frame_range']):
        st, ed = _parse_frame_range(fr)

        # offset 1-based/0-based theo vid
        is_one_based = one_based_map.get(vid, False) if force_one_based is None else force_one_based
        if is_one_based:
            st += 1
            ed += 1

        # clamp theo (vid,pid) nếu có
        info = pose_idx.get((vid, pid))
        if info:
            mn, mx = info['min'], info['max']
            st_c = max(st, mn)
            ed_c = min(ed, mx)
        else:
            st_c, ed_c = st, ed  # không có info => giữ nguyên

        if ed_c < st_c:
            drop_mask.append(True)
            new_ranges.append(None)
        else:
            drop_mask.append(False)
            new_ranges.append(_fmt_range(st_c, ed_c))

    df['frame_range'] = new_ranges
    df = df[~pd.Series(drop_mask).values].reset_index(drop=True)
    return df

def _merge_run_windows(df_pred_run_group, allow_gap):
    """
    Gộp các cửa sổ Pred='Run' liên tiếp (theo (vid,pid)).
    allow_gap: số frame cho phép dính nhau (nên đặt = step_frames).
    Trả về list dict [{'vid','pid','p_st','p_ed','n_windows','score_min','score_max'}]
    """
    rows = []
    for _, r in df_pred_run_group.iterrows():
        st, ed = _parse_frame_range(r['frame_range'])
        rows.append((st, ed, float(r.get('score', 0.0))))
    rows.sort(key=lambda x: x[0])

    merged = []
    for st, ed, sc in rows:
        if not merged:
            merged.append({'p_st': st, 'p_ed': ed, 'n_windows': 1,
                           'score_min': sc, 'score_max': sc})
        else:
            last = merged[-1]
            if st <= last['p_ed'] + allow_gap:
                last['p_ed'] = max(last['p_ed'], ed)
                last['n_windows'] += 1
                last['score_min'] = min(last['score_min'], sc)
                last['score_max'] = max(last['score_max'], sc)
            else:
                merged.append({'p_st': st, 'p_ed': ed, 'n_windows': 1,
                               'score_min': sc, 'score_max': sc})
    return merged

def _load_gold(gold_csv_path: str):
    """
    Đọc gold CSV: cột ['VideoName','PID','Action','Start_frame','End_frame'].
    Trả về dict: (vid, pid)-> list [{'g_st','g_ed','action','matched':False}]
    """
    gdf = pd.read_csv(gold_csv_path)
    # Chuẩn hoá tên cột
    gdf = gdf.rename(columns={
        'VideoName': 'vid',
        'PID': 'pid',
        'Action': 'action',
        'Start_frame': 'g_st',
        'End_frame': 'g_ed'
    })
    # ép kiểu
    gdf['vid'] = gdf['vid'].apply(canon_vid)
    gdf['pid'] = gdf['pid'].astype(str).str.strip()
    gdf['g_st'] = pd.to_numeric(gdf['g_st'], errors='coerce').fillna(0).astype(int)
    gdf['g_ed'] = pd.to_numeric(gdf['g_ed'], errors='coerce').fillna(0).astype(int)

    gold = {}
    for (vid, pid), grp in gdf.groupby(['vid','pid']):
        items = []
        for _, r in grp.iterrows():
            items.append({'g_st': int(r['g_st']),
                          'g_ed': int(r['g_ed']),
                          'action': str(r.get('action', 'Run')),
                          'matched': False})
        gold[(vid, pid)] = items
    return gold

def build_predicted_intervals(df_pred: pd.DataFrame, step_frames: int):
    """
    Từ df_pred (các cửa sổ có cột ['vid','pid','frame_range','Pred','score']),
    tạo list predicted intervals theo từng (vid,pid).
    """
    df = df_pred.copy()
    df['vid'] = df['vid'].apply(canon_vid) 
    df['pid'] = df['pid'].astype(str).str.strip()

    df_run = df[df['Pred'] == 'Run'].copy()
    pred_intervals = []  # list of dicts

    for (vid, pid), grp in df_run.groupby(['vid','pid']):
        merged = _merge_run_windows(grp[['frame_range', 'score']], allow_gap=step_frames)
        for m in merged:
            pred_intervals.append({
                'vid': vid, 'pid': pid,
                'p_st': m['p_st'], 'p_ed': m['p_ed'],
                'n_windows': m['n_windows'],
                'score_min': m['score_min'], 'score_max': m['score_max']
            })
    return pred_intervals

def add_true_label_by_overlap(df_out: pd.DataFrame, gold_csv_path: str):
    """
    Thêm cột TrueLabel cho từng window bằng rule overlap với gold:
    gold có cột: VideoName, PID, Action, Start_frame, End_frame.
    """
    gdf = pd.read_csv(gold_csv_path).rename(columns={
        'VideoName':'vid', 'PID':'pid', 'Action':'action',
        'Start_frame':'g_st', 'End_frame':'g_ed'
    })
    gdf['vid']  = gdf['vid'].apply(canon_vid)
    gdf['pid']  = gdf['pid'].astype(str).str.strip()
    gdf['g_st'] = pd.to_numeric(gdf['g_st'], errors='coerce').fillna(0).astype(int)
    gdf['g_ed'] = pd.to_numeric(gdf['g_ed'], errors='coerce').fillna(0).astype(int)

    gold = {}
    for (vid, pid), grp in gdf.groupby(['vid','pid']):
        gold[(vid, pid)] = [(int(r.g_st), int(r.g_ed), str(r.action)) for _, r in grp.iterrows()]

    vids = df_out['vid'].apply(canon_vid)
    pids = df_out['pid'].astype(str).str.strip()
    y_true = []
    for vid, pid, fr in zip(vids, pids, df_out['frame_range']):
        st, ed = _parse_frame_range(fr)
        intervals = gold.get((vid, pid), [])
        is_run = any(_overlap(st, ed, g_st, g_ed) for (g_st, g_ed, _a) in intervals)
        y_true.append('Run' if is_run else 'Walk')

    out = df_out.copy()
    out['TrueLabel'] = y_true
    return out


def match_and_score(pred_intervals, gold_dict, prefer='most_abnormal'):
    """
    Ghép 1-1 giữa predicted intervals và gold (overlap là đúng).
    prefer: 'most_abnormal' => sort theo score_min tăng dần (SVM score âm hơn = bất thường hơn).
            'longest'       => ưu tiên khoảng dự đoán dài hơn trước.
    Trả về:
      metrics = {'TP':..,'FP':..,'FN':..,'precision':..,'recall':..,'f1':..}
      matches_df: DataFrame chi tiết từng predicted interval và gold matched (nếu có).
    """
    # Sắp xếp predicted để ghép ổn định
    if prefer == 'most_abnormal':
        pred_intervals = sorted(pred_intervals, key=lambda d: d['score_min'])
    elif prefer == 'longest':
        pred_intervals = sorted(pred_intervals, key=lambda d: (d['p_ed']-d['p_st']), reverse=True)
    else:
        pred_intervals = list(pred_intervals)

    # clone flags
    gold_matched = {(k): [dict(item) for item in v] for k, v in gold_dict.items()}

    matches = []  # lưu từng predicted và gold matched (nếu có)
    TP = 0
    for p in pred_intervals:
        vid, pid = p['vid'], p['pid']
        glist = gold_matched.get((vid, pid), [])
        found_idx = -1
        for i, g in enumerate(glist):
            if not g['matched'] and _overlap(p['p_st'], p['p_ed'], g['g_st'], g['g_ed']):
                found_idx = i
                break
        if found_idx >= 0:
            glist[found_idx]['matched'] = True
            g = glist[found_idx]
            TP += 1
            matches.append({
                **p,
                'matched': True,
                'g_st': g['g_st'], 'g_ed': g['g_ed'], 'action': g['action']
            })
        else:
            matches.append({
                **p,
                'matched': False,
                'g_st': None, 'g_ed': None, 'action': None
            })

    # Đếm FP/FN
    FP = sum(1 for m in matches if not m['matched'])
    FN = 0
    for (vid, pid), glist in gold_matched.items():
        FN += sum(1 for g in glist if not g['matched'])

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0

    metrics = {'TP': TP, 'FP': FP, 'FN': FN,
               'precision': precision, 'recall': recall, 'f1': f1}

    matches_df = pd.DataFrame(matches, columns=[
        'vid','pid','p_st','p_ed','n_windows','score_min','score_max',
        'matched','g_st','g_ed','action'
    ])
    return metrics, matches_df

def evaluate_run_detection(df_pred: pd.DataFrame, gold_csv_path: str,
                           step_frames: int, out_dir="run_walking_results_csv"):
    """
    Pipeline tính điểm:
      1) Tạo predicted intervals từ df_pred.
      2) Đọc gold, match overlap, tính TP/FP/FN/P/R/F1.
      3) Lưu matches & summary ra CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    pred_intervals = build_predicted_intervals(df_pred, step_frames=step_frames)
    gold = _load_gold(gold_csv_path)
    metrics, matches_df = match_and_score(pred_intervals, gold, prefer='most_abnormal')

    # Lưu
    matches_csv = os.path.join(out_dir, "run_matches_detail.csv")
    matches_df.to_csv(matches_csv, index=False)
    summary_csv = os.path.join(out_dir, "run_metrics_summary.csv")
    pd.DataFrame([metrics]).to_csv(summary_csv, index=False)

    print("=== Event-level metrics (overlap-based) ===")
    print(metrics)
    print(f"Saved matches: {matches_csv}")
    print(f"Saved summary: {summary_csv}")
    return metrics, matches_df

# =========================
# MAIN
# =========================
def main():
    TRAIN_DIRS = [
        'JPRecorded-feats-newv2',
    ]
    TEST_DIRS  = [
        'JPRecorded-feats-newv2',
    ]

    # Load train (chỉ Walk)
    df_train_list = []
    for d in TRAIN_DIRS:
        feats = load_all_wr_features(d)
        df_full = build_dataset_wavelet_wr(feats)
        df_walk = df_full[df_full['vid'].str.startswith("Walk")]
        df_train_list.append(df_walk)
    df_train = pd.concat(df_train_list, ignore_index=True) if df_train_list else pd.DataFrame()

    # Load test (chỉ Running)
    df_test_list = []
    for d in TEST_DIRS:
        feats = load_all_wr_features(d)          # <<< dùng wr
        df_full = build_dataset_wavelet_wr(feats)  # <<< dùng wr
        df_run = df_full[df_full['vid'].str.startswith("Run")]
        df_test_list.append(df_run)
    df_test = pd.concat(df_test_list, ignore_index=True) if df_test_list else pd.DataFrame()

    print("Train windows:", len(df_train), "| Test windows:", len(df_test))

    # Train SVM
    df_train = oversample_walking(df_train, factor=3)
    model, n_walk = train_one_class_svm_on_walking(df_train)
    print(f"Trained One-Class SVM on {n_walk} walking windows.")

    # Predict & export
    df_pred = predict_and_export(
        df_test, model,
        out_dir="run_walking_results_csv",
        out_suffix="_eval.csv",
        gold_csv_path="Toyota-running action.csv",     # nếu muốn
        pose_dir="JPrecorded_track/run"      # <== thư mục chứa nhiều CSV pose
    )

    GOLD_PATH = "Toyota-running action.csv"
    step_frames = int(STEP_SEC * FP_RATE)
    evaluate_run_detection(df_pred, GOLD_PATH, step_frames, out_dir="run_walking_results_csv")



if __name__ == '__main__':
    main()
