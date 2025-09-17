#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from itertools import groupby
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from scipy.stats import entropy
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
import pywt
# --- Cấu hình chung ---
FP_RATE    = 5        # fps
WINDOW_SEC = 5        # cửa sổ sliding (giây)
STEP_SEC   = 1.0      # bước (giây)
DIST1_IDX  = 1        # index cho dist1
DIST9_IDX  = 9        # index cho dist9
ANGLE_IDX  = 0        # index cho angle0
PEAK_THRESH= 0.3      # ngưỡng peak để tính peak_duration

# --- Thiết lập logging ---
# logging.basicConfig(
#     filename='debug_client_v2.log',
#     filemode='w',
#     level=logging.DEBUG,
#     format='%(asctime)s %(levelname)s: %(message)s'
# )


def load_all_features(base_dir):
    data = {}
    for video in os.listdir(base_dir):
        p_v = os.path.join(base_dir, video)
        if not os.path.isdir(p_v): continue
        data[video] = {}
        for pid_dir in os.listdir(p_v):
            p_p = os.path.join(p_v, pid_dir)
            if not os.path.isdir(p_p): continue
            pid = pid_dir.split('_',1)[1]
            data[video][pid] = {}
            for fn in os.listdir(p_p):
                if not fn.endswith('.npy'): continue
                arr = np.load(os.path.join(p_p, fn))
                arr2d = arr.reshape(-1,1) if arr.ndim==1 else arr
                data[video][pid][fn[:-4]] = arr2d
    return data

# --- DWT features only, level=1 ---
def extract_dwt_coeffs(col: np.ndarray, wavelet='db1', level=2):
    # Trả về mảng coefficients: [A1, D1]
    coeffs = pywt.wavedec(col, wavelet=wavelet, level=level)
    return np.concatenate(coeffs)

def time_to_frame(t_sec, fps=FP_RATE, mode='floor'):
    return int(np.floor(t_sec*fps)) if mode=='floor' else int(np.ceil(t_sec*fps))

def detect_intervals(arr2d, video=None, pid=None, fps=FP_RATE):
    wlen = int(WINDOW_SEC * fps)
    step = int(STEP_SEC   * fps)
    T, _ = arr2d.shape
    raw_preds = []
    for st in range(0, T - wlen + 1, step):
        ed = st + wlen
        win = arr2d[st:ed, :]
        f1  = extract_features(win[:, DIST1_IDX], fps)
        f9  = extract_features(win[:, DIST9_IDX], fps)
        fa0 = extract_features(win[:, ANGLE_IDX], fps)
        lbl  = rule_predict(f1, f9, fa0, fps)
        # logging.debug(
        #     f"[RAW] {video}, PID={pid}, window={st}-{ed} ({st/fps:.1f}s–{ed/fps:.1f}s), "
        #     f"dist1_range={f1['range']:.2f}, dist9_range={f9['range']:.2f}, ang0_range={fa0['range']:.2f}, pred={lbl}"
        # )
        if lbl:
            raw_preds.append((st, ed, lbl))
    # merge intervals
    merged = []
    for st, ed, lbl in raw_preds:
        if merged and merged[-1][2] == lbl and st <= merged[-1][1] + step:
            merged[-1] = (merged[-1][0], ed, lbl)
        else:
            merged.append((st, ed, lbl))
    return merged


def overlap(i1, i2):
    a1,a2 = i1; b1,b2 = i2
    return max(a1,b1) < min(a2,b2)

def load_gt_from_csv(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=['Pid'])
    df['time_start'] = pd.to_timedelta(df['Time start']).dt.total_seconds()
    df['time_end']   = pd.to_timedelta(df['Time end']).dt.total_seconds()
    df['vid']        = df['Name video'].astype(str).str.strip()
    df['pid_str']    = df['Pid'].astype(int).astype(str)
    df['f0'] = df['time_start'].apply(lambda t: time_to_frame(t, FP_RATE, 'floor'))
    df['f1'] = df['time_end'].apply(lambda t: time_to_frame(t, FP_RATE, 'ceil'))
    # ensure f1 > f0
    mask = df['f1'] <= df['f0']
    df.loc[mask, 'f1'] = df.loc[mask, 'f0'] + FP_RATE

    gt = {}
    for _, r in df.iterrows():
        key = (r.vid, r.pid_str)
        gt.setdefault(key, []).append((r.f0, r.f1, r.Action))
    return gt

def build_dataset(feats_data, gt_dict):
    records = []
    wlen = int(WINDOW_SEC * FP_RATE)
    step = int(STEP_SEC   * FP_RATE)

    for (vid, pid), ivals in gt_dict.items():
        feats = feats_data.get(vid, {}).get(pid)
        if not feats: continue
        arr = feats.get('dist_arr')
        if arr is None: continue
        T, _ = arr.shape
        for st in range(0, T - wlen + 1, step):
            ed = st + wlen
            seg1 = arr[st:ed, DIST1_IDX]
            seg9 = arr[st:ed, DIST9_IDX]
            # DWT level1
            c1 = extract_dwt_coeffs(seg1)
            c9 = extract_dwt_coeffs(seg9)
            feats_vector = np.hstack([c1, c9])
            # label
            labs = [lbl for (g0,g1,lbl) in ivals if max(st,g0)<min(ed,g1)]
            if any(l=='Falling' for l in labs):    lab='Falling'
            elif any(l=='Staggering' for l in labs): lab='Staggering'
            else:                                   lab='Other'
            rec = dict(zip([f'f{i}' for i in range(len(feats_vector))], feats_vector))
            rec['label'] = lab
            records.append(rec)
    return pd.DataFrame.from_records(records)


def train_with_ensemble_cv(X_train, y_train_enc):
    # 1) Khởi tạo base learners
    clf_rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf_xgb = XGBClassifier(
        objective        = 'multi:softprob',
        num_class        = 3,
        n_estimators     = 500,
        max_depth        = 8,
        learning_rate    = 0.1,
        use_label_encoder=False,      
        eval_metric      ='mlogloss',
        random_state     =42,
        n_jobs           =-1
    )

    # 2) Tạo VotingClassifier (soft voting)
    ensemble = VotingClassifier(
        estimators=[
            ('rf', clf_rf),
            ('xgb', clf_xgb),
        ],
        voting='soft',
        weights=[1, 1],  # ưu tiên RF
        n_jobs=-1
    )

    # 3) Đánh giá qua Stratified K-Fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        ensemble,
        X_train,
        y_train_enc,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1
    )
    print("Ensemble CV f1_macro scores:", scores)
    print("Mean f1_macro:", scores.mean())

    # 4) Fit ensemble lên toàn bộ train
    ensemble.fit(X_train, y_train_enc)
    return ensemble

def main():
    feats_v1 = load_all_features('client_v1_feat')
    feats_v2 = load_all_features('client_v2_feat')
    feats_v3 = load_all_features('client_v3_feat')
    feats_v4 = load_all_features('client_v4_feat')

    gt_v1 = load_gt_from_csv('Toyota_data - client_v1.csv')
    gt_v2 = load_gt_from_csv('Toyota_data - client_v2.csv')
    gt_v3 = load_gt_from_csv('Toyota_data - client_v3.csv')
    gt_v4 = load_gt_from_csv('Toyota_data - client_v4.csv')

    df_train = pd.concat([
        build_dataset(feats_v1, gt_v1),
        build_dataset(feats_v2, gt_v2),
        build_dataset(feats_v3, gt_v3)
    ], ignore_index=True)
    df_test  = build_dataset(feats_v4, gt_v4)

    X = df_train.drop(columns='label')
    y = df_train['label']

    counts = df_train['label'].value_counts()
    n_stag = counts['Staggering']

    df_fall    = df_train[df_train.label == 'Falling']
    df_stag    = df_train[df_train.label == 'Staggering']
    df_other   = df_train[df_train.label == 'Other']

    df_fall_up   = df_fall.sample(    n_stag, replace=True,  random_state=42)
    df_other_dn  = df_other.sample(   n_stag, replace=True, random_state=42)

    df_train_balanced = pd.concat([
        df_fall_up,
        df_stag,
        df_other_dn
    ], ignore_index=True)

    df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Final balanced:", df_train_balanced.label.value_counts())

    X_train, y_train = df_train_balanced.drop(columns='label'), df_train_balanced['label']
    X_test,  y_test  = df_test .drop(columns='label'), df_test ['label']

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)   # e.g. ['Falling','Other','Staggering'] → [0,1,2]
    y_test_enc  = le.transform(y_test)

    ensemble = train_with_ensemble_cv(X_train, y_train_enc)

    # 5) Đánh giá trên test
    y_pred_enc = ensemble.predict(X_test)
    y_pred     = le.inverse_transform(y_pred_enc)
    print("\n=== Ensemble on Test ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n",
        confusion_matrix(y_test, y_pred, labels=le.classes_))
if __name__ == '__main__':
    main()