#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- Cấu hình chung ---
FP_RATE    = 5        # fps
WINDOW_SEC = 3        # cửa sổ sliding (giây)
STEP_SEC   = 1.0      # bước (giây)
DIST1_IDX  = 1        # index cho dist1
DIST9_IDX  = 9        # index cho dist9
ANGLE_IDX  = 0        # index cho angle0
PEAK_THRESH= 0.4      # ngưỡng peak để tính peak_duration

# --- Thiết lập logging ---
logging.basicConfig(
    filename='debug_client_v2.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)


def load_all_features(base_dir='client_v2_feat'):
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


def time_to_frame(t_sec, fps=FP_RATE, mode='floor'):
    return int(np.floor(t_sec*fps)) if mode=='floor' else int(np.ceil(t_sec*fps))


def extract_features(col, fps=FP_RATE):
    # tính slope
    diffs = np.diff(col) * fps
    slope_max = np.max(diffs) if len(diffs)>0 else 0.0
    slope_min = np.min(diffs) if len(diffs)>0 else 0.0

    maxv   = np.max(col)
    minv   = np.min(col)
    rng    = maxv - minv
    stdv   = np.std(col)
    meanv  = np.mean(col)
    n_osc  = int(np.count_nonzero(np.diff(np.sign(col-meanv))))

    # peak_duration: số frame liên tiếp col>PEAK_THRESH
    peak_dur = 0
    for k, grp in groupby(col > PEAK_THRESH):
        if k:
            peak_dur = max(peak_dur, len(list(grp)))

    # tính biên độ giữa đỉnh và đáy liên tiếp
    # tìm chỉ số các đỉnh và đáy đơn giản: local maxima/minima
    peaks, troughs = [], []
    for i in range(1, len(col)-1):
        if col[i] > col[i-1] and col[i] > col[i+1]: peaks.append(col[i])
        if col[i] < col[i-1] and col[i] < col[i+1]: troughs.append(col[i])
    # ghép xen kẽ peaks và troughs lấy amplitude lớn nhất
    peak_trough_amp = 0.0
    seq = sorted(peaks + troughs, key=lambda x: -abs(x - meanv))
    if peaks and troughs:
        # dùng max difference giữa bất kỳ đỉnh và đáy
        peak_trough_amp = max(abs(p - t) for p in peaks for t in troughs)

    return {
        'max':             maxv,
        'min':             minv,
        'range':           rng,
        'std':             stdv,
        'n_osc':           n_osc,
        'peak_dur':        peak_dur,
        'slope_max':       slope_max,
        'slope_min':       slope_min,
        'peak_trough_amp': peak_trough_amp
    }


def rule_predict(dist1_feats, dist9_feats, ang0_feats, fps=FP_RATE):
    SLOPE_THRESH   = 0.4 * fps
    AMP_HIGH       = 0.7
    AMP_LOW        = 0.06
    AMP_PT_THRESH  = 0.06
    OSC_THRESH     = 3

    # Falling: spike hoặc amplitude mạnh trên dist
    if (
        dist1_feats['slope_max'] > SLOPE_THRESH or
        dist1_feats['slope_min'] < -SLOPE_THRESH or
        dist1_feats['range']    > AMP_HIGH or
        dist9_feats['slope_max'] > SLOPE_THRESH or
        dist9_feats['slope_min'] < -SLOPE_THRESH or
        dist9_feats['range']    > AMP_HIGH
    ):
        return 'Falling'

    # Staggering: dao động cả dist hoặc góc với biên độ phù hợp
    if (
        ((dist1_feats['n_osc'] >= OSC_THRESH and AMP_LOW <= dist1_feats['range'] <= AMP_HIGH and dist1_feats['peak_trough_amp'] >= AMP_PT_THRESH)
         or
         (dist9_feats['n_osc'] >= OSC_THRESH and AMP_LOW <= dist9_feats['range'] <= AMP_HIGH and dist9_feats['peak_trough_amp'] >= AMP_PT_THRESH))
        or
        (ang0_feats['n_osc'] >= OSC_THRESH and AMP_LOW <= ang0_feats['range'] <= AMP_HIGH and ang0_feats['peak_trough_amp'] >= AMP_PT_THRESH)
    ):
        return 'Staggering'

    return None


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
        logging.debug(
            f"[RAW] {video}, PID={pid}, window={st}-{ed} ({st/fps:.1f}s–{ed/fps:.1f}s), "
            f"dist1_range={f1['range']:.2f}, dist9_range={f9['range']:.2f}, ang0_range={fa0['range']:.2f}, pred={lbl}"
        )
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


def main():
    feats_data = load_all_features('client_v2_feat')

    df = pd.read_csv('Toyota_data - client_v2.csv').dropna(subset=['Pid'])
    df['time_start'] = pd.to_timedelta(df['Time start']).dt.total_seconds()
    df['time_end']   = pd.to_timedelta(df['Time end']).dt.total_seconds()
    df['vid']        = df['Name video'].astype(str).str.strip()
    df['pid_str']    = df['Pid'].astype(int).astype(str)
    df['f0']         = df['time_start'].apply(lambda t: time_to_frame(t, FP_RATE, 'floor'))
    df['f1']         = df['time_end'].apply(lambda t: time_to_frame(t, FP_RATE, 'ceil'))
    mask = df['f1'] <= df['f0']
    df.loc[mask, 'f1'] = df.loc[mask, 'f0'] + FP_RATE

    gt = {}
    for _, r in df.iterrows():
        key = (r.vid, r.pid_str)
        gt.setdefault(key, []).append((r.f0, r.f1, r.Action))

    logging.debug('=== Starting evaluation ===')

    results = {'Falling':{'TP':0,'FP':0,'FN':0},
               'Staggering':{'TP':0,'FP':0,'FN':0}}

    for video, pid_map in feats_data.items():
        for pid, feats in pid_map.items():
            ivals   = gt.get((video, pid), [])
            arr_dist= feats.get('dist_arr')
            raw_ang = feats.get('angle_arr')
            # lấy ang0
            if raw_ang.ndim > 1:
                ang0 = raw_ang[:, ANGLE_IDX]
            else:
                ang0 = raw_ang.flatten()
            if arr_dist is None or ang0 is None:
                continue
            # Đồng bộ độ dài
            min_len = min(arr_dist.shape[0], ang0.shape[0])
            arr_dist_trim = arr_dist[:min_len, :]
            ang0_trim     = ang0[:min_len]
            # Kết hợp angle0 + dist_arr
            arr2d = np.hstack([ang0_trim.reshape(-1,1), arr_dist_trim])

            preds   = detect_intervals(arr2d, video, pid)
            matched = set()

            for pst, ped, plbl in preds:
                f1 = extract_features(arr2d[pst:ped, DIST1_IDX], FP_RATE)
                f9 = extract_features(arr2d[pst:ped, DIST9_IDX], FP_RATE)
                fa0= extract_features(arr2d[pst:ped, ANGLE_IDX], FP_RATE)
                true_lbls = [glbl for (g0,g1,glbl) in ivals if overlap((pst,ped),(g0,g1))]
                true_lbl  = true_lbls[0] if true_lbls else 'Other'
                logging.debug(
                    f"[MERGED] {video}, PID={pid}, frames={pst}-{ped} ({pst/FP_RATE:.1f}s–{ped/FP_RATE:.1f}s), "
                    f"true={true_lbl}, pred={plbl}, "
                    f"dist1_range={f1['range']:.2f}, dist9_range={f9['range']:.2f}, ang0_range={fa0['range']:.2f}"
                )
                overlaps = [i for i,(g0,g1,glbl) in enumerate(ivals) if glbl==plbl and overlap((pst,ped),(g0,g1))]
                if overlaps:
                    results[plbl]['TP'] += 1
                    matched.update(overlaps)
                else:
                    results[plbl]['FP'] += 1

            for i,(g0,g1,glbl) in enumerate(ivals):
                if i not in matched:
                    results[glbl]['FN'] += 1

    for lbl in ['Falling','Staggering']:
        TP = results[lbl]['TP']
        FP = results[lbl]['FP']
        FN = results[lbl]['FN']
        prec = TP/(TP+FP) if TP+FP else 0
        rec  = TP/(TP+FN) if TP+FN else 0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
        print(f"\nLabel={lbl}: TP={TP}, FP={FP}, FN={FN}")
        print(f" Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

if __name__ == '__main__':
    main()